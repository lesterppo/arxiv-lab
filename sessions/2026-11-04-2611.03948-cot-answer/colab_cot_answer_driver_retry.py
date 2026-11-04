"""CoT-vs-answer supervision Colab driver -- arXiv 2611.03948.

Paper claim tested (shrunk experiment, never shrunk model):
on medical multiple-choice QA, segment-weighted SFT (SwS: CoT and
answer losses weighted separately) generally beats uniform-across-
sequence SFT (UaS: equal loss weight on the full CoT+answer
sequence), with general-purpose models benefiting most.

Shrunk design (this driver):
- Dataset: MedMCQA train split (medical MCQ, same task format as
  the paper's MedQA/EMS-MCQA arms; the paper's exact datasets and
  CoT annotations are not re-downloaded here -- MedMCQA's own
  `exp` explanations serve as the CoT, truncated to 600 chars;
  recorded as a limit). Filter: non-empty explanation, valid gold
  option, prompt+CoT+answer <= 480 tokens. Train 240 items, eval 100
  disjoint items, same items/order for both arms (paired comparison).
- Student: try Qwen/Qwen3.5-9B first (9B-first rule), fall back to
  Qwen/Qwen3-8B (proven T4 model) if load fails. NF4 / bf16, LoRA
  r=16 alpha=32 (the standing floors), completion-only loss.
- Arms: base (no adapter) -> UaS (w_cot=1.0, w_ans=1.0) -> SwS
  (w_cot=0.5, w_ans=1.5). The paper's exact SwS weights are not in
  its abstract, so SwS here is the stated operationalization
  "answer tokens weighted 3x CoT tokens, loss normalized by total
  weight" -- the test checks the directional claim that separate
  segment weighting changes MCQ accuracy vs uniform, not the
  paper's exact hyperparameters.
- Eval: greedy generation (max 200 tokens) on the 100 eval items,
  parse the LAST "Answer: [A-D]" match; accuracy over all items,
  accuracy over parsed items, and parse-fail rate reported.
  Metrics + per-item details are written incrementally to
  /content (mid-run download protection); summary to
  /content/cot_answer_summary.json. TRAIN_DONE printed last, on
  success only. No secrets in this file (public ungated models).

  RETRY 2026-11-04 (third attempt, fresh instruction "Re try"):
  two survival changes over the rerun driver, nothing else:
  (1) evaluate() now flushes each per-item details row AND a
  running partial-metrics JSON to /content as each item completes
  -- the rerun lost everything because the old evaluate() wrote
  both CSVs only after the full 100-item eval, so a reclamation
  mid-eval left nothing; (2) after UaS training, the adapter is
  saved to /content and zipped for download, so a future retry --
  if ever authorized -- need not retrain UaS. The banked-base-row
  skip from the rerun driver is kept unchanged.
"""

import csv
import json
import re
import subprocess
import sys

MODELS = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"]
METRICS_CSV = "/content/cot_answer_metrics.csv"
DETAILS_CSV = "/content/cot_answer_details.csv"
SUMMARY_JSON = "/content/cot_answer_summary.json"
# RETRY: running partial metrics, rewritten after every eval item so a
# mid-run download (or a reclamation followed by a host-side poll)
# still sees eval progress even if the full eval never finishes.
PARTIAL_JSON = "/content/cot_answer_partial.json"
# RETRY: UaS adapter save target (dir) + single-file zip for download.
ADAPTER_DIR = "/content/cotans-uas-adapter"
ADAPTER_ZIP = "/content/cotans-uas-adapter.zip"
SEED = 11
N_TRAIN = 240
N_EVAL = 100
MAX_TOKENS = 480
LETTERS = ["A", "B", "C", "D"]

# RERUN 2026-11-04 (second attempt, fresh instruction from Peter):
# skip the base eval when the SAME model loads, reusing the banked
# base row from the first run (accuracy 0.32, parse_fail 0.56 on
# Qwen/Qwen3.5-9B) instead of spending ~38 min re-running it. If a
# different model loads (fallback), base eval runs fresh so the arms
# stay comparable. The downloaded metrics/details CSVs then hold only
# the new arms; the session's merged files are assembled locally
# after download (banked base row + new arms). All other logic is
# identical to the fixed driver (remove_unused_columns=False).
SKIP_BASE_EVAL = True
BANKED_BASE = {"arm": "base", "model": "Qwen/Qwen3.5-9B", "n": 100,
               "accuracy": 0.32, "accuracy_parsed": 0.7272727272727273,
               "parse_fail": 0.56}

ANSWER_RE = re.compile(r"answer:\s*([A-D])", re.IGNORECASE)


def format_prompt(item):
    return ("Question: %s\nA. %s\nB. %s\nC. %s\nD. %s\nReasoning:"
            % (item["question"], item["opa"], item["opb"],
               item["opc"], item["opd"]))


def clean_cot(exp):
    return " ".join((exp or "").split())[:600]


def parse_answer(text):
    """Last 'Answer: X' match, or None (parse fail)."""
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def build_example(tok, prompt, cot, letter, w_cot, w_ans):
    """Tokenize one training example with segment weights.

    Returns (input_ids, labels, weights): prompt tokens masked
    (-100, weight 0), CoT tokens weighted w_cot, answer-segment
    tokens ("\nAnswer: X") weighted w_ans. `tok` needs only
    `__call__(text, add_special_tokens=False).input_ids`.
    """
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    c_ids = tok(" " + cot, add_special_tokens=False).input_ids
    a_ids = tok("\nAnswer: " + letter, add_special_tokens=False).input_ids
    ids = p_ids + c_ids + a_ids
    labels = [-100] * len(p_ids) + c_ids + a_ids
    weights = [0.0] * len(p_ids) + [w_cot] * len(c_ids) + [w_ans] * len(a_ids)
    return ids, labels, weights


def install():
    pkgs = ["transformers==5.20.2", "peft", "bitsandbytes", "accelerate",
            "datasets"]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                       check=False)
    except Exception as e:  # noqa: BLE001
        print("pip install failed (continuing with existing stack): %r" % e,
              flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False, capture_output=True)
    import transformers
    print("transformers=%s" % transformers.__version__, flush=True)


def main():
    install()
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    assert torch.cuda.is_available(), "no GPU: refusing to fake a GPU run"
    dev = "cuda"
    print("GPU: %s total=%.1f GB torch=%s"
          % (torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9,
             torch.__version__), flush=True)

    # --- data: MedMCQA train, deterministic order, filter in order ---
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.shuffle(seed=SEED)
    tok_probe = None
    # Tokenizer is loaded with the model below; for the length filter
    # we first load the tokenizer of the first candidate model.
    from transformers import AutoTokenizer as AT
    tok_probe = AT.from_pretrained(MODELS[0])
    if tok_probe.pad_token is None:
        tok_probe.pad_token = tok_probe.eos_token
    train_items, eval_items = [], []
    for row in ds:
        if len(train_items) >= N_TRAIN and len(eval_items) >= N_EVAL:
            break
        exp = clean_cot(row.get("exp"))
        if len(exp) < 40:
            continue
        cop = row.get("cop")
        if cop is None or not (0 <= int(cop) <= 3):
            continue
        item = {"question": row["question"], "opa": row["opa"],
                "opb": row["opb"], "opc": row["opc"], "opd": row["opd"],
                "gold": LETTERS[int(cop)], "cot": exp}
        prompt = format_prompt(item)
        n = len(tok_probe(prompt, add_special_tokens=False).input_ids) \
            + len(tok_probe(" " + exp, add_special_tokens=False).input_ids) \
            + len(tok_probe("\nAnswer: " + item["gold"],
                            add_special_tokens=False).input_ids)
        if n > MAX_TOKENS:
            continue
        if len(train_items) < N_TRAIN:
            train_items.append(item)
        else:
            eval_items.append(item)
    assert len(train_items) == N_TRAIN, len(train_items)
    assert len(eval_items) == N_EVAL, len(eval_items)
    print("data: train=%d eval=%d" % (len(train_items), len(eval_items)),
          flush=True)

    tok = None
    model = None
    model_id = None
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    for cand in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                cand, quantization_config=bnb, torch_dtype=torch.bfloat16,
                device_map="cuda")
            model.config.use_cache = False
            model_id = cand
            print("model loaded: %s peak=%.2f GB"
                  % (cand, torch.cuda.max_memory_allocated() / 1e9), flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print("model load failed %s: %r" % (cand, e), flush=True)
            model = None
    if model is None:
        raise RuntimeError("no model loaded from %s" % MODELS)

    def evaluate(tag):
        # RETRY: incremental writes. The old version collected all 100
        # rows and wrote both CSVs only at the end, so the 2026-11-04
        # rerun's reclamation mid-UaS-eval left nothing. Here each
        # details row is flushed as its item completes, and a running
        # partial-metrics JSON is rewritten after every item, so a
        # host-side mid-run download always sees the latest progress.
        # The final metrics row is still appended at the end as before.
        model.eval()
        rows = []
        with open(DETAILS_CSV, "a", newline="") as df:
            dw = csv.DictWriter(df, fieldnames=["arm", "idx", "gold", "pred",
                                                "parsed", "correct"])
            if df.tell() == 0:
                dw.writeheader()
            for idx, item in enumerate(eval_items):
                ids = tok(format_prompt(item),
                          return_tensors="pt").input_ids.to(dev)
                with torch.no_grad():
                    gen = model.generate(ids, max_new_tokens=200,
                                         do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                text = tok.decode(gen[0][ids.shape[1]:],
                                  skip_special_tokens=True)
                pred = parse_answer(text)
                row = {"arm": tag, "idx": idx, "gold": item["gold"],
                       "pred": pred or "", "parsed": bool(pred),
                       "correct": bool(pred) and pred == item["gold"]}
                rows.append(row)
                dw.writerow(row)
                df.flush()
                n_done = len(rows)
                parsed_so_far = [r for r in rows if r["parsed"]]
                partial = {"arm": tag, "model": model_id,
                           "completed": n_done, "n": len(eval_items),
                           "accuracy_so_far":
                               sum(r["correct"] for r in rows) / n_done,
                           "accuracy_parsed_so_far":
                               (sum(r["correct"] for r in parsed_so_far)
                                / len(parsed_so_far) if parsed_so_far else 0.0),
                           "parse_fail_so_far":
                               1 - len(parsed_so_far) / n_done}
                with open(PARTIAL_JSON, "w") as pf:
                    json.dump(partial, pf)
                    pf.flush()
                if (idx + 1) % 20 == 0 or idx == 0:
                    print("EVAL %s progress %d/%d: %s"
                          % (tag, n_done, len(eval_items),
                             json.dumps(partial)), flush=True)
        n = len(rows)
        parsed = [r for r in rows if r["parsed"]]
        m = {"arm": tag, "model": model_id, "n": n,
             "accuracy": sum(r["correct"] for r in rows) / n,
             "accuracy_parsed": (sum(r["correct"] for r in parsed) / len(parsed)
                                 if parsed else 0.0),
             "parse_fail": 1 - len(parsed) / n}
        with open(METRICS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(m.keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(m)
        print("EVAL %s: %s" % (tag, json.dumps(m)), flush=True)
        return m

    if SKIP_BASE_EVAL and model_id == BANKED_BASE["model"]:
        print("SKIP base eval: reusing banked base row (model %s)"
              % model_id, flush=True)
        results = [dict(BANKED_BASE)]
    else:
        results = [evaluate("base")]

    def train_arm(tag, w_cot, w_ans):
        nonlocal model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules="all-linear",
                         task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, cfg)
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable()

        examples = []
        for item in train_items:
            ids, labels, weights = build_example(
                tok, format_prompt(item), item["cot"], item["gold"],
                w_cot, w_ans)
            examples.append({"input_ids": ids[:512], "labels": labels[:512],
                             "weights": weights[:512]})

        class DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(examples)

            def __getitem__(self, i):
                return examples[i]

        def collate(batch):
            maxlen = max(len(b["input_ids"]) for b in batch)
            pad = tok.pad_token_id
            input_ids, labels, weights, attn = [], [], [], []
            for b in batch:
                n = maxlen - len(b["input_ids"])
                input_ids.append(b["input_ids"] + [pad] * n)
                labels.append(b["labels"] + [-100] * n)
                weights.append(b["weights"] + [0.0] * n)
                attn.append([1] * len(b["input_ids"]) + [0] * n)
            return {"input_ids": torch.tensor(input_ids),
                    "labels": torch.tensor(labels),
                    "weights": torch.tensor(weights),
                    "attention_mask": torch.tensor(attn)}

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False,
                             num_items_in_batch=None):
                weights = inputs.pop("weights")
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                shift_weights = weights[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    shift_labels.reshape(-1), reduction="none")
                loss = loss.view(shift_labels.size())
                denom = shift_weights.sum().clamp(min=1e-6)
                loss = (loss * shift_weights).sum() / denom
                return (loss, outputs) if return_outputs else loss

        # FIX 2026-11-04 (after the first Colab run died at UaS step 0
        # with KeyError: 'weights'): Trainer's default
        # remove_unused_columns=True drops dataset columns that are
        # not model-forward arguments -- "weights" is such a column,
        # so the collator never saw it. Disabling the removal is the
        # standard fix for custom loss columns. This fixed driver is
        # STAGED ONLY: it has not yet run on Colab (a rerun needs a
        # fresh instruction; see the session README).
        args = TrainingArguments(
            output_dir="/tmp/cotans-%s" % tag, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, num_train_epochs=1,
            learning_rate=1e-4, logging_steps=10, save_strategy="no",
            bf16=True, report_to=[], gradient_checkpointing=True,
            optim="adamw_torch_fused", remove_unused_columns=False)
        trainer = WeightedTrainer(model=peft_model, args=args,
                                  train_dataset=DS(), data_collator=collate)
        trainer.train()
        # RETRY: save the UaS adapter so a future retry -- if ever
        # authorized -- need not retrain UaS from scratch. Saved
        # before eval/unload, while peft_model still holds the trained
        # adapter; zipped to a single file for download. Best effort:
        # a save failure must not kill the evals (the priority is
        # completing UaS eval + SwS train + SwS eval in this attempt).
        if tag == "uas":
            try:
                import shutil
                peft_model.save_pretrained(ADAPTER_DIR)
                shutil.make_archive(ADAPTER_DIR, "zip", ADAPTER_DIR)
                print("UAS adapter saved to %s (+ .zip)" % ADAPTER_DIR,
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print("UAS adapter save failed (continuing to eval): %r"
                      % e, flush=True)
        m = evaluate(tag)
        peft_model = peft_model.unload()
        model = peft_model
        model.config.use_cache = False
        return m

    results.append(train_arm("uas", 1.0, 1.0))
    results.append(train_arm("sws", 0.5, 1.5))

    summary = {"model": model_id, "n_train": N_TRAIN, "n_eval": N_EVAL,
               "arms": results,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY %s" % json.dumps(summary), flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
