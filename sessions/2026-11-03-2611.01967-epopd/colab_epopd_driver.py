"""EP-OPD evidence-bypass Colab driver -- arXiv 2611.01967.

Paper claim tested (shrunk experiment, never shrunk model):
standard on-policy-style training on original-document answers raises
original-doc accuracy but INCREASES evidence bypass (model keeps the
original answer when the document is changed to support a different
one); Evidence-Paired OPD (paired supervision on matched documents
supporting different answers to the same question) recovers much of
the counterfactual loss while preserving most of the original gain.

Shrunk design (this driver):
- Synthetic matched doc-QA: 220 fictional-compound pairs. Each pair
  shares question + filler context and differs ONLY in the
  answer-critical sentence (value A vs value B). Train 160 pairs,
  eval 60 disjoint pairs. No external dataset, no network data fetch.
- Student: try Qwen/Qwen3.5-9B first (9B-first rule), fall back to
  Qwen/Qwen3-8B (proven T4 model) if load fails. NF4 / bf16, LoRA
  r=16 alpha=32 (the standing floors), completion-only loss.
- Arms: base (no adapter) -> single (train on original-doc answers
  only, the standard-OPD proxy) -> paired (train on BOTH matched
  docs, the EP-OPD proxy; explicit margin term omitted -- recorded
  as a limit, paired supervision is the tested component).
- Metrics per arm: original-doc accuracy, counterfactual accuracy,
  bypass rate = P(answer == original gold | counterfactual doc),
  on the same 60 eval pairs. Written incrementally to
  /content/epopd_metrics.csv (mid-run download protection); summary
  to /content/epopd_summary.json. TRAIN_DONE printed last, on
  success only. No secrets in this file (public ungated models).
"""
import csv
import json
import random
import subprocess
import sys

MODELS = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"]
METRICS_CSV = "/content/epopd_metrics.csv"
SUMMARY_JSON = "/content/epopd_summary.json"
SEED = 7

PREFIXES = ["Zorblax", "Kevrin", "Thalor", "Vexin", "Ondral", "Pyrith",
            "Caldwen", "Mirell", "Sorvane", "Taldor", "Nyxal", "Brevik"]
PROPS = [("melting point", "C"), ("boiling point", "C"),
         ("density", "g/cm3"), ("hardness", "Mohs")]


def build_pairs(n):
    """Deterministic matched pairs; filler identical, critical value differs."""
    rng = random.Random(SEED)
    pairs = []
    for i in range(n):
        name = "%s-%d" % (PREFIXES[i % len(PREFIXES)], 100 + i)
        prop, unit = PROPS[i % len(PROPS)]
        a = rng.randint(40, 900)
        b = a + rng.choice([-37, -23, 19, 41, 58])
        filler = ("It was first synthesised in a laboratory setting. "
                  "Samples are stored under inert gas. "
                  "The compound is stable at room temperature.")
        doc_orig = ("Compound %s report. %s Its %s is %d %s. %s"
                    % (name, filler.split(". ")[0] + ".", prop, a, unit,
                       " ".join(filler.split(". ")[1:])))
        # Pair doc: identical except the critical sentence's value.
        doc_cf = doc_orig.replace("%d %s" % (a, unit), "%d %s" % (b, unit))
        q = "What is the %s of compound %s?" % (prop, name)
        pairs.append({"id": i, "name": name, "prop": prop, "unit": unit,
                      "gold_orig": "%d %s" % (a, unit),
                      "gold_cf": "%d %s" % (b, unit),
                      "doc_orig": doc_orig, "doc_cf": doc_cf, "q": q})
    return pairs


def prompt(doc, q):
    return "Document: %s\nQuestion: %s\nAnswer:" % (doc, q)


def install():
    pkgs = ["transformers==5.20.2", "peft", "bitsandbytes", "accelerate"]
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
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    assert torch.cuda.is_available(), "no GPU: refusing to fake a GPU run"
    dev = "cuda"
    print("GPU: %s total=%.1f GB torch=%s"
          % (torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9,
             torch.__version__), flush=True)

    all_pairs = build_pairs(220)
    train_pairs, eval_pairs = all_pairs[:160], all_pairs[160:]
    assert len(eval_pairs) == 60
    # Pair integrity check (the paper's matched-intervention design):
    for p in train_pairs + eval_pairs:
        assert p["doc_orig"] != p["doc_cf"]
        assert p["gold_orig"] != p["gold_cf"]
        assert p["doc_cf"].replace(p["gold_cf"], p["gold_orig"]) == p["doc_orig"]

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
        """Greedy eval on the 60 eval pairs, both doc versions."""
        model.eval()
        rows = []
        for p in eval_pairs:
            outs = {}
            for doc_key, gold_key in (("doc_orig", "gold_orig"),
                                      ("doc_cf", "gold_cf")):
                ids = tok(prompt(p[doc_key], p["q"]),
                          return_tensors="pt").input_ids.to(dev)
                with torch.no_grad():
                    gen = model.generate(ids, max_new_tokens=10,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                text = tok.decode(gen[0][ids.shape[1]:],
                                  skip_special_tokens=True)
                outs[doc_key] = text
            orig_ok = p["gold_orig"] in outs["doc_orig"]
            cf_ok = p["gold_cf"] in outs["doc_cf"]
            bypass = (p["gold_orig"] in outs["doc_cf"]) and not cf_ok
            rows.append({"orig_ok": orig_ok, "cf_ok": cf_ok, "bypass": bypass})
        m = {"arm": tag, "model": model_id, "n": len(rows),
             "orig_acc": sum(r["orig_ok"] for r in rows) / len(rows),
             "cf_acc": sum(r["cf_ok"] for r in rows) / len(rows),
             "bypass": sum(r["bypass"] for r in rows) / len(rows)}
        with open(METRICS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(m.keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(m)
        print("EVAL %s: %s" % (tag, json.dumps(m)), flush=True)
        return m

    results = []
    # Base eval: model has no adapter yet, so plain eval is the base arm.
    results.append(evaluate("base"))

    def train_arm(tag, examples):
        """examples: list of (prompt_text, completion_text)."""
        nonlocal model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules="all-linear",
                         task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, cfg)
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable()

        class DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(examples)

            def __getitem__(self, i):
                pr, comp = examples[i]
                p_ids = tok(pr, add_special_tokens=False).input_ids
                c_ids = tok(" " + comp, add_special_tokens=False).input_ids
                ids = (p_ids + c_ids)[:384]
                labels = ([-100] * len(p_ids) + c_ids)[:384]
                return {"input_ids": ids, "labels": labels}

        def collate(batch):
            maxlen = max(len(b["input_ids"]) for b in batch)
            pad = tok.pad_token_id
            input_ids, labels, attn = [], [], []
            for b in batch:
                n = maxlen - len(b["input_ids"])
                input_ids.append(b["input_ids"] + [pad] * n)
                labels.append(b["labels"] + [-100] * n)
                attn.append([1] * len(b["input_ids"]) + [0] * n)
            return {"input_ids": torch.tensor(input_ids),
                    "labels": torch.tensor(labels),
                    "attention_mask": torch.tensor(attn)}

        args = TrainingArguments(
            output_dir="/tmp/epopd-%s" % tag, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, num_train_epochs=1,
            learning_rate=1e-4, logging_steps=10, save_strategy="no",
            bf16=True, report_to=[], gradient_checkpointing=True,
            optim="adamw_torch_fused")
        trainer = Trainer(model=peft_model, args=args,
                          train_dataset=DS(), data_collator=collate)
        trainer.train()
        # Eval with this adapter active, then drop it for the next arm.
        m = evaluate(tag)
        peft_model = peft_model.unload()
        model = peft_model
        model.config.use_cache = False
        return m

    single_ex = [(prompt(p["doc_orig"], p["q"]), p["gold_orig"])
                 for p in train_pairs]
    results.append(train_arm("single", single_ex))
    paired_ex = []
    for p in train_pairs:
        paired_ex.append((prompt(p["doc_orig"], p["q"]), p["gold_orig"]))
        paired_ex.append((prompt(p["doc_cf"], p["q"]), p["gold_cf"]))
    results.append(train_arm("paired", paired_ex))

    summary = {"model": model_id, "train_pairs": len(train_pairs),
               "eval_pairs": len(eval_pairs), "arms": results,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY %s" % json.dumps(summary), flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
