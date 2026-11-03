"""EP-OPD phase-2 driver -- arXiv 2611.01967 (redesign after phase-1 ceiling).

Phase 1 (same-day, session epopd-1103) found a ceiling effect: base
and single-doc arms both scored orig_acc=1.0, cf_acc=1.0, bypass=0.0
on DISJOINT eval compounds. Fictional compounds carry no closed-book
preference, and training on disjoint items cannot induce one at eval
-- so phase 1 could not discriminate the paper's claim (the paper's
bypass grows with the student's pre-distillation preference for the
original answer on the SAME questions).

Phase 2 redesign (same items at train and eval, the preference is
then induced by the single-doc training itself):
- MEMO set: 60 matched pairs. Arm "single" trains on original docs
  only; arm "paired" (fresh adapter) trains on both matched docs.
  If single-doc preference induces bypass, single should show
  cf_acc < orig_acc and bypass > 0 on MEMO counterfactual docs,
  while paired should recover cf_acc and cut bypass, preserving
  orig_acc -- the paper's EP-OPD claim in miniature.
- HOLDOUT set: 30 disjoint pairs, evaluated for both arms as a
  generality check (paired supervision should not destroy plain
  doc-copying on unseen items).
- Model: Qwen/Qwen3.5-9B first, Qwen/Qwen3-8B fallback; NF4/bf16,
  LoRA r=16/alpha=32, 2 epochs each arm. Metrics written
  incrementally to /content/epopd2_metrics.csv; summary to
  /content/epopd2_summary.json. TRAIN_DONE printed last, on success
  only. No secrets in this file (public ungated models).
"""
import csv
import json
import random
import subprocess
import sys

MODELS = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"]
METRICS_CSV = "/content/epopd2_metrics.csv"
SUMMARY_JSON = "/content/epopd2_summary.json"
SEED = 7

PREFIXES = ["Zorblax", "Kevrin", "Thalor", "Vexin", "Ondral", "Pyrith",
            "Caldwen", "Mirell", "Sorvane", "Taldor", "Nyxal", "Brevik"]
PROPS = [("melting point", "C"), ("boiling point", "C"),
         ("density", "g/cm3"), ("hardness", "Mohs")]


def build_pairs(n):
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
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "transformers==5.20.2", "peft", "bitsandbytes",
                        "accelerate"], check=False)
    except Exception as e:  # noqa: BLE001
        print("pip install failed (continuing): %r" % e, flush=True)
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
    memo_pairs, holdout_pairs = all_pairs[:60], all_pairs[160:190]
    for p in memo_pairs + holdout_pairs:
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

    def evaluate(tag, pairs, set_name):
        model.eval()
        rows = []
        for p in pairs:
            outs = {}
            for doc_key in ("doc_orig", "doc_cf"):
                ids = tok(prompt(p[doc_key], p["q"]),
                          return_tensors="pt").input_ids.to(dev)
                with torch.no_grad():
                    gen = model.generate(ids, max_new_tokens=10,
                                         do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                outs[doc_key] = tok.decode(gen[0][ids.shape[1]:],
                                           skip_special_tokens=True)
            orig_ok = p["gold_orig"] in outs["doc_orig"]
            cf_ok = p["gold_cf"] in outs["doc_cf"]
            bypass = (p["gold_orig"] in outs["doc_cf"]) and not cf_ok
            rows.append({"orig_ok": orig_ok, "cf_ok": cf_ok, "bypass": bypass})
        m = {"arm": tag, "set": set_name, "model": model_id, "n": len(rows),
             "orig_acc": sum(r["orig_ok"] for r in rows) / len(rows),
             "cf_acc": sum(r["cf_ok"] for r in rows) / len(rows),
             "bypass": sum(r["bypass"] for r in rows) / len(rows)}
        with open(METRICS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(m.keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(m)
        print("EVAL %s/%s: %s" % (tag, set_name, json.dumps(m)), flush=True)
        return m

    results = []
    results.append(evaluate("base", memo_pairs, "memo"))

    def train_arm(tag, examples):
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
            output_dir="/tmp/epopd2-%s" % tag, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, num_train_epochs=2,
            learning_rate=1e-4, logging_steps=10, save_strategy="no",
            bf16=True, report_to=[], gradient_checkpointing=True,
            optim="adamw_torch_fused")
        trainer = Trainer(model=peft_model, args=args,
                          train_dataset=DS(), data_collator=collate)
        trainer.train()
        ms = []
        ms.append(evaluate(tag, memo_pairs, "memo"))
        ms.append(evaluate(tag, holdout_pairs, "holdout"))
        model = peft_model.unload()
        model.config.use_cache = False
        return ms

    single_ex = [(prompt(p["doc_orig"], p["q"]), p["gold_orig"])
                 for p in memo_pairs]
    results.extend(train_arm("single", single_ex))
    paired_ex = []
    for p in memo_pairs:
        paired_ex.append((prompt(p["doc_orig"], p["q"]), p["gold_orig"]))
        paired_ex.append((prompt(p["doc_cf"], p["q"]), p["gold_cf"]))
    results.extend(train_arm("paired", paired_ex))

    summary = {"model": model_id, "memo_pairs": len(memo_pairs),
               "holdout_pairs": len(holdout_pairs), "arms": results,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY %s" % json.dumps(summary), flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
