"""Colab driver (FALLBACK, 2026-11-05 rerun): model merging does not
require uniform LoRA ranks.

Fallback changes vs colab_merge_ranks_driver.py (which DIED with a
deterministic CUDA OOM at step 0/15 on Qwen3.5-9B, 14.55/14.56 GB):
  - Base model: Qwen/Qwen3-8B (proven training model; the 9B OOM was in
    Qwen3.5's hybrid gated-delta layers, so a different proven
    architecture is the fallback, not a re-try of the same model).
  - Training sequences capped at 256 tokens (was 512).
  - Batch 1 with gradient accumulation 8 (same effective batch 8).
  - Gradient checkpointing on (+ enable_input_require_grads, required
    with a frozen NF4 base under checkpointing).
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set in main()
    before torch is imported).
Everything else identical: alpha=2r, all-linear targets, lr 1e-4,
1 epoch, completion-only loss, remove_unused_columns=False, same
domains/arms/parsing, incremental metrics writes.

Paper: arXiv:2611.06486 — "Model Merging Does Not Require Uniform LoRA
Ranks". Claims tested (bounded, one T4 day):
  C1: giving each domain expert its own best rank costs simple averaging
      nothing (and can gain up to ~1.8 pp multi-domain accuracy).
  C2: each domain has its own best rank (single-expert evals differ by rank).
  Diagnostic: per-expert delta Frobenius norms — the paper's mechanism for
      why magnitude-based spectral selection drops the small expert.

Design (bounded adaptation, stated honestly):
  - Base: Qwen/Qwen3-8B (fallback driver; 9B OOM'd at step 0/15).
    NF4 QLoRA,
    alpha = 2r, target all-linear, lr 1e-4, 1 epoch, completion-only loss.
  - Domains: math (GSM8K) and multiple-choice QA (ARC-Challenge) —
    deterministic parsing (numeric answer / letter), no judge.
  - Experts: math r=8, math r=32, qa r=8, qa r=32 (4 trainings).
  - Merges (exact, no peft merge internals): averaged delta
    0.5*delta1 + 0.5*delta2, with delta_i = (alpha_i/r_i) B_i A_i = 2 B_i A_i.
    A merged adapter of rank r1+r2 with scaling 1, A = vstack([A1, A2]),
    B = hstack([B1, B2]) reproduces that sum exactly.
    uniform  = math16? No — uniform merge uses r=16? To keep the merge
    comparison clean, uniform = math r=16 is NOT trained; instead the
    uniform arm is math r=8 + qa r=8 (same rank both sides) and the
    non-uniform arm is math r=32 + qa r=8 (each side at its own rank if
    C2 shows math prefers 32 and qa prefers 8; if C2 shows the reverse,
    the non-uniform arm is swapped accordingly — see README for the
    arm actually run, recorded in merge_metrics.csv).
    NOTE: the driver below trains r in {8, 32} only; the uniform arm is
    (8, 8) and the non-uniform arm is (32, 8) unless single-expert
    results invert the preference, in which case it is (8, 32).
    The choice is made by the driver from measured single-expert scores
    and written to the metrics CSV before merging.
  - Eval: GSM8K test first 60, ARC-Challenge test first 60, 2-shot
    prompts (demos disjoint from train/eval), batched greedy decoding.
  - Incremental writes: every phase appends to /content/merge_metrics.csv
    and per-item rows to /content/merge_details.csv, so a reclaimed run
    still leaves numbers (lesson from 2026-11-04 cotans runs).
  - Prints MERGE_DONE as its very last line on completion.

Run via: python3 colab.py train --driver <this file> --rotate
  --results /content/merge_metrics.csv /content/merge_details.csv
  --out-dir <dir> --done-marker MERGE_DONE
"""

import csv
import gc
import json
import math
import os
import re
import subprocess
import sys

METRICS = "/content/merge_metrics.csv"
DETAILS = "/content/merge_details.csv"
ADAPTER_DIR = "/content/adapters"


def _pip():
    pkgs = ["transformers==5.20.2", "peft", "bitsandbytes", "accelerate",
            "datasets", "safetensors"]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                       check=True, timeout=900)
    except Exception as e:  # noqa: BLE001
        print("pip install failed (continuing with existing stack): %r" % e,
              flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   capture_output=True)


def _append_metrics(row):
    new = not os.path.exists(METRICS)
    with open(METRICS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print("METRIC %s" % json.dumps(row), flush=True)


def _append_detail(row):
    new = not os.path.exists(DETAILS)
    with open(DETAILS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    _pip()
    import torch
    import transformers
    print("transformers=%s" % transformers.__version__, flush=True)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import load_dataset
    assert torch.cuda.is_available(), "no GPU: refusing to fake a GPU run"
    print("GPU: %s total=%.1f GB torch=%s"
          % (torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9,
             torch.__version__), flush=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # ---- model probe: Qwen3-8B (fallback driver; 9B OOM'd at step 0) ----
    tok = None
    model_id = None
    for cand in ["Qwen/Qwen3-8B"]:
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            bnb = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                cand, quantization_config=bnb, torch_dtype=torch.bfloat16,
                device_map="auto")
            model.config.use_cache = False
            model_id = cand
            print("PROBE_GREEN %s peak=%.2f GB"
                  % (cand, torch.cuda.max_memory_allocated() / 1e9), flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print("PROBE_FAIL %s: %r" % (cand, e), flush=True)
            torch.cuda.empty_cache()
    assert model_id is not None, "no candidate model loaded"
    _append_metrics({"phase": "probe", "model": model_id,
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                     "acc": "", "n": ""})

    # ---- data ----
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    class DS:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        pad = tok.pad_token_id
        batch_ids, batch_labels, batch_mask = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            batch_ids.append(f["input_ids"] + [pad] * (maxlen - n))
            batch_labels.append(f["labels"] + [-100] * (maxlen - n))
            batch_mask.append(f["attention_mask"] + [0] * (maxlen - n))
        return {"input_ids": torch.tensor(batch_ids),
                "labels": torch.tensor(batch_labels),
                "attention_mask": torch.tensor(batch_mask)}

    # ---- train 4 experts on one base load (unload between experts) ----
    # Build SFT rows (prompt/completion split) per domain.
    math_rows = []
    for ex in gsm.select(range(120)):
        p = "Question: %s\nAnswer:" % ex["question"]
        math_rows.append((p, " " + ex["answer"]))
    qa_rows = []
    for ex in arc.select(range(150)):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        choices = "\n".join("%s. %s" % (l, t) for l, t in zip(labels, texts))
        ans = ex["answerKey"]
        ans_text = texts[labels.index(ans)] if ans in labels else ""
        qa_rows.append(("Question: %s\nChoices:\n%s\nAnswer:"
                        % (ex["question"], choices),
                        " %s. %s" % (ans, ans_text)))
    sft = {"math": [], "qa": []}
    for domain, rows in [("math", math_rows), ("qa", qa_rows)]:
        for p, c in rows:
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            c_ids = tok(c, add_special_tokens=False)["input_ids"]
            ids = (p_ids + c_ids)[:256]
            labels = ([-100] * len(p_ids) + c_ids)[:256]
            sft[domain].append({"input_ids": ids, "labels": labels,
                                "attention_mask": [1] * len(ids)})

    for domain, r in [("math", 8), ("math", 32), ("qa", 8), ("qa", 32)]:
        tag = "%s_r%d" % (domain, r)
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                         target_modules="all-linear",
                         task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, cfg)
        peft_model.enable_input_require_grads()  # grad checkpointing + frozen base
        peft_model.print_trainable_parameters()
        args = TrainingArguments(
            output_dir="/tmp/merge-%s" % tag, per_device_train_batch_size=1,
            gradient_accumulation_steps=8, num_train_epochs=1,
            learning_rate=1e-4, logging_steps=5, save_strategy="no",
            gradient_checkpointing=True,
            remove_unused_columns=False, report_to=[])
        trainer = Trainer(model=peft_model, args=args,
                          train_dataset=DS(sft[domain]), data_collator=collate)
        trainer.train()
        out = os.path.join(ADAPTER_DIR, tag)
        peft_model.save_pretrained(out)
        loss = None
        for entry in trainer.state.log_history:
            if "loss" in entry:
                loss = entry["loss"]
        _append_metrics({"phase": "train", "model": model_id, "adapter": tag,
                         "train_loss": loss, "acc": "", "n": ""})
        # Unload adapter -> clean base for the next expert (peft unload()
        # removes the adapter WITHOUT merging, verified in peft 0.21.2).
        model = peft_model.unload()
        del peft_model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ---- delta norms (CPU, from adapter files) ----
    from safetensors.torch import load_file, save_file
    import glob

    def delta_norm(adapter_path):
        sd = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
        total = 0.0
        mods = {}
        for k, v in sd.items():
            if k.endswith("lora_A.weight"):
                mods.setdefault(k[:-len("lora_A.weight")], {})["A"] = v.float()
            elif k.endswith("lora_B.weight"):
                mods.setdefault(k[:-len("lora_B.weight")], {})["B"] = v.float()
        for m, ab in mods.items():
            if "A" in ab and "B" in ab:
                d = 2.0 * (ab["B"] @ ab["A"])
                total += float((d * d).sum())
        return math.sqrt(total)

    norms = {}
    for domain, r in [("math", 8), ("math", 32), ("qa", 8), ("qa", 32)]:
        tag = "%s_r%d" % (domain, r)
        norms[tag] = delta_norm(os.path.join(ADAPTER_DIR, tag))
        _append_metrics({"phase": "delta_norm", "model": model_id,
                         "adapter": tag, "delta_frobenius": round(norms[tag], 2),
                         "acc": "", "n": ""})

    # ---- eval (batched greedy) ----
    def fewshot(domain):
        if domain == "math":
            demos = []
            for ex in gsm.select(range(1000, 1002)):
                demos.append("Question: %s\nAnswer: %s"
                             % (ex["question"], ex["answer"]))
            return "\n\n".join(demos) + "\n\n"
        demos = []
        for ex in arc.select(range(1000, 1002)):
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            choices = "\n".join("%s. %s" % (l, t)
                                for l, t in zip(labels, texts))
            demos.append("Question: %s\nChoices:\n%s\nAnswer: %s. %s"
                         % (ex["question"], choices, ex["answerKey"],
                            texts[labels.index(ex["answerKey"])]
                            if ex["answerKey"] in labels else ""))
        return "\n\n".join(demos) + "\n\n"

    def eval_items(m, domain, adapter_path, label):
        if adapter_path is None:
            m_eval = m
            peft_eval = None
        else:
            peft_eval = PeftModel.from_pretrained(m, adapter_path)
            peft_eval.eval()
            m_eval = peft_eval
        if domain == "math":
            items = []
            for ex in gsm_test.select(range(60)):
                items.append((fewshot("math") + "Question: %s\nAnswer:"
                              % ex["question"],
                              float(re.findall(r"-?\d+\.?\d*", ex["answer"].split("####")[-1])[0])
                              if "####" in ex["answer"] else None))
            max_new = 160
        else:
            items = []
            for ex in arc_test.select(range(60)):
                labels = ex["choices"]["label"]
                texts = ex["choices"]["text"]
                choices = "\n".join("%s. %s" % (l, t)
                                    for l, t in zip(labels, texts))
                items.append((fewshot("qa") + "Question: %s\nChoices:\n%s\nAnswer:"
                              % (ex["question"], choices), ex["answerKey"]))
            max_new = 24
        tok.padding_side = "left"
        correct = 0
        with torch.no_grad():
            for s in range(0, len(items), 8):
                chunk = items[s:s + 8]
                enc = tok([c[0] for c in chunk], return_tensors="pt",
                          padding=True).to(m.device)
                out = m_eval.generate(**enc, max_new_tokens=max_new,
                                      do_sample=False,
                                      pad_token_id=tok.pad_token_id)
                for j, o in enumerate(out):
                    gen = tok.decode(o[enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
                    gold = chunk[j][1]
                    if domain == "math":
                        nums = re.findall(r"-?\d+\.?\d*", gen.replace(",", ""))
                        ok = bool(nums) and gold is not None and \
                            abs(float(nums[-1]) - gold) < 1e-6
                    else:
                        mm = re.search(r"\b([A-D])\b", gen)
                        ok = bool(mm) and mm.group(1) == gold
                    correct += int(ok)
                    _append_detail({"model": label, "domain": domain,
                                    "idx": s + j, "correct": int(ok),
                                    "gold": gold, "gen": gen[:300]})
        acc = correct / len(items)
        _append_metrics({"phase": "eval", "model": model_id, "adapter": label,
                         "domain": domain, "acc": round(acc, 4), "n": len(items)})
        if peft_eval is not None:
            m_eval = peft_eval.unload()
        tok.padding_side = "right"
        return acc

    model.config.use_cache = True
    results = {}
    results["base_math"] = eval_items(model, "math", None, "base")
    results["base_qa"] = eval_items(model, "qa", None, "base")
    for domain, r in [("math", 8), ("math", 32), ("qa", 8), ("qa", 32)]:
        tag = "%s_r%d" % (domain, r)
        results[tag] = eval_items(model, domain,
                                  os.path.join(ADAPTER_DIR, tag), tag)
    # Non-uniform arm chosen from measured single-expert scores.
    math_best = 32 if results["math_r32"] >= results["math_r8"] else 8
    qa_best = 8 if results["qa_r8"] >= results["qa_r32"] else 32
    print("SINGLE_BEST math_r=%d qa_r=%d" % (math_best, qa_best), flush=True)

    def build_merged(name, tag1, tag2):
        sd1 = load_file(os.path.join(ADAPTER_DIR, tag1,
                                     "adapter_model.safetensors"))
        sd2 = load_file(os.path.join(ADAPTER_DIR, tag2,
                                     "adapter_model.safetensors"))
        merged = {}
        for k in sd1:
            if k in sd2 and k.endswith("lora_A.weight"):
                merged[k] = torch.cat([sd1[k], sd2[k]], dim=0)
            elif k in sd2 and k.endswith("lora_B.weight"):
                merged[k] = torch.cat([sd1[k], sd2[k]], dim=1)
        out = os.path.join(ADAPTER_DIR, name)
        os.makedirs(out, exist_ok=True)
        save_file(merged, os.path.join(out, "adapter_model.safetensors"))
        # Config: rank = r1 + r2, alpha = r1 + r2 (scaling 1), same targets.
        import shutil
        src_cfg = os.path.join(ADAPTER_DIR, tag1, "adapter_config.json")
        cfg = json.load(open(src_cfg))
        r1 = cfg["r"]
        cfg2 = json.load(open(os.path.join(ADAPTER_DIR, tag2,
                                           "adapter_config.json")))
        cfg["r"] = r1 + cfg2["r"]
        cfg["lora_alpha"] = cfg["r"]
        json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"),
                  indent=2)
        return out

    uniform = build_merged("merged_uniform", "math_r8", "qa_r8")
    nonuniform = build_merged("merged_nonuniform",
                              "math_r%d" % math_best, "qa_r%d" % qa_best)
    for name, path in [("merged_uniform", uniform),
                       ("merged_nonuniform", nonuniform)]:
        for domain in ["math", "qa"]:
            results[name + "_" + domain] = eval_items(model, domain, path, name)

    print("RESULTS %s" % json.dumps(results), flush=True)
    print("MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
