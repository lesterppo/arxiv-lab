"""Colab driver — Budget Internalization (arXiv:2611.26172), SFT ablation.

Paper claim tested: "SFT on artificially truncated traces reproduces
budget internalization, whereas SFT on naturally short traces does not"
-- i.e. the fixed rollout cap becomes a learned constraint, not just a
training-cost control. The full RLVR version is out of reach for one
free-T4 day; the paper's own SFT ablation is the bounded-scale test.

Design (all numbers are the experiment-shrink, per the 2026-10-27 rule):
  Model: Qwen/Qwen3.5-9B-Base first (standing pick), NF4 QLoRA r=16/a=32,
         bf16, transformers==5.20.2 (Qwen3.5 needs 5.x -- 2026-11-20
         RLCDR lesson). Fallback via env MODEL: Qwen/Qwen3-8B with
         TRANSFORMERS_PIN='<5.0' (its proven recipe).
  Budget B = 256 tokens (env BUDGET).
  Phase 0 probe: load NF4, 5-token generation, print PROBE_OK.
  Phase 1 traces: GSM8K train, first N_TRAIN=400 problems, 4-shot greedy
         traces from the base model, keep correct-answer ones.
    Arm A "truncated": traces longer than B, CoT cut at B tokens, then
         a forced "#### <answer>" ending.
    Arm B "natural": traces naturally <= B tokens.
    Cap SFT_MAX_EXAMPLES=150 per arm.
  Phase 2 SFT: one epoch each, fresh LoRA per arm, batch 2 x accum 2,
         seq 512, lr 2e-4. Adapters saved to /content/adapter_<arm>.
  Phase 3 eval: held-out GSM8K test, first N_EVAL=100 problems, greedy,
         budgets {128, 256, 512, 1024} for base + arm A + arm B.
         Metrics: accuracy, mean generated tokens, parse-fail rate.
  Internalization signature: arm A beats base near B=256 but falls
  BELOW base at 1024; arm B tracks base at 1024.

Writes /content/budget_internalization_metrics.csv and
/content/budget_internalization_summary.json, prints TRAIN_DONE last.

OOM lessons baked in (2026-11-20 RLCDR leg 2): generate() only (never a
manual forward that materializes [batch, seq, vocab] logits),
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, no
prepare_model_for_kbit_training(), gradient checkpointing on, small
eval batches (8, or 4 at budget 1024).
"""
import csv
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B-Base")
TRANSFORMERS_PIN = os.environ.get("TRANSFORMERS_PIN", "==5.20.2")
BUDGET = int(os.environ.get("BUDGET", "256"))
N_TRAIN = int(os.environ.get("N_TRAIN", "400"))
N_EVAL = int(os.environ.get("N_EVAL", "100"))
SFT_MAX = int(os.environ.get("SFT_MAX_EXAMPLES", "150"))
EVAL_BUDGETS = [128, 256, 512, 1024]

FEWSHOT = """Solve the problem step by step, then give the final answer after ####.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Solution: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.
#### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Solution: There are originally 3 cars. 2 more arrive. 3 + 2 = 5.
#### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Solution: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left.
#### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Solution: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.
#### 8

"""

ANS_RE = re.compile(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)")


def log(msg):
    print(msg, flush=True)


def extract_answer(text):
    """Last '#### <number>' in text, as a float; None if absent."""
    if not text:
        return None
    m = ANS_RE.findall(text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def install():
    log("== pip installs ==")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"transformers{TRANSFORMERS_PIN}", "bitsandbytes",
                    "accelerate", "peft", "datasets", "hf_transfer"],
                   check=True)
    log("== installs done ==")


def main():
    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling)

    log(f"== model: {MODEL} | transformers pin: {TRANSFORMERS_PIN} | B={BUDGET} ==")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generation
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    # ---- Phase 0: probe ----
    probe_ids = tok("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**probe_ids, max_new_tokens=5, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    log("PROBE_OK: " + tok.decode(out[0][probe_ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip())
    log(f"VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")

    # ---- Phase 1: traces ----
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    train_qs = list(ds_train.select(range(N_TRAIN)))
    log(f"== Phase 1: generating traces for {len(train_qs)} problems ==")

    def batch_generate(prompts, max_new, batch=8):
        outs = []
        t0 = time.time()
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new,
                                     do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            for j, g in enumerate(gen):
                n_in = enc["input_ids"].shape[1]
                outs.append(tok.decode(g[n_in:], skip_special_tokens=True))
            if (i // batch) % 10 == 9:
                log(f"  gen {len(outs)}/{len(prompts)} "
                    f"({time.time() - t0:.0f}s elapsed)")
        return outs

    prompts = [FEWSHOT + f"Question: {ex['question']}\nSolution:"
               for ex in train_qs]
    completions = batch_generate(prompts, 512)
    usable = []  # (reasoning_text, answer, n_tokens)
    for ex, comp in zip(train_qs, completions):
        ans = extract_answer(comp)
        gold = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        if ans is not None and abs(ans - gold) < 1e-6:
            reasoning = comp.split("####")[0].strip()
            n_tok = len(tok(reasoning)["input_ids"])
            usable.append((reasoning, gold, n_tok))
    log(f"usable correct traces: {len(usable)}/{len(train_qs)}")

    def fmt_answer(gold):
        return f"#### {int(gold) if float(gold).is_integer() else gold}"

    arm_trunc, arm_nat = [], []
    for reasoning, gold, n_tok in usable:
        if n_tok > BUDGET:
            ids = tok(reasoning)["input_ids"][:BUDGET]
            cut = tok.decode(ids, skip_special_tokens=True).strip()
            arm_trunc.append(cut + "\n" + fmt_answer(gold))
        else:
            arm_nat.append(reasoning + "\n" + fmt_answer(gold))
    log(f"arm truncated candidates: {len(arm_trunc)}; "
        f"arm natural candidates: {len(arm_nat)}")
    # Pair each completion with its question (same order as usable).
    q_trunc = [train_qs[i] for i, (_, _, n) in enumerate(usable)
               if n > BUDGET][:SFT_MAX]
    q_nat = [train_qs[i] for i, (_, _, n) in enumerate(usable)
             if n <= BUDGET][:SFT_MAX]
    arm_trunc, arm_nat = arm_trunc[:SFT_MAX], arm_nat[:SFT_MAX]
    log(f"arm sizes: truncated={len(arm_trunc)} natural={len(arm_nat)}")

    # ---- Phase 2: SFT per arm ----
    from peft import LoraConfig, get_peft_model

    def sft(arm_name, examples, questions):
        log(f"== Phase 2: SFT arm {arm_name} ({len(examples)} ex) ==")
        texts = [FEWSHOT + f"Question: {q['question']}\nSolution: {c}"
                 + tok.eos_token
                 for c, q in zip(examples, questions)]
        tok.padding_side = "right"
        enc = tok(texts, truncation=True, max_length=512)
        from datasets import Dataset as HFDataset
        dset = HFDataset.from_dict({"input_ids": enc["input_ids"],
                                    "attention_mask": enc["attention_mask"]})
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj",
                                         "o_proj", "gate_proj", "up_proj",
                                         "down_proj"],
                         task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, cfg)
        peft_model.gradient_checkpointing_enable()
        args = TrainingArguments(
            output_dir=f"/content/ckpt_{arm_name}", per_device_train_batch_size=2,
            gradient_accumulation_steps=2, num_train_epochs=1,
            learning_rate=2e-4, bf16=True, logging_steps=10,
            save_strategy="no", report_to=[], seed=42,
            optim="adamw_torch", max_grad_norm=1.0)
        trainer = Trainer(model=peft_model, args=args, train_dataset=dset,
                          data_collator=DataCollatorForLanguageModeling(
                              tok, mlm=False))
        trainer.train()
        peft_model.save_pretrained(f"/content/adapter_{arm_name}")
        log(f"SFT arm {arm_name} done; adapter saved")
        base = peft_model.unload()
        del peft_model
        torch.cuda.empty_cache()
        return base

    base_model = model
    if arm_trunc:
        base_model = sft("truncated", arm_trunc, q_trunc)
    if arm_nat:
        base_model = sft("natural", arm_nat, q_nat)
    model = base_model
    model.config.use_cache = True

    # ---- Phase 3: eval ----
    log("== Phase 3: eval accuracy vs generation budget ==")
    eval_qs = list(ds_test.select(range(N_EVAL)))

    def eval_model(mdl, tag):
        rows = []
        mdl.eval()
        for budget in EVAL_BUDGETS:
            prompts_e = [FEWSHOT + f"Question: {ex['question']}\nSolution:"
                         for ex in eval_qs]
            bs = 4 if budget >= 1024 else 8
            completions_e = batch_generate(prompts_e, budget, batch=bs)
            correct, gen_toks, fails = 0, 0, 0
            for ex, comp in zip(eval_qs, completions_e):
                ans = extract_answer(comp)
                gold = float(ex["answer"].split("####")[-1].strip()
                             .replace(",", ""))
                if ans is None:
                    fails += 1
                elif abs(ans - gold) < 1e-6:
                    correct += 1
                gen_toks += len(tok(comp)["input_ids"])
            row = {"model": tag, "budget": budget, "n": len(eval_qs),
                   "accuracy": round(correct / len(eval_qs), 4),
                   "mean_gen_tokens": round(gen_toks / len(eval_qs), 1),
                   "parse_fail_rate": round(fails / len(eval_qs), 4)}
            rows.append(row)
            log(f"eval {tag} budget={budget}: acc={row['accuracy']} "
                f"mean_tok={row['mean_gen_tokens']} fail={row['parse_fail_rate']}")
            torch.cuda.empty_cache()
        return rows

    all_rows = []
    all_rows += eval_model(model, "base")
    from peft import PeftModel
    for arm_name in ("truncated", "natural"):
        path = f"/content/adapter_{arm_name}"
        if not os.path.isdir(path):
            log(f"skip eval arm {arm_name}: no adapter")
            continue
        mdl = PeftModel.from_pretrained(model, path)
        all_rows += eval_model(mdl, arm_name)
        del mdl
        torch.cuda.empty_cache()

    with open("/content/budget_internalization_metrics.csv", "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "budget", "n",
                                           "accuracy", "mean_gen_tokens",
                                           "parse_fail_rate"])
        w.writeheader()
        w.writerows(all_rows)
    summary = {"model": MODEL, "budget_B": BUDGET,
               "n_train_problems": N_TRAIN,
               "usable_traces": len(usable),
               "arm_sizes": {"truncated": len(arm_trunc),
                             "natural": len(arm_nat)},
               "n_eval": N_EVAL, "rows": all_rows}
    with open("/content/budget_internalization_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log("== metrics written to /content ==")
    log("TRAIN_DONE")


if __name__ == "__main__":
    install()
    main()
