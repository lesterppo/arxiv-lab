#!/usr/bin/env python3
"""Agentic-OPD (arXiv:2611.12799) -- SFT-stage checkpoint-pool variant.

Paper claim: in a training run, the best checkpoint fails on problems earlier
checkpoints solved; the union of solved problems across checkpoints far exceeds
any single one. Agentic-OPD treats the best-validation checkpoint as student
and all saved checkpoints as frozen teacher candidates: HCN (Highest Coverage
Next, Eq. 7-9) selects teachers by weighted coverage of the student's unsolved
problems; LAD (Locality-Aware Distillation, Eq. 10-13) confines transfer to
failed student rollouts on problems where the teacher performs better.

DELIBERATE DEVIATION (documented limit): the paper's prerequisite is a full
GRPO run producing the checkpoint pool. That is too heavy for one free-T4
session, so this test builds the pool with sequential SFT instead:
  Stage A: SFT base -> C1 on task A (GSM8K CoT, 400 ex).
  Stage B: continue SFT C1 -> C2 on task B (SVAMP answer-only, 400 ex),
           a different distribution + conflicting (short-answer) format
           that induces forgetting of A.
  Stage C: Agentic-OPD with student=C2, teacher pool={base, C1}.
LAD's turn-level top-k KL (Eq. 13) is adapted to SFT on the teacher's
correct sampled rollout text for LAD-admitted problems (loss on completion
tokens only) -- a bounded, stated simplification of the KL objective.

NAIVE CONTROL (this driver): the ablation that closes the attribution gap
in the run-f README ("no naive-distillation control"). Same experiment --
same data, stages, evals (A_val/A_test/B_test, n=60), seeds, LoRA config,
same student C2, same 100-problem pool, same G=4 sampling -- with TWO
differences only:
  (1) NO HCN: teacher fixed to C1. Rationale: HCN selected C1 in run f
      anyway (true gains {base: 4, C1: 23} on A_val, recomputed offline),
      and the paper's naive baseline distills the best checkpoint without
      selection; the value of HCN's selection is therefore already
      measured separately as the base-vs-C1 gain gap, not confounded into
      this arm.
  (2) NO LAD: distill on ALL pool problems that have a correct teacher
      trace (no "teacher beats student" admission filter). Student
      rollouts are still sampled, purely to report the attribution split
      (how many naive examples fall on problems LAD would have admitted
      vs rejected); they do not affect example selection.
The comparison that matters: naive C3' vs HCN+LAD C3 (43/33/52) vs C2
(24/21/46) -- how much of the +31.7/+20.0/+10.0 pp recovery survives
without HCN+LAD filtering.
No adapters from run f were banked (see session README), so Stages A/B
are retrained from scratch here; wall time ~2h-2h15m like run f.
Evening launch: reclamation risk is real; metrics + results flush after
every phase and the host-side mid-run downloader polls them.

Model: Qwen/Qwen3-8B, NF4 QLoRA r=16/alpha=32, bf16, transformers<5.0,
no prepare_model_for_kbit_training, batch 4 / seq 512, expandable_segments.
Rationale: Qwen3.5-9B OOM'd at step 0/15 on 2026-11-05; 8B is the proven
fallback, and two sequential trainings + a distillation need the proven model.

Phases print progress; metrics CSV + results JSON are flushed to /content
after EVERY phase (reclamation-safe). Final stdout line is TRAIN_DONE.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import math
import random
import re
import subprocess
import sys
import time
import urllib.request

import pandas as pd

METRICS_CSV = "/content/agentic_opd_metrics_naive.csv"
RESULTS_JSON = "/content/agentic_opd_results_naive.json"
MODEL_ID = "Qwen/Qwen3-8B"
SEED = 123
random.seed(SEED)

RESULTS = {
    "paper": "2611.12799",
    "title": "Agentic-OPD: Recovering Forgotten Agent Capabilities via On-Policy Distillation",
    "variant": "NAIVE-DISTILLATION CONTROL (no HCN, no LAD) of the SFT-stage checkpoint-pool variant",
    "control": {
        "teacher": "C1 (fixed; HCN selected C1 in run f anyway, true gains {base: 4, C1: 23})",
        "admission": "none: every pool problem with >=1 correct teacher trace is distilled",
        "student_sampling": "still sampled (G=4) for the attribution split only; does not affect selection",
        "compares_against": {"C3_hcn_lad": {"A_val": 43, "A_test": 33, "B_test": 52},
                              "C2": {"A_val": 24, "A_test": 21, "B_test": 46}},
    },
    "model": MODEL_ID,
    "model_rationale": "Qwen3.5-9B OOM'd at step 0/15 on 2026-11-05; Qwen3-8B is the proven fallback; two sequential SFT stages + distillation need the proven model",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target": "q,k,v,o,gate,up,down"},
    "evals": {},
    "pool_observation": {},
    "naive": {},
    "opd": {},
    "phases": [],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flush_results():
    with open(RESULTS_JSON, "w") as f:
        json.dump(RESULTS, f, indent=2)


def record_metric(phase, model, split, n, correct, notes=""):
    acc = correct / n if n else 0.0
    exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["phase", "model", "split", "n", "correct", "acc", "notes"])
        w.writerow([phase, model, split, n, correct, f"{acc:.4f}", notes])
    log(f"METRIC {phase} {model} {split}: {correct}/{n} = {acc:.4f} {notes}")
    return acc


# ---------------------------------------------------------------- data
def clean_gsm8k_answer(ans):
    ans = re.sub(r"<<[^>]*>>", "", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def gold_number_gsm8k(ans):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans)
    return float(m.group(1).replace(",", "")) if m else None


def load_data():
    log("downloading GSM8K + SVAMP parquet from HF hub")
    for name, url in [
        ("gsm8k_train", "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"),
        ("gsm8k_test", "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"),
        ("svamp_train", "https://huggingface.co/datasets/ChilleD/SVAMP/resolve/main/data/train-00000-of-00001.parquet"),
        ("svamp_test", "https://huggingface.co/datasets/ChilleD/SVAMP/resolve/main/data/test-00000-of-00001.parquet"),
    ]:
        urllib.request.urlretrieve(url, f"/tmp/{name}.parquet")
    gtr = pd.read_parquet("/tmp/gsm8k_train.parquet")
    gte = pd.read_parquet("/tmp/gsm8k_test.parquet")
    str_ = pd.read_parquet("/tmp/svamp_train.parquet")
    ste = pd.read_parquet("/tmp/svamp_test.parquet")
    # schema validation (DAPO lesson: validate real fields before training)
    assert list(gtr.columns) == ["question", "answer"], gtr.columns
    assert list(gte.columns) == ["question", "answer"], gte.columns
    assert "question_concat" in str_.columns and "Answer" in str_.columns
    assert sum("####" in a for a in gtr["answer"]) == len(gtr)
    assert sum("####" in a for a in gte["answer"]) == len(gte)
    log(f"data OK: gsm8k train {len(gtr)} test {len(gte)}; svamp train {len(str_)} test {len(ste)}")

    def prompt(q):
        return f"{q}\nSolution:"

    # Stage A train: GSM8K train rows 0..399, CoT target
    a_train = []
    for _, r in gtr.iloc[0:400].iterrows():
        a_train.append({"prompt": prompt(r["question"]),
                        "completion": " " + clean_gsm8k_answer(r["answer"]),
                        "gold": gold_number_gsm8k(r["answer"])})
    # Stage B train: SVAMP train rows 0..399, answer-only target
    b_train = []
    for _, r in str_.iloc[0:400].iterrows():
        b_train.append({"prompt": prompt(r["question_concat"]),
                        "completion": " " + str(r["Answer"]).strip(),
                        "gold": float(str(r["Answer"]).strip().replace(",", ""))})
    # Eval sets
    a_val = [{"prompt": prompt(r["question"]), "gold": gold_number_gsm8k(r["answer"])}
             for _, r in gte.iloc[0:60].iterrows()]
    a_test = [{"prompt": prompt(r["question"]), "gold": gold_number_gsm8k(r["answer"])}
              for _, r in gte.iloc[100:160].iterrows()]
    b_test = [{"prompt": prompt(r["question_concat"]),
               "gold": float(str(r["Answer"]).strip().replace(",", ""))}
              for _, r in ste.iloc[0:60].iterrows()]
    # LAD pool: GSM8K train rows 400..599 (disjoint from stage A), 200 problems
    lad_pool = [{"prompt": prompt(r["question"]),
                 "gold": gold_number_gsm8k(r["answer"]),
                 "cot": clean_gsm8k_answer(r["answer"])}
                for _, r in gtr.iloc[400:500].iterrows()]
    log(f"sets: A_train {len(a_train)} B_train {len(b_train)} A_val {len(a_val)} "
        f"A_test {len(a_test)} B_test {len(b_test)} LAD_pool {len(lad_pool)}")
    return a_train, b_train, a_val, a_test, b_test, lad_pool


# ---------------------------------------------------------------- model
def install():
    log("installing deps")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.46,<5.0", "bitsandbytes", "accelerate",
                    "peft", "datasets", "hf_transfer"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False)
    log("deps installed")


def build_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    log("loading tokenizer + base model (NF4)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    log(f"VRAM after load: {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated, "
        f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB peak")
    return tok, model


# ---------------------------------------------------------------- train/eval helpers
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_number(text):
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    nums = NUM_RE.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def numbers_equal(pred, gold):
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= 1e-6 * max(1.0, abs(gold))


def make_batches(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]


def train_sft(tok, model, examples, lr, epochs, tag):
    """Plain SFT loop, HF built-in loss (proven soup recipe), batch 4 / seq 512."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_cosine_schedule_with_warmup

    class SFTData(Dataset):
        def __init__(self, exs):
            self.rows = []
            for e in exs:
                p = tok(e["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(e["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
                ids = (p + c)[:512]
                labels = ([-100] * len(p) + c)[:512]
                self.rows.append((ids, labels))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        mx = max(len(ids) for ids, _ in batch)
        pad = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for ids, lab in batch:
            n = mx - len(ids)
            input_ids.append(ids + [pad] * n)
            labels.append(lab + [-100] * n)
            attn.append([1] * len(ids) + [0] * n)
        return (torch.tensor(input_ids), torch.tensor(labels),
                torch.tensor(attn))

    # OOM fix (2026-11-12 run e): without gradient checkpointing, bnb's
    # dequantized bf16 weights are retained for backward across all layers
    # (~14 GiB). Checkpointing recomputes them per segment instead.
    torch.cuda.empty_cache()
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    model.train()
    model.config.use_cache = False
    GRAD_ACCUM = 2  # micro-batch 2 x accum 2 = effective batch 4
    dl = DataLoader(SFTData(examples), batch_size=2, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=0.0)
    total = (len(dl) * epochs + GRAD_ACCUM - 1) // GRAD_ACCUM
    sched = get_cosine_schedule_with_warmup(opt, 3, total)
    step = 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        for mbi, (input_ids, labels, attn) in enumerate(dl):
            input_ids, labels, attn = (input_ids.cuda(), labels.cuda(), attn.cuda())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / GRAD_ACCUM).backward()
            if (mbi + 1) % GRAD_ACCUM == 0 or mbi == len(dl) - 1:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or step == total:
                    log(f"{tag} step {step}/{total} loss {out.loss.item():.4f} "
                        f"elapsed {time.time()-t0:.0f}s "
                        f"VRAM {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    log(f"{tag} done: {step} steps, {time.time()-t0:.0f}s, "
        f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    return step


def run_eval(tok, model, items, max_new, bs=24, tag="eval"):
    """Greedy eval; returns (correct, solved_list)."""
    import torch
    model.eval()
    model.config.use_cache = True  # training sets it False; generation needs KV cache
    tok.padding_side = "left"
    solved, correct = [], 0
    for batch in make_batches(items, bs):
        prompts = [b["prompt"] for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        log(f"{tag} batch {len(solved)//len(batch)+1} done")
        for i, b in enumerate(batch):
            gen = tok.decode(out[i][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            ok = numbers_equal(extract_number(gen), b["gold"])
            solved.append(1 if ok else 0)
            correct += 1 if ok else 0
        torch.cuda.empty_cache()
    tok.padding_side = "right"
    log(f"{tag}: {correct}/{len(items)}")
    return correct, solved


def run_sample(tok, model, items, g, max_new, bs=8, temp=0.8):
    """Sample G rollouts per problem; returns list of (prompt, [completion,...])."""
    import torch
    model.eval()
    model.config.use_cache = True  # training sets it False; generation needs KV cache
    tok.padding_side = "left"
    out_rows = []
    for batch in make_batches(items, bs):
        prompts = [b["prompt"] for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                 temperature=temp, top_p=0.95, num_return_sequences=g,
                                 pad_token_id=tok.eos_token_id)
        plen = enc["input_ids"].shape[1]
        for i, b in enumerate(batch):
            comps = []
            for j in range(g):
                comps.append(tok.decode(out[i * g + j][plen:], skip_special_tokens=True))
            out_rows.append((b, comps))
        torch.cuda.empty_cache()
    tok.padding_side = "right"
    return out_rows


def build_naive_examples(rows):
    """Naive pool selection: every problem with >=1 correct teacher trace
    contributes its first correct teacher rollout as an SFT example.
    Student rollouts are used ONLY for the attribution split (how many
    naive examples fall on problems LAD would have admitted, i.e. teacher
    beats student, vs rejected). Returns (examples, stats). Pure -- no
    torch, unit-testable on CPU.
    rows: list of (problem_dict, student_completions, teacher_completions),
    each completion a raw generated string; problem_dict has "gold"."""
    examples, stats = [], {"pool_n": len(rows), "covered_n": 0,
                           "examples_n": 0, "mean_s": 0.0, "mean_q": 0.0,
                           "lad_admitted_equiv_n": 0, "lad_rejected_equiv_n": 0}
    s_sum, q_sum = 0.0, 0.0
    for b, s_comps, t_comps in rows:
        s_rate = sum(1 for c in s_comps
                     if numbers_equal(extract_number(c), b["gold"])) / len(s_comps)
        q_rate = sum(1 for c in t_comps
                     if numbers_equal(extract_number(c), b["gold"])) / len(t_comps)
        s_sum += s_rate
        q_sum += q_rate
        correct_traces = [c for c in t_comps
                          if numbers_equal(extract_number(c), b["gold"])]
        if correct_traces:
            stats["covered_n"] += 1
            if q_rate > s_rate:
                stats["lad_admitted_equiv_n"] += 1
            else:
                stats["lad_rejected_equiv_n"] += 1
            examples.append({"prompt": b["prompt"],
                             "completion": " " + correct_traces[0].strip(),
                             "gold": b["gold"]})
    stats["examples_n"] = len(examples)
    if rows:
        stats["mean_s"] = s_sum / len(rows)
        stats["mean_q"] = q_sum / len(rows)
    return examples, stats


# ---------------------------------------------------------------- main
def main():
    install()
    import torch
    a_train, b_train, a_val, a_test, b_test, lad_pool = load_data()
    tok, model = build_model()

    def set_adapter(name):
        # NOTE (run-f lesson): peft disable_adapter() called bare is a
        # no-op -- it returns a context manager that must be entered.
        # This driver never evaluates with the adapter disabled (no HCN
        # phase), so set_adapter is only ever called with a real name;
        # fail loudly on None instead of silently running the wrong model.
        assert name is not None, "set_adapter(None) is a no-op trap; use eval_all(None) instead"
        model.set_adapter(name)

    def eval_all(tag, adapter_name):
        # Base eval (adapter_name None) runs with the adapter PROPERLY
        # disabled via the context manager; with the zero-init LoRA this
        # is the base model either way, but the context form is the
        # correct one (run f's bare call evaluated as C2 by accident).
        import contextlib
        ctx = model.disable_adapter() if adapter_name is None else contextlib.nullcontext()
        with ctx:
            if adapter_name is not None:
                set_adapter(adapter_name)
            res = {}
            for split, items, mx in [("A_val", a_val, 120), ("A_test", a_test, 120),
                                     ("B_test", b_test, 48)]:
                c, solved = run_eval(tok, model, items, mx, tag=f"{tag}/{split}")
                record_metric(tag, adapter_name or "base", split, len(items), c)
                res[split] = {"n": len(items), "correct": c,
                              "acc": c / len(items), "solved": solved}
            RESULTS["evals"][tag] = res
            flush_results()
        return res

    # --- base eval
    log("PHASE 0: base eval")
    base = eval_all("base", None)
    RESULTS["phases"].append("base_eval")
    flush_results()

    # --- Stage A: SFT on task A -> C1
    log("PHASE A: SFT on GSM8K CoT (task A) -> C1")
    train_sft(tok, model, a_train, lr=2e-4, epochs=1, tag="stageA")
    model.save_pretrained("/content/adapter_c1")
    RESULTS["phases"].append("stageA")
    flush_results()
    c1 = eval_all("C1", "default")
    torch.cuda.empty_cache()

    # --- Stage B: continue SFT on task B -> C2
    log("PHASE B: continue SFT on SVAMP answer-only (task B) -> C2")
    train_sft(tok, model, b_train, lr=2e-4, epochs=1, tag="stageB")
    model.save_pretrained("/content/adapter_c2")
    RESULTS["phases"].append("stageB")
    flush_results()
    c2 = eval_all("C2", "default")
    torch.cuda.empty_cache()

    # --- Paper observation: checkpoint-pool union on A
    log("PHASE POOL: union coverage on A_val / A_test")
    for split in ["A_val", "A_test"]:
        u = [a or b for a, b in zip(c1[split]["solved"], c2[split]["solved"])]
        gap = sum(u) - c2[split]["correct"]
        RESULTS["pool_observation"][split] = {
            "C1_solved": c1[split]["correct"], "C2_solved": c2[split]["correct"],
            "union_solved": sum(u), "gap_union_minus_C2": gap,
            "forgotten_by_C2": sum(1 for a, b in zip(c1[split]["solved"], c2[split]["solved"]) if a == 1 and b == 0),
        }
        log(f"POOL {split}: C1 {c1[split]['correct']} C2 {c2[split]['correct']} "
            f"union {sum(u)} gap +{gap}")
    flush_results()

    # --- NAIVE POOL (control: NO HCN, NO LAD): teacher fixed to C1.
    # HCN is skipped entirely -- run f selected C1 anyway (true gains
    # {base: 4, C1: 23}), so the naive teacher is C1 by construction.
    # Sample G=4 from student C2 AND teacher C1 on the same 100-problem
    # pool; distill on every problem with >=1 correct teacher trace.
    log(f"PHASE NAIVE-POOL: G=4 rollouts, student C2 vs teacher C1 (fixed), pool n={len(lad_pool)}")
    model.load_adapter("/content/adapter_c1", adapter_name="c1")
    set_adapter("default")  # student C2
    s_rows = run_sample(tok, model, lad_pool, g=4, max_new=120)
    set_adapter("c1")  # teacher C1
    t_rows = run_sample(tok, model, lad_pool, g=4, max_new=120)
    rows = [(b, s_comps, t_comps)
            for (b, s_comps), (_, t_comps) in zip(s_rows, t_rows)]
    opd_examples, naive_stats = build_naive_examples(rows)
    RESULTS["naive"] = {"pool_n": naive_stats["pool_n"], "G": 4,
                        "teacher": "C1 (fixed, no HCN)",
                        "covered_n": naive_stats["covered_n"],
                        "opd_examples_n": naive_stats["examples_n"],
                        "mean_s": naive_stats["mean_s"],
                        "mean_q": naive_stats["mean_q"],
                        "lad_admitted_equiv_n": naive_stats["lad_admitted_equiv_n"],
                        "lad_rejected_equiv_n": naive_stats["lad_rejected_equiv_n"],
                        "run_f_lad_admitted_n": 65}
    record_metric("NAIVE", "C2-vs-C1", "pool", len(lad_pool),
                  naive_stats["covered_n"],
                  notes=f"covered {naive_stats['covered_n']}/{len(lad_pool)} "
                  f"mean_s {naive_stats['mean_s']:.3f} mean_q {naive_stats['mean_q']:.3f} "
                  f"LAD-equiv admitted {naive_stats['lad_admitted_equiv_n']} "
                  f"rejected {naive_stats['lad_rejected_equiv_n']}")
    log(f"NAIVE-POOL: covered {naive_stats['covered_n']}/{len(lad_pool)} problems, "
        f"{len(opd_examples)} teacher traces, mean_s {naive_stats['mean_s']:.3f} "
        f"mean_q {naive_stats['mean_q']:.3f}; "
        f"LAD would have admitted {naive_stats['lad_admitted_equiv_n']}, "
        f"rejected {naive_stats['lad_rejected_equiv_n']}")
    RESULTS["phases"].append("naive-pool")
    flush_results()
    torch.cuda.empty_cache()

    # --- Stage C: NAIVE distillation (all covered teacher traces, no LAD filter)
    if opd_examples:
        log(f"PHASE C: naive distillation on {len(opd_examples)} teacher traces (no LAD filter)")
        set_adapter("default")  # student C2
        train_sft(tok, model, opd_examples, lr=1e-4, epochs=1, tag="stageC-naive")
        model.save_pretrained("/content/adapter_c3naive")
        import shutil
        shutil.make_archive("/content/agentic_opd_adapter_naive", "zip",
                            "/content/adapter_c3naive")
        RESULTS["phases"].append("stageC-naive")
        flush_results()
        c3 = eval_all("C3naive", "default")
        RESULTS["opd"] = {
            "recovery_A_test": c3["A_test"]["acc"] - c2["A_test"]["acc"],
            "recovery_A_val": c3["A_val"]["acc"] - c2["A_val"]["acc"],
            "retention_B_test": c3["B_test"]["acc"] - c2["B_test"]["acc"],
            "vs_C1_A_test": c3["A_test"]["acc"] - c1["A_test"]["acc"],
        }
        log(f"NAIVE-C3: A_test {c2['A_test']['acc']:.3f} -> {c3['A_test']['acc']:.3f} "
            f"(+{RESULTS['opd']['recovery_A_test']:.3f}); "
            f"B_test {c2['B_test']['acc']:.3f} -> {c3['B_test']['acc']:.3f} "
            f"({RESULTS['opd']['retention_B_test']:+.3f})")
    else:
        log("PHASE C: no covered problems -- naive distillation skipped, verdict INCONCLUSIVE path")
        RESULTS["opd"] = {"skipped": "no covered problems"}
    flush_results()

    log("ALL PHASES DONE")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
