#!/usr/bin/env python3
"""SFT data source/mixture (arXiv:2611.14912) -- "Rethinking 'RL Generalizes,
SFT Memorizes': The Role of SFT Data".

Paper claims (Qwen2.5/Qwen3): (1) data SOURCE is critical -- a carefully
mixed SFT dataset substantially outperforms data generated solely by a
larger model; (2) data SCALE -- matching the number of correct rollouts
between SFT and RL greatly improves SFT generalization; (3) larger datasets
cover more tail solution methods, which carry generalizable signal.

Bounded test of claim (1) only (the SFT-vs-RL arms of claim (2) need a full
RL run and are out of reach in one free-T4 session): two SFT arms on GSM8K,
matched on every confound -- same N, same # correct rollouts (all examples
correct by construction), same epochs/lr/batch, same base model:
  Arm S (single-source): N self-generated correct rollouts only
      (rejection-sampled from the base model itself -- the paper's
      "data generated solely by a [single] model" condition).
  Arm M (mixed): N/3 GSM8K human-written solutions + N/3 self-generated
      correct rollouts + N/3 equation-format solutions (calculator
      expressions extracted from human solutions -- a format-diverse
      third source).
Eval: greedy GSM8K test + SVAMP test (transfer/generalization), base model
as reference. If M beats S on the held-out transfer split at matched N,
claim (1) is directionally reproduced.

Model: Qwen/Qwen3.5-9B, NF4 QLoRA r=16/alpha=32, bf16, transformers<5.0,
grad checkpointing + micro-batch 2 (the 2026-11-12 Agentic-OPD OOM fix).
Standing rule (2026-10-27): try Qwen3.5-9B first; on OOM/provision failure
the 8B fallback driver (same code, MODEL_ID Qwen/Qwen3-8B) is staged in
this session. Qwen3.5-9B probe was GREEN 2026-10-30/31 (7.88 GiB loaded);
its full-trainer OOMs predate the grad-checkpointing fix.

Phases print progress; metrics CSV + results JSON are flushed to /content
after EVERY phase (reclamation-safe). Final stdout line is TRAIN_DONE.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import random
import re
import subprocess
import sys
import time
import urllib.request

import pandas as pd

METRICS_CSV = "/content/sft_mixture_metrics.csv"
RESULTS_JSON = "/content/sft_mixture_results.json"
MODEL_ID = "Qwen/Qwen3.5-9B"
SEED = 123
random.seed(SEED)

N_ARM = 480
EPOCHS = 2
LR = 2e-4
EVAL_N = 150
GEN_QUESTIONS = 350
GEN_G = 4

RESULTS = {
    "paper": "2611.14912",
    "title": "Rethinking 'RL Generalizes, SFT Memorizes': The Role of SFT Data",
    "variant": "Bounded test of claim (1) only: single-source vs mixed SFT data, matched N/correct-count/epochs/lr",
    "model": MODEL_ID,
    "model_rationale": "Standing rule: try Qwen3.5-9B first (probe GREEN 2026-10-30/31); 8B fallback driver staged; grad-checkpointing fix from 2026-11-12 applied",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target": "q,k,v,o,gate,up,down"},
    "arms": {"N": N_ARM, "epochs": EPOCHS, "lr": LR},
    "evals": {},
    "generation": {},
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


def equation_format(ans, gold):
    """Third source: calculator expressions extracted into a compact chain.

    'Janet has 3 apples... <<3*4=12>> ... <<12+5=17>> #### 17' becomes
    '3*4=12; 12+5=17\n#### 17'. Falls back to the cleaned human solution
    when a row carries no calculator annotations.
    """
    exprs = re.findall(r"<<([^>]*)>>", ans)
    if not exprs:
        return clean_gsm8k_answer(ans)
    chain = "; ".join(e.strip() for e in exprs)
    return f"{chain}\n#### {gold:g}" if float(gold).is_integer() else f"{chain}\n#### {gold}"


def load_data():
    log("downloading GSM8K + SVAMP parquet from HF hub")
    for name, url in [
        ("gsm8k_train", "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"),
        ("gsm8k_test", "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"),
        ("svamp_test", "https://huggingface.co/datasets/ChilleD/SVAMP/resolve/main/data/test-00000-of-00001.parquet"),
    ]:
        urllib.request.urlretrieve(url, f"/tmp/{name}.parquet")
    gtr = pd.read_parquet("/tmp/gsm8k_train.parquet")
    gte = pd.read_parquet("/tmp/gsm8k_test.parquet")
    ste = pd.read_parquet("/tmp/svamp_test.parquet")
    assert list(gtr.columns) == ["question", "answer"], gtr.columns
    assert list(gte.columns) == ["question", "answer"], gte.columns
    assert "question_concat" in ste.columns and "Answer" in ste.columns
    assert sum("####" in a for a in gtr["answer"]) == len(gtr)
    assert sum("####" in a for a in gte["answer"]) == len(gte)
    log(f"data OK: gsm8k train {len(gtr)} test {len(gte)}; svamp test {len(ste)}")

    def prompt(q):
        return f"{q}\nSolution:"

    # Eval sets (disjoint from every train row used below)
    gsm_test = [{"prompt": prompt(r["question"]), "gold": gold_number_gsm8k(r["answer"])}
                for _, r in gte.iloc[0:EVAL_N].iterrows()]
    svamp_test = [{"prompt": prompt(r["question_concat"]),
                   "gold": float(str(r["Answer"]).strip().replace(",", ""))}
                  for _, r in ste.iloc[0:EVAL_N].iterrows()]
    # Human source: GSM8K train rows 0..319 (160 human + 160 equation-format)
    human_rows = gtr.iloc[0:160]
    eq_rows = gtr.iloc[160:320]
    # Self-generation pool: GSM8K train rows 1000..1349 (disjoint from above).
    # 350 questions x cap-2 kept = up to 700 correct rollouts; arms need 640.
    gen_rows = gtr.iloc[1000:1000 + GEN_QUESTIONS]
    gen_pool = [{"prompt": prompt(r["question"]), "gold": gold_number_gsm8k(r["answer"])}
                for _, r in gen_rows.iterrows()]
    log(f"sets: gsm_test {len(gsm_test)} svamp_test {len(svamp_test)} "
        f"human {len(human_rows)} eq {len(eq_rows)} gen_pool {len(gen_pool)}")
    return gtr, gsm_test, svamp_test, human_rows, eq_rows, gen_pool


def build_arms(human_rows, eq_rows, selfgen_correct):
    """Build the two matched arms. `selfgen_correct` is a list of
    (prompt, completion, gold) with verified-correct completions.
    Returns (arm_s, arm_m, counts dict). Raises if the self-gen pool is short.
    """
    need_self_s = N_ARM
    need_self_m = N_ARM // 3
    if len(selfgen_correct) < need_self_s + need_self_m:
        raise RuntimeError(
            f"self-gen pool short: have {len(selfgen_correct)}, "
            f"need {need_self_s + need_self_m}")

    def prompt_of(q):
        return f"{q}\nSolution:"

    arm_s = [{"prompt": p, "completion": " " + c.strip(), "gold": g}
             for (p, c, g) in selfgen_correct[:need_self_s]]
    human_ex = [{"prompt": prompt_of(r["question"]),
                 "completion": " " + clean_gsm8k_answer(r["answer"]),
                 "gold": gold_number_gsm8k(r["answer"])}
                for _, r in human_rows.iterrows()]
    eq_ex = [{"prompt": prompt_of(r["question"]),
              "completion": " " + equation_format(r["answer"], gold_number_gsm8k(r["answer"])),
              "gold": gold_number_gsm8k(r["answer"])}
             for _, r in eq_rows.iterrows()]
    self_m = [{"prompt": p, "completion": " " + c.strip(), "gold": g}
              for (p, c, g) in selfgen_correct[need_self_s:need_self_s + need_self_m]]
    arm_m = human_ex + eq_ex + self_m
    random.shuffle(arm_m)
    assert len(arm_s) == N_ARM and len(arm_m) == N_ARM
    counts = {"arm_s_selfgen": len(arm_s),
              "arm_m_human": len(human_ex), "arm_m_eq": len(eq_ex),
              "arm_m_selfgen": len(self_m)}
    return arm_s, arm_m, counts


# ---------------------------------------------------------------- model
def install():
    log("installing deps")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==5.20.2"  # 5.x REQUIRED for Qwen3.5 (qwen3_5 arch); 4.x dies with KeyError qwen3_5 (2026-11-13), "bitsandbytes", "accelerate",
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
    """Plain SFT loop, HF built-in loss (proven soup recipe), micro-batch 2."""
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
    # dequantized bf16 weights are retained for backward across all layers.
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
    model.config.use_cache = True
    tok.padding_side = "left"
    solved, correct = [], 0
    for batch in make_batches(items, bs):
        prompts = [b["prompt"] for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
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
    """Sample G rollouts per problem; returns list of (item, [completions])."""
    import torch
    model.eval()
    model.config.use_cache = True
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


# ---------------------------------------------------------------- main
def main():
    install()
    import gc
    import torch
    gtr, gsm_test, svamp_test, human_rows, eq_rows, gen_pool = load_data()
    tok, model = build_model()

    def eval_all(tag, adapter_active):
        # adapter_active True -> LoRA on; False -> base (adapter disabled).
        # NOTE: peft disable_adapter() bare call is a no-op trap (2026-11-12
        # run f lesson) -- use the context manager instead.
        res = {}
        for split, items, mx in [("gsm8k_test", gsm_test, 160),
                                 ("svamp_test", svamp_test, 80)]:
            if adapter_active:
                c, solved = run_eval(tok, model, items, mx, tag=f"{tag}/{split}")
            else:
                with model.disable_adapter():
                    c, solved = run_eval(tok, model, items, mx, tag=f"{tag}/{split}")
            record_metric(tag, "adapter" if adapter_active else "base",
                          split, len(items), c)
            res[split] = {"n": len(items), "correct": c,
                          "acc": c / len(items), "solved": solved}
        RESULTS["evals"][tag] = res
        flush_results()
        return res

    # --- base eval
    log("PHASE 0: base eval")
    base = eval_all("base", False)
    RESULTS["phases"].append("base_eval")
    flush_results()

    # --- generation: self-generated correct rollouts from the base model
    log(f"PHASE GEN: sample G={GEN_G} on {len(gen_pool)} GSM8K-train questions")
    with model.disable_adapter():
        rows = run_sample(tok, model, gen_pool, g=GEN_G, max_new=160, temp=0.8)
    selfgen_correct = []
    n_q_with_correct = 0
    for b, comps in rows:
        kept = 0
        for c in comps:
            if numbers_equal(extract_number(c), b["gold"]) and kept < 2:
                selfgen_correct.append((b["prompt"], c, b["gold"]))
                kept += 1
        if kept:
            n_q_with_correct += 1
    RESULTS["generation"] = {
        "questions": len(gen_pool), "G": GEN_G, "temp": 0.8,
        "questions_with_correct": n_q_with_correct,
        "correct_rollouts_kept": len(selfgen_correct),
        "need": N_ARM + N_ARM // 3,
    }
    log(f"GEN: {n_q_with_correct}/{len(gen_pool)} questions yielded a correct "
        f"rollout; kept {len(selfgen_correct)} (need {N_ARM + N_ARM // 3})")
    RESULTS["phases"].append("generation")
    flush_results()

    arm_s, arm_m, counts = build_arms(human_rows, eq_rows, selfgen_correct)
    RESULTS["arms"]["counts"] = counts
    log(f"ARMS: S={len(arm_s)} (self-gen only); M={len(arm_m)} "
        f"(human {counts['arm_m_human']} + eq {counts['arm_m_eq']} + "
        f"self-gen {counts['arm_m_selfgen']})")
    flush_results()

    # --- Arm S: single-source SFT
    log("PHASE S: SFT single-source (self-gen only) -> adapter_s")
    train_sft(tok, model, arm_s, lr=LR, epochs=EPOCHS, tag="armS")
    model.save_pretrained("/content/adapter_s")
    RESULTS["phases"].append("armS_train")
    flush_results()
    eval_s = eval_all("armS", True)
    torch.cuda.empty_cache()

    # --- rebuild a fresh model+adapter for Arm M (no cross-arm contamination)
    log("PHASE REBUILD: fresh model for Arm M")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    tok, model = build_model()

    # --- Arm M: mixed-source SFT
    log("PHASE M: SFT mixed-source (human + eq + self-gen) -> adapter_m")
    train_sft(tok, model, arm_m, lr=LR, epochs=EPOCHS, tag="armM")
    model.save_pretrained("/content/adapter_m")
    RESULTS["phases"].append("armM_train")
    flush_results()
    eval_m = eval_all("armM", True)

    # --- headline comparison
    RESULTS["comparison"] = {
        "gsm8k_test": {"base": base["gsm8k_test"]["acc"],
                       "armS": eval_s["gsm8k_test"]["acc"],
                       "armM": eval_m["gsm8k_test"]["acc"],
                       "M_minus_S_pp": 100 * (eval_m["gsm8k_test"]["acc"] - eval_s["gsm8k_test"]["acc"])},
        "svamp_test": {"base": base["svamp_test"]["acc"],
                       "armS": eval_s["svamp_test"]["acc"],
                       "armM": eval_m["svamp_test"]["acc"],
                       "M_minus_S_pp": 100 * (eval_m["svamp_test"]["acc"] - eval_s["svamp_test"]["acc"])},
    }
    log(f"COMPARISON gsm8k: base {base['gsm8k_test']['acc']:.3f} "
        f"S {eval_s['gsm8k_test']['acc']:.3f} M {eval_m['gsm8k_test']['acc']:.3f} "
        f"(M-S {RESULTS['comparison']['gsm8k_test']['M_minus_S_pp']:+.1f}pp); "
        f"svamp: base {base['svamp_test']['acc']:.3f} "
        f"S {eval_s['svamp_test']['acc']:.3f} M {eval_m['svamp_test']['acc']:.3f} "
        f"(M-S {RESULTS['comparison']['svamp_test']['M_minus_S_pp']:+.1f}pp)")
    flush_results()

    log("ALL PHASES DONE")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
