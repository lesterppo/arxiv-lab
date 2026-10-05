"""
FTW vs GRPO A/B on Qwen3.5-9B (NF4 QLoRA) — synthetic integer arithmetic.
CHECKPOINT-RESUME version: saves a checkpoint to /content/ckpt.pt every
2 steps. The orchestrator (host VM) downloads each checkpoint; on reclaim
it uploads the latest one as /content/ckpt_resume.pt and the driver
restores adapter / optimizer / RNG / curve / replay buffer and continues.
Reclaims become annoying, not fatal.

Paper: arXiv:2610.03361 "Follow the Winners: Conservative Policy Improvement
with the Cross-Entropy Method for Critic-Free RFT".

Claims tested:
  (1) FTW matches GRPO final performance (held-out accuracy).
  (2) FTW uses less GPU memory (no critic, no per-prompt group rollouts;
      replay buffer lives on CPU).

Arms (same generation budget, same LR/LoRA/optimizer):
  --mode grpo : G=4 rollouts/prompt, per-prompt group-mean baseline,
                standardized advantages, clipped policy-gradient update.
  --mode ftw  : 1 rollout/prompt into a CPU replay buffer (FIFO, cap 256);
                each step an ORDINAL FILTER keeps the top 25% by return and
                a CEM-style weighted MLE update is applied toward the elite
                winners (r > 0 only).

Proxy limits (honest): synthetic multi-step arithmetic, NOT Sokoban/Search-R1;
QLoRA (r=16) not full fine-tune; 12 update steps; binary-ish reward.

Env (driver runs ON the Colab VM; no tunnel):
  RUN_ID     : e.g. ftw-grpo-20261005 (printed in logs; orchestrator keys
               checkpoints by it)
  FTW_MODE   : grpo | ftw
  MODEL_DIR  : optional local weights dir (skips download)

Self-provisioning on startup:
  1. Weights via aria2c (parallel segmented; skips if present).
     (hf_transfer is deprecated; HF_XET_HIGH_PERFORMANCE OOMs 12GB VMs.)
  2. Resume: if /content/ckpt_resume.pt exists, restore and continue.
     Prints RESUMED_FROM_STEP=N or FRESH_START.

Outputs: /content/ckpt.pt (every 2 steps), /content/ftw_<mode>_results.json,
         /content/ftw_<mode>_curve.csv
"""
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_CANDIDATES = ["Qwen/Qwen3.5-9B-Instruct", "Qwen/Qwen3.5-9B"]
HF_REPO = "Qwen/Qwen3.5-9B"
SEED = 11
N_TRAIN = 48
N_HELDOUT = 24
STEPS = 12
LR = 2e-5
CLIP = 0.2
LORA_R = 16
# GRPO: 3 prompts x 4 rollouts = 12 generations/step
GRPO_BATCH = 3
G = 4
# FTW: 12 prompts x 1 rollout = 12 generations/step (same generation budget)
FTW_BATCH = 12
BUF_CAP = 256
ELITE_FRAC = 0.25
ROLLOUT_TOKENS = 224
# Checkpointing (every step: sessions die fast, minimize loss window)
CKPT_EVERY = 1
CKPT_PATH = "/content/ckpt.pt"
RESUME_PATH = "/content/ckpt_resume.pt"
WEIGHT_FILES = [
    "model.safetensors-00001-of-00004.safetensors",
    "model.safetensors-00002-of-00004.safetensors",
    "model.safetensors-00003-of-00004.safetensors",
    "model.safetensors-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
]

PROMPT_TMPL = ("Solve the problem. Show brief reasoning, then put the final "
               "integer answer in \\boxed{{}}.\nProblem: {q}")


# ---------------------------------------------------------------- Weights
def _has_weights(d):
    return all(os.path.isfile(os.path.join(d, f)) for f in WEIGHT_FILES)


def ensure_weights():
    """Return a dir with the model files; aria2c-download if missing."""
    d = os.environ.get("MODEL_DIR")
    if d and _has_weights(d):
        print(f"weights OK (MODEL_DIR): {d}", flush=True)
        return d
    d = "/content/weights"
    if _has_weights(d):
        print(f"weights OK (cached): {d}", flush=True)
        return d
    os.makedirs(d, exist_ok=True)
    r = subprocess.run(["which", "aria2c"], capture_output=True)
    if r.returncode != 0:
        print("installing aria2...", flush=True)
        subprocess.run(["apt-get", "install", "-y", "-q", "aria2"],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    base = f"https://huggingface.co/{HF_REPO}/resolve/main/"
    t_dl = time.time()
    shards = [f for f in WEIGHT_FILES if f.endswith(".safetensors")]
    small = [f for f in WEIGHT_FILES if not f.endswith(".safetensors")]
    procs = []
    for f in shards:
        p = subprocess.Popen(
            ["aria2c", "-x", "8", "-s", "8", "-k", "1M",
             "-d", d, "-o", f, base + f],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append((f, p))
    for f, p in procs:
        rc = p.wait()
        if rc != 0 or not os.path.isfile(os.path.join(d, f)):
            raise RuntimeError(f"aria2c failed for {f} (rc={rc})")
        print(f"weights: {f} OK", flush=True)
    for f in small:
        subprocess.run(["curl", "-sL", "-o", os.path.join(d, f), base + f],
                       check=True)
    for f in small:
        if f.endswith(".json"):
            json.load(open(os.path.join(d, f)))  # validate
    total_gb = sum(os.path.getsize(os.path.join(d, f)) for f in shards) / 1e9
    print(f"weights complete: {d} ({total_gb:.1f} GB shards, "
          f"{(time.time()-t_dl)/60:.1f} min)", flush=True)
    return d


# ---------------------------------------------------------------- Checkpoint
def save_checkpoint(step, run_id, mode, model, opt, curve, buf_lists,
                    h_before, peak_so_far, wall_so_far, t0):
    ckpt = {
        "step": step,
        "run_id": run_id,
        "mode": mode,
        "h_before": h_before,
        "curve": curve,
        "peak_gb_so_far": max(peak_so_far,
                              torch.cuda.max_memory_allocated() / 1e9),
        "wall_so_far_min": wall_so_far + (time.time() - t0) / 60,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "lora": {k: v.cpu() for k, v in model.state_dict().items()
                 if "lora_" in k},
        "opt": opt.state_dict(),
        "buf": buf_lists,
    }
    torch.save(ckpt, CKPT_PATH)
    mb = os.path.getsize(CKPT_PATH) / 1e6
    print(f"CKPT_SAVED step={step} ({mb:.0f} MB) -> {CKPT_PATH}", flush=True)


def load_resume_ckpt():
    if not os.path.isfile(RESUME_PATH):
        return None
    return torch.load(RESUME_PATH, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- Model / task (unchanged)
def make_problems(n, seed):
    """Synthetic multi-step integer arithmetic; answers computed exactly."""
    rng = random.Random(seed)
    items, seen = [], set()
    while len(items) < n:
        t = rng.randrange(6)
        if t == 0:
            a = rng.randint(200, 1999); b = rng.randint(200, 1999)
            c = rng.randint(12, 98); d = rng.randint(1000, 99999)
            q = f"Compute ({a} + {b}) x {c} - {d}."
            ans = (a + b) * c - d
        elif t == 1:
            a = rng.randint(25, 250); b = rng.randint(25, 250)
            c = rng.randint(25, 250); d = rng.randint(25, 250)
            e = rng.randint(1000, 50000)
            q = f"Compute {a} x {b} + {c} x {d} - {e}."
            ans = a * b + c * d - e
        elif t == 2:
            a = rng.randint(500, 5000); b = rng.randint(100, 499)
            c = rng.randint(50, 500); d = rng.randint(50, 500)
            q = f"Compute ({a} - {b}) x ({c} + {d})."
            ans = (a - b) * (c + d)
        elif t == 3:
            a = rng.randint(15, 95); b = rng.randint(15, 95)
            c = rng.randint(12, 60); d = rng.randint(1000, 50000)
            q = f"Compute {a} x {b} x {c} - {d}."
            ans = a * b * c - d
        elif t == 4:
            a = rng.randint(100, 999); b = rng.randint(100, 999)
            c = rng.randint(100, 999); d = rng.randint(15, 80)
            e = rng.randint(1000, 99999)
            q = f"Compute ({a} + {b} + {c}) x {d} - {e}."
            ans = (a + b + c) * d - e
        else:
            a = rng.randint(30, 300); b = rng.randint(30, 300)
            c = rng.randint(30, 300); d = rng.randint(30, 300)
            e = rng.randint(500, 20000)
            q = (f"A warehouse ships {a} crates with {b} bolts each, then "
                 f"{c} crates with {d} bolts each. {e} bolts are defective. "
                 f"How many good bolts are there in total?")
            ans = a * b + c * d - e
        if ans <= 0 or (q in seen):
            continue
        seen.add(q)
        items.append((PROMPT_TMPL.format(q=q), str(ans)))
    return items


def reward_fn(text, ans):
    m = re.findall(r"\\boxed\{\s*(-?[\d,]+)\s*\}", text)
    if m:
        return 1.0 if m[-1].replace(",", "") == ans else 0.0
    m2 = re.findall(r"(-?[\d,]+)", text)
    return 1.0 if m2 and m2[-1].replace(",", "") == ans else 0.0


def seq_logps(model, prompt_ids, comp_ids, no_grad=True):
    inp = torch.cat([prompt_ids, comp_ids], dim=1)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        logits = model(inp).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    off = prompt_ids.shape[1] - 1
    idx = torch.arange(comp_ids.shape[1], device=inp.device)
    return lp[off + idx, comp_ids[0]]


def pick_model_id():
    local = os.environ.get("MODEL_DIR")
    if local and os.path.isdir(local):
        print(f"MODEL_ID_OK {local} (local dir)", flush=True)
        return local
    for mid in MODEL_CANDIDATES:
        try:
            AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            print(f"MODEL_ID_OK {mid}", flush=True)
            return mid
        except Exception as e:
            print(f"MODEL_ID_FAIL {mid}: {str(e)[:150]}", flush=True)
    raise RuntimeError("no candidate model id worked: " + str(MODEL_CANDIDATES))


def load_model():
    mid = pick_model_id()
    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        mid, quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable()
    model.train()
    try:
        from bitsandbytes.optim import AdamW8bit
        opt = AdamW8bit(model.parameters(), lr=LR)
        print("optimizer: AdamW8bit", flush=True)
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        print("optimizer: AdamW", flush=True)
    return model, tok, opt, mid


def gen_one(model, tok, prompt_ids, n_seq=1):
    """Generate in eval mode (train mode + grad checkpoint = garbage)."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model.generate(
            prompt_ids, max_new_tokens=ROLLOUT_TOKENS,
            do_sample=True, temperature=0.8, top_p=0.95,
            pad_token_id=tok.eos_token_id,
            num_return_sequences=n_seq)
    if was_training:
        model.train()
    return out


def eval_heldout(model, tok, problems):
    """Greedy held-out accuracy."""
    acc, was_training = 0, model.training
    model.eval()
    with torch.no_grad():
        for q, ans in problems:
            pids = tok(q, return_tensors="pt").input_ids.cuda()
            out = model.generate(pids, max_new_tokens=ROLLOUT_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][pids.shape[1]:], skip_special_tokens=True)
            acc += reward_fn(text, ans)
    if was_training:
        model.train()
    return acc / len(problems)


# ---------------------------------------------------------------- Training (checkpoint-aware)
def train_grpo(model, tok, opt, problems, wr, fout, start_step, curve,
               ckpt_fn):
    for step in range(start_step, STEPS + 1):
        batch = random.sample(problems, GRPO_BATCH)
        rollouts, rewards_all = [], []
        for q, ans in batch:
            pids = tok(q, return_tensors="pt").input_ids.cuda()
            out = gen_one(model, tok, pids, n_seq=G)
            for gi in range(G):
                cids = out[gi:gi + 1, pids.shape[1]:]
                text = tok.decode(cids[0], skip_special_tokens=True)
                r = reward_fn(text, ans)
                rlp = seq_logps(model, pids, cids)
                rollouts.append((pids, cids, r, rlp))
                rewards_all.append(r)
        losses = []
        for bi in range(GRPO_BATCH):
            grp = rollouts[bi * G:(bi + 1) * G]
            rs = torch.tensor([g[2] for g in grp])
            adv = (rs - rs.mean()) / (rs.std() + 1e-6)
            for (pids, cids, r, rlp), a in zip(grp, adv.tolist()):
                clp = seq_logps(model, pids, cids, no_grad=False)
                ratio = torch.exp(clp - rlp)
                at = torch.tensor(a, device=ratio.device)
                pg = torch.min(ratio * at,
                               torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * at)
                losses.append(-pg.mean())
        opt.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mr = sum(rewards_all) / len(rewards_all)
        curve.append({"step": step, "mean_reward": round(mr, 4),
                      "loss": round(loss.item(), 4),
                      "n_rollouts": len(rollouts)})
        wr.writerow([step, round(mr, 4), round(loss.item(), 4)])
        fout.flush()
        print(f"[grpo] step {step}/{STEPS} reward={mr:.3f} "
              f"loss={loss.item():.4f}", flush=True)
        if step % CKPT_EVERY == 0 or step == STEPS:
            ckpt_fn(step, curve, [])
    return curve


def train_ftw(model, tok, opt, problems, wr, fout, start_step, curve, buf,
              ckpt_fn):
    for step in range(start_step, STEPS + 1):
        batch = random.sample(problems, FTW_BATCH)
        rewards_all = []
        for q, ans in batch:
            pids = tok(q, return_tensors="pt").input_ids.cuda()
            out = gen_one(model, tok, pids, n_seq=1)
            cids = out[:, pids.shape[1]:]
            text = tok.decode(cids[0], skip_special_tokens=True)
            r = reward_fn(text, ans)
            rewards_all.append(r)
            buf.append((pids.cpu(), cids.cpu(), r))
            if len(buf) > BUF_CAP:
                buf.pop(0)
        # ORDINAL FILTER: top-k by return; CEM-style MLE toward elite winners
        k = max(8, int(len(buf) * ELITE_FRAC))
        elites = sorted(buf, key=lambda x: -x[2])[:k]
        winners = [e for e in elites if e[2] > 0]
        if winners:
            losses = []
            for pids_c, cids_c, r in winners:
                pids, cids = pids_c.cuda(), cids_c.cuda()
                clp = seq_logps(model, pids, cids, no_grad=False)
                losses.append(-clp.mean())
            opt.zero_grad()
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_v = loss.item()
        else:
            loss_v = float("nan")
        mr = sum(rewards_all) / len(rewards_all)
        curve.append({"step": step, "mean_reward": round(mr, 4),
                      "loss": round(loss_v, 4) if loss_v == loss_v else None,
                      "n_rollouts": len(batch),
                      "buf_size": len(buf), "n_winners": len(winners)})
        wr.writerow([step, round(mr, 4),
                     round(loss_v, 4) if loss_v == loss_v else "nan"])
        fout.flush()
        print(f"[ftw] step {step}/{STEPS} reward={mr:.3f} buf={len(buf)} "
              f"winners={len(winners)}/{k} loss={loss_v:.4f}", flush=True)
        if step % CKPT_EVERY == 0 or step == STEPS:
            ckpt_fn(step, curve, buf)
    return curve


# ---------------------------------------------------------------- Main
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grpo", "ftw"],
                    default=os.environ.get("FTW_MODE", "grpo"))
    args = ap.parse_args()
    run_id = os.environ.get("RUN_ID", f"ftw-{args.mode}-20261005")
    t0 = time.time()
    print(f"mode={args.mode} run_id={run_id} seed={SEED}", flush=True)

    train_p = make_problems(N_TRAIN, SEED)
    held_p = make_problems(N_HELDOUT, SEED + 1000)
    print(f"problems: {len(train_p)} train / {len(held_p)} heldout", flush=True)

    # 1. weights (aria2c if needed)
    weights_dir = ensure_weights()
    os.environ["MODEL_DIR"] = weights_dir

    # 2. resume?
    ckpt = load_resume_ckpt()
    if ckpt:
        print(f"RESUMED_FROM_STEP={ckpt['step']}", flush=True)
    else:
        print("FRESH_START", flush=True)

    # 3. model
    t_ld = time.time()
    model, tok, opt, mid = load_model()
    print(f"MODEL_LOAD_DONE dl_time_min={(time.time()-t_ld)/60:.1f}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # 4. restore state or fresh
    if ckpt:
        model.load_state_dict(ckpt["lora"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        random.setstate(ckpt["rng"]["python"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        curve = ckpt["curve"]
        h_before = ckpt["h_before"]
        peak_so_far = ckpt.get("peak_gb_so_far", 0)
        wall_so_far = ckpt.get("wall_so_far_min", 0)
        buf = [(torch.tensor(p).unsqueeze(0), torch.tensor(c).unsqueeze(0), r)
               for p, c, r in ckpt.get("buf", [])]
        start_step = ckpt["step"] + 1
        print(f"restored: curve={len(curve)} steps, buf={len(buf)}, "
              f"h_before={h_before:.3f}", flush=True)
    else:
        h_before = eval_heldout(model, tok, held_p)
        print(f"heldout BEFORE: {h_before:.3f}", flush=True)
        curve, buf, start_step = [], [], 1
        peak_so_far, wall_so_far = 0, 0

    # 5. train (or skip if already complete)
    csv_path = f"/content/ftw_{args.mode}_curve.csv"
    fout = open(csv_path, "w", newline="")
    wr = csv.writer(fout)
    wr.writerow(["step", "mean_reward", "loss"])
    for e in curve:
        wr.writerow([e["step"], e["mean_reward"],
                     e["loss"] if e["loss"] is not None else "nan"])
    fout.flush()

    def ckpt_fn(step, curve_now, buf_now):
        buf_lists = [(p[0].tolist(), c[0].tolist(), r)
                     for p, c, r in buf_now]
        save_checkpoint(step, run_id, args.mode, model, opt, curve_now,
                        buf_lists, h_before, peak_so_far, wall_so_far, t0)

    if start_step <= STEPS:
        if args.mode == "grpo":
            train_grpo(model, tok, opt, train_p, wr, fout,
                       start_step, curve, ckpt_fn)
            n_rollouts = STEPS * GRPO_BATCH * G
        else:
            train_ftw(model, tok, opt, train_p, wr, fout,
                      start_step, curve, buf, ckpt_fn)
            n_rollouts = STEPS * FTW_BATCH
    else:
        n_rollouts = STEPS * (GRPO_BATCH * G if args.mode == "grpo"
                              else FTW_BATCH)
        print("training already complete, skipping", flush=True)
    fout.close()

    h_after = eval_heldout(model, tok, held_p)
    peak_gb = max(peak_so_far, torch.cuda.max_memory_allocated() / 1e9)
    wall = wall_so_far + (time.time() - t0) / 60
    print(f"heldout AFTER: {h_after:.3f} (delta {h_after-h_before:+.3f})",
          flush=True)
    print(f"peak VRAM: {peak_gb:.2f} GB | wall: {wall/60:.1f} min",
          flush=True)

    results = {
        "paper": "arXiv:2610.03361",
        "mode": args.mode,
        "run_id": run_id,
        "model_id": mid,
        "seed": SEED,
        "steps": STEPS,
        "n_train": N_TRAIN,
        "n_heldout": N_HELDOUT,
        "n_rollouts": n_rollouts,
        "curve": curve,
        "heldout_before": round(h_before, 4),
        "heldout_after": round(h_after, 4),
        "heldout_delta": round(h_after - h_before, 4),
        "peak_vram_gb": round(peak_gb, 2),
        "wall_time_min": round(wall / 60, 1),
    }
    out = f"/content/ftw_{args.mode}_results.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"RESULTS_OK {out}", flush=True)


if __name__ == "__main__":
    main()
