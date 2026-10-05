"""
FTW vs GRPO A/B on Qwen3.5-9B (NF4 QLoRA) — synthetic integer arithmetic.

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

Run on Colab T4 (driver runs ON the VM; no tunnel needed):
  python3 colab.py new -s ftw-grpo --gpu T4
  python3 colab.py exec -s ftw-grpo --timeout 900 --code "
  import subprocess, sys
  # NOTE (2026-10-05): transformers<5.0 does NOT recognize Qwen3.5's
  # `qwen3_5` model type (ValueError on load). Working env verified live:
  # transformers from git main + peft 0.21.0.
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                  'git+https://github.com/huggingface/transformers.git',
                  'peft==0.21.0', 'accelerate', 'bitsandbytes'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
                 check=True)
  print('DEPS_OK')"
  # NOTE: Qwen/Qwen3.5-9B-Instruct does NOT exist on the Hub (2026-10-05);
  # Qwen/Qwen3.5-9B (base) is the working ID (driver probes both).
  python3 colab.py upload -s ftw-grpo colab_ftw_grpo_9b.py /content/ftw_driver.py
  python3 colab.py console -s ftw-grpo --cmd \
    "FTW_MODE=grpo nohup python3 /content/ftw_driver.py > /content/run_grpo.log 2>&1 & echo LAUNCHED"
  python3 colab.py logs -s ftw-grpo /content/run_grpo.log -n 30   # poll
  python3 colab.py download -s ftw-grpo /content/ftw_grpo_results.json .

Outputs: /content/ftw_<mode>_results.json, /content/ftw_<mode>_curve.csv
"""
import csv
import math
import os
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_CANDIDATES = ["Qwen/Qwen3.5-9B-Instruct", "Qwen/Qwen3.5-9B"]
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

PROMPT_TMPL = ("Solve the problem. Show brief reasoning, then put the final "
               "integer answer in \\boxed{{}}.\nProblem: {q}")


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


def train_grpo(model, tok, opt, problems, wr, fout):
    curve = []
    for s in range(STEPS):
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
        curve.append({"step": s + 1, "mean_reward": round(mr, 4),
                      "loss": round(loss.item(), 4),
                      "n_rollouts": len(rollouts)})
        wr.writerow([s + 1, round(mr, 4), round(loss.item(), 4)])
        fout.flush()
        print(f"[grpo] step {s+1}/{STEPS} reward={mr:.3f} "
              f"loss={loss.item():.4f}", flush=True)
    return curve


def train_ftw(model, tok, opt, problems, wr, fout):
    curve = []
    buf = []  # CPU replay buffer: (pids_cpu, cids_cpu, reward)
    for s in range(STEPS):
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
        curve.append({"step": s + 1, "mean_reward": round(mr, 4),
                      "loss": round(loss_v, 4) if loss_v == loss_v else None,
                      "n_rollouts": len(batch),
                      "buf_size": len(buf), "n_winners": len(winners)})
        wr.writerow([s + 1, round(mr, 4),
                     round(loss_v, 4) if loss_v == loss_v else "nan"])
        fout.flush()
        print(f"[ftw] step {s+1}/{STEPS} reward={mr:.3f} buf={len(buf)} "
              f"winners={len(winners)}/{k} loss={loss_v:.4f}", flush=True)
    return curve


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grpo", "ftw"],
                    default=os.environ.get("FTW_MODE", "grpo"))
    args = ap.parse_args()
    t0 = time.time()
    print(f"mode={args.mode} seed={SEED}", flush=True)

    train_p = make_problems(N_TRAIN, SEED)
    held_p = make_problems(N_HELDOUT, SEED + 1000)
    print(f"problems: {len(train_p)} train / {len(held_p)} heldout", flush=True)

    model, tok, opt, mid = load_model()
    torch.cuda.reset_peak_memory_stats()

    h_before = eval_heldout(model, tok, held_p)
    print(f"heldout BEFORE: {h_before:.3f}", flush=True)

    csv_path = f"/content/ftw_{args.mode}_curve.csv"
    fout = open(csv_path, "w", newline="")
    wr = csv.writer(fout)
    wr.writerow(["step", "mean_reward", "loss"])

    if args.mode == "grpo":
        curve = train_grpo(model, tok, opt, train_p, wr, fout)
        n_rollouts = STEPS * GRPO_BATCH * G
    else:
        curve = train_ftw(model, tok, opt, train_p, wr, fout)
        n_rollouts = STEPS * FTW_BATCH
    fout.close()

    h_after = eval_heldout(model, tok, held_p)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0
    print(f"heldout AFTER: {h_after:.3f} (delta {h_after-h_before:+.3f})",
          flush=True)
    print(f"peak VRAM: {peak_gb:.2f} GB | wall: {wall/60:.1f} min",
          flush=True)

    results = {
        "paper": "arXiv:2610.03361",
        "mode": args.mode,
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
    import json
    json.dump(results, open(out, "w"), indent=1)
    print(f"RESULTS_OK {out}", flush=True)


if __name__ == "__main__":
    main()
