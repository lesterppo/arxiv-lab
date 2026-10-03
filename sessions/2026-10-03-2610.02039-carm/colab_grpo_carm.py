"""
CARM live test on Google Colab T4 — GRPO A/B: standard vs CARM masking.

Paper: arXiv:2610.02039.

Design: tiny synchronous GRPO on Qwen2.5-0.5B-Instruct + LoRA. Each step:
roll out G completions per prompt, then run TWO update epochs over the same
batch — the 2nd epoch trains on stale rollouts, which is exactly where the
paper's off-policy masking bites. --mode selects the sequence mask:
  standard: keep iff |mean(log r_i)| <= log(1+eps)   (geometric-mean rule)
  carm:     keep iff mean(|log r_i|) <= log(1+eps)    (cancellation-aware)

Task: generated arithmetic word problems, verifiable \\boxed{} reward.
Metrics per step -> /content/carm_grpo_<mode>.csv:
  step, mean_reward, mask_rate, mean_abs_logratio_kept, loss

Run on Colab (T4):
  !pip install -q transformers peft accelerate bitsandbytes
  !python colab_grpo_carm.py --mode carm      # then --mode standard
Compare the two CSVs: reward curve, mask behavior, stability.

Expected (paper direction): CARM masks the high-bidirectional-drift
responses the standard rule waves through, giving a stabler climb.
Small scale here (48 prompts) -> directional, not conclusive; the CSVs
are the evidence.
"""
import argparse
import csv
import math
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
N_PROMPTS = 48
G = 8
ROLLOUT_TOKENS = 96
LR = 1e-5
EPS = 0.2
CLIP = 0.2
BAND = math.log1p(EPS)
LORA_R = 16


def gen_problems(n, seed, hard=False):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if hard:
            # multi-step: the 0.5B model does NOT solve these instantly,
            # so gradients (and policy drift) stay alive
            a, b, c = rng.randint(3, 19), rng.randint(3, 19), rng.randint(2, 9)
            pat = rng.choice([0, 1, 2])
            if pat == 0:
                ans = (a + b) * c
                q = f"What is ({a} + {b}) times {c}?"
            elif pat == 1:
                ans = a * b + c
                q = f"What is {a} times {b} plus {c}?"
            else:
                a, b = rng.randint(4, 12), rng.randint(4, 12)
                c = rng.randint(2, 20)
                ans = a * b - c
                q = f"What is {a} times {b} minus {c}?"
            out.append((f"{q} Put the final answer in \\boxed{{}}.", ans))
            continue
        a, b = rng.randint(2, 49), rng.randint(2, 49)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            q = f"What is {a} plus {b}?"
        elif op == "-":
            a, b = max(a, b), min(a, b)
            ans = a - b
            q = f"What is {a} minus {b}?"
        else:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            ans = a * b
            q = f"What is {a} times {b}?"
        out.append((f"{q} Put the final answer in \\boxed{{}}.", ans))
    return out


def boxed_reward(text, ans):
    m = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", text)
    return 1.0 if m and int(m[-1]) == ans else 0.0


def std_mask(logr):
    return abs(sum(logr) / len(logr)) <= BAND


def carm_mask(logr):
    return sum(abs(x) for x in logr) / len(logr) <= BAND


def seq_logps(model, tok, prompt_ids, comp_ids, no_grad=True):
    """Log-probs of comp_ids under model given prompt. Returns (logps, mask)."""
    inp = torch.cat([prompt_ids, comp_ids], dim=1)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        logits = model(inp).logits[0]
    lp = torch.log_softmax(logits, dim=-1)
    # logp of each completion token: logits position len(prompt)-1+i -> token i
    off = prompt_ids.shape[1] - 1
    idx = torch.arange(comp_ids.shape[1], device=inp.device)
    return lp[off + idx, comp_ids[0]].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standard", "carm"], required=True)
    ap.add_argument("--prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--group", type=int, default=G)
    ap.add_argument("--epochs", type=int, default=2,
                    help="update epochs per rollout batch (higher = staler rollouts = more drift)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--hard", action="store_true",
                    help="multi-step arithmetic (harder; sustains drift)")
    ap.add_argument("--full-ft", action="store_true",
                    help="train all params instead of LoRA (bigger drift)")
    args = ap.parse_args()
    mask_fn = std_mask if args.mode == "standard" else carm_mask

    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    if not args.full_ft:
        model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=32,
                                                 lora_dropout=0.05,
                                                 target_modules=["q_proj", "v_proj"],
                                                 task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    problems = gen_problems(args.prompts, SEED, hard=args.hard)

    tag = f"{args.mode}_e{args.epochs}{"_hard" if args.hard else ""}{"_full" if args.full_ft else ""}"
    csv_path = f"/content/carm_grpo_{tag}.csv"
    fout = open(csv_path, "w", newline="")
    wr = csv.writer(fout)
    wr.writerow(["step", "mode", "mean_reward", "mask_rate",
                 "mean_abs_logratio_kept", "loss"])

    step = 0
    for pi in range(0, len(problems), 4):  # minibatch of 4 prompts
        batch = problems[pi:pi + 4]
        # ---- rollout ----
        rollouts = []  # (prompt_ids, comp_ids, reward, rollout_logps)
        rewards_all = []
        for q, ans in batch:
            pids = tok(q, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                out = model.generate(pids, max_new_tokens=ROLLOUT_TOKENS,
                                     do_sample=True, temperature=0.8,
                                     top_p=0.95, pad_token_id=tok.eos_token_id,
                                     num_return_sequences=args.group)
            for gi in range(args.group):
                cids = out[gi:gi + 1, pids.shape[1]:]
                # strip padding/eos tail
                text = tok.decode(cids[0], skip_special_tokens=True)
                r = boxed_reward(text, ans)
                rlp = seq_logps(model, tok, pids, cids)
                rollouts.append((pids, cids, r, rlp))
                rewards_all.append(r)
        # ---- N update epochs over the same (now stale) batch ----
        for epoch in range(args.epochs):
            # group advantages per prompt
            idx = 0
            losses, kept, abs_lrs = [], 0, []
            mrs, mks = [], 0
            for bi, (q, ans) in enumerate(batch):
                grp = rollouts[bi * args.group:(bi + 1) * args.group]
                rs = torch.tensor([g[2] for g in grp])
                adv = (rs - rs.mean()) / (rs.std() + 1e-6)
                for (pids, cids, r, rlp), a in zip(grp, adv.tolist()):
                    clp = seq_logps(model, tok, pids, cids, no_grad=False)
                    logr = (clp - rlp).tolist()
                    if not mask_fn(logr):
                        mks += 1
                        continue
                    kept += 1
                    abs_lrs.append(sum(abs(x) for x in logr) / len(logr))
                    ratio = torch.exp(clp - rlp)
                    a_t = torch.tensor(a, device=ratio.device)
                    pg = torch.min(ratio * a_t,
                                   torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_t)
                    losses.append(-pg.mean())
            opt.zero_grad()
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_v = loss.item()
            else:
                loss_v = float("nan")
            step += 1
            mr = sum(rewards_all) / len(rewards_all)
            wr.writerow([step, args.mode, round(mr, 4),
                         round(mks / max(len(rollouts), 1), 4),
                         round(sum(abs_lrs) / max(len(abs_lrs), 1), 4),
                         round(loss_v, 4) if loss_v == loss_v else "nan"])
            fout.flush()
            print(f"step {step} [{args.mode}] reward={mr:.3f} "
                  f"masked={mks}/{len(rollouts)} loss={loss_v:.4f}", flush=True)
    fout.close()
    model.save_pretrained(f"/content/carm_lora_{tag}")
    print(f"done. csv -> {csv_path}")


if __name__ == "__main__":
    main()
