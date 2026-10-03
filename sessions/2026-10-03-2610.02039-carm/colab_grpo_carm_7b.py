"""
CARM deeper test — GRPO A/B on Qwen2.5-7B-Instruct (4-bit QLoRA) with GSM8K.

Paper: arXiv:2610.02039.

Why deeper than the 0.5B runs: the 0.5B toy could not sustain the Goldilocks
drift regime (LoRA drift << band; full-FT diverged). A 7B model on real
GSM8K reasoning moves more per update, giving the masks material to bite.

  --mode standard : keep iff |mean(log r_i)| <= log(1+eps)
  --mode carm     : keep iff mean(|log r_i|) <= log(1+eps)

Metrics -> /content/carm_7b_<mode>.csv per step:
  step, mode, mean_reward, mask_rate, mean_abs_logratio_kept, max_abs_logratio, loss

Run on Colab T4 (account with capacity):
  !pip install -q transformers peft accelerate bitsandbytes datasets
  python colab_grpo_carm_7b.py --mode carm      # then --mode standard
"""
import argparse
import csv
import math
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 7
N_PROMPTS = 16
G = 2
BATCH_PROMPTS = 2
ROLLOUT_TOKENS = 384
LR = 2e-5
EPS = 0.2
CLIP = 0.2
BAND = math.log1p(EPS)
LORA_R = 16


def load_gsm8k(n, seed):
    """GSM8K test problems; reward = numeric answer match."""
    import json
    import urllib.request
    url = ("https://raw.githubusercontent.com/openai/"
           "grade-school-math/master/grade_school_math/data/test.jsonl")
    try:
        raw = urllib.request.urlopen(url, timeout=120).read().decode()
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    except Exception:
        # fallback: HF datasets lib
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        rows = [{"question": ds[i]["question"], "answer": ds[i]["answer"]}
                for i in range(len(ds))]
    rng = random.Random(seed)
    picks = rng.sample(rows, min(n, len(rows)))
    out = []
    for r in picks:
        m = re.search(r"####\s*(-?[\d,]+)", r["answer"])
        ans = m.group(1).replace(",", "")
        q = r["question"]
        out.append((f"{q}\nSolve in at most 5 short steps. Put the final numeric answer in \\boxed{{}}.", ans))
    return out


def reward_fn(text, ans):
    m = re.findall(r"\\boxed\{\s*(-?[\d,]+)\s*\}", text)
    if m:
        return 1.0 if m[-1].replace(",", "") == ans else 0.0
    m2 = re.findall(r"(-?[\d,]+)", text)
    return 1.0 if m2 and m2[-1].replace(",", "") == ans else 0.0


def std_mask(logr):
    return abs(sum(logr) / len(logr)) <= BAND


def carm_mask(logr):
    return sum(abs(x) for x in logr) / len(logr) <= BAND


def seq_logps(model, tok, prompt_ids, comp_ids, no_grad=True):
    inp = torch.cat([prompt_ids, comp_ids], dim=1)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        logits = model(inp).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    off = prompt_ids.shape[1] - 1
    idx = torch.arange(comp_ids.shape[1], device=inp.device)
    return lp[off + idx, comp_ids[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standard", "carm"], required=True)
    ap.add_argument("--prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--group", type=int, default=G)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()
    mask_fn = std_mask if args.mode == "standard" else carm_mask
    mask_name = "geometric-mean" if args.mode == "standard" else "CARM"

    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=32,
                                             lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable()
    model.train()
    try:
        from bitsandbytes.optim import AdamW8bit
        opt = AdamW8bit(model.parameters(), lr=args.lr)
        print("optimizer: AdamW8bit", flush=True)
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print("optimizer: AdamW", flush=True)
    import json as _json, os as _os
    _hp = "/content/hard_problems.json"
    if _os.path.exists(_hp):
        problems = [tuple(x) for x in _json.load(open(_hp))][:args.prompts]
        print(f"loaded {len(problems)} HARD problems from {_hp}", flush=True)
    else:
        problems = load_gsm8k(args.prompts, SEED)
    print(f"loaded {len(problems)} GSM8K problems; mask={mask_name}", flush=True)

    csv_path = f"/content/carm_7b_{args.mode}.csv"
    fout = open(csv_path, "w", newline="")
    wr = csv.writer(fout)
    wr.writerow(["step", "mode", "mean_reward", "mask_rate",
                 "mean_abs_logratio_kept", "max_abs_logratio", "loss"])

    step = 0
    nb = len(problems) // BATCH_PROMPTS
    for bi in range(nb):
        batch = problems[bi * BATCH_PROMPTS:(bi + 1) * BATCH_PROMPTS]
        rollouts, rewards_all = [], []
        for q, ans in batch:
            pids = tok(q, return_tensors="pt").input_ids.cuda()
            was_training = model.training
            model.eval()
            with torch.no_grad():
                out = model.generate(pids, max_new_tokens=ROLLOUT_TOKENS,
                                     do_sample=True, temperature=0.8,
                                     top_p=0.95, pad_token_id=tok.eos_token_id,
                                     num_return_sequences=args.group)
            if was_training:
                model.train()
            if bi == 0 and q == batch[0][0]:
                with open("/content/rollout_debug.txt", "a") as df:
                    df.write(f"=== batch {bi} ans={ans}\n")
                    df.write(tok.decode(out[0][pids.shape[1]:],
                                        skip_special_tokens=True) + "\n")
            for gi in range(args.group):
                cids = out[gi:gi + 1, pids.shape[1]:]
                text = tok.decode(cids[0], skip_special_tokens=True)
                r = reward_fn(text, ans)
                rlp = seq_logps(model, tok, pids, cids)
                rollouts.append((pids, cids, r, rlp))
                rewards_all.append(r)
        for epoch in range(args.epochs):
            losses, kept, mks, abs_lrs, max_lr = [], 0, 0, [], 0.0
            for pbi in range(len(batch)):
                grp = rollouts[pbi * args.group:(pbi + 1) * args.group]
                rs = torch.tensor([g[2] for g in grp])
                adv = (rs - rs.mean()) / (rs.std() + 1e-6)
                for (pids, cids, r, rlp), a in zip(grp, adv.tolist()):
                    clp = seq_logps(model, tok, pids, cids, no_grad=False)
                    logr = (clp - rlp).tolist()
                    if not mask_fn(logr):
                        mks += 1
                        continue
                    kept += 1
                    alr = sum(abs(x) for x in logr) / len(logr)
                    abs_lrs.append(alr)
                    max_lr = max(max_lr, max((abs(x) for x in logr), default=0.0))
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
                         round(max_lr, 4),
                         round(loss_v, 4) if loss_v == loss_v else "nan"])
            fout.flush()
            print(f"step {step} [{args.mode}] reward={mr:.3f} "
                  f"masked={mks}/{len(rollouts)} max|lr|={max_lr:.3f} "
                  f"loss={loss_v:.4f}", flush=True)
    fout.close()
    print(f"done. csv -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
