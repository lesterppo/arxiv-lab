"""Colab driver — marginal value of verifier-based RL beyond SFT
(arXiv:2611.07639, 2026-11-07 daily training track).

Paper: "Measuring the Marginal Value of Verifier-Based RL Beyond
Supervised Fine-Tuning". Core claim: at pre-registered scale, verifier-based
RL (GRPO) adds NO detectable improvement in-distribution over SFT alone
(+0.22 pts, 95% CI [-0.06, +0.51]) and only a modest gain under
distribution shift (+0.33 pts OOD); SFT accounts for most of the gain.
The paper's own benchmark (gpu-train-debug) is SYNTHETIC and calibrated
so a mid-tier model scores 0.60-0.75.

Bounded adaptation (one T4 day, stated honestly):
  - Synthetic verifiable task in the paper's own spirit: multi-step
    arithmetic word problems (6 template families), exact-integer answers,
    programmatic verifier (last-integer extraction). In-distribution:
    3-4 ops, 2-3 digit numbers. OOD: 5-6 ops, 3-4 digit numbers, new
    surface wording (length + surface shift).
  - Arms (same base, same train pool): base (0-shot... 2-shot eval),
    SFT (400 gold CoT demos, 1 epoch, QLoRA r16), SFT+GRPO (manual
    GRPO, verifiable reward, 32 steps x 16 completions, lr 1e-5).
  - Eval: greedy, 2-shot, 200 in-dist + 200 OOD held-out, exact match.
  - Model: Qwen/Qwen3.5-9B NF4-QLoRA tried first per standing picks
    (env MODEL_ID overrides, e.g. Qwen/Qwen3-8B for the fallback run).
  - Deviations from the paper, recorded: (a) GRPO implemented manually
    (plain torch, no TRL) to stay in the proven stack; single update per
    batch, PPO-style clipped ratio, group-normalized advantage;
    (b) beta=0 (no reference model: saves ~5GB VRAM on T4);
    (c) 32 GRPO steps (paper: 60) and lr 1e-5 (paper: 1e-6 full-model;
    LoRA adapters need a higher LR); (d) arithmetic task instead of
    GPU-failure diagnosis (same mechanism: SFT vs SFT+GRPO, in-dist vs
    OOD, programmatic verifier).
  - Incremental writes: every phase appends to /content/rlvr_metrics.csv
    and per-item rows to /content/rlvr_details.csv; adapter zips saved
    after each arm (reclamation lesson).
  - Prints RLGRPO_DONE as its very last line on completion.

Run via: python3 colab.py train --driver <this file> --rotate
  --session rlvr-1107 --monitor-timeout 14400
  --results /content/rlvr_metrics.csv /content/rlvr_details.csv
  /content/adapters/sft.zip /content/adapters/grpo.zip
  --out-dir <session dir> --done-marker RLGRPO_DONE
  --registry-file <isolated registry>
"""

import csv
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys

METRICS = "/content/rlvr_metrics.csv"
DETAILS = "/content/rlvr_details.csv"
ADAPTER_DIR = "/content/adapters"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
N_SFT = 400
N_GRPO_PROMPTS = 128
GRPO_STEPS = 32
N_EVAL = 200
MAX_NEW_EVAL = 192
MAX_NEW_ROLL = 256
SEQ_CAP = 512
SEED = 1107


def _pip():
    pkgs = ["transformers==5.20.2", "peft", "bitsandbytes", "accelerate",
            "safetensors"]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                       check=True, timeout=1200)
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


# ---------------------------------------------------------------- task
# Six template families. Each problem: question text, gold CoT, answer int.
# gen_problem(rng, ood) -> (question, cot, answer). Pure Python; mirrored
# in the local CPU test (test_rlvr_task_cpu.py).
def _fmt(n):
    return "%d" % n


def gen_problem(rng, ood=False):
    """One multi-step arithmetic word problem.

    The narrative is built FROM the sampled op chain, so the question text
    always matches the gold CoT. Families differ only in surface form
    (nouns/verbs); family 5 adds an irrelevant distractor number.
    In-dist: 3-4 ops, 2-3 digit numbers. OOD: 5-6 ops, 3-4 digit numbers.
    """
    fam = rng.choice([1, 2, 3, 4, 5, 6])
    if ood:
        n_ops = rng.choice([5, 6])
        lo, hi = 100, 9999
    else:
        n_ops = rng.choice([3, 4])
        lo, hi = 11, 999
    start = rng.randint(lo, hi)
    ops = []  # (symbol, value); keep the running value positive
    val = start
    tries = 0
    while len(ops) < n_ops and tries < 300:
        tries += 1
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            v = rng.randint(lo, hi)
            ops.append(("+", v))
            val += v
        elif op == "-":
            if val <= lo + 1:
                continue
            v = rng.randint(lo, val - 1)
            ops.append(("-", v))
            val -= v
        else:
            v = rng.randint(2, 9)
            ops.append(("*", v))
            val *= v
    if len(ops) < n_ops:
        return gen_problem(rng, ood)
    steps = []
    cur = start
    for sym, v in ops:
        cur = cur + v if sym == "+" else cur - v if sym == "-" else cur * v
        steps.append((sym, v, cur))
    names = ["Maya", "Liam", "Noor", "Chen", "Aisha", "Diego"]
    name = rng.choice(names)
    items = rng.choice(["notebooks", "boxes", "marbles", "tickets",
                        "bottles", "stickers"])
    if fam == 1:
        intro = "A warehouse had %d %s." % (start, items)

        def sent(sym, v):
            return {"+": "A delivery added %d %s." % (v, items),
                    "-": "%d %s were shipped out." % (v, items),
                    "*": "The stock grew by a factor of %d." % v}[sym]
        ask = "How many %s are in the warehouse now?" % items
    elif fam == 2:
        intro = "%s had $%d." % (name, start)

        def sent(sym, v):
            return {"+": "%s earned $%d." % (name, v),
                    "-": "%s spent $%d." % (name, v),
                    "*": "The savings grew by a factor of %d." % v}[sym]
        ask = "How much money does %s have now?" % name
    elif fam == 3:
        intro = "A factory makes %d parts per day." % start

        def sent(sym, v):
            return {"+": "Output rises by %d parts." % v,
                    "-": "Output falls by %d parts." % v,
                    "*": "Output is multiplied by %d." % v}[sym]
        ask = "How many parts per day does the factory make now?"
    elif fam == 4:
        intro = "There are %d chairs in a hall." % start

        def sent(sym, v):
            return {"+": "%d chairs are added." % v,
                    "-": "%d chairs are removed." % v,
                    "*": "The number of chairs grows by a factor of %d." % v}[sym]
        ask = "How many chairs are in the hall now?"
    elif fam == 5:
        distractor = rng.randint(lo, hi)
        intro = ("A shop had %d %s. (The shop also stores %d pallets, "
                 "which are not counted here.)" % (start, items, distractor))

        def sent(sym, v):
            return {"+": "It bought %d more %s." % (v, items),
                    "-": "It sold %d %s." % (v, items),
                    "*": "Its stock grew by a factor of %d." % v}[sym]
        ask = "How many %s does the shop have now?" % items
    else:
        intro = "A baker made %d loaves." % start

        def sent(sym, v):
            return {"+": "The baker made %d more loaves." % v,
                    "-": "The baker sold %d loaves." % v,
                    "*": "The batch grew by a factor of %d." % v}[sym]
        ask = "How many loaves does the baker have now?"
    sentences = [intro] + [sent(sym, v) for sym, v, _ in steps]
    q = " ".join(sentences) + " " + ask
    cot_lines = []
    cur = start
    for i, (sym, v, c) in enumerate(steps):
        if sym == "+":
            cot_lines.append("Step %d: %d + %d = %d." % (i + 1, cur, v, c))
        elif sym == "-":
            cot_lines.append("Step %d: %d - %d = %d." % (i + 1, cur, v, c))
        else:
            cot_lines.append("Step %d: %d x %d = %d." % (i + 1, cur, v, c))
        cur = c
    cot = "\n".join(cot_lines) + "\nFinal: %d" % val
    return q, cot, val


def extract_answer(text):
    """Last integer after 'Final:' if present, else last integer anywhere."""
    m = re.findall(r"Final:\s*(-?\d+)", text)
    if m:
        return int(m[-1])
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def make_exemplars(rng):
    ex = []
    for _ in range(2):
        q, cot, ans = gen_problem(rng, ood=False)
        ex.append((q, cot))
    return ex


def eval_prompt(question, exemplars):
    s = ""
    for q, cot in exemplars:
        s += "Question: %s\n%s\n\n" % (q, cot)
    s += ("Question: %s\nThink step by step, then write the final answer "
          "as: Final: <number>" % question)
    return s


def train_prompt(question, exemplars):
    return eval_prompt(question, exemplars)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    _pip()
    import torch
    import transformers
    print("transformers=%s" % transformers.__version__, flush=True)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, PeftModel
    assert torch.cuda.is_available(), "no GPU: refusing to fake a GPU run"
    print("GPU: %s total=%.1f GB torch=%s"
          % (torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9,
             torch.__version__), flush=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    rng = random.Random(SEED)

    # ---- data (seeded, disjoint pools) ----
    exemplars = make_exemplars(random.Random(SEED + 1))
    train_pool = [gen_problem(rng, ood=False) for _ in range(600)]
    eval_in = [gen_problem(random.Random(SEED + 2 + i), ood=False)
               for i in range(N_EVAL)]
    eval_ood = [gen_problem(random.Random(SEED + 5000 + i), ood=True)
                for i in range(N_EVAL)]
    sft_data = train_pool[:N_SFT]
    grpo_prompts = [q for q, _, _ in train_pool[N_SFT:N_SFT + N_GRPO_PROMPTS]]
    grpo_answers = [a for _, _, a in train_pool[N_SFT:N_SFT + N_GRPO_PROMPTS]]
    print("data: sft=%d grpo_prompts=%d eval_in=%d eval_ood=%d"
          % (len(sft_data), len(grpo_prompts), len(eval_in), len(eval_ood)),
          flush=True)

    # ---- model ----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto")
    base.config.use_cache = False
    print("LOADED %s peak=%.2f GB"
          % (MODEL_ID, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    _append_metrics({"phase": "probe", "model": MODEL_ID,
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                     "acc": "", "n": ""})

    @torch.no_grad()
    def evaluate(model, split, n=None, arm="base"):
        model.eval()
        items = eval_in if split == "in" else eval_ood
        if n:
            items = items[:n]
        correct, fails, lens = 0, 0, []
        for i in range(0, len(items), 16):
            batch = items[i:i + 16]
            prompts = [eval_prompt(q, exemplars) for q, _, _ in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=SEQ_CAP).to(model.device)
            out = model.generate(**enc, max_new_tokens=MAX_NEW_EVAL,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for j, (q, _, ans) in enumerate(batch):
                gen = tok.decode(out[j][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                pred = extract_answer(gen)
                lens.append(len(gen))
                if pred is None:
                    fails += 1
                elif pred == ans:
                    correct += 1
                _append_detail({"phase": "eval", "split": split,
                                "arm": arm, "q": q[:200],
                                "gold": ans, "pred": pred if pred is not None else "",
                                "gen": gen[:400].replace("\n", " ")})
        acc = correct / len(items)
        print("EVAL split=%s n=%d acc=%.4f parse_fail=%d mean_chars=%.0f"
              % (split, len(items), acc, fails, sum(lens) / max(1, len(lens))),
              flush=True)
        return acc

    # ---- phase 1: base eval (calibration check, 100 each) ----
    acc_in = evaluate(base, "in", n=100, arm="base")
    acc_ood = evaluate(base, "ood", n=100, arm="base")
    _append_metrics({"phase": "eval_base", "model": MODEL_ID,
                     "acc_in": round(acc_in, 4), "acc_ood": round(acc_ood, 4),
                     "n": 100, "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})

    # ---- phase 2: SFT (completion-only CE) ----
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj",
                                      "o_proj", "gate_proj", "up_proj",
                                      "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    model.config.use_cache = False

    def sft_batch(examples):
        prompts = [train_prompt(q, exemplars) for q, _, _ in examples]
        comps = ["\n" + cot + tok.eos_token for _, cot, _ in examples]
        pids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        cids = [tok(c, add_special_tokens=False)["input_ids"] for c in comps]
        maxlen = max(len(a) + len(b) for a, b in zip(pids, cids))
        maxlen = min(maxlen, SEQ_CAP)
        input_ids, labels, attn = [], [], []
        for a, b in zip(pids, cids):
            ids = (a + b)[:maxlen]
            lab = ([-100] * len(a) + b)[:maxlen]
            pad = maxlen - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return (torch.tensor(input_ids, device=model.device),
                torch.tensor(labels, device=model.device),
                torch.tensor(attn, device=model.device))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    model.train()
    sft_losses = []
    for step in range(0, len(sft_data), 4):
        chunk = sft_data[step:step + 4]
        ids, labs, att = sft_batch(chunk)
        out = model(input_ids=ids, labels=labs, attention_mask=att)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sft_losses.append(float(out.loss))
        if (step // 4 + 1) % 25 == 0:
            print("SFT step %d loss=%.4f peak=%.2f GB"
                  % (step // 4 + 1, sft_losses[-1],
                     torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print("SFT done mean_loss=%.4f" % (sum(sft_losses) / len(sft_losses)),
          flush=True)
    model.save_pretrained("/content/adapters/sft")
    shutil.make_archive("/content/adapters/sft", "zip",
                        "/content/adapters", "sft")
    _append_metrics({"phase": "sft_done", "model": MODEL_ID,
                     "mean_loss": round(sum(sft_losses) / len(sft_losses), 4),
                     "n": N_SFT,
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    acc_in = evaluate(model, "in", arm="sft")
    acc_ood = evaluate(model, "ood", arm="sft")
    _append_metrics({"phase": "eval_sft", "model": MODEL_ID,
                     "acc_in": round(acc_in, 4), "acc_ood": round(acc_ood, 4),
                     "n": N_EVAL,
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})

    # ---- phase 3: manual GRPO from the SFT adapter ----
    model.train()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    G = 4  # generations per prompt
    B_PROMPTS = 4  # prompts per step -> 16 sequences
    grng = random.Random(SEED + 99)
    prompt_order = list(range(len(grpo_prompts)))
    grng.shuffle(prompt_order)
    step_rewards = []
    for step in range(GRPO_STEPS):
        idxs = [prompt_order[(step * B_PROMPTS + k) % len(prompt_order)]
                for k in range(B_PROMPTS)]
        qs = [grpo_prompts[i] for i in idxs]
        golds = [grpo_answers[i] for i in idxs]
        prompts = [eval_prompt(q, exemplars) for q in qs]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=SEQ_CAP).to(model.device)
        # rollouts: G samples per prompt (temperature sampling)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=MAX_NEW_ROLL, do_sample=True,
                temperature=0.9, top_p=0.95, num_return_sequences=G,
                pad_token_id=tok.pad_token_id)
        # NOTE: left padding: prompt tokens for sequence j start at
        # offset P - plen_j in the padded sequence (P = padded prompt len).
        P = enc["input_ids"].shape[1]
        seqs = gen  # [B*G, L]
        prompt_lens = []
        for j in range(len(qs)):
            plen = int(enc["attention_mask"][j].sum())
            prompt_lens += [plen] * G
        comp_texts, comp_ids, rewards = [], [], []
        for j, seq in enumerate(seqs):
            plen = prompt_lens[j]
            off = P - plen  # left-pad offset of the prompt in seq
            ids = seq[off + plen:].tolist()
            # trim at eos
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id)]
            comp_ids.append(ids)
            text = tok.decode(ids, skip_special_tokens=True)
            comp_texts.append(text)
            gold = golds[j // G]
            pred = extract_answer(text)
            rewards.append(1.0 if pred == gold else 0.0)
        # group-normalized advantage (per prompt group of G)
        adv = []
        for g0 in range(0, len(rewards), G):
            grp = rewards[g0:g0 + G]
            mu = sum(grp) / len(grp)
            var = sum((r - mu) ** 2 for r in grp) / len(grp)
            sd = math.sqrt(var) + 1e-8
            adv += [(r - mu) / sd for r in grp]
        mean_r = sum(rewards) / len(rewards)
        step_rewards.append(mean_r)
        # old logprobs (no grad) then one clipped-ratio update
        full_ids, full_attn = [], []
        maxlen = 0
        for j, seq in enumerate(seqs):
            plen = prompt_lens[j]
            off = P - plen
            ids = seq[off:off + plen + len(comp_ids[j])].tolist()
            maxlen = max(maxlen, len(ids))
        maxlen = min(maxlen, SEQ_CAP)
        for j, seq in enumerate(seqs):
            plen = prompt_lens[j]
            off = P - plen
            ids = seq[off:off + plen + len(comp_ids[j])].tolist()[:maxlen]
            pad = maxlen - len(ids)
            full_ids.append(ids + [tok.pad_token_id] * pad)
            full_attn.append([1] * len(ids) + [0] * pad)
        full_ids = torch.tensor(full_ids, device=model.device)
        full_attn = torch.tensor(full_attn, device=model.device)
        with torch.no_grad():
            logits_old = model(input_ids=full_ids,
                               attention_mask=full_attn).logits
        # per-token logprob of completion tokens under old policy
        import torch.nn.functional as F
        logp_old_all = F.log_softmax(logits_old.float(), dim=-1)
        # completion mask: positions predicting completion tokens
        comp_lens = [len(c) for c in comp_ids]
        old_lp = []
        for j in range(len(seqs)):
            plen = prompt_lens[j]
            clen = comp_lens[j]
            # logits at positions plen-1 .. plen+clen-2 predict tokens
            # plen .. plen+clen-1
            lp = []
            for t in range(clen):
                pos = plen - 1 + t
                if pos < 0 or pos >= logp_old_all.shape[1]:
                    continue
                lp.append(logp_old_all[j, pos, full_ids[j, plen + t]].item())
            old_lp.append(sum(lp) / max(1, len(lp)))
        old_lp = torch.tensor(old_lp, device=model.device)
        adv_t = torch.tensor(adv, device=model.device, dtype=torch.float32)
        logits = model(input_ids=full_ids,
                       attention_mask=full_attn).logits
        logp_all = F.log_softmax(logits.float(), dim=-1)
        new_lp = []
        for j in range(len(seqs)):
            plen = prompt_lens[j]
            clen = comp_lens[j]
            lp = []
            for t in range(clen):
                pos = plen - 1 + t
                if pos < 0 or pos >= logp_all.shape[1]:
                    continue
                lp.append(logp_all[j, pos, full_ids[j, plen + t]])
            new_lp.append(torch.stack(lp).mean() if lp else
                          torch.tensor(0.0, device=model.device))
        new_lp = torch.stack(new_lp)
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 0.8, 1.2)
        # Standard PPO surrogate on sequence-mean logprobs:
        # -mean(min(ratio*A, clip(ratio)*A)).
        loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        print("GRPO step %d/%d mean_reward=%.3f loss=%.4f peak=%.2f GB"
              % (step + 1, GRPO_STEPS, mean_r, float(loss),
                 torch.cuda.max_memory_allocated() / 1e9), flush=True)
        del gen, seqs, logits_old, logits
        gc.collect()
        torch.cuda.empty_cache()
    print("GRPO done mean_reward_last8=%.3f"
          % (sum(step_rewards[-8:]) / len(step_rewards[-8:])), flush=True)
    model.save_pretrained("/content/adapters/grpo")
    shutil.make_archive("/content/adapters/grpo", "zip",
                        "/content/adapters", "grpo")
    _append_metrics({"phase": "grpo_done", "model": MODEL_ID,
                     "mean_reward_last8": round(sum(step_rewards[-8:]) / 8, 4),
                     "steps": GRPO_STEPS,
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    acc_in = evaluate(model, "in", arm="grpo")
    acc_ood = evaluate(model, "ood", arm="grpo")
    _append_metrics({"phase": "eval_grpo", "model": MODEL_ID,
                     "acc_in": round(acc_in, 4), "acc_ood": round(acc_ood, 4),
                     "n": N_EVAL,
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    print("RLGRPO_DONE", flush=True)


if __name__ == "__main__":
    main()
