"""SCARL real-RLVR Colab driver — arXiv 2610.40407 (Ke et al.).

Follow-up to the 2026-11-01 CPU session (estimator properties only):
a small bounded BASE vs SCARL advantage-shaping comparison on a real
model, on GSM8K-style reasoning.

Design (shrink the EXPERIMENT, never the model):
- Model: Qwen3-8B (current-gen 8B class, proven T4 NF4-QLoRA default),
  NF4, bf16, LoRA r=16/alpha=32 on all 7 linears.
- Two arms, identical seeds / prompt order / LoRA init: BASE uses
  group-relative advantages (reward minus group mean); SCARL reshapes
  them with lam=1.0 per the session's CPU operationalization:
  gap = |confidence - outcome-certified confidence| (1 for the
  positive-advantage class, 0 for the negative class, 0 for
  zero-advantage responses); raw |A|*(1+lam*gap), renormalized within
  each class so the class's total |advantage| budget is preserved.
  Confidence = mean token probability of the generated completion.
- 3 steps x 12 prompts x group-of-4 rollouts per arm; eval on held-out
  GSM8K problems: greedy accuracy (32), sampled Pass@4 (16x4), and
  mean next-token entropy on eval prefixes (16) as the collapse probe.
- Metrics CSV is written incrementally (flushed per row) to
  /content/scarl_rlvr_metrics.csv; a summary JSON goes to
  /content/scarl_rlvr_summary.json. TRAIN_DONE is printed as the VERY
  LAST line, only on success.

Pitfalls honoured: torchao uninstalled right after pip installs;
bf16 (never fp16); eval batches small and logits sliced before
float() (the 2026-10-28 FRA run OOM'd materializing full-vocab
logits at batch 32); rollouts in model.eval(); no secrets in this
file (public ungated model, no HF token needed).
"""

import csv
import json
import math
import os
import random
import re
import sys

# ---------------------------------------------------------------- pure
# Stdlib-only helpers (importable and testable without torch).


def extract_number(text):
    """Last number in `text` (handles commas, decimals, negatives)."""
    if not text:
        return None
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def extract_answer(text):
    """Answer after 'The answer is' if present, else the last number."""
    if not text:
        return None
    m = re.findall(r"[Tt]he answer is\s*:?\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except ValueError:
            pass
    return extract_number(text)


def gold_answer(solution):
    """GSM8K gold: number after '####'."""
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", solution or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def clean_solution(solution):
    """Strip calculator annotations; '#### N' -> 'The answer is N'."""
    s = re.sub(r"<<[^>]*>>", "", solution or "")
    s = re.sub(r"####\s*(-?\d[\d,]*\.?\d*)", r"The answer is \1", s)
    return " ".join(s.split())


def group_advantages(rewards):
    """Group-relative advantages: reward minus group mean."""
    n = len(rewards)
    mean = sum(rewards) / n if n else 0.0
    return [r - mean for r in rewards]


def scarl_advantages(rewards, confidences, lam=1.0):
    """SCARL reshaping, same operationalization as the CPU session.

    Never re-signs; preserves each class's total |advantage| budget
    exactly via within-class renormalization; reduces to base when a
    class's confidences agree.
    """
    A = group_advantages(rewards)
    out = [0.0] * len(A)
    for sign in (1, -1):
        idx = [i for i, a in enumerate(A) if (a > 1e-12 and sign == 1)
               or (a < -1e-12 and sign == -1)]
        if not idx:
            continue
        target = 1.0 if sign == 1 else 0.0
        gaps = [abs(confidences[i] - target) for i in idx]
        raw = [abs(A[i]) * (1.0 + lam * g) for i, g in zip(idx, gaps)]
        budget = sum(abs(A[i]) for i in idx)
        total = sum(raw)
        if total <= 0:
            for i in idx:
                out[i] = A[i]
            continue
        for i, r in zip(idx, raw):
            out[i] = math.copysign(budget * (r / total), A[i])
    return out


# Embedded fallback problems (original word problems, not GSM8K copies),
# used only if the GSM8K download fails.
FALLBACK_PROBLEMS = [
    ("A baker makes 5 trays with 12 muffins on each tray. He sells 23 "
     "muffins in the morning. How many muffins are left?", 37),
    ("Sara reads 18 pages every day. How many pages does she read in "
     "2 weeks?", 252),
    ("A classroom has 6 rows with 8 desks in each row. 9 desks are "
     "broken. How many desks are usable?", 39),
    ("Tom buys 4 packs of pens with 7 pens in each pack and gives 5 "
     "pens to his friend. How many pens does he keep?", 23),
    ("A farmer collects 96 eggs and packs them into boxes of 8. How "
     "many boxes does he fill?", 12),
    ("Lily had 45 stickers. She buys 3 sheets with 12 stickers each. "
     "How many stickers does she have now?", 81),
    ("A train travels 65 miles per hour for 3 hours, then 40 miles "
     "per hour for 2 hours. How many miles does it travel in total?",
     275),
    ("A shop sells notebooks for $4 each. Mia buys 9 notebooks and "
     "pays with $50. How much change does she get?", 14),
    ("There are 7 bags with 15 marbles in each bag. 28 marbles are "
     "red and the rest are blue. How many blue marbles are there?",
     77),
    ("A gardener plants 8 rows of 14 tulips and 6 rows of 9 roses. "
     "How many flowers does she plant in total?", 166),
    ("Jack saves $12 a week for 9 weeks, then spends $38 on a game. "
     "How much money does he have left?", 70),
    ("A rectangle is 15 cm long and 8 cm wide. What is its area in "
     "square cm?", 120),
    ("A school orders 240 juice boxes for 8 classes with the same "
     "number of students. How many juice boxes does each class get?",
     30),
    ("Emma runs 3 km on Monday, 5 km on Wednesday, and 4 km on "
     "Friday. How many km does she run in 4 such weeks?", 48),
    ("A movie ticket costs $11 for adults and $7 for children. A "
     "family of 2 adults and 3 children goes to the movies. How much "
     "do they pay in total?", 43),
    ("A warehouse has 17 shelves with 24 cans on each shelf. Workers "
     "remove 156 cans. How many cans remain?", 252),
]


def build_prompt(question, exemplars):
    """2-shot completion-style prompt for a base model."""
    parts = []
    for q, sol in exemplars:
        parts.append("Question: %s\nAnswer: %s" % (q, sol))
    parts.append("Question: %s\nAnswer:" % question)
    return "\n\n".join(parts)


# ---------------------------------------------------------------- main
def main():
    import subprocess

    # 1. Installs (Colab ships torchao 0.10, peft needs >0.16 -> remove).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.46,<5.0", "bitsandbytes", "accelerate",
                    "peft", "datasets"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torchao"], check=False)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    print("cuda:", torch.cuda.get_device_name(0),
          "total_gib: %.2f" % (torch.cuda.get_device_properties(0).total_memory
                                / 2**30), flush=True)

    # 2. Data: GSM8K test, first 48 = train pool, next 32 = held-out
    #    eval, first 16 of held-out = sampled Pass@4 pool.
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        train_ds = load_dataset("openai/gsm8k", "main", split="train")
        problems = []
        for ex in ds.select(range(80)):
            g = gold_answer(ex["answer"])
            if g is not None:
                problems.append((ex["question"], g))
        exemplars = []
        for ex in train_ds:
            sol = clean_solution(ex["answer"])
            if 100 < len(sol) < 600 and "####" not in sol:
                exemplars.append((ex["question"], sol))
            if len(exemplars) == 2:
                break
        print("gsm8k loaded: %d problems, %d exemplars"
              % (len(problems), len(exemplars)), flush=True)
        if len(problems) < 80 or len(exemplars) < 2:
            raise RuntimeError("gsm8k short: problems=%d exemplars=%d"
                               % (len(problems), len(exemplars)))
    except Exception as e:  # noqa: BLE001 - fallback must be robust
        print("gsm8k load failed (%r); using embedded fallback" % e,
              flush=True)
        problems = list(FALLBACK_PROBLEMS) * 5  # 80 items
        exemplars = [
            ("A store has 9 shelves with 6 toys on each shelf. 14 toys "
             "are sold. How many toys remain?",
             "The store starts with 9 * 6 = 54 toys. After selling 14, "
             "54 - 14 = 40 remain. The answer is 40"),
            ("Ben walks 2 miles each morning for 5 mornings. How many "
             "miles does he walk in total?",
             "Ben walks 2 * 5 = 10 miles. The answer is 10"),
        ]
    train_pool = problems[:48]
    eval_pool = problems[48:80]
    pass_pool = eval_pool[:16]

    # 3. Metrics CSV (incremental, flushed per row).
    csv_path = "/content/scarl_rlvr_metrics.csv"
    fieldnames = ["arm", "phase", "step", "n_prompts", "n_groups",
                  "mean_reward", "pass_at_4", "accuracy", "entropy",
                  "mean_len", "peak_vram_gib"]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    def row(**kw):
        r = {f: "" for f in fieldnames}
        r.update(kw)
        writer.writerow(r)
        csv_file.flush()
        print("ROW", r, flush=True)

    def peak_vram():
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # 4. Model: Qwen3-8B base, NF4, bf16.
    MODEL = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    print("model loaded, peak_vram_gib:", peak_vram(), flush=True)

    def make_arm(seed):
        """Fresh LoRA adapter with identical init across arms."""
        torch.manual_seed(seed)
        random.seed(seed)
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj",
                                         "o_proj", "gate_proj", "up_proj",
                                         "down_proj"],
                         task_type="CAUSAL_LM")
        m = get_peft_model(base, cfg)
        m.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
        m.train()
        return m

    def batch_generate(model, prompts, do_sample, temp=1.0,
                       max_new=160, batch=4):
        """Greedy or sampled generation, small batches, eval mode."""
        model.eval()
        outs = []
        old_side = tok.padding_side
        tok.padding_side = "left"
        try:
            for i in range(0, len(prompts), batch):
                chunk = prompts[i:i + batch]
                enc = tok(chunk, return_tensors="pt", padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=do_sample,
                        temperature=temp if do_sample else None,
                        top_p=0.95 if do_sample else None,
                        pad_token_id=tok.pad_token_id)
                for j, g in enumerate(gen):
                    new = g[enc["input_ids"].shape[1]:]
                    outs.append(tok.decode(new, skip_special_tokens=True))
        finally:
            tok.padding_side = old_side
            model.train()
        return outs

    def mean_logprobs(model, prompts, completions):
        """Mean logprob per completion token (slice logits before
        float(): never materialize full-vocab fp32 logits)."""
        model.train()
        tok.padding_side = "right"
        out = []
        for prompt, comp in zip(prompts, completions):
            p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
            c_ids = tok(comp, add_special_tokens=False)["input_ids"]
            if not c_ids:
                out.append(-20.0)
                continue
            ids = torch.tensor([p_ids + c_ids], device=device)
            with torch.enable_grad():
                logits = model(input_ids=ids).logits
            # Positions predicting completion tokens only.
            start = len(p_ids) - 1
            sl = logits[0, start:start + len(c_ids), :]  # [L, V] bf16
            logp = torch.log_softmax(sl.float(), dim=-1)
            tgt = torch.tensor(c_ids, device=device)
            lp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            out.append(lp)
        return out

    def greedy_eval(model, pool, batch=8):
        prompts = [build_prompt(q, exemplars) for q, _ in pool]
        outs = batch_generate(model, prompts, do_sample=False, batch=batch)
        correct = 0
        for (_, gold), text in zip(pool, outs):
            pred = extract_answer(text)
            if pred is not None and abs(pred - gold) < 1e-6:
                correct += 1
        return correct / len(pool) if pool else 0.0

    def sampled_pass_at_4(model, pool):
        groups = []
        for q, gold in pool:
            pr = build_prompt(q, exemplars)
            cs = batch_generate(model, [pr] * 4, do_sample=True,
                                temp=1.0, batch=4)
            ok = any((lambda p: p is not None and abs(p - gold) < 1e-6)
                     (extract_answer(c)) for c in cs)
            groups.append(1.0 if ok else 0.0)
        return sum(groups) / len(groups) if groups else 0.0

    def prefix_entropy(model, pool):
        """Mean next-token entropy on eval prefixes (last position
        only, sliced before float())."""
        model.eval()
        tok.padding_side = "right"
        ents = []
        try:
            for i in range(0, len(pool), 8):
                chunk = [build_prompt(q, exemplars)
                         for q, _ in pool[i:i + 8]]
                enc = tok(chunk, return_tensors="pt", padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = model(**enc).logits[:, -1, :]  # [B, V]
                probs = torch.softmax(logits.float(), dim=-1)
                ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                ents.extend(ent.tolist())
        finally:
            model.train()
        return sum(ents) / len(ents) if ents else 0.0

    summary = {"model": MODEL, "lam": 1.0, "steps": 3,
               "prompts_per_step": 12, "group_size": 4, "arms": {}}

    # 5. Baseline eval (identical init for both arms, measured once).
    probe = make_arm(1234)
    base_acc = greedy_eval(probe, eval_pool)
    base_pass = sampled_pass_at_4(probe, pass_pool)
    base_ent = prefix_entropy(probe, eval_pool[:16])
    row(arm="BASELINE", phase="eval", step=-1, n_prompts=len(eval_pool),
        accuracy=round(base_acc, 4), pass_at_4=round(base_pass, 4),
        entropy=round(base_ent, 4), peak_vram_gib=peak_vram())
    summary["baseline"] = {"greedy_accuracy": base_acc,
                           "pass_at_4": base_pass, "entropy": base_ent}
    # Free the probe adapter's optimizer state (adapter itself is small);
    # the next arm rebuilds a fresh adapter on the same base.
    del probe

    # 6. Arms.
    for arm in ("BASE", "SCARL"):
        model = make_arm(1234)
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=1e-5)
        arm_rows = []
        for step in range(3):
            step_prompts = train_pool[step * 12:(step + 1) * 12]
            full_prompts = [build_prompt(q, exemplars)
                            for q, _ in step_prompts]
            # Rollouts: one group of 4 per prompt (eval mode inside).
            torch.manual_seed(1000 + step)
            random.seed(1000 + step)
            all_groups = []
            for pr in full_prompts:
                cs = batch_generate(model, [pr] * 4, do_sample=True,
                                    temp=1.0, batch=4)
                all_groups.append(cs)
            # Train: per group, logprobs + advantages + loss.
            opt.zero_grad()
            step_rewards, step_pass, step_lens = [], [], []
            for (q, gold), pr, cs in zip(step_prompts, full_prompts,
                                         all_groups):
                rewards = []
                for c in cs:
                    pred = extract_answer(c)
                    rewards.append(1.0 if (pred is not None
                                           and abs(pred - gold) < 1e-6)
                                   else 0.0)
                step_rewards.extend(rewards)
                step_pass.append(1.0 if max(rewards) > 0 else 0.0)
                step_lens.extend(len(tok(c, add_special_tokens=False)
                                     ["input_ids"]) for c in cs)
                lps = mean_logprobs(model, [pr] * 4, cs)
                confs = [float(torch.exp(lp.mean()).item()) for lp in lps]
                if arm == "SCARL":
                    advs = scarl_advantages(rewards, confs, lam=1.0)
                else:
                    advs = group_advantages(rewards)
                # Loss: -mean_i A_i * mean_t logp (per-group, /ngroups).
                loss = 0.0
                for adv, lp in zip(advs, lps):
                    loss = loss - adv * lp.mean()
                (loss / len(all_groups)).backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()
            mr = sum(step_rewards) / len(step_rewards)
            pa = sum(step_pass) / len(step_pass)
            ml = sum(step_lens) / len(step_lens)
            row(arm=arm, phase="train", step=step,
                n_prompts=len(step_prompts), n_groups=len(all_groups),
                mean_reward=round(mr, 4), pass_at_4=round(pa, 4),
                mean_len=round(ml, 1), peak_vram_gib=peak_vram())
            arm_rows.append({"step": step, "mean_reward": mr,
                             "pass_at_4": pa})
        acc = greedy_eval(model, eval_pool)
        pas = sampled_pass_at_4(model, pass_pool)
        ent = prefix_entropy(model, eval_pool[:16])
        row(arm=arm, phase="eval", step=3, n_prompts=len(eval_pool),
            accuracy=round(acc, 4), pass_at_4=round(pas, 4),
            entropy=round(ent, 4), peak_vram_gib=peak_vram())
        summary["arms"][arm] = {"train_steps": arm_rows,
                                "eval": {"greedy_accuracy": acc,
                                         "pass_at_4": pas,
                                         "entropy": ent},
                                "peak_vram_gib": peak_vram()}
        # Drop the adapter before the next arm (base weights stay).
        model = model.unload() if hasattr(model, "unload") else model
        del model
        torch.cuda.empty_cache()
        print("arm done:", arm, flush=True)

    summary["peak_vram_gib"] = peak_vram()
    with open("/content/scarl_rlvr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    csv_file.close()
    print("summary:", json.dumps(summary), flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
