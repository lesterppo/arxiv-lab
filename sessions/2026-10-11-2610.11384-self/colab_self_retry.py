"""Colab T4 script: SELF vs hindsight-SFT on a synthetic agentic env.

Reproduction target: arXiv:2610.11384 ("Environmental Feedback Modeling
Matters: Rethinking Feedback Treatment in Agentic Hindsight
Self-Distillation"). The paper's SELF framework jointly optimizes
(a) environmental-feedback modeling (predict the env's textual response to
an action) and (b) hindsight self-distillation (distill guidance from a
feedback-conditioned self-teacher into the policy), and reports that the
two reinforce each other (+4.1pp over GRPO on tau-bench with Qwen3-8B).

Bounded T4 version:
  * Env: deterministic calculator-task agent. State = running total;
    actions = "add k" / "sub k" / "mul 2" / "mul 3"; goal = reach a target
    in <= 5 steps. The env returns textual feedback after every action.
  * Teacher: scripted BFS solver (30% of episodes open with a random
    mistake, then recover). Failed episodes are hindsight-relabeled to the
    actually-reached total (HER-style).
  * Arm 1 (baseline): hindsight self-distillation only = SFT on teacher
    trajectories, loss on action tokens (feedback present as context).
    This is the "simply conditioning the teacher on feedback" baseline
    the paper calls insufficient.
  * Arm 2 (SELF): same SFT + auxiliary env-feedback modeling loss
    (predict feedback tokens given history+action).
  * Eval: held-out targets, greedy rollouts, exact task success rate;
    plus held-out feedback-token perplexity for both arms (the
    mutually-reinforcing claim: joint training should also model
    feedback better).

Self-contained for Colab: pip installs at top of main() (incl. torchao
uninstall per the 2026-10-04 quirk), writes /content/self_results.json.
Importing this module has NO side effects (no pip, no torch) so the
env/teacher/data logic is CPU-testable on the VM.

Model: Qwen3-8B (the paper's own model; text-only causal LM - Qwen3.5-9B
is a multimodal Qwen3_5ForConditionalGeneration build, so a text-only
QLoRA run would needlessly load its vision tower on the T4).
NF4 QLoRA r=16/alpha=32, bf16, transformers<5.0, gradient checkpointing.
"""

# --------------------------------------------------------------------------
# 1. synthetic agentic env (stdlib only - CPU testable)
# --------------------------------------------------------------------------
import json
import math
import random
import re
import time
from collections import deque

MAX_STEPS = 5
MAX_LEN = 256


def apply_action(total, action):
    """Returns (new_total, feedback). Deterministic."""
    try:
        op, num_s = action.strip().split()
        num = int(num_s)
    except Exception:
        return total, "error: malformed action (total unchanged)"
    if op == "add" and 1 <= num <= 9:
        return total + num, f"total={total + num}"
    if op == "sub" and 1 <= num <= 9:
        if total - num < 0:
            return total, "error: total cannot go below 0 (total unchanged)"
        return total - num, f"total={total - num}"
    if op == "mul" and num in (2, 3):
        return total * num, f"total={total * num}"
    return total, "error: illegal action (total unchanged)"


def bfs(start, target, max_depth=MAX_STEPS):
    """Shortest action list from start to target, or None."""
    if start == target:
        return []
    seen = {start}
    q = deque([(start, [])])
    cands = ([f"add {k}" for k in range(1, 10)] +
             [f"sub {k}" for k in range(1, 10)] + ["mul 2", "mul 3"])
    while q:
        s, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for a in cands:
            ns, _ = apply_action(s, a)
            if ns == s or ns > 999 or ns in seen:
                continue
            if ns == target:
                return path + [a]
            seen.add(ns)
            q.append((ns, path + [a]))
    return None


def teacher_episode(start, target, rng, mistake_p=0.3):
    """Scripted teacher rollout with per-step env feedback.

    Returns (steps, reached, final_total) with steps = [(action, feedback)].
    ``mistake_p`` of episodes open with a random (possibly illegal) action,
    then recover with BFS.
    """
    steps, total, remaining = [], start, MAX_STEPS
    if rng.random() < mistake_p:
        a = rng.choice([f"add {rng.randint(1, 9)}",
                        f"sub {rng.randint(1, 9)}", "mul 2", "mul 3",
                        "div 2"])
        total, fb = apply_action(total, a)
        steps.append((a, fb))
        remaining -= 1
    while remaining > 0 and total != target:
        plan = bfs(total, target, remaining)
        if plan is None:
            break
        a = plan[0]
        total, fb = apply_action(total, a)
        steps.append((a, fb))
        remaining -= 1
    reached = (total == target)
    if reached and steps:
        steps[-1] = (steps[-1][0], f"goal reached: total={total}")
    return steps, reached, total


def build_tasks(n, seed):
    rng = random.Random(seed)
    tasks, guard = [], 0
    while len(tasks) < n and guard < n * 50:
        guard += 1
        start, target = rng.randint(0, 20), rng.randint(10, 120)
        if bfs(start, target) is not None:
            tasks.append((start, target))
    return tasks


def episode_text(start, target, steps):
    lines = [f"Task: reach {target}. Start: {start}."]
    for a, fb in steps:
        lines.append(f"> {a}")
        lines.append(fb)
    return "\n".join(lines) + "\n"


def make_data(n_train=120, n_eval=28, seed=0):
    """Teacher episodes with hindsight relabeling of failures."""
    rng = random.Random(seed)
    train, eval_tasks = [], build_tasks(n_eval, seed + 999)
    for start, target in build_tasks(n_train, seed):
        steps, reached, final = teacher_episode(start, target, rng)
        if not reached:
            target = final  # hindsight relabel -> relabeled goal reached
            if steps:
                a, _ = steps[-1]
                steps[-1] = (a, f"goal reached: total={final}")
        train.append((start, target, steps))
    return train, eval_tasks


# --------------------------------------------------------------------------
# 2. training + eval (heavy deps installed lazily inside main)
# --------------------------------------------------------------------------

def _install_deps():
    import subprocess
    import sys

    def pip(*pkgs):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        *pkgs], check=True)

    pip("torch", "bitsandbytes", "peft", "accelerate", "transformers<5.0")
    # Colab ships torchao 0.10; peft needs >0.16 (2026-10-04 quirk)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torchao"], capture_output=True)


def main():
    _install_deps()
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import (LoraConfig, get_peft_model,
                      prepare_model_for_kbit_training)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B",
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    PAD = tok.pad_token_id

    def encode_episode(start, target, steps):
        pieces = [(f"Task: reach {target}. Start: {start}.\n", "ctx")]
        for a, fb in steps:
            pieces.append((f"> {a}\n", "act"))
            pieces.append((f"{fb}\n", "fb"))
        ids, kinds = [], []
        for i, (txt, kind) in enumerate(pieces):
            t = tok(txt, add_special_tokens=(i == 0))["input_ids"]
            ids.extend(t)
            kinds.extend([kind] * len(t))
        ids, kinds = ids[:MAX_LEN], kinds[:MAX_LEN]
        lp = [t if k == "act" else -100 for t, k in zip(ids, kinds)]
        lf = [t if k == "fb" else -100 for t, k in zip(ids, kinds)]
        return ids, lp, lf

    def batchify(rows):
        # rows: list of (ids, lp, lf)
        out = {}
        for k, data in (("input_ids", [r[0] for r in rows]),
                        ("labels_p", [r[1] for r in rows]),
                        ("labels_f", [r[2] for r in rows])):
            ts = [torch.tensor(x) for x in data]
            pad = PAD if k == "input_ids" else -100
            out[k] = torch.nn.utils.rnn.pad_sequence(
                ts, batch_first=True, padding_value=pad).cuda()
        out["attention_mask"] = (out["input_ids"] != PAD).long()
        return out

    def train_arm(episodes, feedback_loss_w, tag, steps=100, lr=2e-4):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B", quantization_config=bnb,
            torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
        peft_cfg = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_cfg)
        model.train()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        rows = [encode_episode(s, t, st) for s, t, st in episodes]
        rng = random.Random(0)
        t0 = time.time()
        for step in range(steps):
            idx = rng.sample(range(len(rows)), 2)
            b = batchify([rows[i] for i in idx])
            out = model(input_ids=b["input_ids"],
                        attention_mask=b["attention_mask"])
            logits = out.logits[:, :-1, :].contiguous()
            flat = logits.view(-1, logits.size(-1))
            nll = torch.nn.functional.cross_entropy(
                flat, b["labels_p"][:, 1:].contiguous().view(-1),
                ignore_index=-100)
            loss = nll
            nll_f = torch.tensor(float("nan"))
            if feedback_loss_w > 0:
                nll_f = torch.nn.functional.cross_entropy(
                    flat, b["labels_f"][:, 1:].contiguous().view(-1),
                    ignore_index=-100)
                loss = loss + feedback_loss_w * nll_f
            (loss / 4).backward()
            if (step + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
            if (step + 1) % 30 == 0:
                print(f"[{tag}] step {step + 1}/{steps} "
                      f"loss={loss.item():.3f} nll_act={nll.item():.3f} "
                      f"nll_fb={nll_f.item():.3f} ({time.time() - t0:.0f}s)",
                      flush=True)
        model.eval()
        return model

    @torch.no_grad()
    def rollout(model, start, target):
        hist = f"Task: reach {target}. Start: {start}.\n"
        total = start
        for _ in range(MAX_STEPS):
            ids = tok(hist, return_tensors="pt").to("cuda")
            gen = model.generate(
                **ids, max_new_tokens=12, do_sample=False,
                pad_token_id=tok.eos_token_id)
            txt = tok.decode(gen[0][ids["input_ids"].shape[1]:])
            m = re.search(r">\s*([a-z]+\s+\d+)", txt)
            if not m:
                return False
            action = m.group(1)
            total, fb = apply_action(total, action)
            hist += f"> {action}\n{fb}\n"
            if total == target:
                return True
        return False

    @torch.no_grad()
    def feedback_ppl(model, episodes):
        tot_nll, tot_tok = 0.0, 0
        for s, t, st in episodes:
            b = batchify([encode_episode(s, t, st)])
            out = model(input_ids=b["input_ids"],
                        attention_mask=b["attention_mask"])
            logits = out.logits[:, :-1, :].contiguous()
            lab = b["labels_f"][:, 1:].contiguous().view(-1)
            nll = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), lab, ignore_index=-100,
                reduction="sum")
            tot_nll += nll.item()
            tot_tok += (lab != -100).sum().item()
        return math.exp(tot_nll / tot_tok)

    torch.manual_seed(0)
    t_all = time.time()
    train_eps, eval_tasks = make_data()
    print(f"data: {len(train_eps)} train episodes, "
          f"{len(eval_tasks)} eval tasks", flush=True)
    rng = random.Random(1234)
    probe = []
    for s, t in eval_tasks:
        steps, reached, final = teacher_episode(s, t, rng, mistake_p=0.3)
        probe.append((s, t if reached else final, steps))

    results = {"n_train": len(train_eps), "n_eval": len(eval_tasks)}
    for tag, w in (("sft_only", 0.0), ("self_joint", 0.5)):
        model = train_arm(train_eps, w, tag)
        ok = sum(rollout(model, s, t) for s, t in eval_tasks)
        ppl = feedback_ppl(model, probe)
        results[tag] = {"success_rate": ok / len(eval_tasks),
                        "success_n": ok,
                        "feedback_ppl": round(ppl, 2)}
        print(f"[{tag}] success {ok}/{len(eval_tasks)} = "
              f"{ok / len(eval_tasks):.3f}, feedback ppl {ppl:.2f}",
              flush=True)
        del model
        torch.cuda.empty_cache()
    results["elapsed_s"] = round(time.time() - t_all, 1)
    with open("/content/self_results_retry.json", "w") as f:
        json.dump(results, f, indent=1)
    print("wrote /content/self_results.json", flush=True)
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
