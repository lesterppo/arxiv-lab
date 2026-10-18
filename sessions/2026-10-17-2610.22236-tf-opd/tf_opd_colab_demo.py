#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TF-OPD Colab demo — Training-Free On-Policy Distillation (arXiv:2610.22236).

STAGED, NOT EXECUTED. Run on Google Colab (T4) via:
    hermes-colab-cli/colab.py exec ...  (or paste into a Colab cell)

What it does (all forward passes, no gradients):
  1. Loads Qwen2.5-0.5B-Instruct from Hugging Face (public, ~0.5B params).
  2. Builds two synthetic specialists from the base by adding small
     Gaussian perturbations to disjoint layer subsets ("math" perturbs
     MLP down-projections, "code" perturbs attention q/k projections).
     These STAND IN for RL-tuned specialists; they are synthetic deltas,
     not trained models — the demo exercises the fusion machinery, not
     real specialist consolidation.
  3. Each specialist greedy-generates short answers to toy domain prompts
     (its "rollouts"), then reads prompt+answer once with teacher forcing
     while forward hooks accumulate per-layer Gram matrices G = X^T X / n.
  4. Demonstrates the §3.4 repetition-loop filter (DEFLATE
     compressibility < 5% over the last 20k chars) on a crafted looping
     answer.
  5. Fuses every nn.Linear layer with the paper's closed form
     (Eq. 2, float64, alpha=0.9 off-diagonal shrinkage); biases combine
     by task arithmetic (§3.3).
  6. Prints per-layer unreproduced-share diagnostics (Eq. 4) and a tiny
     perplexity eval (base / specialists / fused on held-out toy prompts).

Expected runtime on a Colab T4: a few minutes.
"""

# ---- Colab setup: installs first, torchao removed for peft compat ----
# !pip install -q transformers torch accelerate
# !pip uninstall -y torchao

import copy
import math
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ALPHA = 0.9          # Eq. 3 off-diagonal shrinkage
N_ROLLOUTS = 64      # toy-scale stand-in for the paper's few-hundred
MAX_NEW_TOKENS = 96

MATH_PROMPTS = [
    "Solve: 17 * 23 =",
    "What is the derivative of x^3 + 2x?",
    "If a train travels 120 km in 2 hours, its speed is",
    "Compute: (48 + 52) / 4 =",
]
CODE_PROMPTS = [
    "Write a Python function to reverse a string:",
    "Write a Python function to check if a number is even:",
    "Complete: def add(a, b):",
    "Write a Python loop that prints numbers 1 to 5:",
]


def detect_repetition_loop(text, window=20000, ratio=0.05):
    """Paper §3.4: flag when the last `window` chars compress under
    DEFLATE to less than `ratio` of their length."""
    tail = text[-window:].encode("utf-8", errors="ignore")
    if not tail:
        return False
    return len(zlib.compress(tail, 9)) / len(tail) < ratio


def make_specialist(base_sd, seed, target_substrings):
    """Synthetic specialist: base weights + small Gaussian deltas on the
    linear layers whose names contain any of `target_substrings`."""
    g = torch.Generator().manual_seed(seed)
    sd = copy.deepcopy(base_sd)
    n_touched = 0
    for name, p in sd.items():
        if p.dim() == 2 and any(s in name for s in target_substrings):
            sd[name] = p + 0.02 * torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1])
            n_touched += 1
    print(f"specialist(seed={seed}): perturbed {n_touched} linear layers "
          f"matching {target_substrings}")
    return sd


class GramCollector:
    """Forward hooks accumulating per-layer X^T X / n_tokens (§3.2)."""

    def __init__(self, model):
        self.grams = {}
        self.handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                dev = next(mod.parameters()).device
                self.grams[name] = torch.zeros(mod.in_features, mod.in_features,
                                              dtype=torch.float64, device=dev)
                self.handles.append(
                    mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(mod, inputs, output):
            x = inputs[0].detach().to(torch.float64).reshape(-1, inputs[0].shape[-1])
            self.grams[name] += (x.T @ x) / x.shape[0]  # per-token average
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def collect_grams(model, tok, prompts):
    """Greedy-generate one answer per prompt, then one teacher-forced
    pass over prompt+answer per prompt while hooks accumulate grams."""
    collector = GramCollector(model)
    n_kept, n_loop = 0, 0
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)["input_ids"]
        gen = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0], skip_special_tokens=True)
        if detect_repetition_loop(text):  # §3.4 filter
            n_loop += 1
            continue
        full = tok(text, return_tensors="pt").to(model.device)["input_ids"]
        model(full)  # teacher-forced read; hooks accumulate
        n_kept += 1
    collector.remove()
    print(f"rollouts: kept={n_kept} dropped_loop={n_loop}")
    # average over kept answers so long answers don't dominate (§3.2)
    return {k: v / max(n_kept, 1) for k, v in collector.grams.items()}


def shrink(G, alpha=ALPHA):
    return alpha * G + (1 - alpha) * torch.diag(torch.diag(G))


@torch.no_grad()
def fuse_state_dicts(base_sd, spec_sds, spec_grams):
    """Eq. 2 per linear layer in float64; biases by task arithmetic."""
    fused = {}
    for name, Wb in base_sd.items():
        if name in spec_grams[0] and Wb.dim() == 2:
            Wb64 = Wb.double()
            Gt = [shrink(g[name].double().cpu()) for g in spec_grams]
            S = sum(Gt)
            T = sum(G @ (sd[name].double() - Wb64) for G, sd in zip(Gt, spec_sds))
            Dstar = torch.linalg.pinv(S) @ T
            fused[name] = (Wb64 + Dstar).to(Wb.dtype)
        else:  # biases / norms: task arithmetic (§3.3)
            acc = Wb.clone()
            for sd in spec_sds:
                if name in sd:
                    acc = acc + (sd[name] - Wb)
            fused[name] = acc
    return fused


@torch.no_grad()
def unreproduced_share(Dstar, Dd, G):
    # Eq. 4: tr[(D*-Dd)^T G (D*-Dd)] / tr[Dd^T G Dd]
    diff = Dstar.double() - Dd.double()
    num = torch.trace(diff.T @ G.double() @ diff)
    den = torch.trace(Dd.double().T @ G.double() @ Dd.double())
    return float(num / den) if den > 0 else 0.0


@torch.no_grad()
def perplexity(model, tok, prompts):
    nll, n = 0.0, 0
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)["input_ids"]
        out = model(ids, labels=ids)
        nll += out.loss.item() * ids.shape[1]
        n += ids.shape[1]
    return math.exp(nll / n)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    base_sd = {k: v.cpu() for k, v in base.state_dict().items()}

    # 1-2. synthetic specialists (stand-ins for RL-tuned ones)
    sd_math = make_specialist(base_sd, seed=0, target_substrings=["mlp.down_proj"])
    sd_code = make_specialist(base_sd, seed=1, target_substrings=["self_attn.q_proj",
                                                                  "self_attn.k_proj"])

    # 3. on-policy statistics: each specialist reads its own answers
    grams = []
    for tag, sd, prompts in [("math", sd_math, MATH_PROMPTS),
                             ("code", sd_code, CODE_PROMPTS)]:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16).to(device).eval()
        m.load_state_dict({k: v.to(device) for k, v in sd.items()})
        print(f"--- collecting grams: {tag} specialist ---")
        grams.append(collect_grams(m, tok, prompts * (N_ROLLOUTS // len(prompts))))
        del m
        torch.cuda.empty_cache() if device == "cuda" else None

    # 4. loop-filter sanity check on a crafted looping answer
    loop_txt = ("the answer is 42 and the answer is 42 and ") * 1200
    ok_txt = " ".join(MATH_PROMPTS * 200)
    print("loop filter: looping ->", detect_repetition_loop(loop_txt),
          "| normal ->", detect_repetition_loop(ok_txt))

    # 5. closed-form fusion
    fused_sd = fuse_state_dicts(base_sd, [sd_math, sd_code], grams)
    fused = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    fused.load_state_dict({k: v.to(device) for k, v in fused_sd.items()})

    # 6a. diagnostics: Eq. 4 unreproduced share, median over fused layers
    for i, tag in enumerate(["math", "code"]):
        shares = []
        for name in grams[i]:
            Gt = shrink(grams[i][name].double().cpu())
            Dd = (sd_math if i == 0 else sd_code)[name].double() - base_sd[name].double()
            Dstar = fused_sd[name].double() - base_sd[name].double()
            shares.append(unreproduced_share(Dstar, Dd, Gt))
        shares = sorted(shares)
        print(f"eps_{tag}: median={shares[len(shares)//2]:.4f} "
              f"p90={shares[int(0.9*len(shares))]:.4f} "
              f"({len(shares)} linear layers)")

    # 6b. tiny eval: perplexity on held-out toy prompts
    held_math = ["Solve: 91 - 37 =", "What is 12 squared?"]
    held_code = ["Write a Python function to square a number:",
                 "Complete: def greet(name):"]
    models = {"base": base, "fused": fused}
    for tag, sd in [("math-spec", sd_math), ("code-spec", sd_code)]:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16).to(device).eval()
        m.load_state_dict({k: v.to(device) for k, v in sd.items()})
        models[tag] = m
    print(f"{'model':<10} {'ppl_math':>9} {'ppl_code':>9}")
    for tag, m in models.items():
        print(f"{tag:<10} {perplexity(m, tok, held_math):>9.3f} "
              f"{perplexity(m, tok, held_code):>9.3f}")


if __name__ == "__main__":
    main()
