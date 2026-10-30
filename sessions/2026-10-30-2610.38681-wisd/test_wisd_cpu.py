# CPU check for WISD (arXiv 2610.38681) — stdlib + numpy only, no GPU.
# Validates, on toy data, the exact operators the Colab driver uses:
#  (1) group-relative magnitude pruning (tau=1/6, groups of 128 input
#      columns) + per-output-row L2-norm restore;
#  (2) tilted-target math: log q_w = (1+1/alpha) log pi0 - (1/alpha) log pi_w
#      - logZ, alpha=2; alpha -> infinity recovers pi0;
#  (3) top-K cache with force-included sampled token + renormalised KL_K;
#  (4) GSM8K prompt formatting + final-answer parsing (#### rule + fallback).
# The Colab driver mirrors these functions (same formulas, same constants).
import math
import re
import numpy as np

rng = np.random.default_rng(20261030)
TAU = 1.0 / 6.0
GROUP = 128
ALPHA = 2.0
TOPK = 64


def prune_weights(W, tau=TAU, group=GROUP):
    """Prune |w| < tau * group-max (groups along input dim), then restore
    each output row's L2 norm. Returns (pruned, mask, pruned_fraction)."""
    W = np.asarray(W, dtype=np.float64)
    out_dim, in_dim = W.shape
    assert in_dim % group == 0
    G = W.reshape(out_dim, in_dim // group, group)
    gmax = np.abs(G).max(axis=2, keepdims=True)
    mask = (np.abs(G) >= tau * gmax).reshape(out_dim, in_dim)
    P = W * mask
    orig_norm = np.linalg.norm(W, axis=1, keepdims=True)
    new_norm = np.linalg.norm(P, axis=1, keepdims=True)
    scale = orig_norm / np.maximum(new_norm, 1e-12)
    P = P * scale
    return P, mask, float(1.0 - mask.mean())


def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def tilted_target(logp0, logpw, alpha=ALPHA):
    logq = (1.0 + 1.0 / alpha) * logp0 - (1.0 / alpha) * logpw
    logq = logq - np.log(np.exp(logq - logq.max(axis=-1, keepdims=True)).sum(
        axis=-1, keepdims=True)) - logq.max(axis=-1, keepdims=True) \
        if False else log_softmax(logq, axis=-1)
    # NOTE: log_softmax above already normalises; the expression is
    # (1+1/a)logp0 - (1/a)logpw - logZ with logZ = logsumexp(...).
    return logq


def topk_cache(logq, logp0, sampled_ids, K=TOPK):
    """Per position: top-K of logq, force-include sampled token (replace
    lowest entry), renormalised logq_K / logp0_K over the kept set."""
    T, V = logq.shape
    idx = np.argpartition(-logq, K - 1, axis=1)[:, :K]
    for t in range(T):
        s = int(sampled_ids[t])
        if s not in idx[t]:
            worst = int(np.argmin(logq[t, idx[t]]))
            idx[t, worst] = s
    rows = np.arange(T)[:, None]
    lq = logq[rows, idx]
    lp = logp0[rows, idx]
    lq_n = lq - np.log(np.exp(lq - lq.max(axis=1, keepdims=True)).sum(
        axis=1, keepdims=True)) - lq.max(axis=1, keepdims=True)
    lp_n = lp - np.log(np.exp(lp - lp.max(axis=1, keepdims=True)).sum(
        axis=1, keepdims=True)) - lp.max(axis=1, keepdims=True)
    return idx, lq_n, lp_n


def kl_restricted(logp_student, logq_target):
    p = np.exp(logp_student)
    return float((p * (logp_student - logq_target)).sum())


PROMPT_TEMPLATE = ("Solve the following math problem step by step. "
                   "Give the final answer after '####'.\n\nProblem: {q}\n\nSolution:")


def make_prompt(question):
    return PROMPT_TEMPLATE.format(q=question)


def parse_answer(text):
    """GSM8K-style: number after last '####'; fallback: last number."""
    if text is None:
        return None
    m = None
    if "####" in text:
        tail = text.split("####")[-1].replace("$", "")
        m = re.search(r"(-?\d[\d,]*(?:\.\d+)?)", tail)
    if m is None:
        nums = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", text.replace("$", ""))
        if not nums:
            return None
        return float(nums[-1].replace(",", ""))
    return float(m.group(1).replace(",", ""))


def num_equal(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6


# ---- (1) pruning -----------------------------------------------------------
W = rng.normal(0, 1, size=(8, 256))
P, mask, frac = prune_weights(W)
assert P.shape == W.shape
assert 0.0 < frac < 0.9, f"pruned fraction implausible: {frac}"
assert np.allclose(np.linalg.norm(P, axis=1), np.linalg.norm(W, axis=1), rtol=1e-9), \
    "row L2 norms not restored"
# every surviving entry was >= threshold in its group; every pruned one below
G = W.reshape(8, 2, GROUP)
gmax = np.abs(G).max(axis=2, keepdims=True)
thr = np.broadcast_to(TAU * gmax, G.shape).reshape(8, 256)
assert np.all((np.abs(W) >= thr) == mask), "mask != group-relative threshold rule"
assert np.all(P[~mask] == 0.0), "pruned entries not zero"
print(f"PASS pruning: frac={frac:.3f} row-norms restored (max rel err "
      f"{np.max(np.abs(np.linalg.norm(P, axis=1) / np.linalg.norm(W, axis=1) - 1)):.2e})")

# ---- (2) tilted target ------------------------------------------------------
logits0 = rng.normal(0, 2, size=(5, 20))
logitsw = logits0 + rng.normal(0, 1, size=(5, 20))  # weakened: perturbed logits
logp0 = log_softmax(logits0, axis=-1)
logpw = log_softmax(logitsw, axis=-1)
logq = tilted_target(logp0, logpw, ALPHA)
assert np.allclose(np.exp(logq).sum(axis=-1), 1.0, atol=1e-9), "q_w not normalised"
# alpha -> infinity recovers pi0
logq_inf = tilted_target(logp0, logpw, alpha=1e9)
assert np.allclose(logq_inf, logp0, atol=1e-6), "alpha->inf must recover pi0"
# manual logZ check on one position
t = 0
raw = 1.5 * logp0[t] - 0.5 * logpw[t]
logZ = np.log(np.exp(raw - raw.max()).sum()) + raw.max()
assert np.allclose(logq[t], raw - logZ, atol=1e-12), "logZ mismatch"
kl0q = float((np.exp(logp0) * (logp0 - logq)).sum(axis=-1).mean())
assert kl0q >= -1e-9, "KL(pi0||q_w) negative"
print(f"PASS tilted target: normalised, alpha->inf recovers pi0, "
      f"mean KL(pi0||q_w)={kl0q:.4f}")

# ---- (3) top-K cache + restricted KL ----------------------------------------
sampled = np.array([int(np.argmax(logits0[t])) for t in range(5)])
idx, lq_n, lp_n = topk_cache(logq, logp0, sampled, K=4)
assert idx.shape == (5, 4)
for t in range(5):
    assert int(sampled[t]) in idx[t], "sampled token missing from cache set"
    assert np.allclose(np.exp(lq_n[t]).sum(), 1.0, atol=1e-9)
    assert np.allclose(np.exp(lp_n[t]).sum(), 1.0, atol=1e-9)
# KL of a distribution to itself is 0; KL to a different target is >= 0
assert abs(kl_restricted(lq_n[0], lq_n[0])) < 1e-9
kl_k = kl_restricted(lp_n[0], lq_n[0])
assert kl_k >= -1e-9
# student == pi0 restricted: KL_K(pi0_K || q_K) equals the restricted KL
print(f"PASS top-K cache: K=4 sets include sampled token, renormalised, "
      f"KL_K(pi0||q) pos0={kl_k:.4f}")

# ---- (4) prompt + parsing ----------------------------------------------------
q = "Janet sells 16 eggs a day. She makes $2 per egg. How much does she make?"
prompt = make_prompt(q)
assert q in prompt and "####" in prompt
cases = [
    ("First 16*2=32. #### 32", 32.0),
    ("#### $1,234.50", 1234.50),
    ("The answer is 7.", 7.0),  # fallback: last number
    ("No numbers here.", None),
    ("#### -12.5", -12.5),
]
for text, gold in cases:
    got = parse_answer(text)
    assert (got is None and gold is None) or num_equal(got, gold), \
        f"parse {text!r}: got {got}, want {gold}"
print("PASS prompt/parsing: #### rule + last-number fallback, numeric equality")

# ---- (5) GSM8K load (best effort; driver on Colab does the real load) -----
try:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"INFO gsm8k train rows={len(ds)} columns={ds.column_names}")
    assert "question" in ds.column_names and "answer" in ds.column_names
    print("PASS gsm8k load on VM")
except Exception as e:
    print(f"SKIP gsm8k load on VM ({type(e).__name__}: {str(e)[:120]}); "
          "Colab driver loads it and asserts columns.")

print("CPU_CHECK_ALL_PASS")
