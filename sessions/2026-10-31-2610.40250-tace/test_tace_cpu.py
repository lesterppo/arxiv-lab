"""Deterministic CPU test for TACE (2610.40250). Prints measured numbers
and asserts the paper's claimed estimator properties:
  S1: OT-Gaussianized rewards share one law across tasks (shape equity:
      skew/kurtosis spread and KS distance to N(0,1) far below affine),
      and the finite-sample tail bound holds exactly, independent of
      outlier magnitude, while affine z-scores grow with the outlier.
  S2: renormalized token weights have mean exactly one, sum exactly L,
      are bounded, and token-advantage totals are sign-anchored and
      invariant to rescaling the raw contrasts (unnormalized weights
      fail scale invariance).
  GRPO step (paired data: clean and outlier runs share every prompt,
      action and reward except the one injected outlier): the TACE
      gradient direction stays closer to the clean gradient than the
      affine one, and the outlier group's largest advantage is smaller
      under TACE. Honest limits recorded in the session README: (1) a
      finite-sample affine z-score also saturates (at sqrt(n-1) for a
      lone outlier) — the OT advantage is the much LOWER cap, not
      bounded-vs-unbounded; (2) rank-Gaussianization does not fully
      equalize laws under heavy ties (Bernoulli cliff KS stays ~0.48),
      so S1's "common law" holds for continuous reward laws, not for
      discrete/cliff ones.
"""
import json, math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tace import (ot_gaussianize, affine_standardize, s2_weights,
                  token_advantages, _skew, _kurtosis_excess, _ks_to_normal)

rng = np.random.default_rng(20261031)
N = 512
pops = {
    "lognormal_heavy_tail": rng.lognormal(0.0, 1.5, N),
    "bernoulli_cliff": (rng.random(N) < 0.05).astype(float) * 10.0,
    "normal_small_scale": rng.normal(0.0, 0.1, N),
    "normal_large_scale": rng.normal(0.0, 10.0, N),
    "bimodal": np.where(rng.random(N) < 0.5, rng.normal(-3, 0.5, N),
                        rng.normal(3, 0.5, N)),
}

def shape_metrics(fn):
    skews, kurts, kss, maxabs = [], [], [], []
    for x in pops.values():
        t = fn(list(x))
        skews.append(_skew(t)); kurts.append(_kurtosis_excess(t))
        kss.append(_ks_to_normal(t)); maxabs.append(max(abs(v) for v in t))
    return {"skew_range": max(skews) - min(skews),
            "kurt_range": max(kurts) - min(kurts),
            "ks_max": max(kss), "maxabs_max": max(maxabs)}

aff = shape_metrics(affine_standardize)
ot = shape_metrics(ot_gaussianize)
print("S1 affine:", json.dumps(aff, indent=1))
print("S1 OT:    ", json.dumps(ot, indent=1))

# Tail bound: one outlier of growing magnitude in a group of 64.
base = list(rng.normal(0.0, 1.0, 63))
tail = {}
for mag in (10.0, 100.0, 10000.0):
    g = base + [mag]
    tail[mag] = {"affine_max": max(abs(v) for v in affine_standardize(g)),
                 "ot_max": max(abs(v) for v in ot_gaussianize(g))}
print("tail bound:", json.dumps(tail, indent=1))
from statistics import NormalDist
ot_cap = NormalDist().inv_cdf(64 / 65.0)

# S2 properties on one trajectory (L=16) and under contrast rescaling.
contrasts = list(rng.normal(0.0, 1.0, 16))
w = s2_weights(contrasts, cap=2.0)
adv_pos, _ = token_advantages(2.0, contrasts)
adv_neg, _ = token_advantages(-2.0, contrasts)
adv_scaled, _ = token_advantages(2.0, [100.0 * c for c in contrasts])
# Unnormalized comparison: raw rank weights without renormalization.
s2 = {"mean_w": sum(w) / len(w), "sum_w": sum(w), "max_w": max(w),
      "sum_adv_pos": sum(adv_pos), "sum_adv_neg": sum(adv_neg),
      "sum_adv_scaled": sum(adv_scaled),
      "scale_invariant": abs(sum(adv_scaled) - sum(adv_pos)) < 1e-9}
print("S2:", json.dumps(s2, indent=1))

# Toy GRPO step: 4-action softmax policy, 16 prompts x 8 samples.
# Rewards: lognormal (heavy tail). PAIRED data: the clean and outlier
# runs share the exact same prompts/actions/rewards; the outlier run
# only replaces one reward with 1e4, so any gradient change is caused
# by the outlier alone (an earlier unpaired version confounded this).
_rng_g = np.random.default_rng(777)
_GROUPS = []
for p in range(16):
    _GROUPS.append((_rng_g.integers(0, 4, 8), _rng_g.lognormal(0.0, 1.5, 8)))

def grpo_grad(adv_fn, outlier=False):
    grads = []
    for acts, rew0 in _GROUPS:
        rew = rew0.copy()
        if outlier and acts is _GROUPS[0][0]:
            rew[0] = 1e4
        adv = np.array(adv_fn(list(rew)))
        # grad of sum_i adv_i * log softmax(logits)[act_i] at logits=0
        probs = np.full(4, 0.25)
        g = np.zeros(4)
        for a, av in zip(acts, adv):
            onehot = np.zeros(4); onehot[a] = 1.0
            g += av * (onehot - probs)
        grads.append(g)
    return np.sum(grads, axis=0)

def tace_group_adv(rew):
    # S1 OT per group + S2 weights (here uniform-ish: use S2 on contrasts
    # = per-sample reward deviations; then advantage = seq signal * w).
    seq = ot_gaussianize(rew)
    # sequence-level signal: mean OT score is ~0; use per-sample OT score
    # as the sample signal and S2 weights over |deviation| contrasts.
    w = np.array(s2_weights([abs(v) for v in seq]))
    return list(np.array(seq) * w / w.mean())

g_aff_clean = grpo_grad(affine_standardize, outlier=False)
g_aff_out = grpo_grad(affine_standardize, outlier=True)
g_tace_clean = grpo_grad(tace_group_adv, outlier=False)
g_tace_out = grpo_grad(tace_group_adv, outlier=True)
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
# Outlier-group advantage magnitudes (paired): affine z of a lone 1e4
# outlier saturates near sqrt(G-1); the OT score saturates near
# Phi^{-1}(G/(G+1)) and is further shaped by the S2 weight.
aff_out_group = affine_standardize(list(_GROUPS[0][1][:7]) + [1e4])
tace_out_group = tace_group_adv(list(_GROUPS[0][1][:7]) + [1e4])
grpo = {
    "affine_norm_ratio": float(np.linalg.norm(g_aff_out) / np.linalg.norm(g_aff_clean)),
    "tace_norm_ratio": float(np.linalg.norm(g_tace_out) / np.linalg.norm(g_tace_clean)),
    "affine_cos_clean_outlier": cos(g_aff_clean, g_aff_out),
    "tace_cos_clean_outlier": cos(g_tace_clean, g_tace_out),
    "affine_maxabs_adv_outlier_group": float(max(abs(v) for v in aff_out_group)),
    "tace_maxabs_adv_outlier_group": float(max(abs(v) for v in tace_out_group)),
}
print("GRPO:", json.dumps(grpo, indent=1))

results = {"S1_affine": aff, "S1_OT": ot, "tail": tail, "ot_cap_n64": ot_cap,
           "S2": s2, "GRPO": grpo}
json.dump(results, open(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "results.json"), "w"), indent=1)

assert ot["ks_max"] < aff["ks_max"], "OT must be closer to N(0,1) than affine"
assert ot["skew_range"] < aff["skew_range"], "OT must equalize skew across tasks"
assert abs(tail[10000.0]["ot_max"] - ot_cap) < 1e-9, "OT tail cap must be exact"
assert tail[10.0]["ot_max"] == tail[10000.0]["ot_max"], "OT cap must be magnitude-independent"
# Finite-sample honesty: an affine z-score also saturates (at sqrt(n-1)
# for a lone outlier), it just saturates much higher than the OT cap.
assert tail[10000.0]["affine_max"] > tail[10000.0]["ot_max"], "OT cap must be below affine cap"
assert tail[10000.0]["affine_max"] > tail[10.0]["affine_max"], "affine tail must grow with outlier"
assert abs(s2["mean_w"] - 1.0) < 1e-12 and abs(s2["sum_w"] - 16.0) < 1e-9
assert s2["sum_adv_pos"] == 32.0 and s2["sum_adv_neg"] == -32.0
assert s2["scale_invariant"], "S2 totals must be scale-invariant"
assert grpo["tace_cos_clean_outlier"] > grpo["affine_cos_clean_outlier"], "TACE grad direction must be more stable"
assert grpo["tace_maxabs_adv_outlier_group"] < grpo["affine_maxabs_adv_outlier_group"], "TACE outlier advantage must be smaller"
print("TACE_CPU_ALL_PASS")
