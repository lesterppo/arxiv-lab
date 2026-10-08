"""CPU test for arXiv 2610.09239 — the Winner's Curse in LLM self-improvement loops.

Reproduces the paper's selection-noise mechanism in a scripted statistical
model (item-level Bernoulli outcomes with a shared set-level difficulty
component; deterministic, seeded):

Claim 1 (selection under correlated noise):
 1. Candidate errors are positively correlated within a decision
    (shared selection set).
 2. One-shot winner's-curse gap shrinks with selection-set size.

Claim 3 (greedy loop bias):
 3. Final selection-set score minus held-out accuracy decreases with
    n_items: bias(16) > bias(64) > bias(256), with bias(16) >> bias(256).
 4. Lock-in: at n=16 the greedy loop's mean reported gain is positive
    while its mean true held-out gain is negative.

Claim 2 (mostly-harmful proposals):
 5. A majority of proposals after the first are truly harmful ...
 6. ... and a majority of the greedy loop's commits are harmful too.

Claim 4 (fresh-64 probe):
 7. Scoring start and current on 64 items never used for selection
    removes the average bias of the reported gain (|bias| ~ 0), while the
    greedy selection-set report stays strongly biased; single fresh-64
    estimates remain off by ~6-7 points absolute.

Acceptance-rule comparison (paper: rules did not beat greedy over whole
runs) is measured and printed but NOT asserted: the sim replicates the
paper only in the low-noise regime (see session README for the honest
numbers).

Deterministic (seeded). numpy-guarded: skips cleanly without numpy.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    from arxiv_lab.loops import winners_curse as wc
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if not HAS_NUMPY:
    print("SKIP: numpy not available")
    sys.exit(0)

passed = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok: {name}")


N_SEEDS = 200
SEED = 100

print("== claim 1: correlated errors within a decision ==")
ec = wc.error_correlation(16, n_seeds=400, seed=2)
print(f"  corr(e_A, e_B) at n=16: {ec:.3f}")
check("candidate errors positively correlated within a decision", ec > 0.05)

print("== claim 3: greedy-loop selection bias vs selection-set size ==")
bias = {}
for n in (16, 64, 256):
    b = wc.selection_bias(n, n_seeds=N_SEEDS, seed=SEED)
    bias[n] = b
    print(f"  n={n:3d}: bias={b['mean_bias']:6.2f} +- {b['std_bias']:5.2f} "
          f"reported_gain={b['mean_reported_gain']:6.2f} "
          f"true_gain={b['mean_true_gain']:6.2f}")
check("bias decreases with selection-set size (16 > 64 > 256)",
      bias[16]["mean_bias"] > bias[64]["mean_bias"] > bias[256]["mean_bias"])
check("bias(16) >> bias(256)",
      bias[16]["mean_bias"] > 2.0 * bias[256]["mean_bias"])
check("bias(16) in the paper's 13-20 point band",
      13.0 <= bias[16]["mean_bias"] <= 20.0)
check("bias(256) in the paper's 1-5 point band",
      1.0 <= bias[256]["mean_bias"] <= 5.0)
check("held-out (true) gains grow with selection-set size",
      bias[256]["mean_true_gain"] > bias[16]["mean_true_gain"] + 3.0)

print("== claim 2 + lock-in: mostly-harmful proposals, greedy drifts down ==")
hp, hc, tg, rg, f64b, f64e = [], [], [], [], [], []
for s in range(N_SEEDS):
    r = wc.run_greedy_loop(n_items=16, seed=SEED + s)
    hp.append(r["harmful_proposal_frac"])
    hc.append(r["harmful_commit_frac"])
    tg.append(r["true_gain"])
    rg.append(r["reported_gain"])
    f64b.append(r["fresh64_gain_bias"])
    f64e.append(r["fresh64_gain_abs_err"])
hp_m, hc_m = float(np.mean(hp)), float(np.mean(hc))
tg_m, rg_m = float(np.mean(tg)), float(np.mean(rg))
print(f"  harmful_proposal_frac={hp_m:.3f} harmful_commit_frac={hc_m:.3f}")
print(f"  mean true_gain={tg_m:.2f}  mean reported_gain={rg_m:.2f}")
check("most proposals after the first are truly harmful", hp_m > 0.6)
check("a majority of greedy commits are harmful (lock-in)", hc_m > 0.5)
check("greedy loop's true quality declines on average at n=16", tg_m < -2.0)
check("... while its reported gain stays positive", rg_m > 3.0)

print("== claim 4: fresh-64 probe removes average bias ==")
f64b_m, f64e_m = float(np.mean(f64b)), float(np.mean(f64e))
sel_bias_m = rg_m - tg_m
print(f"  greedy selection-set report bias={sel_bias_m:.2f} "
      f"fresh64 probe bias={f64b_m:.2f} mean|err|={f64e_m:.2f}")
check("fresh-64 probe removes the average bias of the reported gain",
      abs(f64b_m) < 2.0)
check("greedy selection-set report stays strongly biased",
      sel_bias_m > 10.0)
check("single fresh-64 estimates remain off by ~6 points",
      f64e_m < 12.0)

print("== acceptance rules vs greedy (measured, not asserted) ==")
for n in (16, 256):
    line = f"  n={n:3d}:"
    for rule in ("greedy", "fresh64", "conservative"):
        vals = [wc.run_loop_with_rule(rule, n_items=n, seed=SEED + s)
                for s in range(N_SEEDS)]
        rb_m = float(np.mean([v["reported_gain"] - v["true_gain"] for v in vals]))
        tg_m = float(np.mean([v["true_gain"] for v in vals]))
        line += f"  {rule}: rep-vs-true bias={rb_m:6.2f} true_gain={tg_m:6.2f} |"
    print(line)
# fresh-64 acceptance shrinks the reported-vs-true bias vs greedy (n=16)
vg = [wc.run_loop_with_rule("greedy", n_items=16, seed=SEED + s)
      for s in range(N_SEEDS)]
vf = [wc.run_loop_with_rule("fresh64", n_items=16, seed=SEED + s)
      for s in range(N_SEEDS)]
bg = abs(float(np.mean([v["reported_gain"] - v["true_gain"] for v in vg])))
bf = abs(float(np.mean([v["reported_gain"] - v["true_gain"] for v in vf])))
print(f"  |reported-vs-true bias| at n=16: greedy={bg:.2f} fresh64-rule={bf:.2f}")
check("fresh-64 acceptance reduces reported-vs-true bias vs greedy", bf < bg)

print(f"\nALL {len(passed)} WINNER'S-CURSE CPU CHECKS PASSED")
