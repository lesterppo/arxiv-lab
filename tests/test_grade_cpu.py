"""CPU test for arXiv 2610.07553 — GRADE gradient admission.

Reproduces the paper's closed-loop admission math in a synthetic
multi-task gradient field (deterministic, seeded):

Mechanism 1 (keep fraction, Eq. 2):
 1. r* follows pool geometry: orthogonal tasks -> r* -> 1 (no filtering);
    redundant/aligned tasks -> r* -> 1/T (strong pruning).
 2. Safeguards: inter-cosine floor at 1e-8, hard fallback 0.5 on
    degenerate denominator, global clamp to [0.1, 1.0].
 3. Forward-probe admission: top-kb by a_i = d_i * d_ref admits a higher
    fraction of reference-aligned samples than the raw pool (the probe
    is an unbiased inner-product estimate, App. M).

Mechanism 2 (step-level gate, Sec. 4.2):
 4. Gate latches once the probe-loss EMA plateaus (relative slope <
    eps_rel), and afterwards skips steps whose probe loss exceeds the
    EMA while committing steps below it.

Deterministic (seeded). numpy-guarded: skips cleanly without numpy.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    from arxiv_lab.training.grade import (
        r_star,
        cosine_stats,
        probe_scores,
        admit_topk,
        AdmissionGate,
        make_task_field,
        simulate_admission,
    )
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if not HAS_NUMPY:
    print("SKIP: numpy not available")
    sys.exit(0)

rng = np.random.default_rng(7)
passed = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok: {name}")


print("== r* geometry ==")
# Orthogonal tasks: cos_inter ~ 0 -> r* -> 1
r_orth = r_star(0.20, 0.0, T=5)
print(f"  orthogonal: r*={r_orth:.3f}")
check("orthogonal pool -> r* ~= 1 (no filtering)", r_orth > 0.95)
# Fully redundant: cos_inter == cos_intra -> r* -> 1/T
r_red = r_star(0.20, 0.20, T=5)
print(f"  redundant:  r*={r_red:.3f}")
check("redundant pool -> r* ~= 1/T", abs(r_red - 0.2) < 0.02)
# Paper's production regime (Table 2 rows): r* in [0.378, 0.779]
for ci, ce, lo, hi in [(0.2042, 0.0097, 0.75, 0.82),
                       (0.2114, 0.0579, 0.35, 0.42),
                       (0.1392, 0.0316, 0.39, 0.46)]:
    r = r_star(ci, ce, T=7)
    print(f"  paper row ci={ci} ce={ce}: r*={r:.3f}")
    check(f"paper regime r* in [{lo},{hi}]", lo <= r <= hi)

print("== r* safeguards ==")
check("negative inter-cosine floored -> r* ~= 1",
      abs(r_star(0.2, -0.05, T=5) - 1.0) < 1e-6)
check("degenerate denominator -> fallback 0.5",
      r_star(-0.1, 0.0, T=5) == 0.5)
check("clamp upper", abs(r_star(10.0, 1e-9, T=5) - 1.0) < 1e-6)
check("clamp lower", r_star(1e-9, 10.0, T=5) == 0.1)

print("== cosine_stats ==")
dirs = make_task_field(K=4, q=32, seed=1)
grads = np.vstack([dirs[k] + 0.05 * rng.standard_normal((25, 32))
                   for k in range(4)])
labels = np.repeat(np.arange(4), 25)
ci, ce = cosine_stats(grads, labels)
print(f"  cos_intra={ci:.3f} cos_inter={ce:.3f}")
check("intra >> inter on near-orthogonal tasks", ci > 0.9 and abs(ce) < 0.2)
r = r_star(ci, ce, T=4)
check("near-orthogonal -> r* close to 1", r > 0.9)

print("== probe admission ==")
res = simulate_admission(K=4, q=32, n_cand=200, kb=50, seed=3, noise=0.3,
                         conflict_task=3, n_probes=4)
print(f"  pool aligned frac={res['pool_align_frac']:.3f} "
      f"admitted={res['admitted_align_frac']:.3f}")
print(f"  pool conflict frac={res['pool_conflict_frac']:.3f} "
      f"admitted={res['admitted_conflict_frac']:.3f}")
check("admission enriches aligned samples",
      res["admitted_align_frac"] > res["pool_align_frac"] + 0.1)
check("admission suppresses the conflicting task",
      res["admitted_conflict_frac"] < res["pool_conflict_frac"] - 0.1)
# score is magnitude-weighted projection, not normalized cosine
s = probe_scores(np.array([2.0, -1.0]), 0.5)
check("probe_scores = d_i * d_ref", bool(np.allclose(s, [1.0, -0.5])))
adm = admit_topk([0.1, 0.9, 0.3, 0.7], 2)
check("admit_topk picks highest scores", set(adm.tolist()) == {1, 3})

print("== admission gate ==")
# decreasing-then-flat probe loss: gate must latch, then skip bad steps
gate = AdmissionGate(alpha=0.1, eps_rel=0.01)
losses = [1.0 - 0.01 * t for t in range(30)]  # steady descent
losses += [0.70] * 15                          # plateau
decisions = [gate.observe(L, t) for t, L in enumerate(losses)]
check("gate latches during plateau", gate.latched)
print(f"  latched at step {gate.latch_step}")
# now feed a damaging step (probe above EMA) -> skip; good step -> commit
d_bad = gate.observe(0.95, 99)
check("damaging step skipped after latch", d_bad == "skip")
d_good = gate.observe(0.60, 100)
check("constructive step committed after latch", d_good == "commit")
check("skip counted", gate.n_skipped >= 1)

print(f"\nALL {len(passed)} GRADE CPU CHECKS PASSED")
