"""CPU test for arXiv 2610.12995 — CIG-SFT conditional-information-gain weighting.

Reproduces the paper's closed-loop weighting math in a synthetic
token field (deterministic, seeded): boilerplate tokens predictable with
or without the task input must get ~1x-or-lower weight, substance tokens
only predictable with the task input must be strongly upweighted.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    from arxiv_lab.training.cig_sft import cig_weights, weighted_nll, simulate_admission
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

print("== cig_weights ==")
# zero CIG everywhere -> uniform weights
w = cig_weights([-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0])
check("zero CIG -> uniform", np.allclose(w, [1.0, 1.0, 1.0]))
check("mean-1 normalized", abs(w.mean() - 1.0) < 1e-9)
# high-CIG token upweighted, low-CIG downweighted
w = cig_weights([-0.1, -5.0, -5.0], [-6.0, -5.1, -4.9])
check("informative token upweighted", w[0] > 2.0)
check("boilerplate tokens downweighted", w[1] < 1.0 and w[2] < 1.0)
# clipping bounds the weight ratio
import math as _math
w = cig_weights([0.0], [-100.0], clip=5.0)
check("clip bounds weight", w[0] <= _math.exp(5.0) + 1e-9)

print("== weighted_nll ==")
nll = weighted_nll([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
check("uniform weights = mean NLL", abs(nll - 2.0) < 1e-9)
nll = weighted_nll([1.0, 100.0], [1.0, 0.0])
check("zero weight ignores token", abs(nll - 1.0) < 1e-9)

print("== simulate_admission ==")
res = simulate_admission(seed=7)
print(f"  boiler_w={res['boiler_w']:.3f} substance_w={res['substance_w']:.3f} "
      f"enrichment={res['enrichment']:.2f}x")
check("boilerplate weight below 1", res["boiler_w"] < 1.0)
check("substance weight above 1", res["substance_w"] > 1.0)
check("enrichment strong (>3x)", res["enrichment"] > 3.0)
# determinism
res2 = simulate_admission(seed=7)
check("deterministic", res == res2)

print(f"\nALL {len(passed)} CIG-SFT CPU CHECKS PASSED")
