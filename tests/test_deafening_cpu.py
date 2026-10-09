"""CPU test for arXiv 2610.09835 — Adam-epsilon intervention miniature.

Tests the paper's mechanism (not the LLM harness): persistent one-sided
gradients on absent tokens get amplified by Adam's sqrt(v_hat)
normalization into full-sized updates; raising eps for the output
projection dampens them while leaving normal-sized gradients alone.

Deterministic (seeded). numpy-guarded: skips cleanly without numpy.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    from arxiv_lab.training.deafening import (
        adam_row_drift,
        steady_update_magnitude,
        dampening_ratio,
    )
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


LR = 2e-4
print("== tiny-gradient amplification (paper Sec. 3 mechanism) ==")
# Persistent one-sided gradient g=1e-4 (absent token's softmax gradient):
# with eps=1e-8, sqrt(v_hat) -> |g|, so |update| -> lr (full-sized).
u = steady_update_magnitude(1e-4, lr=LR, eps=1e-8)
print(f"  steady |update| eps=1e-8: {u:.2e} (lr={LR:.0e})")
check("tiny gradient -> full-sized update at eps=1e-8", abs(u - LR) / LR < 0.05)

# Simulated drift matches the closed form after burn-in.
d = adam_row_drift(1e-4, 500, lr=LR, eps=1e-8)
print(f"  500-step drift eps=1e-8: {d:.4f} (~500*lr={500*LR:.4f})")
check("simulated drift ~= steps*lr at eps=1e-8", abs(d - 500 * LR) / (500 * LR) < 0.10)

print("== raised eps dampens the tiny-gradient regime ==")
ratio = dampening_ratio(g_small=1e-4, eps_low=1e-8, eps_high=1e-2,
                        steps=500, lr=LR)
print(f"  drift(eps=1e-2)/drift(eps=1e-8): {ratio:.4f}")
check("raised eps removes >80% of rare-token drift", ratio < 0.2)

print("== selectivity: normal gradients untouched ==")
# A real learning signal has |g| >> eps_high: then |update| -> lr either way.
u_big_low = steady_update_magnitude(1.0, lr=LR, eps=1e-8)
u_big_high = steady_update_magnitude(1.0, lr=LR, eps=1e-2)
print(f"  |update| g=1.0: eps=1e-8 -> {u_big_low:.2e}, eps=1e-2 -> {u_big_high:.2e}")
check("real learning (g=1.0) barely affected by raised eps",
      abs(u_big_high - u_big_low) / u_big_low < 0.02)

print("== sanity ==")
check("zero gradient -> zero drift",
      adam_row_drift(0.0, 100, lr=LR, eps=1e-8) == 0.0)
check("drift grows with steps",
      adam_row_drift(1e-4, 1000, lr=LR) > adam_row_drift(1e-4, 100, lr=LR))

print(f"\nALL {len(passed)} DEAFENING CPU CHECKS PASSED")
