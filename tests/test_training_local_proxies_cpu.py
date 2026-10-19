"""CPU test for arXiv 2610.23145 — Local proxies for memory-efficient training.

Reproduces the paper's three checkable claims on CPU (numpy, seeded):

1. EXACTNESS (deep linear nets): per-layer least-squares proxies
   (block input -> global output) yield proxy-propagated gradients equal to
   exact backprop gradients (paper: "in theory, this approach is exact for
   deep linear networks").
2. ERROR BOUND (nonlinear residual nets): with a residual-form proxy, the
   gradient-estimation error shrinks as the proxy distillation loss is
   optimized (paper: "optimizing the distillation loss improves gradient
   estimation"; error bounded with the appropriate residual proxy).
3. BEATS ZEROTH-ORDER (paper's LLM-finetuning claim, CPU analog): on a
   synthetic classifier, proxy-local training reaches lower loss than
   MeZO-style SPSA within the same step budget, with strictly fewer
   retained activation tensors than backprop.

Deterministic (seeded). numpy-guarded (skips cleanly without numpy).

Run:  python3 tests/test_training_local_proxies_cpu.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
except ImportError:
    np = None

if np is None:
    print("SKIP: numpy not available")
    sys.exit(0)

from arxiv_lab.training.local_proxies import (  # noqa: E402
    linear_exactness,
    residual_error_bound,
    three_way_comparison,
)


def main():
    # — Claim 1: exactness on deep linear nets —
    max_rel_err = linear_exactness()
    print("linear exactness: max rel grad err = %.3e" % max_rel_err)
    assert max_rel_err < 1e-8, "proxy grads not exact on deep linear net"

    # — Claim 2: distillation loss down -> gradient error down —
    mses, coss = residual_error_bound()
    print("residual proxy checkpoints:")
    for m, c in zip(mses, coss):
        print("  distill MSE=%.6f  grad cos-sim=%.6f" % (m, c))
    assert mses[-1] < mses[0], "proxy distillation did not reduce MSE"
    assert coss[-1] > coss[0], "gradient estimate did not improve"
    # monotone-ish improvement of the estimate with better proxies
    assert coss[-1] > 0.97, "final proxy grad cosine sim too low"
    corr = np.corrcoef(np.log10(np.array(mses) + 1e-12), coss)[0, 1]
    print("  corr(log MSE, cos-sim) = %.3f" % corr)
    assert corr < -0.7, "expected strong negative corr(MSE, grad quality)"

    # — Claim 3: proxy-local beats MeZO at comparable (lower) memory —
    res = three_way_comparison()
    f = res["final"]
    s = res["slots"]
    print("three-way final loss: backprop=%.4f proxy=%.4f mezo=%.4f"
          % (f["backprop"], f["proxy"], f["mezo"]))
    print("activation slots: %s" % s)
    assert s["proxy"] < s["backprop"], "no memory win vs backprop"
    assert s["mezo"] == 0
    assert f["proxy"] < f["mezo"], "proxy-local did not beat zeroth-order"
    assert f["proxy"] <= 1.35 * f["backprop"], \
        "proxy-local degraded too far vs backprop"
    # all methods actually learned something
    assert f["backprop"] < 0.55, "backprop baseline did not learn"

    print("\nALL LOCAL-PROXY CHECKS PASSED")


if __name__ == "__main__":
    main()
