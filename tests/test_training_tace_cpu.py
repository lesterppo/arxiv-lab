"""CPU tests for TACE (arXiv 2610.40250).

Property checks (stdlib only): S1 OT tail cap exact and
magnitude-independent and below the affine cap; S1 shape equity improves
over affine on continuous laws; S2 weights mean exactly one, sum exactly
L, totals sign-anchored and scale-invariant.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.training.tace import (
    affine_standardize,
    ks_to_normal,
    ot_gaussianize,
    ot_tail_cap,
    s2_weights,
    skew,
    token_advantages,
)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


def main():
    # S1 tail bound: exact, magnitude-independent, below affine cap.
    base = [float(i) for i in range(63)]
    for mag in (10.0, 100.0, 10000.0):
        g = base + [mag]
        check("OT max == cap for mag %g" % mag,
              abs(max(abs(v) for v in ot_gaussianize(g)) - ot_tail_cap(64)) < 1e-9)
    check("OT cap below affine cap at 1e4",
          ot_tail_cap(64) < max(abs(v) for v in affine_standardize(base + [10000.0])))
    # S1 shape equity on continuous laws (lognormal vs wide normal).
    import random
    rng = random.Random(20261031)
    heavy = [rng.lognormvariate(0.0, 1.5) for _ in range(512)]
    wide = [rng.gauss(0.0, 10.0) for _ in range(512)]
    aff_sk = abs(skew(affine_standardize(heavy)) - skew(affine_standardize(wide)))
    ot_sk = abs(skew(ot_gaussianize(heavy)) - skew(ot_gaussianize(wide)))
    check("OT skew gap below affine skew gap", ot_sk < aff_sk)
    check("OT KS below affine KS on heavy tail",
          ks_to_normal(ot_gaussianize(heavy)) < ks_to_normal(affine_standardize(heavy)))
    # S2 properties.
    contrasts = [rng.gauss(0.0, 1.0) for _ in range(16)]
    w = s2_weights(contrasts, cap=2.0)
    check("S2 mean exactly one", abs(sum(w) / len(w) - 1.0) < 1e-12)
    check("S2 sum exactly L", abs(sum(w) - 16.0) < 1e-9)
    adv_pos, _ = token_advantages(2.0, contrasts)
    adv_neg, _ = token_advantages(-2.0, contrasts)
    adv_scaled, _ = token_advantages(2.0, [100.0 * c for c in contrasts])
    check("S2 total sign-anchored +", abs(sum(adv_pos) - 32.0) < 1e-9)
    check("S2 total sign-anchored -", abs(sum(adv_neg) + 32.0) < 1e-9)
    check("S2 scale-invariant", abs(sum(adv_scaled) - sum(adv_pos)) < 1e-9)
    print("ALL PASS test_training_tace_cpu")


if __name__ == "__main__":
    main()
