"""CPU tests for the Masked Erosion reproduction (arXiv:2610.38392).

Fast directional checks (3 seeds): masking is exact, exposure under a
single failure is large, freeze/ablation audits accept nothing, larger
batches accept more genuine deletions, and the analytic prediction is
the exact binomial P(Bin(N, p_flip) >= 3).
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.harness.masked_erosion import (
    analytic_acceptance,
    deletion_experiment,
    min_coverage,
    paired_test_accept,
    run_improvement_loop,
)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


def main():
    # Masking: deleting one safeguard alone changes nothing; deleting it
    # while its covering partner fails exposes large harm.
    for env in ("coding", "web"):
        dels = deletion_experiment(20261030, env)
        for i, (masked, exposed) in dels.items():
            check("masked delta ~0 (%s S%d: %.4f)" % (env, i, masked),
                  abs(masked) < 0.005)
            check("exposed delta large (%s S%d: %.4f)" % (env, i, exposed),
                  exposed > 0.02)
    # Full harness covers every channel twice.
    check("full coverage is 2", min_coverage(0b1111) == 2)
    check("one deletion drops min coverage to 1",
          min_coverage(0b1110) == 1)

    # Paired-test edge cases: all-zero diffs never accept; three flips
    # in 20 accept; two flips do not.
    check("zero diffs rejected", not paired_test_accept([0] * 20))
    check("3 flips accept",
          paired_test_accept([1, 1, 1] + [0] * 17))
    check("2 flips rejected",
          not paired_test_accept([1, 1] + [0] * 18))

    # Analytic prediction: exact binomial, monotone in N and friction.
    check("analytic zero friction is nominal size",
          analytic_acceptance(0.0, 20) == 0.05)
    a20 = analytic_acceptance(0.08, 20)
    a80 = analytic_acceptance(0.08, 80)
    check("analytic monotone in N (%.3f < %.3f)" % (a20, a80), a20 < a80)
    check("analytic monotone in friction",
          analytic_acceptance(0.02, 40) < analytic_acceptance(0.08, 40))

    # Gates: freeze and ablation audits preserve depth; larger batches
    # accept more genuine deletions under the paired gate.
    for gate in ("freeze", "ablation_audit"):
        runs = [run_improvement_loop(20261030 + k, gate, 20, "coding", True)
                for k in range(3)]
        check("%s accepts nothing" % gate,
              all(r["accepted"] == 0 for r in runs))
    small = [run_improvement_loop(20261030 + k, "paired", 10, "coding", True)
             for k in range(3)]
    large = [run_improvement_loop(20261030 + k, "paired", 80, "coding", True)
             for k in range(3)]
    check("batch 80 accepts more than batch 10",
          sum(r["accepted"] for r in large) > sum(r["accepted"] for r in small))
    check("batch 80 loses more depth",
          sum(r["depth_lost"] for r in large) >= sum(r["depth_lost"] for r in small))
    print("ALL CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
