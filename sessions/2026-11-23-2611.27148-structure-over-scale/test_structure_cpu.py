#!/usr/bin/env python3
"""Deterministic CPU test for the Structure Over Scale Colab driver
(arXiv:2611.27148).

Extracts + execs the driver's PURE HELPERS block (stdlib only, no
torch) and checks:

* E0 answer extraction / normalisation / equality (boxed, Answer:
  line, frac, tolerance).
* E1 group-relative advantages (GRPO): zero-mean within group, zero
  when the group has no variance.
* E2 spectral stats: effective rank / top-1 share on known spectra
  (single direction -> eff rank 1, top1 1; flat spectrum of k ->
  eff rank k, top1 1/k; scale invariance).
* E3 the r x r singular-value trick used by update_spectrum: with
  A = U_A diag(S_A) Vh_A and B = U_B diag(S_B) Vh_B, the nonzero
  singular values of BA equal those of diag(S_B) Vh_B U_A diag(S_A)
  -- verified against a direct SVD of B@A on random matrices
  (numpy; the driver uses the identical torch calls).
* E4 determinism: repeated extraction gives identical results.

Run: python3 test_structure_cpu.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "colab_structure_over_scale.py")

src = open(DRIVER).read()
start = src.index("# ---------------- PURE HELPERS (stdlib) ----------------")
end = src.index("# ---------------- END PURE HELPERS ----------------")
pure = src[start:end]
ns = {}
exec(pure, ns)

extract_boxed = ns["extract_boxed"]
extract_answer = ns["extract_answer"]
answers_equal = ns["answers_equal"]
spectral_stats = ns["spectral_stats"]
group_advantages = ns["group_advantages"]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def svd_trick(A, B):
    """Mirror of the driver's update_spectrum core (numpy version)."""
    U_A, S_A, _ = np.linalg.svd(A, full_matrices=False)
    U_B, S_B, Vh_B = np.linalg.svd(B, full_matrices=False)
    C = np.diag(S_B) @ Vh_B @ U_A @ np.diag(S_A)
    return np.linalg.svd(C, compute_uv=False)


def main():
    print("E0 answer helpers")
    check(extract_boxed("x \\boxed{42} y") == "42", "boxed extraction")
    check(extract_boxed("a \\boxed{\\frac{1}{2}} b") == "\\frac{1}{2}",
          "boxed with nested braces")
    check(extract_boxed("no box") is None, "no boxed -> None")
    check(extract_boxed("first \\boxed{1} second \\boxed{2}") == "2",
          "LAST boxed wins")
    check(extract_answer("work...\nAnswer: 42") == "42",
          "Answer: line extraction")
    check(extract_answer("work \\boxed{7}") == "7", "boxed beats Answer:")
    check(extract_answer("") is None, "empty -> None")
    check(answers_equal("42", "42"), "exact match")
    check(answers_equal("1/2", "\\frac{1}{2}"), "frac normalisation")
    check(answers_equal("0.5", "\\frac{1}{2}"), "frac vs decimal")
    check(answers_equal("$1,000", "1000"), "currency/comma strip")
    check(not answers_equal("42", "43"), "mismatch detected")
    check(not answers_equal(None, "42"), "None pred fails")

    print("E1 group advantages")
    adv = group_advantages([1, 0, 0, 0], 4)
    check(abs(sum(adv)) < 1e-9, "advantages zero-mean within group")
    check(adv[0] > 0 and all(a < 0 for a in adv[1:]),
          "solved sample gets positive advantage")
    adv2 = group_advantages([1, 1, 1, 1], 4)
    check(all(a == 0.0 for a in adv2), "no variance -> zero advantage")
    adv3 = group_advantages([1, 0, 0, 0, 0, 0, 1, 1], 4)
    check(len(adv3) == 8 and abs(sum(adv3[:4])) < 1e-9
          and abs(sum(adv3[4:])) < 1e-9,
          "two groups handled independently")

    print("E2 spectral stats")
    er, t1 = spectral_stats([5.0])
    check(er == 1.0 and t1 == 1.0, "single direction: eff rank 1, top1 1")
    er, t1 = spectral_stats([1.0] * 8)
    check(abs(er - 8.0) < 1e-9 and abs(t1 - 0.125) < 1e-9,
          "flat spectrum of 8: eff rank 8, top1 1/8")
    er, t1 = spectral_stats([10.0, 1.0, 1.0, 1.0])
    check(er < 4.0 and t1 > 0.9, "concentrated spectrum: low eff rank")
    er1, _ = spectral_stats([2.0, 1.0])
    er2, _ = spectral_stats([200.0, 100.0])
    check(abs(er1 - er2) < 1e-9, "scale invariance")
    er, t1 = spectral_stats([])
    check(er == 0.0 and t1 == 0.0, "empty spectrum -> zeros")

    print("E3 r x r singular-value trick vs direct SVD")
    rng = np.random.default_rng(0)
    for trial in range(20):
        r = int(rng.integers(2, 9))
        d_in = int(rng.integers(10, 40))
        d_out = int(rng.integers(10, 40))
        A = rng.normal(size=(r, d_in))
        B = rng.normal(size=(d_out, r))
        direct = np.linalg.svd(B @ A, compute_uv=False)
        trick = svd_trick(A, B)
        check(np.allclose(np.sort(direct)[::-1][:r],
                          np.sort(trick)[::-1], atol=1e-8),
              "trial %d: trick singular values match direct SVD "
              "(r=%d)" % (trial, r))
        # Concentrated vs distributed: a rank-1 update must read as
        # eff rank ~1 however large r is.
        B1 = rng.normal(size=(d_out, 1)) @ rng.normal(size=(1, r))
        A1 = rng.normal(size=(r, d_in))
        s1 = svd_trick(A1, B1)
        er1, t1_1 = spectral_stats(s1.tolist())
        check(er1 < 1.5 and t1_1 > 0.9,
              "rank-1 update reads concentrated (eff rank %.2f)" % er1)

    print("E4 determinism")
    check(group_advantages([1, 0, 0, 0], 4)
          == group_advantages([1, 0, 0, 0], 4),
          "advantages deterministic")
    check(spectral_stats([3.0, 1.0]) == spectral_stats([3.0, 1.0]),
          "spectral stats deterministic")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
