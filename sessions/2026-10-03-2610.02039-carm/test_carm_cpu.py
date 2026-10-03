"""
CARM (arXiv:2610.02039) — CPU mechanism test on synthetic log-ratios.

1. Cancellation demo: sequences with opposing +-a drifts are ACCEPTED by
   the standard geometric-mean mask for arbitrarily large a, but REJECTED
   by CARM once a exceeds the band.
2. No false positives: clean (small-noise) sequences are accepted by both.
3. Joint bound (qualitative form of the paper's theorem): among accepted
   sequences, CARM bounds the max per-token |log ratio|; the standard mask
   does not.
Run: python3 test_carm_cpu.py
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carm import standard_mask, carm_mask, drift_stats

EPS = 0.2
BAND = math.log1p(EPS)


def main():
    # 1. cancellation: half +a, half -a
    print("== cancellation: +-a drifts ==")
    for a in (0.1, 0.25, 0.5, 1.0, 2.0):
        seq = [a] * 32 + [-a] * 32
        s, c = standard_mask(seq, EPS), carm_mask(seq, EPS)
        print(f"  a={a:4.2f}: standard={'KEEP' if s else 'drop'}  "
              f"CARM={'KEEP' if c else 'drop'}")
        if a > BAND + 1e-9:
            assert s and not c, \
                "standard must accept canceling drift; CARM must reject it"
    print("  -> standard mask accepts arbitrarily large canceling drift; "
          "CARM rejects it. Flaw reproduced, fix verified.")

    # 2. no false positives on clean sequences
    print("== clean sequences (small noise) ==")
    import random
    rng = random.Random(0)
    for trial in range(20):
        seq = [rng.gauss(0, 0.05) for _ in range(64)]
        assert standard_mask(seq, EPS) and carm_mask(seq, EPS), \
            "both masks must accept clean sequences"
    print("  20/20 clean sequences accepted by both masks.")

    # 3. joint bound (the paper's theorem, qualitative form): for CARM-accepted
    # sequences, (frac outside band) * (mean excess beyond band) <= band.
    # Proof sketch: sum of excesses <= sum |x_i| <= n*band for accepted seqs.
    # The standard mask admits sequences violating this arbitrarily badly.
    print("== joint bound: frac_outside * mean_excess <=")
    worst_carm_joint = 0.0
    worst_std_joint = 0.0
    n_std = n_carm = 0
    for ai in range(0, 41):
        for bi in range(0, 41):
            a, b = ai * 0.1, bi * 0.1
            seq = [a] * 16 + [-b] * 16 + [0.0] * 32
            st = drift_stats(seq, EPS)
            joint = st["frac_outside"] * st["mean_excess"]
            if standard_mask(seq, EPS):
                n_std += 1
                worst_std_joint = max(worst_std_joint, joint)
            if carm_mask(seq, EPS):
                n_carm += 1
                worst_carm_joint = max(worst_carm_joint, joint)
    print(f"  standard: accepted {n_std}, worst joint = {worst_std_joint:.3f} "
          f"(band={BAND:.3f})")
    print(f"  CARM:     accepted {n_carm}, worst joint = {worst_carm_joint:.3f} "
          f"(band={BAND:.3f})")
    assert worst_carm_joint <= BAND + 1e-9, \
        "CARM-accepted sequences must satisfy the joint bound"
    assert worst_std_joint > 5 * BAND, \
        "standard mask must admit sequences badly violating the joint bound"
    print("  -> CARM-accepted sequences satisfy frac_outside*mean_excess <= band; "
          "standard mask admits violations "
          f"{worst_std_joint / BAND:.0f}x worse. Paper's joint-bound direction confirmed.")

    print("\nALL CARM CHECKS PASSED.")


if __name__ == "__main__":
    main()
