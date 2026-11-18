#!/usr/bin/env python3
"""CPU/numerical sanity check of the Ambient SFT loss (arXiv:2611.23229).

Loss (task spec):  L = L_SFT + lam * T^2 * KL( softmax(z_teacher / T)
                                              || softmax(z_student / T) )
Teacher = frozen base model; student = fine-tuned model. Checked on toy
logits with numpy only. Two limits the paper's abstract claims:

  * T -> 1 : reduces to standard knowledge distillation (T^2 * KL = KL).
  * T -> inf: reverts to regular SFT.

The T -> inf claim needs care: the UNSCALED KL term -> 0 (both tempered
distributions -> uniform), but the T^2-SCALED term used in practice tends
to a NONZERO logit-matching constant (Hinton et al. 2015: grad of T^2*KL
-> (z_s - z_t)/N). So under the T^2-scaled form the T -> inf limit is
logit matching, not plain SFT; the "reverts to SFT" reading holds for the
unscaled KL term (or, equivalently, for lam -> 0). This check measures
both quantities so the distinction is on the record, not assumed.
"""
import numpy as np

rng = np.random.default_rng(0)
VOCAB = 32000  # Qwen-scale vocab, sampled sparsely below for speed


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def kl_tempered(z_t, z_s, T):
    p = softmax(z_t / T)
    q = softmax(z_s / T)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def t2_kl_limit_constant(z_t, z_s):
    """Hinton T->inf limit of T^2 * KL: (1/2N) * sum((dz_t - dz_s)^2)
    where dz are mean-centred logits (softmax is shift-invariant)."""
    dt = z_t - z_t.mean()
    ds = z_s - z_s.mean()
    return float(0.5 * np.mean((dt - ds) ** 2))


def main():
    # Toy logits: teacher ~ N(0, 4), student = teacher + small noise
    # (a partially-trained student close to its frozen teacher).
    z_t = rng.normal(0, 2.0, size=512)
    z_s = z_t + rng.normal(0, 0.5, size=512)

    print("=== Ambient SFT loss limits (toy logits, n=512) ===")
    # 1. T = 1: T^2 * KL == KL (standard KD term)
    kl1 = kl_tempered(z_t, z_s, 1.0)
    print(f"T=1   : KL={kl1:.6f}  T^2*KL={1.0**2 * kl1:.6f}  (identical: standard KD)")

    # 2. T sweep: unscaled KL -> 0, T^2*KL -> Hinton constant
    const = t2_kl_limit_constant(z_t, z_s)
    print(f"Hinton T->inf limit of T^2*KL (logit-MSE const): {const:.6f}")
    for T in (2.0, 4.0, 8.0, 16.0, 100.0, 1000.0):
        kl = kl_tempered(z_t, z_s, T)
        print(f"T={T:>7.0f}: KL={kl:.8f}  T^2*KL={T*T*kl:.6f}"
              f"  (ratio to limit: {T*T*kl/const:.4f})")

    # 3. Assertions
    assert abs(kl_tempered(z_t, z_s, 1.0) - kl1) < 1e-12
    kl_1000 = kl_tempered(z_t, z_s, 1000.0)
    assert kl_1000 < 1e-4, f"unscaled KL at T=1000 should be ~0, got {kl_1000}"
    t2_1000 = 1000.0 ** 2 * kl_1000
    assert abs(t2_1000 / const - 1.0) < 0.02, (
        f"T^2*KL at T=1000 should converge to Hinton const, got {t2_1000} vs {const}")
    # 4. Identical teacher/student -> KL = 0 at every T
    assert kl_tempered(z_t, z_t, 4.0) == 0.0
    # 5. KL is non-negative and grows as student diverges
    z_far = z_t + rng.normal(0, 4.0, size=512)
    assert kl_tempered(z_t, z_far, 2.0) > kl_tempered(z_t, z_s, 2.0)
    print("ALL ASSERTIONS PASSED")
    print()
    print("CONCLUSION: T=1 is exactly standard KD. Unscaled KL -> 0 as T -> inf,")
    print("but the T^2-scaled term used in the loss converges to a nonzero")
    print(f"logit-matching constant ({const:.4f} here), NOT to 0. The abstract's")
    print("'T -> inf reverts to SFT' holds for the unscaled KL term; with the")
    print("T^2 factor the limit is logit matching. Colab driver therefore")
    print("treats T as trust-in-teacher (small T = strong KD, large T = weak")
    print("regularisation toward the frozen base) and reports per-arm results")
    print("rather than assuming the T -> inf limit equals plain SFT.")


if __name__ == "__main__":
    main()
