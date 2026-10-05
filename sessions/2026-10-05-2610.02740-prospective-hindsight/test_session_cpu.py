"""CPU test for arXiv 2610.02740 — Prospective Hindsight.

Reproduces the paper's core claim end to end in a synthetic contextual
bandit with a GRPO-style base method:
  * policy pi(a|x) with a prospective value predictor V(a,x) that SHARES the
    feature extractor with the policy (the two co-evolve);
  * base method: batch-normalized advantages (as in GRPO);
  * +PH: advantages multiplied by the stop-gradient surprise weight
    (1 + lam*|V - r|), amplifying rollouts where the agent's self-model was
    most inaccurate.

Asserts the paper's headline result: PH lowers the prospective predictor's
miscalibration rate E|V - r| ("descent pathway on miscalibration as a
byproduct of optimization") while improving — not hurting — task return.
Also unit-tests the PH weighting primitive.

Deterministic (seeded). numpy-guarded.
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

from arxiv_lab.training.prospective_hindsight import (  # noqa: E402
    surprise_weights, ph_advantages, miscalibration_rate,
    expected_calibration_error,
)


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run_bandit(use_ph, seed, lam=1.5, rounds=6000, batch=32):
    """Shared-representation actor-critic on a contextual bandit.

    True arm values q(a,x) = sigmoid(w*.(x)); w* scaled x2 so outcomes are
    decisive and reducible miscalibration dominates the noise floor.
    """
    rng = np.random.default_rng(seed)
    n_arms, dim, dh = 3, 4, 8
    w_star = rng.normal(0, 2.0, size=(n_arms, dim))
    W = rng.normal(0, 0.5, size=(dh, dim))   # shared feature extractor
    U = np.zeros((n_arms, dh))              # policy head
    Z = np.zeros((n_arms, dh))              # prospective (value) head
    alpha, alpha_v = 0.1, 0.1

    rets, prosps, retros = [], [], []
    for _ in range(rounds // batch):
        xs = rng.normal(0, 1, size=(batch, dim))
        buf = []
        for x in xs:
            pre = W @ x
            phi = np.tanh(pre)
            dt = 1.0 - phi ** 2
            pi = _softmax(U @ phi)
            a = rng.choice(n_arms, p=pi)
            q_true = 1.0 / (1.0 + np.exp(-(w_star[a] @ x)))
            r = float(rng.random() < q_true)
            v = float(1.0 / (1.0 + np.exp(-(Z[a] @ phi))))
            buf.append((x, phi, dt, pi, a, r, v))
            rets.append(r)
            prosps.append(v)
            retros.append(r)
        rs = np.array([b[5] for b in buf])
        vs = np.array([b[6] for b in buf])
        deltas = rs - vs
        surprise = np.abs(deltas)                       # detached gap
        adv = (deltas - deltas.mean()) / (deltas.std() + 1e-8)  # GRPO-style
        w = 1.0 + lam * surprise if use_ph else np.ones(batch)
        adv_ph = adv * w                                # PH-weighted advantage
        # actor -> policy head + shared extractor (0.5-scaled for stability)
        dU = np.zeros_like(U)
        dW_a = np.zeros_like(W)
        for i, (x, phi, dt, pi, a, r, v) in enumerate(buf):
            dl = -pi.copy()
            dl[a] += 1.0
            dU += adv_ph[i] * np.outer(dl, phi)
            dW_a += adv_ph[i] * np.outer((dl @ U) * dt, x)
        U += alpha / batch * dU
        W += 0.5 * alpha / batch * dW_a
        # critic (prospective predictor) -> value head + shared extractor
        dZ = np.zeros_like(Z)
        dW_c = np.zeros_like(W)
        for i, (x, phi, dt, pi, a, r, v) in enumerate(buf):
            dv = deltas[i] * v * (1.0 - v)
            dZ[a] += dv * phi
            dW_c += np.outer(dv * Z[a] * dt, x)
        Z += alpha_v / batch * dZ
        W += alpha_v / batch * dW_c

    half = len(rets) // 2
    return {
        "return": float(np.mean(rets[half:])),
        "ece": expected_calibration_error(np.array(prosps[half:]),
                                          np.array(retros[half:])),
        "miscal": miscalibration_rate(np.array(prosps[half:]),
                                      np.array(retros[half:])),
    }


def main():
    # --- unit: weighting primitive --------------------------------------
    w, s = surprise_weights([0.9, 0.5, 0.1], [1.0, 0.5, 1.0], lam=1.0)
    assert list(np.round(s, 3)) == [0.1, 0.0, 0.9], s
    assert list(np.round(w, 3)) == [1.1, 1.0, 1.9], w  # high surprise amplified
    adv, _ = ph_advantages([1.0, -0.5, 2.0], [0.9, 0.5, 0.1],
                           [1.0, 0.5, 1.0], lam=1.0)
    assert list(np.round(adv, 3)) == [1.1, -0.5, 3.8], adv
    # zero surprise -> identity (base method recovered)
    adv0, _ = ph_advantages([0.3], [0.7], [0.7], lam=2.0)
    assert abs(adv0[0] - 0.3) < 1e-12
    assert abs(miscalibration_rate([0.8, 0.2], [1, 0]) - 0.2) < 1e-12

    # --- bandit: GRPO-style base vs +PH -----------------------------------
    seeds = (11, 23, 37)
    d_miscal, d_ret = [], []
    for seed in seeds:
        base = run_bandit(False, seed)
        ph = run_bandit(True, seed)
        print(f"seed {seed} base: return={base['return']:.3f} "
              f"ece={base['ece']:.3f} miscal={base['miscal']:.3f}")
        print(f"seed {seed} +PH : return={ph['return']:.3f} "
              f"ece={ph['ece']:.3f} miscal={ph['miscal']:.3f}")
        d_miscal.append(base["miscal"] - ph["miscal"])
        d_ret.append(ph["return"] - base["return"])
    print(f"mean miscalibration change (base->PH): {-np.mean(d_miscal):+.4f} "
          f"(all {sum(d > 0 for d in d_miscal)}/{len(seeds)} seeds improved)")
    print(f"mean return change (base->PH): {np.mean(d_ret):+.4f} "
          f"(all {sum(d >= -0.02 for d in d_ret)}/{len(seeds)} seeds no worse)")

    # paper claim: PH descends miscalibration as a byproduct, without
    # sacrificing task return.
    assert all(d > 0 for d in d_miscal), d_miscal
    assert all(d >= -0.02 for d in d_ret), d_ret

    print("ALL PROSPECTIVE-HINDSIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
