"""CPU tests for arxiv_lab.training.rpo (arXiv:2610.17767).

Validates the group-relative advantage rule, the root-token loss pieces,
and the paper's signature claim in a seeded toy world: with thinking
useful only on hard items and costly everywhere, group rollouts with
forced THINK/DIRECT arms must produce difficulty-aware adaptive
thinking (think-rate hard >> easy) with accuracy preserved.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.training.rpo import (
    group_advantages,
    kl_penalty,
    root_token_losses,
    simulate_adaptive_thinking,
    THINK_COST,
)

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


def need_numpy():
    if not HAS_NUMPY:
        print("SKIP (numpy missing)")
        sys.exit(0)


def test_group_advantages_zero_sum():
    need_numpy()
    adv = group_advantages([1.0, 0.85, 0.0, -0.15])
    assert abs(adv.sum()) < 1e-9
    # group-relative: identical rewards -> zero advantages -> no update
    adv2 = group_advantages([0.5, 0.5, 0.5, 0.5])
    assert (abs(adv2) < 1e-12).all()


def test_group_advantages_ordering():
    need_numpy()
    adv = group_advantages([1.0, 0.85, 1.0, 0.85])
    # above-mean arms positive, below-mean negative
    assert (adv[:1] > 0).all() and (adv[1:2] < 0).all()


def test_kl_penalty_nonnegative():
    need_numpy()
    kl = kl_penalty([-0.7, -1.2], [-0.69, -1.2])
    assert (kl >= -1e-12).all()
    kl0 = kl_penalty([-0.7], [-0.7])
    assert abs(kl0[0]) < 1e-12


def test_root_token_loss_sign():
    need_numpy()
    # Positive advantage on the sampled root token -> loss decreases as
    # its log-prob rises (policy gradient direction).
    l1, _, t1 = root_token_losses([-0.5], [1.0])
    l2, _, t2 = root_token_losses([-0.1], [1.0])
    assert l2 < l1
    # KL term only active with a reference + beta > 0
    _, kl0, _ = root_token_losses([-0.5], [1.0])
    assert kl0 == 0.0
    _, kl1, t3 = root_token_losses([-0.5], [1.0], logp_ref=[-0.4], beta=1.0)
    assert kl1 > 0.0 and t3 != l1


def test_adaptive_gap_emerges():
    res = simulate_adaptive_thinking(n_steps=4000, lr=0.5, seed=7)
    assert res["think_rate_hard"] > 0.6, res
    assert res["think_rate_easy"] < 0.4, res
    assert res["adaptive_gap"] > 0.3, res
    # accuracy preserved: mixture of the four arms' correctness rates
    # (0.95/0.97 easy, 0.25/0.80 hard) averages ~0.74
    assert res["mean_accuracy"] > 0.70, res


def test_simulation_deterministic():
    a = simulate_adaptive_thinking(n_steps=500, seed=3)
    b = simulate_adaptive_thinking(n_steps=500, seed=3)
    assert a == b


def test_think_cost_documented():
    assert THINK_COST == 0.15


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL RPO CPU TESTS PASS")
