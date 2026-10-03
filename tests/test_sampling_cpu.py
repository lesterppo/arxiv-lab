"""CPU tests for arxiv_lab.sampling (FERPO, arXiv:2610.02198).

Ports experiments/2026-10-03b/test_ferpo.py.

Claim A — KL regularization keeps SNIS weights well behaved: the effective
sample size of the self-normalized importance weights must rise monotonically
as the target stays closer to the rollout policy (higher temperature alpha).

Claim B — forward-KL encourages coverage of multiple high-value modes while
reverse-KL favors a subset: with actor capacity K=2 below the 3 target modes,
the forward-KL fit must cover strictly more modes than the reverse-KL fit.

Seeded and deterministic. Requires numpy (optional dependency); skipped
gracefully without it.

Run:  python3 tests/test_sampling_cpu.py
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from arxiv_lab.sampling import (snis_weights, forward_kl_fit, reverse_kl_fit,
                                mode_coverage, expected_log_target)


def test_kl_limit_keeps_weights_well_behaved():
    rng = np.random.default_rng(42)
    alphas = [0.25, 0.5, 1.0, 2.0]
    ess = []
    for alpha in alphas:
        _, _, e = snis_weights(rng, 200000, alpha=alpha)
        ess.append(e)
        print('  alpha=%.2f -> ESS=%.1f%%' % (alpha, 100 * e))
    assert all(b > a for a, b in zip(ess, ess[1:])), \
        'ESS must rise monotonically with alpha (tighter KL limit -> better weights)'
    assert ess[-1] > 2 * ess[0], 'effect should be substantial, not marginal'


def test_forward_kl_covers_modes():
    rng = np.random.default_rng(42)
    a, w, ess = snis_weights(rng, 200000, alpha=1.0)
    print('  SNIS ESS=%.1f%%' % (100 * ess,))
    mu_f, sg_f, pi_f = forward_kl_fit(a, w, n_comp=2)
    mu_r, sg_r, pi_r = reverse_kl_fit(np.random.default_rng(43), n_comp=2,
                                      init_mu=np.array([-2.5, -3.5]))
    cov_f = mode_coverage(mu_f, pi_f)
    cov_r = mode_coverage(mu_r, pi_r)
    print('  forward-KL fit: mu=%s sig=%s pi=%s -> covers %d/3 modes' %
          (np.round(mu_f, 2), np.round(sg_f, 2), np.round(pi_f, 2), cov_f))
    print('  reverse-KL fit: mu=%s sig=%s pi=%s -> covers %d/3 modes' %
          (np.round(mu_r, 2), np.round(sg_r, 2), np.round(pi_r, 2), cov_r))
    print('  E[log target]: forward=%.3f reverse=%.3f' %
          (expected_log_target(np.random.default_rng(44), mu_f, sg_f, pi_f),
           expected_log_target(np.random.default_rng(45), mu_r, sg_r, pi_r)))
    assert cov_f >= 2, 'forward-KL should cover at least 2 of 3 modes'
    assert cov_f > cov_r, 'forward-KL must cover strictly more modes than reverse-KL'


def main():
    if not HAS_NUMPY:
        print("SKIPPED: numpy not installed (arxiv_lab.sampling is numpy-optional)")
        return
    print('claim A: KL limit keeps SNIS weights well behaved')
    test_kl_limit_keeps_weights_well_behaved()
    print('claim B: forward-KL mode coverage vs reverse-KL mode seeking')
    test_forward_kl_covers_modes()
    print('ALL SAMPLING CHECKS PASSED')


if __name__ == '__main__':
    main()
