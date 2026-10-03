"""CPU tests for arxiv_lab.memory (CMP, arXiv:2610.02070).

Ports experiments/2026-10-03b/test_cmp.py. Paper claims under test:
  1. Identification fails for never-retrieved memories under store-level
     intervention (retrieval-level positivity violation).
  2. Intervening on retrieval with known propensities + SNIPW restores
     identification (near-unbiased utility for those memories).
  3. Discrimination between required and non-required memories improves.

Seeded and deterministic. Requires numpy (optional dependency); skipped
gracefully without it.

Run:  python3 tests/test_memory_cpu.py
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

from arxiv_lab.memory import (
    MemoryStore, naive_store_intervention, cmp_balanced_design, auc)


def test_cmp():
    rng = np.random.default_rng(7)
    store = MemoryStore(rng)
    n_supp = len(store.suppressed)
    print('suppressed required memories: %d of %d (%.0f%%)' %
          (n_supp, len(store.required), 100 * n_supp / len(store.required)))

    naive = naive_store_intervention(store)
    ate, valid = cmp_balanced_design(store, np.random.default_rng(11))
    cmp_agg = np.nanmean(np.where(valid, ate, np.nan), axis=1)

    labels = np.zeros(store.n_mem); labels[store.required] = 1
    auc_naive = auc(naive, labels)
    auc_cmp = auc(np.nan_to_num(cmp_agg, nan=-1.0), labels)
    print('AUC required vs non-required: naive=%.3f CMP=%.3f' % (auc_naive, auc_cmp))

    supp = store.suppressed
    retr_req = np.array([m for m in store.required if m not in set(supp.tolist())])
    bias_naive = float(abs(naive[supp].mean() - store.true_u[supp].mean()))
    bias_cmp = float(abs(np.nanmean(ate[supp][valid[supp]]) - store.true_u[supp].mean()))
    print('never-retrieved required: true_u=%.3f naive=%.3f cmp=%.3f' % (
        store.true_u[supp].mean(), naive[supp].mean(),
        np.nanmean(ate[supp][valid[supp]])))
    print('other required:           true_u=%.3f naive=%.3f' % (
        store.true_u[retr_req].mean(), naive[retr_req].mean()))
    print('non-required:             true_u=%.3f naive=%.3f cmp=%.3f' % (
        store.true_u[20:].mean(), naive[20:].mean(),
        np.nanmean(ate[20:][valid[20:]])))

    # claim 1: positivity violation — naive attributes exactly 0 utility
    assert np.all(naive[supp] == 0.0), 'never-retrieved memories must score 0 under naive'
    assert bias_naive > 0.05, 'naive must be badly biased for suppressed memories'
    # claim 2: CMP restores identification (near-unbiased)
    assert bias_cmp < 0.05, 'CMP should be near-unbiased for suppressed memories'
    # claim 3: discrimination improves
    assert auc_cmp > auc_naive + 0.15, 'CMP must improve required/non-required AUC'
    assert auc_cmp > 0.65
    print('ALL CMP CHECKS PASSED')


def main():
    if not HAS_NUMPY:
        print("SKIPPED: numpy not installed (arxiv_lab.memory is numpy-optional)")
        return
    test_cmp()
    print('ALL MEMORY CHECKS PASSED')


if __name__ == '__main__':
    main()
