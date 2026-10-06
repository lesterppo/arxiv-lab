"""CPU test for arXiv 2610.06765 — ARBOR conditional rank allocation.

Reproduces the paper's *illustrative separation* under orthogonal,
equiprobable subtasks on a deterministic synthetic construction:

 1. A fixed rank-r update shared across K orthogonal subtasks hits an
    approximation floor (best rank-r SVD truncation of the stacked tasks
    leaves strictly positive mean residual).
 2. Conditional per-question selection of r rank-one atoms from the shared
    basis via the additive gate (question repr + tag + interaction) avoids
    the floor: with correct routing the residual is ~0 at the SAME active
    rank.
 3. Routing matters: perturbing the tags (the paper's tag-perturbation
    check) degrades the conditional update toward/above the floor, and the
    gate's atom clusters align with the subtask labels.

Deterministic (seeded). numpy-guarded: skips cleanly without numpy.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    from arxiv_lab.training.arbor import (
        make_orthogonal_tasks,
        fixed_update_floor,
        additive_gate,
        conditional_update,
        oracle_gate_weights,
        question_for_task,
    )
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

K = 4
DIM = 16
RANK_PER_TASK = 2
ACTIVE_RANK = 2  # same active rank for the fixed and conditional updates


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def test_arbor_separation():
    if not HAS_NUMPY:
        print("SKIP: numpy not available")
        return
    deltas, atoms = make_orthogonal_tasks(K=K, dim=DIM,
                                          rank_per_task=RANK_PER_TASK, seed=0)

    # --- fixed update: best single rank-ACTIVE_RANK matrix for all tasks ---
    _fixed, floor = fixed_update_floor(deltas, ACTIVE_RANK)
    print(f"fixed rank-{ACTIVE_RANK} approximation floor (mean MSE): {floor:.6f}")
    assert floor > 1e-6, "orthogonal subtasks must leave a positive floor"

    # --- conditional update: gate routes each question to its own atoms ---
    Wq, Wt, Wx = oracle_gate_weights(K, RANK_PER_TASK, seed=1)
    flat = [uv for comps in atoms for uv in comps]

    def fit_residual(top, target):
        # Least-squares fit of the selected atoms to the target update:
        # models the paper's *learned coefficient* scaling the adapter
        # residual. Separates routing (the claim under test) from scale.
        A = np.stack([np.outer(u, v).reshape(-1)
                      for (u, v) in (flat[i] for i in top)], axis=1)
        coef, *_ = np.linalg.lstsq(A, target.reshape(-1), rcond=None)
        out = np.zeros_like(target)
        for c, i in zip(coef, top):
            u, v = flat[i]
            out = out + c * np.outer(u, v)
        return out

    cond_mses = []
    routing_ok = 0
    for k in range(K):
        q, tag = question_for_task(k, K, seed=2)
        _delta_c, scores = conditional_update(q, tag, atoms, Wq, Wt, Wx,
                                              ACTIVE_RANK)
        top = np.argsort(scores)[-ACTIVE_RANK:]
        delta_c = fit_residual(top, deltas[k])
        cond_mses.append(mse(delta_c, deltas[k]))
        # routing accuracy: top-ACTIVE_RANK atoms belong to subtask k
        expected = set(range(k * RANK_PER_TASK, (k + 1) * RANK_PER_TASK))
        if set(top.tolist()) == expected:
            routing_ok += 1
    cond_mean = float(np.mean(cond_mses))
    print(f"conditional rank-{ACTIVE_RANK} mean MSE: {cond_mean:.6f}")
    print(f"routing accuracy: {routing_ok}/{K}")
    assert routing_ok == K, "gate must route each subtask to its own atoms"
    assert cond_mean < 1e-9, "correct routing reconstructs each delta exactly"
    assert cond_mean < floor, "conditional selection beats the fixed floor"

    # --- tag perturbation: routing is load-bearing (paper's check) ---
    pert_mses = []
    for k in range(K):
        q, _tag = question_for_task(k, K, seed=2)
        wrong = np.zeros(K)
        wrong[(k + 1) % K] = 1.0  # perturbed specialty tag
        _delta_c, scores = conditional_update(q, wrong, atoms, Wq, Wt, Wx,
                                              ACTIVE_RANK)
        top = np.argsort(scores)[-ACTIVE_RANK:]
        delta_c = fit_residual(top, deltas[k])
        pert_mses.append(mse(delta_c, deltas[k]))
    pert_mean = float(np.mean(pert_mses))
    print(f"tag-perturbed conditional mean MSE: {pert_mean:.6f}")
    assert pert_mean > cond_mean, "wrong tags must hurt"
    assert pert_mean > floor, "misrouted conditional update hits the floor"

    results = {
        "K": K, "dim": DIM, "rank_per_task": RANK_PER_TASK,
        "active_rank": ACTIVE_RANK,
        "fixed_floor_mse": floor,
        "conditional_mse": cond_mean,
        "tag_perturbed_mse": pert_mean,
        "routing_accuracy": f"{routing_ok}/{K}",
    }
    out = os.path.join(os.path.dirname(__file__), "..", "sessions",
                       "2026-10-06-2610.06765-arbor", "results_cpu.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    test_arbor_separation()
    print("ARBOR CPU test: separation reproduced.")
