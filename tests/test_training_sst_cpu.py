"""CPU tests for arxiv_lab.training.sst (Subspace Tuning, arXiv:2610.22293).

Claim under test (bounded, simulated scale): restricting each weight update
to the *bottom* singular subspace of the weight matrix (Subspace Tuning)
substantially improves retention of a previously learned task during
sequential fine-tuning -- the paper's learning-forgetting Pareto claim
(protecting up to 99% of the subspace; optimizer-agnostic; composes with
SFT).

Setup: a 2-layer MLP (20 -> 64 -> 2) trained sequentially on two synthetic
binary tasks sharing the input space (independent random teachers, so
task B interferes with task A). Protocol: train task A with plain SGD to
convergence, then train task B with (a) plain SGD and (b) SST
(protect_frac=0.9, SVD refreshed every step at this scale). Metrics:
retention = task-A accuracy after task-B training; learning = task-B
accuracy after task-B training.

RESULT: the claim reproduces directionally. Plain SGD: retention 0.630,
learning 1.000. SST(0.9): retention 0.868 (+0.238), learning 0.750.
Pareto sweep (strong two-sided reading): pf=0.5 -> 0.850/0.835;
pf=0.7 -> 0.863/0.790; pf=0.99 -> 0.873/0.720. The toy shows a real
retention/learning trade-off -- at this scale the free bottom subspace is
tiny (~1% of blocks), so target-task learning costs 0.17-0.28; the paper's
"preserved" target performance is the large-scale regime where the bottom
subspace is still enormous. Honest limits: synthetic tasks share no
general-capability structure; the scale-appropriate 9B LoRA test is staged
for Colab (sessions/2026-10-16-2610.22293-sst/colab_sst_9b.py).

Fidelity note: an earlier draft of the module removed only the top-top
block ``U_r (U_rT G V_r) V_rT``; that reading showed NO retention effect
(flat 0.623-0.637 across protect_frac) -- the top-bottom/bottom-top blocks
carry enough interference to destroy retention. The two-sided projection
``(I - U_r U_rT) G (I - V_r V_rT)`` used here is the faithful reading of
"eliminating changes along the principal directions", and it is what
reproduces the effect.

Seeded and deterministic. Requires numpy (optional dependency); skipped
gracefully without it.

Run:  python3 tests/test_training_sst_cpu.py
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


def need_numpy():
    if not HAS_NUMPY:
        print("SKIP: numpy not installed")
        sys.exit(0)


from arxiv_lab.training.sst import (
    principal_directions,
    project_grad_to_bottom_subspace,
    sst_update,
)

SEED = 20261016
D_IN, D_HID, D_OUT = 20, 64, 2
N = 400
LR = 0.1
STEPS_A = 300
STEPS_B = 300
PROTECT_FRAC = 0.9


def make_task(rng, n=N, d=D_IN):
    x = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (x @ w > 0).astype(int)
    return x, y


def init_params(rng):
    return [rng.normal(scale=np.sqrt(2.0 / D_IN), size=(D_IN, D_HID)),
            np.zeros(D_HID),
            rng.normal(scale=np.sqrt(2.0 / D_HID), size=(D_HID, D_OUT)),
            np.zeros(D_OUT)]


def forward(params, x):
    w1, b1, w2, b2 = params
    h = np.maximum(0.0, x @ w1 + b1)
    logits = h @ w2 + b2
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return h, e / e.sum(axis=1, keepdims=True)


def grads_and_acc(params, x, y):
    w1, b1, w2, b2 = params
    h, p = forward(params, x)
    n = len(x)
    acc = float((p.argmax(axis=1) == y).mean())
    dlogits = p.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    dw2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ w2.T
    dh[h <= 0.0] = 0.0
    dw1 = x.T @ dh
    db1 = dh.sum(axis=0)
    return [dw1, db1, dw2, db2], acc


def train(params, x, y, steps, lr, protect_frac=None):
    for _ in range(steps):
        grads, _acc = grads_and_acc(params, x, y)
        if protect_frac is None:
            params = [p - lr * g for p, g in zip(params, grads)]
        else:
            params = sst_update(params, grads, lr, protect_frac)
    return params


def run_sequential(seed=SEED):
    rng = np.random.default_rng(seed)
    xa, ya = make_task(rng)
    xb, yb = make_task(rng)

    base = train(init_params(rng), xa, ya, STEPS_A, LR)
    _, acc_a_after_a = grads_and_acc(base, xa, ya)

    plain = train([p.copy() for p in base], xb, yb, STEPS_B, LR,
                  protect_frac=None)
    _, plain_retention = grads_and_acc(plain, xa, ya)
    _, plain_learning = grads_and_acc(plain, xb, yb)

    sst = train([p.copy() for p in base], xb, yb, STEPS_B, LR,
                protect_frac=PROTECT_FRAC)
    _, sst_retention = grads_and_acc(sst, xa, ya)
    _, sst_learning = grads_and_acc(sst, xb, yb)

    return {
        "acc_a_after_a": acc_a_after_a,
        "plain_retention": plain_retention,
        "plain_learning": plain_learning,
        "sst_retention": sst_retention,
        "sst_learning": sst_learning,
    }


def test_projection_removes_principal_components():
    need_numpy()
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=(24, 16))
    g = rng.normal(size=(24, 16))
    gp = project_grad_to_bottom_subspace(g, w, 0.9)
    u_r, v_r = principal_directions(w, 0.9)
    # No update component may remain along the protected directions,
    # on either side.
    assert np.abs(u_r.T @ gp).max() < 1e-8, "left leakage into top subspace"
    assert np.abs(gp @ v_r).max() < 1e-8, "right leakage into top subspace"
    # ...but the projected gradient is not zero: learning can proceed.
    assert np.linalg.norm(gp) > 0.01 * np.linalg.norm(g)


def test_sst_improves_retention_with_learning_tradeoff():
    """Pin the reproduced Pareto effect (faithful two-sided reading)."""
    need_numpy()
    res = run_sequential()
    print(f"task A after task-A training : {res['acc_a_after_a']:.3f}")
    print(f"plain SGD: retention {res['plain_retention']:.3f}, "
          f"learning {res['plain_learning']:.3f}")
    print(f"SST(0.9):  retention {res['sst_retention']:.3f}, "
          f"learning {res['sst_learning']:.3f}")
    assert res["acc_a_after_a"] > 0.95, "task A must be learned first"
    assert res["plain_retention"] < 0.90, (
        "sanity: plain SGD must actually forget (else the test is vacuous)")
    # The paper's retention claim, reproduced directionally at toy scale.
    assert res["sst_retention"] >= res["plain_retention"] + 0.15, (
        "SST must substantially improve retention")
    # Honest limit: at this scale the free bottom subspace is tiny, so
    # target-task learning pays a visible cost (paper's "preserved"
    # performance is the large-scale regime).
    assert res["sst_learning"] >= 0.70, (
        "SST must still learn the target task reasonably")


def main():
    test_projection_removes_principal_components()
    test_sst_improves_retention_with_learning_tradeoff()
    print("ALL SST CPU TESTS PASSED")


if __name__ == "__main__":
    main()
