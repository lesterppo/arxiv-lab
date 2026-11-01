"""Perturb-Select-Vote (PSV): training-free inference-time ensembles.

Paper: arXiv 2610.40649 "Rethinking Neural Thickets Without Weight
Perturbations" (Dufour et al., submitted 29 Oct 2026).

One-line claim: the gains attributed to random weight perturbations of a
pretrained model do not require weight-space geometry — keeping the model
entirely frozen and adding a random bias to its output logits matches or
exceeds full-weight perturbations on most reasoning tasks, once perturbed
predictors are selected on a small validation set and aggregated by voting
(Perturb-Select-Vote). Small weight perturbations act primarily through
their induced logit shifts, so a direct logit bias realizes the same
benefit without touching any weight.

This module implements PSV as a deterministic CPU simulation (see the
2026-11-01 session for the measured reproduction): a frozen softmax
classifier trained on class-prior-skewed data (so its errors are a
systematic prior bias, the kind a constant logit shift can correct),
two perturbation arms — full-weight (matrix + bias, sigma 0.3) and
logit-only (bias vector only, sigma 0.5) — each with K=32 members,
validation selection of the top M=8 members, and majority voting.
numpy-guarded per repo conventions.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

N_CLASSES = 4
N_FEATURES = 12
K_MEMBERS = 32
TOP_M = 8
SIGMA_WEIGHT = 0.3
SIGMA_LOGIT = 0.5
# (variant, train priors, signal strength); validation/test are balanced.
VARIANTS = (
    ("A", (0.55, 0.15, 0.15, 0.15), 2.0),
    ("B", (0.50, 0.20, 0.15, 0.15), 1.6),
    ("C", (0.40, 0.30, 0.15, 0.15), 2.0),
)
SEEDS = (1, 7, 42)


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for psv")


def generate_prior_shift_data(rng, priors, signal, n):
    """Balanced-or-skewed multiclass data: class signal in one feature.

    Each example's class dimension carries +signal; all other features are
    standard normal noise. ``priors`` controls the class mix, so a skewed
    training split teaches the frozen classifier a systematic prior bias
    that a constant logit shift can correct.
    """
    _need_numpy()
    y = rng.choice(N_CLASSES, size=n, p=priors)
    X = rng.normal(size=(n, N_FEATURES))
    X[np.arange(n), y] += signal
    return X, y


def train_softmax_regression(X, y, seed, epochs=400, lr=0.1):
    """Plain gradient-descent softmax regression (the frozen base model)."""
    _need_numpy()
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.1, size=(X.shape[1], N_CLASSES))
    b = np.zeros(N_CLASSES)
    Y = np.eye(N_CLASSES)[y]
    n = len(y)
    for _ in range(epochs):
        logits = X @ W + b
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (X.T @ g)
        b -= lr * g.sum(axis=0)
    return W, b


def accuracy(W, b, X, y):
    """Classification accuracy of one (W, b) predictor."""
    _need_numpy()
    return float((np.argmax(X @ W + b, axis=1) == y).mean())


def majority_vote_accuracy(members, X, y):
    """Majority-vote accuracy; ties resolve to the lowest class index."""
    _need_numpy()
    preds = np.stack([np.argmax(X @ W + b, axis=1) for W, b in members])
    out = np.zeros(X.shape[0], dtype=int)
    for j in range(X.shape[0]):
        out[j] = int(np.argmax(np.bincount(preds[:, j], minlength=N_CLASSES)))
    return float((out == y).mean())


def make_perturbed_members(W, b, kind, seed):
    """K perturbed predictors. ``weight`` perturbs matrix + bias;
    ``logit`` keeps the model frozen and perturbs the bias vector only
    (a constant shift of the output logits)."""
    _need_numpy()
    members = []
    for k in range(K_MEMBERS):
        r = np.random.default_rng(seed * 1000 + k)
        if kind == "weight":
            members.append((W + r.normal(scale=SIGMA_WEIGHT, size=W.shape),
                            b + r.normal(scale=SIGMA_WEIGHT, size=b.shape)))
        elif kind == "logit":
            members.append((W, b + r.normal(scale=SIGMA_LOGIT, size=b.shape)))
        else:
            raise ValueError("kind must be 'weight' or 'logit'")
    return members


def select_and_vote(members, X_val, y_val, X_test, y_test):
    """Select top-TOP_M members by validation accuracy, then vote.

    Returns (selvote accuracy, vote-all accuracy, best-member accuracy).
    """
    _need_numpy()
    val = [accuracy(W, b, X_val, y_val) for W, b in members]
    order = np.argsort(val)[::-1]
    selected = [members[i] for i in order[:TOP_M]]
    return (majority_vote_accuracy(selected, X_test, y_test),
            majority_vote_accuracy(members, X_test, y_test),
            accuracy(selected[0][0], selected[0][1], X_test, y_test))


def run_experiment():
    """Full PSV experiment: 3 prior-shift variants x 3 seeds = 9 runs.

    Data RNG seeding mirrors the session's exploratory run so the
    numbers below reproduce exactly (data seed = seed * 17 + 1 per run).
    """
    _need_numpy()
    rows = []
    for variant, priors, signal in VARIANTS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 17 + 1)
            X_tr, y_tr = generate_prior_shift_data(rng, priors, signal, 1200)
            X_va, y_va = generate_prior_shift_data(rng, (0.25,) * 4, signal, 400)
            X_te, y_te = generate_prior_shift_data(rng, (0.25,) * 4, signal, 800)
            W, b = train_softmax_regression(X_tr, y_tr, seed)
            row = {"variant": variant, "seed": seed,
                   "base": accuracy(W, b, X_te, y_te)}
            for kind in ("weight", "logit"):
                members = make_perturbed_members(W, b, kind, seed)
                sel, all_, best = select_and_vote(members, X_va, y_va, X_te, y_te)
                row[kind + "_selvote"] = sel
                row[kind + "_voteall"] = all_
                row[kind + "_best"] = best
            rows.append(row)
    means = {k: float(np.mean([r[k] for r in rows]))
             for k in ("base", "weight_selvote", "weight_voteall", "weight_best",
                       "logit_selvote", "logit_voteall", "logit_best")}
    return {
        "runs": rows,
        "means": means,
        "logit_beats_weight_runs": sum(1 for r in rows
                                       if r["logit_selvote"] > r["weight_selvote"]),
        "logit_beats_base_runs": sum(1 for r in rows
                                     if r["logit_selvote"] > r["base"]),
        "logit_selvote_beats_voteall_runs": sum(
            1 for r in rows if r["logit_selvote"] > r["logit_voteall"]),
        "n_runs": len(rows),
    }
