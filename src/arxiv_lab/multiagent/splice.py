"""SPLICE: locating and closing information gaps in multi-agent systems.

Paper: arXiv 2610.40186 "Locating and Closing Information Gaps in
Multi-Agent Systems" (Gong et al., submitted 29 Oct 2026).

One-line claim: in a fixed multi-agent workflow, a missing edge can leave
the receiver without evidence it needs (an information gap), and adding
that edge can also displace a correct answer with an incorrect upstream
conclusion — so SPLICE locates a possible gap per query, adds one directed
edge, re-executes the receiver and affected downstream nodes, and lets a
selector trained on PROCESS features of the edit (how far the receiver
moved, where its new content came from, whether the change reached the
final answer, plus observed conflict / coverage) choose between the
candidate and the original answer. On the paper's benchmarks this raises
task score by 8.0% (Web tool use) and 5.7% mean (reasoning/planning)
at 1.25-1.44x the skeleton's cost.

This module implements the mechanism as a deterministic CPU simulation
(see the 2026-10-31 session for the measured reproduction): a skeleton
workflow Retriever A -> Receiver -> Final, an unconnected Retriever B,
three query categories (GAP / NOGAP / TRAP), a gap locator, candidate
re-execution, noisy process-feature observation, and a logistic-regression
selector trained on a dev split. numpy-guarded per repo conventions.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

REEXEC_SHARE = 0.5  # receiver + downstream share of one workflow run


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.multiagent.splice")


def make_queries(n, seed, gap_frac=0.25, trap_frac=0.30):
    """Synthetic queries. Returns (queries, rng).

    GAP: deciding fact held only by B (skeleton incomplete, candidate
    correct). NOGAP: deciding fact held by A, B unrelated (no probe).
    TRAP: A correct, B contradicts A on the deciding key (candidate
    wrong — the displacement hazard).
    """
    _need_numpy()
    rng = np.random.default_rng(seed)
    cats = rng.choice(["GAP", "NOGAP", "TRAP"], size=n,
                      p=[gap_frac, 1 - gap_frac - trap_frac, trap_frac])
    queries = []
    for i, cat in enumerate(cats):
        key = "k%d" % i
        correct = int(rng.integers(0, 2))
        q = {"id": i, "category": str(cat), "key": key, "correct": correct}
        if cat == "GAP":
            q["A"] = {}
            q["B"] = {key: correct}
        elif cat == "NOGAP":
            q["A"] = {key: correct}
            q["B"] = {"u%d" % i: 0}
        else:
            q["A"] = {key: correct}
            q["B"] = {key: 1 - correct}
        queries.append(q)
    return queries, rng


def run_skeleton(q):
    """Skeleton run: receiver sees A only. (answer, evidence)."""
    if q["key"] not in q["A"]:
        return None, dict(q["A"])
    return q["A"][q["key"]], dict(q["A"])


def run_candidate(q):
    """Candidate run: receiver sees A + B, B wins on conflict."""
    vis = dict(q["A"])
    vis.update(q["B"])
    if q["key"] not in vis:
        return None, vis
    return vis[q["key"]], vis


def locate_gap(q, skel_answer):
    """Gap locator: incomplete skeleton answer, or B's output shares the
    deciding key (relevance probe)."""
    if skel_answer is None:
        return True
    return q["key"] in q["B"]


def process_features(q, skel_answer, skel_ev, cand_answer, cand_ev, rng):
    """Observed process features [movement, provenance, final_reach,
    conflict_obs, coverage_obs] with probe noise: a real conflict is
    missed w.p. 0.20; a trap shows spurious coverage w.p. 0.15."""
    _need_numpy()
    new_keys = [k for k in cand_ev if k not in skel_ev]
    changed = [k for k in cand_ev if k in skel_ev and cand_ev[k] != skel_ev[k]]
    movement = (len(new_keys) + len(changed)) / max(1, len(cand_ev))
    provenance = 1.0  # all new content arrived via the added edge
    final_reach = 1.0 if cand_answer != skel_answer else 0.0
    true_conflict = 1.0 if changed else 0.0
    true_coverage = 1.0 if (q["key"] in new_keys) else 0.0
    conflict_obs = true_conflict if rng.random() >= 0.20 else 0.0
    coverage_obs = true_coverage
    if true_coverage == 0.0 and q["category"] == "TRAP":
        coverage_obs = 1.0 if rng.random() < 0.15 else 0.0
    return np.array([movement, provenance, final_reach, conflict_obs,
                     coverage_obs])


def train_selector(X, y, steps=2000, lr=0.5):
    """Deterministic full-batch logistic regression. Returns a model
    tuple consumed by selector_predict."""
    _need_numpy()
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        g = p - y
        w -= lr * (Xs.T @ g) / len(y)
        b -= lr * g.mean()
    return (w, b, mu, sd)


def selector_predict(model, X):
    """Boolean adoption decisions for feature rows X."""
    _need_numpy()
    w, b, mu, sd = model
    z = ((X - mu) / sd) @ w + b
    return (1.0 / (1.0 + np.exp(-z)) >= 0.5)


def run_experiment(n_dev=400, n_test=400, seed=20261031):
    """Dev-train the selector, evaluate on held-out test queries.

    Returns measured numbers: probe rate, cost ratio, accuracies for
    skeleton / always-adopt / SPLICE selector / oracle, and per-category
    adoption rates on the test split.
    """
    _need_numpy()
    dev, rng = make_queries(n_dev, seed)
    test, _ = make_queries(n_test, seed + 1)

    def evaluate(queries, rng):
        rows = []
        for q in queries:
            skel_ans, skel_ev = run_skeleton(q)
            skel_ok = (skel_ans == q["correct"])
            if not locate_gap(q, skel_ans):
                rows.append({"q": q, "probed": False, "skel_ok": skel_ok,
                             "cand_ok": skel_ok, "feat": None,
                             "adopted": False})
                continue
            cand_ans, cand_ev = run_candidate(q)
            rows.append({"q": q, "probed": True, "skel_ok": skel_ok,
                         "cand_ok": (cand_ans == q["correct"]),
                         "feat": process_features(q, skel_ans, skel_ev,
                                                  cand_ans, cand_ev, rng),
                         "adopted": None})
        return rows

    dev_rows = evaluate(dev, rng)
    probed = [r for r in dev_rows if r["probed"]]
    X = np.array([r["feat"] for r in probed])
    y = np.array([1.0 if r["cand_ok"] else 0.0 for r in probed])
    model = train_selector(X, y)
    test_rows = evaluate(test, np.random.default_rng(seed + 2))
    for r in test_rows:
        if r["probed"]:
            r["adopted"] = bool(selector_predict(
                model, r["feat"].reshape(1, -1))[0])

    def acc(mode):
        ok = 0
        for r in test_rows:
            if mode == "skeleton":
                ok += r["skel_ok"]
            elif mode == "always_adopt":
                ok += r["cand_ok"] if r["probed"] else r["skel_ok"]
            elif mode == "splice":
                ok += (r["cand_ok"] if r["adopted"] else r["skel_ok"]) \
                    if r["probed"] else r["skel_ok"]
            elif mode == "oracle":
                ok += (r["cand_ok"] if r["cand_ok"] else r["skel_ok"]) \
                    if r["probed"] else r["skel_ok"]
        return ok / len(test_rows)

    probed_test = [r for r in test_rows if r["probed"]]
    trap = [r for r in test_rows
            if r["probed"] and r["q"]["category"] == "TRAP"]
    gap = [r for r in test_rows
           if r["probed"] and r["q"]["category"] == "GAP"]
    return {
        "n_dev": n_dev, "n_test": n_test, "seed": seed,
        "probe_rate": len(probed_test) / len(test_rows),
        "cost_ratio": 1.0 + REEXEC_SHARE * len(probed_test) / len(test_rows),
        "acc_skeleton": acc("skeleton"),
        "acc_always_adopt": acc("always_adopt"),
        "acc_splice": acc("splice"),
        "acc_oracle": acc("oracle"),
        "trap_adopt_rate": float(np.mean([r["adopted"] for r in trap])),
        "gap_adopt_rate": float(np.mean([r["adopted"] for r in gap])),
        "selector_weights": model[0].tolist(),
    }
