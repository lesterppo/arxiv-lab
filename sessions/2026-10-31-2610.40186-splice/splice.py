"""SPLICE (2610.40186) — Selective Probing and Link Insertion for Closing
information gaps in multi-agent Execution, reproduced as a deterministic
CPU simulation of a fixed multi-agent workflow.

Workflow (skeleton): Retriever A -> Receiver R -> Final F. A second
retriever B exists but is NOT connected to R in the skeleton, so evidence
held only by B can never reach R: an information gap.

SPLICE step, per query:
  1. Run the skeleton.
  2. Locate a possible gap: (a) R's answer is incomplete (a required key
     is missing), or (b) an unconnected node's output shares the query's
     deciding key (relevance probe). If neither fires, no candidate.
  3. Add one directed edge B -> R, re-execute R and downstream F
     (receiver + downstream = 0.5 of workflow cost, a stated modelling
     choice), producing a candidate answer.
  4. Adoption: a selector trained on PROCESS features of the edit
     (content movement, provenance, final-reach, observed conflict,
     observed coverage gain) chooses candidate vs original.

Query categories (ground truth, hidden from the selector):
  GAP   — deciding fact held only by B; skeleton incomplete/wrong,
          candidate correct.
  NOGAP — deciding fact held by A; B's output is on unrelated keys
          (relevance probe does not fire) -> no candidate, skeleton right.
  TRAP  — A holds the correct deciding fact; B holds a CONTRADICTING
          value for the same key. The receiver trusts edge-delivered
          content on conflict, so the candidate is WRONG: adopting it
          displaces a correct answer. This is the paper's central hazard:
          movement / final-reach features are 1 for both GAP and TRAP,
          so a final-output-only rule cannot separate them; only the
          conflict / coverage process features can.
Feature observation is noisy, as in any real probe: a trap's conflict is
missed with probability 0.20, and a trap can show spurious coverage with
probability 0.15. The selector is a small logistic regression (numpy,
deterministic gradient descent) fitted on a dev split and evaluated on a
held-out test split — mirroring the paper's "selector trained on process
features" without claiming the paper's exact feature set (the abstract
names the feature families, not their formulas).
"""
import numpy as np

REEXEC_SHARE = 0.5  # receiver + downstream share of one workflow run


def make_queries(n, seed, gap_frac=0.25, trap_frac=0.30):
    """Return list of query dicts. nogap share = 1 - gap - trap."""
    rng = np.random.default_rng(seed)
    cats = rng.choice(["GAP", "NOGAP", "TRAP"], size=n,
                      p=[gap_frac, 1 - gap_frac - trap_frac, trap_frac])
    queries = []
    for i, cat in enumerate(cats):
        key = f"k{i}"
        correct = int(rng.integers(0, 2))
        q = {"id": i, "category": str(cat), "key": key, "correct": correct}
        if cat == "GAP":
            q["A"] = {}                      # A lacks the deciding fact
            q["B"] = {key: correct}          # only B holds it, correct
        elif cat == "NOGAP":
            q["A"] = {key: correct}
            q["B"] = {f"u{i}": 0}            # unrelated keys only
        else:  # TRAP
            q["A"] = {key: correct}
            q["B"] = {key: 1 - correct}      # contradicts A on the key
        queries.append(q)
    return queries, rng


def receiver_answer(visible):
    """Receiver decision: missing deciding key -> incomplete (wrong);
    conflicting values -> trust the edge-delivered (B) value."""
    return visible


def run_skeleton(q):
    """Skeleton: R sees A only. Returns (answer, evidence)."""
    if q["key"] not in q["A"]:
        return None, dict(q["A"])  # incomplete -> wrong
    return q["A"][q["key"]], dict(q["A"])


def run_candidate(q):
    """Candidate: R sees A + B (B wins on conflict)."""
    vis = dict(q["A"])
    vis.update(q["B"])
    if q["key"] not in vis:
        return None, vis
    return vis[q["key"]], vis


def locate_gap(q, skel_answer, skel_ev):
    """Gap locator: incomplete skeleton answer, or B's output shares the
    deciding key (relevance probe). NOGAP fires neither."""
    if skel_answer is None:
        return True
    return q["key"] in q["B"]


def process_features(q, skel_answer, skel_ev, cand_answer, cand_ev, rng):
    """Process features of the edit (observed, with probe noise)."""
    new_keys = [k for k in cand_ev if k not in skel_ev]
    changed = [k for k in cand_ev if k in skel_ev and cand_ev[k] != skel_ev[k]]
    movement = (len(new_keys) + len(changed)) / max(1, len(cand_ev))
    provenance = (len(new_keys) + len(changed)) / max(
        1, len(new_keys) + len(changed))  # all new content came via the edge
    final_reach = 1.0 if cand_answer != skel_answer else 0.0
    true_conflict = 1.0 if changed else 0.0
    true_coverage = 1.0 if (q["key"] in new_keys) else 0.0
    # Noisy observation: miss a real conflict w.p. 0.20; spurious coverage
    # on a trap w.p. 0.15.
    conflict_obs = true_conflict if rng.random() >= 0.20 else 0.0
    coverage_obs = true_coverage
    if true_coverage == 0.0 and q["category"] == "TRAP":
        coverage_obs = 1.0 if rng.random() < 0.15 else 0.0
    return np.array([movement, provenance, final_reach, conflict_obs,
                     coverage_obs])


def train_selector(X, y, steps=2000, lr=0.5, seed=0):
    """Deterministic logistic regression (full-batch GD)."""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1])
    b = 0.0
    for _ in range(steps):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (Xs.T @ g) / len(y)
        b -= lr * g.mean()
    return (w, b, mu, sd)


def selector_predict(model, X):
    w, b, mu, sd = model
    z = ((X - mu) / sd) @ w + b
    return (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)


def run_experiment(n_dev=400, n_test=400, seed=20261031):
    dev, rng = make_queries(n_dev, seed)
    test, _ = make_queries(n_test, seed + 1)

    def evaluate(queries, rng, model=None):
        rows = []
        for q in queries:
            skel_ans, skel_ev = run_skeleton(q)
            skel_ok = (skel_ans == q["correct"])
            if not locate_gap(q, skel_ans, skel_ev):
                rows.append({"q": q, "probed": False, "skel_ok": skel_ok,
                             "cand_ok": skel_ok, "feat": None,
                             "adopted": False})
                continue
            cand_ans, cand_ev = run_candidate(q)
            cand_ok = (cand_ans == q["correct"])
            feat = process_features(q, skel_ans, skel_ev, cand_ans, cand_ev,
                                    rng)
            rows.append({"q": q, "probed": True, "skel_ok": skel_ok,
                         "cand_ok": cand_ok, "feat": feat,
                         "adopted": None})
        return rows

    dev_rows = evaluate(dev, rng)
    # Train selector: adopt (1) iff the candidate is correct, on probed dev.
    probed = [r for r in dev_rows if r["probed"]]
    X = np.array([r["feat"] for r in probed])
    y = np.array([1.0 if r["cand_ok"] else 0.0 for r in probed])
    model = train_selector(X, y)
    test_rows = evaluate(test, np.random.default_rng(seed + 2))
    for r in test_rows:
        if r["probed"]:
            r["adopted"] = bool(selector_predict(
                model, r["feat"].reshape(1, -1))[0])

    def accuracy(rows, mode):
        ok = 0
        for r in rows:
            if mode == "skeleton":
                ok += r["skel_ok"]
            elif mode == "always_adopt":
                ok += r["cand_ok"] if r["probed"] else r["skel_ok"]
            elif mode == "splice":
                if r["probed"]:
                    ok += r["cand_ok"] if r["adopted"] else r["skel_ok"]
                else:
                    ok += r["skel_ok"]
        return ok / len(rows)

    def accuracy_oracle(rows):
        # Oracle adopts exactly when the candidate is correct.
        ok = sum((r["cand_ok"] if (r["probed"] and r["cand_ok"])
                  else r["skel_ok"]) for r in rows)
        return ok / len(rows)

    probed_test = [r for r in test_rows if r["probed"]]
    trap_test = [r for r in test_rows
                 if r["probed"] and r["q"]["category"] == "TRAP"]
    gap_test = [r for r in test_rows
                if r["probed"] and r["q"]["category"] == "GAP"]
    out = {
        "n_dev": n_dev, "n_test": n_test, "seed": seed,
        "probe_rate": len(probed_test) / len(test_rows),
        "cost_ratio": 1.0 + REEXEC_SHARE * len(probed_test) / len(test_rows),
        "acc_skeleton": accuracy(test_rows, "skeleton"),
        "acc_always_adopt": accuracy(test_rows, "always_adopt"),
        "acc_splice": accuracy(test_rows, "splice"),
        "acc_oracle": accuracy_oracle(test_rows),
        "trap_adopt_rate": float(np.mean([r["adopted"] for r in trap_test])),
        "gap_adopt_rate": float(np.mean([r["adopted"] for r in gap_test])),
        "n_trap_test": len(trap_test), "n_gap_test": len(gap_test),
        "selector_weights": model[0].tolist(),
    }
    return out
