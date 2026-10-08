"""The Winner's Curse in LLM self-improvement loops.

Paper: arXiv 2610.09239 "The Winner's Curse in LLM Self-Improvement Loops:
Selection Noise, Lock-in, and Acceptance Rules".

One-line claim: keep-if-better on a small selection set is selection under
measurement noise — the greedy loop's final selection-set score exceeds
held-out accuracy by ~13-20 points with 16 selection items (~1-5 with 256),
most post-first proposals are truly harmful (lock-in), and stricter
acceptance rules do not beat greedy acceptance over whole runs.

What this module implements (the mechanism, not the LLM harness):

  * Item-level statistical stand-in. A candidate has a true quality q in
    points (0-100). On a selection set of n items, item j has a difficulty
    d_j (fixed per selection set, shared across candidates) and the
    candidate's per-item outcome is Bernoulli(clip(q/100 + d_j/100)).
    The shared difficulty makes candidate errors correlated within a
    decision (paper claim 1); the selection score is the mean in points,
    with item noise ~50/sqrt(n) points at q=50.
  * Proposal distribution (paper claim 2): q ~ N(q_current - delta,
    sigma_prop), i.e. most proposals after the first are truly harmful.
  * Greedy keep-if-better loop reusing one selection set across
    generations: each generation the incumbent and K proposals are scored
    with fresh evaluation noise on the same set; the winner's score is the
    reported score. Selecting the argmax over noisy scores is the
    winner's curse: E[reported - held-out] ~ c_K * 50/sqrt(n) points.
  * Acceptance rules: "greedy", "fresh64" (commit only if the winning
    candidate also beats the incumbent on 64 fresh items never used for
    selection), "conservative" (commit only if the selection-score gap
    exceeds z * 50/sqrt(n), a one-SE-style significance bar).

Simulation-honesty note: q and the noise are scripted distributions, not
LLMs. This tests the paper's *selection-noise mechanism* (winner's curse
bias, lock-in, acceptance rules); it cannot speak to LLM rewriting
behavior. The bias magnitudes fall out of the Bernoulli item model with
no extra tuning.

numpy-guarded per repo conventions.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.loops.winners_curse")


# ---------------------------------------------------------------------------
# Module-level constants (numpy-free)
# ---------------------------------------------------------------------------

Q0 = 50.0                # true quality (points) of the starting instruction
N_ITEMS_GRID = (16, 64, 256)   # selection-set sizes from the paper
ITEM_STD_POINTS = 50.0   # Bernoulli(p=0.5) item-score std, in points
HELDOUT_ITEMS = 4096     # items for the "true" held-out accuracy estimate
FRESH64_ITEMS = 64       # fresh items for the claim-4 diagnostic / rule
Z_CONSERVATIVE = 1.0     # one-SE-style significance bar for the rule
SIGMA_DIFFICULTY = 20.0  # std of per-item difficulty, in points

ACCEPTANCE_RULES = ("greedy", "fresh64", "conservative")


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def make_selection_set(rng, n_items, sigma_difficulty=SIGMA_DIFFICULTY):
    """Per-item difficulties d_j (points), fixed for one selection set.

    The same difficulty vector is shared by every candidate scored on the
    set, which is what makes candidate errors correlated within a decision
    (paper claim 1).
    """
    _need_numpy()
    return rng.normal(0.0, sigma_difficulty, size=n_items)


def _item_probs(q, item_diff):
    q = np.asarray(q, dtype=float)
    p = q[..., None] / 100.0 + item_diff[None, :] / 100.0
    return np.clip(p, 0.0, 1.0)


def score_candidates(q, item_diff, rng):
    """Selection-set scores (points) for qualities q on one selection set.

    Fresh Bernoulli evaluation noise is drawn on every call; the item
    difficulties (the set itself) stay fixed.
    """
    _need_numpy()
    p = _item_probs(q, np.asarray(item_diff, dtype=float))
    draws = rng.random(p.shape) < p
    return 100.0 * draws.mean(axis=-1)


def heldout_score(q, rng, n_items=HELDOUT_ITEMS,
                  sigma_difficulty=SIGMA_DIFFICULTY):
    """Score on n_items fresh items never used for selection.

    New item difficulties and new evaluation noise: the held-out accuracy
    analogue. With n_items=HELDOUT_ITEMS this is effectively the true
    quality; with n_items=FRESH64_ITEMS it is the paper's 64-item probe.
    """
    _need_numpy()
    fresh_diff = make_selection_set(rng, n_items, sigma_difficulty)
    return score_candidates(q, fresh_diff, rng)


def generate_candidates(q_current, n, rng, delta=2.0, sigma_prop=3.0):
    """Proposal distribution: q ~ N(q_current - delta, sigma_prop).

    Centered below the incumbent, so most proposals are truly harmful
    (paper claim 2). With delta=2, sigma_prop=3, P(harmful) ~= 0.75.
    """
    _need_numpy()
    q = rng.normal(q_current - delta, sigma_prop, size=n)
    return np.clip(q, 1.0, 99.0)


def error_correlation(n_items, n_seeds=400, seed=0,
                      sigma_difficulty=SIGMA_DIFFICULTY):
    """Pairwise correlation of candidate score errors within a decision.

    Two equal-quality candidates share one selection set per decision; their
    errors e = score - true q decompose into a shared set component
    (item difficulties) plus independent evaluation noise. Correlating the
    two candidates' errors across many decisions (fresh set each time)
    estimates corr(e_A, e_B) > 0 (paper claim 1).
    """
    _need_numpy()
    rng = np.random.default_rng(seed)
    ea = np.empty(n_seeds)
    eb = np.empty(n_seeds)
    for s in range(n_seeds):
        item_diff = make_selection_set(rng, n_items, sigma_difficulty)
        sc = score_candidates(np.array([Q0, Q0]), item_diff, rng)
        ea[s] = sc[0] - Q0
        eb[s] = sc[1] - Q0
    ea = ea - ea.mean()
    eb = eb - eb.mean()
    denom = np.sqrt((ea ** 2).sum() * (eb ** 2).sum())
    if denom == 0:
        return 0.0
    return float(np.dot(ea, eb) / denom)


def winners_curse_gap(n_items, n_candidates=8, n_seeds=400, seed=0,
                      sigma_difficulty=SIGMA_DIFFICULTY):
    """One-shot winner's curse: E[selection score - held-out score].

    K candidates of *equal* true quality; pick the argmax on the selection
    set. The gap is pure selection bias (no quality differences).
    """
    _need_numpy()
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_seeds)
    for s in range(n_seeds):
        item_diff = make_selection_set(rng, n_items, sigma_difficulty)
        q = np.full(n_candidates, Q0)
        sel = score_candidates(q, item_diff, rng)
        best = int(np.argmax(sel))
        h = float(heldout_score(np.array([Q0]), rng)[0])
        gaps[s] = sel[best] - h
    return float(gaps.mean())


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------

def _run_loop(rule, n_items, n_gens, n_candidates, delta, sigma_prop,
              sigma_difficulty, seed, z_conservative):
    _need_numpy()
    if rule not in ACCEPTANCE_RULES:
        raise ValueError(f"unknown acceptance rule {rule!r}; "
                         f"choose from {ACCEPTANCE_RULES}")
    rng = np.random.default_rng(seed)
    sel_diff = make_selection_set(rng, n_items, sigma_difficulty)

    q_cur = Q0
    # Starting instruction scored once on the selection set (reported base).
    s_start = float(score_candidates(np.array([q_cur]), sel_diff, rng)[0])
    # ... and on 64 fresh items never used for selection (claim-4 probe).
    f64_start = float(heldout_score(np.array([q_cur]), rng, FRESH64_ITEMS)[0])

    n_commits = 0
    harmful_commits = 0
    proposals_total = 0
    harmful_proposals = 0
    true_gain_per_commit = []

    s_reported = s_start
    for _ in range(n_gens):
        q_prop = generate_candidates(q_cur, n_candidates, rng, delta,
                                     sigma_prop)
        proposals_total += n_candidates
        harmful_proposals += int(np.sum(q_prop < q_cur))
        s_prop = score_candidates(q_prop, sel_diff, rng)
        s_cur = float(score_candidates(np.array([q_cur]), sel_diff, rng)[0])
        best = int(np.argmax(s_prop))

        accept = False
        if rule == "greedy":
            accept = bool(s_prop[best] > s_cur)
        elif rule == "fresh64":
            # commit only if the winner also beats the incumbent on 64
            # fresh items never used for selection
            if s_prop[best] > s_cur:
                f_diff = make_selection_set(rng, FRESH64_ITEMS,
                                            sigma_difficulty)
                f_cur = float(
                    score_candidates(np.array([q_cur]), f_diff, rng)[0])
                f_best = float(
                    score_candidates(np.array([q_prop[best]]), f_diff,
                                     rng)[0])
                accept = bool(f_best > f_cur)
        else:  # conservative: one-SE-style significance bar on the gap
            se = ITEM_STD_POINTS / np.sqrt(n_items)
            accept = bool((s_prop[best] - s_cur) > z_conservative * se)

        if accept:
            if q_prop[best] <= q_cur:
                harmful_commits += 1
            true_gain_per_commit.append(float(q_prop[best] - q_cur))
            q_cur = float(q_prop[best])
            s_reported = float(s_prop[best])
            n_commits += 1
        else:
            s_reported = s_cur

    # Final reported score: the winning score of the last generation's
    # comparison on the selection set (what the loop actually observed).
    s_final = s_reported
    h_final = float(heldout_score(np.array([q_cur]), rng)[0])
    f64_final = float(heldout_score(np.array([q_cur]), rng, FRESH64_ITEMS)[0])
    true_gain = q_cur - Q0
    return {
        "rule": rule,
        "n_items": n_items,
        "n_gens": n_gens,
        "n_candidates": n_candidates,
        "seed": seed,
        # paper claim 3: reported (selection-set) gain vs true held-out gain
        "reported_gain": s_final - s_start,
        "true_gain": true_gain,
        # paper claim 3 headline number: final selection-set score minus
        # held-out accuracy of the final instruction
        "selection_bias": s_final - h_final,
        "final_true_quality": q_cur,
        "n_commits": n_commits,
        # lock-in diagnostics (paper claim 2)
        "harmful_commit_frac": (harmful_commits / n_commits
                                if n_commits else 0.0),
        "harmful_proposal_frac": harmful_proposals / proposals_total,
        "mean_true_gain_per_commit": (float(np.mean(true_gain_per_commit))
                                      if true_gain_per_commit else 0.0),
        # paper claim 4 diagnostic: scoring start and current on 64 fresh
        # items never used for selection
        "fresh64_gain": f64_final - f64_start,
        "fresh64_gain_bias": (f64_final - f64_start) - true_gain,
        "fresh64_gain_abs_err": abs((f64_final - f64_start) - true_gain),
    }


def run_greedy_loop(n_items=16, n_gens=8, n_candidates=6, delta=2.0,
                    sigma_prop=3.0, sigma_difficulty=SIGMA_DIFFICULTY,
                    seed=0):
    """One greedy keep-if-better loop; returns the metric dict (see _run_loop)."""
    return _run_loop("greedy", n_items, n_gens, n_candidates, delta,
                     sigma_prop, sigma_difficulty, seed, Z_CONSERVATIVE)


def run_loop_with_rule(rule, n_items=16, n_gens=8, n_candidates=6, delta=2.0,
                       sigma_prop=3.0, sigma_difficulty=SIGMA_DIFFICULTY,
                       seed=0, z_conservative=Z_CONSERVATIVE):
    """One loop under an acceptance rule; rule in ACCEPTANCE_RULES."""
    return _run_loop(rule, n_items, n_gens, n_candidates, delta, sigma_prop,
                     sigma_difficulty, seed, z_conservative)


def selection_bias(n_items, n_seeds=300, n_gens=8, n_candidates=6, delta=2.0,
                   sigma_prop=3.0, sigma_difficulty=SIGMA_DIFFICULTY,
                   seed=0, rule="greedy"):
    """Mean E[final selection-set score - held-out accuracy] over seeds.

    The paper's headline quantity (13-20 pts at n=16, 1-5 pts at n=256
    for the greedy loop), plus mean reported/true gains for context.
    """
    _need_numpy()
    biases, rep, tru = [], [], []
    for s in range(n_seeds):
        r = _run_loop(rule, n_items, n_gens, n_candidates, delta, sigma_prop,
                      sigma_difficulty, seed + s, Z_CONSERVATIVE)
        biases.append(r["selection_bias"])
        rep.append(r["reported_gain"])
        tru.append(r["true_gain"])
    biases = np.asarray(biases)
    return {
        "rule": rule,
        "n_items": n_items,
        "n_seeds": n_seeds,
        "mean_bias": float(biases.mean()),
        "std_bias": float(biases.std()),
        "mean_reported_gain": float(np.mean(rep)),
        "mean_true_gain": float(np.mean(tru)),
    }
