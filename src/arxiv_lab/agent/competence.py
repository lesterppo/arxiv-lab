"""Competence-aware orchestration for LLM multi-agent systems (arXiv:2610.13144).

Paper claim implemented: a two-stage framework for automatic multi-agent
orchestration. Stage 1 models the backbone's *task-conditioned competence*
for the current query (here: online per-agent, per-task-type reliability
estimates from observed outcomes). Stage 2 uses that signal to adapt *role
activation* and the *communication graph*: high estimated competence ->
a single role answers alone; medium -> proposer + verifier; low -> a
multi-role deliberation round. The paper reports that for higher-competence
backbones this lifts accuracy (87.22 -> 89.97) while cutting token cost by
36.26%, and for lower-competence backbones adds +4.58 points over fixed
orchestration schemes.

Provides ``CompetenceTracker`` (stage 1), ``orchestrate`` (stage 2),
``simulate_mas`` (seeded synthetic multi-agent evaluation), and the
fixed ``solo`` / ``full`` baseline orchestrators.

Requires numpy (optional dependency): ``pip install numpy``.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def _need_numpy():
    if np is None:
        raise ImportError(
            "arxiv_lab.agent.competence requires numpy (optional dependency); "
            "install it with `pip install numpy`"
        )


# ---------------------------------------------------------------------------
# Stage 1: task-conditioned competence modelling
# ---------------------------------------------------------------------------

class CompetenceTracker:
    """Online per-agent, per-task-type competence estimates.

    Each observation is a binary outcome (agent answered correctly or not)
    for a given task type. The estimate is an exponentially-weighted moving
    average initialised at ``prior`` (conservative 0.5), so it adapts to
    distribution shift without storing history.
    """

    def __init__(self, alpha=0.08, prior=0.5):
        _need_numpy()
        self.alpha = alpha
        self.prior = prior
        self._est = {}  # (agent_id, task_type) -> float

    def observe(self, agent_id, task_type, correct):
        key = (agent_id, task_type)
        prev = self._est.get(key, self.prior)
        self._est[key] = (1.0 - self.alpha) * prev + self.alpha * float(correct)

    def estimate(self, agent_id, task_type):
        return self._est.get((agent_id, task_type), self.prior)

    def best(self, agent_ids, task_type):
        """Agent id with the highest estimated competence (ties -> first)."""
        return max(agent_ids, key=lambda a: self.estimate(a, task_type))


# ---------------------------------------------------------------------------
# Stage 2: adaptive role activation + communication graph
# ---------------------------------------------------------------------------

SOLO, PAIR, DELIBERATE = "solo", "pair", "deliberate"

# Token model: prompt grows with collaboration (transcript re-reads).
MODE_TOKENS = {
    SOLO: 700,          # 1 call x (500 prompt + 200 completion)
    PAIR: 2000,         # proposer 700 + verifier 1300 (reads the proposal)
    DELIBERATE: 5200,   # 3 roles x 2 rounds, growing transcript
}


def select_mode(est_competence, hi=0.85, mid=0.60):
    """Map the (query-conditioned) estimated competence to a mode."""
    if est_competence >= hi:
        return SOLO
    if est_competence >= mid:
        return PAIR
    return DELIBERATE


def query_competence(tracker, agent_id, task_type, difficulty_proxy,
                     difficulty_weight=0.45):
    """Query-conditioned competence: base estimate minus difficulty penalty.

    Mirrors the paper's stage 1 ("task-conditioned competence for the
    current query"): the per-agent, per-task estimate is adjusted by a
    cheap observable difficulty proxy (query features / a fast probe).
    The default weight is deliberately conservative (larger than the true
    data-generating penalty), so collaboration is spent cautiously.
    """
    base = tracker.estimate(agent_id, task_type)
    return float(min(1.0, max(0.0, base - difficulty_weight * difficulty_proxy)))


def communication_graph(mode, roles):
    """Directed edges (speaker -> listener) for the chosen mode."""
    if mode == SOLO:
        return []
    if mode == PAIR:
        proposer, verifier = roles[0], roles[1]
        return [(proposer, verifier), (verifier, proposer)]
    # DELIBERATE: round-robin, every role reads every other role each round
    edges = []
    for i, a in enumerate(roles):
        for j, b in enumerate(roles):
            if i != j:
                edges.append((a, b))
    return edges


def orchestrate(tracker, agent_ids, task_type, difficulty_proxy=None,
                hi=0.85, mid=0.60):
    """Two-stage orchestration: estimate competence, then adapt the graph.

    With ``difficulty_proxy`` given, the mode is chosen on the
    query-conditioned competence (paper's stage 1); otherwise on the raw
    per-task estimate. Role ranking always uses the base estimates.

    Returns (mode, roles, edges) where roles are the activated agent ids
    in activation order and edges the communication graph.
    """
    ranked = sorted(agent_ids, key=lambda a: -tracker.estimate(a, task_type))
    if difficulty_proxy is None:
        cond = tracker.estimate(ranked[0], task_type)
    else:
        cond = query_competence(tracker, ranked[0], task_type, difficulty_proxy)
    mode = select_mode(cond, hi, mid)
    n_roles = {SOLO: 1, PAIR: 2, DELIBERATE: 3}[mode]
    roles = ranked[:n_roles]
    return mode, roles, communication_graph(mode, roles)


# ---------------------------------------------------------------------------
# Seeded simulation: hidden ground-truth competence, noisy outcomes
# ---------------------------------------------------------------------------

TASK_TYPES = ("math", "code", "qa", "logic")


def simulate_mas(rng, n_queries, agent_competence, task_mix=None,
                 policy="adaptive", seed_outcomes=0, difficulty_noise=0.10):
    """Run a seeded multi-agent simulation.

    ``agent_competence``: dict agent_id -> dict task_type -> true base
    success probability in [0,1]. Each query also draws a difficulty
    ``d ~ Beta(2, 5)``; the true per-query success probability is
    ``clip(base - 0.25 * d, 0.02, 0.99)``. The orchestrator observes only
    a noisy proxy ``d_hat = clip(d + Normal(0, difficulty_noise), 0, 1)``
    (the cheap query-difficulty signal of the paper's stage 1), so
    collaboration is spent on hard queries and skipped on easy ones.

    ``policy``: "adaptive" (two-stage competence-aware), "full" (always
    3-role deliberation), "solo" (always the best-estimated single agent).

    Returns dict with accuracy, total tokens, mode histogram, and the
    tracker (for calibration checks).
    """
    _need_numpy()
    agents = list(agent_competence.keys())
    tracker = CompetenceTracker()
    out_rng = np.random.default_rng(seed_outcomes or rng.integers(2**31))

    if task_mix is None:
        task_mix = [1.0 / len(TASK_TYPES)] * len(TASK_TYPES)

    correct = 0
    tokens = 0
    modes = {SOLO: 0, PAIR: 0, DELIBERATE: 0}

    for _ in range(n_queries):
        task = TASK_TYPES[out_rng.choice(len(TASK_TYPES), p=task_mix)]
        difficulty = float(out_rng.beta(2, 5))
        d_hat = float(min(1.0, max(0.0,
                                   difficulty + out_rng.normal(0, difficulty_noise))))
        ranked = sorted(agents, key=lambda a: -tracker.estimate(a, task))

        if policy == "adaptive":
            mode, roles, _ = orchestrate(tracker, agents, task,
                                         difficulty_proxy=d_hat)
        elif policy == "full":
            mode, roles = DELIBERATE, ranked[:3]
        elif policy == "solo":
            mode, roles = SOLO, ranked[:1]
        else:
            raise ValueError(f"unknown policy {policy!r}")
        modes[mode] += 1
        tokens += MODE_TOKENS[mode]

        # Each activated agent draws an independent outcome at its true
        # query-conditioned competence.
        def qp(a):
            p = agent_competence[a][task] - 0.25 * difficulty
            return min(0.99, max(0.02, p))

        if mode == SOLO:
            ok = out_rng.random() < qp(roles[0])
            tracker.observe(roles[0], task, ok)
        elif mode == PAIR:
            proposer, verifier = roles[0], roles[1]
            p1, p2 = qp(proposer), qp(verifier)
            prop_ok = out_rng.random() < p1
            # The verifier's own-answer correctness: an unbiased stage-1
            # signal for its task competence (also reused as the
            # correctness of any substituted answer below).
            ver_latent = out_rng.random() < p2
            if prop_ok:
                # Verifier confirms (prob p2) -> correct; false-alarms and
                # substitutes its own answer -> correct iff ver_latent.
                ok = (out_rng.random() < p2) or ver_latent
            else:
                # Verifier catches the error (prob p2) and substitutes its
                # own answer -> correct iff ver_latent; else agrees wrong.
                ok = (out_rng.random() < p2) and ver_latent
            tracker.observe(proposer, task, prop_ok)
            tracker.observe(verifier, task, ver_latent)
        else:  # DELIBERATE: majority of the three independent draws
            draws = [out_rng.random() < qp(a) for a in roles]
            ok = sum(draws) >= 2
            for a, d in zip(roles, draws):
                tracker.observe(a, task, d)

        correct += ok

    return {
        "accuracy": correct / n_queries,
        "tokens": tokens,
        "modes": modes,
        "tracker": tracker,
    }


def _spearman(x, y):
    """Rank correlation via argsort (no scipy dependency)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def query_calibration_spearman(tracker, agent_competence, rng, n=300,
                               task_types=TASK_TYPES, difficulty_noise=0.10):
    """Spearman between the stage-1 query-conditioned competence signal
    and the true query-conditioned competence, over ``n`` fresh queries.

    This is the faithful stage-1 check for the paper ("task-conditioned
    competence for the current query"): for each query the orchestrator's
    own top-ranked agent is used, its query-conditioned estimate
    (base estimate minus the difficulty proxy) is compared against that
    agent's true per-query success probability.
    """
    _need_numpy()
    agents = list(agent_competence.keys())
    est, true = [], []
    for _ in range(n):
        task = task_types[rng.integers(len(task_types))]
        difficulty = float(rng.beta(2, 5))
        d_hat = float(min(1.0, max(0.0,
                                   difficulty + rng.normal(0, difficulty_noise))))
        top = tracker.best(agents, task)
        est.append(query_competence(tracker, top, task, d_hat))
        true.append(agent_competence[top][task] - 0.25 * difficulty)
    return _spearman(est, true)


def calibration_spearman(tracker, agent_competence, task_types=TASK_TYPES):
    """Spearman correlation between estimated and true competence.

    NOTE: measured on the selectively-observed stream this is affected by
    bandit-style selection bias (rarely-activated agent x task cells stay
    near the prior). Prefer ``query_calibration_spearman`` for the
    stage-1 claim.
    """
    _need_numpy()
    est, true = [], []
    for a, prof in agent_competence.items():
        for t in task_types:
            est.append(tracker.estimate(a, t))
            true.append(prof[t])
    return _spearman(est, true)


def make_backbone(rng, kind):
    """Hidden ground-truth competence profiles.

    ``kind`` "high": specialist agents, most task types at 0.88-0.97
    (paper's higher-competence backbone, ~87-90% observed accuracy);
    "low": 0.62-0.75 (paper's lower-competence backbone). Three roles,
    four task types; each agent has one weak spot so routing matters.
    """
    _need_numpy()
    agents = ["coder", "reasoner", "researcher"]
    lo, hi = (0.88, 0.97) if kind == "high" else (0.62, 0.75)
    prof = {}
    for a in agents:
        # each agent has one weak spot to make routing matter
        weak = rng.choice(TASK_TYPES)
        prof[a] = {
            t: float(rng.uniform(lo - 0.13, lo) if t == weak else rng.uniform(lo, hi))
            for t in TASK_TYPES
        }
    return prof
