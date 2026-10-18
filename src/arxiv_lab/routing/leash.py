"""LEASH: uncertainty-aware weak-to-strong escalation (arXiv:2610.22280).

One-line claim: a weak agent escalates to a strong agent only when its
*latent uncertainty* is high — measured training-free, with no extra forward
pass, by comparing the current hidden state against states from its past
successful and failed trajectories — and this beats random escalation at
matched escalation rates. Three layers: (L1) initial task uncertainty routes
up front, (L2) step-level uncertainty escalates mid-task, (L3) a shadow
observer keeps watching the strong agent after handoff and intervenes on
sustained high uncertainty.

    u(h) = cos(h, fail_centroid) - cos(h, success_centroid), calibrated to [0, 1]

``TaskSim`` is the deterministic validation harness used for the CPU tests:
multi-step tasks with a latent trouble process driving both hidden-state
distributions and outcomes. Requires numpy (optional dependency).
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def _need_numpy():
    if np is None:
        raise ImportError(
            "arxiv_lab.routing.leash requires numpy (optional dependency); "
            "install it with `pip install numpy`"
        )


# One canonical latent geometry shared by every simulator instance, so an
# uncertainty estimator fit on past trajectories stays valid across tasks.
# Built lazily (numpy) and cached; the module imports without numpy.
_CANON = {}


def _canonical_centroids(dim=64):
    _need_numpy()
    rng = np.random.default_rng(0)
    a = rng.normal(size=dim); a /= np.linalg.norm(a)
    b = rng.normal(size=dim); b -= (b @ a) * a; b /= np.linalg.norm(b)
    c_succ = 0.9 * a + 0.1 * b
    c_fail = 0.1 * a + 0.9 * b
    return c_succ / np.linalg.norm(c_succ), c_fail / np.linalg.norm(c_fail)


def canonical_centroids(dim=64):
    """Shared (success, failure) latent centroids; deterministic in dim."""
    if dim not in _CANON:
        _CANON[dim] = _canonical_centroids(dim)
    return _CANON[dim]


class TaskSim:
    """Deterministic weak/strong-agent task simulator (validation harness).

    Each task has a difficulty in [0, 1]. A latent trouble process drives
    both the weak agent's hidden-state distribution (toward the failure
    centroid) and its per-step failure probability. The weak agent succeeds
    on easy tasks and fails on hard ones; the strong agent is frontier-like
    (rarely troubled, robust when troubled).
    """

    def __init__(self, seed=0, n_steps=6, dim=64, noise=0.25):
        _need_numpy()
        self.rng = np.random.default_rng(seed)
        self.n_steps = n_steps
        self.dim = dim
        self.noise = noise
        self.c_succ, self.c_fail = canonical_centroids(dim)

    def sample_task(self):
        _need_numpy()
        difficulty = float(self.rng.uniform(0.05, 0.95))
        feat = float(np.clip(difficulty + self.rng.normal(0, 0.12), 0, 1))
        return {"difficulty": difficulty, "feat": feat}

    def _hidden(self, trouble, strong=False):
        c = self.c_fail if trouble else self.c_succ
        h = c + self.rng.normal(0, self.noise, self.dim)
        if strong:
            h += self.rng.normal(0, self.noise * 0.5, self.dim)
        return h / np.linalg.norm(h)

    def weak_rollout(self, task):
        """Full weak-agent trajectory: (list of (hidden, trouble), success)."""
        _need_numpy()
        d = task["difficulty"]
        steps, ok = [], True
        for _ in range(self.n_steps):
            trouble = self.rng.random() < (0.10 + 0.40 * d)
            p_ok = 0.60 if trouble else 0.97
            if self.rng.random() > p_ok:
                ok = False
            steps.append((self._hidden(trouble), trouble))
        return steps, ok

    def strong_finish(self, task, from_step, intervene=False):
        """Strong agent completes steps [from_step, n_steps); (ok, states)."""
        _need_numpy()
        d = task["difficulty"]
        states = []
        for _ in range(from_step, self.n_steps):
            trouble = self.rng.random() < (0.03 + 0.10 * d)
            if intervene and trouble:
                trouble = False  # shadow-observer corrective nudge
            states.append(self._hidden(trouble, strong=True))
            p_ok = 0.95 if trouble else 0.995
            if self.rng.random() > p_ok:
                return False, states
        return True, states

    def strong_full(self, task):
        _need_numpy()
        ok, _ = self.strong_finish(task, 0)
        return ok


class UncertaintyEstimator:
    """u(h) = cos(h, fail) - cos(h, succ), fit on past trajectories.

    Fit on the weak agent's own history: states from failed trajectories
    form the failure centroid, states from successful ones the success
    centroid. No extra forward pass at decision time.
    """

    def __init__(self):
        self.c_succ = None
        self.c_fail = None
        self._lo = self._hi = None

    def fit(self, bank):
        """bank: iterable of (states: [h...], success: bool)."""
        _need_numpy()
        ss = [h for states, ok in bank if ok for h in states]
        fs = [h for states, ok in bank if not ok for h in states]
        if not ss or not fs:
            raise ValueError("bank needs both successful and failed trajectories")
        self.c_succ = np.mean(ss, axis=0); self.c_succ /= np.linalg.norm(self.c_succ)
        self.c_fail = np.mean(fs, axis=0); self.c_fail /= np.linalg.norm(self.c_fail)
        raw = [self.raw(h) for states, _ in bank for h in states]
        self._lo, self._hi = float(np.min(raw)), float(np.max(raw))
        return self

    def raw(self, h):
        _need_numpy()
        return float(h @ self.c_fail - h @ self.c_succ)

    def __call__(self, h):
        _need_numpy()
        u = (self.raw(h) - self._lo) / max(self._hi - self._lo, 1e-9)
        return float(np.clip(u, 0, 1))


class LeashPolicy:
    """Three-layer escalation controller.

    theta1: L1 task-uncertainty routing threshold (on the task feature proxy).
    theta2: L2 step-uncertainty escalation threshold.
    l3:     enable the L3 shadow observer after handoff.
    """

    def __init__(self, est: UncertaintyEstimator, theta1=0.5, theta2=0.6, l3=True):
        self.est, self.theta1, self.theta2, self.l3 = est, theta1, theta2, l3

    def initial_uncertainty(self, task):
        return float(np.clip(task["feat"], 0, 1)) if np is not None else 0.0

    def run_task(self, sim: TaskSim, task):
        """Returns (success, escalated: bool, strong_steps: int)."""
        _need_numpy()
        if self.initial_uncertainty(task) > self.theta1:
            ok, _ = sim.strong_finish(task, 0)          # L1: route up front
            return ok, True, sim.n_steps
        steps, weak_ok = sim.weak_rollout(task)        # weak acts, L2 watches
        for t, (h, _) in enumerate(steps):
            if self.est(h) > self.theta2:              # L2: escalate mid-task
                ok, sstates = sim.strong_finish(task, t + 1, intervene=False)
                if self.l3:
                    ok = self._shadow_observe(sim, task, t + 1, sstates, ok)
                return ok, True, sim.n_steps - (t + 1)
        return weak_ok, False, 0

    def _shadow_observe(self, sim: TaskSim, task, from_step, sstates, ok):
        """L3: after handoff, keep watching the strong agent; on 2+
        consecutive high-uncertainty steps, intervene once (re-run the
        remainder with the corrective nudge)."""
        _need_numpy()
        if ok:
            return True
        highs = 0
        for h in sstates:
            if self.est(h) > self.theta2:
                highs += 1
                if highs >= 2:
                    ok2, _ = sim.strong_finish(task, from_step, intervene=True)
                    return ok2
            else:
                highs = 0
        return False


def auc(scores, labels):
    """Binary AUC of scores against boolean labels."""
    _need_numpy()
    scores = np.asarray(scores); labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores)
    s, l = scores[order], labels[order]
    n1, n0 = int(l.sum()), int((~l).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = np.argsort(np.argsort(s)) + 1
    return float((ranks[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(sim, policy, tasks, rng):
    """Run a policy over tasks; returns success/escalation aggregates."""
    _need_numpy()
    succ, esc, ssteps = 0, 0, 0
    for task in tasks:
        ok, e, ss = policy.run_task(sim, task)
        succ += ok; esc += e; ssteps += ss
    n = len(tasks)
    return {"success": succ / n, "escalation_rate": esc / n,
            "mean_strong_steps": ssteps / n, "n": n}


def evaluate_random(sim, tasks, target_rate, rng):
    """Random-escalation baseline at (approximately) the target rate."""
    _need_numpy()
    succ, esc, ssteps = 0, 0, 0
    for task in tasks:
        if rng.random() < target_rate:
            ok = sim.strong_full(task); e, ss = True, sim.n_steps
        else:
            _, ok = sim.weak_rollout(task); e, ss = False, 0
        succ += ok; esc += e; ssteps += ss
    n = len(tasks)
    return {"success": succ / n, "escalation_rate": esc / n,
            "mean_strong_steps": ssteps / n, "n": n}
