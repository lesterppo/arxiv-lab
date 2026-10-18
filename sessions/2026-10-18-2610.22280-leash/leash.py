"""LEASH: Uncertainty-Aware Weak-to-Strong Escalation (2610.22280) — CPU sim.

Faithful reimplementation of the paper's three-layer mechanism, tested against
a synthetic weak/strong agent simulator:

  L1 — initial task uncertainty -> early routing (weak vs strong agent)
  L2 — step-level uncertainty  -> runtime escalation (weak hands off mid-task)
  L3 — shadow observer          -> continued oversight of the strong agent

Uncertainty signal (paper's core, training-free, no extra forward pass):
compare the weak model's current hidden state against states from its PAST
successful and failed trajectories; a state that looks more like past
failures => higher uncertainty:

    u(h) = cos(h, fail_centroid) - cos(h, success_centroid)   (calibrated to [0,1])

Test design: synthetic multi-step tasks where a latent "trouble" process
drives both outcomes and hidden-state distributions. The estimator is fit on
past trajectories only; escalation policies are evaluated on held-out tasks
against a RANDOM-escalation baseline at MATCHED escalation rates (the paper's
headline comparison: +6.55-11.94 pts over random at matched rates on OSWorld).

stdlib + numpy only.
"""
import numpy as np


# ---------------------------------------------------------------- simulator
# One canonical latent geometry shared by every simulator instance, so an
# uncertainty estimator fit on past trajectories stays valid across tasks
# and paired comparisons (per-task seeds) remain meaningful.
def _canonical_centroids(dim=64):
    rng = np.random.default_rng(0)
    a = rng.normal(size=dim); a /= np.linalg.norm(a)
    b = rng.normal(size=dim); b -= (b @ a) * a; b /= np.linalg.norm(b)
    c_succ = 0.9 * a + 0.1 * b
    c_fail = 0.1 * a + 0.9 * b
    return c_succ / np.linalg.norm(c_succ), c_fail / np.linalg.norm(c_fail)


_CANON = {}


def canonical_centroids(dim=64):
    if dim not in _CANON:
        _CANON[dim] = _canonical_centroids(dim)
    return _CANON[dim]


class TaskSim:
    """Multi-step task environment with a weak and a strong agent.

    Latent trouble process: at each step the weak agent may enter 'trouble';
    trouble drives BOTH the hidden-state distribution (toward the failure
    centroid) and the per-step failure probability. The strong agent is
    uniformly more capable but still degrades in trouble.

    Regime (motivated by the paper's OSWorld numbers: weak-alone moderate,
    strong clearly better): weak succeeds on easy tasks and fails on hard
    ones, so *which* tasks get escalated is the discriminating decision.
    """

    def __init__(self, seed=0, n_steps=6, dim=64, noise=0.25):
        self.rng = np.random.default_rng(seed)
        self.n_steps = n_steps
        self.dim = dim
        self.noise = noise
        # canonical latent geometry, shared across instances
        self.c_succ, self.c_fail = canonical_centroids(dim)

    # -- latent dynamics ----------------------------------------------------
    def sample_task(self):
        difficulty = float(self.rng.uniform(0.05, 0.95))
        # noisy observable proxy of difficulty (what L1 sees)
        feat = np.clip(difficulty + self.rng.normal(0, 0.12), 0, 1)
        return {"difficulty": difficulty, "feat": feat}

    def _hidden(self, trouble, strong=False):
        c = self.c_fail if trouble else self.c_succ
        h = c + self.rng.normal(0, self.noise, self.dim)
        if strong:
            h += self.rng.normal(0, self.noise * 0.5, self.dim)
        return h / np.linalg.norm(h)

    def weak_rollout(self, task):
        """Full weak-agent trajectory: list of (hidden, trouble) + success."""
        d = task["difficulty"]
        steps, ok = [], True
        for _ in range(self.n_steps):
            trouble = self.rng.random() < (0.10 + 0.40 * d)
            # weak degrades in trouble (the escalation premise) but survives
            # easy tasks; strong stays robust (frontier-like rescue headroom)
            p_ok = 0.60 if trouble else 0.97
            if self.rng.random() > p_ok:
                ok = False
            steps.append((self._hidden(trouble), trouble))
        return steps, ok

    def strong_finish(self, task, from_step, intervene=False):
        """Strong agent completes remaining steps; returns (success, states).

        Frontier-like: rarely troubled, robust when troubled — the rescue
        headroom that makes escalation worthwhile (cf. paper: weak small
        open model vs Claude Opus 5)."""
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
        ok, _ = self.strong_finish(task, 0)
        return ok


# ------------------------------------------------------- uncertainty estimator
class UncertaintyEstimator:
    """u(h) = cos(h, fail) - cos(h, succ), fit on past trajectories."""

    def __init__(self):
        self.c_succ = None
        self.c_fail = None
        self._lo = self._hi = None

    def fit(self, bank):
        """bank: list of (states: [h...], success: bool)."""
        ss = [h for states, ok in bank if ok for h in states]
        fs = [h for states, ok in bank if not ok for h in states]
        assert ss and fs, "bank needs both outcomes"
        self.c_succ = np.mean(ss, axis=0); self.c_succ /= np.linalg.norm(self.c_succ)
        self.c_fail = np.mean(fs, axis=0); self.c_fail /= np.linalg.norm(self.c_fail)
        raw = [self.raw(h) for states, _ in bank for h in states]
        self._lo, self._hi = float(np.min(raw)), float(np.max(raw))

    def raw(self, h):
        return float(h @ self.c_fail - h @ self.c_succ)

    def __call__(self, h):
        u = (self.raw(h) - self._lo) / max(self._hi - self._lo, 1e-9)
        return float(np.clip(u, 0, 1))


# ------------------------------------------------------------- LEASH policy
class LeashPolicy:
    """Three-layer escalation controller.

    theta1: L1 task-uncertainty routing threshold (on task feature proxy).
    theta2: L2 step-uncertainty escalation threshold.
    l3:     enable shadow observer after handoff.
    """

    def __init__(self, est: UncertaintyEstimator, theta1=0.5, theta2=0.6, l3=True):
        self.est, self.theta1, self.theta2, self.l3 = est, theta1, theta2, l3

    def initial_uncertainty(self, task):
        # L1: cheap task-level proxy (paper: "initial task uncertainty")
        return float(np.clip(task["feat"], 0, 1))

    def run_task(self, sim: TaskSim, task):
        """Returns (success, escalated: bool, strong_steps: int)."""
        if self.initial_uncertainty(task) > self.theta1:
            # L1: route to strong immediately
            ok, _ = sim.strong_finish(task, 0)
            return ok, True, sim.n_steps
        # weak acts; L2 monitors each step
        steps, weak_ok = sim.weak_rollout(task)
        for t, (h, _) in enumerate(steps):
            if self.est(h) > self.theta2:
                # L2: hand off remaining steps to strong
                ok, sstates = sim.strong_finish(task, t + 1, intervene=False)
                if self.l3:
                    ok = self._shadow_observe(sim, task, t + 1, sstates, ok)
                return ok, True, sim.n_steps - (t + 1)
        return weak_ok, False, 0

    def _shadow_observe(self, sim: TaskSim, task, from_step, sstates, ok):
        """L3: after handoff, keep watching the strong agent's states; if
        uncertainty stays high for 2 consecutive steps, apply one corrective
        intervention (re-run the remainder with the nudge)."""
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


# ------------------------------------------------------- evaluation harness
def evaluate(sim, policy, tasks, rng):
    succ, esc, ssteps = 0, 0, 0
    for task in tasks:
        ok, e, ss = policy.run_task(sim, task)
        succ += ok; esc += e; ssteps += ss
    n = len(tasks)
    return {"success": succ / n, "escalation_rate": esc / n,
            "mean_strong_steps": ssteps / n, "n": n}


def evaluate_random(sim, tasks, target_rate, rng):
    """Random escalation baseline at (approximately) the matched rate."""
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


def auc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores)
    s, l = scores[order], labels[order]
    n1, n0 = l.sum(), (~l).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = np.argsort(np.argsort(s)) + 1
    return float((ranks[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
