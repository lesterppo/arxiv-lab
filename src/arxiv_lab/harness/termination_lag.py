#!/usr/bin/env python3
"""Termination lag in coding agents (arXiv:2611.14960).

Paper claim: coding agents solve a task well before they stop working on it.
Termination lag = steps between an agent's first correct patch and the end
of its run. Measured on SWE-bench: 38-82% of successful runs achieve a
correct solution on their initial source-code edit; post-solution steps are
a median 35-37% (Qwen3-Coder) to 54-60% (Laguna-XS.2) of successful runs;
agents spend most post-solution steps verifying their work and exploring
the repository; in 1-8% of cases continued activity undoes the earlier
correct patch, leaving the final solution incorrect.

Miniature: synthetic SWE-style tasks. A deterministic simulated agent
explores, then edits a one-bug repo; a test-suite verifier checks any patch
(1 step per check). Three stop policies are compared on the *same* tasks
with paired RNG streams, so the pre-solution phase is identical across
policies and only the post-solution behaviour differs:
  - "continue"     : the paper-observed behaviour -- after the first correct
                     patch, keep going for extra verification / exploration /
                     re-edit steps, then stop.
  - "stop_on_pass" : run the test suite once after the first correct patch
                     and stop on a pass (lag = 1 confirmation step).
  - "fixed_budget" : run a fixed number of steps regardless of state.
Stdlib only, deterministic (SplitMix64, same construction as wikitrace so
the Rust second stage reproduces the exact streams).
"""

import math

# ---------------------------------------------------------------- RNG
MASK64 = (1 << 64) - 1


class RNG:
    """SplitMix64. `next_u64` returns the next 64-bit draw."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        # 53-bit precision float in [0, 1)
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def bernoulli(self, p):
        return self.uniform() < p

    def randint(self, lo, hi):
        # inclusive lo, exclusive hi
        return lo + (self.next_u64() % (hi - lo))

    def poisson(self, lam):
        # Knuth's algorithm; fine for the small lambdas used here.
        limit = math.exp(-lam)
        k = 0
        prod = 1.0
        while True:
            prod *= self.uniform()
            if prod <= limit:
                return k
            k += 1


# ---------------------------------------------------------------- tasks
# First-edit solve probability per difficulty. Mean over the 1/3-1/3-1/3
# mix is 0.60, inside the paper's 38-82% band.
P_FIRST = {"easy": 0.82, "medium": 0.60, "hard": 0.38}
P_REFIX = 0.55          # prob a later edit fixes the bug (before solved)
P_UNDO = 0.15           # prob a post-solution re-edit undoes the fix
MAX_EDITS = 6           # agent gives up after this many failed edits
MAX_STEPS = 20          # hard cap on run length


class Task:
    __slots__ = ("idx", "difficulty")

    def __init__(self, idx, difficulty):
        self.idx = idx
        self.difficulty = difficulty


def gen_tasks(n, seed):
    rng = RNG(seed)
    diffs = ["easy", "medium", "hard"]
    return [Task(i, diffs[rng.randint(0, 3)]) for i in range(n)]


# ---------------------------------------------------------------- verifier
def verify(patch_correct):
    """The test suite: deterministic, passes iff the current patch is correct."""
    return bool(patch_correct)


# ---------------------------------------------------------------- agent run
class RunResult:
    __slots__ = ("task", "policy", "steps", "first_correct_step",
                 "first_edit_solved", "final_correct", "lag", "post_share",
                 "undo_to_incorrect", "n_verify", "n_explore", "n_reedit",
                 "n_undos", "solved")

    def __init__(self, task, policy):
        self.task = task
        self.policy = policy
        self.steps = 0
        self.first_correct_step = None
        self.first_edit_solved = False
        self.final_correct = False
        self.lag = 0
        self.post_share = 0.0
        self.undo_to_incorrect = False
        self.n_verify = 0       # post-solution counts only
        self.n_explore = 0
        self.n_reedit = 0
        self.n_undos = 0
        self.solved = False


def run_task(task, policy, rng, mean_extra=3.0, p_undo=P_UNDO,
             fixed_budget=10, mean_pre_explore=1.0):
    """Simulate one agent run. Returns a RunResult.

    Step accounting: every EDIT / VERIFY / EXPLORE is one step. `lag` is the
    number of steps after the first correct patch (paper definition).
    """
    res = RunResult(task, policy)
    patch_correct = False
    step = 0
    edit_no = 0

    def do_edit():
        nonlocal patch_correct, step, edit_no
        step += 1
        edit_no += 1
        if patch_correct:
            # re-edit of an already-correct patch: may undo the fix
            if rng.bernoulli(p_undo):
                patch_correct = False
                res.n_undos += 1
        elif edit_no == 1:
            if rng.bernoulli(P_FIRST[task.difficulty]):
                patch_correct = True
                res.first_correct_step = step
                res.first_edit_solved = True
        else:
            if rng.bernoulli(P_REFIX):
                patch_correct = True
                if res.first_correct_step is None:
                    res.first_correct_step = step
        return patch_correct

    def do_verify():
        nonlocal step
        step += 1
        res.n_verify += 1
        return verify(patch_correct)

    def do_explore():
        nonlocal step
        step += 1
        res.n_explore += 1

    # --- pre-solution phase: explore the repo, then edit until the first
    # correct patch or give up. Counters are snapshotted at the solve point
    # so the reported composition is post-solution only.
    for _ in range(1 + rng.poisson(mean_pre_explore)):
        if step >= MAX_STEPS:
            break
        step += 1  # pre-solution exploration (not counted in post shares)
    edits = 0
    while edits < MAX_EDITS and step < MAX_STEPS:
        do_edit()
        edits += 1
        if patch_correct:
            break
    if not patch_correct:
        res.steps = step
        res.final_correct = False
        res.solved = False
        return res
    res.solved = True
    snap_v, snap_e, snap_r = res.n_verify, res.n_explore, 0
    # (edits are not counted in n_reedit; re-edits are post-solution only,
    # so count them from here on by tracking edits after the snapshot)
    post_edits = 0

    def do_reedit():
        nonlocal post_edits
        post_edits += 1
        return do_edit()

    # --- post-solution phase, by policy.
    if policy == "stop_on_pass":
        # One confirmation test, then stop.
        assert do_verify()
    elif policy == "fixed_budget":
        while step < fixed_budget and step < MAX_STEPS:
            r = rng.uniform()
            if r < 0.55:
                do_verify()
            elif r < 0.80:
                do_explore()
            else:
                do_reedit()
    elif policy == "continue":
        extra = 1 + rng.poisson(mean_extra)
        for _ in range(extra):
            if step >= MAX_STEPS:
                break
            r = rng.uniform()
            if r < 0.55:          # verify work
                do_verify()
            elif r < 0.80:        # explore repo
                do_explore()
            else:                 # re-edit: may undo, may re-fix later
                do_reedit()
    else:
        raise ValueError(f"unknown policy {policy!r}")

    res.steps = step
    res.final_correct = patch_correct
    res.lag = step - res.first_correct_step
    res.post_share = res.lag / step if step else 0.0
    res.undo_to_incorrect = (not patch_correct)
    res.n_verify -= snap_v
    res.n_explore -= snap_e
    res.n_reedit = post_edits
    return res


# ---------------------------------------------------------------- experiment
def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def run_experiment(policy, tasks, seed, **kw):
    """Run `policy` on all tasks with paired per-task RNG streams.

    The RNG is re-seeded per (task, policy) from (seed, task.idx), so the
    pre-solution draws are identical across policies: any difference in the
    metrics is caused by the stop policy alone.
    """
    runs = []
    for t in tasks:
        rng = RNG((seed ^ (t.idx * 0x9E3779B97F4A7C15)) & MASK64)
        runs.append(run_task(t, policy, rng, **kw))
    solved = [r for r in runs if r.solved]
    n = len(runs)
    first_edit_solved = sum(1 for r in solved if r.first_edit_solved)
    total_lag = sum(r.lag for r in solved)
    return {
        "policy": policy,
        "n_tasks": n,
        "n_solved": len(solved),
        "success_rate": len(solved) / n,
        "first_edit_solve_share": (first_edit_solved / len(solved)) if solved else 0.0,
        "mean_lag": sum(r.lag for r in solved) / len(solved) if solved else 0.0,
        "median_lag": _median([r.lag for r in solved]),
        "median_post_share": _median([r.post_share for r in solved]),
        "mean_post_share": sum(r.post_share for r in solved) / len(solved) if solved else 0.0,
        "final_correct_rate": sum(1 for r in solved if r.final_correct) / len(solved) if solved else 0.0,
        "undo_to_incorrect_rate": sum(1 for r in solved if r.undo_to_incorrect) / len(solved) if solved else 0.0,
        "mean_undos": sum(r.n_undos for r in solved) / len(solved) if solved else 0.0,
        "mean_total_steps": sum(r.steps for r in runs) / n,
        "post_verify_share": (sum(r.n_verify for r in solved) / total_lag) if total_lag else 0.0,
        "post_explore_share": (sum(r.n_explore for r in solved) / total_lag) if total_lag else 0.0,
        "post_reedit_share": (sum(r.n_reedit for r in solved) / total_lag) if total_lag else 0.0,
    }


def compare_policies(tasks, seed, **kw):
    return {p: run_experiment(p, tasks, seed, **kw)
            for p in ("continue", "stop_on_pass", "fixed_budget")}
