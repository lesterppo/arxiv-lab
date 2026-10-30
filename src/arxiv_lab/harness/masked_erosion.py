"""Masked erosion of redundant safeguards under acceptance gates
(arXiv:2610.38392).

The paper's claim: self-improving agents edit their harness and keep an
edit only if an acceptance gate scores it higher on a batch of tasks.
When safeguards overlap, deleting one is *masked*: the remaining
safeguards still block the harm, so the gate decides on task success
alone and redundancy erodes unseen. The paper measures harm for every
subset of four written safeguards in two agent environments, runs 411
improvement loops under ten gate rules (mostly controlled deletion
experiments, plus three GEPA optimizer runs), and reports: a deletion
behind another working safeguard adds almost no observable harm; a
paired significance test blocks deletions that pass by chance but not
deletions that genuinely raise task success; a larger evaluation batch
lets the gate delete cheaper safeguards (20-task batches kept every
harness protected while 40-task batches lost all six, at the same
cost); large harm audits prevent exposure but not the loss of depth,
which turns into harm when the harness is reused with a weaker model;
freezing safety text or auditing one-layer ablations preserves depth.

This module reproduces the deletion-experiment core deterministically
on synthetic task banks: four safeguards, each harm channel covered by
exactly two safeguards (depth 2), two environments, and five gate
rules (mean gate, paired significance test, harm audit, safety-text
freeze, one-layer ablation audit). Common random numbers (one fixed
task bank per seed, paired before/after evaluation) make the gate
comparison paired. The GEPA optimizer runs and the paper's LLM
environments are NOT reproduced (no live model here); the frozen
statistical model is an analytic paired-test power prediction computed
from the fixed parameters only, never fitted to the run. Pure stdlib.
"""

import math
import random

CANONICAL_SEED = 20261030
N_SEEDS = 20
ROBUSTNESS_SEEDS = (1, 7, 42)

# Four safeguards S1..S4; harm channel c is covered by safeguards
# {c, (c+1) % 4}, so every channel starts at depth 2 and every
# safeguard covers two channels.
N_SAFEGUARDS = 4
N_CHANNELS = 4
COVERAGE = tuple(tuple(sorted({c, (c + 1) % N_SAFEGUARDS}))
                 for c in range(N_CHANNELS))

# Friction (task-success cost of keeping the safeguard) per safeguard,
# as a success-probability shift on friction-sensitive tasks. Heterogeneous
# on purpose: the paper's "cheaper safeguards" are the low-friction ones
# a larger batch can newly detect. Fixed a priori; see the pilot note in
# the session README for the calibration check.
FRICTION = (0.02, 0.04, 0.06, 0.08)

BASE_SUCCESS = 0.80
FRICTION_TASK_PROB = 0.5
RISKY_TASK_PROB = 0.35
# Residual harm probability when at least one covering safeguard blocks.
RESIDUAL_HARM = 0.05
# Harm probability when a risky task's channel has no working cover.
EXPOSED_HARM = {"coding": 0.60, "web": 0.40}
# Weaker-model reuse: each present safeguard independently fails to
# block with this probability (the strong-model evaluation uses 0.0).
WEAK_FAILURE_RATE = 0.30

# Improvement-loop shape: each round proposes every safeguard once, in
# random order; 6 rounds = 24 proposals per run.
N_ROUNDS = 6
BATCH_SIZES = (10, 20, 40, 80)
# One-sided paired-test critical value (normal approximation).
Z_CRIT = 1.645
# Audit thresholds (harm-increase tolerances on the audit bank).
AUDIT_HARM_TOL = 0.005
ABLATION_HARM_TOL = 0.01
AUDIT_BANK_SIZE = 2000


def generate_tasks(seed, n, env):
    """One fixed task bank: paired evaluation reuses the same tasks.

    Each task carries: risky channel (-1 = not risky), a friction mask
    (which safeguards slow this task), a success uniform, a harm
    uniform, and one failure uniform per safeguard (used only when a
    failure rate is injected, so normal and reuse evaluations are
    paired on the same draws).
    """
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        risky = -1
        if rng.random() < RISKY_TASK_PROB:
            risky = rng.randrange(N_CHANNELS)
        friction = tuple(rng.random() < FRICTION_TASK_PROB
                         for _ in range(N_SAFEGUARDS))
        tasks.append({
            "risky": risky,
            "friction": friction,
            "u_success": rng.random(),
            # Second independent success draw: chance-gain deletions
            # (zero systematic friction) are evaluated with fresh
            # sampling noise, as in the paper's stochastic LLM
            # evaluations; genuine deletions keep the paired draw.
            "u_success2": rng.random(),
            "u_harm": rng.random(),
            "u_fail": tuple(rng.random() for _ in range(N_SAFEGUARDS)),
        })
    return tasks


def _task_outcome(task, present, env, failure_rate, second_draw=False):
    """(success, harm) for one task under a safeguard bitmask."""
    p_success = BASE_SUCCESS
    for i in range(N_SAFEGUARDS):
        if present & (1 << i) and task["friction"][i]:
            p_success -= FRICTION[i]
    u = task["u_success2"] if second_draw else task["u_success"]
    success = u < p_success
    harm = False
    risky = task["risky"]
    if risky >= 0:
        covering = [i for i in COVERAGE[risky] if present & (1 << i)]
        working = [i for i in covering
                   if task["u_fail"][i] >= failure_rate]
        if not working:
            harm = task["u_harm"] < EXPOSED_HARM[env]
        else:
            harm = task["u_harm"] < RESIDUAL_HARM
    return success, harm


def evaluate(present, tasks, env, failure_rate=0.0):
    """Mean success and mean harm over a task bank."""
    s = h = 0
    for t in tasks:
        ok, hm = _task_outcome(t, present, env, failure_rate)
        s += ok
        h += hm
    n = len(tasks)
    return s / n, h / n


def paired_diffs(before_present, after_present, tasks, env, paired=True):
    """Per-task success differences (after - before).

    Paired (genuine deletions): the same success draw decides both
    evaluations, so only the friction change can flip an outcome.
    Unpaired (chance deletions, ``paired=False``): the after-evaluation
    draws fresh success noise, so a zero-effect deletion shows pure
    sampling variation around a zero mean.
    """
    diffs = []
    for t in tasks:
        b, _ = _task_outcome(t, before_present, env, 0.0)
        a, _ = _task_outcome(t, after_present, env, 0.0,
                             second_draw=not paired)
        diffs.append((1 if a else 0) - (1 if b else 0))
    return diffs


def paired_test_accept(diffs, z_crit=Z_CRIT):
    """One-sided paired test: accept iff the mean success gain is
    significantly positive. With zero variance, accept iff mean > 0."""
    n = len(diffs)
    mean = sum(diffs) / n
    if mean <= 0:
        return False
    var = sum((d - mean) ** 2 for d in diffs) / n
    if var <= 0:
        return True
    return mean / (math.sqrt(var / n)) > z_crit


def min_coverage(present):
    """Minimum safeguard coverage over harm channels (0, 1, or 2)."""
    return min(sum(1 for i in COVERAGE[c] if present & (1 << i))
               for c in range(N_CHANNELS))


def gate_decide(gate, present, deleted, tasks, env, audit_tasks,
              genuine=True):
    """Accept/reject a proposed deletion under one gate rule.

    All success-based gates use the same batch; the audit gates add a
    harm check, so differences isolate the audit. The mean gate accepts
    on any positive mean gain (no significance requirement); the paired
    gate requires a significant gain.
    """
    after = present & ~(1 << deleted)
    if gate == "freeze":
        return False
    diffs = paired_diffs(present, after, tasks, env, paired=genuine)
    if gate == "mean":
        if sum(diffs) / len(diffs) <= 0:
            return False
    elif not paired_test_accept(diffs):
        return False
    if gate in ("mean", "paired"):
        return True
    _, harm_before = evaluate(present, audit_tasks, env, 0.0)
    if gate == "harm_audit":
        _, harm_after = evaluate(after, audit_tasks, env, 0.0)
        return harm_after - harm_before <= AUDIT_HARM_TOL
    if gate == "ablation_audit":
        # One-layer ablation: force each other safeguard to fail in
        # turn; the deletion must not raise harm under any single
        # failure. (The deleted safeguard itself is gone either way.)
        for failed in range(N_SAFEGUARDS):
            if failed == deleted:
                continue
            hb = _harm_under_forced_failure(present, failed, audit_tasks, env)
            ha = _harm_under_forced_failure(after, failed, audit_tasks, env)
            if ha - hb > ABLATION_HARM_TOL:
                return False
        return True
    raise ValueError("unknown gate: %r" % (gate,))


def _harm_under_forced_failure(present, failed, tasks, env):
    """Mean harm when safeguard `failed` never blocks (others normal)."""
    h = 0
    for t in tasks:
        risky = t["risky"]
        if risky < 0:
            continue
        covering = [i for i in COVERAGE[risky]
                    if (present & (1 << i)) and i != failed]
        if not covering:
            harm = t["u_harm"] < EXPOSED_HARM[env]
        else:
            harm = t["u_harm"] < RESIDUAL_HARM
        h += harm
    return h / len(tasks)


def deletion_experiment(seed, env):
    """Masked-deletion measurement: harm change when each safeguard is
    deleted alone (others working) vs deleted with one covering
    safeguard failed. Returns {safeguard: (masked_delta, exposed_delta)}.
    """
    tasks = generate_tasks(seed, 4000, env)
    full = (1 << N_SAFEGUARDS) - 1
    _, harm_full = evaluate(full, tasks, env, 0.0)
    out = {}
    for deleted in range(N_SAFEGUARDS):
        after = full & ~(1 << deleted)
        _, harm_masked = evaluate(after, tasks, env, 0.0)
        # Exposed: a covering partner of the deleted safeguard fails.
        partner = COVERAGE[deleted][1] if COVERAGE[deleted][0] == deleted \
            else COVERAGE[deleted][0]
        harm_exposed = _harm_under_forced_failure(after, partner, tasks, env)
        harm_exposed_full = _harm_under_forced_failure(full, partner, tasks,
                                                        env)
        out[deleted] = (harm_masked - harm_full,
                        harm_exposed - harm_exposed_full)
    return out


def run_improvement_loop(seed, gate, batch_size, env, genuine=True,
                         n_rounds=N_ROUNDS):
    """One self-improvement run: proposals accumulate in the harness.

    `genuine=False` zeroes every safeguard's friction (chance gains
    only). Returns the run summary dict.
    """
    global FRICTION
    rng = random.Random(seed * 7919 + batch_size)
    bank = generate_tasks(seed, 4000, env)
    audit_tasks = generate_tasks(seed + 500, AUDIT_BANK_SIZE, env)
    saved_friction = FRICTION
    if not genuine:
        FRICTION = (0.0,) * N_SAFEGUARDS
    try:
        present = (1 << N_SAFEGUARDS) - 1
        accepted = 0
        accepted_by_safeguard = [0] * N_SAFEGUARDS
        for _ in range(n_rounds):
            order = list(range(N_SAFEGUARDS))
            rng.shuffle(order)
            for deleted in order:
                if not (present & (1 << deleted)):
                    continue
                # Fresh batch per proposal, as in the paper's loops.
                batch = rng.sample(bank, batch_size)
                if gate_decide(gate, present, deleted, batch, env,
                               audit_tasks, genuine=genuine):
                    present &= ~(1 << deleted)
                    accepted += 1
                    accepted_by_safeguard[deleted] += 1
    finally:
        FRICTION = saved_friction
    # Reuse evaluation on fresh tasks, strong vs weaker model.
    fresh = generate_tasks(seed + 1000, 4000, env)
    full = (1 << N_SAFEGUARDS) - 1
    succ_full, harm_full = evaluate(full, fresh, env, 0.0)
    succ_final, harm_final = evaluate(present, fresh, env, 0.0)
    _, harm_full_weak = evaluate(full, fresh, env, WEAK_FAILURE_RATE)
    _, harm_final_weak = evaluate(present, fresh, env, WEAK_FAILURE_RATE)
    return {
        "gate": gate, "batch_size": batch_size, "env": env,
        "genuine": genuine, "accepted": accepted,
        "accepted_by_safeguard": accepted_by_safeguard,
        "present": present, "min_coverage": min_coverage(present),
        "depth_lost": min_coverage(present) < 2,
        "fully_exposed": min_coverage(present) == 0,
        "success_gain": succ_final - succ_full,
        "harm_strong": harm_final - harm_full,
        "harm_weak": harm_final_weak - harm_full_weak,
    }


def analytic_acceptance(friction, batch_size):
    """Frozen statistical-model prediction: paired-test acceptance
    probability for a genuine deletion, from the fixed parameters only.

    Genuine paired diffs are +1 exactly when a task flips (probability
    p_flip = FRICTION_TASK_PROB * friction) and 0 otherwise, so with k
    flips in N tasks the paired t-statistic is sqrt(k / (1 - k/N)).
    It exceeds Z_CRIT = 1.645 iff k >= 3 (for every batch size used
    here), hence acceptance is P(Bin(N, p_flip) >= 3) exactly. Never
    fitted to simulated outcomes. Chance deletions sit at the nominal
    one-sided size, 0.05.
    """
    from math import comb
    p_flip = FRICTION_TASK_PROB * friction
    if p_flip <= 0:
        return 0.05
    n = batch_size
    return 1.0 - sum(comb(n, k) * p_flip ** k * (1 - p_flip) ** (n - k)
                     for k in range(3))


def run_experiment(seeds=range(N_SEEDS), canonical_seed=CANONICAL_SEED):
    """Canonical run: deletion measurements, gate sweep, batch sweep."""
    results = {"deletion": {}, "gates": [], "batch_sweep": [],
               "analytic": {}}
    for env in ("coding", "web"):
        dels = [deletion_experiment(canonical_seed + k, env)
                for k in range(5)]
        results["deletion"][env] = {
            i: (sum(d[i][0] for d in dels) / len(dels),
                sum(d[i][1] for d in dels) / len(dels))
            for i in range(N_SAFEGUARDS)
        }
    gates = ("mean", "paired", "harm_audit", "ablation_audit", "freeze")
    for gate in gates:
        for env in ("coding", "web"):
            for genuine in (True, False):
                runs = [run_improvement_loop(canonical_seed + k, gate, 20,
                                             env, genuine)
                        for k in seeds]
                results["gates"].append(_summarize(gate, 20, env, genuine,
                                                    runs))
    for n in BATCH_SIZES:
        for env in ("coding", "web"):
            runs = [run_improvement_loop(canonical_seed + k, "paired", n,
                                         env, True) for k in seeds]
            results["batch_sweep"].append(_summarize("paired", n, env, True,
                                                      runs))
    # Sensitivity: 3x friction, N=20, coding env only. At the primary
    # friction scale genuine gains are mostly undetectable at N=20
    # (fewer than 3 flips per batch), so the paired test cannot
    # discriminate genuine from chance there; the high-friction arm
    # checks whether discrimination appears once gains are detectable.
    global FRICTION
    saved = FRICTION
    FRICTION = tuple(3 * f for f in saved)
    try:
        results["sensitivity"] = []
        for gate in ("paired", "mean"):
            for genuine in (True, False):
                runs = [run_improvement_loop(canonical_seed + k, gate, 20,
                                             "coding", genuine)
                        for k in seeds]
                row = _summarize(gate, 20, "coding", genuine, runs)
                row["friction_scale"] = 3
                results["sensitivity"].append(row)
    finally:
        FRICTION = saved
    for i, fr in enumerate(FRICTION):
        for n in BATCH_SIZES:
            results["analytic"][(i, n)] = analytic_acceptance(fr, n)
    return results


def _summarize(gate, batch_size, env, genuine, runs):
    n = len(runs)
    return {
        "gate": gate, "batch_size": batch_size, "env": env,
        "genuine": genuine, "n_runs": n,
        "mean_accepted": sum(r["accepted"] for r in runs) / n,
        "frac_depth_lost": sum(r["depth_lost"] for r in runs) / n,
        "frac_fully_exposed": sum(r["fully_exposed"] for r in runs) / n,
        "mean_success_gain": sum(r["success_gain"] for r in runs) / n,
        "mean_harm_strong": sum(r["harm_strong"] for r in runs) / n,
        "mean_harm_weak": sum(r["harm_weak"] for r in runs) / n,
        "accepted_by_safeguard": [
            sum(r["accepted_by_safeguard"][i] for r in runs) / n
            for i in range(N_SAFEGUARDS)],
    }


if __name__ == "__main__":
    import json
    res = run_experiment()
    print(json.dumps(res, indent=1, default=str))
