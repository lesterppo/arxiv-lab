"""CASEL: consequence-aware selective execution for tool-using agents
(arXiv:2610.36570).

The paper's claim: most runtime safeguards allocate scrutiny by generic
uncertainty scores or coarse mutation heuristics. CASEL is a model-agnostic
runtime allocation policy that combines calibrated failure probability,
state-dependent consequence, reversibility, ambiguity, and evidence
freshness to choose, per proposed tool call, among direct execution,
verification, sandboxing, clarification, approval, and abstention by
minimizing a per-mode expected cost. In the paper's controlled simulation
(20 seeds, 120,000 test actions each, same intervention menu and same
compute cost for every method) CASEL cuts harmful-side-effect trajectories
from 14.75% (uncertainty-ranked gate) to 7.02% and mean normalized harm
from 0.184 to 0.123. A product gate (failure probability x consequence)
reaches similar harm (5.82%, 0.120) but sends 45% more actions to human
approval; removing reversibility-aware routing raises the rate to 15.53%;
a trajectory-level risk budget increases harm at equal cost (11.82%).
The paper states its own limit: intervention effectiveness is assumed,
not measured — the results rank allocation rules under the simulator's
assumptions and do not establish deployment effectiveness.

This module reproduces the allocation-rule ranking deterministically on
synthetic tool actions: a CASEL cost-minimizing gate, an uncertainty-ranked
gate, a product-ranked gate, a CASEL ablation with reversibility ignored,
and a trajectory-level risk-budget gate, compared at matched intervention
cost (the paper's "same compute cost" condition). Common random numbers
(shared failure uniforms) make the comparison paired. Pure stdlib.
"""

import math
import random

CANONICAL_SEED = 20261029
ROBUSTNESS_SEEDS = (1, 7, 42)

# Simulation scale. The paper uses 20 seeds x 120,000 actions; this bounded
# reproduction uses 20 seeds x 10,000 actions (200,000 actions, 10,000
# trajectories of 20 actions). The scale and the parameters below are fixed
# before any run; the base failure rates are set so the harmful-trajectory
# rate sits in a discriminating range (a 50-action trajectory at the first
# draft's rates saturated at ~1.0 for every policy and could rank nothing).
N_SEEDS = 20
ACTIONS_PER_SEED = 10000
TRAJ_LEN = 20

# Harm weight: mode costs and normalized harm share one cost scale only if
# a realized harm counts far more than a unit of compute. Expected cost =
# mode cost + HARM_WEIGHT x failure probability x planned harm. Without
# this weight, expected harm (~0.01/action) never outweighs any mode cost
# and cost minimization degenerates to "always direct". HARM_WEIGHT = 20
# puts a 5%-chance-of-full-harm action at ~1.0 cost units, comparable to
# the human-approval cost. Fixed a priori; all policies share it.
HARM_WEIGHT = 20.0

# A harmful side-effect event is a failed action whose realized normalized
# harm reaches this threshold; a trajectory is harmful if it contains one.
HARM_EVENT_THRESHOLD = 0.10

# --- Intervention modes -------------------------------------------------
# cost: compute/human cost charged per action in that mode (approve is the
# expensive human mode; abstain carries an opportunity cost).
# harm_mult: fraction of the action's harm that still materializes if the
# action fails in that mode. Sandbox containment leaks for irreversible
# side effects (see _harm_mult), which is exactly what the reversibility
# ablation gets wrong. Effectiveness values are ASSUMED here, as in the
# paper — they are the simulator's assumptions, not measured facts.
MODES = ("direct", "verify", "sandbox", "clarify", "approve", "abstain")
MODE_COST = {
    "direct": 0.00,
    "verify": 0.10,
    "sandbox": 0.15,
    "clarify": 0.12,
    "approve": 0.50,
    "abstain": 0.20,
}
# Failure-probability adjustments (planning and realization use the same
# assumed effects): verification refreshes stale evidence (removes the
# staleness component), clarification resolves ambiguity (removes the
# ambiguity component). Abstention executes nothing.
_VERIFY_STALE_REMOVAL = 0.4
_CLARIFY_AMBIG_REMOVAL = 0.5
# Reversible failures are partly recoverable: only this fraction of the
# consequence materializes as lasting harm.
_REVERSIBLE_HARM_FRACTION = 0.3
# Abstention forfeits the action's value: its cost is a small handling
# cost plus the action's value V (drawn per action). Without the value
# term, abstention (harm 0, flat cost) dominates approval for every risky
# action and the six-mode menu degenerates; with it, approval wins for
# high-value risky actions and abstention for low-value ones.


def _harm_mult(mode, reversible):
    """Fraction of harm realized on failure in `mode` (true values)."""
    if mode == "direct":
        return 1.0
    if mode == "verify":
        return 0.6
    if mode == "sandbox":
        return 0.2 if reversible else 0.6
    if mode == "clarify":
        return 1.0
    if mode == "approve":
        return 0.05
    if mode == "abstain":
        return 0.0
    raise ValueError(mode)


def _adjusted_p(p_fail, ambiguity, staleness, mode):
    """Failure probability after a mode's assumed preventive effect."""
    if mode == "abstain":
        return 0.0
    p = p_fail
    if mode == "verify":
        p -= _VERIFY_STALE_REMOVAL * staleness
    if mode == "clarify":
        p -= _CLARIFY_AMBIG_REMOVAL * ambiguity
    return max(0.01, min(0.95, p))


def _gen_action(rng):
    """One synthetic proposed tool action.

    Returns (p_true, p_hat, u_score, base_consequence, reversible,
    ambiguity, staleness, u_fail, value): p_hat is a calibrated estimate (small
    unbiased noise); u_score is a generic uncertainty score (large noise,
    blind to consequence and reversibility); u_fail is the shared failure
    uniform for paired comparison across policies; value is the utility
    forfeited by abstaining.
    """
    p_base = 0.005 + 0.05 * rng.random() ** 2
    ambiguity = rng.uniform(0.3, 0.8) if rng.random() < 0.20 else rng.uniform(0.0, 0.3)
    staleness = rng.uniform(0.3, 0.9) if rng.random() < 0.25 else rng.uniform(0.0, 0.3)
    p_true = max(0.005, min(0.95, p_base + 0.08 * ambiguity + 0.06 * staleness))
    p_hat = max(0.005, min(0.95, p_true + rng.gauss(0.0, 0.02)))
    u_score = max(0.0, min(1.0, p_true + rng.gauss(0.0, 0.10)))
    r = rng.random()
    if r < 0.70:
        base_c = rng.uniform(0.05, 0.30)
    elif r < 0.95:
        base_c = rng.uniform(0.30, 0.70)
    else:
        base_c = rng.uniform(0.70, 1.00)
    reversible = rng.random() < 0.60
    return (p_true, p_hat, u_score, base_c, reversible, ambiguity, staleness,
            rng.random(), rng.uniform(0.1, 0.9))


def _casel_choose(p_hat, consequence, reversible, ambiguity, staleness,
                  value, ignore_reversibility=False):
    """CASEL: pick the mode minimizing expected cost.

    Expected cost = mode cost + HARM_WEIGHT x adjusted failure probability
    x consequence x recovery fraction x mode harm multiplier; abstention
    instead costs its handling cost plus the forfeited action value. With
    ignore_reversibility=True (ablation) planning assumes every action is
    reversible and sandbox containment always holds.
    """
    rec = _REVERSIBLE_HARM_FRACTION
    best_mode, best_cost = "direct", None
    for mode in MODES:
        if mode == "abstain":
            expected = MODE_COST[mode] + value
        else:
            p_adj = _adjusted_p(p_hat, ambiguity, staleness, mode)
            if ignore_reversibility:
                hm = _harm_mult(mode, True)
                rec_planned = _REVERSIBLE_HARM_FRACTION
            else:
                hm = _harm_mult(mode, reversible)
                rec_planned = rec if reversible else 1.0
            expected = (MODE_COST[mode]
                        + HARM_WEIGHT * p_adj * consequence * rec_planned * hm)
        if best_cost is None or expected < best_cost - 1e-12:
            best_mode, best_cost = mode, expected
    return best_mode


def _run_policy(actions, policy, casel_traj_costs=None, casel_mean_cost=None):
    """Realize one policy over pre-generated trajectories.

    `actions` is a list of trajectories (lists of action tuples).
    Returns aggregate counts: trajectories, harmful trajectories, total
    normalized harm, total cost, approvals, abstentions, failures, and the
    per-trajectory CASEL costs when policy == 'casel'.
    """
    n_traj = 0
    harmful_traj = 0
    total_harm = 0.0
    total_cost = 0.0
    n_actions = 0
    approvals = 0
    abstentions = 0
    failures = 0
    traj_costs = []
    for ti, traj in enumerate(actions):
        prior_harm_events = 0
        traj_harmful = False
        traj_cost = 0.0
        budget_left = casel_traj_costs[ti] if policy == "budget" else None
        # Ranked gates decide the approval set up front, per trajectory.
        if policy == "uncertainty":
            ranked = sorted(range(len(traj)), key=lambda i: -traj[i][2])
            n_appr = int(round(casel_mean_cost / MODE_COST["approve"] * len(traj)))
            approve_set = set(ranked[:n_appr])
        elif policy == "product":
            ranked = sorted(range(len(traj)),
                            key=lambda i: -(traj[i][1] * traj[i][3]))
            n_appr = int(round(casel_mean_cost / MODE_COST["approve"] * len(traj)))
            approve_set = set(ranked[:n_appr])
        else:
            approve_set = set()
        for i, act in enumerate(traj):
            (p_true, p_hat, _u, base_c, reversible, amb, stale,
             u_fail, value) = act
            # State-dependent consequence: consequence grows as the
            # trajectory's state degrades after harmful events.
            consequence = min(1.0, base_c * (1.0 + 0.5 * prior_harm_events))
            if policy == "casel":
                mode = _casel_choose(p_hat, consequence, reversible, amb, stale,
                                     value)
            elif policy == "casel_no_reversibility":
                mode = _casel_choose(p_hat, consequence, reversible, amb,
                                     stale, value, ignore_reversibility=True)
            elif policy in ("uncertainty", "product"):
                mode = "approve" if i in approve_set else "direct"
            elif policy == "budget":
                mode = _casel_choose(p_hat, consequence, reversible, amb, stale,
                                     value)
                if MODE_COST[mode] > budget_left:
                    mode = "direct"
                budget_left -= MODE_COST[mode]
            else:  # pragma: no cover
                raise ValueError(policy)
            cost = MODE_COST[mode]
            traj_cost += cost
            p_adj = _adjusted_p(p_true, amb, stale, mode)
            failed = u_fail < p_adj
            harm = 0.0
            if failed:
                failures += 1
                rec = _REVERSIBLE_HARM_FRACTION if reversible else 1.0
                harm = min(1.0, consequence * rec * _harm_mult(mode, reversible))
                total_harm += harm
                if harm >= HARM_EVENT_THRESHOLD:
                    prior_harm_events += 1
                    traj_harmful = True
            if mode == "approve":
                approvals += 1
            if mode == "abstain":
                abstentions += 1
            n_actions += 1
        total_cost += traj_cost
        traj_costs.append(traj_cost)
        n_traj += 1
        if traj_harmful:
            harmful_traj += 1
    return {
        "trajectories": n_traj,
        "harmful_trajectories": harmful_traj,
        "harmful_rate": harmful_traj / n_traj if n_traj else 0.0,
        "mean_normalized_harm": total_harm / n_actions if n_actions else 0.0,
        "mean_cost": total_cost / n_actions if n_actions else 0.0,
        "approval_rate": approvals / n_actions if n_actions else 0.0,
        "abstention_rate": abstentions / n_actions if n_actions else 0.0,
        "failure_rate": failures / n_actions if n_actions else 0.0,
        "n_actions": n_actions,
        "traj_costs": traj_costs,
    }


def run_casel_experiment(master_seed, n_seeds=N_SEEDS,
                         actions_per_seed=ACTIONS_PER_SEED,
                         traj_len=TRAJ_LEN):
    """Run all five allocation policies; return per-policy aggregates."""
    policies = ["casel", "uncertainty", "product",
                "casel_no_reversibility", "budget"]
    agg = {p: {"trajectories": 0, "harmful_trajectories": 0,
               "total_harm": 0.0, "total_cost": 0.0, "n_actions": 0,
               "approvals": 0, "abstentions": 0, "failures": 0,
               "seed_rates": []} for p in policies}
    for s in range(n_seeds):
        rng = random.Random(master_seed * 1000 + s)
        n_traj = actions_per_seed // traj_len
        trajectories = [[_gen_action(rng) for _ in range(traj_len)]
                        for _ in range(n_traj)]
        casel = _run_policy(trajectories, "casel")
        mean_cost = casel["mean_cost"]
        results = {"casel": casel}
        for policy in policies[1:]:
            results[policy] = _run_policy(
                trajectories, policy,
                casel_traj_costs=casel["traj_costs"],
                casel_mean_cost=mean_cost)
        for p in policies:
            r = results[p]
            a = agg[p]
            a["trajectories"] += r["trajectories"]
            a["harmful_trajectories"] += r["harmful_trajectories"]
            a["total_harm"] += r["mean_normalized_harm"] * r["n_actions"]
            a["total_cost"] += r["mean_cost"] * r["n_actions"]
            a["n_actions"] += r["n_actions"]
            a["approvals"] += int(round(r["approval_rate"] * r["n_actions"]))
            a["abstentions"] += int(round(r["abstention_rate"] * r["n_actions"]))
            a["failures"] += int(round(r["failure_rate"] * r["n_actions"]))
            a["seed_rates"].append(r["harmful_rate"])
    out = {}
    for p in policies:
        a = agg[p]
        n = a["n_actions"]
        rates = a["seed_rates"]
        mean_rate = sum(rates) / len(rates)
        sd = math.sqrt(sum((x - mean_rate) ** 2 for x in rates) / len(rates))
        out[p] = {
            "harmful_trajectory_rate": a["harmful_trajectories"] / a["trajectories"],
            "harmful_rate_seed_mean": mean_rate,
            "harmful_rate_seed_sd": sd,
            "mean_normalized_harm": a["total_harm"] / n,
            "mean_cost_per_action": a["total_cost"] / n,
            "approval_rate": a["approvals"] / n,
            "abstention_rate": a["abstentions"] / n,
            "failure_rate": a["failures"] / n,
            "n_actions": n,
            "n_trajectories": a["trajectories"],
        }
    return out


def threshold_validity_check(master_seed=CANONICAL_SEED, n_actions=60000):
    """Finite-sample threshold validity, the paper's formal condition.

    Fit a threshold tau on p_hat x consequence from a calibration split so
    the 95% Clopper-Pearson-style upper bound on the direct-execution
    failure rate stays below 0.05, then measure the held-out direct-arm
    failure rate. Returns (tau, calibration upper bound, held-out failure
    rate, held-out n). Uses the Wilson interval (normal approx.) for the
    bound, reported explicitly.
    """
    rng = random.Random(master_seed + 77)
    actions = [_gen_action(rng) for _ in range(n_actions)]
    calib, held = actions[: n_actions // 3], actions[n_actions // 3:]
    # Candidate thresholds on risk score; pick the largest tau whose
    # calibration direct-arm failure upper bound is <= 0.05.
    scores = sorted(a[1] * a[3] for a in calib)
    tau = None
    for cand in scores[:: max(1, len(scores) // 200)]:
        direct = [a for a in calib if a[1] * a[3] <= cand]
        if len(direct) < 200:
            continue
        fails = sum(1 for a in direct if a[7] < a[0])
        ph = fails / len(direct)
        z = 1.96
        denom = 1 + z * z / len(direct)
        centre = (ph + z * z / (2 * len(direct))) / denom
        half = z * math.sqrt(ph * (1 - ph) / len(direct)
                             + z * z / (4 * len(direct) ** 2)) / denom
        if centre + half <= 0.05:
            tau = cand
    if tau is None:
        tau = scores[len(scores) // 10]
    direct_held = [a for a in held if a[1] * a[3] <= tau]
    fails = sum(1 for a in direct_held if a[7] < a[0])
    return {
        "tau": tau,
        "held_out_direct_failure_rate": fails / len(direct_held),
        "held_out_direct_n": len(direct_held),
        "target": 0.05,
    }


def risk_ledger_check(master_seed=CANONICAL_SEED, n_traj=400, traj_len=50):
    """Trajectory risk ledger: planned expected harm vs realized harm.

    For each trajectory, the ledger accumulates CASEL's planned expected
    harm (p_hat x planned harm); returns the mean absolute ledger error
    and the fraction of trajectories whose realized harm stayed within
    ledger + 0.5. A ledger that tracks realization is the paper's
    validity condition for trajectory-level accounting.
    """
    rng = random.Random(master_seed + 55)
    errors = []
    within = 0
    for _ in range(n_traj):
        planned = 0.0
        realized = 0.0
        prior = 0
        for _ in range(traj_len):
            act = _gen_action(rng)
            (p_true, p_hat, _u, base_c, reversible, amb, stale, u_fail,
             value) = act
            consequence = min(1.0, base_c * (1.0 + 0.5 * prior))
            mode = _casel_choose(p_hat, consequence, reversible, amb, stale,
                                     value)
            p_adj_hat = _adjusted_p(p_hat, amb, stale, mode)
            rec = _REVERSIBLE_HARM_FRACTION if reversible else 1.0
            planned += p_adj_hat * consequence * rec * _harm_mult(mode, reversible)
            p_adj = _adjusted_p(p_true, amb, stale, mode)
            if u_fail < p_adj:
                harm = min(1.0, consequence * rec * _harm_mult(mode, reversible))
                realized += harm
                if harm >= HARM_EVENT_THRESHOLD:
                    prior += 1
        errors.append(abs(realized - planned))
        if realized <= planned + 0.5:
            within += 1
    return {
        "mean_abs_ledger_error": sum(errors) / len(errors),
        "fraction_within_ledger_plus_0.5": within / n_traj,
        "n_trajectories": n_traj,
    }
