"""Python side of the anytime-stopping Python-vs-Rust parity sweep.

Mirrors the `anytime-stopping parity <kind>` subcommands exactly (same
JSON keys, same draw order, same seed handling) so a parsed-JSON
comparison is exact: every compared value is an integer or a bool.
Usage:

    python3 tests/parity_gen.py <kind> <args...>
    # kinds: rng <seed> <n>
    #        trajectory <seed> <budget> <p_solvable> <p_fix>
    #                     <p_change_after_self_pass>
    #                     <p_change_after_self_fail> <p_break>
    #                     <p_self_false_pass> <p_self_false_fail>
    #        experiment <seed> <n_runs> <same 8 params...>

Not part of the Rust build; kept under tests/ as the parity oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "..", "..", "src"))

from arxiv_lab.agent.anytime_stopping import (
    POLICIES,
    SplitMix64,
    TrajectoryParams,
    evaluate_all,
    generate_trajectory,
    trajectory_digest,
)


def parse_params(args):
    return TrajectoryParams(
        budget=int(args[0]),
        p_solvable=float(args[1]),
        p_fix=float(args[2]),
        p_change_after_self_pass=float(args[3]),
        p_change_after_self_fail=float(args[4]),
        p_break=float(args[5]),
        p_self_false_pass=float(args[6]),
        p_self_false_fail=float(args[7]),
    )


def rng_kind(seed, n):
    r = SplitMix64(seed)
    u64s = [r.next_u64() for _ in range(n)]
    r = SplitMix64(seed)
    mantissas = [r.next_u64() >> 11 for _ in range(n)]
    r = SplitMix64(seed)
    bern = [r.bernoulli(0.5) for _ in range(n)]
    print(json.dumps({"kind": "rng", "seed": seed, "u64": u64s,
                      "mantissa": mantissas, "bernoulli_half": bern}))


def trajectory_kind(seed, params):
    rng = SplitMix64(seed)
    traj = generate_trajectory(rng, params)
    stops = {name: fn(traj) for name, fn in POLICIES.items()}
    print(json.dumps({
        "kind": "trajectory", "seed": seed,
        "steps": [s.to_list() for s in traj],
        "digest": trajectory_digest(traj),
        "stops": {
            "baseline": stops["baseline"],
            "first_self_pass": stops["first_self_pass"],
            "double_pass": stops["double_pass"],
            "double_pass_cumulative": stops["double_pass_cumulative"],
            "oracle": stops["oracle"],
        },
    }))


def experiment_kind(seed, n_runs, params):
    rng = SplitMix64(seed)
    trajs = [generate_trajectory(rng, params) for _ in range(n_runs)]
    ev = evaluate_all(trajs, params)
    # Integer aggregates only, so the comparison is exact.
    pol = ev["policies"]
    n = len(trajs)
    pf = ev["double_pass_post_fire"]
    out = {
        "kind": "experiment", "seed": seed, "n_runs": n_runs,
        # Integer fields only (the float changed_after_fire_rate is
        # excluded so the comparison stays exact).
        "post_fire": {
            "n_fired": pf["n_fired"],
            "changed_after_fire": pf["changed_after_fire"],
            "repairs_after_fire": pf["repairs_after_fire"],
            "breaks_after_fire": pf["breaks_after_fire"],
        },
        "n_true_first_pass": ev["hindsight_at_first_self_pass"]
                             ["n_true_first_pass"],
        "quality_wins": ev["quality_delta_double_pass_vs_baseline"]["wins"],
        "quality_losses":
            ev["quality_delta_double_pass_vs_baseline"]["losses"],
        "quality_ties":
            ev["quality_delta_double_pass_vs_baseline"]["ties"],
    }
    for name in ("baseline", "first_self_pass", "double_pass",
                 "double_pass_cumulative", "oracle"):
        rows = [(fn(traj)) for traj in trajs for fn in [POLICIES[name]]]
        out[name] = {
            "sum_stop_steps": sum(rows),
            "n_final_pass": sum(1 for traj, s in zip(trajs, rows)
                                if traj[s - 1].hidden_pass),
            "n_fired": sum(1 for s in rows if s < params.budget),
        }
    print(json.dumps(out))


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "rng":
        rng_kind(int(sys.argv[2]), int(sys.argv[3]))
    elif kind == "trajectory":
        trajectory_kind(int(sys.argv[2]), parse_params(sys.argv[3:11]))
    elif kind == "experiment":
        experiment_kind(int(sys.argv[2]), int(sys.argv[3]),
                        parse_params(sys.argv[4:12]))
    else:
        raise SystemExit("unknown kind: %s" % kind)
