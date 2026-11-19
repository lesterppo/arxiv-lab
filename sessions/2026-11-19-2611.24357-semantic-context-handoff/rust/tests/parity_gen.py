"""Python side of the semantic-context-handoff Python-vs-Rust parity sweep.

Mirrors the `semantic-context-handoff parity <kind>` subcommands exactly
(same JSON keys, same draw order, same seed handling) so a parsed-JSON
comparison is exact: every compared value is an integer or a bool.
Usage:

    python3 tests/parity_gen.py <kind> <args...>
    # kinds: rng <seed> <n>
    #        task <seed> <13 params...>
    #        experiment <seed> <n_tasks> <13 params...>
    # 13 params, in order: steps facts_per_step delayed_frac max_delay
    # min_fact_tokens max_fact_tokens state_retention compression
    # state_overhead recall lookup_overhead fp_per_lookup
    # pass_step_fraction

Not part of the Rust build; kept under tests/ as the parity oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "..", "..", "src"))

from arxiv_lab.agent.semantic_context_handoff import (
    POLICIES,
    SplitMix64,
    TaskParams,
    evaluate_all,
    evaluate_policy,
    generate_task,
    task_digest,
)


def parse_params(args):
    return TaskParams(
        steps=int(args[0]),
        facts_per_step=int(args[1]),
        delayed_frac=float(args[2]),
        max_delay=int(args[3]),
        min_fact_tokens=int(args[4]),
        max_fact_tokens=int(args[5]),
        state_retention=float(args[6]),
        compression=float(args[7]),
        state_overhead=int(args[8]),
        recall=float(args[9]),
        lookup_overhead=int(args[10]),
        fp_per_lookup=float(args[11]),
        pass_step_fraction=float(args[12]),
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


def task_kind(seed, params):
    rng = SplitMix64(seed)
    task = generate_task(rng, params)
    out = {
        "kind": "task", "seed": seed,
        "facts": [[f.key, f.produced_step, f.needed_step, f.tokens,
                   f.retained] for f in task.facts],
        "needs": task.needs,
        "digest": task_digest(task),
        "policies": {},
    }
    for p in POLICIES:
        r = evaluate_policy(task, p, params, seed=seed, task_index=0)
        out["policies"][p] = {
            "steps_ok": r["steps_ok"],
            "task_pass": r["task_pass"],
            "strict_task_pass": r["strict_task_pass"],
            "total_tokens": r["total_tokens"],
            "recoveries": r["recoveries"],
            "recovery_hits": r["recovery_hits"],
        }
    print(json.dumps(out))


def experiment_kind(seed, n_tasks, params):
    rng = SplitMix64(seed)
    tasks = [generate_task(rng, params) for _ in range(n_tasks)]
    res = evaluate_all(tasks, params, seed=seed)
    # Integer aggregates only, so the comparison is exact.
    out = {"kind": "experiment", "seed": seed, "n_tasks": n_tasks,
           "policies": {}}
    for p in POLICIES:
        a = res[p]
        # Recompute the integer sums from the per-task results (the
        # float rates in evaluate_all's output are derived from them).
        passes = strict = steps_ok = steps = tokens = 0
        recoveries = hits = 0
        for i, task in enumerate(tasks):
            r = evaluate_policy(task, p, params, seed=seed, task_index=i)
            passes += 1 if r["task_pass"] else 0
            strict += 1 if r["strict_task_pass"] else 0
            steps_ok += r["steps_ok"]
            steps += r["steps"]
            tokens += r["total_tokens"]
            recoveries += r["recoveries"]
            hits += r["recovery_hits"]
        out["policies"][p] = {
            "tasks": n_tasks,
            "passes": passes,
            "strict_passes": strict,
            "steps_ok": steps_ok,
            "steps": steps,
            "tokens": tokens,
            "recoveries": recoveries,
            "recovery_hits": hits,
        }
        assert a["tasks"] == n_tasks
    print(json.dumps(out))


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "rng":
        rng_kind(int(sys.argv[2]), int(sys.argv[3]))
    elif kind == "task":
        task_kind(int(sys.argv[2]), parse_params(sys.argv[3:16]))
    elif kind == "experiment":
        experiment_kind(int(sys.argv[2]), int(sys.argv[3]),
                        parse_params(sys.argv[4:17]))
    else:
        raise SystemExit("unknown kind: %s" % kind)
