"""Python side of the LeBet Python-vs-Rust parity sweep.

Mirrors the `lebet parity <kind>` subcommands exactly (same JSON keys,
same draw order, same seed handling) so a parsed-JSON comparison is
exact. Usage:

    python3 tests/parity_gen.py <kind> <args...>
    # kinds: rng <seed> <n> | trajs <seed> <n_succ> <n_fail>
    #        mats <seed> <n_succ> <n_fail> | mutate <seed> <n>
    #        wealth <seed> <n_succ> <n_fail> | experiment <seed>

Not part of the Rust build; kept under tests/ as the parity oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from arxiv_lab.agent import lebet


def rng_kind(seed, n):
    r = lebet.Rng(seed)
    u64 = [r.next_u64() for _ in range(n)]
    r = lebet.Rng(seed)
    uniform = [r.uniform() for _ in range(n)]
    r = lebet.Rng(seed)
    randint6 = [r.randint(6) for _ in range(n)]
    r = lebet.Rng(seed)
    gauss = [r.gauss(0.0, 1.0) for _ in range(n)]
    print(json.dumps({"kind": "rng", "seed": seed, "u64": u64,
                      "uniform": uniform, "randint6": randint6, "gauss": gauss}))


def trajs_kind(seed, ns, nf):
    rng = lebet.Rng(seed)
    data = lebet.gen_dataset(rng, ns, nf)
    out = [{"label": t["label"],
            "steps": [[a, ok, s, o, tok] for (a, ok, s, o, tok) in t["steps"]]}
           for t in data]
    print(json.dumps({"kind": "trajs", "seed": seed, "trajs": out}))


def mats_kind(seed, ns, nf):
    rng = lebet.Rng(seed)
    data = lebet.gen_dataset(rng, ns, nf)
    out = []
    for t in data:
        mat = lebet.prefix_matrix(t)
        rows = ["".join("1" if b else "0" for b in row) for row in mat]
        out.append({"label": t["label"], "mat": rows})
    print(json.dumps({"kind": "mats", "seed": seed, "mats": out}))


def mutate_kind(seed, n):
    rng = lebet.Rng(seed)
    cur = (frozenset([0, 2]), frozenset([5]))
    out = []
    for _ in range(n):
        cur = lebet._mutate(rng, cur)
        out.append(lebet.monitor_to_text(cur))
    print(json.dumps({"kind": "mutate", "seed": seed, "monitors": out}))


def wealth_kind(seed, ns, nf):
    rng = lebet.Rng(seed)
    data = lebet.gen_dataset(rng, ns, nf)
    mats = [(lebet.prefix_matrix(t), t["label"]) for t in data]
    mon = (frozenset([4, 8]), frozenset([2]), frozenset([3]))
    q = lebet.baseline_fire_rate(mon, mats)
    out = []
    for mat, label in mats[:8]:
        series = lebet.betting_wealth_series(mon, mat, q, 0.75)
        step = lebet.betting_intervene_step(mon, mat, q, 0.15, 0.75)
        out.append({"label": label, "wealth": series, "intervene": step})
    print(json.dumps({"kind": "wealth", "seed": seed, "q": q, "trajs": out}))


def experiment_kind(seed):
    r = lebet.run_experiment(seed=seed)
    print(json.dumps(r))


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "rng":
        rng_kind(int(sys.argv[2]), int(sys.argv[3]))
    elif kind == "trajs":
        trajs_kind(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif kind == "mats":
        mats_kind(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif kind == "mutate":
        mutate_kind(int(sys.argv[2]), int(sys.argv[3]))
    elif kind == "wealth":
        wealth_kind(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif kind == "experiment":
        experiment_kind(int(sys.argv[2]))
    else:
        raise SystemExit("unknown kind: %s" % kind)
