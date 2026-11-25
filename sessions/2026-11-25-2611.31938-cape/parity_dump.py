#!/usr/bin/env python3
"""Generate parity cases for the Rust port of CAPE (arXiv:2611.31938).

Runs the Python compiler on random worlds and dumps one case per line
to rust/tests/parity_cases.txt, in a std-parseable format:
  n_aspects cap budget cfg1,cfg2,... comp negcalls queries certified completed_n total_calls opt_comp opt_negcalls
where each cfg is a digit string (one char per aspect = value index)
and certified is 0/1. The Rust parity test rebuilds the identical
world from the digit strings and requires 0 mismatches on every field.
Deterministic: fixed seed list.
"""
import os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from arxiv_lab.agent.cape import (
    make_world, compile_policy, exact_optimum,
)

OUT = os.path.join(os.path.dirname(__file__), "rust", "tests", "parity_cases.txt")

def cfg_str(w, h):
    return "".join(str(w.aspects[i].values.index(v)) for i, v in enumerate(h))

def main():
    rng = random.Random(20261125)
    cases = []
    # (n_cases, n_configs range, cap, budget range)
    plan = [
        (250, (4, 8), (1, 2), (50, 300)),
        (150, (10, 12), (2, 2), (100, 400)),
        (40, (6, 8), (3, 3), (200, 400)),
        (4, (12, 12), (3, 3), (1000, 1000)),
    ]
    n_done = 0
    for n_cases, n_range, cap_range, budget_range in plan:
        for _ in range(n_cases):
            ncfg = rng.randint(*n_range)
            cap = rng.randint(*cap_range)
            budget = rng.randint(*budget_range)
            seed = rng.randint(0, 10 ** 6)
            w = make_world(n_configs=ncfg, n_aspects=4, seed=seed)
            r = compile_policy(w, cap=cap, budget=budget)
            opt, _ = exact_optimum(w, cap=cap)
            cfgs = ",".join(cfg_str(w, h) for h in w.configs)
            cases.append(
                "4 %d %d %s %d %d %d %d %d %d %d %d"
                % (cap, budget, cfgs, r.value[0], r.value[1], r.queries,
                   1 if r.certified else 0, len(r.completed), r.total_calls,
                   opt[0], opt[1]))
            n_done += 1
            if n_done % 50 == 0:
                print("generated %d/%d" % (n_done, 444), flush=True)
    with open(OUT, "w") as f:
        f.write("\n".join(cases) + "\n")
    print("wrote %d cases to %s" % (len(cases), OUT))

if __name__ == "__main__":
    main()
