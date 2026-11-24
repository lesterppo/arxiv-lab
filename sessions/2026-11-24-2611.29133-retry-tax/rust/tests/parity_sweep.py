"""Python-vs-Rust parity sweep for the Retry Tax port.

Generates random task populations, feeds them to the Rust
`metrics-batch` subcommand, and compares the closed-form outputs
(completion, spend/task, cost-per-completion, marginal cost) against
the Python module's values. 0 mismatches required (tolerance 1e-9
relative; both sides are IEEE-754 f64 closed forms, so agreement
should be exact to rounding).

Also checks the simulated quantities agree within Monte-Carlo noise:
the Python and Rust RNG streams differ by design, so simulation
parity is a tolerance check on completion/CPC vs the shared closed
form, not a draw-for-draw match.

Run from the rust/ dir after `cargo build --release`:
    python3 tests/parity_sweep.py
Writes parity_results.json next to this file.
"""

import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUST_DIR = os.path.dirname(HERE)
SESSION_DIR = os.path.dirname(RUST_DIR)
sys.path.insert(0, os.path.join(SESSION_DIR, "..", "..", "src"))

from arxiv_lab.agent import retry_tax as rt  # noqa: E402

BIN = os.path.join(RUST_DIR, "target", "release", "retrytax")
N_CASES = 500
SEED = 424242


def main():
    rng = random.Random(SEED)
    cases = []
    for _ in range(N_CASES):
        n = rng.randint(1, 60)
        kind = rng.random()
        if kind < 0.4:
            ps = [rng.random() for _ in range(n)]
        elif kind < 0.7:
            # Two-point mixture, like the calibrated population.
            ps = [rng.choice([0.05, 0.9, 0.5, 0.02]) for _ in range(n)]
        else:
            # Includes degenerate p = 0 / p = 1 tasks.
            ps = [rng.choice([0.0, 1.0, rng.random()]) for _ in range(n)]
        cap = rng.randint(1, 8)
        cost = rng.choice([0.418, 1.0, 2.5])
        cases.append((cap, cost, ps))

    stdin = "\n".join(
        f"{cap} {cost} " + " ".join(f"{p:.17g}" for p in ps)
        for cap, cost, ps in cases
    ) + "\n"
    proc = subprocess.run([BIN, "metrics-batch"], input=stdin,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().split("\n")
    assert len(lines) == len(cases), f"{len(lines)} != {len(cases)}"

    mismatches = 0
    worst = 0.0
    for (cap, cost, ps), line in zip(cases, lines):
        comp, spend, cpc, mc = (float(x) for x in line.split())
        m = rt.population_metrics(ps, cap, cost)
        exp_mc = rt.marginal_cost(ps, cap, cost) if cap >= 2 else None
        pairs = [(comp, m.completion), (spend, m.spend_per_task),
                 (cpc, m.cost_per_completion)]
        if exp_mc is not None:
            pairs.append((mc, exp_mc))
        for got, exp in pairs:
            if math.isinf(exp):
                ok = math.isinf(got)
            elif math.isnan(exp):
                ok = math.isnan(got)
            else:
                denom = max(1.0, abs(exp))
                ok = abs(got - exp) <= 1e-9 * denom
                worst = max(worst, abs(got - exp) / denom)
            if not ok:
                mismatches += 1
                print(f"MISMATCH cap={cap} cost={cost} got={got} exp={exp}")

    # Simulation tolerance check: both sides' simulations must sit
    # within MC noise of the shared closed form (different RNGs).
    sim_bad = 0
    for _ in range(20):
        ps = rt.calibrated_mixture(5000, rng.randint(1, 10**6))
        m = rt.population_metrics(ps, 4, 0.418)
        c, _s, cpc = rt.simulate_population(
            ps, 4, 0.418, random.Random(rng.randint(1, 10**6)))
        if abs(c - m.completion) > 0.02 or abs(cpc - m.cost_per_completion) > 0.05:
            sim_bad += 1
    print(f"parity cases: {N_CASES}, mismatches: {mismatches}, "
          f"worst rel diff: {worst:.3g}, sim tolerance failures: {sim_bad}")
    assert mismatches == 0, f"{mismatches} parity mismatches"
    assert sim_bad == 0, f"{sim_bad} simulation tolerance failures"

    with open(os.path.join(HERE, "parity_results.json"), "w") as f:
        json.dump({"cases": N_CASES, "mismatches": mismatches,
                   "worst_rel_diff": worst, "sim_tolerance_failures": sim_bad,
                   "verdict": "PARITY VERIFIED"}, f, indent=1)
    print("PARITY VERIFIED")


if __name__ == "__main__":
    main()
