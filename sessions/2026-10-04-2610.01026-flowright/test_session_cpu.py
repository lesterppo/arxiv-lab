"""Session-level CPU test for 2610.01026 FloWright co-evolution.

Re-runs the session's hardened-difficulty experiment at reduced size (via
the same code path, without touching the canonical results.json) and asserts
the paper's core directional claims hold: co-evolution beats single-role
evolution, and the flat shared-reward baseline leaves the generator
under-evolved relative to co-evolution (credit-assignment pathology visible
in skills).

Deterministic, pure stdlib.

Run:  python3 sessions/2026-10-04-2610.01026-flowright/test_session_cpu.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import experiment


def main():
    # Reduced size, same code path as experiment.main(); does not write
    # results.json.
    experiment.GENERATIONS = 20
    experiment.TRAIN_N = 48
    experiment.TEST_N = 96
    experiment.SEEDS = [0, 1, 2]
    d = experiment.run_difficulty("hardened")

    def mean_acc(cond):
        vals = [d["seeds"][str(s)]["conditions"][cond]["test_acc"]
                for s in experiment.SEEDS]
        return sum(vals) / len(vals)

    def mean_skill(cond, role):
        vals = [d["seeds"][str(s)]["conditions"][cond]["skills"][role]
                for s in experiment.SEEDS]
        return sum(vals) / len(vals)

    coev, single, flat = (mean_acc(c) for c in ("coev", "single", "flat"))
    print("reduced: coev=%.3f single=%.3f flat=%.3f" % (coev, single, flat))

    # Paper's core claim: co-evolving roles beats evolving one role alone.
    assert coev > single + 0.20, \
        "coev should clearly beat single-role: %.3f vs %.3f" % (coev, single)
    # Co-evolution must actually work (near-ceiling on the stratified test).
    assert coev > 0.85, "coev should reach high accuracy: %.3f" % coev
    # Single-role is capped by the fixed verifier/refiner bottleneck.
    assert single < 0.70, "single-role should be capped: %.3f" % single
    # Credit-assignment pathology: under the flat shared reward the
    # generator gets no directional signal once verifier+refiner compensate,
    # so it ends less evolved than under structured co-evolution.
    gen_flat = mean_skill("flat", "generator")
    gen_coev = mean_skill("coev", "generator")
    print("reduced: generator skill flat=%.3f coev=%.3f" % (gen_flat, gen_coev))
    assert gen_flat < gen_coev, \
        "flat should leave the generator less evolved: %.3f vs %.3f" \
        % (gen_flat, gen_coev)

    print("session test passed")


if __name__ == "__main__":
    main()
