"""CPU test for arXiv 2610.08048 -- DAEDALUS bootstrapping agent memory.

Tests the *protocol* on the scripted VaultWorld stand-ins (no LLM here):

 1. Solver loop accepts a heuristic only after >=1 failure followed by Ns
    consecutive successes with it in context (paper's validation rule);
    too-easy tasks (never failed) and too-hard tasks (Nf failures) yield
    no heuristic.
 2. The acceptance rule filters extractor errors: a wrongly-diagnosed
    heuristic cannot reach Ns consecutive successes, so it is rejected.
 3. The explorer calibrates difficulty: too-easy outcomes raise the room
    count, too-hard outcomes lower it.
 4. Consolidation dedupes redundant heuristics (one per convention).
 5. End-to-end: run_daedalus builds a non-empty bank from 12 sessions and
    the frozen bank transfers to held-out tasks (success_rate above the
    no-memory baseline).

Deterministic (seeded). numpy-guarded: skips cleanly without numpy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from arxiv_lab.daedalus import (
    ACCEPTED, TOO_EASY, TOO_HARD, CONVENTIONS,
    Task, VaultWorld, Solver, Extractor, Explorer, MemoryBank,
    make_task, solver_loop, run_daedalus, evaluate,
)
import random


def _quiet_task(seed=0, convention="color", n_rooms=6):
    return make_task("t", convention, n_rooms, random.Random(seed))


def test_env_judge():
    t = _quiet_task()
    w = VaultWorld(t, random.Random(0))
    rooms = w.scan().replace("Rooms: ", "").split(", ")
    assert set(rooms) == set(t.rooms)
    assert w.enter_code("0000") == "Wrong code. The keypad buzzes."
    assert not w.succeeded()
    assert w.enter_code(t.codes[t.code_room]) == "The vault clicks open."
    assert w.succeeded()
    assert "the keypad code is" in w.read_note(t.code_room)


def test_acceptance_rule():
    # effective heuristic: base fails sometimes, heuristic nearly always works
    t = _quiet_task(seed=3)
    solver = Solver(seed=11, p_inspect=0.0, p_slip=0.0)  # never inspects alone
    extractor = Extractor(seed=1, p_error=0.0)
    outcome, h = solver_loop(t, solver, extractor, Ns=3, Nf=6)
    assert outcome == ACCEPTED, outcome
    assert h is not None and h.convention == t.convention
    # too easy: solver that always knows -> no heuristic
    solver2 = Solver(seed=11, p_inspect=1.0, p_parse=1.0, max_tries=1,
                     p_slip=0.0)
    outcome2, h2 = solver_loop(t, solver2, extractor, Ns=3, Nf=6)
    assert outcome2 == TOO_EASY and h2 is None, (outcome2, h2)


def test_acceptance_filters_extractor_errors():
    # extractor always wrong -> heuristic can never earn Ns consecutive wins
    t = _quiet_task(seed=5)
    solver = Solver(seed=11, p_inspect=0.0, p_slip=0.0)
    bad_extractor = Extractor(seed=2, p_error=1.0)
    outcome, h = solver_loop(t, solver, bad_extractor, Ns=3, Nf=4)
    assert outcome == TOO_HARD and h is None, (outcome, h)


def test_explorer_calibration():
    ex = Explorer(seed=0)
    t = ex.propose(6)
    assert ex.feasible(t)
    harder = ex.refine(t, TOO_EASY)
    easier = ex.refine(t, TOO_HARD)
    assert harder.difficulty > t.difficulty
    assert easier.difficulty < t.difficulty
    assert harder.difficulty <= 10 and easier.difficulty >= 4


def test_consolidation_dedupes():
    from arxiv_lab.daedalus import Heuristic, _HEURISTIC_TEXT
    bank = MemoryBank()
    bank.accept(Heuristic(_HEURISTIC_TEXT["color"], "color", "a"))
    bank.accept(Heuristic(_HEURISTIC_TEXT["color"], "color", "b"))
    bank.accept(Heuristic(_HEURISTIC_TEXT["initial"], "initial", "c"))
    frozen = bank.consolidate()
    assert [h.convention for h in frozen] == ["color", "initial"]
    assert bank.acceptance_rate(10) == 0.3


def test_end_to_end_transfer():
    bank, log = run_daedalus(n_sessions=12, seed=7)
    assert bank.consolidated, "bank should be non-empty"
    assert len(log) == 12
    assert any(e["outcome"] == ACCEPTED for e in log)
    rng = random.Random(99)
    heldout = [make_task("h-%d" % i, rng.choice(CONVENTIONS), 7, rng)
               for i in range(24)]
    base = evaluate(heldout, solver_seed=1, heuristics=(), n_rollouts=5)
    mem = evaluate(heldout, solver_seed=1,
                   heuristics=bank.consolidated, n_rollouts=5)
    assert mem["success_rate"] > base["success_rate"], (base, mem)
    print("baseline %.3f -> daedalus %.3f (pass^5 %.3f -> %.3f)" % (
        base["success_rate"], mem["success_rate"],
        base["pass_5"], mem["pass_5"]))


def main():
    test_env_judge()
    print("env_judge OK")
    test_acceptance_rule()
    print("acceptance_rule OK")
    test_acceptance_filters_extractor_errors()
    print("acceptance_filters_errors OK")
    test_explorer_calibration()
    print("explorer_calibration OK")
    test_consolidation_dedupes()
    print("consolidation OK")
    test_end_to_end_transfer()
    print("end_to_end OK")
    if not HAS_NUMPY:
        print("(numpy absent -- numeric asserts skipped)")


if __name__ == "__main__":
    main()
