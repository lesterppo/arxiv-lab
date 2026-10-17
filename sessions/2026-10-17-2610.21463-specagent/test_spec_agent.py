"""CPU tests for arxiv_lab.agent.spec_agent (arXiv:2610.21463).

Paper claims under test (sim-scale, deterministic grid tool-world):
  1. Quality retention: SpecAgent success within tolerance of the
     strong-only baseline (paper: within ~1 point on tau2-Bench).
  2. Cost reduction: strong-model *decoded* tokens substantially reduced vs
     strong-only (paper: roughly halved memory traffic / wall time; at most
     half as many generated tokens in the edge-cloud setting).
  3. Invariant: the weak model never executes an environment-modifying
     action -- it holds the first write; only the orchestrator executes
     committed/repaired writes.
  4. Invariant: every committed step passed strong-model verification
     (longest-valid-prefix rule).
  5. Invariant: every drafted read was executed by the weak model against
     the real environment (grounded in real observations).

Seeded and deterministic. stdlib only.

Run:  python3 src/arxiv_lab/agent/test_spec_agent.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from arxiv_lab.agent.spec_agent import (
    READ, WRITE, TERMINAL,
    Step, ToolWorld, WeakPolicy, StrongPolicy, SpecAgent,
    run_strong_only, run_weak_only, evaluate,
)

SEED = 7
N_TASKS = 64


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def test_protocol_unit():
    print("[unit] draft/verify/commit mechanics on a fixed world")
    rng = random.Random(0)
    world = ToolWorld(6, [(2, 2), (4, 4)])
    weak, strong = WeakPolicy(err=0.0), StrongPolicy()

    # weak drafts: reads executed for real, exactly one held write at the end
    seg = weak.draft(world, rng)
    reads, tail = seg[:-1], seg[-1]
    check(all(s.kind == READ and s.executed and s.executor == "weak"
              and s.observation is not None for s in reads),
          "draft reads executed by weak with real observations")
    check(tail.kind in (WRITE, TERMINAL) and not tail.executed
          and tail.executor is None,
          "first write held unexecuted at end of draft")
    check(all(e != "weak" or t not in ToolWorld.WRITE_TOOLS
              for e, t in world.write_log),
          "weak executed no writes while drafting")

    # verifier: longest valid prefix + repair of first rejected step
    bad = [Step("scan", (3,), READ, observation=world.scan(3),
                executed=True, executor="weak"),
           Step("collect", (), WRITE)]  # invalid: not standing on a target
    verdicts, repair = strong.verify(bad, world, random.Random(1))
    strong2 = StrongPolicy(miss=0.0, false_alarm=0.0)
    verdicts, repair = strong2.verify(bad, world, random.Random(1))
    check(verdicts == [True, False], "verdicts mark the bad write, keep the read")
    check(repair is not None and repair.name == "goto",
          "first rejected step repaired with a goto")

    # run() commits only verified steps
    agent = SpecAgent(WeakPolicy(err=0.0), StrongPolicy(miss=0.0, false_alarm=0.0))
    res = agent.run(ToolWorld(6, [(1, 0)]), random.Random(3))
    check(res["success"], "noiseless spec run solves the task")
    check(all(v is True for _, _, v in res["commits"]),
          "every committed step carried a positive verdict")
    check(all(e != "weak" or t not in ToolWorld.WRITE_TOOLS
              for e, t in res["write_log"]),
          "full run: weak never executed a write")


def test_quality_and_cost():
    print("[eval] %d seeded tasks: quality retention + token reduction" % N_TASKS)
    summary, _ = evaluate(N_TASKS, SEED)
    s = summary
    print("  strong-only success : %.3f" % s["strong_success"])
    print("  spec success        : %.3f" % s["spec_success"])
    print("  weak-only success   : %.3f" % s["weak_success"])
    print("  strong decoded tok  : strong-only=%.0f spec=%.0f ratio=%.3f"
          % (s["mean_strong_decoded"]["strong_only"],
             s["mean_strong_decoded"]["spec"], s["strong_token_ratio"]))
    print("  strong prefill tok  : spec=%.0f/episode" % s["mean_strong_prefill"]["spec"])
    print("  weak decoded tok    : spec=%.0f weak-only=%.0f"
          % (s["mean_weak_decoded"]["spec"], s["mean_weak_decoded"]["weak_only"]))
    print("  segments/ep=%.2f repairs/ep=%.2f"
          % (s["mean_segments_per_episode"], s["mean_repairs_per_episode"]))

    check(s["spec_success"] >= s["strong_success"] - 0.10,
          "spec within 10 pts of strong-only "
          "(%.3f vs %.3f)" % (s["spec_success"], s["strong_success"]))
    check(s["strong_token_ratio"] <= 0.60,
          "strong decoded tokens cut to <=60%% of strong-only "
          "(ratio %.3f)" % s["strong_token_ratio"])
    check(s["mean_strong_decoded"]["spec"] < s["mean_strong_decoded"]["strong_only"],
          "strictly fewer strong decoded tokens than strong-only")
    check(s["spec_success"] - s["weak_success"] >= 0.30,
          "spec recovers >=30 pts of the weak->strong gap "
          "(+%.3f)" % (s["spec_success"] - s["weak_success"]))


def test_invariants():
    print("[eval] protocol invariants over %d seeded spec episodes" % N_TASKS)
    summary, _ = evaluate(N_TASKS, SEED)
    v = summary["invariant_violations"]
    print("  violations:", v)
    check(v["weak_executed_write"] == 0,
          "weak model never executed an environment-modifying action")
    check(v["commit_without_verdict"] == 0,
          "every committed step passed verification (longest valid prefix)")
    check(v["ungrounded_read"] == 0,
          "every drafted read grounded in a real observation")


def main():
    test_protocol_unit()
    test_quality_and_cost()
    test_invariants()
    print("ALL SPECAGENT TESTS PASSED")


if __name__ == "__main__":
    main()
