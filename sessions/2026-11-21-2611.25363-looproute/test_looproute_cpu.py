"""CPU test for LoopRoute (arXiv:2611.25363) — deterministic, stdlib only.

Live experiments (all run here, no model calls):
  E1  Router unit behaviour: triggers on the 2nd occurrence of a
      (state, action) pair (threshold=2), not on the 1st; resets after an
      assist; can re-trigger on a later recurrence.
  E2  Oracle planner sanity: plan_to_goal returns a legal plan that cooks
      the meal from every tested start state.
  E3  False-trigger check: on unlocked tasks the fast policy never repeats
      a pair, so LoopRoute must request 0 assists.
  E4  Main experiment (60 seeded tasks, mixed locked/unlocked): fast-only
      vs LoopRoute-oracle at threshold 2, plus a threshold-3 sensitivity.
      Asserts the paper's directional claim: LoopRoute improves task
      success over the standalone fast policy with only limited assistant
      use (few calls per task).
  E5  Determinism: the main experiment repeated gives identical numbers.

Usage: python3 test_looproute_cpu.py  (writes results.json next to itself)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.agent.looproute import (  # noqa: E402
    FastPolicy, KitchenWorld, LoopRouter, oracle_assistant,
    plan_to_goal, run_episode, run_experiment,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(1000, 1060))


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def main():
    results = {}

    # E1 — router unit behaviour.
    r = LoopRouter(threshold=2)
    s, a = ("room",), "open chest"
    check("E1 no trigger on 1st occurrence", not r.should_assist(s, a))
    r.note(s, a)
    check("E1 trigger on 2nd occurrence", r.should_assist(s, a))
    plan = r.assist(s, a, ["go pantry", "take brass key"])
    check("E1 assist returns plan", plan == ["go pantry", "take brass key"])
    check("E1 count resets after assist", not r.should_assist(s, a))
    r.note(s, a)
    check("E1 re-trigger after recurrence", r.should_assist(s, a))
    r3 = LoopRouter(threshold=3)
    r3.note(s, a)
    check("E1 threshold=3 silent on 2nd", not r3.should_assist(s, a))
    r3.note(s, a)
    check("E1 threshold=3 triggers on 3rd", r3.should_assist(s, a))
    results["E1_router_units"] = "all passed"

    # E2 — oracle planner sanity on 10 seeds (mixed locked/unlocked).
    for seed in range(2000, 2010):
        env = KitchenWorld(seed, locked=(seed % 2 == 0))
        plan = plan_to_goal(env)
        check(f"E2 plan exists seed={seed}", len(plan) > 0, f"len={len(plan)}")
        env.reset()
        done = False
        for act in plan:
            _, _, done, info = env.step(act)
            assert info["ok"], f"illegal plan step {act!r} seed={seed}"
        check(f"E2 plan cooks seed={seed}", done)
    results["E2_oracle_plans"] = "10/10 plans legal and complete"

    # E3 — false-trigger check on unlocked tasks.
    policy = FastPolicy()
    assists = 0
    solved = 0
    for seed in range(3000, 3020):
        router = LoopRouter(threshold=2)
        res = run_episode(KitchenWorld(seed, locked=False), policy,
                          router=router, assistant=oracle_assistant)
        assists += res["assists"]
        solved += res["success"]
    check("E3 zero assists on loop-free tasks", assists == 0,
          f"assists={assists}")
    check("E3 fast policy solves unlocked tasks", solved == 20,
          f"solved={solved}/20")
    results["E3_false_trigger"] = {"assists": assists, "solved": solved}

    # E4 — main experiment.
    exp2 = run_experiment(SEEDS, threshold=2)
    exp3 = run_experiment(SEEDS, threshold=3)
    f, o = exp2["fast_only"], exp2["looproute_oracle"]
    gain = o["success_rate"] - f["success_rate"]
    print(f"E4 fast-only:    success={f['success_rate']:.3f} "
          f"mean_steps={f['mean_steps']:.1f}")
    print(f"E4 looproute K=2: success={o['success_rate']:.3f} "
          f"mean_steps={o['mean_steps']:.1f} "
          f"mean_assists={o['mean_assists']:.2f} "
          f"mean_llm_calls={o['mean_llm_calls']:.2f}")
    print(f"E4 looproute K=3: success={exp3['looproute_oracle']['success_rate']:.3f} "
          f"mean_assists={exp3['looproute_oracle']['mean_assists']:.2f}")
    check("E4 looproute beats fast-only by >=30pp", gain >= 0.30,
          f"gain={gain:.3f}")
    check("E4 limited assistant use (<=3 calls/task)",
          o["mean_llm_calls"] <= 3.0, f"calls={o['mean_llm_calls']:.2f}")
    check("E4 looproute mean steps < fast-only",
          o["mean_steps"] < f["mean_steps"],
          f"{o['mean_steps']:.1f} vs {f['mean_steps']:.1f}")
    results["E4_main_K2"] = {"fast_only": f, "looproute_oracle": o,
                             "gain_pp": round(gain * 100, 1)}
    results["E4_main_K3"] = exp3["looproute_oracle"]

    # E5 — determinism.
    again = run_experiment(SEEDS, threshold=2)
    check("E5 deterministic repeat", again == exp2)
    results["E5_deterministic"] = True

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print("ALL CHECKS PASSED — results written to results.json")


if __name__ == "__main__":
    main()
