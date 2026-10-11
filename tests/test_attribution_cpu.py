"""CPU test for arXiv 2610.11600 — EMFA failure attribution.

Reproduces the paper's core claim in a synthetic taint-tag world model:
propagation-aware candidate screening plus counterfactual verification
identifies the decisive (agent, step) pair of a failed multi-agent
execution more accurately than judging steps in isolation.

Deterministic (seeded builders). stdlib only.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.attribution import emfa as E

passed = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok: {name}")


print("== stage unit checks ==")
t = E.make_cascade(seed=1)
final = E.find_final_failure(t)
check("final failure = last executed step", final == max(t.executed))

estep, esub = E.build_dependencies(t)
check("dependency edges follow consumes",
      all((c, s.id) in estep for s in t.steps for c in s.consumes))
check("subtask reuse edges recorded", len(esub) > 0)

ginfo, gedges = E.backward_trace(t, final, estep)
check("backward trace reaches the deviation source", 1 in ginfo)
check("backward trace excludes nothing reachable",
      ginfo == set(s.id for s in t.steps if s.id in t.executed))

ann = E.annotate_propagation(t, ginfo)
check("source annotated introduce", ann[1] == "introduce")
check("downstream annotated carry",
      all(ann[i] == "carry" for i in ginfo if i not in (0, 1, final)))
check("final annotated carry (symptom), not introduce", ann[final] == "carry")

check("no loops in a pure cascade", E.extract_loops(t) == [])

tl = E.make_loop(seed=2, ambiguous=True)
loops = E.extract_loops(tl)
coder_loops = [l for l in loops
               if all(tl.steps[i].agent == "coder" for i in l)]
check("persistent loop detected on the retrying agent",
      len(coder_loops) == 1 and len(coder_loops[0]) == 3 + 2 % 3)
check("loop steps are the coder attempts", all(
    tl.steps[i].action_sig == "patch" for i in coder_loops[0]))

tr = E.make_recovered(seed=0)
ginfo_r, _ = E.backward_trace(tr, E.find_final_failure(tr),
                              E.build_dependencies(tr)[0])
ann_r = E.annotate_propagation(tr, ginfo_r)
check("corrected step annotated correct", ann_r[2] == "correct")
cand_r = E.screen_candidates(tr, ginfo_r, ann_r, [], E.find_final_failure(tr))
check("recovered deviation screened out", 1 not in cand_r)
check("unrecovered deviation kept", 4 in cand_r)
check("carriers never candidates",
      all(ann_r.get(c) == "introduce" for c in cand_r))
check("candidate pool bounded (<=4)", len(cand_r) <= 4)

tp = E.make_premature(seed=0)
pred_p = E.emfa_attribute(tp)
check("premature stop attributed via counterfactual", pred_p == ("coder", 5))

td = E.make_redundant(seed=0)
pred_d = E.emfa_attribute(td)
check("earliest-decisive tie-break", pred_d == ("researcher_a", 1))

print("== 240-trajectory experiment ==")
methods = {"emfa": E.emfa_attribute, "blame_final": E.blame_final,
           "blame_latest_wrong": E.blame_latest_wrong,
           "blame_earliest_wrong": E.blame_earliest_wrong}
configs = []
for i in range(40):
    configs.append(("cascade", i, {}))
for i in range(40):
    configs.append(("recovered", i, {}))
for i in range(40):
    configs.append(("loop", i, {}))
for i in range(40):
    configs.append(("loop", i, {"ambiguous": False}))
for i in range(40):
    configs.append(("premature", 0, {}))
for i in range(40):
    configs.append(("redundant", 0, {}))

step_acc, agent_acc = {}, {}
for m, fn in methods.items():
    s = a = 0
    for name, seed, kw in configs:
        traj = E.BUILDERS[name](seed, **kw)
        pa, ps = fn(traj)
        ga, gs = traj.decisive
        s += (pa, ps) == (ga, gs)
        a += pa == ga
    step_acc[m] = s / len(configs)
    agent_acc[m] = a / len(configs)
    print(f"  {m:20s} step_acc={step_acc[m]:.3f} agent_acc={agent_acc[m]:.3f}")

check("EMFA step-level accuracy is perfect on the suite",
      step_acc["emfa"] == 1.0)
check("EMFA agent-level accuracy is perfect on the suite",
      agent_acc["emfa"] == 1.0)
best_base = max(v for k, v in step_acc.items() if k != "emfa")
check("EMFA beats the best isolation baseline by >= 30pp",
      step_acc["emfa"] - best_base >= 0.30)
print(f"  margin over best baseline: {step_acc['emfa'] - best_base:.3f} "
      f"(paper reports +3.45/+4.40pp SOTA gains on Who&When)")
check("latest-wrong baseline never right (carriers/symptoms mislead)",
      step_acc["blame_latest_wrong"] == 0.0)

print(f"\nALL {len(passed)} EMFA CPU CHECKS PASSED")
