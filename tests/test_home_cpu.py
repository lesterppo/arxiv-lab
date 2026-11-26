#!/usr/bin/env python3
"""CPU test for HoMe (arXiv 2611.34624) -- deterministic, stdlib only.

E1  HoMe loop vs starting harness: accuracy gain + fewer query tokens,
    and the evolved harness reaches the exhaustive-oracle accuracy.
E2  Localized (trace-attributed) edits vs random edits: mean accuracy
    gain per formal evaluation over 12 seeds.
E3  Candidate gating (smoke tests): formal evaluations consumed to
    reach the final accuracy, gated vs ungated.
E4  Oracle sanity: the exhaustive optimum is the expected harness
    (all-versions, recency 1.0, top1, fuzzy) and scores 1.0.
E5  Determinism: identical results across repeated runs.
E6  Unit checks: memory construction, failure attribution, edit order.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arxiv_lab.agent.home import (  # noqa: E402
    Harness, build_memory, dominant_cause, evaluate, failure_breakdown,
    home_loop, localized_candidates, make_world, oracle_best,
    smoke_indices,
)

START = Harness(top_k=3, recency=0.0, revision="all", fuzzy=False,
                answer="majority")
fails = []


def check(name, cond, detail=""):
    print(("ok " if cond else "FAIL ") + name + (f" [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- E6 units
cases = make_world(seed=0, n_cases=48)
mem_all = build_memory(cases, "all")
mem_latest = build_memory(cases, "latest")
n_hist = sum(1 for c in cases if c.historical)
n_syn = sum(1 for c in cases if c.query_attribute != c.attribute)
check("world mix", 0 < n_hist < 48 and 0 < n_syn < 48,
      f"historical={n_hist}/48 synonym={n_syn}/48")
# latest-only must drop at least one superseded version somewhere
dropped = sum(len(mem_all[i]) - len(mem_latest[i]) for i in range(48))
check("latest drops superseded versions", dropped > 0, f"dropped={dropped}")

start_res = evaluate(START, cases, mem_all)
bd = failure_breakdown(start_res)
check("start harness is weak", start_res.accuracy < 0.5,
      f"acc={start_res.accuracy:.3f} breakdown={bd}")
check("start failures are STALE-dominant",
      dominant_cause(start_res) == "STALE", f"cause={dominant_cause(start_res)}")
cands = localized_candidates(START, "STALE")
check("localized STALE first fix is recency",
      cands and cands[0].recency == 1.0 and cands[0].revision == "all",
      f"first={cands[0] if cands else None}")
check("localized STALE last fix is the risky latest-only",
      cands and cands[-1].revision == "latest",
      f"last={cands[-1] if cands else None}")

# ---------------------------------------------------------------- E4 oracle
oracle_h, oracle_r = oracle_best(cases)
check("oracle accuracy is 1.0", oracle_r.accuracy == 1.0,
      f"acc={oracle_r.accuracy}")
check("oracle harness is all/fuzzy/top1 with recency > 0",
      oracle_h.revision == "all" and oracle_h.recency > 0
      and oracle_h.answer == "top1" and oracle_h.fuzzy,
      f"{oracle_h}")
check("oracle tokens beat start tokens",
      oracle_r.mean_tokens < start_res.mean_tokens,
      f"oracle={oracle_r.mean_tokens} start={start_res.mean_tokens}")

# ---------------------------------------------------------------- E1 loop
loop = home_loop(cases, START, seed=0)
check("E1 loop beats start accuracy",
      loop.best_result.accuracy > loop.start_result.accuracy,
      f"{loop.start_result.accuracy:.3f} -> {loop.best_result.accuracy:.3f}")
check("E1 loop uses fewer tokens",
      loop.best_result.mean_tokens < loop.start_result.mean_tokens,
      f"{loop.start_result.mean_tokens} -> {loop.best_result.mean_tokens}")
check("E1 loop reaches oracle token cost",
      loop.best_result.mean_tokens == oracle_r.mean_tokens,
      f"loop={loop.best_result.mean_tokens} oracle={oracle_r.mean_tokens}")
check("E1 loop reaches oracle accuracy",
      loop.best_result.accuracy == oracle_r.accuracy,
      f"loop={loop.best_result.accuracy} oracle={oracle_r.accuracy}")
check("E1 loop within the 6-formal-eval budget",
      loop.formal_evals <= 6, f"formal={loop.formal_evals}")
check("E1 memory built once per revision policy",
      loop.constructions <= 2, f"constructions={loop.constructions}")
print(f"    E1 detail: best={loop.best} formal={loop.formal_evals} "
      f"smoke={loop.smoke_evals} rejects={loop.smoke_rejects} "
      f"tried={loop.candidates_tried}")

# ---------------------------------------------------------------- E2 localized vs random
SEEDS = list(range(12))
loc_gains, rnd_gains = [], []
for s in SEEDS:
    cs = make_world(seed=100 + s, n_cases=48)
    l = home_loop(cs, START, seed=s, localized=True)
    r = home_loop(cs, START, seed=s, localized=False)
    loc_gains.append((l.best_result.accuracy - l.start_result.accuracy)
                     / max(1, l.formal_evals))
    rnd_gains.append((r.best_result.accuracy - r.start_result.accuracy)
                     / max(1, r.formal_evals))
loc_mean = sum(loc_gains) / len(loc_gains)
rnd_mean = sum(rnd_gains) / len(rnd_gains)
print(f"    E2 detail: localized gain/formal-eval={loc_mean:.4f} "
      f"random={rnd_mean:.4f}")
check("E2 localized beats random per formal eval",
      loc_mean > rnd_mean + 0.05,
      f"localized={loc_mean:.4f} random={rnd_mean:.4f}")

# ---------------------------------------------------------------- E3 gating
# E3a: the smoke gate rejects the known trap. From the recency-fixed
# harness, the risky latest-only edit loses historical queries: its
# smoke accuracy is below the incumbent's, and its formal accuracy is
# below it too -- so the gate saves a formal evaluation that would
# have been wasted.
fixed = Harness(3, 1.0, "all", False, "majority")
fixed_res = evaluate(fixed, cases, build_memory(cases, "all"))
trap = Harness(3, 1.0, "latest", False, "majority")
sidx = smoke_indices(cases)
smoke_cases = [cases[i] for i in sidx]
smoke_mem_fixed = {j: build_memory(cases, "all")[i]
                   for j, i in enumerate(sidx)}
smoke_mem_trap = {j: build_memory(cases, "latest")[i]
                  for j, i in enumerate(sidx)}
s_fixed = evaluate(fixed, smoke_cases, smoke_mem_fixed)
s_trap = evaluate(trap, smoke_cases, smoke_mem_trap)
trap_res = evaluate(trap, cases, build_memory(cases, "latest"))
print(f"    E3a detail: trap smoke acc={s_trap.accuracy:.3f} vs "
      f"incumbent={s_fixed.accuracy:.3f}; formal trap={trap_res.accuracy:.3f} "
      f"vs incumbent={fixed_res.accuracy:.3f}")
check("E3a smoke gate rejects the trap candidate",
      s_trap.accuracy < s_fixed.accuracy - 1e-9,
      f"trap smoke={s_trap.accuracy} incumbent smoke={s_fixed.accuracy}")
check("E3a trap is genuinely worse formally too",
      trap_res.accuracy < fixed_res.accuracy - 1e-9,
      f"trap={trap_res.accuracy} incumbent={fixed_res.accuracy}")
# E3b: in the full localized loop the gate costs nothing and loses
# nothing (all proposed candidates are improvements, so it never
# fires) -- gated and ungated agree on formal evals and accuracy.
gated_formals, ungated_formals, gated_accs = [], [], []
for s in SEEDS:
    cs = make_world(seed=100 + s, n_cases=48)
    g = home_loop(cs, START, seed=s, gating=True)
    u = home_loop(cs, START, seed=s, gating=False)
    gated_formals.append(g.formal_evals)
    ungated_formals.append(u.formal_evals)
    gated_accs.append(g.best_result.accuracy)
g_mean = sum(gated_formals) / len(gated_formals)
u_mean = sum(ungated_formals) / len(ungated_formals)
print(f"    E3b detail: gated formal-evals mean={g_mean:.2f} "
      f"ungated={u_mean:.2f} gated acc mean="
      f"{sum(gated_accs) / len(gated_accs):.3f}")
check("E3b gating never costs formal evaluations",
      g_mean <= u_mean, f"gated={g_mean:.2f} ungated={u_mean:.2f}")
check("E3b gating keeps the accuracy",
      all(a >= 0.95 for a in gated_accs),
      f"min gated acc={min(gated_accs):.3f}")

# ---------------------------------------------------------------- E5 determinism
loop2 = home_loop(cases, START, seed=0)
check("E5 deterministic loop",
      [(h, a) for h, a, _ in loop.history]
      == [(h, a) for h, a, _ in loop2.history],
      f"histories equal: {len(loop.history)} steps")

print()
if fails:
    print(f"FAILED {len(fails)} checks: {fails}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
