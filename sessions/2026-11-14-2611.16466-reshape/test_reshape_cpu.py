#!/usr/bin/env python3
"""Deterministic CPU test for ReSHAPE (arXiv:2611.16466).

Pre-registered assertions (paper figures in brackets):
  A1  travel: F1_SHOW - F1_ASK >= 0.05 [paper: showing plans raises F1
      by 9.7 points over asking, in travel, with two model families]
  A2  travel: admission_SHOW > admission_ASK
  A3  travel: planner success SHOW > ASK, lift >= 1.15
  A4  shopping: F1_SHOW - F1_ASK >= 0.03 (same direction, second domain)
  A5  pooled ORACLE admission > 0.52 [paper: the one correct trade-off
      alone admits over 52% of the acceptable plans]
  A6  pooled admission ordering: none < ask < show < oracle; F1 ordering:
      show > ask and oracle >= show
  A7  pooled planner success: SHOW > ASK (lift >= 1.15); search-space
      quality predicts what the planner finds: Pearson r between arm F1
      and planner success across the 8 arm-domain points > 0.8
  A8  pooled planner success: ASK < NONE (asking and relaxing the wrong
      dimension unconditionally *dilutes* the space) and SHOW > NONE
  A9  sensitivity: SHOW F1 non-decreasing in k_show (4/8/16); the SHOW
      gain from 4 -> 16 shown plans exceeds the ASK gain from asking 1 ->
      3 questions (concrete reactions scale; abstract questions do not)
  A10 invariants + determinism: metrics in [0,1]; S0 subset of every
      revised space; planner success monotone in K (1/2/4) for SHOW;
      identical seed -> identical metrics on re-run
  A11 unit cases: oracle admits every trade-off plan; hand-checked
      admission/precision/F1 on a known space; planner with K >= |S|
      succeeds iff the space contains an acceptable plan; RNG streams
      reproduce exactly
  A12 paired rescue: pooled, the fraction of instances where the SHOW
      planner succeeds and the NONE planner fails exceeds the reverse
      (SHOW rescues more instances than it loses)
  A13 robustness: F1_SHOW > F1_ASK in both domains at BONUS_FRAC 0.15
      and 0.35 as well as the main 0.25 (ordering is not bonus luck)

Writes results.json next to this file. Stdlib only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reshape
from reshape import (
    DOMAINS, N_ACCEPTABLE, RNG, STRATEGIES, gen_instances, pearson,
    plan, revise, revise_oracle, run_experiment, space_metrics,
    _strategy_seed,
)

SEED = 20261114
N_INSTANCES = 300

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def main():
    results = {}
    instances = {}
    for dom in ("travel", "shopping"):
        instances[dom] = gen_instances(dom, N_INSTANCES, SEED)
        results[dom] = run_experiment(dom, instances[dom], SEED)
    print(f"main experiment ({N_INSTANCES} instances/domain, seed {SEED}, "
          f"planner K={reshape.PLANNER_K}, show K={reshape.SHOW_K}):")
    for dom in ("travel", "shopping"):
        for s in STRATEGIES:
            m = results[dom][s]
            print(f"  {dom:9s} {s:7s} adm={m['admission']:.3f} "
                  f"prec={m['precision']:.3f} f1={m['f1']:.3f} "
                  f"succ={m['planner_success']:.3f} |S|={m['mean_space_size']:.1f}")

    print("assertions:")
    tr, sh = results["travel"], results["shopping"]
    check("A1 travel F1_SHOW - F1_ASK >= 0.05",
          tr["show"]["f1"] - tr["ask"]["f1"] >= 0.05,
          f"got {tr['show']['f1']:.3f} - {tr['ask']['f1']:.3f} "
          f"= {tr['show']['f1'] - tr['ask']['f1']:.3f}")
    check("A2 travel admission_SHOW > admission_ASK",
          tr["show"]["admission"] > tr["ask"]["admission"],
          f"got {tr['show']['admission']:.3f} vs {tr['ask']['admission']:.3f}")
    check("A3 travel planner SHOW > ASK, lift >= 1.15",
          tr["show"]["planner_success"] >= 1.15 * tr["ask"]["planner_success"],
          f"got {tr['show']['planner_success']:.3f} vs "
          f"{tr['ask']['planner_success']:.3f} "
          f"(lift {tr['show']['planner_success'] / tr['ask']['planner_success']:.2f}x)")
    check("A4 shopping F1_SHOW - F1_ASK >= 0.03",
          sh["show"]["f1"] - sh["ask"]["f1"] >= 0.03,
          f"got {sh['show']['f1']:.3f} - {sh['ask']['f1']:.3f} "
          f"= {sh['show']['f1'] - sh['ask']['f1']:.3f}")
    check("A5 pooled ORACLE admission > 0.52",
          sum(results[d]["oracle"]["admission"] for d in results) / 2 > 0.52,
          f"got {sum(results[d]['oracle']['admission'] for d in results) / 2:.3f}")
    # A6 admission ordering (pooled) + F1 ordering
    adm = {s: sum(results[d][s]["admission"] for d in results) / 2
           for s in STRATEGIES}
    f1 = {s: sum(results[d][s]["f1"] for d in results) / 2
          for s in STRATEGIES}
    check("A6a pooled admission: none < ask < show < oracle",
          adm["none"] < adm["ask"] < adm["show"] < adm["oracle"],
          f"got {['%.3f' % adm[s] for s in STRATEGIES]}")
    check("A6b pooled F1: show > ask and oracle >= show",
          f1["show"] > f1["ask"] and f1["oracle"] >= f1["show"],
          f"got {['%.3f' % f1[s] for s in STRATEGIES]}")
    # A7 planner lift + correlation
    succ = {s: sum(results[d][s]["planner_success"] for d in results) / 2
            for s in STRATEGIES}
    check("A7a pooled planner SHOW > ASK, lift >= 1.15",
          succ["show"] >= 1.15 * succ["ask"],
          f"got {succ['show']:.3f} vs {succ['ask']:.3f} "
          f"(lift {succ['show'] / succ['ask']:.2f}x)")
    xs = [results[d][s]["f1"] for d in ("travel", "shopping") for s in STRATEGIES]
    ys = [results[d][s]["planner_success"] for d in ("travel", "shopping")
          for s in STRATEGIES]
    r = pearson(xs, ys)
    check("A7b F1 predicts planner success: Pearson r > 0.8",
          r > 0.8, f"got r={r:.3f}")
    # A8 dilution finding
    check("A8a pooled planner ASK < NONE (asking dilutes the space)",
          succ["ask"] < succ["none"],
          f"got ask {succ['ask']:.3f} vs none {succ['none']:.3f}")
    check("A8b pooled planner SHOW > NONE",
          succ["show"] > succ["none"],
          f"got show {succ['show']:.3f} vs none {succ['none']:.3f}")
    # A9 sensitivity: show K and ask Q
    show_by_k, ask_by_q = {}, {}
    for k in (4, 8, 16):
        show_by_k[k] = sum(
            run_experiment(dom, instances[dom], SEED, show_k=k)[ "show"]["f1"]
            for dom in ("travel", "shopping")) / 2
    for q in (1, 3):
        ask_by_q[q] = sum(
            run_experiment(dom, instances[dom], SEED, ask_q=q)["ask"]["f1"]
            for dom in ("travel", "shopping")) / 2
    print(f"  sensitivity show K 4/8/16: F1="
          f"{['%.3f' % show_by_k[k] for k in (4, 8, 16)]}; "
          f"ask Q 1/3: F1={['%.3f' % ask_by_q[q] for q in (1, 3)]}")
    check("A9a SHOW F1 non-decreasing in k_show",
          show_by_k[4] <= show_by_k[8] <= show_by_k[16],
          f"got {['%.3f' % show_by_k[k] for k in (4, 8, 16)]}")
    check("A9b SHOW gain 4->16 exceeds ASK gain Q 1->3",
          show_by_k[16] - show_by_k[4] > ask_by_q[3] - ask_by_q[1],
          f"got SHOW +{show_by_k[16] - show_by_k[4]:.3f} vs "
          f"ASK +{ask_by_q[3] - ask_by_q[1]:.3f}")
    # A10 invariants + determinism
    ok_range = all(0.0 <= results[d][s][k] <= 1.0
                   for d in results for s in STRATEGIES
                   for k in ("admission", "precision", "f1", "planner_success"))
    check("A10a all metrics in [0, 1]", ok_range)
    subset_ok = True
    for dom in ("travel", "shopping"):
        for inst in instances[dom][:50]:
            for s in ("ask", "show", "oracle"):
                rng = RNG(_strategy_seed(SEED, inst.idx, s))
                if not inst.spec0 <= revise(inst, s, rng):
                    subset_ok = False
    check("A10b S0 is a subset of every revised space", subset_ok)
    succ_k = {}
    for k in (1, 2, 4):
        succ_k[k] = sum(
            run_experiment(dom, instances[dom], SEED, planner_k=k)["show"]["planner_success"]
            for dom in ("travel", "shopping")) / 2
    print(f"  sensitivity planner K 1/2/4 (SHOW): succ="
          f"{['%.3f' % succ_k[k] for k in (1, 2, 4)]}")
    check("A10c SHOW planner success monotone in K",
          succ_k[1] <= succ_k[2] <= succ_k[4],
          f"got {['%.3f' % succ_k[k] for k in (1, 2, 4)]}")
    m2 = {dom: run_experiment(dom, instances[dom], SEED)
          for dom in ("travel", "shopping")}
    check("A10d same seed -> identical metrics on re-run", m2 == results)
    # A11 unit cases
    inst0 = instances["travel"][0]
    oracle_space = revise_oracle(inst0)
    tradeoff_plans = [i for i, p in enumerate(inst0.plans)
                      if inst0.tradeoff_holds(p)]
    check("A11a oracle admits every trade-off plan",
          all(i in oracle_space for i in tradeoff_plans),
          f"{len(tradeoff_plans)} trade-off plans")
    # Hand-checked metrics on a known space: take spec0 of instance 0.
    m0 = space_metrics(inst0, inst0.spec0)
    inter = len(inst0.spec0 & inst0.acceptable)
    exp_adm = inter / len(inst0.acceptable)
    exp_prec = inter / len(inst0.spec0)
    exp_f1 = 2 * exp_prec * exp_adm / (exp_prec + exp_adm)
    check("A11b space_metrics matches hand computation",
          abs(m0["admission"] - exp_adm) < 1e-12
          and abs(m0["precision"] - exp_prec) < 1e-12
          and abs(m0["f1"] - exp_f1) < 1e-12,
          f"adm={m0['admission']:.3f} prec={m0['precision']:.3f} f1={m0['f1']:.3f}")
    rng_a = RNG(_strategy_seed(SEED, 0, "show"))
    space_full = revise(inst0, "show", rng_a, k_show=10 ** 9)
    check("A11c planner with K >= |S| succeeds iff S holds an acceptable plan",
          plan(inst0, space_full, RNG(5), k=len(space_full)) ==
          (len(space_full & inst0.acceptable) > 0))
    r1, r2 = RNG(12345), RNG(12345)
    check("A11d RNG streams reproduce exactly",
          [r1.next_u64() for _ in range(5)] == [r2.next_u64() for _ in range(5)])
    # A12 paired rescue: per-instance planner outcomes, show vs none
    rescue = loss = 0
    n_tot = 0
    for dom in ("travel", "shopping"):
        for inst in instances[dom]:
            outs = {}
            for s in ("none", "show"):
                rng = RNG(_strategy_seed(SEED, inst.idx, s))
                space = revise(inst, s, rng)
                outs[s] = plan(inst, space, rng, k=reshape.PLANNER_K)
            n_tot += 1
            if outs["show"] and not outs["none"]:
                rescue += 1
            elif outs["none"] and not outs["show"]:
                loss += 1
    print(f"  paired rescue: SHOW succeeds/NONE fails on {rescue}/{n_tot} "
          f"instances; reverse on {loss}/{n_tot}")
    check("A12 SHOW rescues more instances than it loses",
          rescue > loss, f"rescue {rescue} vs loss {loss}")
    # A13 robustness across BONUS_FRAC
    robust = True
    detail = {}
    for bf in (0.15, 0.35):
        reshape.BONUS_FRAC = bf
        for dom in ("travel", "shopping"):
            insts = gen_instances(dom, N_INSTANCES, SEED)
            res = run_experiment(dom, insts, SEED)
            detail[(bf, dom)] = (res["show"]["f1"], res["ask"]["f1"])
            if not res["show"]["f1"] > res["ask"]["f1"]:
                robust = False
    reshape.BONUS_FRAC = 0.25
    check("A13 F1_SHOW > F1_ASK at BONUS_FRAC 0.15 and 0.35, both domains",
          robust,
          "; ".join(f"bf={bf} {dom}: show {v[0]:.3f} vs ask {v[1]:.3f}"
                    for (bf, dom), v in detail.items()))

    out = {
        "paper": "2611.16466",
        "seed": SEED,
        "n_instances_per_domain": N_INSTANCES,
        "planner_k": reshape.PLANNER_K,
        "show_k": reshape.SHOW_K,
        "bonus_frac": 0.25,
        "n_constrained": reshape.N_CONSTRAINED,
        "n_acceptable": N_ACCEPTABLE,
        "domains": results,
        "pooled": {
            s: {k: sum(results[d][s][k] for d in results) / 2
                for k in ("admission", "precision", "f1", "planner_success",
                          "mean_space_size")}
            for s in STRATEGIES
        },
        "pearson_f1_vs_planner_success": r,
        "lifts": {
            "show_over_ask_f1_travel": tr["show"]["f1"] - tr["ask"]["f1"],
            "show_over_ask_f1_shopping": sh["show"]["f1"] - sh["ask"]["f1"],
            "show_over_ask_planner_lift_pooled":
                succ["show"] / succ["ask"],
            "oracle_over_none_planner_lift_pooled":
                succ["oracle"] / succ["none"],
            "show_over_none_planner_lift_pooled":
                succ["show"] / succ["none"],
        },
        "sensitivity": {
            "show_f1_by_k_show": {str(k): show_by_k[k] for k in (4, 8, 16)},
            "ask_f1_by_n_questions": {str(q): ask_by_q[q] for q in (1, 3)},
            "show_planner_success_by_k": {str(k): succ_k[k] for k in (1, 2, 4)},
        },
        "paired_rescue": {"show_rescues": rescue, "show_loses": loss,
                          "n_instances": n_tot},
        "robustness_bonus_frac": {
            f"{bf}:{dom}": {"show_f1": v[0], "ask_f1": v[1]}
            for (bf, dom), v in detail.items()
        },
        "assertions_passed": PASSED,
        "assertions_failed": FAILED,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)


if __name__ == "__main__":
    main()
