"""Session experiment: component routing (arXiv:2610.01787), agent track, CPU.

Reproduces the paper's core mechanism synthetically:
 (i)   experience components (locators, procedures, state facts, lessons)
       with measurable recurrence and state-conditionality route to
       weights vs context;
 (ii)  a rule fit on (recurrence, state-conditionality) from two synthetic
       "backbone families" is tested on a held-out family's 24 cells
       (4 types x 2 envs x 3 seeds), mirroring the paper's 24/24 claim;
 (iii) toy demonstrations: note readout degrades after weights-write
       (most for highest recurrence), context gain grows with the
       information gap, weights gain shrinks with the policy gap.

Writes results.json next to this file and prints a summary.
Pure stdlib. Deterministic (MASTER_SEED fixed before any run).
"""

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from arxiv_lab.routing import (
    COMPONENT_TYPES, CONTEXT, WEIGHTS, Component, fit_routing_rule,
)

MASTER_SEED = 20261004
HERE = os.path.dirname(os.path.abspath(__file__))

# ── Synthetic world ────────────────────────────────────────────────────
# Per-type property ranges: (r_lo, r_hi, s_lo, s_hi).
TYPE_PROPS = {
    "locator":    (0.70, 0.95, 0.05, 0.30),
    "lesson":     (0.60, 0.90, 0.10, 0.35),
    "procedure":  (0.15, 0.45, 0.60, 0.90),
    "state_fact": (0.10, 0.40, 0.65, 0.95),
}
ENVS = {"stable-ui": (0.05, -0.05), "dynamic-ui": (-0.05, 0.05)}
SEEDS = (11, 22, 33)
N_PER_CELL = 40

# Backbone families: (kw, kc) gain scalings — "different gain scalings".
FAMS = {
    "memorizer":  (1.50, 0.75),   # consolidates into weights easily
    "in-context": (0.65, 1.35),   # strong in-context learner
    "balanced":   (1.00, 1.00),   # held-out family
}
NOISE_SD = 0.15


def clamp01(x):
    return max(0.01, min(0.99, x))


def gen_pool():
    """One component pool; cells keyed (type, env, seed)."""
    rng = random.Random(MASTER_SEED)
    pool = []
    for kind in COMPONENT_TYPES:
        r_lo, r_hi, s_lo, s_hi = TYPE_PROPS[kind]
        for env, (dr, ds) in ENVS.items():
            for sd in SEEDS:
                cell_rng = random.Random(rng.randrange(1 << 30))
                for _ in range(N_PER_CELL):
                    r = clamp01(cell_rng.uniform(r_lo, r_hi) + dr)
                    s = clamp01(cell_rng.uniform(s_lo, s_hi) + ds)
                    pool.append({"comp": Component(kind, r, s),
                                 "type": kind, "env": env, "seed": sd})
    return pool


def observe(pool, fam, noise_seed):
    """Simulate gain-in-weights vs gain-in-context for one family."""
    kw, kc = FAMS[fam]
    rng = random.Random(noise_seed)
    rows = []
    for item in pool:
        c = item["comp"]
        gw = kw * (0.5 + 1.2 * c.recurrence - 0.8 * c.state_conditionality) \
            + rng.gauss(0.0, NOISE_SD)
        gc = kc * (0.5 - 0.6 * c.recurrence + 1.1 * c.state_conditionality) \
            + rng.gauss(0.0, NOISE_SD)
        rows.append({**item, "gw": gw, "gc": gc,
                     "winner": WEIGHTS if gw >= gc else CONTEXT})
    return rows


def fit_rule(rows_a, rows_b):
    rs = [r["comp"].recurrence for r in rows_a] + \
         [r["comp"].recurrence for r in rows_b]
    ss = [r["comp"].state_conditionality for r in rows_a] + \
         [r["comp"].state_conditionality for r in rows_b]
    ys = [1 if r["winner"] == WEIGHTS else 0 for r in rows_a] + \
         [1 if r["winner"] == WEIGHTS else 0 for r in rows_b]
    return fit_routing_rule(rs, ss, ys)


def heldout_cells(rule, rows):
    """24 cells on the held-out family: rule prediction vs majority winner."""
    cells = {}
    for r in rows:
        key = (r["type"], r["env"], r["seed"])
        cells.setdefault(key, []).append(r)
    score, detail = 0, []
    for key in sorted(cells):
        items = cells[key]
        rbar = sum(i["comp"].recurrence for i in items) / len(items)
        sbar = sum(i["comp"].state_conditionality for i in items) / len(items)
        pred = rule.destination((rbar, sbar))
        w = sum(1 for i in items if i["winner"] == WEIGHTS)
        truth = WEIGHTS if w >= len(items) / 2 else CONTEXT
        ok = pred == truth
        score += ok
        detail.append({"cell": key, "pred": pred, "truth": truth,
                       "rbar": round(rbar, 3), "sbar": round(sbar, 3),
                       "w_frac": round(w / len(items), 2), "match": ok})
    return score, len(cells), detail


def intervention_check(rule, comp, which, delta):
    """Move one property toward the boundary; report p_weights shift."""
    r, s = comp.recurrence, comp.state_conditionality
    p0 = rule.p_weights(r, s)
    if which == "recurrence":
        r2 = clamp01(r + delta)
        s2 = s
    else:
        r2, s2 = r, clamp01(s + delta)
    p1 = rule.p_weights(r2, s2)
    return {"before": (round(r, 3), round(s, 3), round(p0, 3)),
            "after": (round(r2, 3), round(s2, 3), round(p1, 3)),
            "dist_to_boundary_before": round(abs(p0 - 0.5), 3),
            "dist_to_boundary_after": round(abs(p1 - 0.5), 3),
            "dest_before": rule.destination((r, s)),
            "dest_after": rule.destination((r2, s2))}


def observed_winner_after(comp_new, fam, trials=400, seed=7):
    """Majority observed winner of an intervened component under a family."""
    kw, kc = FAMS[fam]
    rng = random.Random(seed)
    w = 0
    r, s = comp_new
    for _ in range(trials):
        gw = kw * (0.5 + 1.2 * r - 0.8 * s) + rng.gauss(0.0, NOISE_SD)
        gc = kc * (0.5 - 0.6 * r + 1.1 * s) + rng.gauss(0.0, NOISE_SD)
        w += gw >= gc
    return WEIGHTS if w >= trials / 2 else CONTEXT


def baselines(rule, rows):
    """Total gain per routing policy on the held-out family."""
    tot = {"rule": 0.0, "all_weights": 0.0, "all_context": 0.0,
           "random": 0.0, "oracle": 0.0}
    for it in rows:
        dest = rule.destination(it["comp"])
        tot["rule"] += it["gw"] if dest == WEIGHTS else it["gc"]
        tot["all_weights"] += it["gw"]
        tot["all_context"] += it["gc"]
        tot["random"] += 0.5 * (it["gw"] + it["gc"])
        tot["oracle"] += max(it["gw"], it["gc"])
    n = len(rows)
    return {k: v / n for k, v in tot.items()}


# ── Claim (iii) toys ───────────────────────────────────────────────────
def blend_readout_quality(recurrence, n=64, q=0.3, trials=800):
    """Note readout after m overlapping weights-writes (m ~ recurrence).

    Instances of a recurring component are genuinely different draws around
    a prototype (e.g. a locator's coordinates differ per episode), not noisy
    copies of one truth. The note is instance x1 (an exact copy kept in
    context reads at 1.0). Writing into weights blends the m instances into
    one vector; readout quality = bit-agreement of sign(blend) with the
    note x1. More recurrences -> the note's own contribution is diluted ->
    readout converges to prototype-vs-note agreement (1-q) < 1.
    """
    rng = random.Random(MASTER_SEED + 99)
    m = 1 + round(9 * recurrence)
    acc = 0.0
    for _ in range(trials):
        proto = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        insts = [[-b if rng.random() < q else b for b in proto]
                 for _ in range(m)]
        note = insts[0]
        mean = [sum(col) / m for col in zip(*insts)]

        def _signbit(av, b):
            if av > 0:
                s = 1
            elif av < 0:
                s = -1
            else:
                s = 1 if rng.random() < 0.5 else -1  # random tie-break
            return s == b

        acc += sum(1 for av, b in zip(mean, note) if _signbit(av, b)) / n
    return acc / trials


def info_gap_curve(q=0.85):
    """Context gain vs information gap (fixed retrieval quality q)."""
    return [(round(g, 2), round(q * g, 4))
            for g in (i / 10 for i in range(0, 10))]


def policy_gap_curve(dim=8):
    """Weights gain vs policy gap: task target drifts from weights vector."""
    rng = random.Random(MASTER_SEED + 7)
    w = [rng.gauss(0, 1) for _ in range(dim)]
    nrm = math.sqrt(sum(x * x for x in w))
    w = [x / nrm for x in w]
    u = [rng.gauss(0, 1) for _ in range(dim)]
    u = [x - sum(a * b for a, b in zip(w, u)) * wi
         for x, wi in zip(u, w)]
    nrm = math.sqrt(sum(x * x for x in u)) or 1.0
    u = [x / nrm for x in u]
    out = []
    for d in (i / 4 for i in range(0, 9)):
        t = [wi + d * ui for wi, ui in zip(w, u)]
        nrm = math.sqrt(sum(x * x for x in t))
        t = [x / nrm for x in t]
        cos = sum(a * b for a, b in zip(w, t))
        out.append((round(d, 2), round(1 - cos, 4), round(cos, 4)))
    return out


def main():
    pool = gen_pool()
    rows_mem = observe(pool, "memorizer", MASTER_SEED + 1)
    rows_ic = observe(pool, "in-context", MASTER_SEED + 2)
    rows_bal = observe(pool, "balanced", MASTER_SEED + 3)

    # (i) type-level destination pattern per family
    pattern = {}
    for fam, rows in (("memorizer", rows_mem), ("in-context", rows_ic),
                      ("balanced", rows_bal)):
        per_type = {}
        for kind in COMPONENT_TYPES:
            items = [r for r in rows if r["type"] == kind]
            wfrac = sum(1 for r in items if r["winner"] == WEIGHTS) / len(items)
            per_type[kind] = round(wfrac, 3)
        pattern[fam] = per_type

    # (ii) rule fit on two families, 24-cell test on held-out
    rule = fit_rule(rows_mem, rows_ic)
    score, total, detail = heldout_cells(rule, rows_bal)

    # interventions toward the boundary
    proc = Component("procedure", 0.30, 0.72)
    lesson = Component("lesson", 0.78, 0.20)
    iv1 = intervention_check(rule, proc, "recurrence", +0.35)
    iv1["observed_winner_after"] = observed_winner_after((0.65, 0.72),
                                                         "balanced")
    iv2 = intervention_check(rule, lesson, "state_conditionality", +0.35)
    iv2["observed_winner_after"] = observed_winner_after((0.78, 0.55),
                                                         "balanced")

    # routing-by-rule vs whole-trajectory baselines
    base = baselines(rule, rows_bal)
    best_single = max(base["all_weights"], base["all_context"])
    headroom = base["oracle"] - best_single
    captured = (base["rule"] - best_single) / headroom if headroom else 0.0

    # (iii) toys
    recs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    readout = [(r, round(blend_readout_quality(r), 4)) for r in recs]
    igap = info_gap_curve()
    pgap = policy_gap_curve()

    results = {
        "paper": "arXiv:2610.01787",
        "master_seed": MASTER_SEED,
        "n_items_per_family": len(pool),
        "type_destination_pattern_wfrac": pattern,
        "rule_coefs": rule.as_dict(),
        "heldout_cells": {"match": score, "total": total, "detail": detail},
        "interventions": {"procedure_recurrence_boost": iv1,
                          "lesson_statecond_boost": iv2},
        "baselines_mean_gain_per_item": {k: round(v, 4)
                                         for k, v in base.items()},
        "rule_margin_over_best_single": round(base["rule"] - best_single, 4),
        "rule_headroom_captured": round(captured, 4),
        "readout_quality_vs_recurrence": readout,
        "context_gain_vs_info_gap": igap,
        "weights_gain_vs_policy_gap": pgap,  # (delta, gap, gain)
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("=== (i) destination pattern: P(weights wins) per type/family ===")
    for fam, pt in pattern.items():
        print("  %-10s %s" % (fam, pt))
    print("=== (ii) rule coefs a_r=%.3f a_s=%.3f b=%.3f ==="
          % (rule.a_r, rule.a_s, rule.b))
    print("held-out 24-cell recovery: %d/%d" % (score, total))
    for d in detail:
        if not d["match"]:
            print("  MISS", d)
    print("intervention 1 (procedure, r+0.35):", iv1["before"], "->",
          iv1["after"], "| dist to boundary %.3f -> %.3f" %
          (iv1["dist_to_boundary_before"], iv1["dist_to_boundary_after"]),
          "| dest %s -> %s, observed winner after: %s" %
          (iv1["dest_before"], iv1["dest_after"],
           iv1["observed_winner_after"]))
    print("intervention 2 (lesson, s+0.35):", iv2["before"], "->",
          iv2["after"], "| dist to boundary %.3f -> %.3f" %
          (iv2["dist_to_boundary_before"], iv2["dist_to_boundary_after"]),
          "| dest %s -> %s, observed winner after: %s" %
          (iv2["dest_before"], iv2["dest_after"],
           iv2["observed_winner_after"]))
    print("=== baselines: mean gain/item on held-out family ===")
    for k, v in base.items():
        print("  %-12s %.4f" % (k, v))
    print("rule margin over best single destination: %.4f "
          "(headroom captured: %.1f%%)" % (base["rule"] - best_single,
                                           100 * captured))
    print("=== (iii) readout quality after weights-write vs recurrence ===")
    for r, q in readout:
        print("  r=%.2f k=%2d quality=%.4f (context copy: 1.0000)"
              % (r, 1 + round(9 * r), q))
    print("=== (iii) context gain vs info gap (q=0.85):",
          [g for _, g in igap])
    print("=== (iii) weights gain vs policy gap:",
          [(d, g) for d, _, g in pgap])


if __name__ == "__main__":
    main()
