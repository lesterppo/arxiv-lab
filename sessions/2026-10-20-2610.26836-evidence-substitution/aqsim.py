"""AQSim — deterministic synthetic aqueous-speciation simulator.

Stand-in for PHREEQC in the evidence-substitution replication
(arXiv 2610.26836, "Same Simulation, Different Evidence").

The paper's mechanism: a deterministic simulator emits a long report; the
harness keeps it out of context and the agent retrieves via a section index.
Agents compute the right value but answer from their own recomputed outputs
("evidence substitution") and misround to a neighboring option.

This module is stdlib-only and fully deterministic given a seed.
"""
import math
import random
from decimal import Decimal, ROUND_HALF_UP

MINERALS = ("calcite", "gypsum", "halite", "dolomite")

# per-mineral coefficients: SI = c0 + c1*pH + c2*log10(Ca) + c3*log10(HCO3)
#   + c4*log10(SO4) + c5*(temp-25)/25 + c6*log10(Mg) + jitter
_COEF = {
    "calcite":  (-3.10, 0.62, 0.55, 0.48, 0.02, -0.10, 0.05),
    "gypsum":   (-2.40, 0.10, 0.50, 0.05, 0.55, -0.05, 0.02),
    "halite":   (-4.20, 0.05, 0.02, 0.02, 0.03,  0.02, 0.02),
    "dolomite": (-5.60, 0.95, 0.55, 0.60, 0.02, -0.12, 0.45),
}
_IONS = ("Na", "Ca", "Mg", "Cl", "SO4", "HCO3")


def sample_params(seed):
    """Deterministic water-chemistry parameters."""
    rng = random.Random(seed)
    params = {
        "pH": round(rng.uniform(6.0, 8.5), 2),
        "temp_C": round(rng.uniform(10.0, 40.0), 1),
    }
    for ion in _IONS:
        params[ion] = round(10 ** rng.uniform(-1.0, 2.0), 4)  # 0.1..100 mmol/L
    return params


def _si(mineral, p, jitter):
    c0, c1, c2, c3, c4, c5, c6 = _COEF[mineral]
    v = (c0 + c1 * p["pH"]
         + c2 * math.log10(p["Ca"]) + c3 * math.log10(p["HCO3"])
         + c4 * math.log10(p["SO4"]) + c5 * (p["temp_C"] - 25.0) / 25.0
         + c6 * math.log10(p["Mg"]) + jitter)
    return v


def simulate(params, seed):
    """Run the simulator. Returns dict with sections and raw values."""
    rng = random.Random(seed ^ 0xA9)
    jitters = {m: rng.uniform(-0.02, 0.02) for m in MINERALS}
    si = {m: _si(m, params, jitters[m]) for m in MINERALS}
    # speciation activities (decorative but deterministic)
    spec = {
        "Ca++": params["Ca"] * 0.62, "Mg++": params["Mg"] * 0.58,
        "Na+": params["Na"] * 0.81, "Cl-": params["Cl"] * 0.79,
        "SO4--": params["SO4"] * 0.35, "HCO3-": params["HCO3"] * 0.74,
    }
    masses = {m: max(0.0, 10.0 * (10 ** si[m] - 1.0) / (10 ** si[m] + 1.0))
              for m in MINERALS}
    return {"params": dict(params), "si": si, "speciation": spec,
            "masses": masses, "seed": seed}


def fmt_report(sim):
    """Render the full simulator report (the authoritative output)."""
    p = sim["params"]
    lines = ["=== AQSIM GEOCHEMISTRY REPORT ===",
             "--- INPUTS ---"]
    lines.append(f"pH = {p['pH']:.2f}")
    lines.append(f"temp_C = {p['temp_C']:.1f}")
    for ion in _IONS:
        lines.append(f"[{ion}] = {p[ion]:.4f} mmol/L")
    lines.append("--- SPECIATION ---")
    for sp, a in sim["speciation"].items():
        lines.append(f"activity({sp}) = {a:.6f}")
    lines.append("--- SATURATION ---")
    for m in MINERALS:
        lines.append(f"SI({m}) = {sim['si'][m]:.6f}")
    lines.append("--- MINERAL_MASSES ---")
    for m in MINERALS:
        lines.append(f"mass({m}) = {sim['masses'][m]:.6f} g")
    lines.append("=== END REPORT ===")
    return "\n".join(lines)


def get_section(report, name):
    """Extract one section from a report (the retrieval tool)."""
    lines = report.splitlines()
    out, capture = [], False
    for ln in lines:
        if ln.strip() == f"--- {name} ---":
            capture = True
            continue
        if capture and ln.startswith("---"):
            break
        if capture:
            out.append(ln)
    return "\n".join(out)


def round3(v):
    """Half-up rounding to 3 decimals (option keying)."""
    return float(Decimal(str(v)).quantize(Decimal("0.001"),
                                          rounding=ROUND_HALF_UP))


def make_question(item_seed):
    """Build one multiple-choice item with a verifiable keyed answer.

    Returns dict: seed, params, mineral, report, question, options (list of
    (letter, value)), answer (letter), keyed_value (rounded true SI).
    The true value is kept > 5e-4 away from any .0005 rounding boundary so
    the key is unambiguous.
    """
    rng = random.Random(item_seed)
    params = sample_params(item_seed)
    mineral = rng.choice(MINERALS)
    sim = simulate(params, item_seed)
    true_v = sim["si"][mineral]
    # avoid rounding-boundary ambiguity
    frac = abs(true_v * 1000.0 - round(true_v * 1000.0))
    assert frac > 5e-4 or abs(frac - 0.5) > 1e-9, "boundary value"
    keyed = round3(true_v)
    assert abs(abs(true_v - keyed) - 0.0005) > 1e-9
    cands = [keyed, round(keyed + 0.001, 3), round(keyed - 0.001, 3),
             round(keyed + 0.007, 3)]
    assert len(set(cands)) == 4
    order = rng.sample(range(4), 4)
    letters = "ABCD"
    options = [(letters[i], cands[j]) for i, j in enumerate(order)]
    answer = letters[order.index(0)]
    q = (f"What is the saturation index SI({mineral}) at pH={params['pH']:.2f}, "
         f"temp={params['temp_C']:.1f}C? Round to 3 decimals. "
         f"Options: " + ", ".join(f"{L}={v:.3f}" for L, v in options))
    return {"seed": item_seed, "params": params, "mineral": mineral,
            "report": fmt_report(sim), "true_si": true_v,
            "keyed_value": keyed, "question": q, "options": options,
            "answer": answer}
