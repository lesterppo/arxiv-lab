#!/usr/bin/env python3
"""CPU checks for the LoRA-reasoning Colab driver (arXiv:2611.17072).

Validates the pure-Python logic without a model or downloads, using
tiny synthetic Banking77-shaped data:
  - normalize_label / parse_pred (label normalisation + parse fails)
  - make_prompt / make_trace_prompt (format, label list, exemplars)
  - extract_trace (gold-label gate, empty/short rejection)
  - compute_ece / macro_f1 (hand-computed values)
  - build_arms (matched items/N, R label-only, T trace+label, short-pool error)
  - strided_indices (retention-subset coverage)
Runs offline, stdlib + the driver module (imported with the Colab-only
imports guarded: the driver imports pandas at top level, so this test
stubs it out when pandas is unavailable).
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas  # noqa: F401
except ImportError:
    sys.modules["pandas"] = types.ModuleType("pandas")

import colab_lora_reasoning_driver_v4 as d

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


LABELS = ["card_arrival", "lost_or_stolen_card", "card_linking",
          "exchange_rate", "pin_blocked"]

# --- normalize_label / parse_pred
check("normalize_label exact", d.normalize_label("card_arrival") == "card_arrival")
check("normalize_label case+space",
      d.normalize_label("Card Arrival") == "card_arrival")
check("normalize_label hyphens",
      d.normalize_label("lost-or-stolen-card") == "lost_or_stolen_card")
check("parse_pred exact", d.parse_pred("card_arrival", LABELS) == "card_arrival")
check("parse_pred case", d.parse_pred("Card Arrival", LABELS) == "card_arrival")
check("parse_pred sentence",
      d.parse_pred("I think this is about card_arrival.", LABELS) == "card_arrival")
check("parse_pred fail", d.parse_pred("I have no idea", LABELS) is None)
check("parse_pred empty", d.parse_pred("", LABELS) is None)
check("parse_pred first line wins",
      d.parse_pred("exchange_rate\ncard_arrival", LABELS) == "exchange_rate")
# v3: reasoning-arm generations -- label on a later "Category:" line
check("parse_pred category line",
      d.parse_pred("The user asks about delivery time.\nCategory: card_arrival",
                   LABELS) == "card_arrival")
check("parse_pred category line case",
      d.parse_pred("The user asks about delivery.\nCategory: Card Arrival",
                   LABELS) == "card_arrival")
check("parse_pred first line beats category line",
      d.parse_pred("card_arrival\nCategory: lost_or_stolen_card",
                   LABELS) == "card_arrival")
check("parse_pred category line wrong label falls through",
      d.parse_pred("Reasoning text here.\nCategory: not_a_label",
                   LABELS) is None)
check("parse_pred full-text fallback",
      d.parse_pred("Some rambling about a lost_or_stolen_card situation.",
                   LABELS) == "lost_or_stolen_card")

# --- prompts
p = d.make_prompt("Where is my card?", LABELS)
check("make_prompt has all labels", all(l in p for l in LABELS))
check("make_prompt has message", "Where is my card?" in p)
check("make_prompt ends Category:", p.rstrip().endswith("Category:"))
tp = d.make_trace_prompt("Where is my card?", LABELS)
check("trace prompt has exemplars",
      "lost_or_stolen_card" in tp and "Reasoning:" in tp)
check("trace prompt has target message", "Where is my card?" in tp)
check("trace prompt ends Reasoning:", tp.rstrip().endswith("Reasoning:"))

# --- extract_trace
good = ("The user asks when the card will arrive after ordering it.\n"
        "Category: card_arrival")
check("extract_trace valid",
      d.extract_trace(good, LABELS, "card_arrival") is not None
      and "arrive" in d.extract_trace(good, LABELS, "card_arrival"))
check("extract_trace wrong label",
      d.extract_trace(good, LABELS, "lost_or_stolen_card") is None)
check("extract_trace no category",
      d.extract_trace("The user asks about delivery.", LABELS, "card_arrival") is None)
check("extract_trace short reasoning",
      d.extract_trace("Ok.\nCategory: card_arrival", LABELS, "card_arrival") is None)
check("extract_trace case-insensitive label",
      d.extract_trace("The user asks about delivery time.\nCategory: Card Arrival",
                      LABELS, "card_arrival") is not None)
check("extract_trace rambles past category still ok",
      d.extract_trace("The user asks about delivery.\nCategory: card_arrival\nExtra junk here.",
                      LABELS, "card_arrival") is not None)

# --- compute_ece / macro_f1
check("ece perfect calibration",
      abs(d.compute_ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) - 0.0) < 1e-9)
# confidences [0.9, 0.9] both correct in bin 4 (0.8-1.0), [0.1, 0.1] both
# wrong in bin 0 (0.0-0.2): |1-0.9|*0.5 + |0-0.1|*0.5 = 0.1
check("ece hand-computed",
      abs(d.compute_ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]) - 0.1) < 1e-9)
check("ece empty", d.compute_ece([], []) == 0.0)
check("macro_f1 perfect",
      abs(d.macro_f1(["a", "b", "a"], ["a", "b", "a"], ["a", "b"]) - 1.0) < 1e-9)
# preds a,b vs golds a,a: class a tp=1 fn=1 -> f1 2/3; class b fp=1 -> f1 0;
# macro over classes present in per = (2/3 + 0)/2 = 1/3
check("macro_f1 hand-computed",
      abs(d.macro_f1(["a", "b"], ["a", "a"], ["a", "b"]) - 1 / 3) < 1e-9)

def _short_pool_raises(traced):
    try:
        d.build_arms(traced[:3])
    except RuntimeError as e:
        return "short" in str(e)
    return False


# --- build_arms (shrink the driver's MIN_ARMS floor for the toy pool)
d.MIN_ARMS = 5
traced = [
    {"prompt": f"PROMPT {i}", "gold": "card_arrival",
     "reasoning": f"The user asks about delivery timing {i}."}
    for i in range(5)
]
arm_r, arm_t, n = d.build_arms(traced)
check("build_arms n", n == 5)
check("build_arms same prompts",
      all(a["prompt"] == b["prompt"] for a, b in zip(arm_r, arm_t)))
check("build_arms R label-only",
      all(a["completion"].strip() == "card_arrival" for a in arm_r))
check("build_arms T has trace+label",
      all("Category: card_arrival" in b["completion"]
          and "delivery timing" in b["completion"] for b in arm_t))
check("build_arms short pool raises", _short_pool_raises(traced))

# --- strided_indices
check("strided full coverage small",
      d.strided_indices(50, 100) == list(range(50)))
idxs = d.strided_indices(1531, 100)
check("strided count", len(idxs) == 100, f"got {len(idxs)}")
check("strided monotonic+spread",
      idxs[0] == 0 and idxs[-1] > 1400 and all(
          b > a for a, b in zip(idxs, idxs[1:])))
check("strided step uniform",
      len({b - a for a, b in zip(idxs, idxs[1:])}) == 1)


# --- load_traced_from_jsonl (v3 trace reuse)
import json as _json
import tempfile as _tempfile
_tmp = _tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
_tmp.write(_json.dumps({"text": "t1", "gold": "card_arrival",
                        "reasoning": "Asks about delivery.",
                        "prompt": "PROMPT t1"}) + "\n")
_tmp.write(_json.dumps({"text": "t2", "gold": "lost_or_stolen_card",
                        "reasoning": "Reports a lost card.",
                        "prompt": "PROMPT t2"}) + "\n")
_tmp.close()
_loaded = d.load_traced_from_jsonl(_tmp.name)
check("load_traced round-trip",
      len(_loaded) == 2 and _loaded[0]["gold"] == "card_arrival"
      and _loaded[1]["reasoning"] == "Reports a lost card."
      and _loaded[0]["prompt"] == "PROMPT t1")
os.unlink(_tmp.name)
d.MIN_ARMS = 2  # toy pool for the build_arms check below
_arms = d.build_arms(_loaded)
check("load_traced feeds build_arms",
      len(_arms[0]) == 2 and len(_arms[1]) == 2 and _arms[2] == 2)

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
print("ALL CPU CHECKS PASSED")
