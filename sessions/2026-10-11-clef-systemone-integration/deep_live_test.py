#!/usr/bin/env python3
"""Deep live test of the systemone backend against clef-flash (runs on VM).

Beyond the 2026-10-07 integration test:
  1. incident triage (noul + choice + score w/ object criteria)
  2. comment triage (content-bot style)
  3. score criteria shapes: array-of-numbers, array-of-strings,
     array-of-objects -> score_value mapping behaviour for each
  4. choice with 2 criteria (lower boundary)
  5. sequential decide calls (no cross-call state leakage)
  6. client-side validation still rejects bad shapes pre-network

Writes /content/clef_deep_result.json
"""
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
for _p in (_here, "/content", os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
if "systemone" in sys.modules:
    del sys.modules["systemone"]
from systemone import SystemoneClient, score_value  # noqa: E402

OUT = "/content/clef_deep_result.json"
BASE = "http://localhost:11434"
MODEL = "clef-flash:latest"

results = {"model": MODEL, "tests": {}, "verdict": "GREEN"}


def _t(cond, msg="assertion failed"):
    assert cond, msg


def run(name, fn):
    t0 = time.time()
    rec = {}
    try:
        fn(rec)
        rec["verdict"] = "PASS"
    except Exception as e:  # noqa: BLE001
        rec["verdict"] = "FAIL"
        rec["error"] = f"{type(e).__name__}: {e}"
        results["verdict"] = "RED"
    rec["elapsed_s"] = round(time.time() - t0, 1)
    results["tests"][name] = rec
    print(f"{name}: {rec['verdict']} ({rec['elapsed_s']}s)")


def C(**kw):
    return SystemoneClient(model=MODEL, base_url=BASE, timeout=300, **kw)


def t1_incident(rec):
    ans = C().decide(
        "Production database latency p99 is 4.2s, error rate 2.1%, "
        "customer checkout is failing intermittently.",
        {"page_oncall": {"type": "noul",
                         "instructions": "Page the on-call engineer?"},
         "severity": {"type": "choice",
                      "instructions": "Incident severity?",
                      "criteria": {"low": "minor, no customer impact",
                                   "medium": "degraded for some customers",
                                   "high": "critical outage, checkout down"}},
         "urgency": {"type": "score",
                     "instructions": "Response urgency?",
                     "criteria": [{"score": 0.0, "description": "can wait"},
                                  {"score": 0.5, "description": "today"},
                                  {"score": 1.0, "description": "now"}]}})
    _t(ans["page_oncall"]["p"] > 0.5, "page_oncall should be likely")
    # "intermittent failures" is genuinely ambiguous between medium/high;
    # the integration property is: recognized as a real incident (not low).
    sev = ans["severity"]
    _t(sev["choice"] in ("medium", "high"),
       f"severity should be medium/high, got {sev['choice']}")
    _t(sev["probabilities"]["low"] < 0.2, "low should carry little mass")
    v = score_value(ans["urgency"])
    _t(v is not None and 0.5 < v <= 1.0, f"urgency value {v}")
    rec["summary"] = {k: (v["p"] if v["type"] == "noul"
                          else v["choice"] if v["type"] == "choice"
                          else round(score_value(v), 3))
                      for k, v in ans.items()}


def t2_comment_triage(rec):
    ans = C().decide(
        "Comment on ppoppo205's post from @crypto_guru_99: "
        "'DM me for 10x guaranteed returns, send 0.1 BTC to bc1qxy2...'",
        {"is_spam": {"type": "noul",
                     "instructions": "Is this comment spam or a scam?"},
         "action": {"type": "choice",
                    "instructions": "What should the agent do?",
                    "criteria": {"reply": "genuine engagement, worth a reply",
                                 "skip": "benign but low-value",
                                 "escalate": "spam/scam/abuse, remove + flag"}},
         "priority": {"type": "score",
                      "instructions": "Handling urgency?",
                      "criteria": [0, 0.5, 1]}})
    _t(ans["is_spam"]["p"] > 0.5, "should look like spam")
    _t(ans["action"]["choice"] == "escalate", "should escalate")
    _t(ans["action"]["probabilities"]["escalate"] > 0.9, "escalate dominant")
    v = score_value(ans["priority"])
    _t(v is not None and v > 0.5, f"priority value {v}")
    rec["summary"] = {"is_spam": round(ans["is_spam"]["p"], 3),
                      "action": ans["action"]["choice"],
                      "priority": round(v, 3)}


def t3_score_shapes(rec):
    out = {}
    shapes = {
        "numbers": [0, 0.25, 0.5, 0.75, 1.0],
        "strings": ["none", "low", "high"],
        "objects": [{"score": 0, "description": "calm"},
                    {"score": 10, "description": "panic"}],
    }
    for label, criteria in shapes.items():
        ans = C().decide(
            "A user reports the app is slow on login.",
            {"s": {"type": "score", "instructions": "How alarming?",
                   "criteria": criteria}})
        a = ans["s"]
        _t(a["type"] == "score", f"{label}: typed")
        _t(abs(sum(a["probabilities"].values()) - 1.0) < 1e-6,
           f"{label}: probs sum to 1")
        v = score_value(a)
        out[label] = {"score_idx": round(a["score"], 3), "value": v}
        if label == "numbers":
            _t(v is not None and 0.0 <= v <= 1.0, f"{label}: value in [0,1]")
        elif label == "strings":
            _t(v is None, f"{label}: non-numeric -> None")
        elif label == "objects":
            _t(v is not None and 0.0 <= v <= 10.0, f"{label}: value in [0,10]")
    rec["shapes"] = out


def t4_choice_two_criteria(rec):
    ans = C().decide(
        "The build is green and all checks pass.",
        {"ship": {"type": "choice", "instructions": "Ship it?",
                  "criteria": {"yes": "all green, ship",
                               "no": "something is off, hold"}}})
    _t(ans["ship"]["choice"] in ("yes", "no"), "valid choice")
    rec["choice"] = ans["ship"]["choice"]


def t5_sequential_no_leak(rec):
    c = C()
    a1 = c.decide("Everything is fine, no alerts.",
                  {"page": {"type": "noul",
                            "instructions": "Page the on-call?"}})
    a2 = c.decide("Database is down, all writes failing.",
                  {"page": {"type": "noul",
                            "instructions": "Page the on-call?"}})
    _t(a2["page"]["p"] > a1["page"]["p"],
       f"down ({a2['page']['p']:.3f}) should page harder than fine "
       f"({a1['page']['p']:.3f})")
    rec["p_fine"] = round(a1["page"]["p"], 3)
    rec["p_down"] = round(a2["page"]["p"], 3)


def t6_client_side_validation(rec):
    c = C()
    cases = [
        ({"q": {"type": "score", "instructions": "x",
                "criteria": {"a": "b"}}}, ValueError),   # score needs array
        ({"q": {"type": "choice", "instructions": "x",
                "criteria": ["a", "b"]}}, ValueError),   # choice needs object
        ({"q": {"type": "noul", "instructions": "x",
                "criteria": [0, 1]}}, ValueError),       # noul: no criteria
    ]
    for questions, exc in cases:
        try:
            c.decide("state", questions)
        except exc:
            continue
        raise AssertionError(f"{questions} did not raise {exc.__name__}")
    rec["cases"] = len(cases)


for name, fn in [("incident_triage", t1_incident),
                 ("comment_triage", t2_comment_triage),
                 ("score_shapes", t3_score_shapes),
                 ("choice_two_criteria", t4_choice_two_criteria),
                 ("sequential_no_leak", t5_sequential_no_leak),
                 ("client_side_validation", t6_client_side_validation)]:
    run(name, fn)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("verdict:", results["verdict"], "->", OUT)
sys.exit(0 if results["verdict"] == "GREEN" else 1)
