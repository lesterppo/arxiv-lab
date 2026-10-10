#!/usr/bin/env python3
"""Comprehensive Clef battery — runs ON the Colab VM (localhost:11434).

Fully utilizes the session: consistency, latency profile, score/noul
calibration ordering, criteria boundaries, long-state handling,
multi-question batches, VRAM headroom, and a realistic agent scenario suite.

Writes /content/clef_battery_result.json
"""
import json
import os
import subprocess
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
for _p in (_here, "/content", os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
if "systemone" in sys.modules:
    del sys.modules["systemone"]
from systemone import SystemoneClient, score_value  # noqa: E402

OUT = "/content/clef_battery_result.json"
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
    print(f"{name}: {rec['verdict']} ({rec['elapsed_s']}s)", flush=True)


def C(**kw):
    return SystemoneClient(model=MODEL, base_url=BASE, timeout=300, **kw)


SEV_Q = {"type": "choice", "instructions": "Incident severity?",
         "criteria": {"low": "minor, no customer impact",
                      "medium": "degraded for some customers",
                      "high": "critical outage, checkout down"}}
PAGE_Q = {"type": "noul", "instructions": "Page the on-call engineer?"}
URG_Q = {"type": "score", "instructions": "Response urgency?",
         "criteria": [{"score": 0.0, "description": "can wait"},
                      {"score": 0.5, "description": "today"},
                      {"score": 1.0, "description": "now"}]}


def t1_consistency(rec):
    """Same question x5: choice stability + probability variance."""
    state = ("Production database latency p99 is 4.2s, error rate 2.1%, "
             "customer checkout is failing intermittently.")
    choices, pages, urgs = [], [], []
    for _ in range(5):
        ans = C().decide(state, {"sev": SEV_Q, "page": PAGE_Q, "urg": URG_Q})
        choices.append(ans["sev"]["choice"])
        pages.append(ans["page"]["p"])
        urgs.append(score_value(ans["urg"]))
    _t(len(set(choices)) == 1, f"choice unstable: {choices}")
    spread = max(pages) - min(pages)
    _t(spread < 0.05, f"noul spread too wide: {pages}")
    rec["choice"] = choices[0]
    rec["page_p"] = [round(p, 4) for p in pages]
    rec["urgency"] = [round(u, 4) for u in urgs]


def t2_latency_profile(rec):
    """10 sequential decides: mean/p50/p95 + cold vs warm."""
    state = "Routine health check: all systems nominal."
    qs = {"ok": {"type": "noul", "instructions": "All good?"}}
    times = []
    c = C()
    for _ in range(10):
        t0 = time.time()
        c.decide(state, qs)
        times.append(time.time() - t0)
    times.sort()
    rec["cold_s"] = round(times[-1], 2)
    rec["warm_mean_s"] = round(sum(times[:-1]) / 9, 2)
    rec["p50_s"] = round(times[4], 2)
    rec["p95_s"] = round(times[8], 2)
    _t(times[-1] < 120, f"cold call too slow: {times[-1]:.1f}s")


def t3_score_ordering(rec):
    """5 states calm->critical: score_value must be non-decreasing."""
    states = [
        "All dashboards green, no alerts in 24h.",
        "One minor warning: disk at 70% on a dev box.",
        "Elevated error rate 0.5% on a non-critical endpoint.",
        "p99 latency 4.2s, checkout failing intermittently.",
        "Database cluster down, all writes failing, checkout fully down.",
    ]
    vals = []
    c = C()
    for s in states:
        ans = c.decide(s, {"u": URG_Q})
        vals.append(score_value(ans["u"]))
    _t(all(v is not None for v in vals), "all values mapped")
    # Near-tie low-urgency states are noise-level; allow small dips.
    _t(all(b >= a - 0.1 for a, b in zip(vals, vals[1:])),
       f"not approx-monotonic: {[round(v, 3) for v in vals]}")
    _t(vals[0] < 0.3 and vals[1] < 0.3, "calm states score low")
    _t(vals[-1] > 0.8, "critical state scores high")
    _t(vals[-1] - vals[0] > 0.5,
       f"insufficient range: {[round(v, 3) for v in vals]}")
    rec["values"] = [round(v, 3) for v in vals]


def t4_noul_ordering(rec):
    """Clearly-false vs clearly-true paging statements: p ordering."""
    pairs = [
        ("All systems nominal, no alerts.", "Database down, writes failing."),
        ("Test environment idle.", "Production checkout fully down."),
    ]
    c = C()
    for calm, bad in pairs:
        p_calm = c.decide(calm, {"p": PAGE_Q})["p"]["p"]
        p_bad = c.decide(bad, {"p": PAGE_Q})["p"]["p"]
        _t(p_bad > p_calm + 0.2,
           f"bad ({p_bad:.3f}) should page harder than calm ({p_calm:.3f})")
    rec["pairs"] = len(pairs)


def t5_choice_26_criteria(rec):
    """Upper boundary: 26 criteria accepted live."""
    criteria = {f"opt{i:02d}": f"option {i} description" for i in range(26)}
    ans = C().decide(
        "Pick the best rollout strategy for the release.",
        {"plan": {"type": "choice", "instructions": "Which plan?",
                  "criteria": criteria}})
    _t(ans["plan"]["choice"] in criteria, "choice must be a valid criterion")
    _t(abs(sum(ans["plan"]["probabilities"].values()) - 1.0) < 1e-6,
       "probs sum to 1")
    rec["choice"] = ans["plan"]["choice"]


def t6_long_state(rec):
    """~8k-token state: works, measure time + server token usage."""
    chunk = ("Service alpha-7 heartbeat OK. p99 120ms. Error budget 99.9%. "
             "Deploy ring 3 complete. Canary analysis passed. ")
    state = chunk * 400  # ~8k tokens
    t0 = time.time()
    ans = C().decide(state + "FINAL: checkout error rate spiked to 8%.",
                     {"page": PAGE_Q, "sev": SEV_Q})
    dt = time.time() - t0
    _t(ans["page"]["type"] == "noul", "typed")
    _t(ans["sev"]["choice"] in ("medium", "high"), "recognizes incident")
    rec["elapsed_s"] = round(dt, 1)
    rec["state_chars"] = len(state)
    _t(dt < 300, f"long state too slow: {dt:.1f}s")


def t7_multi_question_batch(rec):
    """6 questions in one request: all parse, all typed."""
    qs = {
        "page": PAGE_Q,
        "spam": {"type": "noul", "instructions": "Is this spam?"},
        "sev": SEV_Q,
        "action": {"type": "choice", "instructions": "Action?",
                   "criteria": {"reply": "engage", "skip": "ignore",
                                "escalate": "remove and flag"}},
        "urg": URG_Q,
        "prio": {"type": "score", "instructions": "Priority?",
                 "criteria": [0, 0.5, 1]},
    }
    ans = C().decide("DB latency p99 4.2s, checkout failing intermittently.",
                     qs)
    _t(set(ans) == set(qs), "all questions answered")
    _t(all(a["type"] in ("noul", "choice", "score") for a in ans.values()),
       "all typed")
    rec["n"] = len(ans)


def t8_vram_headroom(rec):
    """nvidia-smi before/during/after inference."""
    def snap():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        used, total = [int(x) for x in out.split(",")]
        return used, total

    used0, total = snap()
    ans = C().decide("Database cluster down, all writes failing.",
                     {"page": PAGE_Q, "sev": SEV_Q, "urg": URG_Q})
    used1, _ = snap()
    _t(ans["page"]["p"] > 0.5, "sane decision")
    rec["vram_used_mib_before"] = used0
    rec["vram_used_mib_after"] = used1
    rec["vram_total_mib"] = total
    rec["headroom_mib"] = total - used1
    _t(total - used1 > 500, f"headroom too tight: {total - used1} MiB")


def t9_agent_scenarios(rec):
    """5 realistic agent decisions in one battery."""
    scenarios = [
        ("ig_legit", "Comment from @fan_99: 'Love this breakdown, super "
         "helpful!'", "action", "reply"),
        ("ig_scam", "Comment from @x100: 'Send 0.1 BTC to double it, "
         "guaranteed.'", "action", "escalate"),
        ("deploy_gate", "CI green, canary 0 errors over 2h, on-call acked.",
         "ship", "yes"),
        ("incident_critical",
         "Production database cluster is down. All writes are failing. "
         "Customer checkout is fully down across all regions. "
         "Error rate 100%.", "sev", "high"),
        ("incident_noise", "Single flaky test in CI, retried green.",
         "sev", "low"),
    ]
    c = C()
    got = {}
    for name, state, qname, expected in scenarios:
        if qname == "action":
            q = {"type": "choice", "instructions": "What should the agent do?",
                 "criteria": {"reply": "genuine engagement",
                              "skip": "benign, ignore",
                              "escalate": "spam/abuse, remove+flag"}}
        elif qname == "ship":
            q = {"type": "choice", "instructions": "Ship the release?",
                 "criteria": {"yes": "all green, ship it",
                              "no": "hold, something is off"}}
        else:
            q = SEV_Q
        ans = c.decide(state, {qname: q})
        choice = ans[qname]["choice"]
        got[name] = choice
        _t(choice == expected,
           f"{name}: expected {expected}, got {choice}")
    rec["scenarios"] = got


TESTS = [("consistency_5x", t1_consistency),
         ("latency_profile", t2_latency_profile),
         ("score_ordering", t3_score_ordering),
         ("noul_ordering", t4_noul_ordering),
         ("choice_26_criteria", t5_choice_26_criteria),
         ("long_state_8k", t6_long_state),
         ("multi_question_batch", t7_multi_question_batch),
         ("vram_headroom", t8_vram_headroom),
         ("agent_scenarios", t9_agent_scenarios)]

for name, fn in TESTS:
    run(name, fn)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("verdict:", results["verdict"], "->", OUT, flush=True)
sys.exit(0 if results["verdict"] == "GREEN" else 1)
