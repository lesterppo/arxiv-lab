"""CPU test: signed requirement manifests (arXiv 2611.07483).

Deterministic, exactly scorable simulation of the paper's preregistered
study: free-text vs structured-JSON vs HMAC-signed manifest handoffs
across five threats, with a deterministic predicate verifier vs a noisy
surface-text checker. stdlib only; numpy not needed.

Expected directional reproduction of the paper (magnitudes differ — the
simulator is rule-based, not LLM-based):
  C1  signed unsafe commits under attack << free-text (paper 9.0% vs 69.4%).
  C2  unsigned manifest commits violations under mutation; signed commits
      none (paper 79% vs 0%).
  C3  deterministic verifier detects more violations with fewer false
      alarms than the surface-text checker (paper 78.7% vs 59.0%
      detection, 7% vs 50% false alarms).
  C4  pre-signing injection raises signed-manifest violations (paper
      14.6% -> 37.5% on Qwen3-8B): the signature authenticates the
      corrupted manifest; extraction is the trust boundary.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.multiagent.signed_manifest import (  # noqa: E402
    INJECT_LINE,
    build_handoff,
    check_requirement,
    extract_requirements,
    make_task,
    merge,
    run_experiment,
    sign_manifest,
    unsafe_under_attack,
    verify_manifest,
    BY_ID,
)
import random  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" | " + detail if detail else ""))
    if not cond:
        FAILURES.append(name)


KEY = "orchestrator-hmac-key"

# --- unit: signing / verification -----------------------------------------
m = {"task": "t", "requirements": [{"id": "env", "quote": "deploy only to staging"}]}
signed = sign_manifest(m, KEY)
check("signature verifies on untampered manifest", verify_manifest(signed, KEY))
tampered = dict(signed)
tampered["requirements"] = [{"id": "env", "quote": "deploy only to production"}]
check("tampered manifest fails verification", not verify_manifest(tampered, KEY))
check("wrong key fails verification", not verify_manifest(signed, "other-key"))
check("missing signature fails verification",
      not verify_manifest({"task": "t", "requirements": []}, KEY))
check("signature is deterministic",
      sign_manifest(m, KEY)["signature"] == signed["signature"],
      signed["signature"][:16])

# --- unit: extractor -------------------------------------------------------
rng = random.Random(7)
task = make_task(rng)
reqs = extract_requirements(task["text"])
check("extractor recovers all listed requirements",
      {r["id"] for r in reqs} == {r["id"] for r in task["ground_truth"]})
inj = {"text": task["text"] + "\n" + INJECT_LINE,
       "ground_truth": task["ground_truth"]}
inj_reqs = extract_requirements(inj["text"])
env = [r for r in inj_reqs if r["id"] == "env"]
check("pre-signing injection overrides the env requirement",
      len(env) == 1 and env[0]["allowed"] == {"production"},
      "injected allowed=%s" % (env[0]["allowed"],))

# --- unit: predicates ------------------------------------------------------
dec_ok = {"env": "staging", "email_scope": "internal", "spend_usd": 100,
          "payment_approved": True, "logging": True}
dec_bad = {"env": "production", "email_scope": "external", "spend_usd": 5000,
           "payment_approved": False, "logging": False}
check("predicates accept a conformant decision",
      all(check_requirement(r, dec_ok) for r in BY_ID.values()))
check("predicates reject a violating decision",
      not any(check_requirement(r, dec_bad) for r in BY_ID.values()))

# --- unit: merger refuses on a bad signature ------------------------------
handoff, _ = build_handoff(task, "signed_manifest", KEY, "none")
handoff["manifest"]["requirements"] = [
    {"id": "env", "quote": "deploy only to production", "field": "env",
     "allowed": {"production"}, "safety": True, "tamper_value": "staging"}]
_, refused = merge(handoff, random.Random(1), "none")
check("merger refuses a manifest with a broken signature", refused)

# --- full experiment -------------------------------------------------------
rows, summary = run_experiment(n_per_cell=60, seed=0)
under_attack = unsafe_under_attack(summary)
print("\nunsafe-commit rate under attack (mean of 5 threats):")
for fmt, rate in under_attack.items():
    print("  %-15s %.3f" % (fmt, rate))
print("\nper-cell unsafe-commit rates:")
for fmt in ("free_text", "json_unsigned", "signed_manifest"):
    for threat in ("none", "drop", "alter", "ignore", "mutate", "inject"):
        s = summary[(fmt, threat)]
        print("  %-15s %-6s unsafe=%.3f conformant=%.3f refused=%.3f"
              % (fmt, threat, s["unsafe_commit_rate"],
                 s["conformant_commit_rate"], s["refused_rate"]))
print("\nverifier comparison (over violating / conformant decisions):")
for fmt in ("signed_manifest", "json_unsigned"):
    v = summary[(fmt, "verifier")]
    print("  %-15s detection=%.3f (n=%d) false_alarm=%.3f (n=%d)"
          % (fmt, v["detection_rate"], v["n_violating"],
             v["false_alarm_rate"], v["n_conformant"]))

check("C1: signed unsafe-under-attack << free-text",
      under_attack["signed_manifest"] + 0.15 < under_attack["free_text"],
      "signed=%.3f free_text=%.3f"
      % (under_attack["signed_manifest"], under_attack["free_text"]))
check("C1: signed unsafe-under-attack < 0.25",
      under_attack["signed_manifest"] < 0.25,
      "signed=%.3f" % under_attack["signed_manifest"])
check("C2: signed commits nothing under in-transit mutation",
      summary[("signed_manifest", "mutate")]["unsafe_commit_rate"] == 0.0
      and summary[("signed_manifest", "mutate")]["refused_rate"] == 1.0)
check("C2: unsigned manifest commits violations under mutation",
      summary[("json_unsigned", "mutate")]["unsafe_commit_rate"] > 0.3,
      "unsigned mutate unsafe=%.3f"
      % summary[("json_unsigned", "mutate")]["unsafe_commit_rate"])
v_signed = summary[("signed_manifest", "verifier")]
v_json = summary[("json_unsigned", "verifier")]
check("C3: deterministic verifier detects more than surface checker",
      v_signed["detection_rate"] > v_json["detection_rate"],
      "det %.3f vs %.3f"
      % (v_signed["detection_rate"], v_json["detection_rate"]))
check("C3: deterministic verifier false-alarms less",
      v_signed["false_alarm_rate"] < v_json["false_alarm_rate"],
      "fa %.3f vs %.3f"
      % (v_signed["false_alarm_rate"], v_json["false_alarm_rate"]))
check("C4: pre-signing injection raises signed violations",
      summary[("signed_manifest", "inject")]["unsafe_commit_rate"]
      > summary[("signed_manifest", "none")]["unsafe_commit_rate"],
      "inject=%.3f none=%.3f"
      % (summary[("signed_manifest", "inject")]["unsafe_commit_rate"],
         summary[("signed_manifest", "none")]["unsafe_commit_rate"]))
check("no net loss of conformant commits (signed vs json, no attack)",
      summary[("signed_manifest", "none")]["conformant_commit_rate"]
      >= summary[("json_unsigned", "none")]["conformant_commit_rate"] - 0.05,
      "signed=%.3f json=%.3f"
      % (summary[("signed_manifest", "none")]["conformant_commit_rate"],
         summary[("json_unsigned", "none")]["conformant_commit_rate"]))

# --- determinism -----------------------------------------------------------
_, summary2 = run_experiment(n_per_cell=60, seed=0)
check("experiment is deterministic for a fixed seed",
      summary == summary2)

print("\nfailures: %d" % len(FAILURES))
sys.exit(1 if FAILURES else 0)
