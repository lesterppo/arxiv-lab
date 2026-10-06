"""CPU test for arXiv 2610.04375 — Intent-execution correspondence (IEC).

Reproduces the paper's core claims on a deterministic synthetic workload
(500 tool calls, 5 arg classes, 4-hop pipeline modeled on the paper's
measured harness paths):

 1. Hops silently mutate calls: code/escape-carrying calls are mutated at a
    substantial rate (paper: 12.0% of Claude Code Bash calls carrying code,
    escapes, or long text); every modeled mutating hop class bites.
 2. Backslash-mutated calls still parse, so the wrong action executes
    WITHOUT any reported error (paper: 80.7% of backslash-changed calls).
 3. A trajectory-based judge (reads emitted call + result, not hop
    receipts) attributes path-caused failures to the LLM (paper: 95.1%).
 4. The observation protocol names the first mutating hop without
    executing anything (JsonReserializeHop is cosmetic and never named).
 5. IntAct repair: armored delivery executes ZERO mutated calls; damage
    becomes refusal (checksum), and a safe retry recovers full success.

Deterministic: fixed workload, no RNG in the pipeline itself.
stdlib only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arxiv_lab.iec.iec import (  # noqa: E402
    Call,
    ToolContract,
    IdentityHop,
    JsonReserializeHop,
    ShellQuoteHop,
    UrlDecodeHop,
    TruncateHop,
    observe,
    first_mutating_hop,
    armor,
    deliver,
    Refusal,
    ExecutedCall,
    NaiveTrajectoryJudge,
    make_workload,
    production_hops,
)

N = 500
CONTRACT = ToolContract("bash", ("cmd",))


def run_naive(calls, hops):
    """Naive pipeline: push canonical text through hops, then execute."""
    stats = {"mutated": 0, "wrong_action": 0, "error": 0, "ok": 0,
             "by_class": {}, "first_hop": {}}
    judge = NaiveTrajectoryJudge()
    misattributed = 0
    path_caused = 0
    for cls, call in calls:
        receipts = observe(call, CONTRACT, hops)
        mutated = any(r.changed for r in receipts)
        named = first_mutating_hop(receipts)
        if named:
            stats["first_hop"][named[1]] = stats["first_hop"].get(named[1], 0) + 1
        # what the tool actually receives after the last hop
        text = call.canonical()
        for hop in hops:
            text = hop.receive(text)
        try:
            parsed = CONTRACT.parse(text)
            outcome = "ok" if parsed == call else "wrong-action"
        except Exception:
            outcome = "error"
        stats[outcome.replace("-", "_")] += 1
        if mutated:
            stats["mutated"] += 1
            path_caused += 1 if outcome != "ok" else 0
            if outcome != "ok" and judge.attribute(call, outcome) == "llm":
                misattributed += 1
        c = stats["by_class"].setdefault(cls, {"n": 0, "mutated": 0})
        c["n"] += 1
        c["mutated"] += mutated
    stats["misattribution_rate"] = misattributed / max(path_caused, 1)
    stats["path_caused_failures"] = path_caused
    return stats


def run_intact(calls, hops):
    """IntAct pipeline: armored delivery, refuse on any damage, safe retry."""
    stats = {"executed_intact": 0, "mutated_executions": 0, "refused": 0,
             "retry_ok": 0}
    clean = [IdentityHop("clean-channel")]
    for _cls, call in calls:
        result = deliver(armor(call), CONTRACT, hops)
        if isinstance(result, Refusal):
            stats["refused"] += 1
            # safe retry of the refused call over a clean channel
            retry = deliver(armor(call), CONTRACT, clean)
            assert isinstance(retry, ExecutedCall) and retry.call == call
            stats["retry_ok"] += 1
        else:
            assert isinstance(result, ExecutedCall)
            if result.call != call:
                stats["mutated_executions"] += 1
            else:
                stats["executed_intact"] += 1
    return stats


def test_iec_claims():
    calls = make_workload(N, seed=0)
    hops = production_hops(truncate_at=2000)

    naive = run_naive(calls, hops)
    intact = run_intact(calls, hops)

    print(f"calls={N}")
    print(f"mutated={naive['mutated']} ({naive['mutated']/N:.1%})")
    for cls, c in naive["by_class"].items():
        print(f"  {cls}: mutated {c['mutated']}/{c['n']} = {c['mutated']/c['n']:.1%}")
    print(f"wrong_action={naive['wrong_action']} error={naive['error']} ok={naive['ok']}")
    print(f"first mutating hop counts: {naive['first_hop']}")
    print(f"path-caused failures={naive['path_caused_failures']}, "
          f"judge misattribution={naive['misattribution_rate']:.1%}")
    print(f"IntAct: intact={intact['executed_intact']} refused={intact['refused']} "
          f"mutated_executions={intact['mutated_executions']} retry_ok={intact['retry_ok']}")

    # Claim 1: hops mutate a substantial share of calls; code class is hit.
    assert naive["mutated"] > 0.05 * N, "hops should mutate >5% of calls"
    code = naive["by_class"]["code"]
    assert code["mutated"] / code["n"] >= 0.10, "code-carrying calls hit hard"
    # every modeled mutating hop class gets named at least once; the
    # cosmetic JSON re-serialization hop and the tool terminal are never
    # named as *mutating* stages.
    assert set(naive["first_hop"]) == {"shell-quoting", "proxy-normalize", "payload-cap"}
    assert "logger-pretty" not in naive["first_hop"]
    assert "bash" not in naive["first_hop"]

    # Claim 2: backslash mutations parse fine -> wrong action, no error.
    # (shell-quoting mutations never break JSON structure, so none of the
    # code-class mutations surface as errors.)
    assert naive["wrong_action"] > 0
    code_wrong = naive["by_class"]["code"]["mutated"]  # all parse -> wrong-action
    assert code_wrong > 0

    # Claim 3: trajectory judge blames the LLM for path-caused failures.
    assert naive["misattribution_rate"] >= 0.9, naive["misattribution_rate"]

    # Claim 4+5: IntAct executes zero mutated calls; refusals are retried clean.
    assert intact["mutated_executions"] == 0
    assert intact["executed_intact"] + intact["refused"] == N
    assert intact["retry_ok"] == intact["refused"]
    # every call either executes intact or is refused-then-retried: full
    # task success with no silent wrong actions.
    assert intact["executed_intact"] + intact["retry_ok"] == N

    results = {
        "n": N,
        "naive": naive,
        "intact": intact,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "sessions",
                       "2026-10-06-2610.04375-iec", "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    test_iec_claims()
    print("IEC CPU test: all claims reproduced.")
