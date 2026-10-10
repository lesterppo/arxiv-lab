"""CPU test for arXiv 2610.10590 — Agent-Controlled Forgetting for
Tool-Using Agents: Reversible Context Curation in Practice.

Reproduces the paper's mechanism and experimental design in a scripted,
seeded simulation (no LLM): a noisy tool-use trajectory (big payloads,
tiny useful content) and a dense contrasting workload, comparing
retained history vs reversible forgetting vs an irreversible-truncation
ablation.

Claims under test (paper Sec. 3-4):
 1. Protocol integrity: batch archival is all-or-nothing; unknown /
    duplicate / already-archived ids fail without partial mutation;
    user/assistant messages are not selectable; recovery returns the
    exact original and mints a fresh result id; the stub stays
    addressable. (paper Sec. 3.2)
 2. Token savings on noisy trajectories: the forget arm uses far fewer
    cumulative prompt tokens than retained history. (paper: 231,951 vs
    912,492 provider prompt tokens; ~50% fewer cumulative input tokens)
 3. Reversibility preserves task success: recall probes for facts the
    (lossy) note dropped are answered after recovery; the irreversible
    ablation at the same token budget fails them. (paper: both arms
    passed the primary behavioral oracle)
 4. Workload dependence: on dense payloads where nearly every token is
    signal, the forget arm archives nothing and saves ~nothing.
    (paper: the contrasting application-development pair produced no
    context or cost saving)
 5. The method arm makes more requests (context-management + recovery
    calls) and carries a small latency proxy. (paper: more requests,
    17% longer)

Deterministic (seeded). stdlib only.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from arxiv_lab.context.reversible_forgetting import (
    ForgettingHarness,
    IrreversibleHarness,
    ForgettingError,
    TrajectoryReplay,
    estimate_tokens,
)


def test_batch_atomicity():
    h = ForgettingHarness()
    h.add("user", "do the thing")
    r1 = h.add_tool_result("logs", "body-one " * 50)
    r2 = h.add_tool_result("logs", "body-two " * 50)
    # valid batch commits
    m = h.context_apply([(r1, "note one"), (r2, "note two")])
    assert set(m) == {r1, r2} and len(h.archive) == 2
    assert h._by_result[r1].archived and h._by_result[r2].archived
    # duplicate in batch -> whole batch rejected
    r3 = h.add_tool_result("logs", "body-three " * 50)
    before = h.prompt_tokens()
    try:
        h.context_apply([(r3, "ok"), (r3, "dup")])
        raise AssertionError("expected ForgettingError")
    except ForgettingError:
        pass
    assert not h._by_result[r3].archived
    assert h.prompt_tokens() == before  # nothing committed
    # unknown id -> rejected, nothing committed
    try:
        h.context_apply([(r3, "ok"), ("r999", "bogus")])
        raise AssertionError("expected ForgettingError")
    except ForgettingError:
        pass
    assert not h._by_result[r3].archived
    # already-archived id -> rejected
    try:
        h.context_apply([(r1, "again")])
        raise AssertionError("expected ForgettingError")
    except ForgettingError:
        pass
    print("test_batch_atomicity: all-or-nothing holds; no partial mutation")


def test_recovery_returns_exact_original():
    h = ForgettingHarness()
    orig = "payload with details " * 100 + "SECRET-LINE-42"
    r1 = h.add_tool_result("browser", orig, {"url": "http://x"})
    h.context_apply([(r1, "page about details")])
    assert "SECRET-LINE-42" not in h._by_result[r1].body  # stub, not original
    new_rid = h.context_recover("a1")
    assert new_rid != r1                       # fresh result id
    assert h._by_result[new_rid].body == orig  # exact original back
    assert h._by_result[r1].archived           # original stub addressable
    assert h._by_result[r1].archive_ref == "a1"
    try:
        h.context_recover("a999")
        raise AssertionError("expected ForgettingError")
    except ForgettingError:
        pass
    print("test_recovery_returns_exact_original: exact payload restored, "
          "fresh id %s" % new_rid)


def test_noisy_workload_savings():
    retained = TrajectoryReplay(seed=7, policy=None, workload="noisy").run()
    forget = TrajectoryReplay(seed=7, policy="forget", workload="noisy").run()
    saved = 1.0 - forget["cumulative_prompt_tokens"] / retained["cumulative_prompt_tokens"]
    print("noisy: retained=%d forget=%d saved=%.1f%% archived=%d "
          "recoveries=%d requests %d->%d" % (
              retained["cumulative_prompt_tokens"],
              forget["cumulative_prompt_tokens"], 100 * saved,
              forget["archived"], forget["recoveries"],
              retained["requests"], forget["requests"]))
    assert saved >= 0.50, "expected >=50%% cumulative token savings, got %.1f%%" % (100 * saved)
    assert forget["all_probes_ok"], "reversible arm must pass all recall probes"
    assert forget["recoveries"] > 0, "lossy notes must trigger recovery"
    assert forget["requests"] > retained["requests"], "method arm makes more requests"


def test_reversibility_beats_irreversible():
    rev = TrajectoryReplay(seed=7, policy="forget", workload="noisy").run()
    irr = TrajectoryReplay(seed=7, policy="forget", workload="noisy",
                           harness_cls=IrreversibleHarness).run()
    print("reversible: probes %d/%d ok, recoveries=%d" %
          (rev["probes_ok"], rev["probes"], rev["recoveries"]))
    print("irreversible: probes %d/%d ok" % (irr["probes_ok"], irr["probes"]))
    assert rev["all_probes_ok"]
    assert irr["probes_ok"] < irr["probes"], \
        "irreversible truncation must lose at least one probe the note dropped"
    assert irr["probes_ok"] < rev["probes_ok"]


def test_workload_dependence():
    retained = TrajectoryReplay(seed=11, policy=None, workload="dense").run()
    forget = TrajectoryReplay(seed=11, policy="forget", workload="dense").run()
    saved = 1.0 - forget["cumulative_prompt_tokens"] / retained["cumulative_prompt_tokens"]
    print("dense: retained=%d forget=%d saved=%.1f%% archived=%d" % (
        retained["cumulative_prompt_tokens"],
        forget["cumulative_prompt_tokens"], 100 * saved, forget["archived"]))
    assert forget["archived"] == 0, "dense payloads should never be archived"
    assert abs(saved) < 0.05, "dense workload: expected ~no savings, got %.1f%%" % (100 * saved)


def test_stub_cost_scales_with_note():
    h = ForgettingHarness()
    r1 = h.add_tool_result("logs", "x" * 4000)
    full = h.prompt_tokens()
    h.context_apply([(r1, "short note")])
    stubbed = h.prompt_tokens()
    ratio = stubbed / full
    print("stub/active token ratio: %.3f" % ratio)
    assert ratio < 0.05, "stub must cost a small fraction of the original"
    assert estimate_tokens("x" * 4000) == 1000


if __name__ == "__main__":
    test_batch_atomicity()
    test_recovery_returns_exact_original()
    test_stub_cost_scales_with_note()
    test_noisy_workload_savings()
    test_reversibility_beats_irreversible()
    test_workload_dependence()
    print("ALL FORGET-TESTS PASSED")
