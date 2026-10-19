"""CPU tests for arxiv_lab.harness.schema_shield (arXiv:2610.23342).

Reproduces the paper's core claim on CPU: a lightweight, training-free
repair layer (schema validation + candidate-set name retry + one-shot
constrained arg repair) recovers tool-calling robustness under interface
drift without retraining the caller.

Method: a synthetic 8-tool registry; 24 scripted tasks with golden
pre-drift call sequences (the "caller" is frozen on the old interface, as a
deployed model would be). Six drift scenarios replay the stale calls against
the drifted registry, with and without SchemaShield. Measured: baseline
accuracy, shield accuracy, drift-failure recovery rate, repair harms
(correct calls broken by the repair layer — must be zero), and a no-drift
control proving the layer is a pass-through when nothing drifted.

Deterministic (seeded). Pure stdlib.

Run:  python3 tests/test_harness_schemashield_cpu.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from arxiv_lab.harness import (
    DriftSimulator,
    SchemaShield,
    ToolCall,
    ToolDef,
)


# ── Old (pre-drift) registry ────────────────────────────────────────────────

def _t(name, desc, **params):
    return ToolDef(name, desc, params)


OLD_REGISTRY = {
    "search_web": _t(
        "search_web", "Search the web for current information.",
        query={"type": "string", "required": True},
        limit={"type": "integer", "required": False, "default": 5}),
    "read_file": _t(
        "read_file", "Read a file from the virtual filesystem.",
        path={"type": "string", "required": True}),
    "write_file": _t(
        "write_file", "Write content to a file, creating parents.",
        path={"type": "string", "required": True},
        content={"type": "string", "required": True}),
    "calc": _t(
        "calc", "Evaluate an arithmetic expression.",
        expression={"type": "string", "required": True}),
    "send_email": _t(
        "send_email", "Send an email to a recipient.",
        to={"type": "string", "required": True},
        subject={"type": "string", "required": True},
        body={"type": "string", "required": False, "default": ""}),
    "db_query": _t(
        "db_query", "Run a read-only SQL query against the warehouse.",
        sql={"type": "string", "required": True},
        timeout_s={"type": "integer", "required": False, "default": 30}),
    "translate": _t(
        "translate", "Translate text to a target language.",
        text={"type": "string", "required": True},
        target_lang={"type": "string", "required": True}),
    "summarize": _t(
        "summarize", "Summarize a document in N bullet points.",
        document={"type": "string", "required": True},
        bullets={"type": "integer", "required": False, "default": 3}),
}

DISTRACTORS = [
    _t("web_search_pro", "Pro web search with ranking control.",
       query={"type": "string", "required": True},
       rank_by={"type": "string", "required": False, "default": "relevance"}),
    _t("search_images", "Search the web for images.",
       query={"type": "string", "required": True}),
    _t("file_read_cached", "Read a file with a caching layer.",
       path={"type": "string", "required": True},
       ttl_s={"type": "integer", "required": False, "default": 60}),
    _t("email_draft", "Draft (not send) an email.",
       to={"type": "string", "required": True},
       subject={"type": "string", "required": True}),
    _t("db_exec", "Run a read-WRITE sql statement.",
       sql={"type": "string", "required": True}),
    _t("paraphrase", "Paraphrase text in the same language.",
       text={"type": "string", "required": True}),
]


# ── Scripted tasks: (name, golden pre-drift calls) ──────────────────────────

TASKS = [
    ("t01", [ToolCall("search_web", {"query": "colab t4 vram", "limit": 3})]),
    ("t02", [ToolCall("read_file", {"path": "/data/nums.txt"})]),
    ("t03", [ToolCall("calc", {"expression": "(3+7)*11"})]),
    ("t04", [ToolCall("send_email",
                      {"to": "a@x.io", "subject": "hi", "body": "hello"})]),
    ("t05", [ToolCall("db_query", {"sql": "SELECT 1", "timeout_s": 10})]),
    ("t06", [ToolCall("translate", {"text": "hello", "target_lang": "zh"})]),
    ("t07", [ToolCall("summarize", {"document": "long text", "bullets": 5})]),
    ("t08", [ToolCall("write_file", {"path": "/o.txt", "content": "x"})]),
    ("t09", [ToolCall("search_web", {"query": "qwen3.5"}),
             ToolCall("summarize", {"document": "r1 r2", "bullets": 2})]),
    ("t10", [ToolCall("db_query", {"sql": "SELECT a FROM t"}),
             ToolCall("calc", {"expression": "1+2"})]),
    ("t11", [ToolCall("search_web", {"query": "arxiv api"})]),
    ("t12", [ToolCall("read_file", {"path": "/a.txt"}),
             ToolCall("translate", {"text": "bonjour", "target_lang": "en"})]),
    ("t13", [ToolCall("write_file", {"path": "/b.txt", "content": "y"}),
             ToolCall("read_file", {"path": "/b.txt"})]),
    ("t14", [ToolCall("send_email", {"to": "b@x.io", "subject": "s"})]),
    ("t15", [ToolCall("calc", {"expression": "2**10"})]),
    ("t16", [ToolCall("search_web", {"query": "peft torchao", "limit": 8})]),
    ("t17", [ToolCall("db_query", {"sql": "SELECT now()"})]),
    ("t18", [ToolCall("translate", {"text": "hola", "target_lang": "fr"})]),
    ("t19", [ToolCall("summarize", {"document": "doc"})]),
    ("t20", [ToolCall("search_web", {"query": "grpo"}),
             ToolCall("db_query", {"sql": "SELECT 2"}),
             ToolCall("calc", {"expression": "3*3"})]),
    ("t21", [ToolCall("read_file", {"path": "/c.txt"})]),
    ("t22", [ToolCall("send_email",
                      {"to": "c@x.io", "subject": "r", "body": "b"})]),
    ("t23", [ToolCall("write_file", {"path": "/d.txt", "content": "z"})]),
    ("t24", [ToolCall("calc", {"expression": "10/4"}),
             ToolCall("translate", {"text": "x", "target_lang": "de"})]),
]


# ── Drifted-caller stub ─────────────────────────────────────────────────────
# A deployed caller frozen on the old interface emits the golden pre-drift
# calls. Under description-rewrite drift the caller is additionally misled
# into a few seeded arg-form mistakes (e.g. string where int is expected),
# mirroring how stale models mis-fill under misleading descriptions.

def drifted_caller(task_id, golden_calls, scenario, rng):
    out = []
    for call in golden_calls:
        args = dict(call.arguments)
        if scenario == "desc_rewrite" and rng.random() < 0.25:
            for k, v in list(args.items()):
                if isinstance(v, int):
                    args[k] = str(v)  # misled into string form
                    break
        out.append(ToolCall(call.tool_name, args))
    return out


DRIFTS = {
    "control": {},
    "rename": {
        # Realistic renames: semantics preserved, so a repair layer can
        # plausibly recover them via name/candidate-set matching.
        "renames": {
            "search_web": "web_search_v2",
            "read_file": "file_reader",
            "db_query": "database_query",
            "send_email": "send_mail",
        },
    },
    "distractors": {"distractors": DISTRACTORS},
    "schema_edits": {
        "schema_edits": {
            "search_web": {"query": "q", "limit": "max_results"},
            "send_email": {"to": "recipient"},
        },
        "type_edits": {
            "db_query": {"timeout_s": "string"},
            "summarize": {"bullets": "string"},
        },
    },
    "desc_rewrite": {
        "desc_rewrite": {
            "search_web": "Search the web. The limit parameter takes a "
                          "string like '10'.",
            "summarize": "Summarize; bullets is a string count.",
        },
        "type_edits": {
            "search_web": {"limit": "string"},
            "summarize": {"bullets": "string"},
        },
    },
    "combined": {
        "renames": {
            "search_web": "web_search_v2",
            "read_file": "file_reader",
            "db_query": "database_query",
            "send_email": "send_mail",
            "calc": "q9x",  # deliberately unguessable: honest limit case
        },
        "distractors": DISTRACTORS,
        "schema_edits": {
            "search_web": {"query": "q", "limit": "max_results"},
            "send_email": {"to": "recipient"},
        },
        "type_edits": {
            "search_web": {"limit": "string"},
            "db_query": {"timeout_s": "string"},
            "summarize": {"bullets": "string"},
        },
    },
}


def run_scenario(name, seed=7):
    drift = DRIFTS[name]
    sim = DriftSimulator(OLD_REGISTRY, drift)
    shield = SchemaShield(sim.new_registry)
    rng = random.Random(seed)
    stale = []
    for tid, golden in TASKS:
        stale.extend(drifted_caller(tid, golden, name, rng))
    b_ok, s_ok, details = sim.replay_stale(stale, shield)
    total = len(stale)
    drift_failures = total - b_ok
    recovered = s_ok - b_ok
    # harms: baseline-valid calls the shield broke
    harms = sum(1 for _, b, r in details
                if b and not (r is not None and r.ok))
    by_stage = {}
    for _, b, r in details:
        if not b and r is not None and r.ok:
            by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
    return {
        "scenario": name,
        "total": total,
        "baseline_ok": b_ok,
        "shield_ok": s_ok,
        "drift_failures": drift_failures,
        "recovered": recovered,
        "recovery_rate": recovered / drift_failures if drift_failures else 1.0,
        "harms": harms,
        "by_stage": by_stage,
    }


def main():
    results = {}
    for name in DRIFTS:
        r = run_scenario(name)
        results[name] = r
        print("%-13s total=%2d baseline=%2d shield=%2d "
              "failures=%2d recovered=%2d rate=%.3f harms=%d stages=%s"
              % (name, r["total"], r["baseline_ok"], r["shield_ok"],
                 r["drift_failures"], r["recovered"], r["recovery_rate"],
                 r["harms"], r["by_stage"]))

    # ── Assertions: the paper's core claim, operationalized ──
    ctrl = results["control"]
    assert ctrl["baseline_ok"] == ctrl["total"], "control must be all-valid"
    assert ctrl["shield_ok"] == ctrl["total"] and ctrl["harms"] == 0, \
        "shield must be a pass-through with no drift"

    comb = results["combined"]
    assert comb["baseline_ok"] < comb["total"], "drift must actually break calls"
    assert comb["shield_ok"] > comb["baseline_ok"], \
        "shield must recover a strict subset of drift failures"
    assert comb["recovery_rate"] >= 0.60, \
        "recovery rate %.3f below 0.60 bar" % comb["recovery_rate"]
    assert comb["harms"] == 0, "repair layer must not break valid calls"

    # Honest-limit accounting: every unrecovered combined failure must come
    # from a documented hard class (unguessable rename calc->q9x, or the
    # synonym arg rename to->recipient that no training-free layer can infer).
    sim = DriftSimulator(OLD_REGISTRY, DRIFTS["combined"])
    shield = SchemaShield(sim.new_registry)
    rng = random.Random(7)
    stale = []
    for tid, golden in TASKS:
        stale.extend(drifted_caller(tid, golden, "combined", rng))
    _, _, details = sim.replay_stale(stale, shield)
    hard_tools = {"calc", "send_email"}
    for call, b_ok, res in details:
        if not b_ok and not (res is not None and res.ok):
            assert call.tool_name in hard_tools, \
                "unexpected unrecovered call outside hard classes: %r" % (call,)

    for name, r in results.items():
        assert r["harms"] == 0, "harms in scenario %s" % name
        assert r["shield_ok"] >= r["baseline_ok"], \
            "shield regressed baseline in %s" % name

    # name_retry must fire on renames; arg_repair must fire on schema edits;
    # both stages must contribute under combined drift (the paper's ablation:
    # candidate-set drift is the dominant damage source, schema edits add)
    assert results["rename"]["by_stage"].get("name_retry", 0) > 0
    assert results["schema_edits"]["by_stage"].get("arg_repair", 0) > 0
    comb_stages = results["combined"]["by_stage"]
    assert comb_stages.get("name_retry", 0) > 0
    assert comb_stages.get("name_retry+arg_repair", 0) > 0 or \
        comb_stages.get("arg_repair", 0) > 0

    print("\nALL SCHEMA-SHIELD CHECKS PASSED")
    print("combined: baseline %.3f -> shield %.3f, recovery %.3f, harms %d"
          % (comb["baseline_ok"] / comb["total"],
             comb["shield_ok"] / comb["total"],
             comb["recovery_rate"], comb["harms"]))


if __name__ == "__main__":
    main()
