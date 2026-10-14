"""CPU tests for arxiv_lab.skills.skill_routing (arXiv:2610.18522).

Deterministic, model-free: scripted routers stand in for the LLM so the
harness logic (library build, rendering, parsing, metrics, the collapse
signature, and the preselection mitigation) is validated end to end.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.skills.skill_routing import (
    aggregate,
    build_library,
    collapse_gap,
    parse_loaded,
    preselect_prompt,
    render_library,
    route_prompt,
    routing_scores,
)

GOLDEN = [
    {"name": "csv_to_json", "description": "Convert CSV files to JSON.",
     "instructions": "Read the CSV..."},
    {"name": "date_parse", "description": "Parse dates into ISO format.",
     "instructions": "Use dateutil..."},
]
DISTRACTORS = [
    {"name": "knit_scarf", "description": "Knit a scarf pattern.",
     "instructions": "..."},
    {"name": "pizza_dough", "description": "Pizza dough hydration.",
     "instructions": "..."},
    {"name": "star_chart", "description": "Astrology charts.",
     "instructions": "..."},
    {"name": "csv_to_xml", "description": "Convert CSV files to XML.",
     "instructions": "..."},
    {"name": "herb_garden", "description": "Plan a kitchen herb garden.",
     "instructions": "..."},
    {"name": "chord_prog", "description": "Suggest guitar chord progressions.",
     "instructions": "..."},
    {"name": "marathon_plan", "description": "Build a marathon training plan.",
     "instructions": "..."},
    {"name": "sourdough", "description": "Sourdough starter schedule.",
     "instructions": "..."},
    {"name": "flight_deals", "description": "Find cheap flight deals.",
     "instructions": "..."},
    {"name": "haiku_write", "description": "Compose haikus on a theme.",
     "instructions": "..."},
    {"name": "tax_estimate", "description": "Estimate quarterly taxes.",
     "instructions": "..."},
    {"name": "yoga_flow", "description": "Design a morning yoga flow.",
     "instructions": "..."},
    {"name": "wine_pair", "description": "Pair wine with dinner menus.",
     "instructions": "..."},
    {"name": "chess_open", "description": "Recommend chess openings.",
     "instructions": "..."},
    {"name": "budget_50", "description": "50/30/20 budgeting worksheet.",
     "instructions": "..."},
]
TASKS = [
    {"id": "t1", "prompt": "Turn this CSV into JSON.",
     "golden": ["csv_to_json"]},
    {"id": "t2", "prompt": "What day is 2026-10-14?",
     "golden": ["date_parse"]},
]


def collapsing_router(task, library_text):
    """Scripted stand-in for the paper's collapse: loads nothing once the
    library is large (zero-load phenomenon)."""
    n = library_text.count("\n- ")
    if n > 10:
        return "I can answer directly.\nLOAD: none"
    return "LOAD: %s" % task["golden"][0]


def perfect_router(task, library_text):
    return "LOAD: %s" % task["golden"][0]


def perfect_preselector(task, library_text):
    return "SELECT: %s" % task["golden"][0]


def run_condition(tasks, golden, distractor_pool, n_distractors, router,
                  seed=0, mask=False, preselect=None):
    per_task = []
    for t in tasks:
        lib = build_library(golden, distractor_pool, n_distractors, seed)
        if preselect is not None:
            sel = parse_loaded(preselect(t, render_library(lib)))
            sel_names = {s.lower() for s in sel}
            lib = [s for s in lib if s["name"].lower() in sel_names]
        loaded = parse_loaded(router(t, render_library(lib,
                                                       mask_descriptions=mask)))
        per_task.append(routing_scores(loaded, t["golden"]))
    return aggregate(per_task)


def test_build_library_deterministic():
    a = build_library(GOLDEN, DISTRACTORS, 3, seed=7)
    b = build_library(GOLDEN, DISTRACTORS, 3, seed=7)
    assert [s["name"] for s in a] == [s["name"] for s in b]
    names = {s["name"] for s in a}
    assert {"csv_to_json", "date_parse"} <= names
    assert len(a) == 2 + 3


def test_render_mask_hides_descriptions():
    full = render_library(GOLDEN)
    masked = render_library(GOLDEN, mask_descriptions=True)
    assert "Convert CSV files to JSON." in full
    assert "Convert CSV files to JSON." not in masked
    assert "csv_to_json" in masked


def test_parse_loaded_variants():
    assert parse_loaded("LOAD: csv_to_json\nLOAD: date_parse") == \
        ["csv_to_json", "date_parse"]
    assert parse_loaded("load: CSV_TO_JSON") == ["CSV_TO_JSON"]
    assert parse_loaded("LOAD: none") == []
    assert parse_loaded("SELECT: knit_scarf") == ["knit_scarf"]
    assert parse_loaded("") == []


def test_routing_scores():
    s = routing_scores(["csv_to_json", "pizza_dough"], ["csv_to_json"])
    assert s["recall"] == 1.0
    assert s["wrong_load"] == 1
    assert abs(s["precision"] - 0.5) < 1e-9
    assert s["zero_load"] is False
    s2 = routing_scores([], ["csv_to_json"])
    assert s2["recall"] == 0.0 and s2["zero_load"] is True


def test_collapse_signature_detected():
    base = run_condition(TASKS, GOLDEN, DISTRACTORS, 0, collapsing_router)
    stressed = run_condition(TASKS, GOLDEN, DISTRACTORS, 40,
                             collapsing_router)
    assert base["mean_recall"] == 1.0
    assert stressed["mean_recall"] == 0.0
    assert stressed["zero_load_rate"] == 1.0
    gap = collapse_gap(base, stressed)
    assert gap["recall_drop"] == 1.0
    assert gap["zero_load_rise"] == 1.0


def test_explicit_preselection_restores_recall():
    # Perfect implicit routing collapses under the scripted router; routing
    # only over the preselected (golden) subset restores recall to 1.0.
    collapsed = run_condition(TASKS, GOLDEN, DISTRACTORS, 40,
                              collapsing_router)
    mitigated = run_condition(TASKS, GOLDEN, DISTRACTORS, 40,
                              collapsing_router,
                              preselect=perfect_preselector)
    assert collapsed["mean_recall"] == 0.0
    assert mitigated["mean_recall"] == 1.0


def test_prompts_contain_task_and_library():
    lib_text = render_library(GOLDEN)
    rp = route_prompt(TASKS[0], lib_text)
    assert "LOAD: <skill name>" in rp and TASKS[0]["prompt"] in rp
    pp = preselect_prompt(TASKS[0], lib_text)
    assert "SELECT: <skill name>" in pp


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL SKILL ROUTING CPU TESTS PASS")
