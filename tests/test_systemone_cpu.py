"""CPU test for arxiv_lab.backends.systemone (Clef typed-decision client).

No network: validates request-shape building (including the live-verified
quirks — questions/criteria must be objects, never lists; choice criteria
need 2-26 entries) and response parsing against the verbatim 2026-10-05
clef-flash:latest fixture. A live test runs only when SYSTEMONE_BASE_URL is
set (e.g. a Colab VM running clef-flash via Ollama >= 0.35).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.backends.systemone import (
    SystemoneClient, decide, build_request, parse_answers, score_value,
    QUESTION_TYPES,
)

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except Exception as e:  # noqa: BLE001 - test harness reports, not raises
        FAIL.append((name, f"{type(e).__name__}: {e}"))


S = "some state"


def expect_raises(name, exc, fn):
    def _f():
        try:
            fn()
        except exc:
            return
        except Exception as e:  # noqa: BLE001
            raise AssertionError(
                f"expected {exc.__name__}, got {type(e).__name__}: {e}")
        raise AssertionError(f"expected {exc.__name__}, nothing raised")
    check(name, _f)


GOOD_QUESTIONS = {
    "page_oncall": {
        "type": "noul",
        "instructions": "Should the on-call engineer be paged?",
    },
    "severity": {
        "type": "choice",
        "instructions": "How severe is this incident?",
        "criteria": {
            "low": "Minor impact, no customer-facing effect",
            "medium": "Degraded experience for some customers",
            "high": "Critical outage affecting checkout",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "Rate the urgency of a response.",
        "criteria": [
            {"score": 0.0, "description": "no urgency"},
            {"score": 0.5, "description": "handle today"},
            {"score": 1.0, "description": "drop everything"},
        ],
    },
}

# Verbatim response shape from the 2026-10-05 live T4 test.
FIXTURE = {
    "model": "clef-flash:latest",
    "answers": {
        "page_oncall": {"type": "noul", "noul": 0.9280018157114909},
        "severity": {
            "type": "choice",
            "choice": "high",
            "probabilities": {
                "high": 0.5481548640781726,
                "low": 0.04502363207630507,
                "medium": 0.4068215038455223,
            },
            "confidence": 0.23991788568091543,
        },
    },
    "usage": {"input_tokens": 269, "output_tokens": 0},
}


def t_build_ok():
    body = build_request("clef-flash:latest", "db latency p99 4.2s",
                         GOOD_QUESTIONS)
    assert body["model"] == "clef-flash:latest"
    assert set(body["questions"]) == {"page_oncall", "severity", "urgency"}
    assert body["questions"]["severity"]["criteria"]["high"].startswith(
        "Critical")


def t_parse_fixture():
    ans = parse_answers(FIXTURE)
    assert ans["page_oncall"] == {"type": "noul", "p": 0.9280018157114909}
    sev = ans["severity"]
    assert sev["type"] == "choice" and sev["choice"] == "high"
    probs = sev["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-9, probs
    assert probs["high"] > probs["medium"] > probs["low"]
    assert sev["confidence"] == 0.23991788568091543


def t_parse_score():
    # Shape observed live 2026-10-07: score = expected criteria-array index.
    ans = parse_answers({"answers": {"urgency": {
        "type": "score", "score": 1.4635095461108958,
        "legend": {"0": 0, "1": 0.5, "2": 1},
        "probabilities": {"0": 0.09641359668229478,
                          "1": 0.34366326052451457,
                          "2": 0.5599231427931907},
        "confidence": 0.165}}})
    u = ans["urgency"]
    assert u["type"] == "score"
    assert abs(u["score"] - 1.4635095461108958) < 1e-12
    assert abs(sum(u["probabilities"].values()) - 1.0) < 1e-9
    assert u["legend"] == {"0": 0, "1": 0.5, "2": 1}
    assert u["confidence"] == 0.165


def t_score_value_numeric():
    ans = parse_answers({"answers": {"urgency": {
        "type": "score", "score": 1.4635095461108958,
        "legend": {"0": 0, "1": 0.5, "2": 1},
        "probabilities": {"0": 0.09641359668229478,
                          "1": 0.34366326052451457,
                          "2": 0.5599231427931907}}}})["urgency"]
    v = score_value(ans)
    assert abs(v - (0 * 0.09641359668229478 + 0.5 * 0.34366326052451457
                    + 1 * 0.5599231427931907)) < 1e-9


def t_score_value_object_criteria():
    ans = parse_answers({"answers": {"p": {
        "type": "score", "score": 0.663,
        "legend": {"0": {"score": 0, "description": "low"},
                   "1": {"score": 1, "description": "high"}},
        "probabilities": {"0": 0.337, "1": 0.663}}}})["p"]
    assert abs(score_value(ans) - 0.663) < 1e-9


def t_score_value_non_numeric():
    ans = parse_answers({"answers": {"p": {
        "type": "score", "score": 1.1,
        "legend": {"0": "low", "1": "high"},
        "probabilities": {"0": 0.4, "1": 0.6}}}})["p"]
    assert score_value(ans) is None


def t_question_types():
    assert set(QUESTION_TYPES) == {"noul", "choice", "score"}


def t_criteria_boundaries_ok():
    q2 = {"type": "choice", "instructions": "x",
          "criteria": {"a": "da", "b": "db"}}
    q26 = {"type": "choice", "instructions": "x",
           "criteria": {f"c{i}": "d" for i in range(26)}}
    body = build_request("m", S, {"q2": q2, "q26": q26})
    assert len(body["questions"]["q26"]["criteria"]) == 26


def t_bool_rejected():
    for bad in ({"answers": {"q": {"type": "noul", "noul": True}}},
                {"answers": {"q": {"type": "score", "score": False,
                                   "legend": {}, "probabilities": {}}}}):
        try:
            parse_answers(bad)
        except RuntimeError:
            continue
        raise AssertionError(f"bool accepted: {bad}")


def t_choice_int_probs_to_float():
    ans = parse_answers({"answers": {"c": {
        "type": "choice", "choice": "a",
        "probabilities": {"a": 1, "b": 0}}}})["c"]
    assert ans["probabilities"] == {"a": 1.0, "b": 0.0}
    assert ans["confidence"] is None


def t_unicode_passthrough():
    body = build_request("m", "延迟 p99 为 4.2 秒 🚨",
                         {"q": {"type": "noul",
                                "instructions": "要叫醒值班吗？"}})
    assert "🚨" in body["state"]


def t_extra_keys_passthrough():
    body = build_request("m", S, {"q": {"type": "noul",
                                        "instructions": "x",
                                        "temperature": 0.1}})
    assert body["questions"]["q"]["temperature"] == 0.1


def t_score_value_empty_probs():
    ans = {"type": "score", "score": 0.0, "legend": {"0": 0},
           "probabilities": {}, "confidence": None}
    assert score_value(ans) == 0.0


for good in (t_build_ok, t_parse_fixture, t_parse_score,
               t_score_value_numeric, t_score_value_object_criteria,
               t_score_value_non_numeric, t_question_types,
               t_criteria_boundaries_ok, t_bool_rejected,
               t_choice_int_probs_to_float, t_unicode_passthrough,
               t_extra_keys_passthrough, t_score_value_empty_probs):
    check(good.__name__, good)

# Shape violations must fail fast, before any network call.
expect_raises("questions_as_list", ValueError,
              lambda: build_request("m", S, [{"type": "noul"}]))
expect_raises("questions_empty", ValueError,
              lambda: build_request("m", S, {}))
expect_raises("state_empty", ValueError,
              lambda: build_request("m", "  ", GOOD_QUESTIONS))
expect_raises("bad_type", ValueError,
              lambda: build_request("m", S, {"q": {"type": "maybe",
                                                   "instructions": "x"}}))
expect_raises("missing_instructions", ValueError,
              lambda: build_request("m", S, {"q": {"type": "noul"}}))
expect_raises("choice_criteria_as_list", ValueError,
              lambda: build_request("m", S, {"q": {"type": "choice",
                                                   "instructions": "x",
                                                   "criteria": ["a", "b"]}}))
expect_raises("choice_criteria_missing", ValueError,
              lambda: build_request("m", S, {"q": {"type": "choice",
                                                   "instructions": "x"}}))
expect_raises("choice_criteria_too_few", ValueError,
              lambda: build_request("m", S, {"q": {
                  "type": "choice", "instructions": "x",
                  "criteria": {"only": "one"}}}))
expect_raises("choice_criteria_too_many", ValueError,
              lambda: build_request("m", S, {"q": {
                  "type": "choice", "instructions": "x",
                  "criteria": {f"c{i}": "d" for i in range(27)}}}))
expect_raises("score_criteria_as_object", ValueError,
              lambda: build_request("m", S, {"q": {
                  "type": "score", "instructions": "x",
                  "criteria": {"a": "b"}}}))
expect_raises("score_criteria_missing", ValueError,
              lambda: build_request("m", S, {"q": {"type": "score",
                                                   "instructions": "x"}}))
expect_raises("score_criteria_empty", ValueError,
              lambda: build_request("m", S, {"q": {"type": "score",
                                                   "instructions": "x",
                                                   "criteria": []}}))
expect_raises("noul_with_criteria", ValueError,
              lambda: build_request("m", S, {"q": {"type": "noul",
                                                   "instructions": "x",
                                                   "criteria": ["a"]}}))
expect_raises("parse_no_answers", RuntimeError,
              lambda: parse_answers({"model": "m"}))
expect_raises("parse_unknown_type", RuntimeError,
              lambda: parse_answers(
                  {"answers": {"q": {"type": "weird"}}}))
expect_raises("parse_noul_not_number", RuntimeError,
              lambda: parse_answers(
                  {"answers": {"q": {"type": "noul", "noul": "high"}}}))

# Live test: only when pointed at a real endpoint.
LIVE = os.environ.get("SYSTEMONE_BASE_URL")


def t_live():
    base = LIVE
    model = os.environ.get("SYSTEMONE_MODEL", "clef-flash:latest")
    ans = decide("Production database latency p99 is 4.2s, error rate 2.1%.",
                 {"page_oncall": {"type": "noul",
                                  "instructions": "Page the on-call?"},
                  "severity": {"type": "choice",
                               "instructions": "Incident severity?",
                               "criteria": {
                                   "low": "minor",
                                   "medium": "degraded",
                                   "high": "critical"}},
                  "urgency": {"type": "score",
                              "instructions": "Response urgency?",
                              "criteria": [0, 0.5, 1]}},
                 model=model, base_url=base, timeout=120)
    assert ans["page_oncall"]["type"] == "noul"
    assert 0.0 <= ans["page_oncall"]["p"] <= 1.0
    assert ans["severity"]["choice"] in ("low", "medium", "high")
    v = score_value(ans["urgency"])
    assert v is not None and 0.0 <= v <= 1.0, v
    print("live answers:", ans)


if LIVE:
    check("t_live", t_live)
else:
    print("SKIP t_live (SYSTEMONE_BASE_URL not set)")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for name, err in FAIL:
    print("FAIL", name, "-", err)
sys.exit(1 if FAIL else 0)
