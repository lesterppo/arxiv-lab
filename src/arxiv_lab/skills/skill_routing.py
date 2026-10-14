"""Skill routing collapse and its mitigations (arXiv:2610.18522).

Paper: "Lost in Implicit Routing: How LLM Agents Fail to Load Relevant
Skills at Scale".

One-line claim: when agents route implicitly over skill metadata, adding
irrelevant skills collapses loaded-skill recall (73.6% -> 27.4% at 1,000
irrelevant skills; zero-load rate 9.3% -> 51.6%). Masking distractor
descriptions largely restores loading; explicit preselection from the same
library retains 73.9% recall.

This module is the LLM-agnostic harness: skill-library construction,
prompt rendering for the implicit-routing and explicit-preselection
protocols, a deterministic loader-line parser, and the routing metrics
(recall, zero-load rate, wrong-load count, precision). The LLM itself is
injected as a ``router`` callable so the harness is testable on CPU with
scripted routers and drivable live against any chat model.
"""

from __future__ import annotations

import random
import re

__all__ = [
    "Skill",
    "Task",
    "build_library",
    "render_library",
    "route_prompt",
    "preselect_prompt",
    "parse_loaded",
    "routing_scores",
    "aggregate",
    "collapse_gap",
]

# A skill is a plain dict: {"name": str, "description": str,
# "instructions": str}.  A task is {"id": str, "prompt": str,
# "golden": [skill names]}.

Skill = dict
Task = dict

_LOAD_RE = re.compile(r"^\s*(?:LOAD|SELECT)\s*:\s*([A-Za-z0-9_\-]+)\s*$",
                      re.IGNORECASE | re.MULTILINE)


def build_library(golden_skills, distractor_pool, n_distractors, seed=0):
    """Assemble a library: all golden skills + ``n_distractors`` sampled
    without replacement from ``distractor_pool`` (seeded)."""
    rng = random.Random(seed)
    distractors = rng.sample(distractor_pool, min(n_distractors,
                                                  len(distractor_pool)))
    lib = list(golden_skills) + list(distractors)
    rng.shuffle(lib)
    return lib


def render_library(skills, mask_descriptions=False):
    """Render the skill catalogue block. When ``mask_descriptions`` is true
    only names are shown (the paper's masking mitigation)."""
    lines = ["## Skill library (%d skills)" % len(skills)]
    for s in skills:
        if mask_descriptions:
            lines.append("- %s" % s["name"])
        else:
            lines.append("- %s :: %s" % (s["name"], s["description"]))
    return "\n".join(lines)


def route_prompt(task, library_text):
    """Implicit-routing prompt: the agent sees task + full library and must
    emit LOAD lines before solving."""
    return (
        "You are an agent with a library of reusable skills. A skill's "
        "instructions are loaded into your context only after you load it.\n\n"
        "Task:\n%s\n\n%s\n\n"
        "First, decide which skills to load for this task. Write one line "
        "per skill you load, exactly in this format:\n"
        "LOAD: <skill name>\n"
        "Load every skill whose instructions you will need, and no others. "
        "If no skill is needed, write exactly: LOAD: none\n"
        "Then solve the task." % (task["prompt"], library_text)
    )


def preselect_prompt(task, library_text):
    """Explicit-preselection prompt: a dedicated selection call over the
    same library (the paper's mitigation)."""
    return (
        "You are selecting skills for a task. Read the task and the skill "
        "catalogue, then select the skills whose instructions are directly "
        "needed to solve the task.\n\n"
        "Task:\n%s\n\n%s\n\n"
        "Write one line per selected skill, exactly in this format:\n"
        "SELECT: <skill name>\n"
        "Select only skills directly needed. If none apply, write exactly: "
        "SELECT: none" % (task["prompt"], library_text)
    )


def parse_loaded(text, none_token="none"):
    """Deterministically extract LOAD:/SELECT: lines -> ordered, deduped
    skill names (case-insensitive match later)."""
    names = []
    seen = set()
    for m in _LOAD_RE.finditer(text or ""):
        name = m.group(1).strip()
        if name.lower() == none_token:
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            names.append(name)
    return names


def routing_scores(loaded, golden):
    """Per-task routing metrics.

    ``loaded``: names parsed from the agent reply; ``golden``: the task's
    golden skill names. Matching is case-insensitive.
    """
    g = {x.lower() for x in golden}
    l = {x.lower() for x in loaded}
    hits = l & g
    return {
        "recall": len(hits) / len(g) if g else 1.0,
        "zero_load": len(loaded) == 0,
        "wrong_load": len(l - g),
        "precision": len(hits) / len(l) if l else 1.0,
        "n_loaded": len(loaded),
        "n_golden": len(g),
    }


def aggregate(per_task_scores):
    """Mean metrics over tasks; also returns raw counts."""
    n = len(per_task_scores)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for k in ("recall", "precision"):
        out["mean_" + k] = sum(s[k] for s in per_task_scores) / n
    out["zero_load_rate"] = sum(
        1 for s in per_task_scores if s["zero_load"]) / n
    out["mean_wrong_load"] = sum(
        s["wrong_load"] for s in per_task_scores) / n
    return out


def collapse_gap(agg_baseline, agg_stressed):
    """Recall drop and zero-load rise of a stressed condition relative to
    the no-distractor baseline (the paper's collapse signature)."""
    return {
        "recall_drop": agg_baseline["mean_recall"]
        - agg_stressed["mean_recall"],
        "zero_load_rise": agg_stressed["zero_load_rate"]
        - agg_baseline["zero_load_rate"],
    }
