#!/usr/bin/env python3
"""Live DART experiment (arXiv 2611.34928) on MATH-500 via NVIDIA NIM.

Generator: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning (gpt-oss-20b is
retired from NIM -- health check HTTP 410 Gone, 2026-11-28).

Design (matched trace budget k=3 per greedy-unsolved problem):
  screening: greedy (temp 0) on N Level-5 MATH-500 problems.
  MC arm:    3 fresh samples (temp 0.7) per unsolved problem.
  DART arm:  1 step revision (temp 0.7, reference-answer guidance +
             meta-thought) + 2 completions from the revised step.
  Recovery = fraction of unsolved problems solved by >= 1 trace.
The revision target is the top position-weighted step (the NIM endpoint
exposes no token logprobs, so the position-only proxy is used -- see the
session README and the E6 oracle ablation in tests/test_dart_cpu.py).

State is saved to live_state.json after every problem, so the run is
resumable: re-run with the same args to continue. The NIM client is NOT
part of this session (surrogate credentials must never be committed):
it is imported from $NIM_CLIENT_PATH (default /tmp/dart/nim_client.py).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/arxiv-lab-build/src"))
sys.path.insert(0, os.environ.get("NIM_CLIENT_PATH", "/tmp/dart"))

from arxiv_lab.agent.dart import (  # noqa: E402
    answers_equal,
    extract_answer,
    rank_steps,
    split_steps,
)
from nim_client import generate  # noqa: E402  (surrogate client, /tmp only)

SESSION = Path(__file__).resolve().parent
STATE_FILE = SESSION / "live_state.json"
MATH500 = Path("/tmp/math500.json")

SOLVE_SUFFIX = "\n\nSolve step by step. Put your final answer in \\boxed{}."
K = 3
MAX_TOKENS = 16384
TEMP_SAMPLE = 0.7


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"screen": {}, "mc": {}, "dart": {}}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=1))


def is_box_only(step: str) -> bool:
    s = step.strip()
    return bool(re.fullmatch(r"\\boxed\{.*\}", s, re.DOTALL))


def solve_prompt(problem: str) -> str:
    return problem + SOLVE_SUFFIX


def revise_prompt(problem: str, steps: list[str], j: int, gold: str) -> str:
    numbered = "\n\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
    return (
        f"{problem}\n\nA previous attempt at this problem went wrong. "
        f"Its reasoning steps were:\n\n{numbered}\n\n"
        f"The correct final answer is {gold}. Step {j + 1} is where the "
        f"reasoning should be repaired.\n\n"
        f"First write one line starting with 'Meta-thought:' (one sentence "
        f"on what went wrong in that step). Then, after a blank line, write "
        f"a corrected replacement for Step {j + 1} only, as a single "
        f"paragraph. Do not solve the whole problem."
    )


def complete_prompt(problem: str, prefix_steps: list[str], revised: str) -> str:
    body = "\n\n".join(prefix_steps + [revised])
    return (
        f"{problem}\n\nContinue this partial solution step by step to the "
        f"final answer. Put your final answer in \\boxed{{}}.\n\n{body}"
    )


def parse_revision(text: str) -> tuple[str, str]:
    """Split a revision reply into (meta-thought, replacement step)."""
    text = (text or "").strip()
    meta = ""
    m = re.match(r"(?i)meta-?thought:\s*(.*)", text)
    if m:
        first, _, rest = text.partition("\n")
        meta = first.strip()
        body = rest.strip()
        # drop a leading blank-line-separated second meta line if present
        text = body or m.group(1)
    # if the model returned meta-thought + step in one block, keep the
    # last paragraph as the step when a blank line separates them
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    step = parts[-1] if parts else text
    return meta, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-screen", type=int, default=30)
    ap.add_argument("--phases", default="screen,mc,dart")
    args = ap.parse_args()
    phases = set(args.phases.split(","))

    rows = json.loads(MATH500.read_text())
    level5 = [r for r in rows if r["level"] == 5][: args.n_screen]
    print(f"screening {len(level5)} Level-5 problems", flush=True)
    state = load_state()
    t_start = time.time()

    if "screen" in phases:
        for r in level5:
            uid = r["unique_id"]
            if uid in state["screen"]:
                continue
            out = generate([{"role": "user", "content": solve_prompt(r["problem"])}],
                           temperature=0.0, max_tokens=MAX_TOKENS)
            ans, method = extract_answer(out["content"])
            state["screen"][uid] = {
                "problem": r["problem"], "gold": r["answer"],
                "subject": r["subject"], "trace": out["content"],
                "answer": ans, "extracted_via": method,
                "correct": answers_equal(ans, r["answer"]),
                "seconds": out["seconds"],
                "tokens": out["usage"].get("completion_tokens"),
                "finish_reason": out["finish_reason"],
            }
            save_state(state)
            print(f"screen {uid} correct={state['screen'][uid]['correct']} "
                  f"({out['seconds']}s)", flush=True)

    unsolved = [uid for uid, s in state["screen"].items() if not s["correct"]]
    applicable = [uid for uid in unsolved
                  if (state["screen"][uid].get("trace") or "").strip()]
    print(f"unsolved: {len(unsolved)} / {len(state['screen'])} "
          f"({len(applicable)} DART-applicable, "
          f"{len(unsolved) - len(applicable)} empty-trace)", flush=True)

    if "mc" in phases:
        for uid in applicable:
            if uid in state["mc"]:
                continue
            s = state["screen"][uid]
            traces = []
            for _ in range(K):
                out = generate(
                    [{"role": "user", "content": solve_prompt(s["problem"])}],
                    temperature=TEMP_SAMPLE, max_tokens=MAX_TOKENS)
                ans, method = extract_answer(out["content"])
                traces.append({"answer": ans, "extracted_via": method,
                               "correct": answers_equal(ans, s["gold"]),
                               "seconds": out["seconds"],
                               "finish_reason": out["finish_reason"],
                               "trace": out["content"][-8000:]})
            state["mc"][uid] = {"traces": traces}
            save_state(state)
            print(f"mc {uid} solved={any(t['correct'] for t in traces)}",
                  flush=True)

    if "dart" in phases:
        for uid in applicable:
            if uid in state["dart"]:
                continue
            s = state["screen"][uid]
            steps = split_steps(s["trace"])
            if not steps:
                state["dart"][uid] = {
                    "error": "empty rejected trace (reasoning overflow) — "
                             "not DART-applicable"}
                save_state(state)
                print(f"dart {uid} SKIP empty trace", flush=True)
                continue
            revisable = [i for i, st in enumerate(steps)
                         if not is_box_only(st)]
            if not revisable:
                state["dart"][uid] = {"error": "no revisable step"}
                save_state(state)
                continue
            # position-only ranking restricted to revisable steps
            order = [i for i in rank_steps(steps) if i in revisable]
            j = order[0]
            rev = generate(
                [{"role": "user",
                  "content": revise_prompt(s["problem"], steps, j, s["gold"])}],
                temperature=TEMP_SAMPLE, max_tokens=MAX_TOKENS)
            meta, revised_step = parse_revision(rev["content"])
            completions = []
            for _ in range(K - 1):
                out = generate(
                    [{"role": "user", "content": complete_prompt(
                        s["problem"], steps[:j], revised_step)}],
                    temperature=TEMP_SAMPLE, max_tokens=MAX_TOKENS)
                ans, method = extract_answer(out["content"])
                completions.append({"answer": ans, "extracted_via": method,
                                    "correct": answers_equal(ans, s["gold"]),
                                    "seconds": out["seconds"],
                                    "finish_reason": out["finish_reason"],
                                    "trace": out["content"][-8000:]})
            state["dart"][uid] = {
                "revise_step": j, "n_steps": len(steps),
                "meta_thought": meta, "revised_step": revised_step,
                "revision_seconds": rev["seconds"],
                "revision_finish": rev["finish_reason"],
                "revision_content_chars": len(rev["content"] or ""),
                "completions": completions,
            }
            save_state(state)
            print(f"dart {uid} revised_step={j + 1}/{len(steps)} "
                  f"solved={any(c['correct'] for c in completions)}",
                  flush=True)

    n = len(unsolved)
    applicable = [uid for uid in unsolved
                  if (state["screen"][uid].get("trace") or "").strip()]
    n_app = len(applicable)
    mc_rec = (sum(1 for uid in applicable
                  if any(t["correct"] for t in state["mc"].get(uid, {})
                         .get("traces", []))) / n_app) if n_app else 0.0
    dart_rec = (sum(1 for uid in applicable
                    if any(c["correct"] for c in state["dart"].get(uid, {})
                           .get("completions", []))) / n_app) if n_app else 0.0
    summary = {
        "n_screen": len(state["screen"]),
        "greedy_solved": sum(1 for s in state["screen"].values()
                             if s["correct"]),
        "n_unsolved": n,
        "n_dart_applicable": n_app,
        "n_empty_traces": n - n_app,
        "k": K,
        "mc_recovery": mc_rec,
        "dart_recovery": dart_rec,
        "relative_gain": ((dart_rec - mc_rec) / mc_rec) if mc_rec else None,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    (SESSION / "results.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
