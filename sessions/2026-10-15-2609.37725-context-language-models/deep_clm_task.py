#!/usr/bin/env python3
"""Deep-test task for arXiv 2609.37725 (Context Language Models): live loop.

Design: a synthetic long-horizon research task. 8 questions, each answered
by exactly 2 "golden" docs in a 48-doc pool (32 distractors). A scripted
`search(query)` returns the top-3 keyword-ranked docs (~1650 tokens per
search) into the agent's context file. Budget: 6000 tokens — about 3-4
searches fit, so a 8-question run (~10-16 searches) forces context
management.

Two arms, same task/retriever/grading:
  * clm      — the model may edit its context file at any time via BASH
               (delete/prune/rewrite/compact); edits sync to the next turn.
               Mirrors the paper's zero-shot CLM setup (Sec. 4.1).
  * baseline — fixed harness policy: append-only + head truncation of the
               oldest tool-output lines when over budget (questions pinned).

The model client is injected as chat_fn(system, user_content) -> str, so
this file is committable: no credentials, no network. Grading is
deterministic exact-match (normalized).

Metrics per arm: accuracy, turns, searches, bash edits, token trajectory,
peak tokens, within-budget fraction.
"""
import os
import random
import re
import subprocess

BUDGET_TOKENS = 6000
CHARS_PER_TOKEN = 4
MAX_TURNS = 40
SEED = 7
SAFETY_MULT = 4  # hard cap on file size (x budget) so a runaway arm can't
# blow up the API call; the *reported* budget stays 6000.

QUESTIONS = [
    ("Q1", "Which award did Dr. Lee win in 2019?", "Turing Award"),
    ("Q2", "Where was the 2019 ceremony held?", "San Francisco"),
    ("Q3", "Who did Dr. Lee thank on stage?", "Dr. Rao"),
    ("Q4", "What was the prize money?", "one million dollars"),
    ("Q5", "What project did Dr. Lee announce at the banquet?", "Helios"),
    ("Q6", "What work did the award recognize?", "sparse attention"),
    ("Q7", "Which company sponsored the 2019 ceremony?", "Meridian Labs"),
    ("Q8", "What city will host the 2020 ceremony?", "Boston"),
]

FACT_SENTENCES = {
    "Q1": "In 2019, Dr. Elena Lee received the prestigious Turing Award for her contributions to computer science.",
    "Q2": "The 2019 award ceremony honoring Dr. Lee was held in San Francisco.",
    "Q3": "On stage, Dr. Lee thanked her longtime collaborator Dr. Rao.",
    "Q4": "The prize money for the 2019 award totaled one million dollars.",
    "Q5": "At the banquet, Dr. Lee announced the Helios project.",
    "Q6": "The award recognized Dr. Lee's pioneering work on sparse attention.",
    "Q7": "The 2019 ceremony was sponsored by Meridian Labs.",
    "Q8": "The 2020 ceremony will take place in Boston.",
}

STOP = set(("which what where when will did the and for was were are "
            "with from that this have has had who whom whose".split()))

FILLER_TOPICS = [
    "the annual review covered budget allocations and hiring plans",
    "attendees discussed venue logistics and catering arrangements",
    "the committee published minutes after a lengthy deliberation",
    "reporters asked about future plans and upcoming publications",
    "the program listed keynotes panels and poster sessions",
    "organizers thanked volunteers sponsors and local staff",
    "a retrospective panel recalled milestones of the past decade",
    "the newsletter summarized travel grants and fellowship news",
]


def _keywords(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 3 and w not in STOP]


def _filler(rng, nchars, pool):
    parts = []
    while len(" ".join(parts)) < nchars:
        parts.append(rng.choice(pool))
    return " ".join(parts)[:nchars]


class DocStore(object):
    """48 docs: 2 golden per question + 32 distractors. Deterministic."""

    def __init__(self, seed=SEED):
        rng = random.Random(seed)
        self.docs = []
        did = 0
        for qid, _, _ in QUESTIONS:
            for _ in range(2):
                body = (FACT_SENTENCES[qid] + " " +
                        _filler(rng, 1900, FILLER_TOPICS))
                self.docs.append({"id": "d%02d" % did, "text": body,
                                  "golden_for": qid})
                did += 1
        for _ in range(32):
            # distractors share vocabulary (award, ceremony, 2019, Dr. Lee)
            # but carry no answer-bearing sentence
            lead = rng.choice([
                "The 2019 award season drew wide attention.",
                "Dr. Lee attended the ceremony with colleagues.",
                "Ceremony organizers released the seating plan.",
                "Reporters covered the award extensively.",
            ])
            body = lead + " " + _filler(rng, 2000, FILLER_TOPICS)
            self.docs.append({"id": "d%02d" % did, "text": body,
                              "golden_for": None})
            did += 1

    def search(self, query):
        kws = _keywords(query)
        def score(d):
            t = d["text"].lower()
            return sum(t.count(k) for k in kws)
        ranked = sorted(self.docs, key=score, reverse=True)
        return ranked[:3]


def estimate_tokens(text):
    return len(text) // CHARS_PER_TOKEN


SYSTEM_BASE = """You are a research agent. Your entire working context is a file; each turn you receive its current content. End EVERY turn with exactly one action line:
  SEARCH: <query>   - retrieve 3 documents (appended to your file)
  ANSWER:           - finish; put each answer on its own following line as  Qn: <answer>  for Q1..Q8
Think briefly before the action line. Never fabricate document contents; only use what is in the file.

Example turn:
I need the prize money fact first.
SEARCH: Dr. Lee award prize money 2019

Example final turn:
I have all facts.
ANSWER:
Q1: Turing Award
Q2: San Francisco
Q3: Dr. Rao
Q4: one million dollars
Q5: Helios
Q6: sparse attention
Q7: Meridian Labs
Q8: Boston"""

SYSTEM_CLM = SYSTEM_BASE + """
  BASH: <command>   - run a bash command. YOUR CONTEXT IS THE FILE: edit it freely to manage the 6000-token budget - delete used-up tool output, compact verbose docs to one-liners, keep a FINDINGS section with facts you will need. Examples:
  BASH: sed -i '/^## Search results for .old./,/^$/d' FILE
  BASH: python3 -c "p='FILE'; t=open(p).read(); open(p,'w').write('condensed...')"
The runtime tells you the current token count in the file's first line. Stay under budget; drop or compact tool output once you have extracted its facts."""

SYSTEM_BASELINE = SYSTEM_BASE + """
You have no file-editing tools. When the file exceeds the 6000-token budget, the runtime automatically drops the oldest tool-output lines (your questions are pinned)."""


def _initial_file(arm, path):
    lines = []
    lines.append("[runtime] context ~0 tokens (budget %d). Manage the file to stay under budget." % BUDGET_TOKENS)
    lines.append("# Research task")
    lines.append("Answer all 8 questions. Use SEARCH to retrieve documents.")
    if arm == "clm":
        lines.append("Your context IS this file (%s) - edit it with BASH whenever useful." % path)
    else:
        lines.append("Oldest tool outputs are auto-dropped over budget (questions stay pinned).")
    lines.append("## Questions")
    for qid, q, _ in QUESTIONS:
        lines.append("%s: %s" % (qid, q))
    lines.append("## ---- live context ----")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _refresh_runtime_line(path):
    with open(path) as f:
        lines = f.read().splitlines()
    toks = estimate_tokens("\n".join(lines))
    lines[0] = ("[runtime] context ~%d tokens (budget %d). Manage the file "
                "to stay under budget." % (toks, BUDGET_TOKENS))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return toks


def _truncate_to_budget(path):
    """Baseline fixed policy: drop oldest tool-output lines past the marker."""
    with open(path) as f:
        lines = f.read().splitlines()
    try:
        m = lines.index("## ---- live context ----")
    except ValueError:
        m = 0
    while estimate_tokens("\n".join(lines)) > BUDGET_TOKENS and len(lines) > m + 1:
        del lines[m + 1]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _parse_actions(resp, arm):
    """All action lines in order. Lenient: strips bullets/bold/numbering."""
    actions = []
    for line in resp.strip().splitlines():
        s = line.strip()
        s = re.sub(r"^[\s>*\-•\d.]+", "", s)
        s = s.replace("**", "").strip()
        up = s.upper()
        if up.startswith("SEARCH:"):
            actions.append(("search", s[len("SEARCH:"):].strip()))
        elif arm == "clm" and up.startswith("BASH:"):
            actions.append(("bash", s[len("BASH:"):].strip()))
        elif up.startswith("ANSWER:") or up.startswith("FINAL:"):
            actions.append(("answer", resp))
    return actions


def _parse_action(resp, arm):
    """Backward-compat: last action wins."""
    acts = _parse_actions(resp, arm)
    return acts[-1] if acts else (None, None)


def _parse_answers(resp):
    answers = {}
    for line in resp.splitlines():
        m = re.match(r"\s*(Q[1-8])\s*:\s*(.*)", line)
        if m:
            answers[m.group(1)] = m.group(2).strip()
    return answers


def _norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(the|a|an) ", "", s)
    return s


def grade(answers):
    """Returns (correct, total, detail)."""
    detail, correct = {}, 0
    for qid, _, gold in QUESTIONS:
        got = answers.get(qid, "")
        ng, na = _norm(gold), _norm(got)
        ok = bool(ng) and bool(na) and (ng in na or na in ng)
        detail[qid] = {"gold": gold, "got": got, "ok": bool(ok)}
        correct += ok
    return correct, len(QUESTIONS), detail


TURN_REMINDER = """
---
End THIS turn with exactly one action line: SEARCH: <query> | BASH: <command> (CLM arm only) | ANSWER:
If ANSWER:, follow it with 8 lines of the form  Qn: <answer>  for Q1..Q8. Nothing else after the action line."""


def run_arm(chat_fn, arm, docstore=None, max_turns=MAX_TURNS,
            workdir="/tmp", verbose=False):
    """Run one arm. chat_fn(system, user_content) -> str."""
    assert arm in ("clm", "baseline")
    docstore = docstore or DocStore()
    path = os.path.join(workdir, "clm_ctx_%s.md" % arm)
    _initial_file(arm, path)
    system = SYSTEM_CLM if arm == "clm" else SYSTEM_BASELINE
    traj, n_search, n_bash, nudges = [], 0, 0, 0
    answers = {}
    for turn in range(max_turns):
        with open(path) as f:
            content = f.read()
        toks = estimate_tokens(content)
        traj.append(toks)
        try:
            resp = chat_fn(system, content + TURN_REMINDER)
        except Exception as e:
            traj.append("chat_error: %s" % e)
            break
        actions = _parse_actions(resp, arm)
        if verbose:
            print("[%s t%d] %s (tok %d)" % (
                arm, turn,
                "+".join(k for k, _ in actions) or "none", toks))
        if not actions:
            nudges += 1
            with open(path, "a") as f:
                f.write("[runtime] no valid action parsed; end your next "
                        "turn with SEARCH: / BASH: / ANSWER:.\n")
            _refresh_runtime_line(path)
            if nudges >= 3:
                break
            continue
        done = False
        for kind, payload in actions:
            if kind == "search":
                n_search += 1
                docs = docstore.search(payload or "award")
                block = ("\n## Search results for '%s'\n" % payload +
                         "\n\n".join("[DOC %s]\n%s" % (d["id"], d["text"])
                                     for d in docs))
                with open(path, "a") as f:
                    f.write(block + "\n")
                if arm == "baseline":
                    _truncate_to_budget(path)
                _refresh_runtime_line(path)
            elif kind == "bash":
                n_bash += 1
                try:
                    r = subprocess.run(payload, shell=True, cwd="/tmp",
                                       capture_output=True, text=True,
                                       timeout=60)
                    receipt = "[runtime] bash exit=%d" % r.returncode
                except Exception as e:
                    receipt = "[runtime] bash failed: %s" % e
                with open(path, "a") as f:
                    f.write(receipt + "\n")
                # safety cap only; the reported budget is the model's own job
                with open(path) as f:
                    clines = f.read().splitlines()
                cap = BUDGET_TOKENS * SAFETY_MULT * CHARS_PER_TOKEN
                while len("\n".join(clines)) > cap and len(clines) > 2:
                    del clines[1]
                with open(path, "w") as f:
                    f.write("\n".join(clines) + "\n")
                _refresh_runtime_line(path)
            elif kind == "answer":
                answers = _parse_answers(payload)
                done = True
                break
        if done:
            break
    correct, total, detail = grade(answers)
    with open(path) as f:
        final_text = f.read()
    return {
        "arm": arm,
        "accuracy": "%d/%d" % (correct, total),
        "correct": correct,
        "total": total,
        "detail": detail,
        "turns": len(traj),
        "searches": n_search,
        "bash_edits": n_bash,
        "nudges": nudges,
        "token_traj": [t for t in traj if isinstance(t, int)],
        "peak_tokens": max([t for t in traj if isinstance(t, int)] or [0]),
        "final_tokens": estimate_tokens(final_text),
        "within_budget_frac": (
            sum(1 for t in traj if isinstance(t, int) and t <= BUDGET_TOKENS)
            / max(1, len([t for t in traj if isinstance(t, int)]))),
        "answered": bool(answers),
    }
