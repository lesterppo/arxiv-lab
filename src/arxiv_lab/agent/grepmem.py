"""GrepMem: lossless, index-free memory for long-horizon LLM agents.

arXiv:2611.25123 (Yang, Shi, Zhao, 2026) — "Forget Freely, Grep Later:
Lossless, Index-Free Memory for Long-Horizon LLM Agents".

Claim implemented: an append-only plain-text log of the agent's complete
history plus a regular-expression search tool for on-demand recall is a
lossless memory mechanism that stays competitive with keeping the full
history, while lossy strategies (summarization, masking, recency
truncation) discard details later questions need. The paper trains
Qwen3.5-9B under five context-management strategies and reports 95-97%
accuracy on ContextBench for GrepMem vs at most 21% for lossy baselines,
and +8.8pp over SUPO summarization on BCP-Multi.

This module implements the mechanism (log + regex grep + memory policies)
and a deterministic CPU evaluation harness that isolates the memory policy
from model capability: a perfect keyword QA oracle answers questions from
whatever context lines a policy returns, so accuracy differences between
policies measure information preservation only — exactly the paper's claim.

Stdlib only.
"""

from __future__ import annotations

import random
import re
import statistics

# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

FACT_VERBS = ("is", "is in", "is now", "runs on", "uses", "costs")


class AppendOnlyLog:
    """Append-only plain-text log of an agent's complete history.

    Lines are timestamped turns: observations (tool results / facts),
    agent actions, and distractor chatter. Nothing is ever deleted or
    rewritten — the paper's "forget freely" half: the agent may drop
    lines from its working context, but the log itself is lossless.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line)

    def __len__(self) -> int:
        return len(self.lines)

    def grep(self, pattern: str, max_lines: int | None = None) -> list[str]:
        """Regular-expression search over the log (case-insensitive).

        Returns matching lines in log order. ``max_lines`` caps the
        returned lines (most recent matches win when capped).
        """
        rx = re.compile(pattern, re.IGNORECASE)
        hits = [ln for ln in self.lines if rx.search(ln)]
        if max_lines is not None and len(hits) > max_lines:
            hits = hits[-max_lines:]
        return hits


# ---------------------------------------------------------------------------
# Synthetic research sessions (deterministic task generator)
# ---------------------------------------------------------------------------

ENTITIES = [
    ("Project Atlas", "project"),
    ("Project Beacon", "project"),
    ("Dr. Chen", "person"),
    ("Dr. Okafor", "person"),
    ("Lab 7", "place"),
    ("Server Node 3", "device"),
    ("Experiment 12", "experiment"),
    ("Experiment 27", "experiment"),
    ("Dataset Vega", "dataset"),
    ("Model Orion", "model"),
]

ATTRIBUTES = [
    ("budget", "is", ["$1.2M", "$2.4M", "$3.1M", "$0.8M", "$4.7M"]),
    ("location", "is in", ["Building 7", "Building 2", "Room 14", "Annex B", "Floor 3"]),
    ("port", "runs on", ["port 8080", "port 443", "port 9000", "port 5432"]),
    ("temperature", "is", ["37.5C", "21.0C", "42.3C", "18.9C"]),
    ("status", "is", ["active", "paused", "complete", "pending review"]),
    ("lead", "is led by", ["Dr. Chen", "Dr. Okafor", "Dr. Patel", "Dr. Kim"]),
]

# Paraphrase map: attribute -> synonym used in the question text only.
# The log still uses the canonical word, so a keyword grep on the
# paraphrase misses unless the policy also greps the entity token.
PARAPHRASE = {
    "budget": "funding",
    "location": "whereabouts",
    "port": "endpoint",
    "temperature": "heat",
    "status": "state",
    "lead": "supervisor",
}

STOPWORDS = {
    "what", "is", "the", "of", "does", "have", "has", "for", "in", "on",
    "how", "much", "where", "who", "which", "a", "an", "to", "do",
}


def _fact_line(step: int, entity: str, attr: str, verb: str, value: str,
               kind: str = "OBS") -> str:
    return f"[t{step:04d}] {kind} | {entity} {attr} {verb} {value}"


def _distractor_line(rng: random.Random, step: int) -> str:
    topics = [
        "team standup moved to 10:00",
        "coffee machine needs descaling",
        "weekly report draft saved",
        "calendar invite accepted for Friday",
        "VPN connection re-established",
        "notebook kernel restarted",
        "package update available for numpy",
        "meeting notes filed under Q3",
        "disk usage at 62 percent",
        "backup completed successfully",
    ]
    return f"[t{step:04d}] CHAT | {rng.choice(topics)}"


def generate_session(seed: int, n_lines: int = 500, n_questions: int = 20,
                     update_rate: float = 0.3,
                     paraphrase_rate: float = 0.25) -> dict:
    """Generate one synthetic long-horizon research session.

    Returns {"log": AppendOnlyLog, "questions": [...]} where each question
    is {"text", "entity", "attr", "answer", "position", "paraphrased"}.
    Facts are placed uniformly through the stream; a fraction are later
    updated (latest value wins); a fraction of questions paraphrase the
    attribute word. Distractor lines fill the rest.
    """
    rng = random.Random(seed)
    log = AppendOnlyLog()
    # Choose which (entity, attr) pairs carry facts, and their values.
    pairs = [(e, a) for e, _ in ENTITIES for a in ATTRIBUTES
             if rng.random() < 0.55]
    rng.shuffle(pairs)
    truth: dict[tuple[str, str], str] = {}
    fact_steps: list[tuple[int, str, str]] = []  # (step, entity, attr)
    n_facts = max(n_questions, n_lines // 12)
    fact_positions = sorted(rng.sample(range(n_lines), n_facts))
    fi = 0
    for step in range(n_lines):
        if fi < n_facts and step == fact_positions[fi]:
            fi += 1
            entity, (attr, verb, values) = pairs[rng.randrange(len(pairs))]
            if (entity, attr) in truth and rng.random() > update_rate:
                # Restatement of the current value (no update).
                value = truth[(entity, attr)]
            else:
                value = rng.choice(values)
            truth[(entity, attr)] = value
            log.append(_fact_line(step, entity, attr, verb, value,
                                  kind=rng.choice(["OBS tool:web_search",
                                                   "OBS tool:db_query",
                                                   "OBS tool:api_call"])))
            fact_steps.append((step, entity, attr))
        elif rng.random() < 0.45:
            # Agent action line (non-factual trajectory content).
            log.append(f"[t{step:04d}] ACT agent: "
                       f"{rng.choice(['draft summary', 'run analysis', 'open file', 'call tool', 'review output'])}")
        else:
            log.append(_distractor_line(rng, step))
    # Questions: sample from facts actually in the log.
    questions = []
    rng.shuffle(fact_steps)
    for step, entity, attr in fact_steps[:n_questions]:
        paraphrased = rng.random() < paraphrase_rate
        attr_word = PARAPHRASE[attr] if paraphrased else attr
        if attr == "budget":
            text = (f"How much {attr_word} does {entity} have?"
                    if paraphrased else
                    f"What is the {attr} of {entity}?")
        elif attr == "location":
            text = (f"What are the {attr_word} of {entity}?"
                    if paraphrased else
                    f"What is the {attr} of {entity}?")
        else:
            text = f"What is the {attr_word} of {entity}?"
        questions.append({
            "text": text,
            "entity": entity,
            "attr": attr,
            "answer": truth[(entity, attr)],
            "position": step / max(n_lines - 1, 1),
            "paraphrased": paraphrased,
        })
    return {"log": log, "questions": questions, "truth": truth}


# ---------------------------------------------------------------------------
# Memory policies
# ---------------------------------------------------------------------------

def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def policy_full(log: AppendOnlyLog, question: dict, **kw) -> list[str]:
    """Ceiling: the whole history stays in context."""
    return list(log.lines)


def policy_grepmem(log: AppendOnlyLog, question: dict,
                   max_lines: int = 40, entity_only: bool = False,
                   attr_only: bool = False) -> list[str]:
    """GrepMem: regex search over the append-only log, on demand.

    The agent greps the log for the question's keywords: one pattern per
    keyword group (entity tokens, attribute token). Matches are unioned
    in log order and capped at ``max_lines``. No embeddings, no index,
    no extraction — the log itself is the memory. ``attr_only`` restricts
    patterns to the attribute token (vocabulary-mismatch stress test).
    """
    kws = _keywords(question["text"])
    entity_tokens = [w for w in _keywords(question["entity"])]
    if attr_only:
        attr_tokens = [w for w in kws if w not in entity_tokens]
        patterns = [r"\b" + re.escape(t) + r"\b" for t in attr_tokens]
    elif entity_only:
        patterns = [r"\b" + re.escape(t) + r"\b" for t in entity_tokens]
    else:
        patterns = [r"\b" + re.escape(t) + r"\b" for t in entity_tokens]
        attr_tokens = [w for w in kws if w not in entity_tokens]
        patterns += [r"\b" + re.escape(t) + r"\b" for t in attr_tokens]
    hits: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for ln in log.grep(pat):
            if ln not in seen:
                seen.add(ln)
                hits.append(ln)
    hits.sort(key=log.lines.index)
    if len(hits) > max_lines:
        hits = hits[-max_lines:]
    return hits


def policy_summary(log: AppendOnlyLog, question: dict, block: int = 25,
                   retain: float = 0.15, seed: int = 0) -> list[str]:
    """Lossy summarization baseline (SUPO-style).

    Each block of ``block`` lines is replaced by its first line plus a
    seeded ``retain`` fraction of its fact lines — a summarizer keeps the
    gist and some details, but most detail facts are gone. The most
    recent block is kept whole (recency of the summary itself).
    """
    rng = random.Random(seed)
    out: list[str] = []
    lines = log.lines
    for start in range(0, len(lines), block):
        chunk = lines[start:start + block]
        if start + block >= len(lines):
            out.extend(chunk)
            continue
        out.append(chunk[0])
        facts = [ln for ln in chunk[1:] if " | " in ln and "OBS" in ln]
        keep = int(len(facts) * retain)
        out.extend(rng.sample(facts, keep) if keep else [])
    return out


def policy_recency(log: AppendOnlyLog, question: dict, window: int = 60) -> list[str]:
    """Truncation / masking baseline: only the last ``window`` lines."""
    return list(log.lines[-window:])


def policy_random_mask(log: AppendOnlyLog, question: dict, keep: float = 0.20,
                       seed: int = 0) -> list[str]:
    """Random-masking baseline: a seeded ``keep`` fraction of lines."""
    rng = random.Random(seed)
    return [ln for ln in log.lines if rng.random() < keep]


POLICIES = {
    "full": policy_full,
    "grepmem": policy_grepmem,
    "summary": policy_summary,
    "recency": policy_recency,
    "random_mask": policy_random_mask,
}


# ---------------------------------------------------------------------------
# Deterministic QA oracle (perfect given the facts in context)
# ---------------------------------------------------------------------------

def answer_question(question: dict, context: list[str]) -> str | None:
    """Answer from the supplied context lines only.

    Finds the latest context line mentioning the question's entity and
    attribute word, and extracts the value after the verb. Returns None
    when the fact is not in the context. The oracle is perfect given the
    facts — accuracy differences therefore measure the memory policy.
    """
    entity = question["entity"]
    attr = question["attr"]
    # Anchor on the fact structure "| <entity> <attr> <verb> <value>" so
    # an entity mentioned as another fact's *value* (e.g. "lead is led
    # by Dr. Chen") cannot false-match a question about that entity.
    anchor = re.compile(rf"\| {re.escape(entity)} {re.escape(attr)} "
                        r"(?:is in|is now|is led by|is|runs on|uses|costs) (.+)$")
    best: str | None = None
    for ln in context:
        m = anchor.search(ln)
        if m:
            best = m.group(1).strip()
    return best


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_policy(policy_name: str, sessions: list[dict],
                    **policy_kw) -> dict:
    """Run one policy over generated sessions; return accuracy + cost."""
    policy = POLICIES[policy_name]
    correct = 0
    total = 0
    costs: list[int] = []
    per_position = {"early": [0, 0], "mid": [0, 0], "late": [0, 0]}
    paraphrase = [0, 0]
    for sess in sessions:
        log = sess["log"]
        for q in sess["questions"]:
            ctx = policy(log, q, **policy_kw)
            costs.append(len(ctx))
            ans = answer_question(q, ctx)
            ok = ans == q["answer"]
            correct += ok
            total += 1
            pos = "early" if q["position"] < 0.33 else ("mid" if q["position"] < 0.67 else "late")
            per_position[pos][0] += ok
            per_position[pos][1] += 1
            if q["paraphrased"]:
                paraphrase[0] += ok
                paraphrase[1] += 1
    return {
        "policy": policy_name,
        "accuracy": correct / total if total else 0.0,
        "n": total,
        "mean_context_lines": statistics.fmean(costs) if costs else 0.0,
        "by_position": {k: (v[0] / v[1] if v[1] else 0.0) for k, v in per_position.items()},
        "paraphrase_accuracy": (paraphrase[0] / paraphrase[1] if paraphrase[1] else 0.0),
        "paraphrase_n": paraphrase[1],
    }


def run_suite(seeds=range(5), n_lines: int = 500, n_questions: int = 20,
              grepmem_max_lines: int = 40) -> dict:
    """Full comparison suite over generated sessions (one stream length)."""
    sessions = [generate_session(s, n_lines=n_lines, n_questions=n_questions)
                for s in seeds]
    results = {}
    for name in POLICIES:
        kw = {"max_lines": grepmem_max_lines} if name == "grepmem" else {}
        results[name] = evaluate_policy(name, sessions, **kw)
    return {"n_lines": n_lines, "n_sessions": len(sessions),
            "n_questions": sum(len(s["questions"]) for s in sessions),
            "results": results}
