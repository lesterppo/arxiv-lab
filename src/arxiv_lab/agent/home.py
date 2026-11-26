"""HoMe: Toward Practical Harness Optimization of Memory Systems.

arXiv:2611.34624 "HoMe: Toward Practical Harness Optimization of Memory
Systems for AI Assistants" (Agentscope-AI / ReMe).

Core claim implemented here: a memory system's behaviour is governed by
its harness (code, prompts, configs for memory construction,
maintenance, retrieval, answer generation), and the harness can be
optimized by an evaluation-driven loop rather than by hand. HoMe
iteratively evaluates candidate harnesses, derives each new candidate
from any previously evaluated harness by editing any part of it, and
uses session-level memory versioning plus execution traces to identify
shared failure mechanisms across cases and localize them to the
responsible session and harness component. Early structural exploration
is encouraged via candidate gating; evaluation cost is cut via smoke
tests, adaptive sampling, and reuse of constructed memory. On ReMe /
LongMemEval-S and BEAM, with six formal candidate evaluations per
benchmark after a full initial evaluation, HoMe found harnesses beating
the originally strong harness while using far fewer query-time tokens.

Bounded CPU reproduction (no model, no network; deterministic given a
seed):

* World: scripted multi-session histories. Each case is one entity /
  attribute query over 4 sessions of planted facts, revisions (a later
  session supersedes an earlier value) and distractors. 75% of queries
  ask for the CURRENT value, 25% ask for the value AS OF an earlier
  session (historical queries).
* Harness: top_k, recency weight, revision policy (latest-only vs
  all-versions), fuzzy attribute matching (synonym queries), answer
  rule (top1 vs majority). Query-time tokens = 20 + 12*top_k (+10 if
  fuzzy).
* Traces: every evaluation records, per case, the retrieved records
  (with session + superseded flag) and classifies failures as STALE
  (answered a superseded value), MISSING (nothing retrieved) or OK.
  Session-level versioning makes the failing session visible in the
  trace; the dominant shared cause localizes to a component
  (maintenance for STALE, retrieval/construction for MISSING).
* Loop: full initial evaluation, then up to 6 formal candidate
  evaluations. Candidates come from localized edits for the dominant
  shared failure; each is smoke-tested on 8 cases first (candidate
  gating) and only formally evaluated if the smoke test does not
  regress. Constructed memory is built once and reused across
  candidates (memory-reuse saving is counted).
* Oracle: exhaustive enumeration of the 256-harness space gives the
  independently computed optimum the loop is measured against.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

__all__ = [
    "Case",
    "Fact",
    "Harness",
    "EvalResult",
    "Trace",
    "make_world",
    "build_memory",
    "evaluate",
    "failure_breakdown",
    "dominant_cause",
    "localized_candidates",
    "random_edit",
    "home_loop",
    "oracle_best",
    "smoke_indices",
    "ALL_HARNESSES",
    "N_FORMAL",
    "SMOKE_N",
]

# ---------------------------------------------------------------- world
ATTRIBUTES = ("office", "phone", "project", "manager", "city")
SYNONYM = {"office": "workplace", "phone": "telephone", "project": "assignment",
           "manager": "supervisor", "city": "town"}
ENTITIES = ("ana", "ben", "cy", "dan", "eli", "fay", "gus", "hal")
VALUES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
          "golf", "hotel", "india", "juliet", "kilo", "lima")

N_SESSIONS = 4
TOK_PROMPT = 20
TOK_PER_FACT = 12
TOK_FUZZY = 10
N_FORMAL = 6
SMOKE_N = 8


@dataclass(frozen=True)
class Fact:
    entity: str
    attribute: str
    value: str
    session: int


@dataclass(frozen=True)
class Case:
    """One query over a multi-session history.

    historical=True asks for the value as of `target_session`;
    otherwise the current (latest) value is the gold answer.
    """
    entity: str
    attribute: str
    query_attribute: str  # may be a synonym of `attribute`
    historical: bool
    target_session: int
    gold: str
    history: tuple  # tuple[Fact, ...]


@dataclass(frozen=True)
class Harness:
    top_k: int = 3
    recency: float = 0.0
    revision: str = "all"       # "all" | "latest"
    fuzzy: bool = False
    answer: str = "majority"    # "top1" | "majority"


ALL_HARNESSES = [
    Harness(top_k=k, recency=r, revision=v, fuzzy=f, answer=a)
    for k in (1, 2, 3, 5)
    for r in (0.0, 0.25, 0.5, 1.0)
    for v in ("all", "latest")
    for f in (False, True)
    for a in ("top1", "majority")
]


def make_world(seed: int, n_cases: int = 48) -> list:
    """Deterministic multi-session histories with revisions, synonyms
    and distractors. Every case's gold is the value the harness should
    return; history is shared across cases only within a case."""
    rng = random.Random(seed)
    cases = []
    for _ in range(n_cases):
        entity = rng.choice(ENTITIES)
        attr = rng.choice(ATTRIBUTES)
        n_revisions = rng.choice((1, 1, 2))  # mostly one revision
        sessions = sorted(rng.sample(range(N_SESSIONS), n_revisions + 1))
        vals = rng.sample(VALUES, n_revisions + 1)
        facts = []
        for s, v in zip(sessions, vals):
            facts.append(Fact(entity, attr, v, s))
        # distractors: same attribute, other entities (same sessions)
        for _ in range(rng.randint(1, 3)):
            other = rng.choice([e for e in ENTITIES if e != entity])
            facts.append(Fact(other, attr, rng.choice(VALUES),
                              rng.choice(range(N_SESSIONS))))
        # distractor: same entity, other attribute
        other_attr = rng.choice([a for a in ATTRIBUTES if a != attr])
        facts.append(Fact(entity, other_attr, rng.choice(VALUES),
                          rng.choice(range(N_SESSIONS))))
        historical = rng.random() < 0.25
        if historical:
            target = sessions[rng.randrange(len(sessions))]
            gold = vals[sessions.index(target)]
        else:
            target = N_SESSIONS - 1
            gold = vals[-1]
        query_attr = SYNONYM[attr] if rng.random() < 0.3 else attr
        cases.append(Case(entity, attr, query_attr, historical, target,
                          gold, tuple(facts)))
    return cases


# ---------------------------------------------------------------- memory
def build_memory(cases: list, revision: str = "all") -> dict:
    """Session-level versioned memory, built once per revision policy
    and reused across candidates sharing it (the paper's memory
    reuse). Keyed by case index; each entry is the case's facts sorted
    by (session, insertion order).

    The revision policy applies at CONSTRUCTION time: "latest" drops
    superseded versions from the store entirely, so historical queries
    (which need old versions) can no longer be answered -- the genuine
    tradeoff against the STALE failures "latest" fixes."""
    mem = {}
    for i, c in enumerate(cases):
        facts = sorted(c.history, key=lambda f: (f.session, f.value))
        if revision == "latest":
            latest = {}
            for f in facts:
                key = (f.entity, f.attribute)
                if key not in latest or f.session >= latest[key].session:
                    latest[key] = f
            facts = sorted(latest.values(),
                           key=lambda f: (f.session, f.value))
        mem[i] = tuple(facts)
    return mem


@dataclass
class Trace:
    case_index: int
    retrieved: tuple  # tuple[Fact, ...] in rank order
    answer: str | None
    correct: bool
    cause: str  # "OK" | "STALE" | "MISSING"


@dataclass
class EvalResult:
    accuracy: float
    mean_tokens: float
    traces: tuple
    n: int
    constructions: int = 0


def smoke_indices(cases: list, n: int = SMOKE_N) -> list:
    """Stratified smoke-test indices (the paper's adaptive sampling):
    round-robin over the four (historical x synonym) buckets, so the
    smoke set always covers the failure modes a candidate could break
    (a plain first-8 slice can miss historical queries entirely and
    let a history-breaking candidate through the gate). Deterministic.
    """
    buckets: dict = {}
    for i, c in enumerate(cases):
        key = (1 if c.historical else 0) * 2 + (1 if c.query_attribute != c.attribute else 0)
        buckets.setdefault(key, []).append(i)
    out: list = []
    order = [0, 2, 1, 3]  # current-plain, historical-plain, current-syn, historical-syn
    while len(out) < n:
        progressed = False
        for b in order:
            if buckets.get(b) and len(out) < n:
                out.append(buckets[b].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(out)


def _score(fact: Fact, case: Case, harness: Harness) -> float | None:
    """Retrieval score, or None if the record is not retrievable.
    The query names `query_attribute`; a record matches by attribute
    exactly, or through the synonym map when fuzzy is on. Records with
    no attribute match are not retrievable (so a synonym query with
    fuzzy off yields MISSING, not a wrong-attribute answer). Same-
    entity records score higher than same-attribute other-entity
    distractors; recency breaks ties toward the latest session, and
    with recency 0 ties fall back to insertion order (oldest first,
    so a stale version wins)."""
    ent = 2.0 if fact.entity == case.entity else 0.0
    if fact.attribute == case.query_attribute:
        att = 2.0
    elif harness.fuzzy and SYNONYM.get(fact.attribute) == case.query_attribute:
        att = 1.2
    else:
        return None
    return ent + att + harness.recency * (fact.session / (N_SESSIONS - 1))


def _retrieve(memory: tuple, case: Case, harness: Harness) -> list:
    """Records for this case, honouring the historical session bound,
    ranked by score (ties: recency, then insertion order)."""
    pool = list(memory)
    if case.historical:
        pool = [f for f in pool if f.session <= case.target_session]
    scored = []
    for pos, f in enumerate(pool):
        s = _score(f, case, harness)
        if s is not None:
            scored.append((-s, -f.session if harness.recency > 0 else pos, pos, f))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in scored[:harness.top_k]]


def _answer(retrieved: list, harness: Harness) -> str | None:
    if not retrieved:
        return None
    if harness.answer == "top1":
        return retrieved[0].value
    counts: dict = {}
    order: list = []
    for f in retrieved:
        if f.value not in counts:
            counts[f.value] = 0
            order.append(f.value)
        counts[f.value] += 1
    best = max(order, key=lambda v: (counts[v], -order.index(v)))
    return best


def _classify(case: Case, retrieved: list, answer: str | None) -> str:
    if answer is None:
        return "MISSING"
    if answer == case.gold:
        return "OK"
    # Superseded value of the right key?
    for f in case.history:
        if (f.entity == case.entity and f.attribute == case.attribute
                and f.value == answer and f.value != case.gold):
            return "STALE"
    return "STALE"  # any wrong value here is a stale/distractor value


def evaluate(harness: Harness, cases: list, memory: dict | None = None,
             constructions: int = 0) -> EvalResult:
    """Evaluate one harness on all cases. Memory is built per revision
    policy and passed in (reuse); `constructions` counts builds
    charged to this evaluation (0 when reused)."""
    if memory is None:
        memory = build_memory(cases, harness.revision)
        constructions = 1
    traces = []
    correct = 0
    for i, case in enumerate(cases):
        retrieved = _retrieve(memory[i], case, harness)
        answer = _answer(retrieved, harness)
        ok = answer == case.gold
        correct += 1 if ok else 0
        traces.append(Trace(i, tuple(retrieved), answer, ok,
                            _classify(case, retrieved, answer)))
    tokens = TOK_PROMPT + harness.top_k * TOK_PER_FACT
    if harness.fuzzy:
        tokens += TOK_FUZZY
    return EvalResult(correct / len(cases), float(tokens), tuple(traces),
                      len(cases), constructions)


# ------------------------------------------------------- failure analysis
def failure_breakdown(result: EvalResult) -> dict:
    counts: dict = {}
    for t in result.traces:
        if t.cause != "OK":
            counts[t.cause] = counts.get(t.cause, 0) + 1
    return counts


def dominant_cause(result: EvalResult) -> str | None:
    counts = failure_breakdown(result)
    if not counts:
        return None
    # deterministic: highest count, ties broken alphabetically
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def localized_candidates(harness: Harness, cause: str | None) -> list:
    """Prioritized candidate edits for the dominant shared failure,
    each targeting the localized component. Order matters: the first
    candidate is the component's primary fix."""
    if cause == "STALE":
        cands = []
        if harness.recency < 1.0:
            cands.append(Harness(harness.top_k, 1.0, harness.revision,
                                 harness.fuzzy, harness.answer))
        if harness.answer != "top1":
            cands.append(Harness(harness.top_k, harness.recency,
                                 harness.revision, harness.fuzzy, "top1"))
        if harness.top_k != 1:
            cands.append(Harness(1, harness.recency, harness.revision,
                                 harness.fuzzy, harness.answer))
        if harness.revision != "latest":
            # Risky fix, tried last: kills historical queries; the
            # loop's formal evaluation will reject it if it nets out
            # worse (and the smoke gate should reject it first).
            cands.append(Harness(harness.top_k, harness.recency, "latest",
                                 harness.fuzzy, harness.answer))
        return cands
    if cause == "MISSING":
        cands = []
        if not harness.fuzzy:
            cands.append(Harness(harness.top_k, harness.recency,
                                 harness.revision, True, harness.answer))
        if harness.top_k < 5:
            cands.append(Harness(5, harness.recency, harness.revision,
                                 harness.fuzzy, harness.answer))
        return cands
    return []


def random_edit(harness: Harness, rng: random.Random) -> Harness:
    """One random single-parameter edit (the unrestricted-editing
    baseline the paper contrasts against)."""
    param = rng.choice(("top_k", "recency", "revision", "fuzzy", "answer"))
    if param == "top_k":
        return Harness(rng.choice([k for k in (1, 2, 3, 5)
                                   if k != harness.top_k]),
                       harness.recency, harness.revision, harness.fuzzy,
                       harness.answer)
    if param == "recency":
        return Harness(harness.top_k,
                       rng.choice([r for r in (0.0, 0.25, 0.5, 1.0)
                                   if r != harness.recency]),
                       harness.revision, harness.fuzzy, harness.answer)
    if param == "revision":
        return Harness(harness.top_k, harness.recency,
                       "latest" if harness.revision == "all" else "all",
                       harness.fuzzy, harness.answer)
    if param == "fuzzy":
        return Harness(harness.top_k, harness.recency, harness.revision,
                       not harness.fuzzy, harness.answer)
    return Harness(harness.top_k, harness.recency, harness.revision,
                   harness.fuzzy,
                   "top1" if harness.answer == "majority" else "majority")


# ------------------------------------------------------------------- loop
@dataclass
class LoopResult:
    best: Harness
    best_result: EvalResult
    start_result: EvalResult
    formal_evals: int
    smoke_evals: int
    smoke_rejects: int
    candidates_tried: int
    constructions: int
    history: list = field(default_factory=list)


def _better(a: EvalResult, b: EvalResult) -> bool:
    """Higher accuracy wins; ties break to fewer tokens."""
    if a.accuracy != b.accuracy:
        return a.accuracy > b.accuracy
    return a.mean_tokens < b.mean_tokens


def home_loop(cases: list, start: Harness, seed: int = 0,
              n_formal: int = N_FORMAL, gating: bool = True,
              localized: bool = True) -> LoopResult:
    """HoMe loop: full initial evaluation, then up to `n_formal`
    formal candidate evaluations. Memory is built once per revision
    policy and reused across candidates sharing it. With gating, each
    candidate is smoke-tested on SMOKE_N cases first; a smoke
    regression rejects it without a formal evaluation. With
    localized=False the candidates are random edits (baseline)."""
    rng = random.Random(seed)
    mem_cache: dict = {}
    constructions = 0

    def mem_for(h: Harness, idxs=None):
        nonlocal constructions
        if h.revision not in mem_cache:
            mem_cache[h.revision] = build_memory(cases, h.revision)
            constructions += 1
        full = mem_cache[h.revision]
        if idxs is None:
            return full
        # idxs are full-case indices; re-key by position in the subset
        # so evaluate() can index by enumeration position.
        return {j: full[i] for j, i in enumerate(idxs)}

    start_result = evaluate(start, cases, mem_for(start))
    best, best_result = start, start_result
    formal = smoke = rejects = tried = 0
    history = [(start, start_result.accuracy, start_result.mean_tokens)]
    smoke_idxs = smoke_indices(cases)
    smoke_cases = [cases[i] for i in smoke_idxs]
    tried_harnesses = {start}
    while formal < n_formal:
        cause = dominant_cause(best_result)
        if cause is None:
            break  # nothing left to fix
        cands = (localized_candidates(best, cause) if localized
                 else [random_edit(best, rng)])
        progressed = False
        for cand in cands:
            if formal >= n_formal or cand in tried_harnesses:
                continue
            tried_harnesses.add(cand)
            tried += 1
            if gating:
                smoke += 1
                s_cand = evaluate(cand, smoke_cases, mem_for(cand, smoke_idxs))
                s_best = evaluate(best, smoke_cases, mem_for(best, smoke_idxs))
                if s_cand.accuracy < s_best.accuracy - 1e-9:
                    rejects += 1
                    continue
            formal += 1
            res = evaluate(cand, cases, mem_for(cand))
            if _better(res, best_result):
                best, best_result = cand, res
                history.append((cand, res.accuracy, res.mean_tokens))
                progressed = True
                break  # re-attribute failures from the new best
        if not progressed:
            break  # no candidate in this round improved anything
    # Cost phase: spend remaining formal evaluations on cheaper
    # harnesses that preserve accuracy -- the paper's "far fewer
    # query-time tokens". Each candidate is smoke-gated and counts as
    # a formal evaluation; adoption requires no accuracy loss.
    for k in (1, best.top_k - 1):
            if formal >= n_formal or k < 1 or k >= best.top_k:
                continue
            cand = Harness(k, best.recency, best.revision, best.fuzzy,
                           best.answer)
            if cand in tried_harnesses:
                continue
            tried_harnesses.add(cand)
            tried += 1
            if gating:
                smoke += 1
                s_cand = evaluate(cand, smoke_cases, mem_for(cand, smoke_idxs))
                s_best = evaluate(best, smoke_cases, mem_for(best, smoke_idxs))
                if s_cand.accuracy < s_best.accuracy - 1e-9:
                    rejects += 1
                    continue
            formal += 1
            res = evaluate(cand, cases, mem_for(cand))
            if _better(res, best_result):
                best, best_result = cand, res
                history.append((cand, res.accuracy, res.mean_tokens))
                break  # cheapest working k found; nothing cheaper to try
    return LoopResult(best, best_result, start_result, formal, smoke,
                      rejects, tried, constructions, history)


def oracle_best(cases: list) -> tuple:
    """Exhaustive optimum over the harness space: (best harness,
    best EvalResult). Accuracy first, then fewest tokens."""
    mem_cache: dict = {}
    best_h, best_r = None, None
    for h in ALL_HARNESSES:
        if h.revision not in mem_cache:
            mem_cache[h.revision] = build_memory(cases, h.revision)
        r = evaluate(h, cases, mem_cache[h.revision])
        if best_r is None or _better(r, best_r):
            best_h, best_r = h, r
    return best_h, best_r
