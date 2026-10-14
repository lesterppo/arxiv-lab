"""CPU test for arXiv 2609.37725 — Context Language Models.

Reproduces the paper's mechanism and experimental design in a scripted,
seeded simulation (no LLM): a long-horizon tool-use trajectory where
answer-critical facts are scattered among failed / duplicate / verbose
tool outputs, under a tight context budget.

Two arms:
  * CLM arm — LiveContextFile with a scripted edit policy mirroring the
    paper's observed emergent strategies: prune failed and duplicate tool
    outputs, compact verbose outputs (preserving refs), and maintain a
    findings table of extracted facts. Edits sync to the next-turn input.
  * Baseline arm — the fixed harness policy the paper beats: append-only
    with head truncation when over budget.

The scripted FACT extractor (regex) stands in for the model's own judgment
of what is important — the honest, deterministic analog at CPU scale.

Claims under test (paper abstract + Sec. 4.1):
 1. Protocol integrity: edit batches are all-or-nothing; bad ranges and
    unknown ops raise EditError with no mutation; revision bumps on every
    mutation; sync() returns exactly the current file content.
 2. Budget accounting is deterministic (chars // 4, shared convention).
 3. Under a tight budget, the CLM arm retains all critical facts while the
    truncation baseline loses some (paper: CLM beats fixed context
    management at lower compute on long-horizon tasks).
 4. The model actually edits: the edit log shows in-place updates, and a
    findings table is maintained (paper Fig. 3: 163 in-place tracker
    updates, context held ~6-8K tokens).

Deterministic (seeded). stdlib only.
"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.context.clm import (
    LiveContextFile, EditError, estimate_tokens)

SEED = 7
BUDGET = 6000  # tokens
N_OUTPUTS = 60

FACTS = [
    "Dr. Lee won the Turing Award in 2019.",
    "The ceremony was held in San Francisco.",
    "Dr. Lee thanked collaborator Dr. Rao on stage.",
    "The prize money was one million dollars.",
    "Dr. Lee announced the Helios project at the banquet.",
    "The award recognized work on sparse attention.",
]
# output positions (0-based among tool outputs) carrying the facts — spread
# out, including early ones that head truncation will drop
FACT_POSITIONS = [2, 11, 23, 34, 44, 55]

FILLER_WORDS = ("search result batch nominal recall precision index shard "
                "query latency cache token stream buffer node edge weight "
                "gradient checkpoint epoch loss curve metric baseline delta "
                "server region replica queue worker task job run log trace "
                .split())

TASK_PROMPT = ("Research task: determine which award Dr. Lee won in 2019, "
               "where the ceremony was held, and what was announced there. "
               "Use the tool outputs below.")


def _filler(rng, nchars):
    words = []
    while len(" ".join(words)) < nchars:
        words.append(rng.choice(FILLER_WORDS))
    return " ".join(words)[:nchars]


def build_trajectory(seed=SEED):
    """Task prompt + N_OUTPUTS tool-output lines (one line each)."""
    rng = random.Random(seed)
    fact_at = dict(zip(FACT_POSITIONS, FACTS))
    dup_pool = []
    traj = [TASK_PROMPT]
    for i in range(N_OUTPUTS):
        if i in fact_at:
            body = ("tool[%d] verbose dump %s FACT: %s %s"
                    % (i, _filler(rng, 500), fact_at[i], _filler(rng, 500)))
        elif i % 5 == 0:
            body = "ERROR: search backend timeout (0 results) [tool %d]" % i
        elif i % 6 == 0 and dup_pool:
            body = rng.choice(dup_pool)  # exact duplicate
        else:
            body = "tool[%d] verbose dump %s" % (i, _filler(rng, 1100))
            if len(body) > 900:
                dup_pool.append(body)
        traj.append(body)
    return traj


FACT_RE = re.compile(r"FACT:\s*(.*)")


def _clm_tidy(ctx):
    """One scripted edit pass mirroring the paper's observed strategies.

    Returns True if anything changed.
    """
    lines = ctx.lines
    n = len(lines)
    # Phase A: prune failures and exact duplicates (keep first occurrence).
    fail = [i for i in range(1, n) if lines[i].startswith("ERROR:")]
    seen, dup = {}, []
    for i in range(1, n):
        if lines[i].startswith("[finding") or i in fail:
            continue
        if lines[i] in seen:
            dup.append(i)
        else:
            seen[lines[i]] = i
    # Which facts are already in the findings table?
    recorded = set()
    for l in lines:
        if l.startswith("[finding"):
            m = FACT_RE.search(l)
            if m:
                recorded.add(m.group(1).strip())
    new_findings = []
    for i in range(1, n):
        if (lines[i].startswith("[finding")
                or lines[i].startswith("[compacted")  # already represented
                or i in fail or i in dup):
            continue
        m = FACT_RE.search(lines[i])
        if m:
            fact = re.sub(r"\s*\(ref L\d+\)\s*$", "",
                          m.group(1).strip())
            if fact not in recorded:
                new_findings.append(fact)
                recorded.add(fact)
    if fail or dup:
        ctx.apply_batch(
            [("delete", i, i + 1)
             for i in sorted(set(fail + dup), reverse=True)])
        return True
    # Phase B: compact verbose lines (facts -> one-liners preserving the
    # fact; noise -> head snippet), then append new findings.
    lines, n = ctx.lines, len(ctx.lines)
    ops = []
    for i in range(1, n):
        l = lines[i]
        if l.startswith("[finding") or len(l) < 800:
            continue
        m = FACT_RE.search(l)
        if m:
            ops.append(("replace", i, i + 1,
                        "[compacted] FACT: %s (ref L%d)"
                        % (m.group(1).strip(), i)))
        else:
            ops.append(("replace", i, i + 1,
                        "[compacted] %s..." % l[:100]))
    for f in new_findings:
        ops.append(("append_note", "[finding] FACT: %s" % f))
    if ops:
        ctx.apply_batch(ops)
        return True
    # Last resort: compact the oldest unprotected line.
    for i in range(1, len(ctx.lines)):
        l = ctx.lines[i]
        if l.startswith("[finding") or l.startswith("[compacted"):
            continue
        ctx.apply_batch(
            [("replace", i, i + 1, "[compacted] %s..." % l[:60])])
        return True
    return False


def run_clm(traj, budget=BUDGET):
    ctx = LiveContextFile(budget)
    for out in traj:
        ctx.append(out)
        while ctx.over_budget():
            if not _clm_tidy(ctx):
                break
    return ctx


def run_baseline(traj, budget=BUDGET):
    """Fixed harness policy: append-only + head truncation over budget."""
    ctx = LiveContextFile(budget)
    for out in traj:
        ctx.append(out)
        while ctx.over_budget() and len(ctx.lines) > 1:
            ctx.apply_batch([("delete", 0, 1)])
    return ctx


def retained_facts(ctx):
    text = ctx.sync()
    return sum(1 for f in FACTS if f in text)


# -- tests ------------------------------------------------------------

def test_edit_mechanics():
    ctx = LiveContextFile(1000)
    ctx.append("line0\nline1\nline2")
    r0 = ctx.revision
    ctx.apply_batch([("delete", 1, 2)])
    assert ctx.lines == ["line0", "line2"], ctx.lines
    ctx.apply_batch([("replace", 0, 1, "new0")])
    assert ctx.lines[0] == "new0"
    ctx.apply_batch([("append_note", "[note] hello")])
    assert ctx.lines[-1] == "[note] hello"
    ctx.apply_batch([("rewrite", "a\nb")])
    assert ctx.lines == ["a", "b"]
    assert ctx.sync() == "a\nb"
    assert ctx.revision == r0 + 4
    print("mechanics ok: ops apply, revision bumps, sync reflects edits")


def test_batch_atomicity():
    ctx = LiveContextFile(1000)
    ctx.append("a\nb\nc")
    before = list(ctx.lines)
    for bad in [[("delete", 0, 9)],
                [("replace", -1, 1, "x")],
                [("frobnicate", 0, 1)],
                [("delete", 0, 1), ("delete", 5, 9)]]:
        try:
            ctx.apply_batch(bad)
        except EditError:
            pass
        else:
            raise AssertionError("expected EditError for %r" % (bad,))
    assert ctx.lines == before, "failed batch must not mutate"
    print("atomicity ok: bad batches raise EditError, no partial mutation")


def test_budget_accounting():
    assert estimate_tokens("x" * 4000) == 1000
    ctx = LiveContextFile(10)
    assert not ctx.over_budget()
    ctx.append("x" * 4000)
    assert ctx.tokens() == 1000
    assert ctx.over_budget()
    print("accounting ok: deterministic chars//4, over_budget flag")


def test_clm_retains_facts_within_budget():
    traj = build_trajectory()
    ctx = run_clm(traj)
    n = retained_facts(ctx)
    print("clm: tokens=%d/%d facts=%d/6 edits=%d" % (
        ctx.tokens(), BUDGET, n,
        sum(1 for _, op, _ in ctx.edit_log if op == "edit_batch")))
    assert ctx.tokens() <= BUDGET, "CLM arm must respect the budget"
    assert n == 6, "CLM arm must retain all 6 critical facts, got %d" % n


def test_baseline_truncation_loses_facts():
    traj = build_trajectory()
    ctx = run_baseline(traj)
    n = retained_facts(ctx)
    print("baseline: tokens=%d/%d facts=%d/6" % (ctx.tokens(), BUDGET, n))
    assert ctx.tokens() <= BUDGET
    assert n < 6, ("baseline must lose at least one early fact to "
                   "truncation, got %d/6" % n)


def test_findings_table_and_edit_log():
    traj = build_trajectory()
    ctx = run_clm(traj)
    findings = [l for l in ctx.lines if l.startswith("[finding")]
    assert len(findings) == 6, "findings table must hold 6 facts, got %d" % len(findings)
    batches = [e for e in ctx.edit_log if e[1] == "edit_batch"]
    assert batches, "the model must actually edit, not just append"
    print("findings ok: 6/6 tabled; edit batches=%d (paper Fig.3 analog: "
          "in-place tracker updates)" % len(batches))


if __name__ == "__main__":
    test_edit_mechanics()
    test_batch_atomicity()
    test_budget_accounting()
    test_clm_retains_facts_within_budget()
    test_baseline_truncation_loses_facts()
    test_findings_table_and_edit_log()
    print("ALL CLM TESTS PASSED")
