"""EMFA: Error-Propagation Modeling for Failure Attribution (arXiv:2610.11600).

One-line claim: explicitly modeling how errors propagate across dependent
steps and persist in unresolved interaction loops, then counterfactual-
verifying screened candidates, identifies the decisive (agent, step) pair
of a failed multi-agent execution more accurately than judging steps in
isolation (e.g. blaming the final step, the latest wrong step, or the
earliest wrong step).

Pipeline (paper Sec. 3, five stages):

1. ``find_final_failure`` — locate the step exposing the final symptom
   (paper: scan the last ``w`` steps for the answering step; ``w=5``).
2. ``build_dependencies`` — structured trajectory ``H = (X, S, Estep, Esub)``:
   producer-consumer step edges plus subtask-reuse edges.
3. ``annotate_propagation`` + ``extract_loops`` — the error-propagation
   graph ``Gprop`` (deviation introduced / carried / corrected) and the
   persistent-loop graph ``Gloop`` (repeated unresolved interaction rounds).
4. ``screen_candidates`` — dual-branch screening: propagation branch keeps
   the earliest *unrecovered* deviations (<=2); loop branch keeps the
   earliest step initiating/sustaining each unresolved loop (<=2).
5. ``counterfactual_verify`` — repair each candidate (in chronological
   order) and re-simulate; the earliest candidate whose repair recovers the
   execution is the predicted decisive pair (paper: "take the earliest
   decisive error").

What this module implements (the mechanism, not the LLM harness):
the paper uses an LLM for dependency extraction, propagation annotation,
loop identification, screening and verification judgments. Here those
roles are played by deterministic rules over an explicit taint-tag world
model (see ``simulate``): a deviation introduces a taint tag, carriers
propagate the union of their inputs' tags, verifiers may correct. The
test therefore evaluates the *attribution protocol* — propagation-aware
screening plus counterfactual verification — against naive baselines,
with ground-truth decisive pairs known by construction. No network,
stdlib only.
"""

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# World model: taint-tag multi-agent trajectories with known ground truth.


@dataclass(frozen=True)
class Step:
    """One executed step of a multi-agent trajectory."""
    id: int
    agent: str
    subtask: str
    kind: str            # source|prop|verify|instruct|execute|answer|stop
    consumes: tuple      # ids of prior steps whose outputs this step uses
    action_sig: str = ""  # abstract action signature (loop detection)
    ambiguous: bool = False  # instruction too vague to follow reliably


@dataclass
class Trajectory:
    """A failed multi-agent execution with known ground truth."""
    steps: list                    # Steps in execution order
    taint: dict                    # step id -> frozenset of taint tags (faulty run)
    clean_taint: dict              # step id -> frozenset (fault-free run; {} everywhere)
    introduced: dict               # step id -> frozenset of tags *introduced* here
    executed: set                  # ids actually executed (a "stop" halts the run)
    decisive: tuple                # ground-truth (agent, step id)
    expected: str = "ok"


def _out_taint(kind, input_taints, new_tag=None, corrects=False):
    """Deterministic taint propagation for one step."""
    union = frozenset().union(*input_taints) if input_taints else frozenset()
    if corrects:
        return frozenset()
    if kind == "vote":
        # redundant paths: tainted only if EVERY input is tainted
        # (the step succeeds when any one path is clean)
        base = union if input_taints and all(input_taints) else frozenset()
    else:
        base = union
    if new_tag is not None:
        return base | {new_tag}
    return base


def simulate(steps, deviations, stop_halts=True):
    """Forward-simulate a trajectory.

    ``deviations``: dict step id -> tag (fresh deviation introduced there);
    the special tag ``"premature"`` marks a stop step that halts the run
    early (its clean behavior is to continue). Steps of kind ``verify``
    listed in ``corrections`` correct incoming taint instead of carrying it.
    Returns (taint dict, executed set).
    """
    corrections = deviations.pop("_correct", set())
    taint, executed, introduced = {}, set(), {}
    for s in steps:
        inputs = [taint[i] for i in s.consumes if i in taint]
        if s.kind == "verify" and s.id in corrections:
            taint[s.id] = frozenset()
            introduced[s.id] = frozenset()
        elif s.id in deviations:
            taint[s.id] = _out_taint(s.kind, inputs, new_tag=deviations[s.id])
            introduced[s.id] = frozenset({deviations[s.id]})
        else:
            taint[s.id] = _out_taint(s.kind, inputs)
            introduced[s.id] = frozenset()
        executed.add(s.id)
        if stop_halts and s.kind == "stop" and taint[s.id]:
            break  # premature stop: downstream steps never execute
    return taint, executed, introduced


def repair_and_resimulate(traj, cand_id):
    """Counterfactual: repair step ``cand_id`` (output := clean value) and
    re-simulate downstream. Returns (taint dict, executed set)."""
    steps = traj.steps
    taint, executed = {}, set()
    for s in steps:
        if s.id == cand_id:
            taint[s.id] = frozenset()  # minimally repaired: no deviation
        else:
            inputs = [taint[i] for i in s.consumes if i in taint]
            own = traj.introduced.get(s.id, frozenset())
            new_tag = next(iter(own)) if own else None
            taint[s.id] = _out_taint(s.kind, inputs, new_tag=new_tag)
        executed.add(s.id)
        if s.kind == "stop" and taint[s.id]:
            break
    return taint, executed


def succeeds(taint, executed, steps):
    """Success criterion: every step executed and the final step untainted."""
    if len(executed) != len(steps):
        return False
    return not taint[steps[-1].id]


# --------------------------------------------------------------------------
# EMFA stages.


def find_final_failure(traj, w=5):
    """Stage 1: latest step exposing the failure symptom.

    Paper: scan the last ``w`` steps in reverse for the step that directly
    answers the query; fall back to the last executed step. Here the final
    executed step is the symptom anchor (it is tainted by construction).
    """
    exec_ids = [s.id for s in traj.steps if s.id in traj.executed]
    tail = exec_ids[-w:]
    for sid in reversed(tail):
        s = traj.steps[sid]
        if s.kind == "answer":
            return sid
    return exec_ids[-1]


def build_dependencies(traj):
    """Stage 2: structured trajectory H = (X, S, Estep, Esub).

    Estep: producer -> consumer edges from ``consumes`` (plus the
    deterministic instruction->response completion the paper applies).
    Esub: earlier subtask -> later subtask reusing its outputs.
    """
    estep = set()
    for s in traj.steps:
        for c in s.consumes:
            estep.add((c, s.id))
    # deterministic completion: an instruct immediately followed by a
    # response from the addressed consumer implies a dependency
    by_id = {s.id: s for s in traj.steps}
    for a, b in zip(traj.steps, traj.steps[1:]):
        if a.kind == "instruct" and b.consumes == () and b.id in traj.executed:
            estep.add((a.id, b.id))
    esub = set()
    seen_sub = {}
    for s in traj.steps:
        for c in s.consumes:
            ps = by_id[c].subtask
            if ps != s.subtask:
                esub.add((ps, s.subtask))
        seen_sub.setdefault(s.subtask, s.id)
    return estep, esub


def backward_trace(traj, final_id, estep):
    """Stage 3a: backward relevance tracing from the final failure step.

    Collects upstream steps whose outputs are directly or indirectly reused
    (paper's Ginfo); edges keep producer -> consumer direction.
    """
    producers = {}
    for a, b in estep:
        producers.setdefault(b, set()).add(a)
    seen, stack = {final_id}, [final_id]
    while stack:
        cur = stack.pop()
        for p in producers.get(cur, ()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    vedges = {(a, b) for a, b in estep if a in seen and b in seen}
    return seen, vedges


def annotate_propagation(traj, ginfo):
    """Stage 3b: forward annotation of Gprop nodes.

    introduce: output taint has a tag absent from all inputs' taint
    (a fresh deviation originates here).
    carry: tainted output, taint subset of inputs' union (downstream
    carrier / symptom).
    correct: clean output despite tainted inputs (recovered deviation).
    clean: everything untainted.
    """
    ann = {}
    for sid in ginfo:
        s = traj.steps[sid]
        out = traj.taint.get(sid, frozenset())
        in_union = frozenset().union(
            *(traj.taint.get(i, frozenset()) for i in s.consumes)
        ) if s.consumes else frozenset()
        if out - in_union:
            ann[sid] = "introduce"
        elif out:
            ann[sid] = "carry"
        elif in_union:
            ann[sid] = "correct"
        else:
            ann[sid] = "clean"
    return ann


def extract_loops(traj, min_rounds=3):
    """Stage 3c: persistent interaction loops (paper's Gloop).

    A loop is >= ``min_rounds`` executed steps sharing (agent, subtask,
    action_sig) with identical taint sets (no progress), regardless of
    interleaving steps -- loops "may span multiple subtasks and need not
    follow an ordinary information-dependency path". Extracted
    independently of the dependency graph, as in the paper.
    """
    from collections import defaultdict
    occ = defaultdict(list)
    for s in traj.steps:
        if s.id not in traj.executed or not s.action_sig:
            continue
        occ[(s.agent, s.subtask, s.action_sig)].append(s.id)
    loops = []
    for key, ids in occ.items():
        if len(ids) < min_rounds:
            continue
        taints = {traj.taint.get(i) for i in ids}
        if len(taints) == 1:  # same unresolved objective, no progress
            loops.append(sorted(ids))
    return sorted(loops, key=lambda l: l[0])


def _recovered(traj, sid, final_id):
    """A deviation introduced at ``sid`` is recovered iff none of its tags
    reach the final failure step."""
    s = traj.steps[sid]
    in_union = frozenset().union(
        *(traj.taint.get(i, frozenset()) for i in s.consumes)
    ) if s.consumes else frozenset()
    own = traj.taint.get(sid, frozenset()) - in_union
    return not (own & traj.taint.get(final_id, frozenset()))


def screen_candidates(traj, ginfo, ann, loops, final_id):
    """Stage 4: dual-branch candidate screening (<=2 per branch).

    Propagation branch: earliest *unrecovered* deviations (paper:
    "prioritizes the earliest unrecovered deviation that materially
    affects subsequent reasoning or actions"; carriers/symptoms excluded).
    Loop branch: per loop, the ambiguous instruction when one feeds the
    loop, else the first step initiating the unresolved repetition
    (paper: first occurrence selected as the group-level candidate).
    """
    prop = [sid for sid in sorted(ginfo)
            if ann.get(sid) == "introduce" and not _recovered(traj, sid, final_id)]
    xprop = prop[:2]
    xloop = []
    for loop in loops:
        first = loop[0]
        instr = None
        for sid in loop:
            for c in traj.steps[sid].consumes:
                cs = traj.steps[c]
                if cs.kind == "instruct" and cs.ambiguous:
                    instr = c
                    break
            if instr is not None:
                break
        xloop.append(instr if instr is not None else first)
        if len(xloop) >= 2:
            break
    seen, xcand = set(), []
    for sid in xprop + xloop:
        if sid not in seen:
            seen.add(sid)
            xcand.append(sid)
    return xcand  # chronological, <= 4


def counterfactual_verify(traj, candidates):
    """Stage 5: repair each candidate (chronological); the earliest whose
    repair recovers the execution is the decisive step (paper: model-based
    counterfactual assessment + "earliest decisive error" rule)."""
    for sid in sorted(candidates):
        taint, executed = repair_and_resimulate(traj, sid)
        if succeeds(taint, executed, traj.steps):
            return sid
    return sorted(candidates)[0] if candidates else None


def emfa_attribute(traj):
    """Full EMFA pipeline -> predicted (agent, step id)."""
    final_id = find_final_failure(traj)
    estep, _ = build_dependencies(traj)
    ginfo, _ = backward_trace(traj, final_id, estep)
    ann = annotate_propagation(traj, ginfo)
    loops = extract_loops(traj)
    cand = screen_candidates(traj, ginfo, ann, loops, final_id)
    pred = counterfactual_verify(traj, cand)
    return traj.steps[pred].agent, pred


# --------------------------------------------------------------------------
# Naive baselines (judge steps in isolation, no propagation modeling).


def blame_final(traj):
    final_id = find_final_failure(traj)
    return traj.steps[final_id].agent, final_id


def blame_latest_wrong(traj):
    final_id = find_final_failure(traj)
    cands = [s.id for s in traj.steps
             if s.id in traj.executed and s.id != final_id and traj.taint.get(s.id)]
    sid = max(cands) if cands else final_id
    return traj.steps[sid].agent, sid


def blame_earliest_wrong(traj):
    cands = [s.id for s in traj.steps
             if s.id in traj.executed and traj.taint.get(s.id)]
    sid = min(cands) if cands else find_final_failure(traj)
    return traj.steps[sid].agent, sid


# --------------------------------------------------------------------------
# Synthetic scenario builders (ground truth known by construction).


def _mk(ids, specs):
    return [Step(id=i, agent=a, subtask=st, kind=k, consumes=c,
                 action_sig=sig, ambiguous=amb)
            for i, (a, st, k, c, sig, amb) in zip(ids, specs)]


def _finish(steps, deviations, decisive, tag_counter):
    taint, executed, introduced = simulate(steps, dict(deviations))
    clean = {s.id: frozenset() for s in steps}
    return Trajectory(steps=steps, taint=taint, clean_taint=clean,
                      introduced=introduced, executed=executed,
                      decisive=decisive)


def make_cascade(seed=0):
    """Single root cause with pure downstream propagation."""
    k = seed % 3  # extra carrier steps
    specs = [("planner", "plan", "instruct", (), "instruct", False),
             ("researcher", "gather", "source", (0,), "fetch", False)]
    for j in range(2 + k):
        specs.append(("analyst", "analyze", "prop", (1 + j,), f"summarize:{j}", False))
    specs.append(("writer", "write", "answer", (3 + k,), "finalize", False))
    steps = _mk(range(len(specs)), specs)
    return _finish(steps, {1: "t1"}, ("researcher", 1), None)


def make_recovered(seed=0):
    """Early deviation corrected by a verifier; later independent root
    cause is decisive. Defeats earliest-wrong baselines."""
    gap = seed % 2  # 0-1 benign carrier steps before the decisive deviation
    specs = [
        ("planner", "plan", "instruct", (), "instruct", False),
        ("researcher", "gather", "source", (0,), "fetch", False),
        ("verifier", "check", "verify", (1,), "review", False),
        ("analyst", "analyze", "prop", (2,), "summarize", False),
    ]
    for j in range(gap):
        specs.append(("analyst", "analyze", "prop", (3 + j,), "recheck", False))
    base = 4 + gap
    specs += [
        ("coder", "build", "prop", (3 + gap,), "implement", False),
        ("reviewer", "review", "prop", (base,), "audit", False),
        ("writer", "write", "answer", (base + 1,), "finalize", False),
    ]
    steps = _mk(range(len(specs)), specs)
    dev = {1: "t1", base: "t4", "_correct": {2}}
    return _finish(steps, dev, ("coder", base), None)


def make_loop(seed=0, ambiguous=True):
    """Ambiguous instruction -> unresolved retry loop. Decisive = the
    instruction (paper's loop-branch rule); without ambiguity, the first
    failed attempt."""
    rounds = 3 + seed % 3
    specs = [("planner", "plan", "instruct", (), "instruct", ambiguous)]
    prev_test = None
    for r in range(rounds):
        ex_id = 1 + 2 * r
        te_id = 2 + 2 * r
        consumes = (0,) if r == 0 else (0, prev_test)
        specs.append(("coder", "build", "execute", consumes, "patch", False))
        specs.append(("tester", "test", "prop", (ex_id,), "runtests", False))
        prev_test = te_id
    specs.append(("writer", "write", "answer", (prev_test,), "finalize", False))
    steps = _mk(range(len(specs)), specs)
    # Ambiguous instruction: the ONLY deviation is the instruction itself;
    # every loop attempt merely carries its taint (paper: fixing the
    # instruction breaks the loop). Valid instruction: the executor's first
    # attempt introduces the deviation (failure to follow), retries carry.
    dev = {0: "t0"} if ambiguous else {1: "t1"}
    decisive = ("planner", 0) if ambiguous else ("coder", 1)
    return _finish(steps, dev, decisive, None)


def make_premature(seed=0):
    """Early deviation recovered; agent then stops prematurely although
    the task is unresolved. Decisive = the stop step (counterfactually
    decisive; locally it just says 'done')."""
    steps = _mk(range(7), [
        ("planner", "plan", "instruct", (), "instruct", False),
        ("researcher", "gather", "source", (0,), "fetch", False),
        ("verifier", "check", "verify", (1,), "review", False),
        ("analyst", "analyze", "prop", (2,), "summarize", False),
        ("coder", "build", "prop", (3,), "implement", False),
        ("coder", "build", "stop", (4,), "declare_done", False),
        ("writer", "write", "answer", (5,), "finalize", False),
    ])
    dev = {1: "t1", 5: "premature", "_correct": {2}}
    return _finish(steps, dev, ("coder", 5), None)


def make_redundant(seed=0):
    """Two tainted parallel branches feed a voter that succeeds if EITHER
    branch is clean. Repairing either branch recovers -> the paper's
    "earliest decisive error" rule picks the earlier one. Defeats
    latest-wrong baselines."""
    steps = _mk(range(7), [
        ("planner", "plan", "instruct", (), "instruct", False),
        ("researcher_a", "gather", "source", (0,), "fetch_a", False),
        ("analyst_a", "analyze", "prop", (1,), "summarize_a", False),
        ("researcher_b", "gather", "source", (0,), "fetch_b", False),
        ("analyst_b", "analyze", "prop", (3,), "summarize_b", False),
        ("voter", "decide", "vote", (2, 4), "vote", False),
        ("writer", "write", "answer", (5,), "finalize", False),
    ])
    return _finish(steps, {1: "ta", 3: "tb"}, ("researcher_a", 1), None)


BUILDERS = {
    "cascade": make_cascade,
    "recovered": make_recovered,
    "loop": make_loop,
    "premature": make_premature,
    "redundant": make_redundant,
}
