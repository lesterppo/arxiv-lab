"""WikiTrace: evidence-linked memory for LLM agents
(arXiv:2611.14320, "Don't Just Trust Your Memory: Evidence-Linked Memory
Reduces Memory-Induced Degradation in LLM Agents").

Paper claim: external memory distills past trajectories into concise lessons,
but a lesson can be wrong or misapplied, and the agent cannot check it
because the lesson is given without its source trajectory -- so memory can
cause memory-induced degradation (tasks the agent solves without memory but
fails with it). WikiTrace stores each lesson as a concise insight linked to
a compilation (why its trajectory succeeded/failed) and to the raw trajectory
itself. On a new task the agent starts from all insights and, whenever one
lacks details needed to act, conflicts with another, or comes from a failed
task, descends through its links to fill in missing details or correct the
insight before acting. On five agentic benchmarks WikiTrace beats No Memory
by 10.7 points on average and Insights Only by 3.2, and cuts Regression
(successes turned into failures) from 6.5% (Insights Only) to 4.5%.

This module implements the mechanism, not the paper's benchmarks:
  * ``Task`` -- a synthetic multi-step tool-use task: three steps
    (locate / operate / verify), each requiring one of three actions.
    The gold action for the first two steps is fixed; for the third it
    depends on the task's attribute (object kind). The gold procedure is
    computed by the generator, so success is measured, not judged.
  * ``DefaultAgent`` -- a parameterized stand-in for a model agent: per
    step it picks the gold action with probability ``P_DEFAULT`` and a
    random wrong action otherwise. Its choices are seeded per task, so
    every memory arm faces the *same* default choices on the same task
    (paired Gain/Regression, as in the paper).
  * ``Compiler`` -- turns each past run into a compilation (a diagnosis of
    why the run succeeded/failed) and up to three insights (one per step).
    Each insight states what to do, when it applies (a scope: ``any``
    attribute, or one specific attribute), which run it came from, and
    whether that run succeeded. Two honest error sources, matching the
    paper's failure analysis: (a) over-generalization -- a lesson drawn
    from one attribute is recorded with scope ``any``; (b) misattribution
    on failed runs -- with probability ``CORRUPT_P`` the compiler blames
    the wrong action, so the insight claims an action the trajectory
    never took. Both are detectable from the linked trajectory, which
    is the paper's central finding ("a trajectory often still holds the
    evidence needed to catch an erroneous lesson").
  * ``Memory`` -- insights + compilations + raw trajectories, with source
    links (insight -> compilation -> trajectory).
  * ``verify_lesson`` -- the deterministic rule-based stand-in for the
    paper's LLM evidence check: it reads the source compilation, then
    the raw trajectory step of the same type, and returns SUPPORTED
    (the trajectory shows the claimed action succeeding for this
    task's attribute), CONTRADICTED (the trajectory shows a different
    action was taken, or the claimed action failing), UNSUPPORTED (the
    lesson is scoped ``any`` but its source run used a different
    attribute -- an over-generalization the trajectory cannot vouch
    for), or UNKNOWN (no matching step, or the step's observation was
    not recorded). The check is sound by construction: a SUPPORTED
    lesson always prescribes the gold action for the current task.
  * ``solve`` -- three arms: no-memory (default policy), plain lessons
    (``Insights Only``: follow the first applicable lesson blindly),
    and WikiTrace (descend on the paper's three triggers -- missing
    detail, conflict, failed-run origin -- verify each applicable
    lesson against its evidence, follow it only if supported, else
    fall back to the default policy).

RNG: SplitMix64, implemented here in pure Python so the Rust second stage
can reproduce the exact same streams from the same seeds (parity sweep).

Stdlib only. No network, no model API.
"""

# ---------------------------------------------------------------- RNG

_MASK64 = (1 << 64) - 1


class SplitMix64:
    """Deterministic 64-bit RNG (same stream in the Rust port)."""

    def __init__(self, seed):
        self.state = seed & _MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def randbelow(self, n):
        # rejection sampling, unbiased for n <= 2**64
        assert n > 0
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            v = self.next_u64()
            if v < limit:
                return v % n

    def random(self):
        return self.next_u64() / float(1 << 64)

    def choice(self, seq):
        return seq[self.randbelow(len(seq))]


# ---------------------------------------------------------------- parameters

STEP_TYPES = (0, 1, 2)          # 0 locate, 1 operate, 2 verify
STEP_NAMES = ("locate", "operate", "verify")
N_ACTIONS = 3
N_ATTRS = 4                      # object kinds 0..3
P_DEFAULT = 0.6                  # default agent picks the gold action w.p.
CORRUPT_P = 0.20                 # failed-run insight misattributes w.p.
OBSERVED_P = 0.80                # trajectory step observation is recorded w.p.
SCOPE_ANY_P = 0.50               # insight scope recorded as "any" w.p.

SEED = 20261112
N_TRAIN = 12
N_TEST = 600


def gold_action(step_type, attr):
    """Gold action for a step type on a task with the given attribute."""
    if step_type == 0:
        return 0
    if step_type == 1:
        return 1
    return attr % N_ACTIONS  # verify step: gold depends on the object kind


# ---------------------------------------------------------------- tasks


class Task:
    __slots__ = ("tid", "attr")

    def __init__(self, tid, attr):
        self.tid = tid
        self.attr = attr

    def __repr__(self):
        return "Task(%d, attr=%d)" % (self.tid, self.attr)


def gen_tasks(seed, n):
    """Deterministic task set: task i uses SplitMix64(seed, i) for its attr."""
    rng = SplitMix64(seed)
    return [Task(i, rng.randbelow(N_ATTRS)) for i in range(n)]


def default_choices(task, base_seed):
    """The default agent's per-step choices for this task (paired across
    arms: same task -> same choices, so Gain/Regression are paired)."""
    rng = SplitMix64((base_seed * 1000003 + task.tid * 7919) & _MASK64)
    choices = []
    for st in STEP_TYPES:
        gold = gold_action(st, task.attr)
        if rng.random() < P_DEFAULT:
            choices.append(gold)
        else:
            wrong = [a for a in range(N_ACTIONS) if a != gold]
            choices.append(wrong[rng.randbelow(len(wrong))])
    return choices


# ---------------------------------------------------------------- runs


class StepRecord:
    __slots__ = ("step_type", "chosen", "ok", "observed")

    def __init__(self, step_type, chosen, ok, observed):
        self.step_type = step_type
        self.chosen = chosen
        self.ok = ok
        self.observed = observed


class Run:
    __slots__ = ("rid", "task", "steps", "success")

    def __init__(self, rid, task, steps):
        self.rid = rid
        self.task = task
        self.steps = steps
        self.success = all(s.ok for s in steps)


def run_default(task, base_seed, obs_seed):
    """Run the default agent on a task; record the raw trajectory."""
    rng = SplitMix64((obs_seed * 1000003 + task.tid * 104729) & _MASK64)
    steps = []
    for i, st in enumerate(STEP_TYPES):
        chosen = default_choices(task, base_seed)[i]
        ok = chosen == gold_action(st, task.attr)
        observed = rng.random() < OBSERVED_P
        steps.append(StepRecord(st, chosen, ok, observed))
    return Run(task.tid, task, steps)


# ---------------------------------------------------------------- compiler


class Insight:
    __slots__ = ("iid", "step_type", "claimed", "scope", "source_run",
                 "source_success", "corrupted")

    def __init__(self, iid, step_type, claimed, scope, source_run,
                 source_success, corrupted=False):
        self.iid = iid
        self.step_type = step_type
        self.claimed = claimed
        self.scope = scope          # "any" or ("attr", a)
        self.source_run = source_run
        self.source_success = source_success
        self.corrupted = corrupted  # ground truth only; never shown to agent

    def applies_to(self, attr):
        return self.scope == "any" or self.scope[1] == attr

    def __repr__(self):
        return ("Insight(%s, %s -> %d, scope=%r, run=%d, %s)" %
                (self.iid, STEP_NAMES[self.step_type], self.claimed,
                 self.scope, self.source_run,
                 "ok" if self.source_success else "FAIL"))


class Memory:
    """WikiTrace memory M = (R, C, I, E): raw trajectories, compilations,
    insights, and source links (insight -> compilation -> trajectory)."""

    def __init__(self, insights, compilations, trajectories):
        self.insights = insights
        self.compilations = compilations    # rid -> str
        self.trajectories = trajectories    # rid -> Run

    def applicable(self, step_type, attr):
        return [ins for ins in self.insights
                if ins.step_type == step_type and ins.applies_to(attr)]

    def subset(self, source_success):
        """Memory keeping only insights from runs with the given outcome
        (paper section 5.3: memory built from successful / failed runs)."""
        kept = [ins for ins in self.insights
                if ins.source_success == source_success]
        return Memory(kept, self.compilations, self.trajectories)


def compile_memory(runs, seed):
    """Compiler: one compilation + up to three insights per run."""
    rng = SplitMix64(seed)
    insights = []
    compilations = {}
    trajectories = {}
    for run in runs:
        trajectories[run.rid] = run
        if run.success:
            compilations[run.rid] = (
                "Run %d succeeded on attr=%d. Decisions associated with "
                "success: %s." % (run.rid, run.task.attr,
                                 ", ".join("step %s: action %d worked"
                                           % (STEP_NAMES[s.step_type], s.chosen)
                                           for s in run.steps)))
        else:
            failed = [s for s in run.steps if not s.ok]
            compilations[run.rid] = (
                "Run %d FAILED on attr=%d. First failing step: %s with "
                "action %d (observation: step failed). Plausible cause: "
                "wrong action for this object kind." %
                (run.rid, run.task.attr, STEP_NAMES[failed[0].step_type],
                 failed[0].chosen))
        for s in run.steps:
            claimed = s.chosen
            corrupted = False
            if not run.success and rng.random() < CORRUPT_P:
                # Misattribution: the compiler blames an action the run
                # never took (the paper's "wrong guess" about which action
                # mattered, invisible from the success/failure label).
                claimed = rng.choice([a for a in range(N_ACTIONS)
                                      if a != s.chosen])
                corrupted = True
            scope = "any" if rng.random() < SCOPE_ANY_P else ("attr", run.task.attr)
            insights.append(Insight(
                "I-%04d" % len(insights), s.step_type, claimed, scope,
                run.rid, run.success, corrupted))
    return Memory(insights, compilations, trajectories)


# ---------------------------------------------------------------- evidence check


# verify_lesson outcomes
SUPPORTED = "supported"
CONTRADICTED = "contradicted"
UNSUPPORTED = "unsupported"
UNKNOWN = "unknown"


def descent_triggers(lessons):
    """Paper section 3.4: descend when an insight lacks the details needed
    to act (scope 'any' -- it does not say which attribute it holds for),
    conflicts with another insight, or comes from a failed run."""
    if not lessons:
        return False
    if any(not ins.source_success for ins in lessons):
        return True
    if len({ins.claimed for ins in lessons}) > 1:
        return True
    if any(ins.scope == "any" for ins in lessons):
        return True
    return False


def verify_lesson(lesson, task_attr, memory):
    """Rule-based evidence check (deterministic stand-in for the paper's
    LLM check): read the source compilation, then the raw trajectory step
    of the same type. Returns (verdict, n_evidence_reads).

    Sound by construction: SUPPORTED implies the claimed action is the
    gold action for the current task (the trajectory shows that action
    succeeding on a run with the same attribute).
    """
    reads = 2  # one compilation read + one raw-trajectory read
    _comp = memory.compilations.get(lesson.source_run, "")
    run = memory.trajectories.get(lesson.source_run)
    if run is None:
        return UNKNOWN, reads
    step = None
    for s in run.steps:
        if s.step_type == lesson.step_type:
            step = s
            break
    if step is None or not step.observed:
        return UNKNOWN, reads
    if lesson.claimed != step.chosen:
        return CONTRADICTED, reads  # insight misattributes the run
    if not step.ok:
        return CONTRADICTED, reads  # claimed action failed in its own run
    if lesson.scope == "any" and run.task.attr != task_attr:
        return UNSUPPORTED, reads  # over-generalized beyond its evidence
    return SUPPORTED, reads


# ---------------------------------------------------------------- agents


def solve(task, memory, mode, base_seed):
    """Run one task under one memory arm. Returns (success, stats).

    modes: 'none' (no memory), 'plain' (Insights Only: follow the first
    applicable lesson blindly), 'wikitrace' (descend + verify first).
    stats: descents, corrections (steps where WikiTrace's action differs
    from what blind-follow would have taken), evidence_reads.
    """
    choices = default_choices(task, base_seed)
    stats = {"descents": 0, "corrections": 0, "evidence_reads": 0,
             "lessons_followed": 0}
    success = True
    for i, st in enumerate(STEP_TYPES):
        action = choices[i]
        if memory is not None:
            lessons = memory.applicable(st, task.attr)
            if lessons:
                blind = lessons[0].claimed
                if mode == "plain":
                    action = blind
                    stats["lessons_followed"] += 1
                elif mode == "wikitrace":
                    if descent_triggers(lessons):
                        stats["descents"] += 1
                        # Descend through the links in order and stop at
                        # the first lesson the evidence supports (the
                        # agent looks for the fact that settles the
                        # question; every supported lesson prescribes the
                        # same gold action, so stopping early changes no
                        # outcome, only the evidence cost).
                        supported = None
                        for ins in lessons:
                            verdict, reads = verify_lesson(ins, task.attr, memory)
                            stats["evidence_reads"] += reads
                            if verdict == SUPPORTED:
                                supported = ins
                                break
                        if supported is not None:
                            action = supported.claimed
                            stats["lessons_followed"] += 1
                        # else: no verified lesson, or verified lessons
                        # still conflict -> fall back to the default policy
                    else:
                        action = blind
                        stats["lessons_followed"] += 1
                    if action != blind:
                        stats["corrections"] += 1
        if action != gold_action(st, task.attr):
            success = False
    return success, stats


# ---------------------------------------------------------------- experiment


def run_experiment(seed=SEED, n_train=N_TRAIN, n_test=N_TEST):
    """Main experiment. Returns a dict of measured rates."""
    train_tasks = gen_tasks(seed, n_train)
    test_tasks = gen_tasks(seed + 5000, n_test)
    runs = [run_default(t, seed, seed + 7) for t in train_tasks]
    memory = compile_memory(runs, seed + 11)

    arms = {}
    for mode in ("none", "plain", "wikitrace"):
        results = [solve(t, memory if mode != "none" else None, mode, seed)
                   for t in test_tasks]
        arms[mode] = results

    def rate(results):
        return sum(1 for ok, _ in results if ok) / len(results)

    base = [ok for ok, _ in arms["none"]]
    out = {}
    for mode in ("none", "plain", "wikitrace"):
        res = arms[mode]
        ok = [o for o, _ in res]
        gain = sum(1 for b, o in zip(base, ok) if not b and o) / len(base)
        regr = sum(1 for b, o in zip(base, ok) if b and not o) / len(base)
        regr_solved = (sum(1 for b, o in zip(base, ok) if b and not o)
                       / max(1, sum(base)))
        out[mode] = {"success": rate(res), "gain": gain,
                     "regression": regr, "regression_of_solved": regr_solved}
    wt_stats = arms["wikitrace"]
    n_steps = len(test_tasks) * len(STEP_TYPES)
    out["wikitrace"]["descent_rate"] = (
        sum(s["descents"] for _, s in wt_stats) / n_steps)
    out["wikitrace"]["correction_rate"] = (
        sum(s["corrections"] for _, s in wt_stats) / n_steps)
    out["wikitrace"]["evidence_reads_per_task"] = (
        sum(s["evidence_reads"] for _, s in wt_stats) / len(test_tasks))
    out["wikitrace"]["lessons_followed_per_task"] = (
        sum(s["lessons_followed"] for _, s in wt_stats) / len(test_tasks))
    out["plain"]["lessons_followed_per_task"] = (
        sum(s["lessons_followed"] for _, s in arms["plain"]) / len(test_tasks))

    # Paper section 5.3: memory built from successful runs only / failed
    # runs only, same insights otherwise.
    for subset, label in ((True, "succ_only"), (False, "fail_only")):
        mem = memory.subset(subset)
        res = [solve(t, mem, "wikitrace", seed) for t in test_tasks]
        ok = [o for o, _ in res]
        out["wikitrace_" + label] = {
            "success": sum(ok) / len(ok),
            "gain": sum(1 for b, o in zip(base, ok) if not b and o) / len(base),
            "regression": sum(1 for b, o in zip(base, ok) if b and not o) / len(base),
            "n_insights": len(mem.insights),
        }
    out["n_train_runs"] = n_train
    out["n_test_tasks"] = n_test
    out["n_insights"] = len(memory.insights)
    out["n_insights_corrupted"] = sum(1 for i in memory.insights if i.corrupted)
    out["n_insights_from_failed_runs"] = sum(
        1 for i in memory.insights if not i.source_success)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))
