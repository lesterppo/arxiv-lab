"""CPU test for arXiv 2610.02994 — Sentry: conditional failure knowledge.

Reproduces the paper's core claims on a deterministic synthetic agent suite:
 1. Sentry (detect -> retrieve -> recover -> verify -> store-if-verified) beats
    a runtime-intervention baseline that retrieves but never learns (paper:
    +37% avg on agentic benchmarks).
 2. Exposing the FULL playbook in context lowers performance vs conditional
    retrieval (conditional lessons misfire when their failure is absent).
 3. The store gate rejects lessons from unverified recoveries and accepts
    verified ones; learned lessons transfer to held-out tasks.
 4. Recovery verification uses NO task rewards / success labels.

Deterministic: fixed task suite, seeded misfire RNG.
stdlib only.
"""
import inspect
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arxiv_lab.sentry.sentry import (  # noqa: E402
    Action, FailureDetector, FailureEvent,
    Lesson, Playbook, RecoveryVerifier, Sentry,
    INVALID_TOOL_CALL, REPEATED_ACTION, POOR_GROUNDING, PREMATURE_FINISH,
)

ALLOWED = {"search", "lookup", "finish"}
BUDGET = 6
MISFIRE_P = 0.35

TRAP_OF = {None: None, "invalid_args": INVALID_TOOL_CALL,
           "repeated": REPEATED_ACTION, "premature": PREMATURE_FINISH}


def seed_playbook():
    pb = Playbook()
    pb.add(Lesson("L1", INVALID_TOOL_CALL,
                  "a tool call errored on its arguments",
                  "quote the query string and retry the same tool",
                  source_task="seed"))
    pb.add(Lesson("L2", REPEATED_ACTION,
                  "the same action repeated with no progress",
                  "switch to the 'lookup' tool instead of repeating",
                  source_task="seed"))
    return pb


# ---------------------------------------------------------------------------
# Simulated agent
# ---------------------------------------------------------------------------

class SimAgent:
    """Capability-limited agent. Without the matching lesson it falls into the
    task's trap; with it, it applies the guidance. Misfires (arm B only) apply
    conditional lessons whose failure is absent."""

    def __init__(self, rng):
        self.rng = rng
        self.misfires = 0

    def _natural(self, task):
        trap = task["trap"]
        if trap is None:
            return Action("search", {"q": task["q"]}, "ok: 3 results")
        if trap == "invalid_args":
            return Action("search", {"q": task["q"]},  # unquoted -> error
                          "ERROR: bad args, quote the query string")
        if trap == "repeated":
            return Action("search", {"q": task["q"]}, "no results yet")
        if trap == "premature":
            return Action("finish", {}, "")
        raise AssertionError(trap)

    def _guided(self, task, lesson):
        trap = task["trap"]
        if lesson.failure_type == INVALID_TOOL_CALL and trap == "invalid_args":
            return Action("search", {"q": f'"{task["q"]}"'}, "ok: 3 results")
        if lesson.failure_type == REPEATED_ACTION and trap == "repeated":
            return Action("lookup", {"q": task["q"]}, "ok: 1 result")
        if lesson.failure_type == PREMATURE_FINISH and trap == "premature":
            return [Action("search", {"q": task["q"]}, "ok: 3 results"),
                    Action("finish", {}, "done")]
        return None  # guidance does not apply

    def _misfire(self, task, lesson, trace):
        """Apply a conditional lesson when its failure is ABSENT. Costs steps,
        sometimes breaks a working action."""
        self.misfires += 1
        if lesson.failure_type == INVALID_TOOL_CALL:
            # re-quotes an already-fine call: 50% it errors
            if self.rng.random() < 0.5:
                trace.append(Action("search", {"q": f'"{task["q"]}"'},
                                    "ERROR: bad args, double-quoted"))
            else:
                trace.append(Action("search", {"q": f'"{task["q"]}"'},
                                    "ok: 3 results"))
        elif lesson.failure_type == REPEATED_ACTION:
            trace.append(Action("lookup", {"q": task["q"]}, "ok: 1 result"))
        else:  # PREMATURE_FINISH lesson misfires -> needless evidence step
            trace.append(Action("search", {"q": task["q"]}, "ok: 3 results"))

    def _goal_reached(self, task, trace):
        trap = task["trap"]
        if trap == "premature":
            # need evidence before finish
            seen_evidence = any(a.tool in ("search", "lookup") and "ok:" in a.observation
                                for a in trace)
            return seen_evidence and trace and trace[-1].tool == "finish"
        return any("ok:" in a.observation and not self._err(a) for a in trace)

    @staticmethod
    def _err(a):
        return "ERROR" in (a.observation or "")

    def run(self, task, visible_lessons):
        trace = []
        rng = self.rng
        # misfires first: lessons whose failure is absent from this task
        for les in visible_lessons:
            if TRAP_OF[task["trap"]] != les.failure_type:
                if rng.random() < MISFIRE_P:
                    self._misfire(task, les, trace)
                    if len(trace) >= BUDGET:
                        return False, trace
        # matching lesson, if any
        match = next((l for l in visible_lessons
                      if TRAP_OF[task["trap"]] == l.failure_type), None)
        if match is not None:
            guided = self._guided(task, match)
            if isinstance(guided, list):
                trace.extend(guided)
            else:
                trace.append(guided)
            return self._goal_reached(task, trace) and len(trace) <= BUDGET, trace
        # natural behavior
        if task["trap"] == "repeated":
            for _ in range(3):  # falls into the loop
                trace.append(self._natural(task))
            return False, trace
        trace.append(self._natural(task))
        return self._goal_reached(task, trace), trace


# ---------------------------------------------------------------------------
# Task suite: 48 tasks, traps cycle deterministically
# ---------------------------------------------------------------------------

def make_tasks():
    traps = [None, "invalid_args", "repeated", "premature"]
    return [{"id": f"T{i:02d}", "q": f"query-{i}", "trap": traps[i % 4]}
            for i in range(48)]


def run_arm(name, tasks, seed_playbook_fn, use_sentry, learn, full_context,
            rng_seed):
    rng = random.Random(rng_seed)
    agent = SimAgent(rng)
    detector = FailureDetector(allowed_tools=ALLOWED)
    verifier = RecoveryVerifier(detector)
    pb = seed_playbook_fn()
    sentry = Sentry(pb, detector, verifier)
    ok = 0
    for task in tasks:
        if full_context:
            visible = list(pb._lessons)  # whole playbook in context (arm B)
            success, _ = agent.run(task, visible)
        elif use_sentry:
            success, trace = agent.run(task, [])
            if not success:
                failures = detector.detect(trace)
                assert failures, f"no failure detected for {task['id']}"
                lessons = sentry.on_failure(failures[0])
                if lessons:
                    guided = agent._guided(task, lessons[0])
                    after = guided if isinstance(guided, list) else [guided]
                    ok_rec = verifier.verify(trace, after, failures[0])
                    if ok_rec:
                        sentry.record_recovery(lessons[0])
                    success = agent._goal_reached(task, after)
                else:
                    # no matching lesson: generic unguided retry. For premature
                    # traps this only works half the time (seeded) — the agent
                    # has no reliable recovery strategy without a lesson.
                    if task["trap"] == "premature" and rng.random() < 0.5:
                        after = [Action("search", {"q": task["q"]}, "ok: 3 results"),
                                 Action("finish", {}, "done")]
                        ok_rec = verifier.verify(trace, after, failures[0])
                        if learn and sentry.playbook.get("L-prem") is None:
                            cand = Lesson("L-prem", PREMATURE_FINISH,
                                          "agent finished before gathering evidence",
                                          "gather evidence with search/lookup before finish",
                                          source_task=task["id"])
                            sentry.maybe_store(cand, ok_rec)
                        success = agent._goal_reached(task, after)
                    else:
                        success = False
        else:  # arm A: nothing; arm D: static retrieve, no learning
            if use_sentry is False and seed_playbook_fn is not None and name == "D":
                success, trace = agent.run(task, [])
                if not success:
                    failures = detector.detect(trace)
                    lessons = sentry.on_failure(failures[0]) if failures else []
                    if lessons:
                        guided = agent._guided(task, lessons[0])
                        after = guided if isinstance(guided, list) else [guided]
                        success = agent._goal_reached(task, after)
            else:
                success, _ = agent.run(task, [])
        ok += success
    return ok, len(tasks), agent.misfires, sentry


def main():
    tasks = make_tasks()
    train, held = tasks[:32], tasks[32:]

    # --- unit: detector -------------------------------------------------
    d = FailureDetector(allowed_tools=ALLOWED)
    assert d.detect([Action("search", {"q": "x"}, "ERROR: bad args")])[0].ftype \
        == INVALID_TOOL_CALL
    rep = [Action("search", {"q": "x"}, "no results yet")] * 3
    assert d.detect(rep)[0].ftype == REPEATED_ACTION
    assert d.detect([Action("finish", {}, "")])[0].ftype == PREMATURE_FINISH
    ev = d.detect([Action("search", {"q": "x"}, "ok: 3 results"),
                   Action("finish", {}, "done")])
    assert not [e for e in ev if e.ftype == PREMATURE_FINISH], ev
    ev = d.detect([Action("search", {"q": "x"}, "ok: 3 results",
                            thought="I will now invoke QuantumFluxCapacitor")])
    assert any(e.ftype == POOR_GROUNDING for e in ev), ev
    # unknown tool flagged
    ev = d.detect([Action("frobnicate", {}, "ERROR: unknown tool")])
    assert ev[0].ftype == INVALID_TOOL_CALL

    # --- unit: verifier takes no reward/label ---------------------------
    sig = inspect.signature(RecoveryVerifier.verify)
    params = set(sig.parameters)
    assert params == {"self", "trace_before", "trace_after", "failure"}, params

    before = [Action("search", {"q": "x"}, "ERROR: bad args, quote the query")]
    after_ok = [Action("search", {"q": '"x"'}, "ok: 3 results")]
    after_bad = [Action("search", {"q": "x"}, "ERROR: bad args, quote the query")]
    v = RecoveryVerifier(d)
    f = FailureEvent(INVALID_TOOL_CALL, 0, "e")
    assert v.verify(before, after_ok, f) is True
    assert v.verify(before, after_bad, f) is False   # same failure recurs
    assert v.verify(before, [], f) is False

    # --- unit: playbook conditional exposure -----------------------------
    pb = seed_playbook()
    assert pb.retrieve(FailureEvent("no_such_type", 0, "")) == []
    got = pb.retrieve(FailureEvent(INVALID_TOOL_CALL, 0, ""))
    assert {l.id for l in got} == {"L1"} and len(pb) == 2

    # --- arms on train ----------------------------------------------------
    okA, nA, _, _ = run_arm("A", train, seed_playbook, False, False, False, 7)
    okB, nB, misB, _ = run_arm("B", train, seed_playbook, False, False, True, 7)
    okD, nD, _, sentryD = run_arm("D", train, seed_playbook, False, False, False, 7)
    okC, nC, _, sentryC = run_arm("C", train, seed_playbook, True, True, False, 7)

    print(f"train  A none:            {okA}/{nA} = {okA/nA:.3f}")
    print(f"train  B full-in-context: {okB}/{nB} = {okB/nB:.3f}  (misfires={misB})")
    print(f"train  D static-retrieve: {okD}/{nD} = {okD/nD:.3f}")
    print(f"train  C sentry+learn:    {okC}/{nC} = {okC/nC:.3f}  "
          f"(stored={sentryC.stats['stored']})")

    # paper claim 1: Sentry beats runtime-intervention (no-learning) baseline
    assert okC > okD, f"C {okC} should beat D {okD}"
    rel = (okC - okD) / okD if okD else float("inf")
    print(f"C vs D relative improvement: {rel:+.1%} (paper reports +37% avg)")
    # paper claim 2: full playbook in context hurts vs conditional retrieval
    assert okB < okC, f"B {okB} should be below C {okC}"
    assert misB > 0, "expected misfires in full-context arm"
    print(f"full-context misfires: {misB}; B vs C gap: {(okC-okB)/nB:+.1%}")
    # claim 3a: gate accepted the verified premature lesson
    assert sentryC.stats["stored"] >= 1, sentryC.stats
    assert sentryC.playbook.get("L-prem") is not None

    # --- claim 3b: gate rejects unverified lessons -------------------------
    pb2 = seed_playbook()
    s2 = Sentry(pb2, d, v)
    cand = Lesson("L-bogus", PREMATURE_FINISH, "t", "finish immediately",
                  source_task="T99")
    before2 = [Action("finish", {}, "")]
    after2 = [Action("finish", {}, "")]  # failed recovery: same pattern
    f2 = FailureEvent(PREMATURE_FINISH, 0, "e")
    assert v.verify(before2, after2, f2) is False
    assert s2.maybe_store(cand, False) is False
    assert len(pb2) == 2 and s2.stats["rejected"] == 1

    # --- claim 3c: learned lesson transfers to held-out --------------------
    # C keeps its learned playbook; D stays static. Unguided premature
    # recovery succeeds only half the time (same seed both arms).
    rng = random.Random(7)
    for arm, sentry in (("C", sentryC), ("D", sentryD)):
        ok = 0
        agent = SimAgent(random.Random(7))
        hrng = random.Random(7)
        for task in held:
            success, trace = agent.run(task, [])
            if not success:
                failures = d.detect(trace)
                lessons = sentry.on_failure(failures[0]) if failures else []
                if lessons:
                    guided = agent._guided(task, lessons[0])
                    after = guided if isinstance(guided, list) else [guided]
                    success = agent._goal_reached(task, after)
                elif task["trap"] == "premature" and hrng.random() < 0.5:
                    after = [Action("search", {"q": task["q"]}, "ok: 3 results"),
                             Action("finish", {}, "done")]
                    success = agent._goal_reached(task, after)
            ok += success
        print(f"held   {arm}: {ok}/{len(held)} = {ok/len(held):.3f}")
        if arm == "C":
            okCh = ok
        else:
            okDh = ok
    assert okCh > okDh, f"held-out transfer: C {okCh} should beat D {okDh}"

    print("ALL SENTRY CHECKS PASSED")


if __name__ == "__main__":
    main()
