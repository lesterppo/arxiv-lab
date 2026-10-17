"""Trajectory-level speculation with semantic verification (arXiv:2610.21463).

Paper claim implemented: SpecAgent turns the strong model from the *author*
of each tool-use step into a *reviewer* of steps drafted by a weak model.
The weak model executes read-only tool calls itself while drafting (so every
drafted step is grounded in real observations) and *holds* the first action
that would modify the environment. The strong model then verifies the whole
segment in a single parallel prefill pass, commits the longest prefix it
judges semantically valid, and repairs the first rejected step. Because
verification replaces autoregressive decoding with prefill, the strong model
decodes only short verdicts and occasional repairs instead of every step.

Reported on tau2-Bench: within ~1 point of strong-only success while roughly
halving strong-model memory traffic and wall time.

Sim mapping in this module (stdlib only, deterministic, seeded RNG):
  * tool environment: a grid world with a read-only ``scan`` tool and write
    ``goto`` / ``collect`` tools, plus a terminal ``done`` action;
  * weak policy: cheap, noisy greedy planner (:class:`WeakPolicy`);
  * strong policy: accurate verifier / author (:class:`StrongPolicy`),
    expensive per decoded token;
  * token accounting: strong decoded vs prefill tokens, weak decoded tokens.

Public API: ``ToolWorld``, ``Step``, ``WeakPolicy``, ``StrongPolicy``,
``SpecAgent``, ``run_strong_only``, ``run_weak_only``, ``evaluate``,
``TOKEN_COSTS``.
"""

# Token-cost model (per step, in "tokens"; the absolute scale is arbitrary,
# the *ratios* encode the paper's asymmetry: strong decode is expensive,
# verification is a short verdict per step inside one prefill pass).
TOKEN_COSTS = {
    "strong_decode_per_step": 40,    # strong model authoring one step
    "strong_verdict_per_step": 2,    # short accept/reject verdict per step
    "strong_repair": 40,             # strong model repairing one step
    "strong_prefill_per_step": 10,   # parallel prefill over one segment step
    "weak_decode_per_step": 25,      # weak model drafting one step
}

READ = "read"
WRITE = "write"
TERMINAL = "terminal"


class Step:
    """One drafted tool-use step."""

    __slots__ = ("name", "args", "kind", "observation", "executed",
                 "executor", "verdict")

    def __init__(self, name, args, kind, observation=None, executed=False,
                 executor=None):
        self.name = name            # tool name: "scan"/"goto"/"collect"/"done"
        self.args = args            # tuple of args
        self.kind = kind            # READ / WRITE / TERMINAL
        self.observation = observation  # real observation for executed reads
        self.executed = executed    # whether the tool call really ran
        self.executor = executor    # "weak" / "orchestrator" / None
        self.verdict = None         # strong-model verdict (True/False)

    def __repr__(self):
        return ("Step(%r, args=%r, kind=%s, executed=%s, executor=%r, "
                "verdict=%r)" % (self.name, self.args, self.kind,
                                 self.executed, self.executor, self.verdict))


class ToolWorld:
    """Deterministic grid tool-world.

    Tools:
      * ``scan(radius)`` -- READ-ONLY: uncollected targets within Manhattan
        distance ``radius`` of the agent.
      * ``goto(x, y)`` -- WRITE: moves the agent.
      * ``collect()`` -- WRITE: collects a target if the agent stands on one.
      * ``done()`` -- TERMINAL: ends the episode (no env modification).

    Every environment-modifying call is logged as ``(executor, tool)`` in
    ``write_log`` so tests can assert the weak model never executed a write.
    """

    READ_TOOLS = frozenset({"scan"})
    WRITE_TOOLS = frozenset({"goto", "collect"})

    def __init__(self, size, targets, start=(0, 0)):
        self.size = size
        self.targets = frozenset(targets)
        self.pos = start
        self.collected = set()
        self.write_log = []  # list of (executor, tool_name)

    def scan(self, radius, executor="weak"):
        x0, y0 = self.pos
        out = [(x, y) for (x, y) in self.targets
               if (x, y) not in self.collected
               and abs(x - x0) + abs(y - y0) <= radius]
        return tuple(sorted(out))

    def goto(self, x, y, executor="orchestrator"):
        assert 0 <= x < self.size and 0 <= y < self.size, "goto out of bounds"
        self.pos = (x, y)
        self.write_log.append((executor, "goto"))

    def collect(self, executor="orchestrator"):
        ok = self.pos in self.targets and self.pos not in self.collected
        if ok:
            self.collected.add(self.pos)
        self.write_log.append((executor, "collect"))
        return ok

    def is_done(self):
        return self.collected == set(self.targets)

    def uncollected(self):
        return [t for t in self.targets if t not in self.collected]


class WeakPolicy:
    """Cheap, noisy planner: drafts segments, executes reads, holds writes.

    ``draft`` executes up to ``max_reads`` read-only ``scan`` calls for real
    (grounding every read in a true observation) and then appends exactly one
    held step -- the first environment-modifying action (or ``done``) -- with
    ``executed=False``. It never calls a write tool.
    """

    def __init__(self, err=0.15, tok_per_step=None):
        self.err = err
        self.tok_per_step = (tok_per_step if tok_per_step is not None
                             else TOKEN_COSTS["weak_decode_per_step"])
        self.decoded_tokens = 0

    def reset_counters(self):
        self.decoded_tokens = 0

    def draft(self, world, rng, max_reads=3):
        steps = []
        n_reads = 1
        if rng.random() < 0.30:
            n_reads += 1
        if rng.random() < 0.10:
            n_reads += 1
        n_reads = min(n_reads, max_reads)
        for _ in range(n_reads):
            radius = rng.choice((2, 3, 4))
            obs = world.scan(radius, executor="weak")
            self.decoded_tokens += self.tok_per_step
            steps.append(Step("scan", (radius,), READ, observation=obs,
                              executed=True, executor="weak"))
        steps.append(self._choose_write(world, steps, rng))
        self.decoded_tokens += self.tok_per_step
        return steps

    def _choose_write(self, world, steps, rng):
        """Held write: correct greedy action, or a noisy wrong one."""
        if rng.random() < self.err:
            return self._wrong_action(world, rng)
        return self._correct_action(world, steps)

    def _correct_action(self, world, steps):
        if world.is_done():
            return Step("done", (), TERMINAL)
        if world.pos in world.uncollected():
            return Step("collect", (), WRITE)
        obs = steps[-1].observation if steps else ()
        if obs:
            x0, y0 = world.pos
            dest = min(obs, key=lambda c: (abs(c[0] - x0) + abs(c[1] - y0), c))
            return Step("goto", dest, WRITE)
        x0, y0 = world.pos
        nbrs = [(x0 + dx, y0 + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= x0 + dx < world.size and 0 <= y0 + dy < world.size]
        dest = nbrs[0] if nbrs else (x0, y0)
        return Step("goto", dest, WRITE)

    def _wrong_action(self, world, rng):
        """A plausible-but-wrong held action (ground-truth invalid)."""
        x0, y0 = world.pos
        far = [(x, y) for x in range(world.size) for y in range(world.size)
               if abs(x - x0) + abs(y - y0) > 2
               and (x, y) not in world.uncollected()]
        options = []
        if far:
            options.append(Step("goto", rng.choice(far), WRITE))
        if world.pos not in world.uncollected():
            options.append(Step("collect", (), WRITE))  # invalid: nothing here
        if not world.is_done():
            options.append(Step("done", (), TERMINAL))  # premature done
        if not options:
            return Step("goto", (x0, y0), WRITE)
        return rng.choice(options)


class StrongPolicy:
    """Accurate, expensive model: verifier, repairer, and step author.

    ``verify`` scores a whole segment in *one prefill pass* (cheap parallel
    prefill tokens) and decodes only a short verdict per step; ``repair``
    decodes a replacement for the first rejected step; ``author_step`` is the
    strong-only baseline (full decode per step).
    """

    def __init__(self, miss=0.04, false_alarm=0.03, repair_fail=0.03,
                 author_err=0.0):
        self.miss = miss                # P(accept | step truly invalid)
        self.false_alarm = false_alarm  # P(reject | step truly valid)
        self.repair_fail = repair_fail  # P(repair emits a wrong action)
        self.author_err = author_err    # P(authored step is wrong)
        self.decoded_tokens = 0
        self.prefill_tokens = 0

    def reset_counters(self):
        self.decoded_tokens = 0
        self.prefill_tokens = 0

    def verify(self, segment, world, rng):
        self.prefill_tokens += len(segment) * TOKEN_COSTS["strong_prefill_per_step"]
        verdicts = []
        for step in segment:
            ok = self._ground_truth_valid(step, world)
            r = rng.random()
            if ok and r < self.false_alarm:
                ok = False
            elif not ok and r < self.miss:
                ok = True
            step.verdict = ok
            verdicts.append(ok)
        # short verdicts decoded, not full steps
        self.decoded_tokens += len(segment) * TOKEN_COSTS["strong_verdict_per_step"]
        repair = None
        for step, ok in zip(segment, verdicts):
            if not ok:
                repair = self._repair(step, world, rng)
                break
        return verdicts, repair

    def _ground_truth_valid(self, step, world):
        if step.kind == READ:
            return True  # reads really executed; always grounded
        if step.name == "done":
            return world.is_done()
        if step.name == "goto":
            (x, y) = step.args
            x0, y0 = world.pos
            on_target = (x, y) in world.uncollected()
            local_explore = abs(x - x0) + abs(y - y0) <= 2
            return on_target or local_explore
        if step.name == "collect":
            return world.pos in world.uncollected()
        return False

    def _repair(self, bad_step, world, rng):
        self.decoded_tokens += TOKEN_COSTS["strong_repair"]
        if rng.random() < self.repair_fail:
            noisy = WeakPolicy(err=1.0)
            return noisy._wrong_action(world, rng)
        if bad_step.kind == READ:
            # re-issue the read (grounded); the orchestrator executes it
            return Step(bad_step.name, bad_step.args, READ)
        return self._correct_action(world)

    def _correct_action(self, world):
        if world.is_done():
            return Step("done", (), TERMINAL)
        if world.pos in world.uncollected():
            return Step("collect", (), WRITE)
        x0, y0 = world.pos
        dest = min(world.uncollected(),
                   key=lambda c: (abs(c[0] - x0) + abs(c[1] - y0), c))
        return Step("goto", dest, WRITE)

    def author_step(self, world, rng):
        self.decoded_tokens += TOKEN_COSTS["strong_decode_per_step"]
        if rng.random() < self.author_err:
            noisy = WeakPolicy(err=1.0)
            return noisy._wrong_action(world, rng)
        return self._correct_action(world)


def _execute(step, world, executor):
    """Execute one committed step via the orchestrator (never the weak model)."""
    if step.kind == READ:
        obs = world.scan(*step.args, executor=executor)
        step.observation = obs
        step.executed = True
        step.executor = executor
    elif step.kind == WRITE:
        if step.name == "goto":
            world.goto(*step.args, executor=executor)
        elif step.name == "collect":
            world.collect(executor=executor)
        step.executed = True
        step.executor = executor
    # TERMINAL "done": no environment call; handled by the run loop.
    return step


class SpecAgent:
    """The SpecAgent protocol loop: draft -> verify -> commit prefix -> repair."""

    def __init__(self, weak, strong, max_segments=50):
        self.weak = weak
        self.strong = strong
        self.max_segments = max_segments

    def run(self, world, rng):
        commits = []       # (segment_id, step_index, verdict) -- verdict True
        draft_reads = []   # (executed, has_observation, executor) per draft read
        n_repairs = 0
        n_verdict_steps = 0
        seg_id = 0
        terminated = False
        while (not world.is_done() and not terminated
               and seg_id < self.max_segments):
            segment = self.weak.draft(world, rng)
            for s in segment:
                if s.kind == READ:
                    draft_reads.append((s.executed, s.observation is not None,
                                        s.executor))
            verdicts, repair = self.strong.verify(segment, world, rng)
            n_verdict_steps += len(segment)
            # commit the longest valid prefix
            for i, (step, ok) in enumerate(zip(segment, verdicts)):
                if not ok:
                    break
                if step.kind in (READ, WRITE):
                    _execute(step, world, executor="orchestrator")
                elif step.name == "done":
                    terminated = True
                commits.append((seg_id, i, ok))
            # repair the first rejected step (if any)
            if repair is not None:
                n_repairs += 1
                if repair.kind in (READ, WRITE):
                    _execute(repair, world, executor="orchestrator")
                elif repair.name == "done" and world.is_done():
                    terminated = True
            seg_id += 1
        return {
            "success": world.is_done(),
            "terminated_by_done": terminated,
            "n_segments": seg_id,
            "n_commits": len(commits),
            "n_repairs": n_repairs,
            "n_verdict_steps": n_verdict_steps,
            "strong_decoded": self.strong.decoded_tokens,
            "strong_prefill": self.strong.prefill_tokens,
            "weak_decoded": self.weak.decoded_tokens,
            "write_log": list(world.write_log),
            "commits": commits,
            "draft_reads": draft_reads,
        }


def run_strong_only(world, strong, rng, max_steps=60):
    """Baseline: the strong model authors (and the orchestrator executes)
    every step autoregressively."""
    steps = 0
    while not world.is_done() and steps < max_steps:
        step = strong.author_step(world, rng)
        if step.name == "done":
            break
        _execute(step, world, executor="orchestrator")
        steps += 1
    return {
        "success": world.is_done(),
        "n_steps": steps,
        "strong_decoded": strong.decoded_tokens,
        "strong_prefill": strong.prefill_tokens,
        "weak_decoded": 0,
    }


def run_weak_only(world, weak, rng, max_steps=60):
    """Control: the weak model authors and executes every step itself
    (including writes) -- no verification."""
    steps = 0
    while not world.is_done() and steps < max_steps:
        radius = rng.choice((2, 3, 4))
        obs = world.scan(radius, executor="weak")  # weak's own grounding
        weak.decoded_tokens += weak.tok_per_step
        scan_step = Step("scan", (radius,), READ, observation=obs,
                         executed=True, executor="weak")
        step = weak._choose_write(world, [scan_step], rng)
        weak.decoded_tokens += weak.tok_per_step
        if step.name == "done":
            break
        _execute(step, world, executor="weak")
        steps += 1
    return {
        "success": world.is_done(),
        "n_steps": steps,
        "strong_decoded": 0,
        "strong_prefill": 0,
        "weak_decoded": weak.decoded_tokens,
    }


def _make_task(rng, grid, n_targets):
    cells = [(x, y) for x in range(grid) for y in range(grid)
             if (x, y) != (0, 0)]
    return rng.sample(cells, n_targets)


def evaluate(n_tasks=64, seed=7, n_targets=4, grid=6):
    """Run all three protocols on the same seeded task set.

    Returns (summary, episodes): summary holds aggregate rates, mean token
    counts and invariant-violation counts; episodes holds per-task metrics.
    Deterministic given ``seed``.
    """
    episodes = []
    for i in range(n_tasks):
        rng = __import__("random").Random(seed * 100003 + i)
        targets = _make_task(rng, grid, n_targets)

        weak = WeakPolicy()
        strong = StrongPolicy()
        agent = SpecAgent(weak, strong)
        spec = agent.run(ToolWorld(grid, targets), rng)

        strong2 = StrongPolicy()
        strong_only = run_strong_only(ToolWorld(grid, targets), strong2, rng)

        weak3 = WeakPolicy()
        weak_only = run_weak_only(ToolWorld(grid, targets), weak3, rng)

        episodes.append({"task": i, "targets": targets, "spec": spec,
                         "strong_only": strong_only, "weak_only": weak_only})

    def mean(key, proto):
        vals = [e[proto][key] for e in episodes]
        return sum(vals) / len(vals)

    violations = {"weak_executed_write": 0,
                  "commit_without_verdict": 0,
                  "ungrounded_read": 0}
    for e in episodes:
        spec = e["spec"]
        for executor, tool in spec["write_log"]:
            if executor == "weak" and tool in ToolWorld.WRITE_TOOLS:
                violations["weak_executed_write"] += 1
        for _seg, _idx, verdict in spec["commits"]:
            if verdict is not True:
                violations["commit_without_verdict"] += 1
        for executed, has_obs, executor in spec["draft_reads"]:
            if not (executed and has_obs and executor == "weak"):
                violations["ungrounded_read"] += 1

    s_rate = lambda p: sum(1 for e in episodes if e[p]["success"]) / n_tasks
    strong_tok = mean("strong_decoded", "strong_only")
    summary = {
        "n_tasks": n_tasks,
        "seed": seed,
        "strong_success": s_rate("strong_only"),
        "spec_success": s_rate("spec"),
        "weak_success": s_rate("weak_only"),
        "mean_strong_decoded": {
            "strong_only": strong_tok,
            "spec": mean("strong_decoded", "spec"),
        },
        "strong_token_ratio": (mean("strong_decoded", "spec") / strong_tok
                               if strong_tok else float("nan")),
        "mean_strong_prefill": {"spec": mean("strong_prefill", "spec")},
        "mean_weak_decoded": {
            "spec": mean("weak_decoded", "spec"),
            "weak_only": mean("weak_decoded", "weak_only"),
        },
        "mean_segments_per_episode": mean("n_segments", "spec"),
        "mean_repairs_per_episode": mean("n_repairs", "spec"),
        "mean_verdict_steps_per_episode": mean("n_verdict_steps", "spec"),
        "mean_steps_strong_only": mean("n_steps", "strong_only"),
        "invariant_violations": violations,
    }
    return summary, episodes
