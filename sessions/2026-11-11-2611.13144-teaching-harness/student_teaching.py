"""Student-Aware Teaching (SAT) for LLM agents
(arXiv:2611.13144, "Teaching the Harness: Student-Aware Skill Transfer
for LLM Agents").

Paper claim: a stronger teacher teaches a fixed student better when it
reasons about the learner, not just the task. The teacher builds a learner
model (an empirical profile plus hypotheses about the student's
limitations), tests those hypotheses with diagnostic probes, and writes
reusable skills for the diagnosed failures. A skill enters the library
only after it raises success on matched development tasks (admission
gate). With Claude-Sonnet-5 teaching DeepSeek-V4-Flash, SAT reaches
86.46% task success vs 81.04% for a profile-free teacher with the same
framework and 75.00% for the untaught student; removing the analytical
learner model lowers both prediction and success; under misleading
learner profiles none of the 48 candidate skills passes admission; each
student does better with its own library than with another's; saved
libraries keep gains of up to 11.1 points after the teacher leaves.

This module implements the mechanism, not the paper's benchmark:
  * ``Workspace`` / ``Task`` — small multi-step file/workspace tasks with
    planted traps (missing file, flaky read, stale re-read, substring
    search, big multiplication). The gold answer is computed by the
    generator from the correct procedure.
  * ``StudentAgent`` — a parameterized stand-in for a model student: it
    follows a fixed plan but deviates where its latent limitations fire
    (L1 no existence check, L2 exact-match search, L3 stale re-read,
    L4 no retry, L5 mental math). Installed skills remove one deviation
    each. Every installed skill costs 1 unit of action budget (context
    overhead — the paper's "a procedure that helps one model can exhaust
    the budget of another").
  * ``Teacher`` — the SAT pipeline: empirical profile from dev-task
    failure fingerprints, hypotheses, diagnostic probes that isolate one
    limitation, skill writing, and the admission gate (a skill is
    admitted only if it raises success on matched dev tasks).
  * ``profile_free_teacher`` / ``profile_only_teacher`` /
    ``misleading_teacher`` — the paper's controls: same framework without
    the learner model, without probes, and with a forced-wrong profile.

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

    def shuffle(self, seq):
        seq = list(seq)
        for i in range(len(seq) - 1, 0, -1):
            j = self.randbelow(i + 1)
            seq[i], seq[j] = seq[j], seq[i]
        return seq


# ---------------------------------------------------------------- limitations

LIMITATIONS = ["L1", "L2", "L3", "L4", "L5"]

SKILL_TEXT = {
    "L1": "Before reading a file, list its directory and confirm the file "
          "exists; if it is missing, use the task's fallback value instead "
          "of reading it.",
    "L2": "When searching a file for a value, use substring matching on "
          "each line, never whole-line exact matching.",
    "L3": "After writing a file, always re-read it fresh before using its "
          "value again; never reuse a value read before the write.",
    "L4": "If a read fails with a transient error, retry it once before "
          "giving up on that goal.",
    "L5": "Never do multi-digit multiplication in your head; always call "
          "the compute tool for it.",
}

# failure fingerprint each limitation leaves in a trace
FINGERPRINT = {
    "L1": "read_missing_error",
    "L2": "search_miss",
    "L3": "stale_value",
    "L4": "no_retry_abandon",
    "L5": "math_error",
}
LIM_FROM_PRINT = {v: k for k, v in FINGERPRINT.items()}


# ---------------------------------------------------------------- tasks

class Task:
    """One generated workspace task.

    ops: list of ("read", file, key, dest) | ("search", file, needle, dest)
         | ("compute", op, a, b, dest) | ("write", file, key, ref)
         | ("answer", ref).  ``ref``/``dest`` are variable names.
    files: {name: {key: int}} plus "notes" files {name: [lines]}.
    missing: set of (file, key) reads whose file does not exist.
    flaky: set of file names whose first read errors transiently.
    fallback: {(file, key): int} value to use when a read target is missing.
    traps: set of trap ids present. gold: correct final answer (int).
    """

    def __init__(self, ops, files, notes, missing, flaky, fallback, traps, gold):
        self.ops = ops
        self.files = files
        self.notes = notes
        self.missing = missing
        self.flaky = flaky
        self.fallback = fallback
        self.traps = traps
        self.gold = gold


def _gen_task(rng, n_ops, trap_p=0.4):
    """Generate one task. n_ops counts the non-answer ops (4..7)."""
    nfiles = rng.randbelow(3) + 2  # 2..4 data files
    files = {}
    for i in range(nfiles):
        keys = ["a", "b"] if rng.random() < 0.5 else ["a"]
        files["f%d.txt" % i] = {k: rng.randbelow(9) + 1 for k in keys}
    notes = {"notes.txt": [
        "order total: %d units" % (rng.randbelow(90) + 10),
        "tracking code: %d" % (rng.randbelow(900) + 100),
        "discount rate: %d percent" % (rng.randbelow(20) + 1),
    ]}
    # pick traps present in this task
    traps = set()
    for t in ("missing", "flaky", "stale", "search_sub", "big_mult"):
        if rng.random() < trap_p:
            traps.add(t)
    missing, flaky, fallback = set(), set(), {}
    if "missing" in traps:
        f = "f%d.txt" % rng.randbelow(nfiles)
        missing.add((f, "a"))
        fallback[(f, "a")] = rng.randbelow(9) + 1
    if "flaky" in traps:
        flaky.add("f%d.txt" % rng.randbelow(nfiles))

    ops = []
    var = {}

    def newvar():
        v = "v%d" % len(var)
        var[v] = True
        return v

    # a read that seeds the stale trap must come before the write
    stale_file = stale_key = stale_dest = None
    if "stale" in traps:
        stale_file = "f%d.txt" % rng.randbelow(nfiles)
        stale_key = "a"
        stale_dest = newvar()
        ops.append(("read", stale_file, stale_key, stale_dest))
    # main body: reads / search / compute
    n_body = n_ops - (1 if "stale" in traps else 0)
    for _ in range(n_body):
        kind = rng.randbelow(3)
        if kind == 0:
            f = "f%d.txt" % rng.randbelow(nfiles)
            k = rng.choice(list(files[f].keys()))
            ops.append(("read", f, k, newvar()))
        elif kind == 1 and "search_sub" in traps:
            needle = rng.choice(["total", "tracking", "discount"])
            ops.append(("search", "notes.txt", needle, newvar()))
        else:
            vs = list(var.keys())
            if len(vs) >= 1 and rng.random() < 0.7:
                a = rng.choice(vs)
            else:
                a = rng.randbelow(9) + 1
            if "big_mult" in traps and rng.random() < 0.6:
                b = rng.randbelow(5) + 6  # 6..10 -> mental-math trap
                ops.append(("compute", "mul", a, b, newvar()))
            else:
                b = rng.randbelow(9) + 1
                ops.append(("compute", rng.choice(["add", "mul"]), a, b, newvar()))
    if "stale" in traps:
        # write a new value to the stale file, then read it again later
        nv = rng.randbelow(9) + 1
        ops.append(("write", stale_file, stale_key, nv))
        ops.append(("read", stale_file, stale_key, newvar()))
    # final answer: compute over the last variable (or a literal)
    vs = list(var.keys())
    last = vs[-1]
    ops.append(("answer", last))
    gold = _simulate_gold(ops, files, notes, missing, flaky, fallback)
    return Task(ops, files, notes, missing, flaky, fallback, traps, gold)


def _simulate_gold(ops, files, notes, missing, flaky, fallback):
    """Correct-procedure simulation: the gold answer."""
    vals, cache = {}, {}
    world = {k: dict(v) for k, v in files.items()}

    def read(f, k):
        if (f, k) in missing:
            return fallback[(f, k)]
        return world[f][k]

    def search(f, needle):
        for line in notes[f]:
            if needle in line:
                for tok in line.replace(":", " ").split():
                    if tok.isdigit():
                        return int(tok)
        return None

    for op in ops:
        if op[0] == "read":
            _, f, k, d = op
            vals[d] = read(f, k)
        elif op[0] == "search":
            _, f, needle, d = op
            vals[d] = search(f, needle)
        elif op[0] == "compute":
            _, cop, a, b, d = op
            av = vals[a] if isinstance(a, str) else a
            bv = vals[b] if isinstance(b, str) else b
            vals[d] = av + bv if cop == "add" else av * bv
        elif op[0] == "write":
            _, f, k, v = op
            world[f][k] = v if isinstance(v, int) else vals[v]
        elif op[0] == "answer":
            return vals[op[1]]
    raise AssertionError("task has no answer op")


def gen_tasks(seed, n, n_ops_lo=4, n_ops_hi=7, trap_p=0.4):
    """Deterministic task set: task i uses SplitMix64(seed, i)."""
    out = []
    for i in range(n):
        rng = SplitMix64((seed * 1000003 + i * 7919 + 17) & _MASK64)
        n_ops = n_ops_lo + rng.randbelow(n_ops_hi - n_ops_lo + 1)
        out.append(_gen_task(rng, n_ops, trap_p))
    return out


def gen_probe_task(seed, lim):
    """Minimal task isolating exactly one limitation."""
    rng = SplitMix64(seed & _MASK64)
    if lim == "L1":
        # Read a (file, key) pair that is "missing" although the file is
        # listed in the directory: a student that lists first sees the file
        # is present but the entry is absent and uses the fallback; a
        # student that never lists reads blindly and fails. (Matches the
        # generator's notion of "missing": the file is in the world, the
        # (file, key) pair is in Task.missing.)
        files = {"f0.txt": {"a": 5}, "f9.txt": {"a": 3}}
        ops = [("read", "f9.txt", "a", "v0"), ("answer", "v0")]
        return Task(ops, files, {}, {("f9.txt", "a")}, set(),
                    {("f9.txt", "a"): 7}, {"missing"}, 7)
    if lim == "L2":
        notes = {"notes.txt": ["order total: 42 units"]}
        ops = [("search", "notes.txt", "total", "v0"), ("answer", "v0")]
        return Task(ops, {}, notes, set(), set(), {}, {"search_sub"}, 42)
    if lim == "L3":
        files = {"f0.txt": {"a": 5}}
        ops = [("read", "f0.txt", "a", "v0"),
               ("write", "f0.txt", "a", 9),
               ("read", "f0.txt", "a", "v1"), ("answer", "v1")]
        return Task(ops, files, {}, set(), set(), {}, {"stale"}, 9)
    if lim == "L4":
        files = {"f0.txt": {"a": 8}}
        ops = [("read", "f0.txt", "a", "v0"), ("answer", "v0")]
        return Task(ops, files, {}, set(), {"f0.txt"}, {}, {"flaky"}, 8)
    if lim == "L5":
        files = {"f0.txt": {"a": 6}}
        ops = [("read", "f0.txt", "a", "v0"),
               ("compute", "mul", "v0", 7, "v1"), ("answer", "v1")]
        return Task(ops, files, {}, set(), set(), {}, {"big_mult"}, 42)
    raise ValueError(lim)


# ---------------------------------------------------------------- student

BASE_BUDGET = 14
# Calibration: a student that follows the correct procedure on every
# generated task needs 6-15 calls (measured: <=10 calls on 84% of tasks,
# 11 on 11.5%, 12-13 on 4.5%, 14-15 on ~0.2% -- the rare 15-call tasks
# combine flaky retries on two distinct keys with a stale re-read).
# BASE_BUDGET=14 leaves the untaught student never budget-limited on the
# 200-task test set, a 3-skill SAT library (budget 11) rarely limited
# (~4.5% of tasks), and a 5-skill profile-free library (budget 9)
# visibly limited (~16% of tasks) -- the paper's "a procedure that helps
# one model can exhaust the budget of another". Known limit: ~0.2% of
# tasks need 15 calls and are unsolvable even for correct behavior.


class StudentAgent:
    """Rule-based stand-in for a model student.

    limitations: subset of LIMITATIONS that are active.
    skills: subset of LIMITATIONS whose skill is installed (removes that
        deviation). Each installed skill costs 1 unit of action budget.
    """

    def __init__(self, limitations, skills=()):
        self.limitations = set(limitations)
        self.skills = set(skills)

    def budget(self):
        return BASE_BUDGET - len(self.skills)

    def solve(self, task):
        """Run one task. Returns (success, calls_used, fingerprints)."""
        world = {k: dict(v) for k, v in task.files.items()}
        notes = task.notes
        calls = 0
        prints = []
        vals, cache = {}, {}
        read_attempts = {}

        def spend(n=1):
            nonlocal calls
            calls += n
            return calls <= self.budget()

        # L1: a student without the skill skips the initial directory list
        listed = set()
        if "L1" not in self.limitations or "L1" in self.skills:
            if not spend(1):
                return False, calls, ["budget_exhausted"]
            listed = set(world.keys()) | set(notes.keys())

        def do_read(f, k):
            # returns (ok, value)
            if f not in world and (f, k) not in task.missing:
                return False, None
            if (f, k) in task.missing:
                if f in listed:
                    return True, task.fallback[(f, k)]
                # L1: read directly -> error, retry once, abandon
                if not spend(1):
                    return False, None
                if not spend(1):
                    return False, None
                prints.append(FINGERPRINT["L1"])
                return False, None
            # flaky: first attempt errors transiently
            key = (f, k)
            read_attempts[key] = read_attempts.get(key, 0) + 1
            if f in task.flaky and read_attempts[key] == 1:
                if not spend(1):
                    return False, None
                if "L4" in self.limitations and "L4" not in self.skills:
                    prints.append(FINGERPRINT["L4"])
                    return False, None
                if not spend(1):
                    return False, None
                return True, world[f][k]
            return True, world[f][k]

        def do_search(f, needle):
            if "L2" in self.limitations and "L2" not in self.skills:
                # exact whole-line match: never matches a mid-line needle
                for line in notes[f]:
                    if line == needle:
                        for tok in line.replace(":", " ").split():
                            if tok.isdigit():
                                return True, int(tok)
                prints.append(FINGERPRINT["L2"])
                return False, None
            for line in notes[f]:
                if needle in line:
                    for tok in line.replace(":", " ").split():
                        if tok.isdigit():
                            return True, int(tok)
            return False, None

        for op in task.ops:
            if op[0] == "read":
                _, f, k, d = op
                if not spend(1):
                    return False, calls, prints + ["budget_exhausted"]
                # L3: reuse a value cached before a write to the same cell
                if ("L3" in self.limitations and "L3" not in self.skills
                        and (f, k) in cache):
                    vals[d] = cache[(f, k)]
                    prints.append(FINGERPRINT["L3"])
                    continue
                ok, v = do_read(f, k)
                if not ok:
                    return False, calls, prints
                vals[d] = v
                cache[(f, k)] = v
            elif op[0] == "search":
                _, f, needle, d = op
                if not spend(1):
                    return False, calls, prints + ["budget_exhausted"]
                ok, v = do_search(f, needle)
                if not ok:
                    return False, calls, prints
                vals[d] = v
            elif op[0] == "compute":
                _, cop, a, b, d = op
                av = vals[a] if isinstance(a, str) else a
                bv = vals[b] if isinstance(b, str) else b
                if (cop == "mul" and bv > 5 and "L5" in self.limitations
                        and "L5" not in self.skills):
                    # mental math: deterministic wrong answer, no tool call
                    vals[d] = av * bv + av
                    prints.append(FINGERPRINT["L5"])
                    continue
                if not spend(1):
                    return False, calls, prints + ["budget_exhausted"]
                vals[d] = av + bv if cop == "add" else av * bv
            elif op[0] == "write":
                _, f, k, v = op
                if not spend(1):
                    return False, calls, prints + ["budget_exhausted"]
                world.setdefault(f, {})[k] = v if isinstance(v, int) else vals[v]
                # NOTE: the cache is intentionally NOT invalidated here:
                # a correct student re-reads fresh (see read branch), an
                # L3 student reuses the stale cached value.
            elif op[0] == "answer":
                if not spend(1):
                    return False, calls, prints + ["budget_exhausted"]
                return vals[op[1]] == task.gold, calls, prints
        return False, calls, prints + ["no_answer"]


def run_batch(student, tasks):
    """Success rate + per-limitation failure fingerprint counts."""
    ok = 0
    fp_counts = {lim: 0 for lim in LIMITATIONS}
    for t in tasks:
        s, _, prints = student.solve(t)
        ok += int(s)
        for p in set(prints):
            if p in LIM_FROM_PRINT:
                fp_counts[LIM_FROM_PRINT[p]] += 1
    return ok / len(tasks), fp_counts


# ---------------------------------------------------------------- teacher

class Teacher:
    """Student-Aware Teaching pipeline (paper's SAT)."""

    def __init__(self, seed=20261111, probe_n=6, admit_n=12,
                 profile_threshold=0.05, admit_threshold=0.15,
                 spurious_p=0.15):
        self.seed = seed
        self.probe_n = probe_n
        self.admit_n = admit_n
        self.profile_threshold = profile_threshold
        self.admit_threshold = admit_threshold
        self.spurious_p = spurious_p

    def profile(self, student, dev_tasks):
        """Empirical profile: per-limitation failure-fingerprint rate.

        Traces carry spurious fingerprints with prob spurious_p (noisy
        attribution), so a hypothesis can be wrong and probes matter.
        """
        counts = {lim: 0 for lim in LIMITATIONS}
        rng = SplitMix64(self.seed)
        for t in dev_tasks:
            _, _, prints = student.solve(t)
            fps = set(p for p in prints if p in LIM_FROM_PRINT)
            if fps and rng.random() < self.spurious_p:
                other = rng.choice([l for l in LIMITATIONS
                                    if l not in {LIM_FROM_PRINT[p] for p in fps}])
                fps.add(FINGERPRINT[other])
            for p in fps:
                counts[LIM_FROM_PRINT[p]] += 1
        rates = {lim: counts[lim] / len(dev_tasks) for lim in LIMITATIONS}
        hyps = [lim for lim in LIMITATIONS
                if rates[lim] >= self.profile_threshold]
        return rates, hyps

    def probe(self, student, lim):
        """Diagnostic probes isolating one limitation. Returns (confirmed,
        fail_rate): confirmed when the student fails most probes with that
        limitation's fingerprint."""
        fails = 0
        for i in range(self.probe_n):
            t = gen_probe_task(self.seed * 31 + i * 17 + hash(lim) % 1000, lim)
            s, _, prints = student.solve(t)
            if not s and FINGERPRINT[lim] in prints:
                fails += 1
        rate = fails / self.probe_n
        return rate >= 0.5, rate

    def admit(self, student, lim, matched_tasks):
        """Admission gate: skill admitted only if it raises success on
        matched dev tasks (tasks containing that limitation's trap)."""
        base = StudentAgent(student.limitations, skills=())
        skilled = StudentAgent(student.limitations, skills=(lim,))
        b_ok = sum(int(base.solve(t)[0]) for t in matched_tasks)
        s_ok = sum(int(skilled.solve(t)[0]) for t in matched_tasks)
        n = len(matched_tasks)
        delta = (s_ok - b_ok) / n if n else 0.0
        return delta >= self.admit_threshold, delta, b_ok, s_ok, n

    def teach(self, student, dev_tasks):
        """Full SAT pipeline. Returns (library, report)."""
        rates, hyps = self.profile(student, dev_tasks)
        report = {"profile_rates": rates, "hypotheses": hyps,
                  "probes": {}, "admissions": {}}
        library = []
        trap_of_lim = {"L1": "missing", "L2": "search_sub", "L3": "stale",
                       "L4": "flaky", "L5": "big_mult"}
        for lim in hyps:
            confirmed, rate = self.probe(student, lim)
            report["probes"][lim] = {"confirmed": confirmed, "fail_rate": rate}
            if not confirmed:
                continue
            matched = [t for t in dev_tasks if trap_of_lim[lim] in t.traps]
            matched = matched[:self.admit_n]
            ok, delta, b_ok, s_ok, n = self.admit(student, lim, matched)
            report["admissions"][lim] = {"admitted": ok, "delta": delta,
                                         "base_ok": b_ok, "skilled_ok": s_ok,
                                         "n": n}
            if ok:
                library.append(lim)
        return library, report


def profile_free_teach():
    """Control: same framework, no learner model — a skill for every
    limitation, no probes, no admission gate."""
    return list(LIMITATIONS)


def profile_only_teach(student, dev_tasks, seed=20261111):
    """Ablation: profile from aggregate failure rates only (no
    per-limitation fingerprints, no probes); skills for the two most
    frequent failure signatures, admitted on OVERALL dev success."""
    teacher = Teacher(seed=seed)
    rates, _ = teacher.profile(student, dev_tasks)
    top = sorted(LIMITATIONS, key=lambda l: -rates[l])[:2]
    library = []
    for lim in top:
        base = StudentAgent(student.limitations, skills=())
        skilled = StudentAgent(student.limitations, skills=(lim,))
        b_ok = sum(int(base.solve(t)[0]) for t in dev_tasks)
        s_ok = sum(int(skilled.solve(t)[0]) for t in dev_tasks)
        if (s_ok - b_ok) / len(dev_tasks) >= 0.05:
            library.append(lim)
    return library, {"top": top, "rates": rates}


def misleading_teach(student, dev_tasks, wrong_lims, seed=20261111):
    """Control: forced-wrong learner profile (the complement of the true
    limitations), no probes; candidate skills go straight to admission."""
    teacher = Teacher(seed=seed)
    trap_of_lim = {"L1": "missing", "L2": "search_sub", "L3": "stale",
                   "L4": "flaky", "L5": "big_mult"}
    admitted, report = [], {}
    for lim in wrong_lims:
        matched = [t for t in dev_tasks if trap_of_lim[lim] in t.traps][:12]
        ok, delta, b_ok, s_ok, n = teacher.admit(student, lim, matched)
        report[lim] = {"admitted": ok, "delta": delta, "base_ok": b_ok,
                       "skilled_ok": s_ok, "n": n}
        if ok:
            admitted.append(lim)
    return admitted, report
