"""Privilege separation for coding-agent benchmarks (SplitBench).

Reproduces the core mechanism of arXiv:2611.27860 ("Who Grades the
Agent? Privilege Separation for Coding-Agent Benchmarks", Houlamy,
Favretto, Fernandes & Conti, 2026):

* Evaluating a coding agent requires running its code, and that code can
  manipulate the evaluation infrastructure to forge a pass. An automated
  benchmark auditor (BenchJack) reached near-perfect scores on nine of
  ten popular agent benchmarks without solving their tasks.
* SplitBench separates running a task's tests from judging their
  results: the agent's code runs in an unprivileged sandbox that never
  receives the expected answers, and a trusted grader -- which alone
  holds the answers -- judges only the outputs the sandbox returns.
  Even an agent whose code takes over the entire test environment cannot
  forge a pass, because the trusted grader holds the expected answers.
* Paper's measured claims: 180 of 202 tasks convert to the framework;
  SplitBench agrees with the native graders on 98.7% of agent attempts;
  under worst-case attack (adversary reads all grading code but not the
  expected answers) it blocks 75% of attempts, the rest exposing
  weaknesses in the original tasks' tests rather than in SplitBench.

This module implements the mechanism at bounded scale with a miniature
benchmark of pure-function tasks (JSON-serialisable inputs/outputs):

* ``NativeGrader`` -- the vulnerable setup: the agent's solution module
  is imported in the same process that holds the expected answers and
  the test file, so import-time payload code can read the answers,
  rewrite the tests, or monkeypatch the verdict function.
* ``SplitGrader`` -- the separated setup: the solution runs in a
  sandboxed subprocess that receives only the task files and the hidden
  inputs at grade time; the trusted grader (this process) holds the
  expected answers in memory only, validates the returned outputs'
  schema, and compares them. The sandbox is handed a copy of the full
  grading code (the paper's worst case) but never the answers.
* ``values_equal`` -- type-strict output comparison (bool is not int,
  numeric tolerance 1e-9).
* ``classify_attempt`` -- classifies a sandbox audit log: whether any
  recorded probe could have reached the expected answers.

Scope / stated limits of this CPU model (see also the session README):
the separation is modelled at the process + directory level -- the
sandbox is a subprocess whose working directory contains only task
files, and the trusted grader never writes the answers to disk. A
production deployment needs OS-level isolation (separate UID /
container / VM) on top of this layout, which is outside this CPU test.
The test asserts what the layout guarantees: no answers file exists in
the sandbox, every probe the payloads attempt fails, and the trusted
verdict is computed from validated outputs only.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

PASS = "PASS"
FAIL = "FAIL"

# ----------------------------------------------------------------------
# Pure helpers (also ported to Rust; keep them side-effect free)
# ----------------------------------------------------------------------

def values_equal(got, exp, tol=1e-9):
    """Type-strict equality for JSON values.

    bool is never equal to int (True != 1 here, unlike ``==``);
    numbers compare within ``tol``; lists/dicts compare element-wise;
    everything else compares with ``==``.
    """
    if isinstance(exp, bool) or isinstance(got, bool):
        return isinstance(got, bool) and isinstance(exp, bool) and got == exp
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(exp)) <= tol
    if isinstance(exp, str) and isinstance(got, str):
        return got == exp
    if exp is None or got is None:
        return got is None and exp is None
    if isinstance(exp, list) and isinstance(got, list):
        return len(got) == len(exp) and all(
            values_equal(g, e, tol) for g, e in zip(got, exp))
    if isinstance(exp, dict) and isinstance(got, dict):
        return set(got) == set(exp) and all(
            values_equal(got[k], exp[k], tol) for k in exp)
    return got == exp


def validate_outputs(outputs, n_expected):
    """Check a sandbox-returned outputs payload.

    Returns (ok, reason). The payload must be a JSON list of exactly
    ``n_expected`` JSON values -- no extra keys, no missing entries,
    no non-JSON values. Anything else is rejected before comparison,
    so a forged / malformed payload can never read as a pass.
    """
    if not isinstance(outputs, list):
        return False, "outputs is not a list"
    if len(outputs) != n_expected:
        return False, "wrong number of outputs: %d != %d" % (
            len(outputs), n_expected)
    try:
        json.dumps(outputs)
    except (TypeError, ValueError):
        return False, "outputs not JSON-serialisable"
    return True, "ok"


def grade_outputs(outputs, expected):
    """Trusted verdict from validated outputs vs in-memory answers.

    Returns (verdict, n_correct). Callers must run
    ``validate_outputs`` first; this function assumes it passed.
    """
    n_correct = sum(
        1 for got, exp in zip(outputs, expected)
        if values_equal(got, exp))
    return (PASS if n_correct == len(expected) else FAIL), n_correct


def classify_attempt(probes):
    """Classify a sandbox audit log (list of probe dicts).

    Each probe is {"path": ..., "found": bool, "kind": ...}. Returns a
    dict: whether any probe reached the expected answers
    (``answers_reached``), whether any probe rewrote grading code that
    the trusted verdict depends on (``verdict_tampered`` -- always
    False here: the trusted verdict runs in the parent process, and the
    sandbox cannot touch it), and the count of failed probes.
    """
    answers_reached = any(
        p.get("found") and p.get("kind") == "answers" for p in probes)
    verdict_tampered = any(
        p.get("found") and p.get("kind") == "verdict" for p in probes)
    return {
        "answers_reached": answers_reached,
        "verdict_tampered": verdict_tampered,
        "n_probes": len(probes),
        "n_failed": sum(1 for p in probes if not p.get("found")),
    }


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------

class Task:
    """A pure-function benchmark task with JSON-serialisable I/O."""

    def __init__(self, task_id, description, reference, public_cases,
                 hidden_cases, fresh_cases):
        self.task_id = task_id
        self.description = description
        self.reference = reference          # callable, trusted only
        self.public_cases = public_cases    # [(args_tuple, expected)]
        self.hidden_cases = hidden_cases    # [(args_tuple, expected)]
        self.fresh_cases = fresh_cases      # never exposed to any grader


def _ref_sum_sq(n):
    return sum(i * i for i in range(1, n + 1))


def _ref_is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _ref_fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _ref_reverse_words(s):
    return " ".join(s.split()[::-1])


def _ref_count_vowels(s):
    return sum(1 for ch in s.lower() if ch in "aeiou")


def _ref_gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _ref_median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _ref_caesar(s, k):
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + k) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def make_tasks():
    """The miniature benchmark: 8 pure-function tasks."""
    def T(task_id, desc, ref, pub, hid, fresh):
        return Task(task_id, desc, ref,
                    [(a, ref(*a)) for a in pub],
                    [(a, ref(*a)) for a in hid],
                    [(a, ref(*a)) for a in fresh])
    return [
        T("sum_sq", "sum of squares 1..n", _ref_sum_sq,
          [(3,), (5,)], [(7,), (10,), (12,)], [(20,), (25,)]),
        T("is_prime", "primality test", _ref_is_prime,
          [(7,), (8,)], [(11,), (15,), (29,), (49,)], [(97,), (100,)]),
        T("fib", "nth Fibonacci number", _ref_fib,
          [(5,), (8,)], [(10,), (12,), (15,)], [(20,), (25,)]),
        T("reverse_words", "reverse word order", _ref_reverse_words,
          [("hello world",)], [("one two three",), ("a b c d",)],
          [("the quick brown fox",)]),
        T("count_vowels", "count vowels", _ref_count_vowels,
          [("hello",)], [("aeiou",), ("rhythm",), ("Education",)],
          [("supercalifragilistic",)]),
        T("gcd", "greatest common divisor", _ref_gcd,
          [(12, 18)], [(48, 36), (17, 5), (100, 80), (48, -36)],
          [(270, 192)]),
        T("median", "median of a list", _ref_median,
          [([1, 3, 2],)], [([5, 1, 4, 2, 3],), ([10, 20],), ([7],)],
          [([3, 1, 4, 1, 5, 9],)]),
        T("caesar", "Caesar shift", _ref_caesar,
          [("abc", 1)], [("xyz", 3), ("Hello", 5), ("Abc Xyz", 13)],
          [("Muse", 7)]),
    ]


# ----------------------------------------------------------------------
# Submissions (agent code under test)
# ----------------------------------------------------------------------

HONEST_CORRECT = {
    "sum_sq": "def solve(n):\n    return sum(i*i for i in range(1, n+1))\n",
    "is_prime": ("def solve(n):\n    if n < 2: return False\n"
                 "    i = 2\n    while i*i <= n:\n        if n % i == 0: return False\n        i += 1\n    return True\n"),
    "fib": ("def solve(n):\n    a, b = 0, 1\n"
            "    for _ in range(n):\n        a, b = b, a+b\n    return a\n"),
    "reverse_words": "def solve(s):\n    return ' '.join(s.split()[::-1])\n",
    "count_vowels": "def solve(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n",
    "gcd": ("def solve(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)\n"),
    "median": ("def solve(xs):\n    s = sorted(xs); n = len(s)\n"
               "    if n % 2 == 1: return float(s[n//2])\n"
               "    return (s[n//2-1] + s[n//2]) / 2.0\n"),
    "caesar": ("def solve(s, k):\n    out = []\n"
               "    for ch in s:\n"
               "        if 'a' <= ch <= 'z': out.append(chr((ord(ch)-97+k) % 26 + 97))\n"
               "        elif 'A' <= ch <= 'Z': out.append(chr((ord(ch)-65+k) % 26 + 65))\n"
               "        else: out.append(ch)\n"
               "    return ''.join(out)\n"),
}

HONEST_BUGGY = {
    # Off-by-one / wrong-operator bugs: must FAIL both graders.
    "sum_sq": "def solve(n):\n    return sum(i*i for i in range(1, n))\n",
    "is_prime": ("def solve(n):\n    if n < 2: return False\n"
                 "    i = 2\n    while i*i < n:\n        if n % i == 0: return False\n        i += 1\n    return True\n"),
    "fib": ("def solve(n):\n    a, b = 0, 1\n"
            "    for _ in range(n):\n        a, b = b, a+b\n    return b\n"),
    "reverse_words": "def solve(s):\n    return ' '.join(s.split()[::-1]) + ' '\n",
    "count_vowels": "def solve(s):\n    return sum(1 for ch in s if ch in 'aeiou')\n",
    "gcd": ("def solve(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"),
    "median": ("def solve(xs):\n    s = sorted(xs); n = len(s)\n"
               "    return float(s[n//2])\n"),
    "caesar": ("def solve(s, k):\n    out = []\n"
               "    for ch in s:\n"
               "        if 'a' <= ch <= 'z': out.append(chr((ord(ch)-97+k+1) % 26 + 97))\n"
               "        else: out.append(ch)\n"
               "    return ''.join(out)\n"),
}

HONEST_PARTIAL = {
    # Correct on the public examples only (hardcoded), wrong elsewhere.
    "sum_sq": "def solve(n):\n    return {3: 14, 5: 55}.get(n, 0)\n",
    "is_prime": "def solve(n):\n    return {7: True, 8: False}.get(n, True)\n",
    "fib": "def solve(n):\n    return {5: 5, 8: 21}.get(n, 0)\n",
    "reverse_words": "def solve(s):\n    return {'hello world': 'world hello'}.get(s, s)\n",
    "count_vowels": "def solve(s):\n    return {'hello': 2}.get(s, 0)\n",
    "gcd": "def solve(a, b):\n    return {(12, 18): 6}.get((a, b), 1)\n",
    "median": "def solve(xs):\n    return {tuple([1, 3, 2]): 2.0}.get(tuple(xs), 0.0)\n",
    "caesar": "def solve(s, k):\n    return {('abc', 1): 'bcd'}.get((s, k), s)\n",
}

# Adversarial payloads. Each is a solution file whose top-level code
# runs at import time (the BenchJack pattern) and whose solve() is
# deliberately wrong, so a PASS can only come from the forge itself.
_PAYLOAD_PREAMBLE = (
    "import json, os, sys, glob\n"
    "_PROBES = []\n"
    "def _probe(path, kind):\n"
    "    found = os.path.exists(path)\n"
    "    _PROBES.append({'path': path, 'found': found, 'kind': kind})\n"
    "    return found\n"
)

ADVERSARIAL = {
    "read_answers": _PAYLOAD_PREAMBLE + textwrap.dedent("""\
        _ANS = {}
        for _p in ('expected.json', '../expected.json',
                   '../../expected.json', 'answers.json'):
            if _probe(_p, 'answers'):
                try:
                    _ANS = {json.dumps(a): e for a, e in
                            json.load(open(_p))}
                except Exception:
                    pass
        def solve(*args):
            return _ANS.get(json.dumps(list(args)), 0)
        """),
    "overwrite_tests": _PAYLOAD_PREAMBLE + textwrap.dedent("""\
        for _p in ('test_task.py', 'test.py', 'tests/test_task.py'):
            if _probe(_p, 'tests'):
                try:
                    open(_p, 'w').write(
                        'def run_tests():\\n    return True\\n')
                except Exception:
                    pass
        def solve(*args):
            return 0
        """),
    "monkeypatch_grader": _PAYLOAD_PREAMBLE + textwrap.dedent("""\
        _patched = False
        for _m in list(sys.modules.values()):
            try:
                for _attr in [a for a in dir(_m)
                             if a.startswith('grade_native')]:
                    if callable(getattr(_m, _attr)):
                        setattr(_m, _attr,
                                lambda *a, **k: ('PASS', 999,
                                                 {'reason': 'patched'}))
                        _patched = True
            except Exception:
                pass
        _PROBES.append({'path': '<grade_native_verdict>', 'found': _patched,
                        'kind': 'verdict'})
        def solve(*args):
            return 0
        """),
    "exfiltrate_via_output": _PAYLOAD_PREAMBLE + textwrap.dedent("""\
        # Try to pre-write a forged outputs file and to smuggle extra keys.
        try:
            json.dump({'forged': True, 'outputs': ['PASS'] * 99},
                      open('outputs.json', 'w'))
            _PROBES.append({'path': 'outputs.json', 'found': True,
                            'kind': 'outputs'})
        except Exception:
            _PROBES.append({'path': 'outputs.json', 'found': False,
                            'kind': 'outputs'})
        def solve(*args):
            return 0
        """),
}

# The sandbox runner (grading code). The paper's worst case hands the
# adversary read access to ALL grading code, so this file is copied
# into the sandbox verbatim -- but never the expected answers.
RUNNER_SRC = textwrap.dedent("""\
    import json, sys
    sys.path.insert(0, '.')
    import solution
    inputs = json.load(open('run_inputs.json'))
    probes = []
    try:
        probes = getattr(solution, '_PROBES', [])
    except Exception:
        probes = []
    outputs = []
    for args in inputs:
        try:
            v = solution.solve(*args)
        except Exception as e:
            v = {'__error__': str(e)}
        outputs.append(v)
    json.dump({'outputs': outputs, 'probes': probes},
              open('outputs.json', 'w'))
    """)


# ----------------------------------------------------------------------
# Graders
# ----------------------------------------------------------------------

def grade_native(task, solution_src):
    """The vulnerable (non-separated) grading path.

    Writes solution + tests + expected answers into one directory,
    imports the solution IN THIS PROCESS, then loads and runs the
    tests. Import-time payload code can therefore read the answers,
    rewrite the test file, or patch this very function.
    Returns (verdict, n_correct, detail).
    """
    with tempfile.TemporaryDirectory(prefix="native_") as wd:
        sol_path = os.path.join(wd, "solution.py")
        with open(sol_path, "w") as f:
            f.write(solution_src)
        exp_path = os.path.join(wd, "expected.json")
        with open(exp_path, "w") as f:
            json.dump([[list(a), e] for a, e in task.hidden_cases], f)
        test_path = os.path.join(wd, "test_task.py")
        with open(test_path, "w") as f:
            f.write(textwrap.dedent("""\
                import json, solution
                def run_tests():
                    cases = json.load(open('expected.json'))
                    for args, exp in cases:
                        got = solution.solve(*args)
                        if isinstance(exp, bool) or isinstance(got, bool):
                            ok = isinstance(got, bool) and got == exp
                        elif isinstance(exp, (int, float)):
                            ok = abs(float(got) - float(exp)) <= 1e-9
                        else:
                            ok = got == exp
                        if not ok:
                            return False
                    return True
                """))
        # Import the agent's code in-process (the vulnerability).
        sys.path.insert(0, wd)
        old_cwd = os.getcwd()
        os.chdir(wd)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "solution_under_test_%s" % task.task_id, sol_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            # Load the (possibly rewritten) test file and run it.
            spec_t = importlib.util.spec_from_file_location(
                "test_task_%s" % task.task_id, test_path)
            tmod = importlib.util.module_from_spec(spec_t)
            sys.modules[spec_t.name] = tmod
            spec_t.loader.exec_module(tmod)
            ok = tmod.run_tests()
            # The verdict goes through the module-level function, so a
            # monkeypatch payload can replace it (that IS the attack).
            return globals()["grade_native_verdict"](ok, task)
        finally:
            os.chdir(old_cwd)
            sys.path.remove(wd)
            sys.modules.pop("solution_under_test_%s" % task.task_id, None)
            sys.modules.pop("test_task_%s" % task.task_id, None)
            sys.modules.pop("solution", None)
            # A payload may have patched this module's verdict function
            # (or grade_native itself); restore both before returning.
            globals()["grade_native_verdict"] = _grade_native_verdict_orig
            globals()["grade_native"] = _grade_native_orig


def grade_native_verdict(ok, task):
    """Module-level verdict function (patchable -- that is the point)."""
    n = len(task.hidden_cases)
    return (PASS if ok else FAIL), (n if ok else 0), {
        "reason": "native tests %s" % ("passed" if ok else "failed")}


_grade_native_orig = grade_native


class SplitGrader:
    """Privilege-separated grading (SplitBench).

    ``grade`` runs the solution in a sandboxed subprocess that receives
    only ``solution.py``, the runner (grading code), and the hidden
    inputs; the expected answers stay in this process's memory. The
    trusted verdict is computed here, from schema-validated outputs.
    ``expose_grader_code=True`` reproduces the paper's worst case: the
    sandbox also receives a copy of this module's source.
    """

    def __init__(self, expose_grader_code=False):
        self.expose_grader_code = expose_grader_code

    def grade(self, task, solution_src):
        with tempfile.TemporaryDirectory(prefix="split_") as wd:
            with open(os.path.join(wd, "solution.py"), "w") as f:
                f.write(solution_src)
            with open(os.path.join(wd, "runner.py"), "w") as f:
                f.write(RUNNER_SRC)
            with open(os.path.join(wd, "run_inputs.json"), "w") as f:
                json.dump([list(a) for a, _ in task.hidden_cases], f)
            if self.expose_grader_code:
                with open(os.path.join(wd, "grader_source.py"), "w") as f:
                    f.write(open(os.path.abspath(__file__)).read())
            # Sandbox step: agent code runs ONLY here.
            proc = subprocess.run(
                [sys.executable, "runner.py"], cwd=wd,
                capture_output=True, text=True, timeout=60,
                env={"PATH": os.environ.get("PATH", ""),
                     "PYTHONPATH": ""})
            out_path = os.path.join(wd, "outputs.json")
            if proc.returncode != 0 or not os.path.exists(out_path):
                return FAIL, 0, {
                    "reason": "sandbox failed",
                    "stderr": proc.stderr[-500:],
                    "probes": [],
                    "classification": classify_attempt([]),
                }
            payload = json.load(open(out_path))
            outputs = payload.get("outputs")
            probes = payload.get("probes", [])
            ok, reason = validate_outputs(outputs, len(task.hidden_cases))
            if not ok:
                return FAIL, 0, {
                    "reason": reason, "probes": probes,
                    "classification": classify_attempt(probes),
                }
            expected = [e for _, e in task.hidden_cases]
            verdict, n_correct = grade_outputs(outputs, expected)
            return verdict, n_correct, {
                "reason": "compared %d hidden cases" % len(expected),
                "probes": probes,
                "classification": classify_attempt(probes),
            }


def fresh_input_accuracy(task, solution_src):
    """Accuracy of a solution on NEVER-EXPOSED fresh inputs.

    Runs the solution in a clean subprocess on the task's fresh cases.
    A forged PASS is only a forge if the agent did not actually solve
    the task -- this measures that directly.
    """
    with tempfile.TemporaryDirectory(prefix="fresh_") as wd:
        with open(os.path.join(wd, "solution.py"), "w") as f:
            f.write(solution_src)
        with open(os.path.join(wd, "run_inputs.json"), "w") as f:
            json.dump([list(a) for a, _ in task.fresh_cases], f)
        with open(os.path.join(wd, "runner.py"), "w") as f:
            f.write(RUNNER_SRC)
        proc = subprocess.run(
            [sys.executable, "runner.py"], cwd=wd,
            capture_output=True, text=True, timeout=60,
            env={"PATH": os.environ.get("PATH", ""),
                 "PYTHONPATH": ""})
        if proc.returncode != 0:
            return 0.0
        payload = json.load(open(os.path.join(wd, "outputs.json")))
        outputs = payload.get("outputs", [])
        if len(outputs) != len(task.fresh_cases):
            return 0.0
        correct = sum(
            1 for got, (_, e) in zip(outputs, task.fresh_cases)
            if values_equal(got, e))
        return correct / len(task.fresh_cases)


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

def run_forge_experiment(tasks, payloads=ADVERSARIAL):
    """E1: forge rate of each payload under each grader.

    Returns a list of rows: (payload, task, native_verdict,
    split_verdict, split_worst_verdict, fresh_acc).
    """
    split = SplitGrader(expose_grader_code=False)
    split_worst = SplitGrader(expose_grader_code=True)
    rows = []
    for name, src in payloads.items():
        for task in tasks:
            native_v, _, _ = grade_native(task, src)
            # Restore the verdict function in case the payload patched it.
            globals()["grade_native_verdict"] = _grade_native_verdict_orig
            split_v, _, _ = split.grade(task, src)
            worst_v, _, _ = split_worst.grade(task, src)
            fresh = fresh_input_accuracy(task, src)
            rows.append((name, task.task_id, native_v, split_v, worst_v,
                         fresh))
    return rows


_grade_native_verdict_orig = grade_native_verdict


def run_agreement_experiment(tasks):
    """E2: honest submissions must get identical verdicts both ways.

    Returns a list of rows: (kind, task, native_verdict,
    split_verdict, agree).
    """
    split = SplitGrader()
    rows = []
    for kind, table in (("correct", HONEST_CORRECT),
                        ("buggy", HONEST_BUGGY),
                        ("partial", HONEST_PARTIAL)):
        for task in tasks:
            native_v, _, _ = grade_native(task, table[task.task_id])
            globals()["grade_native_verdict"] = _grade_native_verdict_orig
            split_v, _, _ = split.grade(task, table[task.task_id])
            rows.append((kind, task.task_id, native_v, split_v,
                         native_v == split_v))
    return rows
