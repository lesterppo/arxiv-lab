"""RECEIPT: repairing the return assumption of LLM-agent tool calls.

Paper: arXiv 2611.01930 "RECEIPT: Measuring and Repairing the Return
Assumption of LLM Agents in Web Service Operations" (submitted
1 Nov 2026).

One-line claim: an agent harness that launches a command and reads only
its exit status assumes each call returns promptly with its action
finished -- but commands that ask a question before acting break that
assumption: with closed stdin the program can exit 0 without acting
(a false success), and with open-but-unattended stdin it blocks until
the harness timeout (time/token waste). RECEIPT keeps stdin open,
reads what the process is waiting for, and returns that waiting state
to the agent, so the agent answers instead of falsely claiming success.

This module implements that mechanism literally (stdlib only, no
model, no network): three harness modes over the same command
(closed, open-unattended, receipt), a waiting-prompt detector, and a
pure outcome model shared with the Rust port for parity checking.
The "agent" is a scripted answer policy supplied by the test, so the
harness logic itself is what is measured.
"""

import re
import select
import subprocess
import sys

# A line is a waiting prompt if it ends with '?' or carries an
# explicit choice marker such as [y/N]; plain status output matches
# neither, which is what keeps the detector from firing on noise.
_PROMPT_RE = re.compile(r"(\?|\[y/N\]|\[Y/n\]|\[yes/no\])\s*$",
                        re.IGNORECASE)


def extract_waiting_prompt(output):
    """Return the waiting question in `output`, or None.

    Scans lines newest-first: the last prompt-like line is what the
    process is currently waiting on. Returns the stripped line.
    """
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line and _PROMPT_RE.search(line):
            return line
    return None


def simulate_outcome(guarded, mode, policy_answers=True):
    """Pure outcome model: (status, action_done, timed_out, false_success).

    `guarded` -- the command asks a question before acting.
    `mode` -- "closed", "open", or "receipt".
    `policy_answers` -- in receipt mode, the agent policy answers the
    waiting question (False models a policy that declines / stays silent).
    This is the function the Rust port mirrors exactly; the live test
    below checks the real subprocess behaviour against it.
    """
    if not guarded:
        return (0, True, False, False)
    if mode == "closed":
        # EOF on stdin: the program exits 0 having done nothing.
        return (0, False, False, True)
    if mode == "open":
        # Nobody types: the process blocks until the harness timeout.
        return (None, False, True, False)
    if mode == "receipt":
        if policy_answers:
            return (0, True, False, False)
        return (0, False, False, False)
    raise ValueError("unknown mode %r" % (mode,))


class RunResult(object):
    def __init__(self, status, action_done, timed_out=False,
                 prompt_seen=None, false_success=False):
        self.status = status
        self.action_done = action_done
        self.timed_out = timed_out
        self.prompt_seen = prompt_seen
        self.false_success = false_success

    def as_tuple(self):
        return (self.status, self.action_done, self.timed_out,
                self.false_success)


def _action_done(marker_path):
    try:
        with open(marker_path) as f:
            return f.read().strip() == "DONE"
    except OSError:
        return False


def run_closed(argv, marker_path):
    """Harness mode 1: stdin closed (DEVNULL)."""
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=30)
    done = _action_done(marker_path)
    false_success = (proc.returncode == 0 and not done)
    return RunResult(proc.returncode, done, False, None, false_success)


def run_open_unattended(argv, marker_path, timeout=5):
    """Harness mode 2: stdin open, nobody types, harness timeout."""
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    try:
        proc.wait(timeout=timeout)
        done = _action_done(marker_path)
        return RunResult(proc.returncode, done, False, None,
                         proc.returncode == 0 and not done)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return RunResult(None, _action_done(marker_path), True, None, False)


def run_receipt(argv, marker_path, answer_policy, timeout=30):
    """Harness mode 3 (RECEIPT): keep stdin open, surface the wait.

    Reads process output until a waiting prompt appears, hands the
    prompt to `answer_policy` (a callable prompt -> answer string or
    None), writes the answer, and waits for completion. Returns the
    RunResult with `prompt_seen` set to the question surfaced.
    """
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    output = ""
    prompt_seen = None
    try:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            if not ready:
                proc.kill()
                proc.wait()
                return RunResult(None, _action_done(marker_path), True,
                                 prompt_seen, False)
            line = proc.stdout.readline()
            if not line:
                break
            output += line
            prompt = extract_waiting_prompt(output)
            if prompt is not None:
                prompt_seen = prompt
                answer = answer_policy(prompt)
                if answer is None:
                    proc.kill()
                    proc.wait()
                    return RunResult(None, _action_done(marker_path),
                                     False, prompt_seen, False)
                proc.stdin.write(answer + "\n")
                proc.stdin.flush()
                break
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return RunResult(None, _action_done(marker_path), True,
                         prompt_seen, False)
    done = _action_done(marker_path)
    return RunResult(proc.returncode, done, False, prompt_seen,
                     proc.returncode == 0 and not done)


# ---------------------------------------------------------------------------
# Command under test: a helper script (written by the test into a temp dir).
# A guarded command prints a question and acts only on an explicit 'y';
# on EOF it exits 0 without acting -- the closed-input failure mode the
# paper measures. An unguarded twin acts immediately. Both write DONE
# to a marker file iff the action was performed.
# ---------------------------------------------------------------------------

HELPER_SCRIPT = '''\
import sys, os
task, guarded, marker = sys.argv[1], sys.argv[2] == "guarded", sys.argv[3]
if guarded:
    print("Confirm: %s? [y/N] " % task, flush=True)
    line = sys.stdin.readline()
    if line.strip().lower() != "y":
        # EOF (closed input) or a non-y answer: exit 0, action NOT done.
        sys.exit(0)
with open(marker, "w") as f:
    f.write("DONE")
print("done: %s" % task, flush=True)
'''
