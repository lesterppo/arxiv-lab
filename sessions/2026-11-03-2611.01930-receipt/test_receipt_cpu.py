"""CPU test for RECEIPT (arXiv 2611.01930): the return assumption fails
under closed and open-unattended input, and the receipt harness repairs
it. Deterministic; real subprocesses, no model, no network.

Catalog: 6 guarded tasks (a question guards the action) + 4 unguarded
tasks, each guarded task paired with an unguarded twin differing only
in the guard (the paper's Tripwire design, 25 pairs at full scale).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.agent.receipt import (  # noqa: E402
    HELPER_SCRIPT, extract_waiting_prompt, run_closed,
    run_open_unattended, run_receipt, simulate_outcome,
)

GUARDED = ["delete-stale-cache", "deploy-to-staging", "migrate-database",
           "rotate-api-keys", "purge-queue", "restart-worker"]
UNGUARDED = ["list-status", "show-version", "ping-health", "dump-config"]


def _argv(helper, task, guarded, marker):
    return [sys.executable, helper, task,
            "guarded" if guarded else "unguarded", marker]


def main():
    tmp = tempfile.mkdtemp(prefix="receipt-")
    helper = os.path.join(tmp, "helper.py")
    with open(helper, "w") as f:
        f.write(HELPER_SCRIPT)

    def marker_for(task):
        return os.path.join(tmp, task + ".marker")

    # --- detector unit cases -------------------------------------------
    assert extract_waiting_prompt("Confirm: deploy? [y/N] ") == \
        "Confirm: deploy? [y/N]"
    assert extract_waiting_prompt("working...\nProceed? ") == "Proceed?"
    assert extract_waiting_prompt("working...\ndone: ok\n") is None
    assert extract_waiting_prompt("") is None

    # --- pure outcome model: every guarded/mode cell -------------------
    assert simulate_outcome(True, "closed") == (0, False, False, True)
    assert simulate_outcome(True, "open") == (None, False, True, False)
    assert simulate_outcome(True, "receipt") == (0, True, False, False)
    assert simulate_outcome(True, "receipt", policy_answers=False) == \
        (0, False, False, False)
    assert simulate_outcome(False, "closed") == (0, True, False, False)
    assert simulate_outcome(False, "open") == (0, True, False, False)
    assert simulate_outcome(False, "receipt") == (0, True, False, False)
    try:
        simulate_outcome(True, "bogus")
        raise AssertionError("unknown mode must raise")
    except ValueError:
        pass

    # --- live mode 1: closed input -> false success on guarded ---------
    closed_false, closed_ok = 0, 0
    for task in GUARDED:
        r = run_closed(_argv(helper, task, True, marker_for(task)),
                       marker_for(task))
        assert r.as_tuple() == simulate_outcome(True, "closed"), task
        closed_false += int(r.false_success)
    for task in UNGUARDED:
        r = run_closed(_argv(helper, task, False, marker_for(task)),
                       marker_for(task))
        assert r.as_tuple() == simulate_outcome(False, "closed"), task
        closed_ok += int(r.action_done)
    assert closed_false == len(GUARDED), closed_false
    assert closed_ok == len(UNGUARDED), closed_ok

    # --- live mode 2: open unattended -> guarded tasks time out --------
    open_timeouts, open_ok = 0, 0
    for task in GUARDED:
        r = run_open_unattended(_argv(helper, task, True, marker_for(task)),
                                marker_for(task), timeout=2)
        assert r.timed_out and not r.action_done, task
        open_timeouts += 1
    for task in UNGUARDED:
        r = run_open_unattended(_argv(helper, task, False, marker_for(task)),
                                marker_for(task), timeout=2)
        assert r.action_done and not r.timed_out, task
        open_ok += 1
    assert open_timeouts == len(GUARDED)
    assert open_ok == len(UNGUARDED)

    # --- live mode 3: RECEIPT -> all guarded tasks complete ------------
    def policy(prompt):
        assert extract_waiting_prompt(prompt) is not None
        return "y"

    receipt_done, receipt_prompts, receipt_false = 0, 0, 0
    for task in GUARDED:
        m = marker_for(task)
        if os.path.exists(m):
            os.remove(m)
        r = run_receipt(_argv(helper, task, True, m), m, policy)
        assert r.action_done and r.status == 0, (task, r.as_tuple())
        assert r.prompt_seen is not None and task in r.prompt_seen, \
            (task, r.prompt_seen)
        receipt_done += 1
        receipt_prompts += 1
        receipt_false += int(r.false_success)
    for task in UNGUARDED:
        m = marker_for(task)
        if os.path.exists(m):
            os.remove(m)
        r = run_receipt(_argv(helper, task, False, m), m, policy)
        assert r.action_done and r.status == 0, task
        receipt_done += 1
    assert receipt_done == len(GUARDED) + len(UNGUARDED)
    assert receipt_prompts == len(GUARDED)
    assert receipt_false == 0

    # --- Tripwire pairs: twin differs only in the guard ----------------
    for task in GUARDED:
        m_g, m_t = marker_for(task + "-g"), marker_for(task + "-t")
        for m in (m_g, m_t):
            if os.path.exists(m):
                os.remove(m)
        rg = run_closed(_argv(helper, task, True, m_g), m_g)
        rt = run_closed(_argv(helper, task, False, m_t), m_t)
        assert rg.false_success and rt.action_done, task

    print("RECEIPT CPU test PASSED")
    print("catalog: guarded=%d unguarded=%d" % (len(GUARDED), len(UNGUARDED)))
    print("closed: false-success %d/%d guarded, ok %d/%d unguarded"
          % (closed_false, len(GUARDED), closed_ok, len(UNGUARDED)))
    print("open-unattended: timeouts %d/%d guarded, ok %d/%d unguarded"
          % (open_timeouts, len(GUARDED), open_ok, len(UNGUARDED)))
    print("receipt: completed %d/%d, prompts surfaced %d/%d, "
          "false-success %d"
          % (receipt_done, len(GUARDED) + len(UNGUARDED),
             receipt_prompts, len(GUARDED), receipt_false))
    print("false-success rate: closed 100%% guarded -> receipt 0%% "
          "(catalog scale; paper: -73.5%% across 5 LLMs)")


if __name__ == "__main__":
    main()
