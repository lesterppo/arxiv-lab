"""Executable finite-state control for LLM agents on procedure
(arXiv:2611.10884, "Keeping LLM Agents on Procedure: Executable
Finite-State Control for Long-Horizon Instruction Fidelity").

Paper claim: agents drift from a stated-once procedure over long runs, and
restating the instructions or resuming a stopped agent only moves the drift
to another shortcut. An executable finite-state machine (FSM) that keeps the
workflow state OUTSIDE the model's context and enforces it at three points
— tool calls, state transitions, and the end of the agent's turn — recovers
long-horizon reliability: Claude Haiku 4.5 + FSM decided on average 139/200
accounts correctly and by the procedure vs 48 for the strongest baseline
(95% CI of the difference [46, 130]), and held that rate to the end of the
run; across three models the gain tracked how much the plain agent drifted.

This module implements the mechanism, not the paper's benchmark:
  * ``SOPSpec`` — a small account-processing SOP as a state machine
    (states, tools, transition table, per-state validity, branch rules).
  * ``FSMRuntime`` — the executable FSM: state lives in the runtime, not
    in the agent's context; it enforces at (1) tool-call time (rejects
    calls invalid in the current state), (2) state transitions (only
    tabled transitions), (3) end of turn (a decision counts only from a
    terminal state reached by the procedure).
  * ``DriftAgent`` — a parameterized stand-in for a drifting model: it
    knows the correct next step, but with probability p (growing with
    run length, i.e. context growth) it takes a shortcut or a wrong
    branch. On an FSM rejection it retries with the correct action
    (prob. 1-p) up to a retry budget. A ``restate`` arm re-issues the SOP
    every K accounts; restating ADDS tokens to the context, so it cannot
    reduce p — it only moves the drift (paper's finding).
  * ``run_batch`` — 200-account SOP-Bench-style batches; metric: accounts
    decided correctly AND by the procedure.

RNG: SplitMix64, implemented here in pure Python so the Rust second stage
can reproduce the exact same streams from the same seeds (parity sweep).

Stdlib only. No network, no model API.
"""

# ---------------------------------------------------------------- states

START = "START"
IDENTITY_VERIFIED = "IDENTITY_VERIFIED"
BALANCE_CHECKED = "BALANCE_CHECKED"
ELIGIBILITY_CHECKED = "ELIGIBILITY_CHECKED"
FEE_WAIVED = "FEE_WAIVED"
CLOSED = "CLOSED"
ESCALATED = "ESCALATED"

TERMINAL = frozenset({CLOSED, ESCALATED})

TOOLS = ("verify_identity", "check_balance", "check_eligibility",
         "waive_fee", "close_account", "escalate")

# (state, tool) -> next state. Anything not in the table is an invalid call.
TRANSITIONS = {
    (START, "verify_identity"): IDENTITY_VERIFIED,
    (IDENTITY_VERIFIED, "check_balance"): BALANCE_CHECKED,
    (BALANCE_CHECKED, "check_eligibility"): ELIGIBILITY_CHECKED,
    (ELIGIBILITY_CHECKED, "waive_fee"): FEE_WAIVED,
    (ELIGIBILITY_CHECKED, "close_account"): CLOSED,
    (FEE_WAIVED, "close_account"): CLOSED,
    (ELIGIBILITY_CHECKED, "escalate"): ESCALATED,
}


class ProcedureViolation(Exception):
    """Raised when a tool call violates the SOP in the current state."""


class Account:
    """One SOP-Bench-style account. Attributes are fixed at creation.

    eligible: the fee waiver applies. needs_escalation: balance < 0 or the
    account is flagged, so the correct terminal decision is ESCALATED
    instead of CLOSED. The correct terminal decision is therefore:
    escalate iff needs_escalation else close (after waive iff eligible).
    """

    __slots__ = ("id", "balance", "eligible", "needs_escalation")

    def __init__(self, id, balance, eligible, needs_escalation):
        self.id = id
        self.balance = balance
        self.eligible = eligible
        self.needs_escalation = needs_escalation

    def correct_terminal(self):
        return "escalate" if self.needs_escalation else "close"

    def __repr__(self):
        return (f"Account({self.id}, bal={self.balance}, "
                f"eligible={self.eligible}, esc={self.needs_escalation})")


def correct_next_tool(state, account):
    """The unique SOP-correct next tool from `state`, or None if terminal."""
    if state == START:
        return "verify_identity"
    if state == IDENTITY_VERIFIED:
        return "check_balance"
    if state == BALANCE_CHECKED:
        return "check_eligibility"
    if state == ELIGIBILITY_CHECKED:
        if account.needs_escalation:
            return "escalate"
        if account.eligible:
            return "waive_fee"
        return "close_account"
    if state == FEE_WAIVED:
        return "close_account"
    return None  # terminal


def branch_is_valid(state, tool, account):
    """Branch conditions the transition table alone cannot express:
    waive_fee only when eligible; escalate only when escalation is needed;
    close_account from ELIGIBILITY_CHECKED only when the account is
    neither eligible (it must waive first) nor escalating. Without the
    last condition the FSM would let an eligible account skip its waiver
    — a procedure violation the paper's FSM must catch."""
    if tool == "waive_fee":
        return bool(account.eligible)
    if tool == "escalate":
        return bool(account.needs_escalation)
    if tool == "close_account" and state == ELIGIBILITY_CHECKED:
        return (not account.needs_escalation) and (not account.eligible)
    return True


class FSMRuntime:
    """Executable FSM. State lives here, outside the agent's context.

    Enforcement points (paper §3): (1) tool calls — a call invalid in the
    current state is rejected before execution; (2) state transitions —
    only tabled transitions, plus the branch conditions; (3) end of turn
    — ``end_turn`` reports whether the account ended in a terminal state
    reached by the procedure.
    """

    def __init__(self, account):
        self.account = account
        self.state = START
        self.trace = []          # accepted (tool, from_state, to_state)
        self.rejected = []       # (tool, state, reason)
        self.violations = 0      # accepted calls that violated procedure
        self.terminal_decision = None

    def call(self, tool):
        """Attempt a tool call. Returns (accepted, message)."""
        nxt = TRANSITIONS.get((self.state, tool))
        if nxt is None:
            self.rejected.append((tool, self.state, "no transition"))
            return False, f"invalid call: {tool} not allowed in {self.state}"
        if not branch_is_valid(self.state, tool, self.account):
            self.rejected.append((tool, self.state, "branch condition"))
            return False, (f"invalid call: {tool} branch condition fails "
                           f"in {self.state}")
        prev = self.state
        self.state = nxt
        self.trace.append((tool, prev, nxt))
        if self.state in TERMINAL:
            self.terminal_decision = ("escalate" if tool == "escalate"
                                      else "close")
        return True, f"ok: {prev} --{tool}--> {nxt}"

    def end_turn(self):
        """End-of-turn enforcement: counts only a terminal state reached
        by the procedure. Returns (complete, decision)."""
        if self.state in TERMINAL:
            return True, self.terminal_decision
        return False, None


class PlainRuntime:
    """No enforcement: every call executes, whatever the state. Used by
    the plain (and restate) arms. A call that the SOP forbids is recorded
    as a procedure violation; the state still advances along the table
    when a transition exists, else the call is a no-op that still counts
    as a violation (the agent acted outside the procedure)."""

    def __init__(self, account):
        self.account = account
        self.state = START
        self.trace = []
        self.rejected = []
        self.violations = 0
        self.terminal_decision = None

    def call(self, tool):
        nxt = TRANSITIONS.get((self.state, tool))
        valid = nxt is not None and branch_is_valid(
            self.state, tool, self.account)
        if not valid:
            self.violations += 1
        if nxt is not None:
            prev = self.state
            self.state = nxt
            self.trace.append((tool, prev, nxt))
            if self.state in TERMINAL:
                self.terminal_decision = ("escalate" if tool == "escalate"
                                          else "close")
        else:
            # No transition: the call executes but changes nothing.
            self.trace.append((tool, self.state, self.state))
        return True, "executed (no enforcement)"

    def end_turn(self):
        if self.state in TERMINAL:
            return True, self.terminal_decision
        return False, None


# ------------------------------------------------------------------- RNG

class SplitMix64:
    """Deterministic 64-bit RNG (Steele et al.). Same stream in Python and
    in the Rust port, so parity is exact for identical seeds."""

    MASK = (1 << 64) - 1

    def __init__(self, seed):
        self.state = seed & self.MASK

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return (z ^ (z >> 31)) & self.MASK

    def random(self):
        return (self.next_u64() >> 11) / float(1 << 53)

    def choice(self, seq):
        return seq[self.next_u64() % len(seq)]


# ------------------------------------------------------------- drift agent

# Deviation kinds a drifting agent takes instead of the correct next tool.
DEVIATIONS = ("skip_to_terminal", "wrong_branch", "skip_one_step")


def deviation_tool(state, account, kind, rng):
    """Pick the concrete deviating tool for `kind` in `state`."""
    if kind == "skip_to_terminal":
        # Jump straight to a terminal call, skipping the remaining steps.
        return rng.choice(["close_account", "escalate"])
    if kind == "wrong_branch":
        # Take a branch call whose condition fails (or the wrong terminal).
        if state == ELIGIBILITY_CHECKED:
            if account.needs_escalation:
                return "close_account"          # should escalate
            if account.eligible:
                return "close_account"          # skips the waiver
            return rng.choice(["waive_fee", "escalate"])
        return rng.choice(["waive_fee", "close_account", "escalate"])
    # skip_one_step: call a tool from a later state (invalid order).
    later = {"verify_identity": "check_balance",
             "check_balance": "check_eligibility",
             "check_eligibility": "waive_fee"}
    return later.get(correct_next_tool(state, account), "close_account")


class DriftAgent:
    """Stand-in for a drifting model. Knows the SOP's correct next step,
    but with probability p (growing with the run's step count, i.e. with
    context length) deviates. Parameters:

    p0: base per-step deviation probability at the start of the run.
    drift_rate: added probability when steps_done reaches drift_scale.
    drift_scale: steps over which the drift grows to p0 + drift_rate.
    max_retries: retries allowed after an FSM rejection, per step.
    """

    def __init__(self, rng, p0=0.05, drift_rate=0.35, drift_scale=1000,
                 max_retries=3):
        self.rng = rng
        self.p0 = p0
        self.drift_rate = drift_rate
        self.drift_scale = drift_scale
        self.max_retries = max_retries
        self.steps_done = 0
        self.context_tokens = 0  # restating adds tokens; never shrinks

    def p_deviate(self):
        frac = min(1.0, self.steps_done / self.drift_scale)
        return min(0.95, self.p0 + self.drift_rate * frac)

    def restate(self, tokens=400):
        """Re-issue the SOP. It only ADDS context (paper: restating moves
        the drift, it does not remove it)."""
        self.context_tokens += tokens
        self.steps_done += tokens // 10  # restated text ages the run

    def act(self, runtime):
        """Drive one account to a terminal state or failure. Returns a
        dict with the outcome. `runtime` is an FSMRuntime (enforced) or
        a PlainRuntime (unenforced)."""
        account = runtime.account
        invalid_attempts = 0
        retries_used = 0
        guard = 0
        while True:
            guard += 1
            if guard > 200:
                break  # runaway guard: account abandoned, not completed
            correct = correct_next_tool(runtime.state, account)
            if correct is None:
                break  # terminal
            self.steps_done += 1
            tool = correct
            if self.rng.random() < self.p_deviate():
                kind = self.rng.choice(DEVIATIONS)
                tool = deviation_tool(runtime.state, account, kind, self.rng)
            accepted, _msg = runtime.call(tool)
            if not accepted:
                invalid_attempts += 1
                # Rejected by the FSM: retry with the correct action.
                # Each retry is itself a fresh draw (may deviate again).
                ok = False
                for _ in range(self.max_retries):
                    retries_used += 1
                    self.steps_done += 1
                    retry_tool = correct_next_tool(runtime.state, account)
                    if retry_tool is None:
                        ok = True
                        break
                    if self.rng.random() < self.p_deviate():
                        kind = self.rng.choice(DEVIATIONS)
                        retry_tool = deviation_tool(
                            runtime.state, account, kind, self.rng)
                    accepted, _ = runtime.call(retry_tool)
                    if accepted:
                        ok = True
                        break
                    invalid_attempts += 1
                if not ok:
                    break  # retry budget exhausted: account abandoned
                continue
        complete, decision = runtime.end_turn()
        by_procedure = (runtime.violations == 0 and complete)
        correct_decision = (decision == account.correct_terminal())
        return {
            "account": account.id,
            "complete": complete,
            "decision": decision,
            "correct_decision": correct_decision,
            "by_procedure": by_procedure,
            "correct_and_by_procedure": bool(by_procedure and correct_decision),
            "violations": runtime.violations,
            "invalid_attempts": invalid_attempts,
            "retries_used": retries_used,
            "steps": len(runtime.trace),
        }


def make_accounts(n, seed):
    """Deterministic 200-account SOP-Bench-style batch. Mix: 55% plain
    close (eligible=False, no escalation), 25% waiver then close
    (eligible=True), 20% escalation (balance<0 or flagged)."""
    rng = SplitMix64(seed)
    accounts = []
    for i in range(n):
        r = rng.random()
        if r < 0.20:
            accounts.append(Account(i, balance=-rng.next_u64() % 500 - 1,
                                    eligible=False, needs_escalation=True))
        elif r < 0.45:
            accounts.append(Account(i, balance=rng.next_u64() % 5000 + 100,
                                    eligible=True, needs_escalation=False))
        else:
            accounts.append(Account(i, balance=rng.next_u64() % 5000 + 100,
                                    eligible=False, needs_escalation=False))
    return accounts


def run_batch(accounts, arm, seed, p0=0.05, drift_rate=0.35,
              drift_scale=1000, max_retries=3, restate_every=20):
    """Run one arm over `accounts`. arm in {"plain", "restate", "fsm"}.
    Returns (per-account outcome dicts, summary dict)."""
    rng = SplitMix64(seed)
    agent = DriftAgent(rng, p0=p0, drift_rate=drift_rate,
                       drift_scale=drift_scale, max_retries=max_retries)
    outcomes = []
    for idx, account in enumerate(accounts):
        if arm == "restate" and idx > 0 and idx % restate_every == 0:
            agent.restate()
        runtime = FSMRuntime(account) if arm == "fsm" else PlainRuntime(account)
        outcomes.append(agent.act(runtime))
    n = len(outcomes)
    good = sum(o["correct_and_by_procedure"] for o in outcomes)
    quartiles = []
    q = n // 4
    for k in range(4):
        chunk = outcomes[k * q:(k + 1) * q]
        quartiles.append(sum(o["correct_and_by_procedure"] for o in chunk)
                         / len(chunk))
    summary = {
        "arm": arm,
        "n": n,
        "correct_and_by_procedure": good,
        "rate": good / n,
        "correct_decision_only": sum(o["correct_decision"] for o in outcomes) / n,
        "complete_rate": sum(o["complete"] for o in outcomes) / n,
        "quartile_rates": [round(x, 4) for x in quartiles],
        "total_violations": sum(o["violations"] for o in outcomes),
        "total_invalid_attempts": sum(o["invalid_attempts"] for o in outcomes),
        "total_retries": sum(o["retries_used"] for o in outcomes),
        "final_p_deviate": round(agent.p_deviate(), 4),
    }
    return outcomes, summary
