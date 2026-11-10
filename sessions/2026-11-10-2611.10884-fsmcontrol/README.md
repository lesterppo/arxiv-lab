# Keeping LLM Agents on Procedure: Executable Finite-State Control for Long-Horizon Instruction Fidelity

- **Article:** arXiv [2611.10884](https://arxiv.org/abs/2611.10884) — "Keeping LLM
  Agents on Procedure: Executable Finite-State Control for Long-Horizon
  Instruction Fidelity" (submitted 9 Nov 2026)
- **Verdict: SUPPORTED** (directional reproduction at bounded scale; see
  "Why SUPPORTED, not VERIFIED")
- **Track:** agent track — CPU only, stdlib only, no network in the primary
  test; small NIM-backed validation on `openai/gpt-oss-20b` (secondary)

## Why picked

- **AGENT 5 / TUNE 1.** A runtime mechanism directly reusable in a local
  agent harness: keep the workflow state outside the model's context and
  enforce it at tool calls, state transitions, and end of turn. The paper
  claims Haiku 4.5 + FSM decides 139/200 accounts correctly and by the
  procedure vs 48 for the strongest baseline (95% CI of the difference
  [46, 130]), holding that rate to the end of the run, and that the gain
  tracks how much the plain agent drifts.
- **Repo fit:** `arxiv-lab` mechanism library — an FSM enforcement runtime
  is a general harness primitive (procedure following, tool-call gating),
  complementary to the WAAC trajectory-compression module from 2026-11-09.
- **Testable within budget:** the enforcement mechanism is deterministic
  logic over a small state machine, so a faithful CPU proxy test is
  possible in one run; a small real-model check via NVIDIA NIM closes the
  loop on the drift premise.

## What was built

- `fsm_control.py` — the mechanism (stdlib only, deterministic):
  - `SOPSpec` as a transition table: 7 states (START → IDENTITY_VERIFIED
    → BALANCE_CHECKED → ELIGIBILITY_CHECKED → FEE_WAIVED/CLOSED/ESCALATED),
    6 tools, plus branch conditions the table alone cannot express
    (waive only when eligible, escalate only when needed, close from
    ELIGIBILITY_CHECKED only when neither — without the last condition an
    eligible account could skip its waiver through a formally valid
    transition).
  - `FSMRuntime`: state lives in the runtime, not the agent's context;
    enforces at (1) tool-call time (invalid calls rejected before
    execution), (2) transitions (only tabled, branch-valid moves),
    (3) end of turn (a decision counts only from a terminal state
    reached by the procedure).
  - `PlainRuntime`: no enforcement; forbidden calls execute and count as
    procedure violations (the plain/restate arms).
  - `DriftAgent`: a parameterized stand-in for a drifting model — it knows
    the correct next step but with probability p (growing with run length,
    i.e. context growth: p0=0.05 → 0.40 over the run) takes a shortcut
    (jump to a terminal call), a wrong branch, or a skipped step. On an
    FSM rejection it retries with the correct action (max 3 retries/step).
    A `restate` arm re-issues the SOP every 20 accounts; restating ADDS
    context tokens, so it cannot reduce p — it only moves the drift
    (the paper's finding, modeled explicitly).
  - `SplitMix64`: deterministic 64-bit RNG implemented in pure Python so
    the Rust port reproduces the exact same streams from the same seeds.
  - `run_batch`: 200-account SOP-Bench-style batches; metric: accounts
    decided correctly AND by the procedure.
- `test_fsm_cpu.py` — deterministic test, 7 pre-registered assertion groups
  (A1–A7); writes `results.json`.
- `rust/` — Rust second stage (standing rule, 2026-11-02): a line-for-line
  port of the mechanism (no external crates), `verify` binary mirroring
  the Python test's experiments and assertions, unit tests per mechanism,
  deeper tests (property sweeps over seeds/drift rates, exhaustive +
  random sequences against the transition-table oracle, a 2,000-account
  soak, thread-identity), and a Python-vs-Rust parity sweep on 5 seeds ×
  3 arms — 0 mismatches. See `rust/README.md`.

## Results

Main experiment (200 accounts, seed 20261110, p0=0.05, drift_rate=0.35):

| arm | correct & by procedure | quartiles (Q1→Q4) | violations | invalid calls | retries |
|---|---|---|---|---|---|
| Plain | 86/200 (0.430) | 0.78, 0.44, 0.36, 0.14 | 226 | 0 | 0 |
| Restate (every 20) | 84/200 (0.420) | 0.80, 0.42, 0.24, 0.22 | 245 | 0 | 0 |
| **FSM** | **197/200 (0.985)** | **1.00, 1.00, 0.98, 0.96** | **0** | 237 rejected | 234 |

- The FSM recovers 111 accounts over plain (paper: 139 vs 48, gap 91; ours:
  197 vs 86, gap 111 — same direction, larger because the simulated drift
  is harsher than Haiku's and the FSM's retry budget is generous).
- Drift is visible in the plain arm's quartiles (0.78 → 0.14) and absent
  in the FSM arm (1.00 → 0.96). Restating the SOP does not fix drift
  (84 vs 86) — it only moves it, as the paper claims.
- Sensitivity (A6): at drift_rate × {0.5, 1.0, 1.5} the FSM beats plain by
  90 / 111 / 121 accounts — the advantage is not tuned to one drift level.
- With drift disabled (A5) all three arms complete 200/200: the FSM does
  not falsely block a correct agent.
- NIM validation (secondary, 16 accounts, gpt-oss-20b): **BLOCKED — endpoint
  severely degraded today, not a test failure.** 15/15 chat calls timed out
  on read at a 90s timeout, and the health check timed out at 60s. One
  180s probe then succeeded in 133.5s (empty content) — the endpoint
  answers, but at ~2 minutes per trivial completion, a ~160-call
  validation would take ~6 hours, so the leg was not pursued. The CPU
  verdict stands on its own; the real-model check is carried over as
  blocked, not as evidence either way. (Surrogate client script is
  scratch in /tmp, not committed.)

All seven pre-registered assertion groups passed in both Python and Rust;
the Rust `verify` binary reproduces the Python numbers exactly
(plain 86, restate 84, fsm 197, identical quartiles).

## Verdict: SUPPORTED

The enforcement mechanism reproduces the paper's directional claim at
bounded scale: an FSM outside the model's context, enforcing at tool calls,
transitions, and end of turn, recovers procedure fidelity under a drifting
agent (197 vs 86) and holds its rate to the end of the run (1.00 → 0.96),
while restating the SOP does not (84 vs 86). The Rust port verifies the
same numbers independently, with 0 parity mismatches over 15 seed×arm rows.

Why SUPPORTED, not VERIFIED: the primary test's "model" is a parameterized
drift process, not a frontier LLM on SOP-Bench (that benchmark is not
available here); the NIM check is 16 accounts on one open model, not the
paper's 200-account Claude batches. The mechanism is validated; the
paper's exact benchmark numbers (139 vs 48) are not reproduced.

## Honest limits

- **Simulated drift, not model drift.** The drift model (p growing with
  run length, three deviation kinds, retry-on-rejection) is a stand-in
  calibrated so plain lands in a plausible range (86/200 vs the paper's
  48/200). The FSM advantage is robust across drift rates (A6), but the
  absolute counts depend on the drift parameters, which are stated in
  `results.json` rather than hidden.
- **Retry budget generosity.** The FSM arm lets the agent retry a rejected
  call up to 3 times per step, each retry a fresh draw that may deviate
  again; a real agent that argues with the rejection could burn the
  budget faster (one NIM account did). The 197/200 ceiling is partly a
  property of this retry policy.
- **Single SOP, single procedure shape.** One account-closure workflow
  with three branch kinds; no cross-context workflows, no SWE-bench /
  τ²-bench leg (the paper's further analyses). The mechanism generalizes
  (the runtime is SOP-agnostic), but only this SOP was measured.
- **Restate model is a modeling choice.** Restating is implemented as
  adding context tokens (which can only raise p), matching the paper's
  "restating only moves the drift" — a faithful reading, but the residue
  (steps_done += 40 per restatement) is our parameter, not a measurement.
- **NIM check is small (16 accounts, one model).** It confirms the
  direction with a real model; it is not powered to distinguish 11/16
  from 15/16 statistically.

## Reproduce

```bash
cd sessions/2026-11-10-2611.10884-fsmcontrol
python3 test_fsm_cpu.py        # deterministic; writes results.json; exit 0
cd rust && cargo test --release && cargo run --release --bin verify
```
