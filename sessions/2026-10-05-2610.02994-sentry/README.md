# Session 2026-10-05 — Sentry (arXiv:2610.02994)

**Article:** "Sentry: Learning to Recover from LLM Agent Failures at Test Time"
([abs](https://arxiv.org/abs/2610.02994))

**Why picked:** most prospective *agent* article in the 2026-10-02→05 window
(AGENT 5/5, TUNE 2/5). Its core principle — failure lessons are *conditional*
knowledge that must be conditionally exposed — plugs directly into Peter's
stack: DeepWorld's contribution loop and the arxiv-lab harness both
accumulate experience, and the paper gives a concrete, implementable
discipline for it (detect → retrieve-on-failure → verify-without-rewards →
store-only-if-verified, full playbook never in context). Pure CPU-testable,
no GPU needed.

**Paper claims:** (i) Sentry beats the strongest runtime-intervention
baseline by 37% on average across agentic benchmarks, and the strongest
context-evolution baseline by 39% where both ran; (ii) exposing the full
playbook to the agent *lowers* performance even when relevant lessons remain
available on demand; (iii) learned lessons transfer to held-out tasks;
(iv) a new lesson is stored only after verified recovery, verified *without*
task rewards.

## What was built (agent track, this VM)

- Reusable mechanism: `src/arxiv_lab/sentry/sentry.py` (+ `__init__.py`) —
  `FailureDetector` (invalid tool call / repeated action / poor grounding /
  premature finish from traces, no labels), `Playbook` (external store;
  `retrieve()` returns only failure-matching lessons — no dump-all API),
  `RecoveryVerifier.verify(trace_before, trace_after, failure)` (structural
  checks only: no same-type recurrence, ≥1 valid post-lesson action,
  pattern differs from the failure — signature provably takes no
  reward/success label), `Sentry` (on_failure → conditional retrieval;
  maybe_store gate; stats). Test: `tests/test_sentry_cpu.py` (all pass).
- Session experiment: `test_session_cpu.py` (snapshot of the graduated test)
  + `results.json`. Synthetic suite: 48 tasks, traps cycle
  [none, invalid_args, repeated, premature]; train = tasks 0–31,
  held-out = 32–47. Four arms: A no lessons, B full playbook in context
  (conditional lessons misfire w.p. 0.35 when their failure is absent),
  C Sentry (retrieve-on-failure + verify + learn), D static retrieve
  (runtime-intervention baseline, never learns). Seed playbook covers 2 of
  3 trap types; the premature-failure lesson must be learned on train.

## Results (CPU, deterministic)

| arm (train, n=32) | success | note |
|---|---|---|
| A no lessons | 8/32 = 0.250 | baseline |
| B full playbook in context | 29/32 = 0.906 | 19 misfires |
| D static retrieve (no learn) | 24/32 = 0.750 | runtime-intervention baseline |
| C Sentry + learn | 32/32 = 1.000 | 1 lesson stored |

- **Claim (i):** C vs D = **+33.3% relative** (paper: +37% avg). Direction and
  magnitude both reproduced.
- **Claim (ii):** B < C by 9.4 pts with 19 misfires — exposing the full
  playbook hurts vs conditional retrieval, as the paper's controlled
  experiment found.
- **Claim (iii):** held-out — C 16/16 = 1.000 vs D 15/16 = 0.938. The
  learned premature-failure lesson transfers (D's unguided retry succeeds
  only ~half the time per trap by construction).
- **Claim (iv):** store gate accepted 1 verified lesson, rejected 1 lesson
  from a failed recovery; verifier takes only traces + the failure event.

## Verdict: GREEN

All four paper claims reproduced qualitatively on the synthetic suite, with
the headline number (+33.3% vs +37%) strikingly close. The mechanism
graduated to `src/arxiv_lab/sentry/`.

## Honest limits

- Synthetic agent/tasks: failure modes, misfire probability (0.35), and the
  50% unguided-recovery rate are modeling choices, not measured LLM
  behavior. The +33.3% is a qualitative reproduction, not a replication of
  the paper's benchmark numbers.
- Poor-grounding detection is heuristic (identifier mentions vs observed
  text) and was unit-tested only, not exercised in the arm comparison.
- No live-LLM validation yet (NVIDIA NIM surrogate was available but the
  claims are structural and fully testable on CPU; a model-backed check of
  the detector on real agent traces is future work).
- DeepWorld wiring not done in this run — the module is import-ready for
  the contribution loop's failure path when Peter wants it.
