# Semantic Context Handoff: State–Evidence Separation for Long-Horizon Agents (arXiv:2611.24357) — session record

**Article:** "Semantic Context Handoff with On-Demand Evidence for
Long-Horizon Agents", https://arxiv.org/abs/2611.24357 (v1, 2026-11-19).

**Verdict: SUPPORTED** — the mechanism reproduces directionally at
bounded scale with stated limits (see "Honest limits"). All
experiments below were run live on CPU; nothing is a theater run.

## Why picked (AGENT 5, TUNE 1)

State–evidence separation is directly reusable in a local agent
harness: the Hermes AgentRunner already carries a compact memory
snapshot across rounds, and "keep the interpretation, keep the raw
evidence addressable, recover it on demand" is a drop-in context
policy for that loop — cheaper than full history, safer than
summarise-and-discard. No training track: the paper is a mechanism
paper, tested here in the VM's local agent stack. Repo fit:
`arxiv-lab` mechanism library (agent/context area); the graduated
module is `src/arxiv_lab/agent/semantic_context_handoff.py`.

## What was built

- `src/arxiv_lab/agent/semantic_context_handoff.py` (graduated) +
  `tests/test_semantic_context_handoff_cpu.py` (copy in this session):
  a calibrated synthetic long-horizon task model — each task is 12
  steps; each step produces 3 evidence facts; a quarter of facts are
  needed only later (delay 1–4 steps, the paper's crux); a step
  succeeds iff every fact it needs is in context; a task passes iff
  at least 80% of its steps succeed (a missed fact costs a retry).
  Four policies are compared on the SAME tasks (paired comparisons,
  exactly the paper's replay setup): FULL (every fact so far, full
  tokens — the quality ceiling), SUMMARY (compact state only; each
  fact retained with p=0.85 at 30% of its tokens, no recovery), SCH
  (same state + keyed evidence store; a missing needed fact is
  recovered with recall 0.9 at full size + 5-token lookup overhead,
  0.2 expected distractor pulls per lookup), ORACLE (SCH with
  recall=1, no false positives — the recovery upper bound). Stdlib
  only, no model calls, no network. Determinism flows through
  SplitMix64, implemented identically in the Rust port.
- `rust/` — Rust second stage (std only): full port + `verify` binary,
  unit + deep tests, parity sweep, soak. See `rust/README.md`.

## Results (measured vs paper)

10,000 tasks (5 seeds x 2,000), 12 steps, calibrated regime (every
parameter and its source is documented in the module):

| policy | task pass | strict task pass (all steps) | step success | mean tokens/task |
|---|---|---|---|---|
| FULL (ceiling) | 1.0000 | 1.0000 | 1.0000 | 4,677.5 |
| SUMMARY | 0.9288 | 0.3354 | 0.9149 | 1,585.5 |
| **SCH** | **0.9999** | **0.8991** | **0.9911** | **1,614.3** |
| ORACLE | 1.0000 | 1.0000 | 1.0000 | 1,612.3 |

Paper vs measured:

- **Evidence recovery restores the summary loss: +7.11pp task pass**
  (0.9288 → 0.9999) and **+7.63pp step success** (0.9149 →
  0.9911). Paper: 80.0% → 86.9% Pass@1, +6.9pp. The gain matches
  the paper's magnitude almost exactly; the absolute levels are
  higher because the synthetic model isolates the evidence channel
  (no reasoning errors — see Honest limits).
- **SCH is within 0.01pp of the FULL ceiling on task pass** (0.89pp
  on step success), at **65.5% fewer tokens than FULL**
  (1,614 vs 4,678 per task). Paper: 43% fewer tokens than the
  strongest baseline. Mine is larger because the comparison basis
  differs — FULL history, not the paper's strongest (already
  compressed) baseline.
- **Recovery costs essentially nothing over SUMMARY: +1.8% tokens**
  (1,614 vs 1,586 per task). Paper: "essentially unchanged token
  use". Recovery hit rate 0.9003 (recall parameter 0.9); 1.07
  recoveries per task.
- **Per-seed spread is tiny:** SCH task pass 99.95–100.0% across
  seeds; SUMMARY 92.2–93.65%.
- **The strict all-steps pass rate shows the amplification:** SUMMARY
  0.3354 vs SCH 0.8991 — a single lost fact fails a strict task far
  more often than it fails the 80%-steps rule. The headline numbers
  use the retry-tolerant rule; the strict numbers are reported
  alongside, not hidden.

## Mechanism checks (all live)

- **Ablation — recovery is the mechanism, not the summary:** SCH
  with recall=0 collapses EXACTLY to SUMMARY (identical task pass);
  SCH with recall=0.9 sits within 0.0pp of ORACLE on task pass.
- **Monotonicity:** SCH step success rises with recall —
  0.950 / 0.972 / 0.992 / 1.000 at recall 0.5 / 0.7 / 0.9 / 1.0.
- **No-delay sanity:** with delayed_frac=0 (nothing is ever needed
  later), all four policies pass 1.0000 — the model is not rigged to
  punish summarisation.
- **Sensitivity sweep (27 cells: delayed_frac × state_retention ×
  recall):** SCH beats SUMMARY in **27/27** cells (gain +0.5pp to
  +96.5pp — largest where the summary is lossy and delayed load is
  heavy); SCH is within 10pp of FULL in **25/27** cells. The two
  exceptions are both recall=0.7 with a bad summary
  (retention 0.55): delayed_frac 0.35 → SCH 0.870, 13.0pp from FULL;
  delayed_frac 0.55 → SCH 0.708, 29.2pp from FULL. When recovery
  itself is unreliable AND the summary is lossy, SCH cannot fully
  close the gap — regime-dependence, honestly reported.

## Honest limits

- **Synthetic tasks, not SWE-bench.** The paper's 150-task study is
  not public, so tasks are generated by a calibrated model. The
  verdict ceiling is SUPPORTED, never VERIFIED.
- **Evidence availability is the only failure mode here.** Real
  agents also fail by reasoning errors, bad plans, and tool errors,
  which this model does not include — that is why SCH lands at
  0.9999 rather than the paper's 0.869, and why FULL is 1.0000 by
  construction. The comparison isolates the evidence channel; it
  does not predict absolute Pass@1 on real benchmarks.
- **The 80%-steps task rule is an assumption** (a missed fact costs
  a retry; the task fails only when more than a fifth of steps
  fail). The strict all-steps numbers are reported above for
  transparency.
- **Token sizes are modelled, not measured:** fact sizes 10–30
  tokens, state compression 0.3, lookup overhead 5 — explicit
  assumptions, varied in the sweep only through the three
  load-bearing parameters.
- **No model-backed validation.** The mechanism is deterministic
  logic; no LLM calls were needed or made.
