# DART — Diverse Alternative Reasoning Traces for Hard Math Problems

- Article: https://arxiv.org/abs/2611.34928
- Session: `sessions/2026-11-28-2611.34928-dart/`
- Date: 2026-11-28

## Why picked

AGENT 4, TUNE 4. DART is a trace-mining method for post-training data:
start from a rejected trace, decompose it into steps, rank steps by
position-weighted uncertainty, generate deliberately distinct
alternatives for the top steps with reference-answer guidance plus a
meta-thought, and complete the trace. The paper reports recovering 37%
more problems than Monte-Carlo sampling at a matched trace budget, at
~30% lower cost per accepted trace. Directly relevant to the arxiv-lab
agent track (a reusable mining mechanism for a local harness) and to
post-training data generation (TUNE). Repo fit: `arxiv-lab`
(mechanism library); the deterministic core graduates to
`src/arxiv_lab/agent/dart.py`.

## What was built

- `src/arxiv_lab/agent/dart.py` — deterministic mechanism: step
  splitting, position-weighted uncertainty ranking, `\boxed{}` answer
  extraction/normalisation, matched trace-budget accounting, recovery
  rate. Stdlib only.
- `tests/test_dart_cpu.py` — CPU test: unit checks + oracle simulation.
  With uncertainty ranking, DART recovery 0.953 vs MC 0.450 (n=400
  synthetic hard problems, k=4). Ablation: with the position-only
  proxy (no uncertainties) DART LOSES to MC in the oracle (0.172 vs
  0.412) — the ranking is load-bearing; the proxy can only repair
  traps at the tail of a trace.
- `run_nim_experiment.py` — live experiment driver (MATH-500 Level 5,
  NIM `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`; the NIM client
  itself lives in /tmp and is never committed). Matched budget k=3:
  MC = 3 fresh samples; DART = 1 revision + 2 completions.
- No Rust second stage: the standing rule skips the Rust leg on a
  NEGATIVE verdict (see Verdict below).

## Live results (NIM, MATH-500 Level 5)

Generator: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` via NIM
(gpt-oss-20b retired from NIM: health check HTTP 410 Gone).
Pass 2 (final; pass 1 superseded — see below):

- Screening: 70 Level-5 problems, greedy (temp 0) solve rate **56/70
  (80%)**. 14 unsolved; 5 of those had EMPTY greedy traces (reasoning
  consumed the whole 16,384-token budget, no content emitted) and are
  not DART-applicable — no trace to revise. Comparison set: **9
  problems** with a non-empty rejected trace.
- Matched budget k=3 per problem: MC = 3 fresh samples (temp 0.7);
  DART = 1 step revision (temp 0.7, reference-answer guidance +
  meta-thought) + 2 completions from the revised step.
- **MC recovery 3/9 (0.333)** — solved 44, 702, 1454.
  **DART recovery 2/9 (0.222)** — solved 44, 1454.
  **Relative gain −33%.** DART-lite gained nothing MC did not also
  solve, and lost 702 (MC solved it; DART's revision call overflowed
  and returned empty, completions failed).
- Revision quality: 2/9 revision calls returned empty content
  (token-cap overflow: 702, 1003); 2/9 traces were single-step
  (662, 768 — degenerate whole-trace revision). Truncation
  (`finish=length`): MC 6/27 traces, DART completions 1/18,
  revisions 2/9.
- Grading sensitivity (checked, not re-run): strict normalised
  string match is the primary rule above. Under generous grading
  (strip `^\circ`, accept `\frac14`/`\dfrac14`, accept listing both
  roots for `1 \pm \sqrt{19}`) the result is a **tie, 7/9 vs 7/9**.
  In neither grading does DART-lite beat MC.

Pass 1 (superseded, n=30, boxed-only extraction, 8192-token cap):
greedy 23/30; MC 1/7, DART 1/7. It exposed two measurement bugs
fixed before pass 2: gold `x=5` vs pred `5` scored wrong (one
rescore flip), and the 8192-token cap produced empty traces plus an
empty revision (2048-token cap). Pass-1 files kept as
`results_pass1.json` / `live_state_pass1.json` / `live_run_pass1.log`.

## Verdict

**NEGATIVE** — for the DART-lite variant actually tested (position-only
step ranking, because the NIM endpoint exposes no token logprobs).
At a matched 3-trace budget on 9 greedy-unsolved MATH-500 Level-5
problems, DART-lite recovered 2/9 vs MC's 3/9 (−33% relative); under
generous grading it ties 7/9 vs 7/9. The paper's reported +37%
recovery gain does not reproduce. Scope this precisely: the
**published** method ranks steps by position-weighted *uncertainty*,
which this run could not implement; the CPU oracle simulation
(`tests/test_dart_cpu.py` E5) shows uncertainty-ranked DART beating
MC 0.953 vs 0.450 on synthetic hard problems, so the ranking — not
the revision idea — is the load-bearing part, and the published
variant remains untested here. No Rust second stage: the standing
rule skips the Rust leg on a NEGATIVE verdict.

## Honest limits

- The NIM endpoint exposes no token logprobs, so the live run uses the
  position-only ranking proxy (`uncertainty = 1`), not the paper's
  uncertainty ranking. The oracle ablation shows this proxy is the
  weak point: it can only repair late traps. Any live DART win (or
  loss) is a statement about the proxy, not about DART as published.
- DART is a data-mining method: the revision step sees the gold answer
  (reference-answer guidance). That is the paper's design, not leakage
  in the mining framing — but the recovered traces are training data,
  and the recovery rate must not be read as a test-time solve rate.
- Answer checking is string normalisation after `\boxed{}`
  extraction, not symbolic equivalence (`p-q` vs `-q+p` do not match).
- One generator (Nemotron-3-nano reasoning via NIM), one benchmark
  slice (MATH-500 Level 5), small n. No post-training run was done:
  the paper's downstream claim (DART-mined traces improve Llama-3.1-8B
  post-training) is NOT tested here.
