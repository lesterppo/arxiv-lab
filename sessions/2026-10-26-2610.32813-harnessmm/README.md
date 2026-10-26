# Session 2026-10-26 — HarnessMM: Recursive Self-Improvement for Harness Token Efficiency

**Article:** [2610.32813](https://arxiv.org/abs/2610.32813) — "HarnessMM: Recursive Self-Improvement for Harness Token Efficiency"

**Verdict: SUPPORTED** (reproduced directionally at bounded scale; stated limits below)

## Why picked (AGENT=5, TUNE=1)

Fresh 2026-10-26 screen, serving source `api` (export.arxiv.org, 200
fetched, 112 in the 3-day window, 0 overlapping the covered set). Highest
AGENT score of the window: guarded decision reuse is a mechanism that drops
directly into any local agent harness — including this library's own
`src/arxiv_lab/harness/` (alongside Mingbird's prefill/finish-gate/loop
mechanisms) — and it is fully CPU-testable with a simulated task stream.
Repo fit: arxiv-lab agent-harness mechanism library. (The paper's
HumanEval/DeepSeek/Codex numbers come from live models and are not
reproduced here; what is reproduced is the mechanism's directional claim:
reuse preserves success, guards block stale reuse, persistence beats
single-use graphs.)

## What was built

- `src/arxiv_lab/harness/harnessmm.py` — Executable Decision Graph +
  guarded-reuse controller + RSI validation selection, stdlib only.
  Decisions are (exact action, literal args, causal dependencies); a
  receipt activates a certified decision only on full match. Task stream:
  each task has 4 routine steps (shared pool of 20, certified in a
  training phase), 2 novel steps, and 0–2 trap steps. Traps: the required
  decision differs from the stored one while the receipt claims the stored
  one (changed literal arg value, or changed dependency), or no receipt
  is presented at all. The naive control keys on action + identifier only,
  ignoring payload values and dependencies. RSI: validation tasks select
  between the guarded and naive configs (success first, cost second) and
  the winner is deployed on held-out tasks. Cost model: actor reasoning
  callback 100 tokens/step, certified substitution 5 tokens/step.
- `tests/test_harness_harnessmm_cpu.py` — 5 deterministic CPU tests
  (canonical seed 20261026, 400 held-out tasks; robustness seeds 1, 7, 42).
- `metrics.json` — full per-condition table for all four seeds.

## Results (canonical seed 20261026, 400 held-out tasks)

| condition | success | cost/task | trap fallback | false reuses |
|---|---:|---:|---:|---:|
| actor only | 1.000 | 708.5 | 1.000 | 0 |
| guarded (HarnessMM) | 1.000 | 328.5 | 1.000 | 0 |
| naive surface reuse | 0.420 | 256.1 | 0.297 | 305 |
| single-use graph | 1.000 | 704.9 | 1.000 | 0 |

Reading against the paper's claims:

- **C1 reproduced:** guarded reuse preserves actor success exactly
  (1.000 → 1.000) while cutting cost/task by 53.6% (708.5 → 328.5) —
  same direction as the paper's 29.54% (Codex) / 87.93% (DeepSeek)
  savings with success preserved or improved.
- **C2 reproduced:** naive reuse is cheaper still (256.1) but collapses
  success to 0.420 with 305 false reuses; it falls back on only 29.7%
  of traps (the no-receipt ones) and wrongly reuses the arg-change and
  dep-change traps. Guarded reuse falls back on 100% of traps with zero
  false reuses — the protection is the guard, not the reuse.
- **C3 reproduced (direction):** persistent graphs beat single-use
  graphs by 53.4% on cost (328.5 vs 704.9) at equal success. The
  single-use graph reuses only 15 steps: tasks sample routine decisions
  without replacement, so almost no within-task repeat exists.
- **C4 reproduced:** RSI validation selects the guarded config on all
  four seeds (validation success guarded 1.000 vs naive 0.460–0.520).

Robustness seeds 1, 7, 42: guarded success 1.000, trap fallback 1.000,
false reuses 0, cost saving vs actor 54.2–54.6%, naive success
0.487–0.517 on every seed.

## Honest limits / divergences

- Simulated decisions, a perfect actor, and call-equivalent token costs —
  no live LLM, no real HumanEval/Codex/DeepSeek harness, hence SUPPORTED
  rather than VERIFIED.
- The persistent-vs-single-use saving (53.4%) exceeds the paper's MBPP
  ablation (26.91% lower cost than single-use graphs). That is a property
  of this task mix (routine decisions essentially never repeat within a
  task), not a reproduction of MBPP; the direction matches and the
  magnitude should not be quoted as the paper's number.
- Two simulation bugs were caught and fixed during the run, and are
  recorded here rather than hidden: (1) the first naive matcher keyed on
  action + argument names, which wrongly reused novel steps too — it now
  keys on action + identifier, so only traps are wrongly reused;
  (2) the first trap generator made the required decision identical to
  the stored one, so naive reuse was accidentally correct — traps now
  genuinely require a different decision than the receipt claims.
