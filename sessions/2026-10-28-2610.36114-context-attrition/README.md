# Context Is a Pipeline, Not a Window — deterministic CPU reproduction

- **Article:** arXiv:2610.36114, "Context Is a Pipeline, Not a Window:
  Information Attrition in LLM Agent Harnesses"
  (https://arxiv.org/abs/2610.36114) — cross-listed to cs.AI.
- **Scores:** AGENT=5, TUNE=1.
- **Why picked + repo fit:** the paper is a harness-internals mechanism in
  the exact lineage of this library's `src/arxiv_lab/harness/` modules
  (PrefillBudget, FinishGate, LoopDetector): a shipped harness behaviour —
  slicing tool-result JSON at a fixed length — silently deletes
  end-positioned data before the model ever sees it, while the context
  window looks nearly empty. The remedy (evidence-preserving truncation
  at the same byte budget) is directly reusable in a local harness.
- **Verdict: SUPPORTED** (exactly one controlled verdict word).

## What the paper claims

1. In a real kernel-trace fault-diagnosis agent (720 runs, largest prompt
   5.75% of the context window), a third of tool results (3,827/11,189 =
   34.2%) were cut by fixed-length JSON slicing, and **every** cut result
   became broken JSON.
2. Field survival is position-dependent: one field reached the model in
   4.7% of calls, another in 99.3% of the same results; which field held
   the fault's evidence depended on when the faulty process started
   (early/late in the trace), and diagnosis accuracy matched that split.
3. A controlled experiment: the cut's cost appears first as extra work —
   43% more tool calls, twice the computation, yield 0.61 → 0.31 — because
   the agent still finds the answer when it has another way to reach it.
   A replay of 864 real tool results includes a production trace where the
   shipped cut deletes all 128 records of the fault's evidence.
4. The authors leave open whether the cut truly *causes* the accuracy
   drop. A survey of seven frameworks finds two cutting the same unsafe
   way.

## What was built

- `src/arxiv_lab/harness/context_attrition.py` (graduated mechanism,
  stdlib only) + `tests/test_harness_context_attrition_cpu.py`.
- This session's `run_experiment.py` (runner) + `results.json` (all rows).

Synthetic kernel-trace tool results (seeded `random`): a `fault_evidence`
field placed **early** (2nd serialized field), **late** (last field), or
as **records_only** (the evidence is 128 faulty records at the tail of
the records list, mirroring the production trace). Three policies at one
fixed byte budget (11,900 chars = 66th percentile of generated result
sizes, calibrated so ~1/3 of early/late results are cut): `naive_slice`
(the shipped fixed-length cut), `field_aware` (parse, keep trace id +
evidence field + faulty records first, re-serialize valid JSON, record a
`records_omitted` count), and `no_cut` (control). The reader is a
deterministic rule standing in for the model: strict JSON parse, then
tolerant `fault_evidence` salvage (which succeeds iff the evidence object
lies entirely before the cut point), then a targeted re-query that always
returns the evidence field alone ("another way to reach it", max 2).
Yield = tasks solved / tool calls. 300 tasks per cell.

## Results (canonical seed 20261028, n=300/cell)

| policy | position | cut | valid\|cut | survival\|cut | 1st-acc | final-acc | calls | yield |
|---|---|---|---|---|---|---|---|---|
| naive_slice | early | 0.343 | 0.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.000 |
| naive_slice | late | 0.343 | 0.000 | 0.000 | 0.657 | 1.000 | 1.34 | 0.744 |
| naive_slice | records_only | 0.833 | 0.000 | 0.000 | 0.167 | 1.000 | 1.83 | 0.545 |
| field_aware | early/late/records_only | 0.343/0.343/0.833 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.000 |
| no_cut | all | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.000 |

Reading against the paper's four claims:

1. **Broken JSON: reproduced exactly.** Strict JSON validity among cut
   results is 0.000 under the naive slice in every arm and seed.
2. **Position split: reproduced directionally, starker than the paper.**
   Evidence survival among cut results is 1.000 (early) vs 0.000 (late)
   vs 0.000 (records_only) — the paper's 99.3% vs 4.7% split, with the
   simulation's deterministic positions pushing both ends to the
   extremes. First-attempt accuracy follows the same split (1.000 /
   0.657 / 0.167 overall, where the late/records figures equal the
   uncut fractions). The records_only arm reproduces the production
   trace: the cut deletes all 128 evidence records (survival 0.000 among
   cut results).
3. **Cost-first: reproduced directionally.** With the re-query fallback,
   final accuracy recovers to 1.000 everywhere, but the naive slice costs
   +34% calls (late) and +83% calls (records_only) with yield falling
   1.000 → 0.744 / 0.545 — the paper's +43% calls, yield 0.61 → 0.31, in
   direction if not magnitude.
4. **Window nearly empty: reproduced in scale.** The largest payload is
   9.1–10.6% of a 128k-char window while a third of results are cut
   (paper: 5.75%).

Robustness seeds 1, 7, 42 (same n): identical pattern — valid|cut 0.000
everywhere; survival|cut 1.000 / 0.000 / 0.000; late-arm calls 1.26–1.32
and yield 0.759–0.794; records_only calls 1.82–1.84, yield 0.544–0.549.
CPU test `tests/test_harness_context_attrition_cpu.py`: all checks pass
on all four seeds.

Two simulation bugs were caught and fixed mid-run, recorded here:
(a) the first `records_only` generator used n_records 96..160, so
`n_records - 128` went negative and flagged nearly every record faulty —
fixed to 140..168 so the 128-record tail is genuine; (b) the first budget
(8,200 chars) cut 98.7% of results — recalibrated to the 66th percentile
of result sizes (11,900 chars) for the paper's ~1/3 cut rate.

## Honest limits (why SUPPORTED, not VERIFIED)

- Synthetic corpus and a deterministic rule reader, not the paper's real
  kernel traces or a live LLM: salvage models an agent repairing partial
  JSON, and the re-query is guaranteed to succeed, so final-accuracy
  recovery and the exact yield magnitudes are regime-dependent.
- The paper itself leaves the accuracy-drop causation open; this test
  reproduces the attrition mechanism and the cost-first pattern, not a
  causal claim about LLM accuracy.
- Only the slicing behaviour is tested, not the seven-framework survey.
- The `records_only` arm cuts 83.3% (vs 34.3%) because those results are
  larger; the conditional (among-cut) comparison is the fair one there.

## Reproduce

```bash
python3 run_experiment.py   # writes results.json (this directory)
python3 ../../tests/test_harness_context_attrition_cpu.py
```
