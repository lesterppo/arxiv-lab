# 2026-10-20 — 2610.26836 — Evidence substitution under retrieval

**Article:** [Same Simulation, Different Evidence: Where Retrieved Tool Output
Costs LLM Agents Accuracy](https://arxiv.org/abs/2610.26836) (cs.AI,
2026-10-19)

**Why picked (AGENT 5 / TUNE 1):** the paper studies exactly the
context-management choice our own harnesses make: long tool outputs are
kept out of context and retrieved via a section index. Its core finding —
agents compute the right value but answer from their own recomputed
outputs ("evidence substitution"), misrounding to a neighboring option,
costing GPT-5.1/5.2 7.9–11.3 pts; naming the authoritative output recovers
6.3–7.9 pts — is directly actionable for Hermes' file-based tool-output
design. Repo fit: `arxiv-lab` harness area; pairs with the 2026-10-19
SchemaShield-Lite session (tool-interface robustness).

## What was built

- `aqsim.py` — deterministic synthetic aqueous-speciation simulator
  (stand-in for PHREEQC): seeded water-chemistry params → 4-section report
  (INPUTS / SPECIATION / SATURATION / MINERAL_MASSES) with SI values to 6
  decimals; `make_question()` builds MC items whose keyed answer is the
  true SI rounded half-up to 3 decimals, with ±0.001 neighbor distractors
  (boundary-ambiguous values rejected by construction).
- `harness.py` — model-agnostic tool loop with four conditions:
  **A** inline (full report in context), **B** retrieval (report in file;
  tools `list_sections` / `read_section` / `run_simulation`),
  **C** retrieval + named authoritative output (the paper's fix),
  **D** retrieval with the agent's own prior simulation output pre-seeded
  (the paper's exact failure scenario: own high-precision outputs present).
  Includes the failure classifier (`substitution` / `retrieval_failure` /
  `other`).
- `test_evidence_substitution_cpu.py` — 17 deterministic CPU checks: simulator
  exactness, question keying over 40 items, oracle loop per condition,
  classifier on scripted substitution/blind agents. All pass.
- `run_model_backed.py` / `run_condition_d.py` — model-backed runs against
  `openai/gpt-oss-20b` via NVIDIA NIM (surrogate auth; client kept out of
  the repo). 15 fresh items × 4 conditions = 60 agent runs.

## Results

| condition | accuracy | errors | substitution errors |
|---|---|---|---|
| A inline | 15/15 (1.000) | 0 | 0 |
| B retrieval | 15/15 (1.000) | 0 | 0 |
| C named authority | 15/15 (1.000) | 0 | 0 |
| D own-output pre-seeded | 15/15 (1.000) | 0 | 0 |

The model consistently listed sections, read the SATURATION section (or its
own output in D), and rounded correctly. Retrieval cost: **0.0 pts**.
In C, the model ran its own simulation in 10/15 runs yet still answered
from the report — the authority instruction held.

## Verdict: NEGATIVE

The paper's core claim — retrieval-based context management costs accuracy
via evidence substitution — did not reproduce under this fair test:
zero accuracy cost and zero substitution errors in 60 model-backed runs.

**Honest limits:** single small model (gpt-oss-20b); synthetic simulator,
not PHREEQC; single-value lookup items, not 158 scientific MCQs; n=15 per
condition. The paper itself reports model heterogeneity (Claude showed no
retrieval cost either), so this negative is consistent with — not a
refutation of — the paper's broader picture. The "naming the authoritative
output" fix could not be evaluated (no deficit to recover; ceiling effect
on that sub-claim). Nothing graduated to `src/` — a negative banks no
mechanism; the harness and tests are kept here as the clean-negative
record. Raw results: `results.json`, `results_d.json`, `model_run.log`,
`d_run.log`.

## Resume state

None needed — all four conditions completed; no Colab involved.
