# Balancing CoT and answer supervision (arXiv 2611.03948) — training track

- Article: https://arxiv.org/abs/2611.03948 — "Balancing
  Chain-of-Thought and Answer Supervision for Small Language Models
  on Medical Question Answering" (submitted 2 Nov 2026).
- Why picked: AGENT=1, TUNE=5. The claim is a training-method
  claim with a clean two-arm comparison (uniform vs segment-weighted
  loss on CoT+answer sequences) that fits the daily-LoRA harness
  shape exactly. Repo fit: `hermes-gi-egtkg-finetune` (medical
  domain) and the training-method library; the paper's own finding
  is that general-purpose models benefit most on MCQ, which is what
  a Qwen general-purpose arm tests.
- Verdict: **BLOCKED** (no fair test of the paper's comparison was
  possible in this run; see the failure and the resume state below).
  The base arm did complete on Colab and its numbers are banked
  here; they are a reference point, not a verdict on the paper.

## What was built

`colab_cot_answer_driver.py` — MedMCQA SFT comparison on Colab:

- Data: MedMCQA train split, deterministic order (seed 11):
  240 train / 100 disjoint eval items, filtered to non-empty
  explanations (>= 40 chars), valid gold option, and
  prompt+CoT+answer <= 480 tokens. The same items and order feed
  both arms (paired comparison).
- Format: prompt = question + A–D options ending in `Reasoning:`;
  target = MedMCQA `exp` explanation (whitespace-collapsed,
  truncated to 600 chars) + `\nAnswer: X`. Completion-only loss.
- Arms: base (no adapter) -> UaS (w_cot=1.0, w_ans=1.0) ->
  SwS (w_cot=0.5, w_ans=1.5). Model: Qwen/Qwen3.5-9B first
  (9B-first rule), fallback Qwen/Qwen3-8B; NF4/bf16, LoRA r=16 /
  alpha=32, batch 2 x accum 4, 1 epoch, lr 1e-4.
- Eval: greedy generation (max 200 tokens) on the 100 eval items;
  the LAST `Answer: [A-D]` match is parsed; accuracy over all
  items, accuracy over parsed items, and parse-fail rate reported.
  Metrics + per-item details are written incrementally to /content
  (mid-run download protection).
- `test_cot_answer_cpu.py` — CPU test of the driver's pure logic
  (segment-weight masks, last-match answer parsing, CoT cleaning):
  PASSED (no model, no GPU).

## Results — what actually ran

Colab session `cotans-1104`, account `kyu008008` (rotation:
2026-11-03 cyc236es -> kyu008008), Tesla T4 (15.6 GB), isolated
registry. Model loaded first try: **Qwen/Qwen3.5-9B, peak 8.03 GB**
(no fallback needed). Data: train=240, eval=100 as designed.

| arm | accuracy (n=100) | accuracy over parsed | parse fail |
|---|---|---|---|
| base | 0.32 | 0.7273 (32/44) | 0.56 |
| UaS | — not run | — | — |
| SwS | — not run | — | — |

Base detail (from `cot_answer_details.csv`, 100 rows, verified):
44 parsed, 32 correct; gold distribution A31/B21/C22/D26; parsed
predictions A17/B8/C7/D12. Reading: the base model answers the
format rarely (56% of generations contain no `Answer: X` at all —
the prompt ends in `Reasoning:`, so the base model rambles), but
when it does produce the format it is right 72.7% of the time.
That split is exactly why the paper's comparison needs the SFT
arms: fine-tuning should first fix the format (parse fail -> ~0)
and only then does the UaS-vs-SwS accuracy gap become measurable.

## Failure: driver bug at UaS step 0 (DIED, not reclaimed)

After the base eval, UaS training crashed before the first
optimizer step (0/30) with:

```
File ".../colab_cot_answer_driver.py", line 252, in collate
    weights.append(b["weights"] + [0.0] * n)
KeyError: 'weights'
```

Cause (diagnosed from the traceback, not guessed): HF Trainer's
default `remove_unused_columns=True` drops dataset columns that
are not model-forward arguments before the collator runs, and
`weights` is such a column (`labels` survives because the model
accepts it). The collator therefore never saw `weights`. The
train monitor reported outcome DIED, downloaded the two
incremental files (metrics + details, base arm only), and the
session was torn down (server: no active sessions).

## Resume state (for a rerun on a fresh instruction)

- **Do not re-run silently**: per the one-attempt rule, a rerun
  needs Peter's fresh instruction. This session records the failure
  instead of retrying it.
- Fixed driver staged in this session (`colab_cot_answer_driver.py`,
  `remove_unused_columns=False` added to `TrainingArguments` with
  an explanatory comment). The fix is the standard remedy for
  custom loss columns, but it is **unproven on Colab** — the fixed
  driver has only passed `py_compile` + the CPU test, never a GPU
  run. A rerun should watch the first training step explicitly.
- Banked base results: `cot_answer_metrics.csv` (base row) +
  `cot_answer_details.csv` (100 per-item rows) in this session.
  A rerun driver could reuse the base row instead of spending
  ~38 minutes re-running base eval; the staged driver as written
  re-runs everything (base eval took ~38 min: greedy 200-token
  generations on the 9B hybrid model with reference-kernel
  fallbacks for its gated-delta layers are slow).
- Cost profile for a rerun: ~7 min installs + data, ~5 min model
  load, ~38 min base eval, minutes per training arm (30 steps),
  ~38 min per SFT eval — roughly 2.5 hours if all three evals run
  at base speed; SFT evals may be faster if the tuned model emits
  EOS after the answer (not guaranteed: training targets contain
  no EOS token).
- Train-run record: `~/workspace/goals/arxiv-daily-read-and-test/
  hidden_files/cotans-1104/train-run.json` (outcome DIED, PID 962,
  endpoint gpu-t4-s-ass1c0-13uxnp7i4iqzx); rotation file left at
  `2026-11-04 kyu008008`; active account left at kyu008008.

## Honest limits

- The paper's exact SwS weights are not in its abstract, so SwS
  here is a stated operationalization (answer tokens weighted 3x
  CoT tokens, loss normalized by total weight), not the paper's
  hyperparameters; the planned test was of the directional claim,
  not of exact numbers.
- MedMCQA substitutes for the paper's MedQA/EMS-MCQA/Med-O1
  datasets (same task format: medical MCQ; MedMCQA's own
  explanations serve as CoT). The free-text arm (Med-O1, where the
  paper finds UaS best) is not tested here at all.
- The base parse-fail rate (0.56) is partly an artifact of the
  eval prompt ending in `Reasoning:` with no format instruction;
  it measures format compliance under this prompt, not medical
  knowledge (parsed accuracy 0.7273 is the knowledge signal).
- Nothing graduates to `src/arxiv_lab/training/` (graduation needs
  VERIFIED/SUPPORTED).
