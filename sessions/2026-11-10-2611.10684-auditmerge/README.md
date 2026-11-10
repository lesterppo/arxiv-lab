# Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition

- **Article:** arXiv [2611.10684](https://arxiv.org/abs/2611.10684) — "Audit Before
  You Merge: Provenance, Probing, and Continual LoRA Composition"
  (submitted 7 Nov 2026)
- **Verdict: PARTIAL** (some claims tested, remainder blocked by two
  reclamations; resume state recorded below)
- **Track:** training track — Google Colab T4 via `hermes-colab-cli`
  (`colab.py train`)

## Why picked

- **AGENT 2 / TUNE 5.** A directly testable fine-tuning claim with a
  concrete protocol: audit LoRA adapters for provenance (base-model match,
  non-trivial delta norm) before merging, then compare naive merging vs
  Task Arithmetic (λ=0.5) vs a negative-coefficient coordinate search
  (math adapter probe-optimal at λ=-0.5). The paper reports naive merging
  at -10.5pp GSM8K / -18.3pp HumanEval vs Task Arithmetic λ=0.5 within a
  standard error of the unmerged baseline (Llama-3.1-8B-Instruct, 4 adapters).
- **Repo fit:** `soup-daily-finetune` daily-QLoRA harness — adapter auditing
  and merge-method selection are directly reusable there (and in
  `hermes-gi-egtkg-finetune`).
- **Testable within budget:** two small LoRA adapters + merge evals fit one
  Colab T4 session (bounded scale: 2 adapters instead of 4, MBPP instead
  of HumanEval, n=40/30 evals).

## What was built

- `colab_auditmerge_driver.py` — self-contained Colab driver (installs,
  data, two QLoRA trainings, audit, merge + eval arms; prints
  `AUDITMERGE_DONE` on completion; writes metrics CSV + eval details JSONL
  to /content incrementally per phase/arm):
  - **C1 audit** (paper's two checks): base-model match from
    `adapter_config.json`, and delta norm
    ||(α/r)·Σ B·A||_F over all LoRA pairs. Planted bad adapters: a
    wrong-base copy (config edited to `Qwen/Qwen3-4B`) and a zero-delta
    copy (all `lora_B` zeroed, as a fresh PEFT LoRA is). Genuine adapters
    must pass; planted ones must fail (driver asserts).
  - **C2 merge arms** (same base, PEFT weighted adapters): base (Task 0),
    math-only, code-only, merged TA λ=0.5, merged TA λ=1.0 ("naive sum"),
    merged math λ=-0.5 / code λ=+0.5. With 2 adapters, averaging the two
    merged models equals TA-0.5, so the naive arm is the unscaled sum.
  - **Data:** GSM8K train/test, MBPP train/test (MBPP stands in for
    HumanEval — same code-generation shape, runnable without the
    human-eval package). MBPP is loaded from the canonical parquet files:
    `google/mbpp` 401s on the Hub and the bare `mbpp` id breaks
    `datasets` URI parsing, so the driver downloads the parquet files
    directly (schema verified: train 374 / test 500).
  - **Training:** QLoRA r=16/α=32, NF4, bf16, lr 2e-4, micro-batch 2 /
    accum 2, seq 512, grad checkpointing, manual loop (FPRT-proven).
    `model.enable_input_require_grads()` after `get_peft_model` — required
    under transformers 4.x (see "Driver fixes" below).
  - **Evals:** GSM8K greedy 4-shot CoT (FPRT few-shot prompt), MBPP greedy
    with subprocess test execution (incl. `test_setup_code`).

## Run history (three attempts, one fallback)

| attempt | model | outcome | what happened |
|---|---|---|---|
| v1 | Qwen3.5-9B, kyu009009 | DIED ~3.5 min | `google/mbpp` 401s (DatasetNotFoundError). Driver bug, not a model failure. |
| v2 | Qwen3.5-9B, cyc236ha | DIED ~2 min | Bare `mbpp` id breaks datasets URI parsing (HfUriError). Driver bug. |
| v3 | Qwen3.5-9B, cyc236ha | **RECLAIMED** ~2h05m | Model loaded (peak 7.72 GB); math-LoRA training ran at GPU 100% for ~2h with no epoch line — Qwen3.5's linear-attention layers fall back to reference PyTorch kernels ("much slower"), so the 400-micro-batch math epoch was still running when the VM was reclaimed. Nothing survived (driver writes only at phase boundaries). |
| fallback 1 | Qwen3-8B, cyc236es | DIED ~7.5 min | `RuntimeError: element 0 of tensors does not require grad` at the first backward — transformers 4.x + gradient checkpointing in the manual loop. Driver bug; fixed with `enable_input_require_grads()`. |
| fallback 2 | Qwen3-8B, cyc236es | **RECLAIMED** ~09:25 UTC | Math LoRA done (below), code LoRA done, audit asserts passed, base/math-only/code-only/ta05 evals done; reclaimed during the TA-1.0 math eval. No files downloaded (metrics + eval details died with the VM); numbers below are reconstructed from the monitor's captured log tail. |

Model choice: standing 9B-first rule — Qwen3.5-9B tried first
(transformers==5.20.2; 4.x has no `qwen3_5` modeling — FPRT v3 lesson).
Fallback to Qwen3-8B per the automatic-fallback rule (2026-11-05), with
shrink fixes: math train 800→400, math test 60→40, MBPP test 40→30, code
epochs 2→1. Rotation: 2026-11-09 kyu008008 → v1 kyu009009 → v2/v3
cyc236ha → fallback cyc236es (cyc236ha and ppoppo205 were gpu-unavailable
at fallback creation; the wrapper's cross-account retry landed on
cyc236es). Rotation file stays `2026-11-10 cyc236ha`; the account that
actually ran is in each `train-run.json`.

## Results (fallback 2, Qwen3-8B — reconstructed from the log tail)

Math SFT: 400 examples, 100 steps, mean loss **0.4246**, peak **9.29 GB**.
Code SFT: completed (phase 5 ran) — loss line not captured in the log tail.
Audit (phase 4): asserts passed — genuine math/code adapters passed both
checks and both planted bad adapters were rejected — otherwise the driver
would have exited before phase 5. Exact delta norms were in the metrics
CSV that died with the VM, so C1 is supported by the assert-pass only,
not by recorded numbers.

| arm | GSM8K (n=40) | MBPP (n=30) |
|---|---|---|
| base | **0.850** (34/40, derived — see below) | **0.000** (0/30, derived) |
| math-only | 0.625 (25/40) [0.475, 0.775] | — |
| code-only | — | 0.067 (2/30) [0.000, 0.156] |
| **TA-0.5 merged** | **0.775** (31/40) [0.646, 0.904] | **0.067** (2/30) [0.000, 0.156] |
| TA-1.0 ("naive sum") | not run (reclaimed) | not run |
| math λ=-0.5 | not run (reclaimed) | not run |

Base numbers are derived, not directly logged: the driver records McNemar
contingencies for each merged arm vs base. For TA-0.5 vs base on GSM8K:
base-only-correct 6, merged-only-correct 3, χ²=0.444, p=0.505 → base
correct = 31 + 6 − 3 = 34/40. On MBPP: base-only 0, merged-only 2,
χ²=0.5, p=0.4795 → base = 2 + 0 − 2 = 0/30. The arithmetic is auditable
from the two McNemar rows in `recovered_metrics.csv`.

Reading (honest, underpowered at n=40/30):

- **GSM8K:** TA-0.5 (0.775) sits 7.5pp below base (0.85) — McNemar
  p=0.505, not significant; the 95% CIs overlap ([0.646, 0.904] vs base's
  implied [0.74, 0.96]). Directionally consistent with the paper's
  "within a standard error of the unmerged baseline", but the point
  estimate is lower and n=40 cannot resolve it. The single math adapter
  is *worse* than base (0.625 vs 0.85) — the adapter itself hurt, and
  merging partially recovered it. Likely causes (hypotheses, not
  conclusions — eval details were lost): GSM8K gold solutions carry
  `<<...>>` calculator annotations the adapter learned to emit, and 1
  epoch on 400 narrow examples at lr 2e-4 shifted a strong base toward
  terse outputs; n=40 noise cannot be excluded either.
- **MBPP:** floor effect — the base model scores 0/30, so the MBPP leg
  cannot discriminate merge methods. Qwen3-8B here is the *base* model
  (not -Instruct); a zero-shot instruction + ```python prompt does not
  elicit runnable code from it at all, and one epoch of MBPP SFT only
  lifts the code adapter to 2/30. The MBPP numbers (0, 2, 2) are
  uninformative about merging — a broken operationalization for this
  base model, not evidence about the paper's claim (which used
  HumanEval on an instruct model).
- **C2 (naive vs TA-0.5):** untested — TA-1.0 never ran. **C3 (negative
  λ):** untested — the math-neg arm never ran.

## Verdict: PARTIAL

The audit leg's asserts passed and the TA-0.5 merge was measured against
base on GSM8K (no significant difference, wide CIs) — but the naive and
negative-λ arms never ran, the MBPP leg is a floor effect, the adapters
and raw eval details were lost to the second reclamation, and the base
numbers above are derived from McNemar contingencies rather than directly
logged. The paper's headline contrast (naive forgets, TA-0.5 does not)
was not tested.

## Honest limits

- **Reconstructed numbers.** No metrics CSV or eval JSONL survived;
  `recovered_metrics.csv` is transcribed from the monitor's captured log
  tail (the only surviving record). Base accuracies are derived via the
  McNemar arithmetic shown above, not read from a log line.
- **n=40/30 — underpowered.** Every GSM8K CI spans ~25pp; the only
  claim the data can support is "no significant difference".
- **MBPP format fails on the base model.** 0/30 for base means the MBPP
  leg measures prompt-format compatibility, not merging. A rerun should
  use an instruct base or a few-shot code prompt.
- **Math adapter hurt the base.** 0.85 → 0.625 for the single math
  adapter is the most surprising number; without the eval details the
  cause (annotation leakage vs narrow SFT vs noise) cannot be separated.
- **2 adapters, not 4; Qwen3-8B, not Llama-3.1-8B-Instruct.** Bounded
  scale throughout; the paper's 4-adapter numbers are not reproduced.
- **No adapter artifacts.** math.zip / code.zip died with the VM twice;
  nothing is downloadable from this session.

## Resume state (for a rerun)

- Driver: `colab_auditmerge_driver.py` (current version includes the
  parquet MBPP load, `enable_input_require_grads()`, and the
  `CODE_EPOCHS` env override). Launch:
  `python3 colab.py train --driver colab_auditmerge_driver.py --rotate
  -s auditmerge-<date> --gpu T4 --monitor-timeout 14400
  --env MODEL_ID=Qwen/Qwen3-8B N_MATH_TRAIN=400 N_MATH_TEST=40
  N_CODE_TEST=30 CODE_EPOCHS=1 --results /content/auditmerge_metrics.csv
  /content/auditmerge_eval.jsonl /content/adapters/math.zip
  /content/adapters/code.zip --done-marker AUDITMERGE_DONE`
- What completed in fallback 2: data, math SFT (loss 0.4246), code SFT,
  audit asserts, base/math-only/code-only/TA-0.5 evals (numbers above).
- What is missing: TA-1.0 and math-neg arms, adapter zips, metrics CSV,
  eval details. A rerun should (a) use an instruct base or few-shot code
  prompt so MBPP is not a floor effect, (b) strip `<<...>>` annotations
  from GSM8K training targets, (c) flush metrics + adapter zips
  mid-run (the driver only writes at phase boundaries — both reclamations
  lost everything), and (d) run in a morning window: the 9B attempt died
  to slowness (reference kernels) + reclamation, and the 8B fallback was
  reclaimed in its third hour.
- Attempt accounting for the day: v1/v2 driver-bug relaunches, v3
  RECLAIMED (9B), fallback 1 DIED on a driver bug (fixed + relaunched
  once), fallback 2 RECLAIMED. Per the 2026-11-05 rule the automatic
  fallback is spent — a further attempt needs Peter's fresh instruction.

## Reproduce (what was run)

```bash
# fallback 2 (the run that produced the numbers above)
python3 colab.py train --driver colab_auditmerge_driver.py \
  --account cyc236es -s auditmerge-1110e --gpu T4 --monitor-timeout 14400 \
  --env MODEL_ID=Qwen/Qwen3-8B N_MATH_TRAIN=400 N_MATH_TEST=40 \
  N_CODE_TEST=30 CODE_EPOCHS=1 \
  --results /content/auditmerge_metrics.csv /content/auditmerge_eval.jsonl \
  /content/adapters/math.zip /content/adapters/code.zip \
  --out-dir colab-run-8b-v2 --done-marker AUDITMERGE_DONE
```
