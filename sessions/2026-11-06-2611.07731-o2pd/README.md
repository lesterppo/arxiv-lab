# O²PD — Overlap-Gated On-Policy Distillation on Reliable Prefix

**Article:** arXiv:2611.07731 — "O²PD: Overlap-Gated On-Policy Distillation
on Reliable Prefix"
**Link:** https://arxiv.org/abs/2611.07731
**Date:** 2026-11-06 · **Track:** training (Colab T4) · **AGENT 2 / TUNE 5**

## Why picked

A fine-tune method directly adaptable to the daily-LoRA harness
(`soup-daily-finetune` pattern): on-policy distillation with two
concrete, testable design choices — align only on the student's
trajectory prefix (first 20%, claimed more reliable) and fix the
top-k support with a reset atom + overlap-gated alpha-divergence.
Strong TUNE fit: a distillation loss change, not a new architecture.

## Model choice

- Teacher: **Qwen/Qwen3.5-9B** (latest 9B class, tried first per
  standing picks).
- Student: **Qwen/Qwen3-8B** (proven training model).
- Both NF4. Weight-update test, 8–9B class, current-generation
  families only — hard floors respected. Probe: student loads at
  6.16 GB peak (GREEN).

## Design (bounded, one T4 day, GSM8K math)

- On-policy rollouts from the BASE student (1 per problem, temp 0.8,
  200 GSM8K train problems); all arms train from the SAME cached
  teacher supervision, so the comparison isolates the LOSS design.
  Teacher supervision cached offline (sequential loads — peak VRAM
  is one model at a time).
- Arms (same base, same data budget, same steps):
  - `sft` — completion-only CE on gold answers (control).
  - `opd_full` — top-k RKL (renormalized over teacher top-k support),
    full rollout context (the paper's baseline).
  - `o2pd` — prefix 20% only + reset-support atom + overlap-gated
    alpha-divergence (the paper's method).
- Eval: GSM8K test first 100, 2-shot, greedy, exact-match on the
  final number. Diagnostic: teacher top-1 agreement with the student
  token, prefix vs suffix positions (C1's reliability premise).
- Loss formulas (also checked on CPU, see below):
  support = teacher top-k + {reset}; p_reset/q_reset pool the mass
  outside top-k. Overlap o = sum min(p, q); alpha = clip(1 - o, 0.05,
  0.95); if o > 0.5 the reset atom is dropped (support converges to
  the aligned top-k intersection). D_alpha(p||q) =
  (1 - sum p^alpha q^(1-alpha)) / (alpha (1 - alpha)), averaged over
  prefix positions only (t < 0.2 * L).

## Results — CPU loss-math test

`test_o2pd_loss_cpu.py`: **12/12 pass** (pure-Python mirror of the
driver formulas) — reset support sums to 1; alpha->0 limit is RKL
KL(q||p) and alpha->1 limit is FKL KL(p||q); alpha decreases as
overlap grows; support converges (reset dropped) at high overlap and
is kept at low overlap; prefix gate keeps the first 20%; top-k RKL is
0 on match, positive on mismatch, and differs from FKL.

## Results — Colab run

- Session: `o2pd-1106`, account **cyc236ha** (rotation: next after
  2026-11-05 kyu009009), T4, monitor timeout 4h, done marker `O2PD_DONE`.
- Attempt 1 **DIED** on a driver typo (`topk_ids, topk_probs = [], [], []`,
  3 values into 2 targets) at the first teacher-scoring step, after both
  probes went GREEN (student 6.16 GB, teacher 7.73 GB) and all 200
  rollouts were saved.
- Attempt 2 (automatic fallback, session `o2pd-1106b`, same account/models,
  typo fixed, rollouts resumed) **DIED** in the `opd_full` arm's backward
  pass: `torch.AcceleratorError: CUDA error: device-side assert triggered`
  (scatter/gather index out of bounds).
- **Root cause (diagnosed from the downloaded artifacts): vocab mismatch.**
  Teacher top-k ids reach 248,069, but the student (Qwen3-8B) vocab is
  151,936 — Qwen3.5-9B has a ~250k vocab. `probs.gather(1, teacher_topk)`
  indexes out of range and kills the CUDA kernel (error surfaces
  asynchronously at backward). The paper's setup assumes a shared vocab;
  cross-generation Qwen teacher→student breaks that assumption. Only
  0.30% of teacher top-k mass is out-of-vocab (56.4 / 18,967.4 over
  2,513 positions) — small but fatal without a filter.
- **Fix staged in the driver (not yet run):** out-of-vocab teacher
  (id, prob) pairs are replaced by a dummy in-vocab id at prob 0 without
  renormalizing, so the dropped mass flows into the reset atom exactly
  like mass outside top-k; the dropped mass is recorded as a metric.
  Teacher max top-k id is now recorded in the diagnostics.
- **Partial artifacts banked:** teacher scoring complete (200/200,
  `o2pd_teacher.jsonl`); SFT arm trained (loss 0.433, 1 epoch, n=200;
  adapter in `adapters/sft.zip`, 174 MB); teacher diagnostic:
  top-1 agreement with the student's sampled token is 0.1566 on prefix
  positions vs 0.3628 on suffix (n=200). Caveat: agreement is NOT
  supervision reliability — the student samples at temp 0.8, so prefix
  tokens are noisier; this diagnostic does not test C1's premise and
  must not be read as evidence for or against it.
- **Resume state:** upload `o2pd_rollouts.jsonl` + `o2pd_teacher.jsonl`
  to /content (the driver resumes both); the fixed driver retrains all
  three arms from the cached supervision (~30–45 min) and then evals.
  16/200 rollouts were SEQ_CAP-truncated in teacher scoring (prompt +
  rollout > 256 tokens); the same truncation applies in training, so
  the loss sees identical lengths.
- No further automatic attempts (fallback spent). A retry needs fresh
  instruction.

## Verdict

**NEGATIVE** — the paper's headline claim did not reproduce at this
bounded scale: O²PD scored **0.01 (1/100)** on GSM8K test, below both
the full-context OPD baseline **0.14 (14/100)** and the SFT control
**0.05 (5/100)**; base 0.00 (0/100). All arms trained from the same
cached teacher supervision on the same 200 problems, same steps, same
eval — and the setup discriminates (opd_full clearly beats sft and
base), so this is a fair negative, not a broken test. The o2pd
generations are degenerate (blank/newline-only outputs, repetitive
doubling loops), while opd_full generations are coherent math text:
prefix-only (20%) training with the overlap-gated alpha-divergence
leaves the model's continuation behavior unconstrained, and at this
scale it collapses generation instead of improving it. Train losses:
sft 0.4323, opd_full 0.6839, o2pd 1.2638 (mean overlap 0.6253, mean
alpha 0.3746). Limits: single seed, 200 train / 100 eval, 1 epoch,
GSM8K only; cross-generation teacher with the vocab filter (0.30% of
teacher top-k mass dropped into the reset atom); base 0.00 reflects the
base model generating gibberish in this 2-shot format, so absolute
levels are low and the comparison is what carries weight. The
vocab-mismatch diagnosis from the BLOCKED attempts stands as a real
finding (cross-generation Qwen distillation needs the reset-atom
filter).

## Rerun — fixed driver (2026-11-06, Peter's order)

- Fix verified in `colab_o2pd_driver.py` before launch (py_compile OK,
  CPU test 12/12): OOV teacher (id, prob) pairs replaced by dummy id 0
  at prob 0 without renormalizing; dropped mass recorded
  (dropped_teacher_mass 56.4341 over 2,513 positions, 0.30% of top-k
  mass); teacher_max_topk_id 248,069 recorded. The fix held: opd_full
  completed 200 backward passes (attempt 2 died on the first one).
- Launch incident (aborted, no training lost): the first launch
  (session `o2pd-1106c`) silently failed to upload
  `o2pd_rollouts.jsonl` — the train wrapper's `--upload` uses
  `nargs="*"` with store action, so repeating `--upload` twice keeps
  only the LAST file (teacher landed, rollouts did not). The driver
  then regenerated rollouts fresh, which would have mismatched the
  cached teacher scores (conditioned on the original rollout tokens).
  Caught at ROLLOUTS 16/200, session stopped, relaunched as
  `o2pd-1106d` with both files in a single `--upload` flag; verified
  200/200 lines on the VM before training. Root cause is a wrapper
  arg-parsing quirk — worth a wrapper fix, not a driver fix.
- Session `o2pd-1106d`, account **cyc236es** (cyc236ha and ppoppo205
  gpu-unavailable at creation; wrapper cross-account retry). Rotation
  state deliberately left at `2026-11-06 cyc236ha`; tomorrow still
  picks ppoppo205. Active account restored to cyc236ha.
- Monitor died mid-eval (host-side); driver kept running detached;
  re-attached with `train --watch-only --pid 1333`. Outcome **DONE**.
  All 5 result files downloaded (metrics, details, 3 adapter zips).
  Mid-run snapshots + adapter zips were downloaded during the run as
  reclamation protection.
- Results (GSM8K test, first 100, 2-shot, greedy, exact-match):
  base 0.00, sft 0.05, opd_full 0.14, o2pd 0.01.
- Adapters (local-only, ~161 MB each, NOT pushed):
  `hidden_files/o2pd-1106-rerun2/{sft,opd_full,o2pd}.zip`.

## Files

- `colab_o2pd_driver.py` — Colab driver (pip + torchao uninstall,
  sequential NF4 loads, incremental metrics/adapters; vocab filter
  staged for the resume run).
- `test_o2pd_loss_cpu.py` — CPU loss-math test (12/12).
- `o2pd_metrics.csv`, `o2pd_details.csv` (empty — no eval ran),
  `o2pd_rollouts.jsonl`, `o2pd_teacher.jsonl` — downloaded artifacts.
- `adapters/sft.zip` (161 MB) — **stays local, NOT pushed** (exceeds
  GitHub's 100 MB file limit, same precedent as the 2026-11-04 UaS
  adapter); the SFT adapter weights live only on this VM.
- `train-run.json` — `colab.py train` outcome record (attempt 2, DIED).
- Rerun (2026-11-06): `o2pd_metrics_rerun.csv`,
  `o2pd_details_rerun.csv` (400 per-item rows), `train-run-rerun.json`
  (session `o2pd-1106d`, DONE). Rerun adapter zips are local-only in
  `hidden_files/o2pd-1106-rerun2/` (see rerun section).
