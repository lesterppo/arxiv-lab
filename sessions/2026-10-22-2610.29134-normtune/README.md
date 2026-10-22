# Session 2026-10-22 — 2610.29134 — NormTune: Norm-Guided Parameter-Efficient Fine-Tuning

- **Article:** https://arxiv.org/abs/2610.29134 — "NormTune: Norm-Guided Parameter-Efficient Fine-Tuning for Reasoning-Intensive Tasks"
- **Track:** training (Google Colab T4)
- **Why picked:** AGENT=1/TUNE=5. Directly adaptable to the soup-daily-finetune daily NF4-QLoRA harness: the paper's claim is that PEFT under-adapts (smaller/uneven weight-update norms vs full fine-tuning) and that adaptively strengthening insufficiently-updated layers by their update norms closes much of the gap (+24.24% avg for rank-8 LoRA on AIME/MATH/GPQA).
- **Verdict: BLOCKED** — no fair test was possible: the Colab T4 session failed to provision (runtime proxy 404s; session reclaimed before first exec). Script staged below; resume state recorded.

## What was staged

`colab_normtune.py` — self-contained Colab script (pip installs at top, `pip uninstall -y torchao`, metrics CSVs to `/content/normtune/`):

- Model: **Qwen/Qwen3-8B** (Qwen3.5-9B is unloadable under the transformers<5.0 pin — verified 2026-10-21), NF4, rank-8 LoRA (alpha 16), bf16 — the paper's headline rank-8 setting.
- Task: 240 synthetic multi-step arithmetic-reasoning problems (train) / 80 held-out (eval), chain-of-thought traces, exact-match on `Answer: N`.
- Arm A (control): uniform LR 2e-4, 200 steps.
- Arm B (NormTune): 50-step warmup, per-layer update norms ||B@A||_F measured, below-median layers get 3x LR for the remaining 150 steps (per-layer param groups). Same data, same step budget.
- Records per-layer update-norm CV (paper's observation 1: norm-related under-adaptation) and eval exact-match per arm.

## Resume state

- Colab session name: `normtune-20261022` (T4); account kyu008008 (active 2026-10-22).
- Run: `colab.py exec_detach -s normtune-20261022 -f colab_normtune.py --log /content/normtune/run.log` (script must first be uploaded to `/content/colab_normtune.py`; the `upload` channel 404'd on the dead session — retry on a live one, or write via exec heredoc).
- Then `colab.py download -s normtune-20261022 /content/normtune <local>` and compare `eval_control.csv` vs `eval_normtune.csv`.

## Honest limits (when run)

- Control is uniform-LR LoRA, not full fine-tuning (9B-class full FT infeasible on a T4); the test is directional (does norm-guidance beat uniform LoRA at fixed budget), so the strongest attainable verdict is SUPPORTED, not VERIFIED against the paper's full claim.
- The paper's exact strengthening rule is described only in principle ("adaptively strengthens insufficiently updated layers according to their weight-update norms"); the 3x/below-median operationalization is ours.
