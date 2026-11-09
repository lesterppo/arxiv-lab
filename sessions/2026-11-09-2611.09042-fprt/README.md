# FPRT: Failure Profiling and Reachable Trajectory Selection for Reasoning Distillation

- **Article:** arXiv [2611.09042](https://arxiv.org/abs/2611.09042) — "Failure Profiling and Reachable Trajectory Selection for Reasoning Distillation" (submitted 6 Nov 2026)
- **Verdict: PARTIAL** (Stage 1 reproduced with full numbers; Stages 2–3 measured; the SFT comparison is blocked — see "Why PARTIAL")
- **Track:** training track — Google Colab T4, QLoRA

## Why picked

- **AGENT 2 / TUNE 5.** A fine-tuning-data method directly adaptable to the daily 8B NF4-QLoRA harness (`soup-daily-finetune`, `hermes-gi-egtkg-finetune`): profile the student into failure modes, have the teacher generate mode-conditioned corrections, and select the candidate the student itself finds most reachable (BPR). The paper reports +2.4 / +1.8 pts over LEMMA on 5 math benchmarks.
- **Repo fit:** a data-selection stage for the soup fine-tune pipeline — profiling + BPR is a reusable pre-training filter, independent of the base model.
- **Testable within budget:** GSM8K, N_PROFILE=160, K=5, N_CAND=5, N_EVAL=150; arms base / FPRT / control on one T4.

## Method (paper §§3.1–3.5, Table 1, Appendix D)

- **Stage 1 — profile:** K=5 student samples per problem; error rate α + error consistency κ give modes stubborn (α≥0.6, κ≥0.8), inconsistent (α≥0.6, κ<0.8), sporadic (0<α<0.6). τ_rep = most-frequent-wrong-answer trace (stubborn) else a random incorrect trace.
- **Stage 2 — teacher:** N=5 candidates per error problem with mode-specific prompts (reflection r + preserved prefix p + corrective continuation c).
- **Stage 3 — BPR:** average log-prob of the first L=8 sentences of c under the student, conditioned on q + p; select the argmax among correct candidates.
- **Training:** 90% input q⊕τ_rep → r⊕p⊕c, 10% q → p⊕c.
- Paper setup: teacher DeepSeek-V3.2, students Llama-3.1-8B / DeepSeek-Math-7B, MetaMathQA 395K profiled → 62K SFT examples.

## Bounded adaptation (one T4 day, stated honestly)

- GSM8K instead of MetaMathQA; N_PROFILE=160, K=5, N_CAND=5, N_EVAL=150.
- Student **Qwen/Qwen3-8B** (proven QLoRA student). Teacher **Qwen/Qwen3.5-9B** (NF4, generation-only) in v1–v3: a teacher-student gap is required and 9B is the only stronger model loadable on this T4, so the standing 9B-first pick is satisfied by the teacher role. **Fallback v4: teacher = Qwen/Qwen3-8B** (see below) — the gap is gone there.
- QLoRA r=16/alpha=32, NF4, bf16, lr 2e-4, 2 epochs, seq cap 768, micro-batch 2 + accum 2. Eval: greedy, 4-shot CoT, exact-match, Wilson CIs, McNemar FPRT vs control.
- One implementation deviation from the paper, recorded honestly: the teacher generates **one** raw per problem asking for all 5 trajectories in a single JSON array (parsed by `parse_candidates`), not 5 independent samples. Candidates are therefore all-or-nothing per problem on parse.
- Driver prints `FPRT_DONE` last; phases print `FPRT_FAILED:<phase>` and exit 1. Metrics CSV + profiles/candidates/selections/eval-details JSONL flush incrementally to /content (profiles per-problem, candidates per-problem after the retry loop).

## What was built

- `colab_fprt_driver.py` (v4, the version that ran last) — full pipeline: install → profile → teacher → BPR → base eval → FPRT SFT + eval → control SFT + eval → McNemar. Includes resume-from-profiles (`/content/fprt_profiles.jsonl` skips the ~2.5 h profiling phase) and a fail-fast `qwen3_5` registration assert after install.
- `fprt_train200.jsonl` (200 GSM8K train), `fprt_test150.jsonl` (150 GSM8K test).
- `fprt_profiles.jsonl` (160 rows), `fprt_candidates.jsonl` (48 recs), `fprt_selections.jsonl` (1 selection), `fprt_metrics.csv` — final results from the v4 run (session fprt-1109d).
- `paper.txt` — full paper text.

## Run history (four attempts, one day)

- **v1 (session fprt-1109):** stopped ~10 min in, driver fix — profiling generations had no stop string and rambled; `extract_answer` took the LAST number. Not a fallback-model attempt.
- **v2 (session fprt-1109b, kyu008008):** profiling COMPLETED (160 problems, 48 errors, peak 8.98 GB), then DIED at teacher load — `transformers` did not recognise `qwen3_5`. Root cause, verified locally: **no transformers 4.x release contains qwen3_5 modeling** (4.57.0 and 4.57.6 wheels have zero qwen3_5 files); the inherited `transformers>=4.50,<5.0` pin could never load Qwen3.5-9B. Fix: pin `transformers==5.20.2` (the pin every successful Qwen3.5-9B driver since 2026-11-03 uses).
- **v3 (session fprt-1109c, kyu008008):** resumed from profiles; teacher **LOADED Qwen/Qwen3.5-9B, peak 7.72 GB**; main teacher gen 48/48 DONE 09:45:47 UTC — then **RECLAIMED ~10:39–10:44 UTC during the solo-retry loop** (17 retries done, still going). No candidates were ever written (candidates flush only after the retry loop), so all teacher output died with the VM. Slowness cause: transformers 5.20.2 on this image lacks `causal_conv1d` / `flash-linear-attention`, so Qwen3.5's hybrid attention falls back to reference PyTorch kernels (~8.5 min per batch of 4).
- **v4 (session fprt-1109d, cyc236es — kyu008008 and ppoppo205 were gpu-unavailable, wrapper cross-account retry):** AUTOMATIC FALLBACK per the 2026-11-05 rule (one attempt, different model). Teacher Qwen3.5-9B → **Qwen/Qwen3-8B**; solo-retry loop REMOVED (the paper has no retry; it was the long tail that got v3 reclaimed); teacher max_new_tokens 1600 → 1024, batch 4 → 8. Driver DIED at the BPR gate (below) — no SFT ran.

## Results (v4, measured)

- **Profiling (Stage 1):** 160 problems, **48 with errors — inconsistent 20, sporadic 23, stubborn 5**. Failure modes exist in the expected proportions; the profiling machinery works end to end.
- **Teacher (Stage 2):** 48/48 generations, peak 9.04 GB. Parsed candidates: **5 total, from 1 problem** — **47/48 generations yielded zero parseable candidates** (n_candidates distribution: 47×0, 1×5; all 5 parsed candidates correct). Raw lengths averaged 3,586 chars (range 2,540–4,303), i.e. most generations ran to near the 1,024-token cap.
- **BPR (Stage 3):** **1 selection** (idx 52, sporadic, gold 57), mean margin selected-vs-mean **0.0191** (bpr_all −1.3765 ×4, −1.4718 — the 4 top candidates score identically, so BPR barely discriminates here). The driver's ≥10-selection gate then stopped the run: `FPRT_FAILED:bpr:only 1 usable selections (<10)`.
- **SFT comparison (Stage 4):** never ran — no base / FPRT / control accuracies, no adapters.

## Why PARTIAL (not BLOCKED, not NEGATIVE)

- Stage 1 is a genuine reproduction with measured numbers; Stages 2–3 were executed and measured (their output is the 1-selection finding, not an infra failure). Only the headline SFT comparison is blocked, by the driver's own BPR gate. That is exactly "some claims tested, remainder blocked."
- Not NEGATIVE: the 1-selection outcome is a property of this bounded adaptation — a **base** 8B teacher asked to emit a strict 5-trajectory JSON array inside 1,024 tokens — not a fair test of FPRT with the paper's strong chat teacher (DeepSeek-V3.2). The most likely causes are (a) truncation: most raws ran to near the token cap, so the JSON array never closed, and (b) the base model's weak strict-JSON following. Raw teacher texts were not saved (only `raw_chars`), so the two cannot be separated from this run — recorded as a gap, not a conclusion.
- Not BLOCKED: a fair test of profiling was possible and completed; the run's continuation status is "stop" per the fallback rule (v4 was the one automatic fallback attempt).

## Honest limits

- v4 teacher == student: no teacher-student gap, so even a completed v4 could only test the selection mechanism, not distillation from a stronger teacher.
- Candidates come from one JSON-array generation per problem, not 5 independent samples as in the paper.
- Teacher max_new_tokens 1024 in v4 (vs 1600 in v3) likely contributed to truncation; v3's 9B chat teacher never got to write candidates, so no parse-rate comparison exists.
- The BPR margin (0.0191) comes from a single selection (n=1); it says nothing about BPR's discriminative power in general.
- GSM8K at n=150 eval; the paper's +2.4/+1.8 pt effects would be inside noise at this scale even if the SFT arms had run.
- No HF token was set, so Hub downloads were unauthenticated and slow (9B fetch ~4:45 + weights ~1:12 in v3).

## Resume state (for a future retry — needs Peter's fresh say-so)

- Profiles (160 rows), candidates (48 recs), selections (1) are in this session; driver v4 resumes from the profiles file.
- A retry should restore the **9B chat teacher with 1600 tokens** (v3 proved it loads and generates), cap the retry loop instead of removing it (e.g. max 5 solo retries), **save raw teacher generations** (to separate truncation from format failure), and consider per-trajectory generation or a looser parser. The morning window is strongly preferred: v3 lived 2.8 h before evening reclamation, and the full pipeline needs ~2.5–3 h.
- Monitor-reliability lesson from this day (three separate incidents): a wrapper DIED/TIMEOUT report is not trustworthy on its own — the v3 monitor's DIED was false (driver alive), its TIMEOUT was a SIGTERM, and its log tail carried future timestamps. Always verify with a live `ps` on the VM plus the VM-side log before acting on a monitor verdict. Local monitors were SIGTERM'd by host sweeps four times today (~06:16, ~08:15, ~10:10, ~12:15 UTC); the VM-side driver always survived them.
