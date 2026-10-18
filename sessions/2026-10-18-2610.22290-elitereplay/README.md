# EliteReplay (2610.22290) — training track session

**Article:** https://arxiv.org/abs/2610.22290 — *EliteReplay: GRPO + elite
quality-diversity archive replay for preserving solution diversity in RL post-training*
(testable training article; the agent-track pick for today is covered by a separate session)

**Date:** 2026-10-18 · **Fetch source:** n/a (assigned pick) · **Verdict: PARTIAL**

## Why picked + repo fit (AGENT 2 / TUNE 5)
A GRPO modification that pairs the policy-gradient loss with a replay loss on an
"elite" archive (distinct correct solutions mined pre-training via
quality-diversity search over the frozen base model). Directly adaptable to the
`soup-daily-finetune` NF4-QLoRA GRPO harness pattern (r=16/alpha=32, bf16,
transformers<5.0) — the daily 8B/9B LoRA track is exactly where an elite-replay
term would live. TUNE 5; AGENT 2.

**Core claims under test:** (1) GRPO narrows solution diversity; GRPO + elite-archive
replay preserves correct-strategy diversity while improving accuracy. (2) On
text-to-SQL: higher pass@128 than baselines, up to 77% more distinct correct
strategies, preserves most base-model correct strategies.

## What was built
- `elite_common.py` — synthetic text-to-SQL task (stdlib only): seeded problem
  generator (150 train / 60 eval, disjoint, all reference SQL verified executable),
  sqlite execution reward (0/1 result-set equality; non-SELECT → 0), SQL
  normalization fingerprint for strategy diversity, `EliteArchive` (distinct
  correct fingerprints, cap 200).
- `cpu_precheck.py` — 22/22 CPU checks PASSED: data gen, reward semantics
  (exact/wrong/non-SELECT/syntax-error/semantic-variant), fingerprint
  collapse vs distinction, archive dedup + cap, toy-policy archive build.
- `colab_elitereplay_9b.py` — self-contained Colab T4 driver (built by
  `build_driver.py` from `elite_common.py` + `colab_driver_body.py`): pip
  bootstrap (`transformers>=4.46,<5.0`, peft, bitsandbytes, trl, datasets,
  accelerate) then immediate `pip uninstall -y torchao`; NF4 QLoRA
  (r=16/alpha=32, all linears, bf16, paged_adamw_8bit, lr 2e-4); two arms —
  (a) GRPO-only, (b) GRPO + elite replay (custom `EliteGRPOTrainer` overriding
  `_compute_loss` with an auxiliary NLL term on archive samples, coef 0.3);
  G=4, 24 steps/arm, elite archive from frozen base (48 prompts × 8 samples,
  temp 1.0, cap 200); eval pass@1 (greedy, 30 problems), pass@8 + distinct
  correct strategies@8, pass@64 (8 problems); incremental CSV + status +
  final `results.json` in /content.
- `colab_log_partial.txt` — the VM-side log up to reclamation.

## Results
- **CPU pre-check: 22/22 PASSED** — data pipeline, sqlite reward, fingerprint
  diversity metric, and archive construction all verified locally.
- **Colab T4 run (`elitereplay`, account kyu008008): RECLAIMED.** Reached:
  pip installs OK, torchao 0.10 uninstalled, data gen OK (96 train / 60 eval),
  then hit a substantive finding (below). Qwen3-8B fallback download was in
  flight when the VM was recycled (~10 min into the run). No training steps
  executed; no metrics/results.json were produced (`/content` wiped).
- **Finding — 9B model not loadable via HF transformers here:**
  `Qwen/Qwen3.5-9B` fails with *"model type `qwen3_5` but Transformers does not
  recognize this architecture"* under `transformers>=4.46,<5.0` (the soup-pinned
  range). The driver's built-in fallback (`Qwen/Qwen3-8B`) engaged and was
  downloading when reclaimed. The proven `qwen3.5-9b-q4` recipe is Ollama-based;
  for HF-transformers GRPO the track must run at the 8B class (or wait for
  transformers ≥5.x qwen3_5 support).
- **Paper claims (1)+(2): NOT tested** — the (a)-vs-(b) GRPO comparison never ran.

## Verdict: PARTIAL
Harness + data + archive mechanics verified on CPU; the Colab training
comparison was reclaimed before any training step. Resume state recorded below —
a retry is a single re-run of the staged driver.

## Honest limits
- Claims (1) and (2) remain entirely untested; nothing here speaks to whether
  elite replay preserves strategy diversity or lifts pass@k.
- The 9B-class plan degraded to 8B-class (Qwen3-8B) by necessity; the 9B
  recipe is Ollama-only, incompatible with the HF GRPO driver.
- Synthetic text-to-SQL (5 templates, 2 tables) is far narrower than the
  paper's Spider-style benchmark; diversity counts are on template variants.
- GRPO hyperparams (G=4, 24 steps, replay coef 0.3) are budget-bounded, not
  paper-faithful.

## Resume state
- Session `elitereplay` (T4, kyu008008): reclaimed, VM gone, entry auto-pruned.
- Staged driver: `colab_elitereplay_9b.py` (this dir) — re-upload + detach-run
  on a fresh T4 session; fully idempotent (regenerates data, redownloads model).
- Isolated CLI pattern that worked:
  `python3 -m colab_cli.cli --config /tmp/elitereplay-state.json new -s elitereplay --gpu T4`
  (kyu008008 access token needed a manual refresh first — `oauth2.googleapis.com/token`
  via the stored refresh_token; raw CLI's own refresh fell back to interactive prompt).
- Note: another agent's sessions were never touched; shared `sessions.json`
  untouched; active account left as kyu008008.
