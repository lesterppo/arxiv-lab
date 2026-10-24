# Session 2026-10-24 — 2610.31483 "Training Small Language Models for Transferable Tool Selection"

- Article: https://arxiv.org/abs/2610.31483
- Track: training (TUNE=5)
- Verdict: **NEGATIVE** (for the bounded stand-in as tested — see limits)

## Why picked

TUNE=5. Repo fit: the `soup-daily-finetune` daily-8B-NF4-QLoRA harness and the
Soup one-YAML fine-tune CLI both target small-model QLoRA on Colab T4 — a
small-model RL tool-selection method ports directly into that harness. The
paper's core claims: (1) RL post-training lets a small open model (Qwen3-8B)
drive tool selection over 10k+ tool corpora for complex multi-step
instructions, beating Llama-4-Maverick (402B) on ToolBench; (2) the skill
transfers to different retrievers and new tool domains.

## What was built

`colab_rl_toolselect.py` — self-contained Colab T4 driver:

- Phase 0: pip installs (`transformers<5.0`, bitsandbytes, accelerate, peft,
  datasets, safetensors), then immediately `pip uninstall -y torchao`
  (Colab's torchao 0.10 breaks peft's `get_peft_model`).
- Phase 1: deterministic synthetic corpora — 300 tools across 3 domains
  (finance/calendar in-domain, health held-out with paraphrased descriptions);
  200 train + 50 dev + 50 test multi-step tasks, 50 transfer tasks.
  Two simulated retrievers (keyword-overlap vs name-only, a harsher stand-in)
  to test "different retrievers".
- Phase 2: NF4 QLoRA (r=16, alpha=32, bf16) on Qwen3-8B.
- Phase 3: zero-shot baselines — in-domain test + transfer (both retrievers;
  the baseline section runs 3 evals, no test/name-only baseline — see limits).
- Phase 4: bounded REINFORCE-with-baseline (60 opt steps × 4 rollouts, grad
  accum 2; reward = exact-sequence + 0.5·step-F1). **Documented stand-in:**
  the paper uses full RL/GRPO; trl-GRPO was judged too heavy for one free-T4
  run, so this is a bounded REINFORCE surrogate — stated honestly.
- Phase 5: post-RL evals — in-domain (both retrievers) + transfer
  (both retrievers). Metrics to `/content/toolselect_metrics.csv`
  (seq_exact, step_f1, first_tool_acc per split/state/retriever).

`colab_launcher.py` — obsolete; superseded by `hermes-colab-cli`'s `train`
command (create + health check + detached launch + supervised monitoring).

## What happened (honest account)

Day session (2026-10-24): two free-tier reclamations (`ts1024`, `ts1024b`)
during setup, after a proxy auth hang (raw CLI's google-auth refresh ignores
proxy env) and an IndexError in data-gen (fixed, CPU-verified).

Evening verification mandate ("improve the Colab CLI until a complete run
succeeds"), all on `hermes-colab-cli` v3.3 `train`:

1. **verify** (`toolselect-verify`, cyc236ha): baselines completed, then RL
   crashed on first backward — `sample_traj` was `@torch.no_grad()`, so its
   log-prob had no grad_fn. Outcome DIED. Fix: sampling stays no-grad;
   new `logprob_of()` recomputes summed log-prob WITH grad; per-rollout
   backward; adv==0 rollouts skipped. Offline-verified with a toy model.
2. **verify2** (`toolselect-verify2`, cyc236ha): baselines reproduced
   exactly, then DIED on first `sample_traj` — the grad-fix rewrite had
   dropped `return_dict_in_generate=True`. Fix: restored (one line).
3. **verify3** (`toolselect-verify3`, cyc236ha): full run completed
   (baselines + 60 RL steps + 4 post-RL evals + results JSON), but the
   monitor reported DIED — the driver printed `DONE. metrics` BEFORE a
   ~56-line JSON, so the marker fell outside the monitor's 40-line tail
   window and the normally-exited process read as dead. Complete CSV
   preserved as `toolselect_metrics_complete.csv`. Fixes: monitor now
   greps the whole log for the marker; driver prints the marker last.
4. **verify4** (`toolselect-verify4`, cyc236ha): RECLAIMED at RL step 30/60.
   Log showed RL reward saturating: step 10 mean_reward 1.208, steps 20–30
   at the ceiling 1.500 with loss 0.000 (all rollouts perfect → all
   advantages zero → no learning signal). Record: `train-run-reclaimed4.json`.
5. **verify5** (`toolselect-verify5`, cyc236es via fallback — rotation
   picked ppoppo205, creation failed there, `train` fell back; the
   rotation state file was NOT updated by fallback, fixed afterwards in
   the CLI + state file corrected to cyc236es): **DONE**, full run
   ~2h50m, CSV downloaded, session torn down. Four runtime restarts
   killed the monitor mid-run; each time the detached driver survived
   and the monitor re-attached in watch mode. Final CSV is byte-identical
   to verify3's — the result is deterministic across two full runs.

## Verdict: NEGATIVE (bounded stand-in)

Bounded REINFORCE post-training did **not** improve tool selection:

| split / retriever | zero-shot seq_exact | post-RL seq_exact |
|---|---|---|
| in-domain / keyword | 0.92 | 0.86 (−6pp) |
| in-domain / name-only | — (no baseline run) | 0.88 |
| transfer / keyword | 0.88 | 0.86 (−2pp) |
| transfer / name-only | 0.86 | 0.88 (+2pp) |

step_f1 and first_tool_acc moved by ≤1pp everywhere. The zero-shot model
was already near ceiling (0.92/0.88/0.86), and RL reward saturated by
step ~20 (mean_reward 1.500, loss 0.000 through step 60 in verify5):
the training tasks were too easy for the already-strong model, so the
only gradient signal came from the early steps — and the net effect on
held-out tasks was flat-to-negative. This refutes the bounded stand-in,
not the paper's full method (see limits).

## Key numbers

- Zero-shot (50 tasks each): test-kw 0.92, xfer-kw 0.88, xfer-nameonly 0.86;
  reproduced identically 5 times (verify, verify2, verify3, verify4, verify5).
- Post-RL (50 tasks each): test-kw 0.86, test-nameonly 0.88, xfer-kw 0.86,
  xfer-nameonly 0.88; reproduced identically twice (verify3, verify5).
- RL: 60/60 steps; reward 1.208 @ step 10 → 1.500 @ step 20 → 1.500 @ step 60;
  loss 0.162 → 0.000. Adapter saved to `/content/ts_adapter` (not downloaded).
- VRAM: 6.25 GB after load+LoRA; 6.62 GB after RL. Trainable params 43.6M.
- Run time (verify5): ~2h50m launch → DONE (download + baselines ~50m,
  RL ~70m, post-RL evals ~50m).

## Honest limits

- 300-tool synthetic catalog, not 10k; no 402B baseline (no access) — a
  zero-shot 8B vs post-RL 8B comparison tests "RL improves small-model tool
  selection + transfer", not "beats Maverick".
- REINFORCE-with-baseline, not GRPO/PPO — a bounded stand-in, stated as such.
  The paper's full method (larger scale, harder training tasks, GRPO) could
  still work; this test says only that 60 steps of REINFORCE on easy
  synthetic tasks does not help an already-strong 8B.
- Simulated retrievers and paraphrased descriptions, not real retrieval
  stacks or ToolBench.
- Baseline section omits test/name-only, so post-RL test-nameonly 0.88 has
  no zero-shot counterpart.
- Reward ceiling: with reward = exact + 0.5·F1, a perfect rollout scores
  1.5; once all 4 rollouts in a step are perfect, advantages are zero and
  the step teaches nothing — the training set needs harder tasks for RL
  to have signal beyond step ~20.
