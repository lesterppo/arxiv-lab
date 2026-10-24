# Session 2026-10-24 — 2610.31483 "Training Small Language Models for Transferable Tool Selection"

- Article: https://arxiv.org/abs/2610.31483
- Track: training (TUNE=5)
- Verdict: **BLOCKED** (see below)

## Why picked

TUNE=5. Repo fit: the `soup-daily-finetune` daily-8B-NF4-QLoRA harness and the
Soup one-YAML fine-tune CLI both target small-model QLoRA on Colab T4 — a
small-model RL tool-selection method ports directly into that harness. The
paper's core claims: (1) RL post-training lets a small open model (Qwen3-8B)
drive tool selection over 10k+ tool corpora for complex multi-step
instructions, beating Llama-4-Maverick (402B) on ToolBench; (2) the skill
transfers to different retrievers and new tool domains.

## What was built

`colab_rl_toolselect.py` — self-contained Colab T4 driver (staged here, never
executed to completion):

- Phase 0: pip installs (`transformers<5.0`, bitsandbytes, accelerate, peft,
  datasets, safetensors), then immediately `pip uninstall -y torchao`
  (Colab's torchao 0.10 breaks peft's `get_peft_model`).
- Phase 1: deterministic synthetic corpora — 300 tools across 3 domains
  (finance/calendar in-domain, health held-out with paraphrased descriptions);
  200 train + 50 dev + 50 test multi-step tasks, 50 transfer tasks.
  Two simulated retrievers (keyword-overlap vs name-only, a harsher stand-in)
  to test "different retrievers".
- Phase 2: NF4 QLoRA (r=16, alpha=32, bf16) on Qwen3-8B.
- Phase 3: zero-shot baselines — in-domain test + transfer (both retrievers).
- Phase 4: bounded REINFORCE-with-baseline (60 opt steps × 4 rollouts, grad
  accum 2; reward = exact-sequence + 0.5·step-F1). **Documented stand-in:**
  the paper uses full RL/GRPO; trl-GRPO was judged too heavy for one free-T4
  run, so this is a bounded REINFORCE surrogate — stated honestly.
- Phase 5: post-RL evals — in-domain (both retrievers) + transfer
  (both retrievers). Metrics to `/content/toolselect_metrics.csv`
  (seq_exact, step_f1, first_tool_acc per split/state/retriever).

`colab_launcher.py` — detached launcher (Popen, start_new_session) used to
start the run; log at `/content/ts_run.log`.

## What happened (honest account)

1. Colab auth was expired (`token.kyu008008.json`, expired 2026-10-23).
   The CLI's built-in refresh **hangs behind the proxy** (google-auth
   `Request()` ignores proxy env; on refresh failure it falls into the
   interactive OAuth flow and waits forever). Workaround: manual
   `curl` refresh-token POST via proxy env — succeeded, token fresh
   until 2026-10-24 07:32 UTC. Diagnosis worth keeping.
2. Session `ts1024` (T4) created in ~5s; script uploaded; launched detached
   (PID 1003). Run crashed at data-gen: `make_task` chose the 2/3-step
   template independently of the number of sampled tools → `IndexError`.
   Bug fixed locally; data-gen re-verified on VM CPU (200/50/50/50 tasks,
   0 unresolved gold ids, parser OK, retriever recall@12 50/50).
3. Relaunched on `ts1024` (PID 3300) — session **reclaimed** by Colab
   free tier before producing output.
4. Retried on fresh session `ts1024b` (PID 1995) — **reclaimed again**
   (~25 min in, afternoon HKT = worst reclamation window).

Nothing ran past Phase 0 (pip installs). No model was loaded, no baseline or
RL step executed, no metrics CSV exists. **No run was faked.**

## Verdict: BLOCKED

No fair test was possible — two consecutive free-tier reclamations killed the
run during setup. Scripts staged and verified as far as possible (data-gen
unit-verified on VM CPU). This is an infra outcome, not a paper verdict.

## Resume state (for the next attempt)

- Staged script: `~/workspace/arxiv-lab-build/sessions/2026-10-24-2610.31483-toolselect/colab_rl_toolselect.py`
  (bug-fixed; the first-launched copy had the IndexError — do NOT reuse the
  on-VM copy, re-upload from here).
- Isolated CLI state: `~/workspace/goals/arxiv-daily-read-and-test/hidden_files/colab-state-20261024-ts1024.json`
  (currently `{}`; global `~/.config/colab-cli/sessions.json` untouched;
  active account symlink untouched — still `token.kyu008008.json`).
- Protocol: `python3 -m colab_cli.cli --config <state> new -s <name> --gpu T4`
  → `upload` script → `exec -s <name> -f colab_launcher.py`.
- Auth: if calls hang, the kyu008008 token is probably expired again —
  refresh it with the curl refresh-token flow (see §1), never via the CLI's
  interactive flow. Best retry window: early-morning HKT.
- Remaining work: full run (pip → ~16GB Qwen3-8B download → zero-shot
  baselines → 60 RL steps → post-RL + transfer evals), then download
  `/content/toolselect_metrics.csv` and `/content/ts_run.log`.

## Key numbers

- Zero-shot accuracy: not measured (blocked). Post-RL accuracy: not measured.
- VRAM: not measured. Optimizer steps completed: 0/60.
- Data-gen (CPU-verified): 200 train / 50 dev / 50 test / 50 transfer tasks;
  20 in-domain + 10 transfer tools; keyword-retriever full-gold recall@12
  50/50 (recall is easy at k=12/20 — selection+ordering is the real test).
- Infra: 2 sessions created, 2 reclaimed, 0 completed runs.

## Honest limits (even if the run had completed)

- 300-tool synthetic catalog, not 10k; no 402B baseline (no access) — a
  zero-shot 8B vs post-RL 8B comparison tests "RL improves small-model tool
  selection + transfer", not "beats Maverick".
- REINFORCE-with-baseline, not GRPO/PPO — a bounded stand-in, stated as such.
- Simulated retrievers and paraphrased descriptions, not real retrieval
  stacks or ToolBench.
