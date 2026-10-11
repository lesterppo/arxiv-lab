# Session: SELF — Environmental Feedback Modeling in Hindsight Self-Distillation (2026-10-11)

**Article:** [arXiv:2610.11384](https://arxiv.org/abs/2610.11384) —
"Environmental Feedback Modeling Matters: Rethinking Feedback Treatment in
Agentic Hindsight Self-Distillation" (Guo et al.).

**Scores:** AGENT=4, TUNE=5.

## Why picked
Best training-track paper in the 2026-10-11 window and a direct fit for
the `soup-daily-finetune` harness (daily 8B QLoRA on Colab T4): it asks
whether *how* environmental feedback is used in agentic self-distillation
matters, and answers that jointly optimizing feedback modeling (predict
the env's response) with hindsight distillation beats conditioning the
teacher on feedback alone (+4.1pp over GRPO on τ-bench with Qwen3-8B).
If the mechanism holds at small scale, it is a drop-in auxiliary loss
for the production fine-tune loop.

## What was built
- `colab_self.py` — self-contained Colab T4 script (pip installs at top,
  `torchao` uninstalled per the 2026-10-04 quirk, metrics to
  `/content/self_results.json`). Deterministic calculator-task agentic
  env (running total; `add`/`sub`/`mul` actions; textual feedback incl.
  error cases); scripted BFS teacher (30% mistake-then-recover episodes);
  HER-style hindsight relabeling of failures. Two arms on Qwen3-8B NF4
  QLoRA (r=16/α=32, bf16, grad checkpointing): `sft_only` (hindsight
  distillation, loss on action tokens) vs `self_joint` (+ 0.5× auxiliary
  env-feedback prediction loss). Eval: held-out greedy rollouts (task
  success) + held-out feedback perplexity (the mutually-reinforcing
  claim). Importing the module has no side effects — env/teacher/data
  are CPU-testable.
- `test_self_cpu.py` — 7/7 CPU checks: env dynamics incl. error feedback,
  BFS shortest-path validity, teacher replay consistency, hindsight
  relabeling (failures become reached relabeled goals), mistake-episode
  error surfacing, eval-task solvability.

**Model note:** the run uses Qwen3-8B (the paper's own model) rather than
Qwen3.5-9B: Qwen3.5-9B is a multimodal `Qwen3_5ForConditionalGeneration`
build, so a text-only QLoRA run would needlessly load its vision tower on
the 15GB T4. Qwen3-8B is the closest text-only model at the 9B class.

## Results

**Live run BLOCKED — not executed.** Two Colab T4 sessions were lost to
free-tier reclamation today before the training could complete:

1. `arxiv-optchan` (kyu008008, after the Track A battery): freed for the
   fine-tune, then the ~45–75 min SELF script (`colab_self.py`) was
   launched — the session was reclaimed ~30–50 min in, before
   `/content/self_results.json` was written. No partial artifacts
   recoverable.
2. Retry `arxiv-self` (ppoppo205, trimmed script `colab_self_retry.py`:
   120 train / 28 eval, 100 steps/arm, detached via nohup): reclaimed
   ~15 min after creation, during pip installs.

Additionally, fresh GPU allocation on kyu008008 was refused twice
("precondition failed" — quota/capacity). The Colab-ready scripts are
staged below, fully CPU-tested; the live comparison is queued for a
stability window, not faked.

### 2026-10-11 ~15:20 retry (4B adaptation) — also blocked
- Adapted to **Qwen/Qwen3-4B** (`colab_self_4b.py`, then slimmed +
  single-arm `colab_self_4b_arm.py` with `SELF_ARM` env var): the staged
  8B scripts cannot run — Qwen3-8B SIGKILLs 5/5 at `from_pretrained` on
  the 12GB loader (proven 2026-10-06). Qwen3-4B chosen over Qwen3.5-4B
  deliberately (text-only; no vision tower on the 15GB T4).
- Four more T4 sessions reclaimed within ~15–25 min each (kyu009009 →
  cyc236ha → ppoppo205 → kyu008008); cyc236es `gpu-unavailable`.
  Saturday-afternoon peak reclamation; no arm completed.
- Queued cron **`self-retry`** (goal-owned `arxiv-daily-read-and-test`,
  every 2h, next run ~17:19 HKT 2026-10-11): rotates the 5 accounts,
  runs missing arms one-per-session, downloads
  `/content/self_results_<arm>.json`, updates this README, pushes to
  arxiv-lab, disables itself when both arms land. (Peter-approved.)

### 2026-10-11 ~19:19 retry — sft_only GREEN, self_joint reclaimed
- **Arm `sft_only` completed** (Qwen/Qwen3-4B, NF4 QLoRA r=16/α=32,
  bf16, 100 steps, 120 train / 28 eval): elapsed 719 s on a T4 session
  `arxiv-self-sft-only` (kyu009009). Held-out results:
  `self_results_sft_only.json` — rollout success **17/28 = 0.607**,
  feedback perplexity **49.29**.
- Arm `self_joint` launched detached on cyc236ha
  (`arxiv-self-self-joint`) but the session was reclaimed mid-arm
  (pip/model phase, ~13 min in) — no partial artifacts recoverable.
  Next cron occurrence retries `self_joint`.

## Verdict

**Not yet decisive — awaiting the `self_joint` arm.** The `sft_only`
control arm alone establishes the 4B/toy baseline: rollout success
17/28 (0.607) and feedback perplexity 49.29 after hindsight distillation
without feedback modeling. The paper's claim (+4.1pp over GRPO with
Qwen3-8B on τ-bench) can only be judged against this baseline once the
joint arm lands; at toy scale the comparison is directional, not proof.

## Honest limits
- Toy env (calculator task) vs τ-bench/AppWorld; 120 train episodes and
  28 eval tasks — effect sizes are directional, not paper-scale proof.
- 8B→4B adaptation: Qwen3-8B SIGKILLs the 12GB loader 5/5 on free T4, so
  the comparison runs on Qwen/Qwen3-4B instead of the paper's model —
  the +4.1pp claim's scale question stays open even if the direction
  replicates.
- Two arms only (no GRPO/SDPO baselines); the comparison isolates the
  auxiliary feedback-modeling loss, not the full SELF-vs-RL picture.
- Single seed; greedy eval rollouts.
