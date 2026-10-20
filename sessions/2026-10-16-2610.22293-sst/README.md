# Session 2026-10-16 — SST: Subspace Tuning (2610.22293)

**Article:** https://arxiv.org/abs/2610.22293 — "Towards Continual Learning:
Mitigating Forgetting With Subspace Tuning"

## Why picked (AGENT 2 / TUNE 5)

Directly adaptable to the daily-LoRA fine-tune harnesses
(`soup-daily-finetune`, `hermes-gi-egtkg-finetune`): SST is an
optimizer-agnostic recipe that restricts each weight update to the
*bottom* singular subspace of the weight matrix, protecting the
principal directions that carry prior-task knowledge. Composes with SFT
and self-distillation. A continual-finetune harness that learns a new
domain each day without destroying old ones is exactly the use case.

## What was built

- `src/arxiv_lab/training/sst.py` — `principal_directions`,
  `project_grad_to_bottom_subspace` (`(I−U_rU_rᵀ) G (I−V_rV_rᵀ)`, the
  faithful two-sided reading), `sst_update`.
- `tests/test_training_sst_cpu.py` — projection unit test (two-sided
  leakage < 1e-8, gradient non-degenerate) + sequential two-task MLP sim
  (20→64→2, synthetic binary tasks, seed 20261016).
- `colab_sst_9b.py` — scale-appropriate test staged for Colab T4:
  sequential LoRA (r=16/a=32, NF4, bf16) of Qwen3.5-9B, phase A finance
  glossary (240 pairs) → phase B medical glossary (240 pairs), arms
  "plain" vs "sst". SST implemented exactly as the two-sided projection
  via adapter reparameterization `B ← P_left B`, `A ← A P_right` after
  every optimizer step (SVD of frozen W0 per q_proj/v_proj, cached).
  Self-contained (pip installs at top, torchao uninstalled), resumable
  via `/content/sst_run/` markers, metrics to CSV/JSON.

## Results — CPU toy

| | plain SGD | SST (protect 0.9) |
|---|---|---|
| retention (task A after B) | 0.630 | **0.868 (+0.238)** |
| learning (task B) | 1.000 | 0.750 |

Pareto sweep: pf=0.5 → 0.850/0.835; pf=0.7 → 0.863/0.790;
pf=0.99 → 0.873/0.720. Full numbers in `results.json`.

## Verdict: SUPPORTED

The paper's retention claim reproduces directionally at bounded toy
scale with the faithful two-sided projection. Honest limits: (1) the toy
shows a real retention/learning trade-off — at this scale the free
bottom subspace is tiny, so target-task learning costs 0.17–0.28; the
paper's "preserved" target performance is the large-scale regime where
the bottom subspace is still enormous; (2) synthetic tasks share no
general-capability structure; (3) fidelity note: an earlier draft of the
module removed only the top-top block `U_r(U_rᵀGV_r)V_rᵀ` — that reading
showed NO effect (retention flat 0.623–0.637, drift not reduced); the
top-bottom/bottom-top blocks carry enough interference to destroy
retention, and only the two-sided projection reproduces the claim.

## Colab 9B track

**Phase A** (shared): Qwen3.5-9B NF4, LoRA r=16/a=32, finance glossary
(240 pairs, 3 epochs) → task-A accuracy **0.979** on 48 held-out prompts.
(Note: first eval read 0.188 — the eval used right-padding, which breaks
batched decoder generation; fixed to left padding.)

**Phase B arms** (medical glossary, 240 pairs, 3 epochs):

| arm | design | retention (A) | learning (B) |
|---|---|---|---|
| plain | continue phase-A adapter, unrestricted | **0.729** | 1.000 |
| sst v1 | continue phase-A adapter + project (B,A) each step | 0.417 | 1.000 |

The v1 sst arm is **INVALID as a test of the paper**: projecting the
phase-A adapter itself (`B ← P_left·B`, `A ← A·P_right`) amputates prior
knowledge stored in the adapter at step 0 — the paper protects base
weights and restricts only *updates*. The 0.417 (worse than plain)
measures this port flaw, not the paper's mechanism. Lesson recorded.

**Faithful v2 design** (staged, script pushed): freeze phase-A adapter
("default"), train a fresh adapter "b" (identical init both arms);
- "frozen": "b" unrestricted (isolates the freezing effect);
- "sst": "b" with the two-sided SST projection on "b" only, via the
  `B_b ← P_left·B_b`, `A_b ← A_b·P_right` reparameterization (exact).
At eval, "default"+"b" combine exactly by rank-concatenation
(`B_C=[B_A,B_B]`, `A_C=[A_A;A_B]`, r=32/a=64 keeps peft scaling 2.0) —
this peft version's `set_adapter()` takes one adapter only.

**Status: BLOCKED by reclamation.** The Colab T4 session
(`arxiv-sst-20261016`, account kyu008008) was reclaimed by the free tier
while the v2 arms were training. No v2 numbers were produced.

**Resume state:** script `sessions/2026-10-16-2610.22293-sst/colab_sst_9b.py`
(pushed, final); rerun = new T4 session + upload script +
`nohup python3 /content/colab_sst_9b.py >> /content/sst_run/run.log 2>&1 &`.
The script is fully resumable from scratch (deps → weights → phase A →
arms); SVD cache rebuilds automatically. Fixes already in the script:
requires_grad for `PeftModel.from_pretrained`, left padding for eval,
SVD-before-wrap ordering, bf16→fp32 projector cast, two-adapter design
with exact rank-concat combination.

## Reuse

`sst_update` / `project_grad_to_bottom_subspace` graduate to
`src/arxiv_lab/training/`; the Colab script's `B_b ← P_left B_b`,
`A_b ← A_b P_right` reparameterization (applied to the *new* adapter
only, never to the adapter holding prior knowledge) is the drop-in
pattern for adding SST to any LoRA training loop — per-step cost is two
small matmuls per target module. Portable lesson: in LoRA, the "weight"
whose update SST restricts is the *effective* update; the prior-task
adapter must stay frozen and unprojected.

## Colab confirmation (2026-10-18)

**未完成（session reclaimed）。** 早上 GPU 窗口連續被 free tier
回收兩台 T4：
- `sst-tfopd`（kyu009009，TF-OPD 用）：約 36 分鐘被回收
  （見 2610.22236 session README）。
- `sst-v2`（cyc236ha，本 SST 用）：約 47+ 分鐘被回收。

本輪 SST 進展：deps ✓、weights ✓（Qwen3.5-9B 直接 HTTP 下載約 1 分鐘）、
phase A finance LoRA 訓練到 ep1 step100（loss 2.945→2.826，下降中）。
metrics.json 尚未寫入（只在各 arm 結束時保存），且 markers 在
`/content/sst_run/` 下、隨 VM 一起丟失——「便宜 resume」實際等於
全新重跑（腳本本來就支援 from-scratch 全自動）。
誠實記錄：本早晨 free tier 回收壓力大，任何多小時 run 都建議
分段 cron 或 Pro 機器。腳本本身無問題，直接重用即可。

## Drive-persistent checkpoints (added 2026-10-19, unpushed working copy)

`drive_ckpt.py` (stdlib only) mirrors `/content/sst_run/` to the user's Google
Drive so a reclaimed VM resumes instead of restarting:
- `colab_sst_9b.py` calls `drive_ckpt.restore(RUN, SST_SESSION)` at startup
  (before any marker check) and `drive_ckpt.sync(RUN, SST_SESSION)` inside
  `mark()` after every completed phase. All failures are non-fatal — the
  driver falls back to /content-only checkpoints.
- Two backends: Drive FUSE mount (`/content/drive/MyDrive/arxiv-lab-ckpts/`,
  env `DRIVE_CKPT_DIR` overrides) or Drive REST API via stdlib urllib with
  an OAuth2 refresh token in `drive_token.json` (env `DRIVE_TOKEN_JSON`,
  default `/content/drive_token.json`). Sync is size-keyed and idempotent;
  restore only fires on a fresh VM (local markers win otherwise).
- `mint_drive_token.py`: one-time script Peter runs on his own machine to
  create `drive_token.json` (Desktop OAuth client + Drive API enabled).
  The token file is uploaded to the VM at launch and NEVER committed
  (scope `drive.file`: the driver can only touch files it created).
- Launch requirement: copy `drive_ckpt.py` to `/content/` next to the driver.
- Tested locally with a fake mount dir: sync -> wipe -> restore -> no-op
  sync, all OK. The REST backend is code-reviewed but not live-tested
  (needs a real token).

Reusable: SRD and any future training driver can import the same module
(restore at startup, sync after each phase); only the session name changes.

## Colab retry (2026-10-20, Peter-approved "三個都跑")

**RECLAIMED — partial.** Account cyc236ha, T4. Both `colab_sst_9b.py` and
`drive_ckpt.py` uploaded; no `drive_token.json` minted, so Drive sync
failed gracefully as designed (`[sst9b] drive restore: nothing to restore`,
run proceeded from scratch). The three arms run sequentially
(plain → frozen → sst):

| arm | retention (A) | learning (B) | status |
|---|---|---|---|
| plain | 0.750 | 1.000 | done (ep2 step150 loss 1.564) |
| frozen | **0.3125** | 0.6875 | done (ep2 step150 loss 1.571) |
| sst | — | — | **never ran** |

Phase A: task-A acc = 0.9792 (matches the 0.979 baseline).
The VM was reclaimed ~08:30–08:35 HKT, minutes after the frozen-arm
metrics were written — the SST arm (the actual mechanism under test)
never started. `metrics.json` was never written (script writes it per
arm end, but to `/content/sst_run/` which died with the VM; no Drive
token existed to persist it). Numbers above were captured from log
tails only.

Honest read: (1) the frozen arm retaining *worse* than plain (0.3125 vs
0.75) is suspicious, not a paper result — rank-concatenating a fresh
"B" adapter trained unrestricted on medical data apparently interfered
with the frozen A adapter at eval; this smells like a small-adapter /
toy-eval artifact (64 trainable tensors, 48 eval prompts), but it also
flags that the "exact" rank-concat combination is not interference-free
in practice; (2) the toy scale is the dominant caveat (synthetic
templated glossaries, 240 pairs/task); (3) Drive-checkpointing without a
minted token buys nothing — `mint_drive_token.py` still needs Peter's
one-time run on his own machine before any multi-hour SST run is safe.
The decisive arm (sst) remains untested; a future retry should target
morning GPU windows with the token minted.
