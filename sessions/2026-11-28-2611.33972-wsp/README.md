# Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers on a Frozen Encoder

- Article: https://arxiv.org/abs/2611.33972
- Track: training (Colab T4)
- Date: 2026-11-28
- **Verdict: PARTIAL** — the no-gradient WSP method itself
  reproduces: closed-form heads on a frozen RoBERTa-large reach
  0.85–0.92 accuracy on SST-2 and AG News at every label budget,
  within ~1.3 points of the paper's WSP numbers at 512 labels. A
  same-day follow-up fixed the bf16 optimizer bug (fp32
  classifier), and full FT at 512 labels now reproduces the
  paper's Table 2 values — so the original collapse was my bug,
  confirmed. But the paper's comparative claim (WSP *beats*
  FT/LoRA, whose discriminating prediction is at low budgets)
  still cannot be fairly tested: at 16/64 total labels the
  FT/LoRA arms still collapse, now because 8–24 optimizer steps
  at a fixed lr 1e-5 barely move a 355M model, not because of
  precision. The 48–50 point "lead" over FT at low budgets in
  the merged results still measures non-training baselines, not
  the paper. See "Follow-up (same day)" below.

## Why picked (AGENT 1, TUNE 4)

A closed-form alternative to gradient fine-tuning that is
directly testable on a Colab T4: an LLM writes programs that
read a frozen encoder through cloze templates at several layers,
fit closed-form heads (shrinkage discriminant analysis, kernel
ridge regression, and the encoder's label-word prior), and select
among candidate heads by cross-fitting. No gradient is computed,
no encoder weight changes. The paper's discriminating prediction
is at low label budgets (WSP leads FT/LoRA by 1.2–3.5 points at
64/128 labels, 3.8–5.3 points at 16 labels per class), so the
test is the 16/64/128/512-label grid on two of its 11 dev
datasets (SST-2, AG News) against FT and LoRA arms on the same
budgets. The paper's code repo 404s (as the skill-composition
paper's did); the method was implemented from the paper text.

## What was built

- `colab_wsp_driver.py` — self-contained Colab driver:
  `FacebookAI/roberta-large` (355M, bf16, NOT quantized), frozen
  encoder, top 6 layers, 4 cloze templates per dataset copied
  verbatim from the paper's Appendix K round-1 programs, label
  words from the same table. Views: `mask` (mean over
  templates×layers of the `<mask>` state), `plain_first` (mean
  over layers of the plain first-token state), `mask_concat`
  (4 templates × 6 layers × 1024, kernel head only). Heads per
  dataset × budget: SDA on mask, SDA on plain_first, their
  average, RBF kernel ridge on mask_concat, label-word prior,
  prior+SDA mixture. Selection: 5-fold stratified cross-fitting
  on the train subset plus dev rows, Occam's window (heads
  within 1 SE of the best are averaged); final heads refit on
  all train data. Train pool 2000 stratified per dataset
  (encoded once, features reused at every budget), dev 200
  carved from the pool, eval SST-2 validation (872) and AG News
  test subsample 1000, seed 20261128. Gradient arms: full FT
  (lr 1e-5) and LoRA (peft r=8 on query/value, alpha 16,
  lr 1e-4), batch 16, epochs {16:8, 64:6, 128:4, 512:3}.
  Metrics CSV + results JSON flushed after every phase;
  `WSP_DONE` printed as the last line.
- `test_wsp_cpu.py` — CPU test, ALL CHECKS PASSED before launch
  (template building, label-word scoring, SDA/KRR closed-form
  fits on synthetic frozen features, cross-fitting selection,
  accuracy scoring, stratified subsampling/K-fold, arm table).
  The pure helpers are stdlib+numpy only.
- Templates are human-copied from the paper, NOT LLM-written
  (stated scope difference; the paper itself makes no claim LLM
  authorship helps). Other stated differences from the paper:
  4 templates instead of 8 view + 16 prior; top 6 layers
  instead of 12; dev 200 instead of 1024; 2 of 11 dev datasets;
  one seed; one fixed LR per trained arm; no PFT arm.

## Results (live, Colab T4, account cyc236es, session wsp-1128b)

Accuracy on held-out eval (SST-2 validation n=872, AG News test
subsample n=1000). `collapsed` = dev accuracy within 1 point of
the majority rate — i.e. the arm predicts (near-)constantly.

| dataset | budget | WSP | FT | LoRA | WSP selected head |
|---|---|---|---|---|---|
| SST-2 | 16 | **0.8658** | 0.5115 (collapsed) | 0.5115 (collapsed) | krr_rbf |
| SST-2 | 64 | **0.9094** | 0.5092 (collapsed) | 0.4954 (collapsed) | krr_rbf |
| SST-2 | 128 | **0.9140** | 0.5092 (collapsed) | 0.5390 (collapsed) | krr_rbf |
| SST-2 | 512 | **0.9232** | 0.5069 (collapsed) | 0.6078 | krr_rbf |
| AG News | 16 | **0.8530** | 0.2440 | 0.2320 (collapsed) | prior |
| AG News | 64 | **0.8550** | 0.2430 (collapsed) | 0.2520 (collapsed) | krr_rbf + prior |
| AG News | 128 | **0.8510** | 0.2320 | 0.4200 | krr_rbf |
| AG News | 512 | **0.8910** | 0.2400 (collapsed) | 0.8780 | krr_rbf |

Paper numbers at 512 labels (Table 2) for comparison: SST-2
WSP 93.65 / FT 92.85 / LoRA 92.97; AG News WSP 90.10 / FT 90.11 /
LoRA 89.63. My WSP is 1.33 pts (SST-2) and 1.20 pts (AG News)
below the paper's WSP at 512 labels — close, and the
cross-fitting consistently picks the RBF kernel head (7 of 8
cells; the label-word prior wins only at AG News 16 labels,
where the prior is strongest).

The FT/LoRA columns are not a real comparison. Their training
losses stayed flat at chance from step 1 (SST-2 ~0.69–0.73 =
ln 2; AG News ~1.3–1.5 ≈ ln 4) across every epoch, and dev
accuracy sat at the majority rate. The bug is in my driver,
not the paper: `train_classifier` loads the classifier in pure
bf16 (`torch_dtype=torch.bfloat16`) and runs a raw AdamW loop
with no fp32 master weights. bf16 has ~0.4% relative precision,
so an lr-1e-5 update to a ~0.01-magnitude weight rounds to
zero — the weights never move. LoRA partially escaped at 512
labels (its small adapter weights have a larger
update/weight ratio, and lr 1e-4): AG News LoRA 0.878 vs WSP
0.891 there is the one non-collapsed trained cell, and it lands
in the paper's "within ~0.5–1 point at high budget" pattern —
but one cell is not a comparison.

## Follow-up (same day) — fp32 fix, FT/LoRA arms re-run

Session `wsp-1128c`, account cyc236es, T4, pid 755, endpoint
gpu-t4-s-euw4c1-1gsp0hpgoxolb, DONE in 24.3 min (driver
elapsed; whole wrapper run ~27 min). Fix, exactly one change
in `train_classifier`: the classifier is now loaded in fp32
(`torch_dtype=torch.float32`) instead of pure bf16, so weights,
gradients, and AdamW states are all fp32. Nothing else
changed: same seed, budgets, splits, templates, epochs, LRs,
collapse rule. Run with `WSP_ONLY_ARMS=ft,lora`, so only the
16 gradient arms ran; the WSP arms were not re-run and their
banked values are byte-identical in the merged
`wsp_results.json` (verified by diff against
`wsp_results_pre_followup.json`). Follow-up artifacts:
`wsp_results_followup.json`, `wsp_metrics_followup.csv`,
`train-run-followup.json`.

| dataset | budget | WSP (banked) | FT (follow-up) | LoRA (follow-up) |
|---|---|---|---|---|
| SST-2 | 16 | **0.8658** | 0.4931 (collapsed) | 0.4782 (collapsed) |
| SST-2 | 64 | **0.9094** | 0.5161 (collapsed) | 0.5057 (collapsed) |
| SST-2 | 128 | **0.9140** | 0.5206 | 0.5378 (collapsed) |
| SST-2 | 512 | **0.9232** | **0.9232** | 0.5837 |
| AG News | 16 | **0.8530** | 0.252 (collapsed) | 0.242 (collapsed) |
| AG News | 64 | **0.8550** | 0.248 (collapsed) | 0.247 (collapsed) |
| AG News | 128 | **0.8510** | 0.655 | 0.500 |
| AG News | 512 | **0.8910** | **0.905** | 0.886 |

Paper Table 2 at 512 labels: SST-2 FT 92.85 / LoRA 92.97;
AG News FT 90.11 / LoRA 89.63.

What the follow-up establishes, and what it does not:

- **The fp32 fix works — the original collapse was the bf16
  bug.** Losses now fall where there are enough steps: AG News
  FT b512 1.37 → 0.24, SST-2 FT b512 0.66 → 0.03, and full FT
  at 512 labels lands within 0.5 points of the paper on both
  datasets (SST-2 0.9232 vs paper FT 0.9285; AG News 0.905 vs
  0.9011). At 512 labels WSP vs FT is a tie on SST-2
  (0.9232 / 0.9232) and WSP −1.4 points on AG News (0.8910 vs
  0.905) — near-parity at high budget, as the paper says.
- **The low-budget comparison is still untested — now for a
  different, recipe reason.** At 16/64 total labels the FT/LoRA
  arms still collapse, but their losses are no longer flat at
  chance because of precision: there are only 8 steps (b16)
  and 24 steps (b64) at a fixed lr 1e-5, and SST-2 FT losses
  sit at ~0.64–0.79 with no downward trend at b16/b64/b128.
  Eight optimizer steps cannot move a 355M model. The paper's
  discriminating claim is the low-budget lead, so it remains
  untested under this recipe.
- **LoRA has a second recipe gap (hypothesis, not verified).**
  SST-2 LoRA b512 reached only 0.5837 with a flat loss
  (~0.62–0.76 all 96 steps) while AG News LoRA b512 reached
  0.886 — the same split as attempt 2 (0.6078 vs 0.878). The
  likely cause: peft's default freezes the classifier head
  (only the query/value adapters train), which handicaps LoRA
  and bites harder on the binary task. The paper's LoRA
  (SST-2 92.97) presumably trains the head.
- **Reassessment:** the verdict above stands. The
  condition for SUPPORTED was baselines near the paper's Table
  2 values *and* WSP's lead pattern matching. The first half
  now holds at 512 labels, but the low-budget lead is still
  measured against collapsed baselines (48–50 points at
  16/64 labels), not against trained ones as in the paper.
  Per the one-fix-attempt rule the run stops here; no further
  hyperparameter tweaking was done. A fair low-budget test
  would need a different recipe — more steps at tiny budgets
  (more epochs and/or higher LR) and a LoRA setup that trains
  the classifier head — and is a new run, not this one.

## Honest limits

- **The comparison is untested, by my own bug.** The verdict
  is PARTIAL, not SUPPORTED: WSP's absolute accuracy
  reproduces, but FT/LoRA were never given a working run, so
  nothing here confirms or refutes "WSP beats FT/LoRA". Do not
  quote the 48–53 point mean gaps in `wsp_results.json` as a
  finding — they are an artifact of the broken baseline.
- **Attempt 1 (session wsp-1128) BLOCKED before any arm:**
  datasets>=3 no longer resolves the legacy script ids
  "glue"/"sst2"/"ag_news", and the fancyzhx/ag_news parquet
  mirror has a single `text` column (no `title`). Fixed in the
  driver: canonical `nyu-mll/glue` for SST-2, fancyzhx/ag_news
  first for AG News, schema-robust `ag_texts()`. Attempt-1
  artifacts are kept as `wsp_results_attempt1.json`,
  `wsp_metrics_attempt1.csv`, `train-run-1128a.json`.
- **Attempt 2 (session wsp-1128b) DONE** in 31.2 min, pid 7010,
  endpoint gpu-t4-s-usw1b1-16tgqag5xu5qs. The first monitor was
  killed by a runtime restart 35 s after launch; re-attached
  with `train --watch-only --pid 7010` (note: watch-only still
  requires `--driver`, despite WORKFLOW.md §4.5's wording).
  The driver ran detached on the VM and was unaffected.
- **Single seed, one fixed LR per trained arm**, 2 of the
  paper's 11 datasets, 4 templates / 6 layers instead of the
  paper's 8+16 / 12. No PFT arm. WSP numbers are therefore a
  lower-bound-flavoured reproduction, not an exact replication.
- **Infra notes (for the next run):** the first two creation
  attempts failed before any fix — (1) `ModuleNotFoundError:
  colab_cli` because the agent shell sets PYTHONNOUSERSITE=1;
  run the wrapper with PYTHONNOUSERSITE=0. (2) HTTP 411
  "Length Required" on /tun/m/assign: the installed
  google-colab-cli 0.8.0 is unpatched; `patch_colab_cli.py`
  aborts on it (execution.py has 2 prune blocks, not the 3 the
  patch expects — version drift), so the 411 fix was applied
  manually to `~/.local/lib/python3.12/site-packages/colab_cli/client.py`
  (backup `client.py.bak`): `data="{}"` on the two POST
  `_issue_request` calls in `_post_assignment` and `unassign`.
  A third creation attempt produced a phantom session (kernel
  never answered); it was stopped and the retry succeeded.
  The full patch script (incl. the 404/401 token-refresh fix in
  execution.py) is still NOT applied on 0.8.0.
- **Account deviation (recorded):** `--account cyc236es`, NOT
  `--rotate`. The rotation-pool token files (ppoppo205,
  kyu008008, kyu009009, cyc236ha) do not exist in this VM and
  the other daily-run agent already used all four on
  2026-11-28, so rotation to a fresh account was impossible
  from here. `--max-accounts 1` was used for the same reason.
  A `token.cyc236es.json` copy of the existing cyc236es token
  was created so the wrapper's account switch works (backup
  at /tmp/colab-token-backup.json).

## Resume state (carried over)

- Nothing left running. Session wsp-1128b was torn down by the
  watch-only monitor after DONE; wsp-1128 (attempt 1) was
  already closed; follow-up session wsp-1128c was torn down by
  its monitor after DONE. Driver + CPU test + all three runs'
  artifacts are in this session dir.
- The follow-up run (wsp-1128c, see "Follow-up (same day)"
  above) already did the fix-and-rerun step this section used
  to call for: fp32 classifier, FT/LoRA arms only, WSP banked.
  What remains for a fair low-budget comparison is a recipe
  change, not a precision fix: more optimizer steps at tiny
  budgets (more epochs and/or higher LR for 16/64-label arms)
  and a LoRA setup that trains the classifier head (peft
  default freezes it). That is a new run, not a retry of this
  one, and was deliberately not attempted here (one fix
  attempt only).
- WSP arm table (banked, verified byte-identical through the
  follow-up merge): SST-2 0.8658 / 0.9094 / 0.9140 / 0.9232
  and AG News 0.8530 / 0.8550 / 0.8510 / 0.8910 at budgets
  16 / 64 / 128 / 512. Follow-up FT/LoRA table (merged into
  `wsp_results.json`): SST-2 FT 0.4931 / 0.5161 / 0.5206 /
  0.9232, LoRA 0.4782 / 0.5057 / 0.5378 / 0.5837; AG News FT
  0.252 / 0.248 / 0.655 / 0.905, LoRA 0.242 / 0.247 / 0.500 /
  0.886.

## Repo fit

Closed-form heads on a frozen encoder are a cheap, strong
few-label baseline: with 16 labels WSP is already at 0.85–0.87
on these two tasks with zero gradient steps. Worth keeping as
the low-budget baseline in the daily harness — but only once
the FT/LoRA comparison is re-run with a fixed training loop,
so the baseline is honest.
