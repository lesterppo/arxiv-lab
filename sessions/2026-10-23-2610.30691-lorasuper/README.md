# Session 2026-10-23 — 2610.30691 "LoRA Superposition across Pretrained Models"

- **Article:** https://arxiv.org/abs/2610.30691 (cs.AI, published 2026-10-22)
- **Verdict: BLOCKED** (infra — Colab free-tier reclamation mid-run; scripts staged, resume state recorded)
- **Track:** training · **AGENT=1, TUNE=4**

## Why picked + repo fit
The paper claims LoRAs trained independently for the same task on different
pretrained backbones can be **summed and applied unchanged to all
contributing backbones**: exact sum improves all 56 backbone-task pairs
(+30.21pp mean over unadapted), lands within 0.28pp of solo LoRAs, while the
leave-one-model-out sum gives only +0.06pp; solo LoRA updates are nearly
orthogonal in parameter space (the frozen backbone acts as an implicit
selector). Directly relevant to the Soup / soup-daily-finetune LoRA
composition story — a verified superposition rule would change how daily
adapters are merged.

## What was built
- **`colab_lorasuper.py`** — self-contained Colab T4 script (pip installs at
  top, `pip uninstall -y torchao` after installs). Bounded replication:
  2 distinct ViT-B/16 CLIP backbones (`openai/clip-vit-base-patch16`,
  `laion/CLIP-ViT-B-16-laion2B-s12B-b42K`) × 1 task (CIFAR-10 4-class subset:
  300 train / 100 test per class). LoRA r=8/alpha=16 on q,k,v, 3 epochs;
  per-(backbone, arm) 5-epoch frozen-encoder linear probe isolates adapter
  feature quality. Arms per backbone: `none` (unadapted), `solo` (own LoRA),
  `sum` (exact sum of both LoRAs), `lomo` (the *other* backbone's LoRA).
  Deltas extracted as merged-minus-base; orthogonality = mean per-layer
  cosine between solo deltas. Metrics → `/content/lorasuper/metrics.csv` +
  `summary.json` (gain_sum_over_unadapted_pp, gap_solo_minus_sum_pp,
  gain_lomo_over_unadapted_pp per backbone).
- **`cpu_lorasuper_check.py`** — CPU pipeline validation (random-init tiny
  ViTs, synthetic 2-class task; does NOT test the paper's claim). Caught one
  real bug before the GPU run: delta extraction was also matching `.bias`
  tensors (LoRA `bias="none"` leaves them zero; their zero deltas would have
  dragged the orthogonality mean toward 0). Fixed in both scripts (`.weight`
  filter). Check passes: 6 delta layers/backbone, 8/8 probe arms, CSV
  written (`cpu_check_metrics.csv`).

## What happened
- Colab session `lorasuper-20261023` (T4, account kyu008008) provisioned
  cleanly on the first attempt (contrast 2026-10-22's two provisioning
  failures); T4 confirmed live via `torch.cuda`.
- `colab_lorasuper.py` launched via `exec_detach`; pip installs completed;
  CIFAR-10 download progressed to ~51% (~75 kB/s) when last observed.
- The session was then **reclaimed by Colab's free tier** (session entry
  gone, "No active sessions found on server"; `/content/lorasuper/run.log`
  unreachable). Nothing to tear down.
- Per the no-hammering discipline: one provisioning per day; no re-try loop.

## Resume state
Re-run: `python3 colab.py new -s lorasuper-<date> --gpu T4` →
`upload` this directory's `colab_lorasuper.py` → `exec_detach -f` with
`--log /content/lorasuper/run.log` → `download /content/lorasuper/` when
`summary.json` appears. Expected runtime ~60 min (dominated by the slow
torchvision CIFAR mirror; HF model pulls are fast). Then compare
`summary.json` against the paper's headline numbers:
gain_sum_over_unadapted_pp ≈ +30 (all pairs positive),
gap_solo_minus_sum_pp ≈ 0.3, gain_lomo_over_unadapted_pp ≈ 0,
mean_cosine ≈ 0.

## Honest limits (when resumed)
- Bounded scale: 2 backbones × 1 task (paper: 7 × 8). A positive result is
  SUPPORTED-at-bounded-scale, not a full replication.
- Per-(backbone, arm) linear probing differs from the paper's exact
  training protocol; it isolates adapter feature quality, which is what the
  superposition claim is about.
