# 2610.31521 — Jev-DLC: Decision-Level Continual Learning from Typed Feedback

- Article: https://arxiv.org/abs/2610.31521 (submitted 22 Oct 2026, Feng et al.)
- Track: training, tested on Nemotron 3.5 Bolt E4B (live Colab T4 session,
  Ollama, temp 0, think off) — not a weight-training run. Bolt cannot be
  Soup-trained on this T4 (see 2026-10-25 #6: GGUF-only repo, Gemma 4 OOM),
  so this test uses Bolt as the paper's frozen scorer, which is exactly
  the role the paper assigns its scorer.
- Session dir: `sessions/2026-10-25-2610.31521-jevdlc-bolt/`

## Why picked

Screened fresh from the arXiv cs.AI API window 21–22 Oct 2026 (the weekend
gap meant no newer announcements; "fresh" = not previously tested).
Jev-DLC was a carried-forward near-miss from the 2026-10-25 daily run
(TUNE=3). It is the cleanest Bolt-API-testable training paper in the
window: its whole mechanism is bounded channels *above a frozen scorer*,
with no weight updates, no raw history, and no extra scorer call — so a
live Bolt scorer plus compact decision-level state is a faithful test
bed, unlike training papers that need a trainable base model.

## Paper claim (abstract + §4–6, read from the PDF)

Typed corrections become compact causal state above a frozen scorer.
Two channels: conservative (base logits + Dirichlet-smoothed count prior
over the preferred action's semantic identity, Eq. 5–6, α=0.5, weight 2.0)
and responsive (base logits + a correction vector updated by
minimum-change PA-I repair on the directed (preferred, rejected)
contrast, Eq. 7–9, γ=C=1). Combined by geometric mixing with reliability
weights updated by zero-one loss after each revealed feedback
(Eq. 10–11, η=0.25, σ=0.1). Predict-then-update protocol (§6.1): a
correction only helps later events. On TCPA-Repeat: Jev-DLC 44.83%
overall vs frozen scorer 39.44% and Recent-4 history replay 42.40%;
at feedback depth ≥ 2: 56.07%, 7.54 points above the strongest comparison.

## What was built

- `jevdlc_bolt.py` — self-contained VM-side driver (stdlib only).
  Frozen scorer = Bolt's first-token logprobs for the candidate letters
  via Ollama `/api/chat` with `logprobs` + `top_logprobs=20` (verified:
  `/api/generate` silently drops logprobs; chat returns clean single-token
  A–D logprobs). This mirrors the paper's JEV-SFT scorer, which scores
  option tokens at the first output position.
- Synthetic chronological stream (no TCPA-Bench release exists to reuse):
  8 users, each with a hidden fixed ranking over 6 generic action
  identities; each event shows a slate of 4 under a neutral context, and
  the target is the user's top-ranked identity in the slate. Contexts
  are unrelated to the ranking, so the frozen scorer cannot know the
  user — any gain must come from feedback state.
- Methods per event, all committing before the target is revealed:
  frozen scorer (argmax base logprobs); Recent-4 (same scorer re-prompted
  with the last 4 typed feedback lines — the paper's own §7.1 definition);
  Jev-DLC channels (conservative count prior + responsive directed-pair
  score, geometric mixing, reliability update).
- Graduated mechanism: `src/arxiv_lab/training/jevdlc.py` (pure
  decision-level state, no model) + `tests/test_training_jevdlc_cpu.py`
  (model-free, 24-event synthetic stream: frozen 0/24, Jev-DLC 23/24,
  plus causal, bounded-state, reset, and pair-transfer checks — ALL PASS).

## Results (live Bolt, 192/192 events scored, 8 users × 24)

User-macro accuracy:

| slice | frozen | Recent-4 | Jev-DLC | Jev-DLC − Recent-4 |
|---|---|---|---|---|
| overall (n=192) | 30.7% | 36.5% | 37.5% | +1.0 pt |
| depth ≥ 1 (n=184) | 31.0% | 37.0% | 38.0% | +1.1 pt |
| depth ≥ 2 (n=176) | 31.3% | 36.4% | 38.6% | +2.3 pt |
| depth ≥ 4 (n=160) | 31.3% | 37.5% | 39.4% | +1.9 pt |
| depth ≥ 8 (n=128) | 33.6% | 37.5% | 43.0% | +5.5 pt |

Per-user overall: Jev-DLC beats Recent-4 on 5 of 8 users, loses on 3.

**Verdict: SUPPORTED** — by the pre-registered rule (Jev-DLC > Recent-4
overall, depth ≥ 2 margin > 0, Jev-DLC > frozen). The direction matches
the paper at every depth slice, and the gain grows with feedback depth
(+1.0 → +5.5), as the paper's depth table predicts. Magnitudes are
smaller than the paper's (their depth ≥ 2 margin is +7.54 pt) — see limits.

## Honest limits

- This is a mechanism test, not a TCPA-Bench replication: synthetic
  stream, 8 users, generic actions. Absolute levels (37.5% vs the
  paper's 44.83%) are not comparable; only the ordering and the
  depth pattern are.
- Decision-level reconstruction, not the paper's hidden-vector PA-I:
  the responsive channel here is a tanh-bounded directed-pair count,
  not μ·d over a learned 32-dim projection (Ollama exposes no hidden
  states). The conservative channel and the mixing/weight update follow
  the paper's equations exactly.
- Margins are small and the user count is low (5/8 users win); this is
  directional support, not a strong replication.
- Discarded first run: an earlier run used `top_logprobs=5` and skipped
  every event where Bolt was confident enough that a letter fell out of
  the top 5 — selecting only low-confidence events and invalidating its
  verdict. It was discarded, the driver was fixed (top 20 + −20.0 floor
  for missing letters), and the reported run re-ran from scratch with
  the same seed. The reported run has zero scorer errors and 192/192
  events.
- Run notes: live run used the pre-portability driver copy with
  `OUT=/tmp/jevdlc_out`; the committed driver defaults OUT to its own
  directory (env `JEVDLC_OUT` overrides). Bolt serving was untouched by
  the run; supervisor heartbeats continued through it.

## Files

- `jevdlc_bolt.py` — live driver (run on the Bolt session VM)
- `jevdlc_events.csv` — per-event predictions for all 3 methods (192 rows)
- `jevdlc_summary.json` — user-macro accuracies by depth slice + verdict
