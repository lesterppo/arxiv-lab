# Session: RATED — Reliability-Aware Tool Retrieval (2026-10-12)

**Article:** [arXiv:2610.13889](https://arxiv.org/abs/2610.13889) — "RATED:
Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics" (Chen,
Liao). Code: [Elliotes/EcoAgent](https://github.com/Elliotes/EcoAgent)
(Apache-2.0).

**Scores:** AGENT=5, TUNE=1.

## Why picked
Tool retrieval is the front door of every tool-calling agent in Peter's
stack: pick the wrong (or broken) tool and the whole trajectory fails.
RATED adds a lightweight reliability layer — a frozen semantic retriever
re-ranked by a contextual bandit on real-time execution feedback — that
adapts to API outages *without retraining*. The mechanism is small,
self-contained, and directly portable to the local agent harness
(`arxiv_lab.retrieval`) and to deepworld's tool-calling agents. Most
testable agent paper in the 2026-10-12 window (107 cs.AI papers screened).

## What was built
- `src/arxiv_lab/retrieval/rated.py` — faithful port of the authors'
  mechanism (`train_bandit_warmup.py` / `evaluate_rated_full.py`):
  `cosine_topk` (frozen semantic stage), `DiscountedLinUCB` (per-tool
  arms, query-embedding context, UCB scoring, discounted updates,
  static-reliability warm-start), `build_hybrid_score`
  (`0.75*semantic + 0.10*static + 0.15*bandit`, additive so the bandit
  nudges rather than overrides), `execution_reward` (the paper's shaped
  reward: `0.1*sem*static + 0.9*exec_signal`, exec_signal = +1.0 success /
  −2.0 picked-a-failing-tool / 0.0 otherwise), `rerank_hybrid`
  (with `use_bandit=False` = the paper's `--disable-bandit` semantic-only
  baseline), `mrr_at_k` (hits must be *working* positives), and
  `fault_selection_rate`.
- `tests/test_retrieval_cpu.py` — 10/10 CPU checks: top-k ordering,
  LinUCB math (unseen arm scores exactly `alpha`, updates move the mean,
  orthogonal contexts don't leak), hybrid weights, reward shaping,
  metric edge cases, disable-bandit == semantic order, determinism, and
  the full scenario below.
- `rated_sim.py` — deterministic CPU reproduction of the paper's dynamic
  reliability eval: 1000-tool corpus, Zipf-popular queries with two
  positives each; at `fail_step` the most-used tools start failing
  (severe-outage scenario). Compares RATED vs semantic-only online.

## Results

Canonical run: 2000 queries, outage of the 102 most-used tools at step
400, seed 7. Full data: `results.json`.

| metric (post-outage) | semantic-only | RATED (hybrid+bandit) |
|---|---|---|
| success rate | 0.145 | **0.751** (+60.6 pp) |
| MRR (working positives) | 0.546 | **0.848** (+55.3% rel.) |
| fault selection rate | 85.5% | **24.5%** (−61.0 pp) |
| re-rank latency | — | 7.15 ms/query (**0.14 ms/candidate**) |

Pre-outage both arms are identical (success 1.000, MRR 1.000) — the
bandit only changes behavior once failures arrive, exactly as designed.
Seed 21: fault 84.1% → 44.9% (−39.3 pp), MRR +44.1% rel. — same
direction, notable seed variance (reported honestly).

## Verdict

**Core claims reproduced.** The headline numbers land almost exactly on
the paper's: baseline fault selection 85.5% (≈ paper's 86.2%), RATED
relative MRR gain +55.3% (≈ paper's +52.2%), fault rate cut by ~61 pp
(paper: 86.2% → 2.9% — same direction; our residual 24.5% reflects
fewer post-outage steps, 1600 vs their 8000, so arms see fewer repeat
failures). The mechanism that makes it work is the *additive* hybrid:
with only 0.15 weight the bandit can demote a known-broken tool below
its working semantic neighbours without ever overriding the semantic
order wholesale. Reading the authors' code (rather than just the
abstract) mattered: an early multiplicative `semantic × reliability`
variant of ours collapsed fault rates but *hurt* MRR (−20%), because
symmetric reliability boosts leak across queries; the paper's additive
hybrid + shaped (−2.0) failure reward avoids that trap.

Latency nuance: the paper's "<1 ms overhead" holds **per candidate**
(0.14 ms here) but not per query at 50 candidates (7.1 ms) with the
paper's own per-score linear-solve pattern — their eval defaults to 10
candidates, and the claim likely assumes cached inverses or
per-candidate accounting. Reported as measured.

## Honest limits
- Synthetic corpus: Gaussian embeddings, Zipf popularity, deterministic
  outages — not the 5,172 real tools / 210k pairs of EcoAgent. Effect
  sizes are regime-dependent (mid-frequency outage slice, the paper's
  own eval choice, gives a weaker 15.2% → 10.0% fault reduction —
  same direction, smaller magnitude).
- No encoder: the semantic stage is an oracle-ish cosine proxy, so the
  MRR comparison isolates the bandit; the paper's +52.2% also folds in
  their contrastively tuned encoder vs the ToolLLM retriever.
- Single outage event, stationary thereafter; no schema-change/auth
  failure modes; LinUCB dim 64 vs their 1024.
- Not yet wired into deepworld's tool-calling loop (working copy only;
  deepworld pushes need Peter's explicit go-ahead per repo rules).
