# GrepMem: lossless, index-free memory for long-horizon LLM agents

- **Article:** arXiv:2611.25123 — "Forget Freely, Grep Later: Lossless,
  Index-Free Memory for Long-Horizon LLM Agents" (Yang, Shi, Zhao, 2026)
- **Link:** https://arxiv.org/abs/2611.25123
- **Date tested:** 2026-11-20
- **Verdict: SUPPORTED**

## Why picked

- **AGENT 5 / TUNE 2.** GrepMem is a memory mechanism, not a training
  method: an append-only plain-text log of the agent's complete history
  plus a regular-expression search tool for on-demand recall. No
  embeddings, no retrieval index, no memory extraction, no extra model.
- **Repo fit:** directly reusable as a context policy in a local agent
  harness (Hermes AgentRunner memory). The local stack already keeps
  full trajectories; GrepMem says the working context can be dropped
  freely because the log is lossless and grep recovers what is needed.
  It is the memory-side companion to yesterday's Semantic Context
  Handoff (2611.24357), which separates state from evidence: GrepMem
  supplies the lossless store that SCH-style evidence recovery reads
  from.
- **Testable on CPU:** pure logic (log + regex + policies), deterministic
  evaluation, no model calls needed to isolate the mechanism.

## What was built

- `src/arxiv_lab/agent/grepmem.py` (graduated module, stdlib only):
  `AppendOnlyLog` (append-only lines + case-insensitive regex `grep`
  with a most-recent cap), a deterministic synthetic research-session
  generator (facts, updates where the latest value wins, distractor
  lines, paraphrased questions), five memory policies
  (`full`, `grepmem`, `summary`, `recency`, `random_mask`), and a
  deterministic QA oracle that is perfect given the facts in its
  context — so accuracy differences measure information preservation
  only, exactly the paper's claim.
- `tests/test_grepmem_cpu.py`: five live experiments with assertions.
- `rust/` — Rust second stage (see `rust/README.md`): same mechanism
  ported to Rust, `verify` binary, unit + deep tests, Python-vs-Rust
  parity sweep, soak at a scale beyond the Python test.
  **Rust leg verdict: VERIFIED** (independently re-run by the parent:
  `cargo test --release` 16 passed / 0 failed, `verify` ALL CHECKS
  PASSED with numbers identical to the Python `results.json`, parity
  sweep 53/53 cases with 0 mismatches). The port reproduces CPython's
  `random.Random` (MT19937) so generated sessions are byte-identical
  to the Python track; divergences documented in `rust/README.md`
  (single-pass grep, new `gen`/`eval`/`rng` subcommands, `regex` crate
  instead of `re`). Soak (20,000-line stream x 200 questions, 3.35 s):
  GrepMem 0.9850 — the 3 misses are genuine 40-line-cap behaviour at
  scale, not a port difference; full stays 1.0000, recency 0.1700.

## Results (Python CPU test, 5 seeds x 20 questions unless noted)

Main suite, 500-line streams:

| policy | accuracy | mean context lines | paraphrase acc |
|---|---|---|---|
| full (ceiling) | 1.000 | 500.0 | 1.000 |
| **grepmem** | **1.000** | **14.3** | **1.000** |
| summary (SUPO-style) | 0.200 | 44.0 | 0.071 |
| recency (last 60) | 0.250 | 60.0 | 0.071 |
| random_mask (20%) | 0.400 | 97.0 | 0.500 |

- **GrepMem matches the FULL ceiling (1.000 vs 1.000) at 2.9% of the
  context lines** — the paper's "lossless" claim, reproduced
  directionally (paper: 95–97% on ContextBench vs at most 21% for
  lossy baselines; here summary/recency score 0.13–0.25).
- **Scaling (200 → 2000 lines):** GrepMem stays 1.000 at every length;
  recency degrades 0.410 → 0.220, summary 0.250 → 0.130.
- **Grep budget sweep:** accuracy 0.610 at max 5 lines, 0.880 at 10,
  0.990 at 20, 1.000 at 40 — recall is budget-limited, not lossy.
- **Long-stream recall (1 buried fact, 1000 lines):** GrepMem 1.000,
  recency 0.400, summary 0.200.
- **Vocabulary mismatch (honest limit):** attribute-only grep on
  paraphrased questions scores paraphrase accuracy 0.000 — when the
  query vocabulary ("funding") does not match the log vocabulary
  ("budget"), grep finds nothing. The full GrepMem policy is saved by
  its entity-token patterns (paraphrase acc 1.000), but a question
  whose entity is also paraphrased would fail the same way. Grep
  quality is only as good as the patterns the agent issues.

## Honest limits

- Synthetic oracle evaluation, not a real LLM agent: the QA oracle is
  perfect given the facts, so the test measures information
  preservation, not an LLM's ability to issue good grep patterns or
  answer from retrieved lines. A real agent would score below these
  numbers on both GrepMem and FULL.
- The lossy-summary baseline is a simulation (first line + a seeded
  15% of fact lines per 25-line block), not a real summarizer; a strong
  summarizer would retain more than 0.20, a weak one less.
- `random_mask` (0.40–0.52) beats the paper's ≤21% lossy bound because
  keeping a random 20% of a 500-line stream still retains many facts —
  the paper's baselines are harsher than this control.
- Only the memory mechanism was tested; the paper's RL training of
  Qwen3.5-9B under the five strategies was not reproduced.

## Files

- `README.md` (this file), `results.json`, `test_grepmem_cpu.py`
- Graduated: `src/arxiv_lab/agent/grepmem.py`,
  `tests/test_grepmem_cpu.py`
- Rust second stage: `rust/` (`Cargo.toml`, `src/`, `tests/`,
  `README.md`, `results.json`)
