# A Matrix with Steps (arXiv:2611.20230) — session record

**Article:** Paulhart & Schüller, "A Matrix with Steps: An Episodic and
Procedural Memory Method for Tool-Using Agents and Robots",
https://arxiv.org/abs/2611.20230 (v1, 2026-11-17).

**Verdict: SUPPORTED** — the mechanism reproduces directionally at
bounded scale with stated limits (see "Honest limits"). All eight
experiments below were run live on CPU; nothing is a theater run.

## Why picked (AGENT 5)

Direct fit for the hermes agent harness's episodic memory and for the
arxiv-lab mechanism library: a two-part memory (delta-rule associative
matrix as finder + ordered step records as source of truth) with a write
gate that admits only externally verified outcomes — the exact failure
mode the hermes harness needs (agents that repeat what failed and learn
from runs they only *believe* succeeded). No training track: the paper is
a mechanism paper, tested here in the VM's local agent stack.

## What was built

- `src/arxiv_lab/memory/matrix_steps.py` (graduated) + `tests/test_matrix_steps_cpu.py`
  (copy in this session): delta-rule matrix M (Eq. 1), hashing-trick
  bag-of-words embedder (3000-word vocabulary), per-procedure sign
  vectors (64-bit FNV-1a hash -> Mersenne Twister, low bit = sign),
  step records, write gate, deterministic consolidation, two-stage
  first-attempt promotion test, instant demotion, erasure by rebuild.
  Stdlib + numpy only, no model calls, no network.
- `rust/` — Rust second stage (std only): full port + `verify` binary,
  14 unit + 7 deep tests, parity sweep, soak. See `rust/README.md`.

## Results (measured vs paper)

| experiment | paper | measured (Python / Rust) |
|---|---|---|
| 1. write gate + scope isolation | model-only success never recalled; no cross-scope episode returned | model-claimed success excluded; bob's episode never returned for alice; "done" over a failed tool call recorded as failure; person verdict overrides |
| 2. hidden criterion (Plan 4), 200 tasks | self-judged-stops: 96% reported, 35% solved, 1.46 att/task, 110 damaging; verified: 79% solved, 3.25 att/task, 174 damaging; ceiling ~79% | self-judged-stops: **93.5% reported, 21.5% solved**, 1.38 att/task, 144 damaging; verified: **84.5% solved = ceiling 0.845**, 2.14 att/task, 150 damaging; no-memory 83.0% solved, 3.06 att/task, 318 damaging |
| 3. capacity, d=1024 square, 200 procs | found 1.000, next-step 0.997 | **found 1.000, next-step 0.928**, median lookup 2.67 ms |
| 3. capacity, 500 procs | found 0.726, next-step 0.952 | **found 0.994, next-step 0.877**, median lookup 5.36 ms |
| 4. consolidation | >=5 episodes, >=3 successes, top sequence >=60%; confidence (s+1)/(n+2) | no record below any minimum; record at 6 ep / 4 succ with confidence **0.625 = 5/8**, fixed args + failure counts correct |
| 5. revisit first-attempt | every revisit solved at first attempt up to ~10k records | **1.000 on 3000 revisits, 1000 situations** (Rust soak: 1.000 on 30 000 revisits, 10 000 situations) |
| 6. erasure of 10% | erased situations return to no-memory level, others unchanged | **20/20 erased not returned, 180/180 kept found**; max cosine of an erased key to its erased value **0.242** (< 0.50); audit names record + cause |
| 7. demotion | solved 76% -> 97% | **0.760 -> 0.970**, 44 trips -> 1 trip, 1 demotion on first stop |
| 8. promotion honesty, 200 windows | identical variant 0/200; naive rule 93-98/200 | **identical 0/200 promoted (naive 84/200)**; better variant (30% -> 60%) **118/200 promoted** (paper: 64%) |

Rust leg: `verify` **ALL CHECKS PASSED** in ~104 s with numbers
identical to Python on every discrete metric; `cargo test --release`
**21 passed, 0 failed**; parity sweep **26/26, 0 mismatches**;
`soak` **PASSED** (10k revisit 1.000, 4-thread identity bit-for-bit).

## Honest limits

- **Next-step accuracy is below the paper's**: 0.928 vs 0.997 at 200
  procedures, 0.877 vs 0.952 at 500. The finding accuracy matches or
  beats the paper (1.000 / 0.994 vs 1.000 / 0.726), but the step chains
  do not reach the paper's level in this re-implementation.
- **The rebuild that reproduces the paper's finding numbers is not the
  paper's specified rebuild.** The paper specifies alpha=1, beta=1, 10
  passes; with beta=1 the delta-rule step exceeds 2/lambda_max(K K^T)
  once ~1000 near-orthogonal keys are stored, so it oscillates (found
  0.990, next-step 0.888 at 200). The results above use beta=0.25 x 10
  passes — the stable iterative route to the same least-squares solution
  the paper's passes approximate (the paper's own online beta_base is
  0.25). The module keeps beta=1 as the default and documents this.
- **Embedder detail**: the paper's "hash embedder" is reproduced as the
  hashing trick (signed one-hot buckets). A dense random-vector
  embedder was tried first and drowned matrix recall in common-mode
  crosstalk (found 0.21 at 200); templated natural-language texts did
  the same (mean cross-key cosine 0.45 with stopwords). The synthetic
  capacity data therefore uses random word sets from the 3000-word
  vocabulary, as the paper's synthetic runs do.
- **Self-judged solved share is more extreme than the paper's** (21.5%
  vs 35%) — same direction, larger gap; the simulation's method order
  (fastest first) and hidden-limit distribution are a reconstruction,
  not the paper's MuJoCo setup.
- **Not tested**: the `.emm` v6 file format and container (Sec. 5),
  the router/sharding follow-ups, real-text embedders (bge-m3 /
  Qwen3-Embedding), the Qwen tool-choice follow-up, and any real robot
  or user data — all simulations, as in the paper.

## Files

- `test_matrix_steps_cpu.py` — the deterministic CPU test (run: `python3 tests/test_matrix_steps_cpu.py` from the repo root; ~5 min, mostly the capacity rebuilds)
- `results.json` — all measured numbers
- `rust/` — Cargo.toml, Cargo.lock, src/lib.rs, src/bin/matrix_steps.rs, tests/unit.rs, tests/deep.rs, tests/parity_gen.py, tests/parity_sweep.py, README.md, results.json, soak_results.json (no `target/`, no binaries)
- Mechanism (graduated): `src/arxiv_lab/memory/matrix_steps.py`, exported from `src/arxiv_lab/memory/__init__.py`
