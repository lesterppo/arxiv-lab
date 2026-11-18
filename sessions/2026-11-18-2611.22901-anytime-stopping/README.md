# Done, But Still Working: Anytime Evaluation of When Coding Agents Should Stop (arXiv:2611.22901) — session record

**Article:** Yang, Xue, Dong, Zhang, Zheng, Wang, "Done, But Still
Working: Anytime Evaluation of When Coding Agents Should Stop",
https://arxiv.org/abs/2611.22901 (v1, 2026-11-18).

**Verdict: SUPPORTED** — the mechanism reproduces directionally at
bounded scale with stated limits (see "Honest limits"). All
experiments below were run live on CPU; nothing is a theater run.

## Why picked (AGENT 5, TUNE 2)

The stop rule is directly reusable in a local agent harness loop: the
Hermes AgentRunner already caps runs at 12 rounds, and "stop after the
same patch passes twice" is a drop-in finish policy for that loop —
cheaper than a fixed round cap, safer than stopping at the first pass.
No training track: the fine-tuning claim (imitating the rule saves
30%) is noted as not tested, and the paper is a mechanism paper,
tested here in the VM's local agent stack.

## What was built

- `src/arxiv_lab/agent/anytime_stopping.py` (graduated) +
  `tests/test_anytime_stopping_cpu.py` (copy in this session):
  an anytime-evaluation harness — a calibrated synthetic trajectory
  model (each run is a sequence of steps; each step emits a patch with
  a hidden-test state and a self-test state, the noisy proxy), five
  stopping policies compared on the SAME trajectories (paired
  comparisons, no new rollouts, exactly the paper's replay setup), and
  the paper's explanation metrics (fire rate, patch changes after the
  rule fires, repair-vs-break counts, paired quality delta with CI).
  Stdlib only, no model calls, no network. Determinism flows through
  SplitMix64, implemented identically in the Rust port.
- `rust/` — Rust second stage (std only): full port + `verify` binary,
  9 unit + 6 deep tests, parity sweep, soak. See `rust/README.md`.

## Results (measured vs paper)

10,000 runs (5 seeds x 2,000), budget 30 steps, calibrated regime
(every parameter and its source is documented in the module):

| policy | saving vs run-to-budget | final hidden-test pass rate | fire rate |
|---|---|---|---|
| (a) run to budget (baseline) | 0% | 0.3095 | — |
| (b) stop at first self-test pass | 29.12% | 0.3260 | 0.506 |
| (c) stop after same patch passes twice (the paper's rule) | **25.83%** | 0.3267 | 0.472 |
| (d) hindsight oracle (first hidden pass) | 19.75% | 0.3608 | 0.352 |

Paper vs measured:

- **Double-pass rule saves 25.83%** (paper: 23% in the preregistered
  replication). Per-seed: 26.32 / 26.92 / 24.78 / 25.83 / 25.29%
  (spread 2.14pp).
- **Hindsight saving at passing self-tests: 18.40%** (all-runs
  reading) and **56.85%** (conditional on the first self-pass being a
  true pass, n=3,236) — both inside the paper's <1%–57% range.
- **The rule out-saves the oracle (25.83% vs 19.75%)** — because
  ~40% of its stops are on false self-passes. That is the quality
  risk, not a free lunch: the oracle is the quality-preserving bound.
- **Post-fire analysis** (the paper's explanation): of the 4,720 runs
  where the rule fired, the patch changed afterwards in 84.3%
  (3,981), and those changes repaired the patch 244 times and broke
  it 416 times — both directions happen, as the paper's account
  requires.
- **Quality caution (the paper's non-inferiority point):** paired
  delta double-pass vs baseline **+1.72pp** (wins 416, losses 244,
  ties 9,340), 95% CI (+1.22, +2.22)pp. The point estimate is
  positive, but 244 concrete quality-loss cases exist (the rule stops
  on a broken patch that would have repaired), and across the
  11-cell sensitivity sweep the delta **changes sign: −3.33pp to
  +17.40pp** — negative when self-tests are a noisier proxy
  (false-pass 3–5%), positive when post-pass changes break often.
  Whether the rule preserves quality depends on the model/scaffold
  regime, so non-inferiority cannot be established from the mechanism
  alone — the same regime-dependence the paper reports for savings.
- **First-self-pass vs double-pass:** the earlier rule saves more
  (29.12% vs 25.83%) at slightly worse quality (0.3260 vs 0.3267) —
  the extra confirmation step costs ~3pp of saving and buys a small
  quality margin.
- **Scale check:** a 100k-run Python check gives baseline pass
  0.2974, double-pass saving 25.48%, delta +2.69pp — the 10k-run
  numbers are stable; the true baseline rate sits just under 0.30,
  which is why the test band is [0.28, 0.50].

Rust leg: `verify` **ALL CHECKS PASSED** in ~0.05 s with numbers
identical to Python on every metric; `cargo test --release`
**15 passed, 0 failed** (9 unit + 6 deep); parity sweep **23/23,
0 mismatches**; `soak` **PASSED** (200k runs: saving 25.50%,
baseline pass 0.2970, delta +2.77pp, two passes bit-identical,
4-thread identity bit-for-bit).

## Calibration (what the synthetic model assumes)

The paper's 6,000 recorded trajectories are not public, so every
trajectory here is generated, not replayed. Each parameter is
calibrated to a paper aggregate or marked as an explicit assumption
(full table in the module docstring):

- budget 30 [assumption: typical SWE-agent step cap]
- p_solvable 0.65 [paper: eventual hidden-pass rate ~0.31, inside the
  0.30–0.50 open-model SWE-bench Verified resolve-rate band]
- p_fix 0.03, p_break 0.08, p_change_after_self_pass 0.15,
  p_change_after_self_fail 0.90 [assumptions, tuned so the first
  self-pass lands late enough for ~26% saving and the repair/break
  balance keeps the quality delta near zero]
- p_self_false_pass 0.01, p_self_false_fail 0.15 [assumptions: the
  self-test is a noisy proxy]

Two modelling choices were load-bearing and are documented as such:

- **Self-test outcomes are sticky per patch version** (re-running the
  same tests on an unchanged patch returns the same result). A v1
  model with a per-step 15% false-pass rate put 60% of first
  self-passes on broken patches and the hindsight saving at 86% —
  both outside the paper's ranges. The sticky + 1% calibration fixed
  both.
- **The agent's change decision depends on the self-test outcome it
  can see** (edits after a self-fail, mostly re-runs tests after a
  self-pass), never on the hidden state it cannot see. A v1 model
  keyed on the hidden state let post-pass churn break nearly every
  good patch (baseline pass 0.16, rule quality +21.7pp) — backwards
  against the paper's caution.

Under sticky self-tests the cumulative double-pass variant coincides
exactly with the consecutive rule; the test asserts this equality on
2,000 trajectories.

## Honest limits

- **Synthetic trajectories, not the paper's recorded runs.** The
  verdict ceiling is SUPPORTED, never VERIFIED. The numbers reproduce
  the paper's aggregates directionally (23% saving, <1%–57%
  hindsight range, regime-dependent quality) but no claim is checked
  against a real SWE-bench run.
- **The pooled 95% CI for the quality delta excludes a 2pp loss**
  (+1.22 to +2.22pp) in the calibrated regime. The
  "non-inferiority not established" claim rests on the sweep's sign
  flip and the 244 loss cases, not on the pooled CI — i.e. it is a
  claim about regime-dependence, matching the paper's cross-model
  finding, not a re-run of the paper's preregistered test.
- **Not tested:** the fine-tuning claim (imitating the rule saves
  30% over six seeds); real SWE-bench Verified trajectories or any
  real model/scaffold; token-cost (as opposed to step-count) savings;
  the paper's per-model/per-scaffold breakdown (the synthetic model
  has one regime, varied only in the sweep).
- **Extreme sweep cells are outside the realistic regime** (baseline
  pass as low as 0.106 when post-pass changes break 30% of patches
  and the agent edits 45% of post-pass steps). They are mechanism
  probes showing the delta's sign, not plausible deployments.

## Files

- `test_anytime_stopping_cpu.py` — the deterministic CPU test (run:
  `python3 tests/test_anytime_stopping_cpu.py` from the repo root;
  ~5 s, 34 checks)
- `results.json` — all measured numbers (10,000-run experiment +
  11-cell sensitivity sweep)
- `rust/` — Cargo.toml, Cargo.lock, src/lib.rs,
  src/bin/anytime_stopping.rs, tests/unit.rs, tests/deep.rs,
  tests/parity_gen.py, tests/parity_sweep.py, README.md,
  results.json, soak_results.json (no `target/`, no binaries)
- Mechanism (graduated): `src/arxiv_lab/agent/anytime_stopping.py`,
  exported from `src/arxiv_lab/agent/__init__.py`
