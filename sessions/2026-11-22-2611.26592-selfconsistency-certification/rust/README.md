# selfconsistency-rs

Rust port of the self-consistency certification mechanism from
`lesterppo/arxiv-lab` (arXiv:2611.26592 — "Realized versus Population
Reliability: Anytime-Valid Self-Consistency Certification under
Correlated Reasoning", Agbeyegbe & Agbeyegbe, 2026): a mixture
betting e-process over repeated rollouts certifies a prompt only
while the rollouts are conditionally independent; positive
correlation through a per-prompt latent belief splits the null
into a *realized* null (about the drawn belief) and a *population*
null (about average behaviour), the within-unit effective sample
size saturates at 1/rho, and one rollout per fresh prompt restores
population validity.

Python stays the primary verdict carrier; this port cross-checks it.
A failed Rust leg would never overturn the Python verdict.

## What is ported

| Python (`arxiv_lab/agent/selfconsistency.py`) | Rust (`selfconsistency_rs`) |
|---|---|
| `SplitMix64` | `SplitMix64` — same arithmetic, same stream |
| `normal_cdf` (`math.erf`) | `normal_cdf` — libm `erf` via FFI (see below) |
| `normal_ppf` (Acklam) | `normal_ppf` — same coefficients, same op order |
| `rho_from_tau_sigma`, `tau_sigma_from_rho` | same |
| `pi_positive` | `pi_positive` |
| `betting_eprocess`, `crossing_step` | same, over `LAMBDA_GRID` |
| `simulate_unit` (binary + latent) | `simulate_unit` |
| `simulate_prompt_level` | `simulate_prompt_level` |
| `ess_within_unit`, `within_unit_mean_var` | same |
| `icc_anova` | `icc_anova` |
| `theta_hat_from_accuracy` | `theta_hat_from_accuracy` |
| `run_single_unit_experiment` | `run_single_unit_experiment` |
| `run_prompt_level_experiment` | `run_prompt_level_experiment` |
| `tests/test_selfconsistency_cpu.py` (E0–E7) | `selfconsistency verify` (same experiments + assertions) |

OUT OF SCOPE (not ported): the model-backed leg (real rollouts
through NVIDIA NIM) — it needs network access and a real LLM, so
it stays Python-only. Everything ported here is deterministic CPU
logic.

## Build and test

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd sessions/2026-11-22-2611.26592-selfconsistency-certification/rust
# CARGO_TARGET_DIR keeps target/ out of the session dir.
CARGO_TARGET_DIR=/tmp/selfconsistency-rs-target cargo test --release --offline
# 29 unit tests + 10 deep tests
CARGO_TARGET_DIR=/tmp/selfconsistency-rs-target cargo build --release --offline
/tmp/selfconsistency-rs-target/release/selfconsistency verify  # live-test verification
python3 tests/parity_sweep.py                                  # Python-vs-Rust parity
/tmp/selfconsistency-rs-target/release/selfconsistency soak    # 20000x256 scale run
```

Zero external dependencies — stdlib only, so the build works
`--offline` with no crate downloads. (`Cargo.lock` exists but lists
no dependencies.)

## CLI

```bash
# Live-test verification (mirrors the Python CPU test's E0-E7
# experiments and assertions, plus exact-equality checks against
# ../results.json)
selfconsistency verify [--expected ../results.json]
# -> ALL CHECKS PASSED + RESULTS_JSON {...}

# Parity helpers (machine-readable, for tests/parity_sweep.py)
selfconsistency ppf <p> / cdf <x>     # repr + u64 bits
selfconsistency ppf-batch / cdf-batch # values on stdin, one per line
selfconsistency eprocess <csv>        # wealth path as JSON floats
selfconsistency crossing <eps> <csv>  # first crossing step, or null
selfconsistency run-single <seed> <theta> <rho> <n_rollouts> <n_units> [eps]
selfconsistency run-prompt <seed> <theta> <rho> <n_prompts> <n_reps> [eps]
selfconsistency ess-rows <seed> <rho> <m_units> <n_max>
selfconsistency soak [n_units] [n_rollouts] [seed] [theta] [rho]
```

## Live-test verification (2026-11-22)

`verify` **ALL CHECKS PASSED** in ~0.34 s (Python: 11.4 s), with
every number **identical to the Python `../results.json`** (exact
float equality on all experiment fields, checked by the binary
itself against `--expected ../results.json`):

- E0 unit checks: all passed (`ppf(0.975)=1.959964`,
  `pi(-0.15, tau=0.693)=0.414`, e-process crosses on all +1 at
  step 8, never on all −1, ICC 1.0 on perfectly split groups,
  0.0 on identical groups).
- E1 rho=0 control: crossing **0.0045** (<= eps).
- E2 AIME regime (theta=-0.15, rho=0.48, null true): N=64
  crossing **0.272**, N=256 crossing **0.359**, pi **0.414** —
  grows with budget toward pi from below.
- E3 realized-null split (N=256): mu<=0 rate **0.0018**
  (n=1142, valid), mu>0 rate **0.834** (n=858, high power).
- E4 ESS saturation: empirical var 0.9946 / 0.6169 / 0.5155 /
  0.4920 / 0.4850 at n=1/4/16/64/256 vs predicted 1.0000 /
  0.6100 / 0.5125 / 0.4881 / 0.4820; ESS(256) **2.07** ~ 1/rho.
- E5 prompt-level: validity crossing **0.0075** (<= eps); power
  at theta=+0.4, B=256: prompt-level **0.994** vs single-unit
  **0.619**.
- E6 GSM8K regime (theta=+0.96, rho=0.435): crossing **0.834**
  (pi=0.927), gap dormant.
- E7 determinism: repeated run identical.

`cargo test --release` **39 passed, 0 failed** (29 unit + 10 deep).

## Deep tests (scale / properties Python does not reach)

- Exhaustive oracle check: all 2^1..2^10 binary increment
  sequences x 3 lambda grids (default 9-grid, single 0.5,
  {0.1, 0.9}) — **6,138 cases, 0 mismatches** against a
  direct-product wealth oracle.
- Prefix monotonicity: one fixed 2000-unit set at N=1024;
  crossing rates at prefixes 16/64/256/1024 are non-decreasing,
  N=1024 rate > 0.30.
- Prompt-level validity across a rho grid {0.0, 0.2, 0.48, 0.8}
  (theta=-0.15): all rates < 0.08; power across the same grid
  (theta=+0.4): all rates > 0.5.
- ESS saturation across rho: ESS(100000, rho) within 1% of 1/rho
  for rho in {0.05, 0.1, 0.2, 0.48, 0.9}; ESS decreasing in rho,
  increasing in n, bounded by 1/rho.
- Realized-null split property on a fresh seed family (theta in
  {-0.5, -0.15, 0.0}, seed 1234): mu<=0 rate < 0.08 and mu>0
  rate > 0.5 in every case.
- Soak (in `cargo test`, and via `soak`): 20,000 units x 256
  rollouts = 5.12M rollouts, ~0.6 s, crossing **0.3399**
  (pi=0.414). The first 2000 units are bit-identical to Python
  E2 N=256 (rate 0.359); the 0.3399 overall rate reflects the
  additional 18,000 units (0.3378) — sampling variation, not a
  port difference.
- Thread identity: 4 threads running the same (seed, params)
  reproduce the serial results exactly; shard aggregates are
  bit-identical. The mechanism is stateless per run.
- E-process mean wealth under the null (theta=-0.5, rho=0):
  mean wealth at steps 1/8/32/64 stays <= 1.05 over 4000 units.

## Parity sweep: 10,253 cases, 0 mismatches

`python3 tests/parity_sweep.py` (~6 s):

- `ppf`: 5,014 uniforms + edge values (1e-300, 1e-12, 0.02425
  region boundary, 0.999999, …) — **0 mismatches, worst 0 ulp**.
- `cdf`: 5,009 points + edge values — **0 mismatches, worst
  0 ulp**.
- `eprocess`: 203 increment sequences (random lengths 1..64,
  all-+1, all-−1) — full wealth paths bit-identical, crossing
  steps identical.
- Full experiments with identical seeds: E1, E2 N=64/N=256,
  E5 validity + power, E6 (7 cases) and E4 ESS rows (5 rows) —
  every field equal to `../results.json`; plus 15 extra
  experiments on fresh seeds/parameters run live on both sides —
  **0 mismatches**.

## Bit-exactness: the one 1-ulp gap found and fixed

The first build showed a 1-ulp difference in the e-process
wealth path (`1.405555555555556` Rust vs `1.4055555555555557`
Python after a single +1 increment) although every `log1p` and
`exp` term was bit-identical. The cause is the **sum**: CPython
3.12's built-in `sum()` uses **Neumaier compensated summation**
for floats, while Rust's `Iterator::sum` is a naive sequential
accumulation. Verified empirically: the Neumaier formulation in
`py_sum` matches CPython's `sum()` on 20,000 random float lists
with 0 mismatches. The port therefore implements `py_sum` and
uses it wherever the Python source calls `sum()` on floats
(`betting_eprocess`, `crossing_step`, `icc_anova`'s
`ss_between`/`ss_within`, and the E4 mean/var in `verify`).
After the fix, all 10,253 parity cases are 0 mismatches. This
is a deliberate, documented divergence from naive Rust
idiom — not a rounding of results.

## Deliberate divergences from Python

- `normal_cdf` calls libm `erf` through a tiny `extern "C"`
  declaration (`#[link(name = "m")]`). Rust's std has no `erf`;
  this is the exact function CPython's `math.erf` calls, so
  `normal_cdf` is bit-identical. No crates are involved; the
  binary still links only the system libm. (`ln`, `ln_1p`,
  `exp`, `sqrt` likewise resolve to the same libm routines
  CPython uses — verified by the 0-ulp ppf/cdf sweeps.)
- Float sums use `py_sum` (Neumaier, matching CPython 3.12's
  `sum()`) instead of Rust's naive `.sum()` — see the section
  above. This is the only behavioural divergence in the
  arithmetic; it exists precisely to *remove* a divergence.
- `(x - mean) ** 2` in Python is `pow(x, 2.0)`; the port uses
  `x * x`. glibc's `pow` is correctly rounded, so `pow(x, 2.0)`
  equals the correctly rounded `x * x` bit-for-bit; verified by
  the bit-exact ICC tests on mixed groups and by the E4 rows.
- `SingleUnitResult` / `PromptLevelResult` are structs with the
  Python dict's fields in the same order (the two
  `crossing_rate_realized_*` fields are `Option<f64>`, `None`
  when the group is empty, matching Python's `None`). `PartialEq`
  derives make the E7 determinism check a struct equality, the
  same comparison Python's `dict == dict` makes.
- The `verify` binary's JSON parser is a minimal
  recursive-descent parser written for this task (numbers as
  f64, objects/arrays/strings/bools/null) — it only reads
  `../results.json` for the exact-equality comparison.
- Floats print with a Python-`repr`-compatible format
  (`py_repr`: integral floats as `1.0`, scientific notation
  below 1e-4 / at or above 1e16); parsed values round-trip
  exactly.
- The `ppf` / `cdf` / `eprocess` / `crossing` / `run-single` /
  `run-prompt` / `ess-rows` / `soak` subcommands are new
  artifacts for parity and scale testing; they are not Python
  features. `run-single`/`run-prompt` take `eps` as an optional
  trailing argument (default 0.05).
- Errors that Python raises as `ValueError` (`normal_ppf`
  outside (0, 1), `tau_sigma_from_rho` outside [0, 1],
  `within_unit_mean_var` with n = 0) are Rust `panic!`s with
  the same message text. No test exercises them through the
  binary.
- `SplitMix64::new` masks the seed to 64 bits (a no-op for
  `u64`, kept for symmetry with Python's `int(seed) & MASK64`).

No other divergences. In particular: the RNG stream, the
`tau == 0` no-draw rule in `simulate_unit` (the stream
alignment depends on it), the lambda grid, the log-space
max-shift in the e-process, and the experiment loops all
follow the Python source operation for operation.

## What the Rust leg does NOT claim

The Rust port cross-checks the Python implementation; it does
not re-validate the paper. The verdict carrier is the Python
track, and the synthetic-model limitation (random-effects
simulation instead of the paper's Claude Haiku 4.5 rollouts)
applies identically here.
