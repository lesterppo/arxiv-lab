# Signed but Not Safe — requirement integrity in multi-agent handoffs

**Article:** arXiv:2611.07483 — "Signed but Not Safe: Requirement Integrity
in Multi-Agent LLM Handoffs"
**Link:** https://arxiv.org/abs/2611.07483
**Date:** 2026-11-06 · **Track:** agent · **AGENT 5 / TUNE 1**

## Why picked

Multi-agent handoffs are the exact failure surface of Peter's agent work
(Hermes on-device agent, multi-agent orchestration): an orchestrator splits
a request across subagents and merges results, and each handoff can
summarize away, alter, or ignore a requirement. The paper's mechanism —
verbatim-quote + checkable-predicate requirements, HMAC-SHA256-signed
manifest, deterministic verifier — is pure logic, exactly scorable, and
directly reusable as a harness guard. Repo fit: `arxiv-lab` agent
mechanisms (`src/arxiv_lab/multiagent/`), `hermes-colab-cli` infra.

## What was built

- `src/arxiv_lab/multiagent/signed_manifest.py` — the mechanism:
  requirement catalogue (5 checkable requirements), rule-based extractor
  (last line per field wins), HMAC-SHA256 manifest signing + verification,
  simulated merger (faithful / drop / alter / ignore), deterministic
  predicate verifier, noisy surface-text checker (simulated LLM checker).
- `tests/test_multiagent_signedmanifest_cpu.py` — deterministic CPU test:
  3 formats × 6 threats × 60 trajectories (1,080 trajectories), plus
  signing/extraction/predicate unit checks.
- `rust/` — std-only Rust second stage (SHA-256 + HMAC from scratch):
  `verify` reproduces the experiment's claims, `tests/deep.rs` runs the
  extensive suite, `parity_signedmanifest.py` sweeps Python-vs-Rust.
- `model_leg_nim.py` — model-backed leg (gpt-oss-20b via NVIDIA NIM):
  the LLM as requirement extractor and as checker (see below).

## Results — deterministic CPU test (seed 0, n=60/cell)

Unsafe-commit rate under attack (mean of drop/alter/ignore/mutate/inject):

| format | unsafe |
|---|---|
| free_text | 0.920 |
| json_unsigned | 0.790 |
| signed_manifest | **0.160** |

Paper: 69.4% → 9.0%. Direction reproduced; magnitudes differ because the
simulator is rule-based (the merger always violates under drop/alter/
ignore by construction) while the paper's LLM merger sometimes still
conforms. The signed residual (0.160) comes entirely from the inject arm.

Per-cell unsafe-commit rates (signed_manifest): none 0.000, drop 0.000,
alter 0.000, ignore 0.000, mutate 0.000 (refused 60/60), inject 0.800.
Unsigned manifest under mutation: 0.483 unsafe (paper: 79% vs 0%).

Verifier comparison (over violating / conformant decisions):

| checker | detection | false alarms |
|---|---|---|
| deterministic verifier (signed) | 0.789 (n=228) | 0.000 (n=72) |
| surface-text checker (json) | 0.067 (n=254) | 1.000 (n=106) |

Paper: 78.7% vs 59.0% detection, 7% vs 50% false alarms. The
deterministic figure matches the paper's 78.7% closely, but for a
different reason that must not be over-read: in this simulator the
verifier misses exactly the inject-arm violations (48/228), because it
can only check against the manifest it was given — the corrupted
manifest hides the violation. The paper's 78.7% reflects LLM
extraction/predicate imperfection. The surface-text checker here is a
keyword-overlap simulation, far weaker than the paper's real LLM judge
(59%) — it is a lower bound on LLM-checker performance, not a
measurement of one. The model-backed leg below measures a real one.

Pre-signing injection (the trust boundary): signed-manifest violations
rise 0.000 → 0.800 (paper: 14.6% → 37.5% on Qwen3-8B). Higher here
because the rule-based extractor always adopts the injected line,
while the paper's LLM extractor only sometimes does. Conformant commits
with no attack: signed 1.000 vs json 1.000 — no net loss.

## Results — Rust second stage (live)

- `cargo test`: **13/13** — SHA-256 known vectors (incl. 1M-byte),
  HMAC-SHA256 RFC 4231 test case 1 (verified against Python's `hmac`),
  extractor/predicate/verifier units, property sweeps (signed+mutate
  always refused; signed-unsafe only via inject; free-text never
  refuses), 100k-case soak (invariants hold; signed unsafe far below
  free-text), 8-thread identity (20,000 calls, stateless).
- `signed_manifest verify`: **PASSED** — same 8 claims as the Python
  test, on an independent xorshift stream: signed 0.167 vs free-text
  0.940 under attack; detection 0.783 vs 0.105; inject 0.833 vs none
  0.000. The parity sweep caught one real divergence during development
  (Rust `alter` matched on the field name `payment_approved` as
  `payment`, altering logging instead) — fixed, see `rust/README.md`.
- Python-vs-Rust parity sweep: **480 cases, 0 mismatches** —
  PARITY VERIFIED (signatures byte-identical, incl. HMAC-SHA256).

## Results — model-backed leg (gpt-oss-20b via NIM)

Ran live (background, ~2.5 h wall time across three attempts; NIM is
slow and flaky — see run log). Per-case rows in
`model_leg_results.jsonl` (31 rows); script `model_leg_nim.py`.

**Extractor** (the trust boundary): clean tasks 7/8 parse-OK, all
recovering 4/4 ground-truth requirements (overall recovery 0.875;
the one miss is a NIM empty reply, not a model error). Injected
tasks (n=7): the model adopted the injected production line in **0/7**
cases and kept the original staging requirement in 6/7 (the 7th task
had no env requirement to keep). The real model resists this
injection format — unlike the rule-based extractor in the CPU test,
which adopts it by construction (last-line-wins). The paper's
injection effect (14.6% → 37.5%) is model- and format-dependent;
gpt-oss-20b on this wording does not reproduce it.

**Checker** (paired, same decisions): LLM checker detection **6/8**
(6/6 among parsed replies; the 2 misses are NIM parse failures, both
on payment violations), false alarms **0/8**. Deterministic verifier
on the same decisions: **8/8** detection, 0 false alarms. So the real
LLM checker is far stronger than the keyword-overlap simulation in
the CPU test (0.067 detection, 1.000 false alarms) and in line with
the paper's LLM checker (59% detection, 50% false alarms — here
higher detection and lower false alarms on n=8, wide CIs). The
deterministic verifier still wins on this synthetic set, as the paper
reports (78.7% vs 59.0%).

## Verdict

**SUPPORTED** — the mechanism's core claims reproduce directionally at
bounded scale in a deterministic simulator: signed manifests cut unsafe
commits under attack from 0.92 to 0.16, in-transit mutation is refused
60/60, the deterministic verifier beats surface-text checking, and the
pre-signing injection boundary is exactly reproduced (the signature
authenticates the corrupted manifest). Limits: rule-based simulator, not
LLMs — magnitudes differ from the paper where LLM behavior matters
(merger conformance under attack, checker quality, injection adoption);
the model-backed leg measures the LLM-facing parts with gpt-oss-20b.
The Rust port is parity-verified and carries the same verdict.

## Files

- `src/arxiv_lab/multiagent/signed_manifest.py` (+ `tests/test_multiagent_signedmanifest_cpu.py`)
- `rust/` — `Cargo.toml`, `src/lib.rs`, `src/main.rs`, `tests/deep.rs`,
  `README.md`, `results.json` (no `target/`, no binaries)
- `parity_signedmanifest.py`, `model_leg_nim.py`, `model_leg_results.jsonl`
