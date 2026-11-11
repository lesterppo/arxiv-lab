# Teaching the Harness: Student-Aware Skill Transfer for LLM Agents

- **Article:** arXiv [2611.13144](https://arxiv.org/abs/2611.13144) — "Teaching the Harness: Student-Aware Skill Transfer for LLM Agents" (submitted 9 Nov 2026)
- **Verdict: SUPPORTED** (directional reproduction at bounded scale; see "Why SUPPORTED, not VERIFIED")
- **Track:** agent track — CPU only, stdlib only, no network in the primary test

## Why picked

- **AGENT 5 / TUNE 1.** A runtime mechanism directly reusable in a local agent harness: a teacher that reasons about the *learner* — builds a learner model (empirical profile + hypotheses about the student's limitations), tests hypotheses with diagnostic probes, writes reusable skills for diagnosed failures, and admits a skill into the library only after it raises success on matched development tasks. The paper claims Claude-Sonnet-5 teaching DeepSeek-V4-Flash reaches 86.46% task success vs 81.04% for a profile-free teacher with the same framework and 75.00% for the untaught student; under misleading learner profiles none of the 48 candidate skills passes admission; each student does better with its own library than with another's.
- **Repo fit:** `arxiv-lab` mechanism library — a teacher/skill-library pipeline is a general harness primitive (skill admission, learner modeling), complementary to the FSM-control module from 2026-11-10. A natural home for the hermes agent's skill library.
- **Testable within budget:** the whole pipeline is deterministic logic over a small simulated student, so a faithful CPU proxy test is possible in one run.

## What was built

- `student_teaching.py` — the mechanism (stdlib only, deterministic):
  - `Task` / `gen_tasks` — procedurally generated multi-step file/workspace tasks (reads, substring searches, arithmetic, writes, final answer) with planted traps: missing file, flaky first read, stale re-read after a write, substring search, big multiplication. Gold answers are computed by simulating the correct procedure.
  - `StudentAgent` — a parameterized stand-in for a model student: it follows a fixed plan but deviates where its latent limitations fire (L1 no existence check, L2 exact-match search, L3 stale re-read, L4 no retry, L5 mental math). Installed skills remove one deviation each. Every installed skill costs 1 unit of action budget (context overhead — the paper's "a procedure that helps one model can exhaust the budget of another"). Budget calibrated: a correct student needs 6–13 calls (measured: ≤10 on 84% of tasks), so `BASE_BUDGET=14` leaves the untaught student never budget-limited, a 3-skill SAT library rarely limited (~4.5%), and a 5-skill profile-free library visibly limited (~16%).
  - `Teacher` — the SAT pipeline: empirical profile from dev-task failure fingerprints (with noisy attribution — spurious fingerprints at p=0.15, so probes matter), hypotheses above a 0.05 rate, diagnostic probes that isolate one limitation (6 minimal probe tasks each), skill writing, and the admission gate (a skill is admitted only if it raises success on matched dev tasks by ≥0.15).
  - Controls from the paper: `profile_free_teach` (a skill for every limitation, no probes, no gate), `profile_only_teach` (aggregate profile only, no probes, admission on overall dev success), `misleading_teach` (forced-wrong profile, straight to admission).
  - `SplitMix64`: deterministic 64-bit RNG implemented in pure Python so the Rust port reproduces the exact same streams from the same seeds.
- `test_sat_cpu.py` — deterministic test, 7 pre-registered assertion groups (A1–A7); writes `results.json`.
- `rust/` — Rust second stage (standing rule, 2026-11-02): a line-for-line port of the mechanism (no external crates), `verify` binary mirroring the Python test's experiments and assertions, 17 tests passing (11 unit + 5 deep + 1 parity), and a Python-vs-Rust parity sweep on 1,200 cases — 0 mismatches. The `verify` binary reproduces the Python arm rates exactly. See `rust/README.md`.

## Results

Main experiment (120 dev tasks, 200 held-out tasks, seed 20261111):

| arm | success |
|---|---|
| Untaught student A (L1, L3, L5) | 0.400 |
| Profile-free teacher (all 5 skills, no gate) | 0.715 |
| Profile-only ablation (no probes) | 0.780 |
| **SAT (diagnosis + probes + admission)** | **0.955** |
| Untaught student B (L2, L4) | 0.520 |
| SAT student B | 0.980 |
| A with B's library | 0.400 |
| B with A's library | 0.520 |

- Ordering reproduces the paper's direction (SAT > profile-free > untaught), with larger gaps than the paper's (86.46 > 81.04 > 75.00) because the miniature's limitations are harsher and the budget cost of unneeded skills is explicit.
- SAT-A library `['L1', 'L3', 'L5']` (all three true limitations); SAT-B library `['L2', 'L4']`. Profile rates for A: L1 0.233, L2 0.033, L3 0.425, L4 0.042, L5 0.508. All three probes confirmed at fail rate 1.0; admission deltas L1 +0.167, L3 +0.583, L5 +0.333 (n=12 matched tasks each).
- Misleading-profile control: 0 skills admitted (paper: 0/48).
- Library exchange: each student does better with its own library than with the other's (A 0.955 vs 0.400; B 0.980 vs 0.520) — B with A's library is no better than untaught B, as expected: A's skills fix limitations B does not have.
- Ablation: profile-only (no probes) reaches 0.780 < SAT 0.955 — it picks the two highest-rate limitations (L3, L5) but misses L1, and its admission on overall dev success cannot tell that L1 matters on its matched tasks.
- All A1–A7 assertions passed; deterministic re-run reproduces identical libraries and rates.

## Why SUPPORTED, not VERIFIED

- The student is a rule-based stand-in, not a model: skills are policy patches (an installed skill deterministically removes one deviation), whereas the paper's skills are text a model reads and may ignore. The admission gate and the diagnosis machinery are the reproduced mechanism; the student side is a model of a model.
- Absolute success rates are miniature-scale (200 held-out synthetic tasks, 5 hand-built limitation types), not the paper's benchmarks (Terminal-Bench-style tasks with frontier models). The claim reproduced is the *ordering and the gate behavior*, not the 86.46% figure.
- The budget-cost model (1 call per installed skill) is an explicit modeling choice calibrated to the measured call distribution; the paper's budget pressure comes from context, not tool calls.
- No model-backed validation was run: the paper's effect depends on a real teacher model writing skills and a real student following them, which this CPU test does not exercise. (The 2026-11-10 FSM session's NIM leg was BLOCKED by endpoint degradation; no NIM leg was attempted here.)

## Limits

- Probe tasks are minimal by construction (one trap each), so probe accuracy (3/3) is an upper bound on what noisy real diagnostics would give.
- Spurious-fingerprint noise (p=0.15) is a modeling choice; the profile threshold (0.05) and admission threshold (0.15) are ours, not the paper's (the paper does not publish them).
- Only two students (disjoint limitation sets) were tested; library exchange was not tested on overlapping sets.
- ~0.2% of generated tasks need 15 calls (flaky retries on two keys + stale re-read) and are unsolvable even for correct behavior under `BASE_BUDGET=14`; the 2,000-task Rust soak confirms the only such failures are budget failures, never wrong answers.
