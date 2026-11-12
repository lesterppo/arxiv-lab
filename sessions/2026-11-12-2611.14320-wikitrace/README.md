# Don't Just Trust Your Memory: Evidence-Linked Memory (WikiTrace)

- **Article:** arXiv [2611.14320](https://arxiv.org/abs/2611.14320) — "Don't Just Trust Your Memory: Evidence-Linked Memory Reduces Memory-Induced Degradation in LLM Agents" (submitted 10 Nov 2026; Lee, Choi, Baek, Hwang, KAIST)
- **Verdict: SUPPORTED** (directional reproduction at bounded scale; see "Why SUPPORTED, not VERIFIED")
- **Track:** agent track — CPU only, stdlib only, no network in the primary test
- **Code:** https://github.com/yukyeongleee/wikitrace

## Why picked

- **AGENT 5 / TUNE 1.** A memory primitive directly reusable in a local agent harness: each lesson is stored as a concise insight linked to a compilation (why its trajectory succeeded/failed) and to the raw trajectory itself, and the agent descends through those links — whenever an insight lacks details needed to act, conflicts with another, or comes from a failed task — to fill in missing details or correct the insight before acting. The paper claims WikiTrace beats No Memory by 10.7 points on average across five agentic benchmarks (eight suites) and Insights Only by 3.2, and cuts Regression (successes turned into failures) from 6.5% to 4.5%.
- **Repo fit:** `arxiv-lab` mechanism library — an evidence-linked memory store with a descent/check step is a general harness primitive, complementary to the student-teaching module from 2026-11-11. A natural home for the hermes agent's memory (its lessons today are stored without their source runs, exactly the failure mode this paper diagnoses).
- **Testable within budget:** the mechanism is deterministic logic over a small simulated agent, so a faithful CPU proxy test is possible in one run.

## What was built

- `wikitrace.py` — the mechanism (stdlib only, deterministic):
  - `Task` / `gen_tasks` — synthetic multi-step tool-use tasks: three steps (locate / operate / verify), each requiring one of three actions. The gold action for locate/operate is fixed; for verify it depends on the task's attribute (object kind). Gold answers are computed by the generator, so success is measured, not judged.
  - `DefaultAgent` (via `default_choices` / `run_default`) — a parameterized stand-in for a model agent: per step it picks the gold action with probability 0.6 and a random wrong action otherwise. Choices are seeded per task, so every memory arm faces the *same* default choices on the same task (paired Gain/Regression, as in the paper).
  - `compile_memory` — the compiler: one compilation per run (a diagnosis of why the run succeeded/failed, separate templates per outcome) and up to three insights (one per step). Each insight states what to do, when it applies (scope `any` or one specific attribute), which run it came from, and whether that run succeeded. Two honest error sources, matching the paper's failure analysis: (a) over-generalization — a lesson drawn from one attribute is recorded with scope `any`; (b) misattribution on failed runs — with probability 0.20 the compiler blames an action the trajectory never took (the paper's "wrong guess" about which action mattered, invisible from the success/failure label alone). Both are detectable from the linked trajectory, which is the paper's central finding.
  - `Memory` — insights + compilations + raw trajectories, with source links (insight → compilation → trajectory), plus `subset()` for the paper's §5.3 successful/failed-runs-only memories.
  - `verify_lesson` — the deterministic rule-based stand-in for the paper's LLM evidence check: it reads the source compilation, then the raw trajectory step of the same type, and returns SUPPORTED (the trajectory shows the claimed action succeeding for this task's attribute), CONTRADICTED (the trajectory shows a different action was taken, or the claimed action failing), UNSUPPORTED (scoped `any` but sourced from a different attribute — an over-generalization the trajectory cannot vouch for), or UNKNOWN (no matching step, or the step's observation was not recorded; observations drop out with probability 0.20). The check is sound by construction: a SUPPORTED lesson always prescribes the gold action for the current task.
  - `solve` — three arms: no-memory (default policy), plain lessons (`Insights Only`: follow the first applicable lesson blindly), and WikiTrace (descend on the paper's three triggers — missing detail, conflict, failed-run origin — verify applicable lessons in order until one is supported, follow it only if supported, else fall back to the default policy). Descent stops at the first supported lesson: every supported lesson prescribes the same gold action, so stopping early changes no outcome, only the evidence cost.
  - `SplitMix64`: deterministic 64-bit RNG implemented in pure Python so the Rust second stage can reproduce the exact same streams from the same seeds.
- `test_wikitrace_cpu.py` — deterministic test, 8 pre-registered assertion groups (A1–A8); writes `results.json`.
- `rust/` — Rust second stage (standing rule, 2026-11-02): a line-for-line port of the mechanism (no external crates), `verify` binary mirroring the Python test's experiments and assertions, unit + deep + parity tests. See `rust/README.md`.

## Results

Main experiment (12 train runs → 36 insights, 600 held-out tasks, seed 20261112):

| arm | success | Gain | Regression |
|---|---|---|---|
| No Memory | 0.200 | — | — |
| Insights Only (plain lessons) | 0.505 | 0.407 | 0.102 |
| **WikiTrace** | **0.703** | **0.503** | **0.000** |

- Ordering reproduces the paper's direction (WikiTrace > Insights Only > No Memory), with larger gaps than the paper's (90.2 > 87.0 > 79.5) because the miniature's default agent is weak (0.6 per step) and its lesson pool is noisy.
- Memory-induced degradation is the sharpest contrast: plain lessons turn 50.8% of the tasks the baseline solves into failures (Regression 0.102 over all tasks); WikiTrace turns none into failures (Regression 0.000). That zero is a theorem of the miniature, not a measurement accident: the evidence check is sound (a SUPPORTED lesson always prescribes the gold action) and the fallback is the identical default policy, so WikiTrace can never turn a correct default step wrong. In the paper the check is an LLM judgment, so its Regression is 4.5%, not 0 — see "Why SUPPORTED, not VERIFIED".
- Descent mechanics: descent triggers on 100% of consulted steps (every consulted lesson set contains a failed-run lesson, a scope-`any` lesson, or a conflict — a miniature artifact, see Limits), 26.4% of steps end with the action corrected relative to blind-follow, 23.8 evidence reads per task (one compilation + one trajectory read per verification; the descent stops at the first supported lesson).
- Paper §5.3 (memory from successful / failed runs only): WikiTrace with successful-runs-only memory reaches 0.583, with failed-runs-only 0.357, both below the 0.703 with both — same ordering as the paper (both > either alone). Note the miniature's failed-runs-only arm still beats the baseline (0.357 > 0.200) because its insights are evidence-checked; without the check, failed-run lessons are the main regression source (4 of 36 insights are corrupted misattributions, 24 of 36 come from failed runs).

## Why SUPPORTED, not VERIFIED

- The evidence checker is a rule-based stand-in, not an LLM: it compares a lesson's claimed action against the trajectory's recorded action and outcome. The paper's check is a model reading the trajectory and judging support — it can be wrong, miss evidence, or hallucinate. The miniature therefore measures the mechanism with a perfect checker, which is why its Regression is exactly 0 while the paper's is 4.5%.
- The agent is a rule-based stand-in, not a model: the default policy is a fixed per-step success probability, and lessons are policy patches (a followed lesson deterministically sets the action), whereas the paper's agent is a model that may ignore, misread, or misapply a lesson. The descent machinery and the Gain/Regression accounting are the reproduced mechanism; the agent side is a model of a model.
- Absolute success rates are miniature-scale (600 synthetic 3-step tasks, 12 training runs, 3 step types, 4 attributes), not the paper's benchmarks (ALFWorld, ScienceWorld, PlanCraft, τ³-bench, AppWorld with frontier models). The claim reproduced is the *ordering and the regression elimination under a sound check*, not the 90.2% figure.
- The corruption (0.20) and observation-dropout (0.20) rates, the scope-`any` rate (0.50), and the default success probability (0.60) are modeling choices calibrated so the lesson pool is noisy enough for plain memory to regress; the paper does not publish equivalents.

## Limits

- Seed fragility (found by the Rust 200-seed sweep): WikiTrace beats plain lessons on 197/200 seeds and never regresses (Regression 0 on every seed; WikiTrace ≥ No Memory on every seed), but plain lessons beat No Memory on only 74/200 seeds — with a small noisy lesson pool, blind-following often hurts. Assertion A3 (plain > none at the main seed 20261112) is therefore a favorable draw, not a theorem; the robust claims are A1/A2/A4/A5, which hold across the sweep.
- Descent rate 1.0 per consulted step is a miniature artifact: a lesson is consulted at every step and failed-run lessons dominate the pool (24/36 insights), so a trigger almost always fires. The paper's agent descends in 22% of Easy runs and 44% of Hard runs — per run, and only for the few insights it actually applies. The miniature has no notion of "not consulting memory on a step", so its descent rate is not comparable to the paper's.
- Evidence cost (23.8 reads/task) is likewise inflated by the same artifact; the paper reports 1.17× the per-task cost of Insights Only. The miniature's cost model counts one compilation read + one trajectory read per verification, with no token accounting.
- The check can only verify a lesson when the source trajectory recorded the step's observation (80% of steps); unobserved steps return UNKNOWN and the lesson is dropped, which is conservative — a real LLM check could still reason from the compilation alone.
- Only one task family (3 fixed step types, gold depending on attribute only at the verify step) was tested; the paper's benchmarks span household, science, crafting, and tool-use tasks.
- No model-backed validation was run: the paper's effect depends on a real model writing insights and a real agent descending through evidence, which this CPU test does not exercise.
