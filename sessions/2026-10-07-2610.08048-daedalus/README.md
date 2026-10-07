# Session 2026-10-07 — DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks (arXiv:2610.08048)

**Article:** "DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks"
([abs](https://arxiv.org/abs/2610.08048))

**Why picked:** most prospective *agent* article in the 2026-10-07 cs.AI
window (AGENT 5/5). DAEDALUS pairs an explorer (generates challenging yet
solvable tasks, adapts difficulty from outcomes) with a solver; an
extractor derives a heuristic from each solver failure; the heuristic is
accepted only after the solver reaches Ns consecutive successes *with it in
context*, having failed at least once before those successes; accepted
heuristics are consolidated into a frozen bank injected at test time. No
pre-existing tasks, no oracle verifier. Crispest claim: +15.9 pts mean
success, 2.2× pass^5 over no-memory — competitive with training-task
methods at lower inference cost.

**Relation to existing arxiv-lab work:** the CMP module (`memory/cmp.py`)
is memory *consolidation*; DAEDALUS is memory *generation* — complementary,
not a duplicate. IEC (2610.04375) is pre-execution intent checking — also
distinct.

## What was built

- Reusable mechanism: `src/arxiv_lab/daedalus.py` — `VaultWorld`
  (deterministic tool env: scan/inspect/read_note/enter_code, hidden
  room→code conventions: color / initial / position), `Solver`, `Extractor`
  (scripted LLM stand-ins, behavior documented in module docstring),
  `solver_loop` (paper's acceptance rule: Ns=3 consecutive successes after
  ≥1 failure; too-easy/too-hard yield nothing), `Explorer` (propose +
  feasibility check + difficulty refine, Nr=3), `MemoryBank.consolidate`
  (dedupe to one lesson per convention), `run_daedalus`, `evaluate`
  (success rate + pass^k).
- Test: `tests/test_daedalus_cpu.py` (6/6 pass; numpy-guarded).
- Session record: `test_session_daedalus.py` (verbatim snapshot) +
  `results_daedalus.json`.

## Simulation honesty box

No local LLM exists on this VM (no ollama, no GPU, no API keys), so the
explorer/solver/extractor are **scripted behavior models**, not LLMs. The
solver models an LLM that sometimes skips recon (p=0.35 inspect), sometimes
misreads conventions (p=0.5 parse), and follows written advice in context
(5% slip/misapply). The extractor usually diagnoses the failure type but
writes a wrong lesson 25% of the time. What this tests is the paper's
**protocol** — acceptance rule, difficulty calibration, consolidation,
transfer — not whether real LLMs emit or consume such heuristics. Do not
cite these numbers as LLM evidence.

## Results (24 exploration sessions, 48 held-out tasks × 5 rollouts)

| seed | bank coverage | accept rate | base success | DAEDALUS success | Δ | base pass^5 | DAEDALUS pass^5 |
|------|---------------|-------------|--------------|------------------|---|-------------|-----------------|
| 7    | color, initial, position | 1.00 | 0.379 | 0.913 | **+0.533** | 0.021 | 0.688 |
| 123  | color, initial, position | 1.00 | 0.421 | 0.900 | **+0.479** | 0.021 | 0.604 |

Per-convention transfer is uniform (~0.91–0.93 on each of color/initial/
position vs ~0.36–0.49 baseline): lessons learned on self-generated tasks
transfer to held-out tasks of the same families.

Protocol checks (unit-tested): the acceptance rule **rejects extractor
errors** — with a always-wrong extractor the loop ends too-hard and no
heuristic is accepted; too-easy tasks (never failed) contribute nothing;
consolidation dedupes 3 raw heuristics → 2 (one per convention); the
explorer raises difficulty after too-easy and lowers it after too-hard.

Two real bugs found by testing: (1) cue ambiguity — two rooms could share a
color, making the hidden rule ill-defined (fixed: cue enforced unique);
(2) position-convention incoherence — the task's code room was defined
against a creation-time scan order while attempts re-shuffled (fixed:
per-task deterministic scan order via `task_scan_order`).

## Verdict vs the paper's claim

**Mechanism supported, directionally stronger here than in the paper.**
The paper reports +15.9 pts and 2.2× pass^5 on AppWorld/τ²-bench/
AutomationBench; this synthetic setting shows +48–53 pts and pass^5
0.021 → 0.60–0.69. The larger magnitude is expected: VaultWorld makes the
hidden convention *the* bottleneck (base solver ~0.40, memory resolves it
almost fully), whereas real benchmarks have many other bottlenecks. The
paper's core protocol claims replicate: failure→heuristic→validated
acceptance improves held-out success, bad heuristics are filtered by the
Ns-consecutive rule, and gains emerge from a small (24-session) budget.

## Limitations

- Scripted stand-ins, not LLMs (see honesty box): heuristic *quality* and
  LLM instruction-following are not tested, only the validation machinery.
- Deterministic judge (strictly stronger than the paper's LLM judge).
- Synthetic 3-convention world; 100% session acceptance rate here (the
  explorer's difficulty band kept tasks learnable) — real settings will
  see more too-easy/too-hard sessions.
- Baseline pass^5 ≈ 0.02 is a floor effect; report the absolute 0.02 →
  0.60–0.69, not a ratio.
- Single held-out difficulty (7 rooms); 2 seeds.
