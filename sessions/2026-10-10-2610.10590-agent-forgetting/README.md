# 2026-10-10 — 2610.10590 "Agent-Controlled Forgetting for Tool-Using Agents: Reversible Context Curation in Practice"

**Link:** https://arxiv.org/abs/2610.10590 (submitted 6 Oct 2026)

**Why picked (AGENT 5/5):** The mechanism is a harness-level context
manager the acting model drives itself: archive a noisy tool result,
replace it with a short note + archive ref in place, keep the exact
original in a recoverable archive, recover on demand. No training, no
provider dependency — a pure harness protocol. Directly complements
yesterday's 2610.11007 (Curated Context: which files to always load —
this is what to *drop* from tool history while staying recoverable), and
plugs into the local agent stack's tool loop (deepworld/hermes harness)
as a context-curation policy.

**What was built:**
- `src/arxiv_lab/context/reversible_forgetting.py` — `ForgettingHarness`
  implementing the paper's Sec. 3.1–3.2 protocol: ordered message sequence
  H + external archive A; `context_apply(batch)` with batch validation
  (unknown/duplicate/already-archived → `ForgettingError`, all-or-nothing,
  originals staged in A before stubs committed; user/system/assistant not
  selectable); `context_recover(ref)` appends the exact original as a new
  tool result with a fresh result id (no tool re-execution; original stub
  stays addressable). Forgetting = eviction from the active request, not
  deletion.
- `tests/test_context_forget_cpu.py` — seeded simulation replaying the
  paper's experimental design: a noisy tool-use trajectory (24
  observations × 120 lines, ~2.5% signal — the paper's OpenTelemetry
  debugging case) vs a dense contrasting workload (paper's
  application-development pair), comparing retained history vs reversible
  forgetting vs an irreversible-truncation ablation, with scripted recall
  probes on facts the lossy note dropped.

**Results (deterministic, seeded; 6/6 checks pass):**
- Protocol integrity: batch atomicity holds (invalid batch → zero
  mutation); recovery returns the byte-exact original with a fresh id;
  stub costs 0.8% of the original observation's tokens.
- Token savings, noisy workload: cumulative prompt tokens 432,405
  (retained) → 60,694 (forget) = **86.0% saved**, 24/24 observations
  archived, 3 recoveries, requests 26 → 53 (more requests, as in the
  paper: its figures were 231,951 vs 912,492 prompt tokens, ~50% fewer
  cumulative input tokens, USD 1.28–1.44 vs ~4.38, 17% longer).
- Reversibility preserves task success: all 3 recall probes answered
  after recovery; the irreversible ablation at the same token budget
  answers 0/3 — isolating the value of the recoverable archive.
- Workload dependence: on dense payloads the policy archives 0
  observations and saves 0.0% — matching the paper's contrasting pair
  that produced no saving. Savings are workload-dependent, not free.

**Verdict: REPLICATES (mechanism level, in simulation).** The paper's
four empirical signatures — large savings on noisy trajectories,
reversibility preserving correctness, zero savings on dense workloads,
more requests — all reproduce. The reusable module graduated to
`src/arxiv_lab/context/` + CPU test.

**Honest limits:** the "agent" is a scripted policy (note = extracted
signal facts, lossy by one fact), not an LLM choosing what to forget;
token estimates are chars/4, not a real tokenizer; savings magnitude is
a function of the noise ratio we chose (86% vs the paper's ~50–75% —
same direction, different workload). The paper's own caveat stands: the
note's semantic adequacy is not validated by storage-integrity checks —
our probes test exactly the failure mode (dropped fact → recovery).
No deepworld wiring yet: the policy-level takeaway is "archive
aggressively on noisy tool trajectories, keep recovery one call away,
never delete on zero-observation evidence" (shared with 2610.11007).
