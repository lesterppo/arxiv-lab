# Session 2026-10-21 — ClawMMU (arXiv 2610.28093)

**Article:** [ClawMMU: Protection-Coherent Virtual Memory for Stateful LLM Agents](https://arxiv.org/abs/2610.28093)
**Track:** agent · **AGENT=5** / TUNE=1 · **Verdict: SUPPORTED**

## Why picked
Memory protection is the open gap in our agent stack: harnesses don't record
*who may act on* a memory, so an instruction hidden in a fetched page can be
treated as the user's own. ClawMMU labels every memory with
(owner, scope, origins, authority) and checks the labels at four gates —
context entry, store write, store recall, tool-call drive — with the central
rule that *information may rise in authority, but authority never lifts
itself*. Repo fit: `arxiv-lab` memory line (CMP memory, TRACER) and the
Hermes harness; directly addresses the injection/phishing vigilance gap found
in the 2026-10-17 model comparison.

## What was built
- `src/arxiv_lab/memory/clawmmu.py` (stdlib only): authority lattice
  (system 4 > user 3 > tool_trusted 2 > tool 1 > unknown 0), `Label`,
  `MemoryItem`, `ClawMMU` with the four gates, `derive_label` (derived
  content takes the min authority of its sources — no self-lift),
  `endorse` (legitimate elevation only by a higher principal), tamper-evident
  seals (HMAC) on stored items, per-tool authority floors (sensitive tools
  such as `send_email` require USER).
- `tests/test_memory_clawmmu_cpu.py`: deterministic battery with a
  **worst-case fooled model** (obeys the last imperative instruction in
  context regardless of provenance) driving both a naive harness and the
  ClawMMU harness.

## Results (measured)
- 14/14 checks passed.
- Attack-block rate (ClawMMU): **6/6** — direct injection exfil (A1),
  poisoned-memory write capped + exfil (A2), owner/origin spoof quarantine
  (A3), summarization-laundering exfil (A4). The tool-drive gate fires even
  though the model was fully fooled, exactly the paper's claim.
- Legit-pass rate: **4/4** — user-authorized send, read-only tool on
  trusted-tool data, user-endorsed tool content (elevation path works, no
  false-positive lockout); sensitive tool with tool-only drivers correctly
  denied.
- Tamper-evidence: post-write label flip detected at recall (ProtectionFault).
- Naive baseline: 0/2 attacks blocked (both executed) — the gates, not the
  model, are doing the work.

## Verdict: SUPPORTED
Core claim reproduced directionally at bounded scale: provenance-based gates
stop memory attacks even with a fully fooled model, with zero false positives
on legitimate flows.

## Honest limits
- The "fooled model" is a deterministic worst-case stub, not a real LLM; a
  live-model spot check (injection fools the model, gate still blocks) was
  not run.
- Authority floors per tool are hand-set; the paper's full contract/airlock
  formalism is not implemented.
- No adversarial attempt to forge the HMAC seal key was tested (key lives in
  harness memory, as in the paper's trust model).
