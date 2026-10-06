# Session 2026-10-06 — IEC: Intent-Execution Correspondence (arXiv:2610.04375)

**Article:** "Do Tool Calls Execute as Intended? Measuring and Repairing
Intent-Execution Correspondence in LLM Agents"
([abs](https://arxiv.org/abs/2610.04375))

**Why picked:** most prospective *agent* article in the 2026-10-06 cs.AI
window (AGENT 5/5, TUNE 2/5). Its core mechanism — a tool call traverses
several hops and any hop can silently change what executes; observe each
hop without executing to name the first mutating one, then deliver in a
hop-unalterable form or refuse — is directly actionable in Peter's stack:
`hermes-colab-cli` has hit exactly this failure family (proxy swallowing
tunnel POST bodies, mangled `exec --code` payloads). Pure CPU-testable, no
GPU needed. Repo fit: `hermes-colab-cli` (exec transport), `deepworld`
(contribution-loop tool calls), arxiv-lab harness.

**Paper claims:** (i) harness path hops mutate tool calls — Claude Code's
Bash tool changed 12.0% of calls carrying code/escapes/long text; all 10
measured harnesses change a call; (ii) for 80.7% of backslash-changed calls
the wrong action runs *without any reported error*; (iii) trajectory-based
judgment attributes 95.1% of production failures to the LLM although the
path caused more than half; (iv) IntAct: observe-what-each-hop-received
names the first mutating hop, and delivering the call in a form the hop
cannot alter (or refusing it) repairs correspondence.

## What was built (agent track, this VM)

- Reusable mechanism: `src/arxiv_lab/iec/iec.py` (+ `__init__.py`) —
  `Call`/`ToolContract` (canonical JSON wire form; the *receiver's own
  parser* decides sameness), five hop classes (`JsonReserializeHop`
  cosmetic, `ShellQuoteHop` backslash-collapsing, `UrlDecodeHop`
  proxy-normalizing, `TruncateHop` payload cap, `EnvExpandHop`),
  `observe()` (per-hop receipts incl. a terminal tool-side receipt, no
  execution), `first_mutating_hop()` (names the hop that *did* the
  mutating), `armor()`/`deliver()` (IntAct: base64 + sha256 armored form —
  immune to backslash/quote/percent mangling; checksum turns truncation
  into refusal, never silent execution), `NaiveTrajectoryJudge`,
  `make_workload()` (5 trap arg classes). Test: `tests/test_iec_cpu.py`
  (all pass).
- Session record: `test_session_cpu.py` (verbatim snapshot) + `results.json`.

## Results (CPU, deterministic — 500 calls, 4-hop production-like pipeline)

| arg class | mutated | note |
|---|---|---|
| plain | 0/100 | control: clean |
| code (backslashes) | 100/100 | shell-quoting hop |
| longtext (>2k chars) | 100/100 | payload-cap truncation → error |
| envlike ($VARS) | 0/100 | control: clean |
| pct (%2F/%25 URLs) | 100/100 | proxy-normalize hop |

- **Claim (i):** 300/500 = 60% of calls mutated by at least one hop; every
  modeled mutating hop class (`shell-quoting`, `proxy-normalize`,
  `payload-cap`) was correctly named as the first mutating hop for its
  victim class. The cosmetic `JsonReserializeHop` was never named.
- **Claim (ii):** all 200 backslash/percent mutations still parsed → the
  wrong action would execute with **no error** (200 wrong-action, 0
  error among them; paper: 80.7%).
- **Claim (iii):** of 300 path-caused failures the naive trajectory judge
  attributed 100% to the LLM (paper: 95.1%).
- **Claim (iv):** IntAct arm — **0 mutated executions**; 400/500 delivered
  intact, 100 refused (truncation → checksum mismatch), all 100 refused
  calls retried over a clean channel → full task success, zero silent
  wrong actions.

## Verdict: GREEN

All four paper claims reproduced qualitatively; the repair mechanism
(IntAct armoring) graduated to `src/arxiv_lab/iec/`.

## Honest limits

- Synthetic hops/workload: absolute mutation rates (60% overall) are
  modeling choices — the pipeline stacks three aggressive mutating hops,
  while the paper's 12.0% is per-harness on real traffic. The qualitative
  claims (mutation happens, silent wrong actions, misattribution, armor
  repair) are what reproduce, not the exact percentages.
- The "LLM" never emits a malformed call in the sim, so the judge's 100%
  misattribution is by construction; the paper's 95.1% is measured on
  production traces with real LLM errors mixed in.
- base64 armor defeats text-mangling hops but cannot survive truncation —
  by design it converts that case to refusal. A production IntAct would
  need chunked reassembly for large payloads.
- No live-LLM validation yet: the detector/naming protocol on real
  `hermes-colab-cli` exec traces is future work (the module is import-ready
  for it).
