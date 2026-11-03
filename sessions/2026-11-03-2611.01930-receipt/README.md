# RECEIPT — repairing the return assumption (arXiv 2611.01930)

- Article: https://arxiv.org/abs/2611.01930 — "RECEIPT: Measuring and
  Repairing the Return Assumption of LLM Agents in Web Service
  Operations" (submitted 1 Nov 2026).
- Why picked: AGENT=5, TUNE=1. The claim is about the agent harness
  itself — exactly what `arxiv-lab`'s harness library is — and it is
  directly testable on CPU with real subprocesses: a harness that
  judges a tool call by exit status alone misreads commands that ask a
  question before acting. Repo fit: `arxiv-lab` harness library
  (next to the Mingbird trio and VBL), and every local agent stack
  that launches shell commands.
- Verdict: **SUPPORTED** (directional reproduction at bounded scale;
  see limits).

## What was built

`receipt.py` — three harness modes over the same command catalog, plus
a waiting-prompt detector and a pure outcome model:

- `run_closed`: stdin = DEVNULL.
- `run_open_unattended`: stdin open, nobody types, harness timeout.
- `run_receipt`: stdin kept open; process output is read until a
  waiting prompt appears; the prompt is returned as waiting state to
  a scripted answer policy, the answer is written, and the call
  completes. This is the paper's repair, minus the syscall-level
  observation (we read the prompt from the process's own stdout
  instead of from system calls — recorded as a limit).
- `extract_waiting_prompt`: last prompt-like line of the output
  (ends with `?` or a `[y/N]`/`[yes/no]` marker); plain status output
  does not match.
- `simulate_outcome`: pure (guarded, mode) -> outcome model, shared
  with the Rust port for parity checking.

Graduated to `src/arxiv_lab/agent/receipt.py` +
`tests/test_agent_receipt_cpu.py` (same test, run from the repo root).

## Results (deterministic CPU test, real subprocesses)

Catalog: 6 guarded tasks (a question guards the action) + 4
unguarded tasks; each guarded task has an unguarded twin differing
only in the guard (the paper's Tripwire design, 25 pairs at full
scale). The guarded helper exits 0 without acting on EOF — the
closed-input failure mode the paper measures on Litmus.

| harness mode | guarded | unguarded |
|---|---|---|
| closed input | false success 6/6 (status 0, action NOT done) | ok 4/4 |
| open, unattended (2 s timeout) | timeout 6/6 | ok 4/4 |
| RECEIPT (policy answers `y`) | completed 6/6, prompts surfaced 6/6 | ok 4/4 |

- False-success rate on guarded tasks: closed 100% -> RECEIPT 0%
  (catalog scale; the paper reports −73.5% false claims of success
  across 5 LLMs on its full benchmark — we reproduce the direction
  and the mechanism, not that number).
- Tripwire pairs: under closed input the guarded member false-
  succeeds while its unguarded twin succeeds, for all 6 pairs.
- Detector unit cases pass: prompt lines extracted, plain output
  returns None, newest prompt wins.

## Rust second stage

`rust/` — std-only port of the outcome model + prompt detector
(the live-subprocess part is out of scope by design; see
`rust/README.md` for the exact scope and divergences). Live results:

- `receipt verify`: reproduces the model-level checks (catalog
  6/6, 6/6, 10/10, 0 false successes) — PASSED.
- Cargo tests: **5/5** (exhaustive 24 cells vs oracle, 5,000 random
  cases vs oracle, 100,000-case soak invariants, detector cases,
  8-thread identity).
- Python-vs-Rust parity sweep (`parity_receipt.py`): **212 cases,
  0 mismatches** — PARITY VERIFIED.

## Honest limits

- No LLM in the loop: the "agent" is a scripted policy that answers
  `y` to every surfaced prompt. The paper's LLM-level numbers
  (2.9x token multiplier, 7th→3rd ranking move, 73.5% reduction) are
  not reproduced here — only the harness mechanism that causes them.
- Prompt observation is via process stdout, not the paper's
  syscall-level read of what the process waits for; a process that
  waits without printing would not be surfaced by this implementation.
- Catalog scale: 10 invocations vs the paper's 158-invocation Litmus
  and 91,469 production calls; the 100% closed-input false-success
  rate reflects a catalog where every guarded command defaults to
  exit-0-on-EOF, not a population estimate (paper: 11.5%).
- A policy that answers `y` indiscriminately would also confirm
  genuinely dangerous actions — RECEIPT surfaces the wait; it does
  not decide whether the action is authorized. That authorization
  check belongs in the policy, not the harness.
