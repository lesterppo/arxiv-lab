# RECEIPT — Rust second stage (arXiv 2611.01930)

Std-only Rust port of the Python mechanism in `../receipt.py`
(`src/arxiv_lab/agent/receipt.py` in the repo). No crates, no
network, no model.

## Scope (standing rule: port the mechanism module, pure logic only)

Ported:

- `simulate_outcome` — the pure (guarded, mode, policy) -> outcome
  model, exactly (closed+guarded -> status 0, no action, false
  success; open+guarded -> timeout kill; receipt+guarded -> success
  iff the policy answers).
- `extract_waiting_prompt` — newest-first scan for a line ending in
  `?` or a `[y/N]`/`[yes/no]` marker (any case).

NOT ported (out of scope, recorded not forced):

- The live subprocess harnesses (`run_closed`,
  `run_open_unattended`, `run_receipt`): process spawning and
  timeout/kill semantics are OS-interaction, not pure mechanism
  logic, and the Python test already covers them with real
  subprocesses. The Rust `verify` command reproduces the
  model-level checks (catalog counts and all outcome cells) that
  those live runs imply.

## Deliberate divergences from Python

None in the ported logic. Two representational notes (not behaviour
changes):

- Timeout kill is encoded as status `-1` in the `parity` command's
  text output (Rust `Option<i32>` has no text form); the Python
  parity script maps Python `None` to `-1` before comparing.
- The Python detector's regex `(\?|\[y/N\]|\[Y/n\]|\[yes/no\])\s*$`
  (case-insensitive) is implemented as: stripped line ends with `?`,
  or its lowercase form ends with `[y/n]` or `[yes/no]`. The
  `[Y/n]` alternative is subsumed by case-insensitivity.

## Live extensive test (all run live)

- `cargo test`: **5/5** — exhaustive 24 outcome cells vs a
  hand-written oracle table; 5,000 random cases vs the oracle;
  100,000-case soak checking the paper's invariants (a timed-out
  call never claims an action; a false success is status 0 with no
  action, closed+guarded only); detector cases (including a `?`
  mid-output that must NOT match); 8-thread identity (20,000
  calls, stateless).
- `receipt verify`: PASSED — catalog 6/6 closed false successes,
  6/6 open timeouts, 10/10 receipt completions, 0 false successes.
- Python-vs-Rust parity sweep (`../parity_receipt.py`): **212 cases
  (24 exhaustive + 188 random), 0 mismatches** — PARITY VERIFIED.

## Layout

- `Cargo.toml`, `Cargo.lock` — package `receipt`, edition 2021.
- `src/lib.rs` — `Mode`, `simulate_outcome`, `extract_waiting_prompt`.
- `src/main.rs` — `receipt verify` / `receipt parity` CLI.
- `tests/deep.rs` — the extensive tests above.
- `results.json` — the numbers from the live run.

No `target/`, no binary, no secrets in this subsession.
