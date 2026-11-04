# RuntimeTickets — Rust second stage (arXiv 2611.03772)

Std-only Rust port of the Python mechanism in `../runtime_tickets.py`
(`src/arxiv_lab/agent/runtime_tickets.py` in the repo). No crates, no
network, no model.

## Scope (standing rule: port the mechanism module, pure logic only)

Ported:

- `Obligation.is_grounded` — every cited line index exists in the
  skill source.
- `Ticket::tighten` — refuses a new (recipient, object) pair or a
  cap above the host's cap; surviving caps are the per-pair minimum.
- `Ticket::gate` — forward iff pair allowed and amount within cap.
- `run_trace`, `decode`, `decode_gate_trace` — monitor trace and
  deterministic decoder (verified iff obligations grounded, every
  forwarded send within ticket, no effect on a blocked send).
- Baselines: `scanner_flags` and `coarse_allowlist_allows`.
- `catalog_ticket` / `catalog_sends` — the same 10+10 catalog the
  Python test uses, so `verify` and the tests check identical numbers.

NOT ported (out of scope, recorded not forced):

- The paper's LLM obligation-proposal step: out of scope in the
  Python implementation too (obligations are given as data), so
  there is nothing to port.
- Any framework integration or network-egress interception: the
  paper evaluates at the egress boundary of three agent frameworks;
  this port, like the Python test, models sends as data, not as live
  network calls.

## Deliberate divergences from Python

None in the ported logic. Two representational notes (not behaviour
changes):

- Amounts are `i64` minor units in both implementations; the Rust
  oracle iterates the allowed map instead of a direct lookup, so the
  fuzz-vs-oracle tests compare two genuinely different code paths.
- The Python scanner regex `\b(stem\w*)` (case-insensitive) is
  implemented as: lowercase the description, split into alphanumeric
  tokens, flag iff some token starts with a stem (`exfiltrat`,
  `attacker`, `secret`, `credential`, `password`, `steal`, `leak`).
  This is exactly the regex's semantics on token text: a stem must
  start at a word boundary (`topsecret` is clear in both), and a
  stem prefix matches longer words (`credentials`, `exfiltrating`,
  `leakage` flagged in both). The parity sweep covers both sides
  with the same description pool.

## How to run

```bash
cargo test    # 8 integration tests
cargo run -- verify    # model-level checks, must print PASSED
```

The parity sweep is driven from the session root (needs the built
debug binary at `rust/target/debug/runtimetickets`):

```bash
python3 parity_runtimetickets.py    # 420 cases, 0 mismatches
```

## Live results

- `runtimetickets verify`: PASSED (benign 10/10, malicious 0/10,
  scanner overlook 7/10 + overkill 2/10, coarse allow-list
  malicious 8/10, decoder verified, tamper flagged, tighten
  refusals).
- Cargo tests: **8/8** (catalog + cap boundary, 441 exhaustive
  cells vs oracle, 5,000 random cases vs oracle, 100,000-case soak
  invariants, 8-thread identity, tighten rules + 500-case fuzz
  invariant, decoder cases, scanner cases).
- Python-vs-Rust parity sweep: **420 cases, 0 mismatches**.

`rust/target/` is a local build artifact and is never pushed.
