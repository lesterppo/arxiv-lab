# yt-ap9iu-rs

Rust port of the Python test harness for the three papers mentioned in
YouTube video AP9iu_yQoCg, run as a deep extensive test (see DEEP_TEST.md
for results). Single binary, no Python harness dependency at runtime —
Python is only used to *execute* the subject code snippets under test.

## Build and test

```bash
cargo test --offline
cargo build --release --offline
```

## Run

```bash
./target/release/yt-ap9iu-rs p1 --out results.jsonl
./target/release/yt-ap9iu-rs p2 --out results.jsonl
./target/release/yt-ap9iu-rs p3 --out results.jsonl
./target/release/yt-ap9iu-rs all --out results.jsonl
```

Stdout is a JSON pointer per record; the full record (including the
model's reply excerpt) is appended to the `--out` JSONL file.

## Auth

The Anthropic credential is never in this repo. It is fetched as an
`hsurr:` surrogate from authd's Unix socket (`/run/hatch/auth/authd.sock`)
and swapped for the real key only on egress to `api.anthropic.com`.
Model: `claude-haiku-5-5`. Temperature is omitted: the parameter is
deprecated for this model and rejected with HTTP 400.

## Resume

A run interrupted mid-way (e.g. a runtime restart) can be resumed by
re-running the same command with the same `--out` path: P1 resumes from
`results.jsonl.ckpt` (chain code + declared round), P2/P3 skip records
whose identity keys are already in the JSONL. No duplicated records.

## Known harness lessons

- Budget checklist + code replies at ≥1500 tokens for this model:
  at 500 tokens the model can return an empty text block (budget spent,
  apparently including thinking tokens), which looks like a refusal.
- A reply that only *demonstrates* a function (`binary_search([], 5)`)
  is not a revised function: `extract_function` returns None unless a
  block defines `def <name>` / `class <name>`.
- Final-answer classification must handle negated mentions of the wrong
  value ("150,000 is not correct") or it mislabels stood-firm replies
  as reversals; replies mentioning both values are labelled Unclear
  and audited from their opening lines.
