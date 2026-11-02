# AI agent decision dilemmas — three papers from a YouTube video (AP9iu_yQoCg)

Source: YouTube video AP9iu_yQoCg, "Codex、Claude Code 怎麼用才高效？
AI Agent 的決策困境" (How to use Codex/Claude Code efficiently? The AI
agent's decision dilemma), channel wow, 23:34, uploaded 2026-11-01.
The video discusses three papers; this session tests their core
mechanisms in miniature. It is a user-requested video test, not a daily
cron run, so it sits under a video-derived session name rather than a
single arXiv id.

Deviation from the repo's one-session-per-article rule, made
deliberately: the three papers were tested jointly in one harness run
(one video, one subject model, one results file), so splitting them
into three folders would duplicate the shared harness three times.
Each paper's verdict is still stated separately below.

Portability note: this session's Rust harness is VM-specific — it
fetches its Anthropic credential as a surrogate from the authd Unix
socket. It is NOT portable arxiv-lab code and must not be graduated
to `src/` without replacing that auth with an env-var key, per the
repo's portable-Linux rule. No secret value is in this folder; the
code only ever handles the `hsurr:` surrogate string.

Articles (all published 2026-10-26, abstracts verified via the arXiv API):

1. **Your Wish Is My Commit: Over-Compliant Refactoring in AI Coding Agents**
   (arXiv:2610.36346) — Codex/Claude Code keep editing clean code across
   3,000+ turns, rewriting up to 31× before first stopping, and resume
   after declaring the task complete.
2. **Thinking-in-Bets at Test Time Reduces Sycophantic Revisions in LLM Agents**
   (arXiv:2610.34654) — a pre-complaint Claim Checklist (claims, checks,
   confidence) nearly eliminates false reversals under mistaken pushback,
   while a post-complaint checklist recovers little; it also fixes more
   real bugs than the baseline.
3. **Continue or Restart? The Limits of Early Failure Prediction in Coding Agents**
   (arXiv:2610.35971) — within-task early-failure discrimination is near
   chance (AUC ≈ 0.50 at 5k–20k tokens, 147,002 traces); under a 75k-token
   cap, uninterrupted continuation beats both learned and random
   restarting on solve rate at lower mean token cost, with the restart
   gain coming almost entirely from between-task allocation.

## Method

Subject: `claude-haiku-5-5` via the Anthropic API (connector surrogate,
never a raw key in this repo). Temperature is omitted: the parameter is
deprecated for this model and rejected with HTTP 400. This is NOT the
papers' own subjects (Codex/Claude Code agents, Gemini 3.5–4.0, Qwen3-0.6B
monitor), so verdicts are about the mechanisms at small n.

- Python prototype, 2026-11-01 (`python/`): 3 snippets × 5 refactor rounds,
  10 trivia items × 3 conditions, 6 tasks × 2 policies. 77 records in
  `python/results.jsonl`. Note: yt-gem distillation failed that day
  (Gemini session expired, NOT_SIGNED_IN); the papers were identified
  from the video description via yt-dlp (android player client) and
  verified on arXiv.
- Rust deep test, 2026-11-02 (`rust/`, `results.jsonl`, `DEEP_TEST.md`):
  8 snippets × 8 rounds × fresh-call and multi-turn variants (128 calls),
  20 trivia items × 3 conditions + 12 patch-review cases (168 calls),
  10 tasks × 3 reps × continue/restart (88 rounds). 272 records.
  `cargo test` 8/8; release binary 6.8 MB (binary itself not committed).
  A runtime restart interrupted the first P1 run at 30/128 records; the
  harness was made resumable (P1 checkpoint, P2/P3 skip of logged keys)
  and rerun cleanly, with no duplicated records.

Run it (from `rust/`):

```bash
cargo test --offline
cargo build --release --offline
./target/release/yt-ap9iu-rs all --out results.jsonl
```

## Verdicts (Rust deep run; full numbers in DEEP_TEST.md)

- **P1 — PARTIAL.** Fresh re-asking reproduces the resume pattern:
  17/64 rounds changed code, 6/8 chains edited again after saying
  NO CHANGE NEEDED. Multi-turn history does the opposite: 9/64 changed,
  2 resumed rounds — once the model has said NO CHANGE in-conversation,
  it mostly stays stopped. All 128 versions passed their tests; worst
  growth was the Stack chain at 3.28× original length.
- **P2 — NOT REPRODUCED for false reversals; SUPPORTED directionally
  for the true-bug balance.** Trivia false reversals: 0/58
  initially-correct cases in all three arms (baseline, pre-checklist,
  post-checklist) — there is no baseline flipping to reduce at this
  difficulty. False bug reports: 0 unnecessary edits in both arms
  (16 cases). True bug reports: baseline fixed 2/4, pre-checklist
  fixed 4/4 (n = 4 per arm, directional only).
- **P3 — PARTIAL for continuation's solve advantage; NOT REPRODUCED
  for its token saving; AUC UNTESTED.** Continue solved 28/30 vs
  restart 24/30 (gap driven by my_atoi 3/3 vs 0/3 and split_csv_line
  1/3 vs 0/3), but cost more tokens (881 vs 760 mean per run).
  Within-task AUC of confidence vs first-attempt pass is undefined
  for all 10 tasks: easy tasks are all-pass with near-constant
  confidence, and hard tasks produced no CONFIDENCE line at all —
  missing exactly where it would matter. That is no measurable
  signal, not evidence for AUC ≈ 0.50.

## Known limits / harness lessons

- One model, no temperature control, so runs are not bit-reproducible.
- Trivia/patch items are easy and verifiable; the papers' hard settings
  (SWE patch review, report drafts, long coding trajectories, 75k-token
  caps) are not represented.
- Budget checklist + code replies at ≥1500 tokens for this model: at
  500 tokens it can return an empty text block (budget spent,
  apparently including thinking tokens), which masquerades as a
  refusal. The first pre-checklist true-bug run scored 0/4 this way
  and was rerun at 1500 tokens (4/4).
- A reply that only *demonstrates* a function is not a revised
  function; the Rust extractor returns None unless a block defines
  `def <name>` / `class <name>`.
- Final-answer classification must handle negated mentions of the
  wrong value ("150,000 is not correct") or stood-firm replies are
  mislabelled as reversals; both-values replies are labelled Unclear
  and were audited from their opening lines.
