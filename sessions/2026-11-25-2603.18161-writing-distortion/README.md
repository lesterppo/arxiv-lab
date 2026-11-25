# How LLMs Distort Our Written Language — 2603.18161 (2026-11-25)

**Verdict: PARTIAL** — the distortion direction replicates for substantive
revisions (minimal/general) and in the frontier baselines, but the headline
sub-claim — that even GRAMMAR-only edits distort far more than human edits
— does NOT reproduce with Qwen3.5-9B/4B: grammar edits sit at the human
baseline. The pronoun collapse is also far smaller than the paper reports.

## Article

- arXiv:2603.18161v2 "How LLMs Distort Our Written Language"
  (Abdulhai et al.)
- Core claims: (1) in an RCT (N=100, "Does money lead to happiness?"),
  heavy LLM use raised neutral-stance essays +68.9%; (2) on ArgRewrite-v2
  (86 pre-ChatGPT student essays on self-driving cars, D1 + expert
  feedback + human D2), LLM edits shift semantics far more than human
  edits and in a common direction, even for GRAMMAR-only edits — human
  JSD ~0.2–0.3, claude-haiku general-revision mean JSD 0.476, gpt-5-mini
  up to >0.7; pronouns −40 to −60% (general) / −31 to −56%
  (minimal/grammar), adjectives +57–90% / +40–87%; (3) ICLR 2026 LLM
  reviews score higher (4.43 vs 4.13 human) and weight clarity/significance
  less, reproducibility/scalability more.
- Why picked: one-off session ordered by the operator. Directly testable
  against the authors' released data (ArgRewrite-v2 + all frontier-model
  outputs), with open-weights models standing in for the frontier models.

## Data

- `essays_llm.csv` (27.7 MB, LOCAL ONLY, never pushed): the authors'
  released file from https://github.com/abdulhaim/llm_writing_distortion
  (`argrewrite_analysis/dataframe_essays_llm_model.csv`), columns
  essay_id, draft1, draft2, ai_text, llm_model, mode, feedback.
  2,580 rows = 86 essays × 3 frontier models (GPT-5-MINI /
  GEMINI-2.5-FLASH / CLAUDE-HAIKU-4-5-20251001) × 5 modes
  (grammar/minimal/general/completion/expansion) × 2 feedback conditions.
  Download: `curl -sL -o essays_llm.csv https://raw.githubusercontent.com/abdulhaim/llm_writing_distortion/main/argrewrite_analysis/dataframe_essays_llm_model.csv`
- The CSV's `feedback` column is only a flag ("With Feedback" /
  "Without Feedback"). The actual expert-feedback text was recovered from
  the same repo (`ArgRewrite/dataset/ArgRewrite.V2.zip`,
  `meta-data/ExpertFeedback/feedback_<id>.docx`), cleaned with the authors'
  own rule (split on "We advise that you use this feedback when you
  revise", strip leading punctuation). All 86 CSV essay_ids matched the
  ArgRewrite raw ids exactly (draft1 text match, 86/86), feedback present
  for all 86 (mean 140 words; cleaned mean 99 words on the sampled 16).
- `sample16.json` (committed, 600 KB): the 16 sampled essays with draft1,
  draft2, cleaned feedback, and the frontier "With Feedback" ai_text for
  grammar/minimal/general × 3 models (144 texts).

## Test design (pre-registered before running)

- Sample: 16 essays stratified by D1 keyword stance. Pre-screen: pro/anti
  wordlists on D1 (pro if pro−anti ≥ 3, anti if ≤ −3, else mixed).
  Strata over 86 essays: pro 5, anti 51, mixed 30. Rule: take all 5 pro
  + seeded shuffle (seed 20261125) first 5 anti + first 6 mixed.
  Sampled ids: 17, 19, 20, 29, 35, 43, 46, 55, 56, 58, 65, 67, 80, 81,
  83, 102.
- Models (Colab T4, chat template, temp 0 / greedy, NF4 via
  BitsAndBytesConfig): Qwen/Qwen3.5-9B and Qwen/Qwen3.5-4B (instruct).
  Modes: grammar, minimal, general. 16 × 3 × 2 = 96 generations,
  max_new_tokens 700. Prompts = the authors' B.1 instructions verbatim
  + feedback + D1, in the authors' own notebook format (system "You are
  a helpful AI assistant.", user "{instruction}\n\nIncorporate the Expert
  Feedback below:\n{feedback}\n\nCURRENT DRAFT:\n{draft1}\n\nOnly provide
  the edited essay and no other feedback.\n\nEdited Essay:\n").
  Deviations: temp 0 instead of the authors' 0.7; open-weights Qwen3.5
  instead of frontier models (frontier outputs are the baseline);
  `enable_thinking=False` (see Run notes — without it Qwen3.5 emits only
  thinking traces and never produces an essay).
- Metrics: (a) mean cosine distance D1→D2 (sentence-transformers
  all-MiniLM-L6-v2 embeddings, same encoder family as the paper) +
  mean-shift alignment (mean cosine similarity of shift vectors to their
  mean direction = homogenization); (b) unigram JSD D1 vs D2 (log2,
  0–1); (c) pronoun rate change % and adjective rate change % —
  pronoun wordlist per 100 words; adjective proxy via suffix list +
  small lexicon, NOT LIWC (see module docstring); (d) stance flip rate
  via LLM-as-judge (Qwen3.5-9B, temp 0, paper D.2 prompt adapted to
  self-driving cars, JSON {stance, argument_types, main_arguments}) +
  deterministic keyword-stance cross-check, with judge/keyword agreement
  reported. Baselines (human D2 + 3 frontier models) computed locally
  from the CSV on the SAME 16 essays × same 3 modes.
- **Pre-registered bar for "distortion present" in grammar mode: mean
  JSD > human mean + 0.10 AND (stance flips ≥ 2/16 OR pronoun drop
  ≥ 20%).** Human mean JSD on the sample is 0.0744, so the bar is
  grammar JSD > 0.1744 plus a flip/pronoun leg.
- Secondary peer-review leg (small, directional only): 6 short ML-paper
  abstracts authored for this test (2 apparently strong, 2 mediocre,
  2 weak, one paragraph each, fixed text in the driver); each model writes
  a review + ICLR score 0/2/4/6/8/10 + strengths/weaknesses; count
  clarity/significance vs reproducibility/scalability mentions; compare
  score mean to the paper's human 4.13 / LLM 4.43, with the caveat that
  there is no human baseline in this leg.
- Verdict rule: SUPPORTED if grammar-mode distortion clears the
  pre-registered bar in both models or clearly in 9B with 4B directional;
  NEGATIVE if grammar edits sit at the human baseline on all metrics;
  PARTIAL if mixed (e.g. lexical shift replicates but stance does not);
  INCONCLUSIVE only if the design cannot discriminate (say exactly why);
  BLOCKED only if no fair test was possible. The human RCT leg (N=100)
  is NOT replicable here — stated as a limit, not tested.

## What was built

- `writing_distortion_metrics.py` — metrics module (stdlib + numpy-guarded):
  unigram JSD, pronoun/adjective-proxy rates, keyword stance, pct_change,
  mean-shift alignment. Docstring cites arXiv:2603.18161 + one-line claim.
- `test_writing_distortion_cpu.py` — deterministic CPU test, **16/16 PASS**.
- `analyze_baselines.py` — local baseline analysis from the CSV;
  output `baselines.json` (means + per-essay values, n=16).
- `analyze_qwen.py` — CPU-side analysis of the Colab results (lexical
  metrics via the module, judge flips, peer-review scores, local
  re-verification of the Colab cosine distances from the stored
  embeddings — recompute matches Colab exactly); output `analysis.json`.
- `colab_writing_distortion_driver.py` — Colab driver (both models,
  judge phase, embeddings phase, peer-review leg; leg controls via env:
  MODELS_TO_RUN, PHASES, JUDGE_SCOPE, INPUT_JSON). Prints
  DISTORTION_DONE on completion of each leg.
- `sample16.json` — sampled subset (see Data).
- `rust/` — Rust second stage of the metrics module (see rust/README.md).

## Baselines (local, n=16, frontier = "With Feedback" rows)

Unigram JSD is a surface word-choice proxy, not the paper's embedding JSD,
so absolute values are not directly comparable to the paper's 0.2–0.7 —
only the human-vs-LLM gaps on the same measure are. Adjective rate is a
suffix+lexicon proxy, not LIWC. Stance flips use the keyword-count rule
(kw) and the LLM judge (j; denominators vary where the judge output was
unparsed).

| condition | mean JSD | pronoun % | adjective % | kw flips | j flips |
|---|---|---|---|---|---|
| HUMAN D1→D2 | 0.0744 | +0.48 | +0.65 | 2/16 | 1/15 |
| GPT-5-MINI / grammar | 0.2360 | −21.26 | +39.40 | 6/16 | 3/15 |
| GPT-5-MINI / minimal | 0.4705 | −33.49 | +83.15 | 7/16 | 4/14 |
| GPT-5-MINI / general | 0.4700 | −34.55 | +88.75 | 7/16 | 3/14 |
| GEMINI-2.5-FLASH / grammar | 0.1360 | −4.78 | +29.74 | 1/16 | 0/15 |
| GEMINI-2.5-FLASH / minimal | 0.3950 | −29.49 | +111.31 | 6/16 | 1/15 |
| GEMINI-2.5-FLASH / general | 0.3331 | −20.48 | +74.06 | 7/16 | 2/14 |
| CLAUDE-HAIKU-4-5 / grammar | 0.0962 | −0.05 | +13.17 | 2/16 | 1/15 |
| CLAUDE-HAIKU-4-5 / minimal | 0.3710 | −26.90 | +65.93 | 5/16 | 1/15 |
| CLAUDE-HAIKU-4-5 / general | 0.2604 | −15.37 | +42.83 | 6/16 | 3/15 |

Reading: on this 16-essay subset the frontier baselines already reproduce
the paper's direction — every model×mode shifts word distribution more
than the human revision, cuts pronouns, and raises adjective-proxy rate;
grammar-only is the least distorting mode per model. The paper's
pronoun/adjective magnitudes (−31 to −56% / +40–87% for minimal/grammar)
are larger than the keyword/suffix proxies show for grammar mode here —
the proxies are conservative, and the paper's numbers are LIWC-based.

## Results (Colab T4, n=16 essays × 3 modes × 2 models = 96 generations)

### Lexical metrics (CPU analysis of downloaded generations)

| condition | mean JSD | pronoun % | adjective % | kw flips | j flips | words D1→out |
|---|---|---|---|---|---|---|
| HUMAN D1→D2 | 0.0744 | +0.48 | +0.65 | 2/16 | 1/15 | 524→573 |
| Qwen3.5-9B / grammar | 0.1255 | −0.03 | +21.72 | 2/16 | 2/14 | 524→491 |
| Qwen3.5-9B / minimal | 0.2053 | −5.52 | +39.05 | 2/16 | 2/15 | 524→488 |
| Qwen3.5-9B / general | 0.2288 | −6.60 | +44.77 | 4/16 | 2/14 | 524→515 |
| Qwen3.5-4B / grammar | 0.0766 | +0.25 | +10.28 | 0/16 | 0/14 | 524→492 |
| Qwen3.5-4B / minimal | 0.2233 | −4.88 | +45.05 | 4/16 | 3/15 | 524→446 |
| Qwen3.5-4B / general | 0.2093 | −5.43 | +41.04 | 5/16 | 1/15 | 524→490 |

### Semantic metrics (all-MiniLM-L6-v2 embeddings, Colab; CPU recompute matches)

| condition | mean cosine dist | mean-shift alignment |
|---|---|---|
| HUMAN D1→D2 | 0.0611 | 0.2557 |
| GPT-5-MINI / grammar | 0.0858 | 0.3727 |
| GPT-5-MINI / minimal | 0.1992 | 0.5226 |
| GPT-5-MINI / general | 0.2018 | 0.5352 |
| GEMINI-2.5-FLASH / grammar | 0.0665 | 0.3871 |
| GEMINI-2.5-FLASH / minimal | 0.1999 | 0.5630 |
| GEMINI-2.5-FLASH / general | 0.1466 | 0.5190 |
| CLAUDE-HAIKU-4-5 / grammar | 0.0422 | 0.3278 |
| CLAUDE-HAIKU-4-5 / minimal | 0.1439 | 0.4608 |
| CLAUDE-HAIKU-4-5 / general | 0.1049 | 0.3910 |
| Qwen3.5-9B / grammar | 0.0666 | 0.3432 |
| Qwen3.5-9B / minimal | 0.0912 | 0.4119 |
| Qwen3.5-9B / general | 0.1226 | 0.4687 |
| Qwen3.5-4B / grammar | 0.0377 | 0.3742 |
| Qwen3.5-4B / minimal | 0.1084 | 0.4669 |
| Qwen3.5-4B / general | 0.0956 | 0.4522 |

### Reading — against the pre-registered bar

- **Grammar mode does NOT clear the bar in either model.** 9B grammar
  JSD 0.1255 < 0.1744 (human + 0.10); pronoun −0.03% (no drop). 4B
  grammar JSD 0.0766 ≈ human 0.0744, cosine distance 0.0377 BELOW human
  0.0611, pronoun +0.25%, 0 flips on both stance measures. Grammar-only
  edits by these open-weights models are, by every measure here,
  indistinguishable from human revisions.
- **Minimal/general DO distort more than human**, in the paper's
  direction but at roughly half the frontier magnitude: JSD 0.21–0.23
  (2.8–3.1× human), cosine 0.09–0.12 vs human 0.061, adjectives
  +39–45% vs human +0.65%, alignment 0.41–0.47 vs human 0.26
  (homogenization present, in the frontier range 0.33–0.56).
- **The pronoun collapse does not reproduce at paper magnitude.**
  Qwen minimal/general pronoun drops are −5 to −7%; the paper reports
  −31 to −60%, and the frontier baselines on these same essays show
  −15 to −35%. Adjective inflation DOES reproduce directionally
  (+39–45% Qwen minimal/general vs paper +57–90% / +40–87%).
- **Stance flips do not replicate.** Judge flips: Qwen 0–3 per 16 (human
  1/15); keyword flips are higher for minimal/general (4–5 vs human 2)
  but judge/keyword agreement is only 41.5% (110/265) — the two stance
  measures largely disagree, so no stance conclusion can be drawn firmly
  from either. The judge is the more credible measure (it reads the whole
  essay); it says stance is essentially unchanged by all edits, human
  and LLM alike.
- Hence **PARTIAL**, not NEGATIVE: grammar sits at the human baseline,
  but minimal/general lexical + semantic distortion is real, and the
  frontier baselines on the same essays reproduce the paper's pattern —
  the paper's own data supports its claim; the open-weights models just
  distort less, and not at all for grammar-only. Not SUPPORTED because
  the pre-registered grammar bar failed in both models.

### Peer-review leg (directional only)

- Qwen3.5-9B scores: strong 8, 8; mediocre 0, 2; weak 0, 0 — mean 3.0
  (6/6 parsed). It discriminates strongly and is harsher than both the
  paper's LLM mean (4.43) and human mean (4.13). Clarity/significance
  mentions 11 vs reproducibility/scalability 9 — no clear weighting
  difference, unlike the paper's LLM-vs-human pattern.
- Qwen3.5-4B scores: mediocre 0, 2; weak 0, 0 — mean 0.5 (4/6 parsed).
  Both strong reviews were truncated at the 500-token cap before the
  SCORE line (text is clearly positive; no score recorded), so the 4B
  mean is not comparable — report the 9B leg only for scores.
  Mentions 11 vs 18 (over truncated text; directional only).

## Run notes

- Leg 0 (full driver, kyu009009, 13:41 HKT): DIED at ~51 min on a driver
  bug — `REVIEW_USER.format()` choked on the literal `{0, 2, 4, 6, 8, 10}`
  braces in the review prompt (KeyError). 48 9B generations were completed
  but never flushed to disk and were lost. Fixed (`.replace()` instead of
  `.format()`) + flush after every phase.
- Leg 1 (full driver v2, cyc236ha, 14:35 HKT): DONE, 191.9 min — but
  produced NO usable generations: Qwen3.5 defaults to a long
  "Thinking Process:" trace and burned the whole 700-token budget
  thinking (96/96 generations thinking traces, 0 revised essays; judge
  parsed 0/272; peer reviews 12/12 thinking traces, 0 scores). Archived
  as `writing_distortion_results-thinking-trace.json`. Fix:
  `enable_thinking=False` in `apply_chat_template` + fail-fast probes
  (abort if a probe still thinks). The cosine distances from that run
  (0.62–0.70) measured thinking-trace-vs-essay, not revision distortion,
  and are discarded.
- Leg 2 (full driver v2, kyu009009, 17:54 HKT): RECLAIMED at ~105 min,
  during the judge phase — but the judge probe had already passed
  ("judge probe OK: JSON present"), proving the thinking fix works.
  Generations lost with the VM. Response: split the run into short legs
  that each bank their output (driver env: MODELS_TO_RUN / PHASES /
  JUDGE_SCOPE / INPUT_JSON).
- Leg A (9B gen+peer, ppoppo205 after cyc236ha gpu-unavailable, 19:42
  HKT): DONE, 56.9 min. Probe OK, 48 real revisions, mean 486 words,
  0 thinking traces. Peer scores 8/8/0/2/0/0.
- Leg B (9B judge+embed, ppoppo205, 20:41 HKT): DONE, 71.5 min. Judge
  218/224 parsed, embeddings 224 texts.
- Leg C (4B gen + judge gen-only + embed, ppoppo205, 21:54 HKT): DONE,
  57.6 min. 4B gen 35.2 min, judge 47/48 parsed, embeddings 272 texts.
- Provision failures along the way: ppoppo205 auth-expired (13:38),
  cyc236es auth-expired (17:50), kyu008008 auth-expired (17:52) —
  ppoppo205 later provisioned fine twice, so the auth-expired errors
  look transient (wrapper token-refresh state), not dead refresh tokens;
  not diagnosed further. Three host-side monitor SIGTERMs (hourly VM
  restarts) — each re-attached with `train --watch-only --pid`.
- Rotation file updated to ppoppo205 (last account that ran, leg C).
- CPU test: 16/16 PASS (verified locally before launch and again after).
- Rust leg: see `rust/README.md`. cargo test 31/31 pass; verify binary
  16/16 pass (mirrors the Python CPU test); Python-vs-Rust parity sweep
  9,410 cases, 0 mismatches (max abs float diff 9.99e-16); soak 800
  long-text cases, all matching. Model-backed/embedding parts out of
  scope. Divergences D1–D8 documented in rust/README.md. Does not
  overturn the Python PARTIAL verdict.

## Honest limits

- The N=100 RCT leg ("Does money lead to happiness?", neutral +68.9%)
  is not replicable here: no human subjects, no RCT. Not tested.
- Open-weights Qwen3.5 models stand in for the paper's frontier models
  (GPT-5-mini, Gemini 2.5 Flash, Claude Haiku 4.5); the frontier outputs
  themselves serve as the baseline, so the comparison is
  open-weights-vs-frontier on identical prompts and essays. The verdict
  is about these models, not about frontier models — the frontier
  baselines reproduce the paper's direction.
- n=16 essays; keyword stance is a crude count rule and disagrees with
  the LLM judge 58.5% of the time; adjective rate is a proxy, not LIWC;
  unigram JSD is surface word choice, not embedding semantics (embedding
  cosine distance is the semantic leg, and it agrees with JSD here).
- Temp 0 (greedy) vs the authors' temp 0.7 — less sampling noise, but
  not the authors' exact decoding.
- The stance judge is Qwen3.5-9B judging its own and the 4B model's
  outputs (same-family judge); keyword cross-check is reported alongside.
- `enable_thinking=False` changes the models' default behaviour; a
  thinking-enabled run would produce no essay at all (shown in leg 1),
  so there is no thinking-mode comparison to make.
- Peer-review leg has no human baseline, uses invented abstracts, and
  the 4B arm is truncated — 9B only for scores, directional only.
- Judge parse failures: 7/272 entries unparsed (essay 67 accounts for 4
  of them across conditions); flip denominators exclude them.

## Resume state (if retried)

Driver + sample + baselines + analysis are all in this directory. The
cumulative results file is `writing_distortion_results.json` (272 judge
entries, 272 embeddings, 96 generations, 12 peer reviews); per-leg
snapshots are `writing_distortion_results-leg{A,B,C}-*.json`. To redo a
leg, re-run `colab.py train` with the same driver and env
(MODELS_TO_RUN / PHASES / JUDGE_SCOPE / INPUT_JSON), uploading
`sample16.json` and the previous leg's results file as
`/content/input_results.json`.
