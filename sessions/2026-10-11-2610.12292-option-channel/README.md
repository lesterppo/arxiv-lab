# Session: Option-Channel Attack on Typed Decision Guardrails (2026-10-11)

**Article:** [arXiv:2610.12292](https://arxiv.org/abs/2610.12292) — "One Word
Opens the Gate: The Option-Channel Attack on Typed Decision Models as Agent
Guardrails" (Azizi, Potraghloo, Pedram). Code:
[ArminAzizi98/option-channel-attack](https://github.com/ArminAzizi98/option-channel-attack)
(MIT).

**Scores:** AGENT=5, TUNE=1.

## Why picked
The paper evaluates *typed decision models* — models that read text and
return a probability over caller-defined options (name + short definition),
generating no text — in the exact role Peter's stack now uses them:
`src/arxiv_lab/backends/systemone.py` (Clef `clef-flash` via Ollama
`/v1/systemone`, deep-tested live on 2026-10-11) is a typed `choice` /
`score` / `noul` backend for agent guardrails. The paper's claims map
1:1 onto that backend: fail-open vs fail-closed error directions, the
option-channel attack (rename the permissive option's label, definition
untouched), benign-text attacks, confidence-based escalation, and the
deterministic typed-value baseline. Most testable agent paper in the
2026-10-11 window.

## What was built
- `guardbench.py` — synthetic GuardBench-style suite: 96 tool-call
  proposals (16 per policy, 8 allow / 8 deny) built from six typed
  attributes; ground truth *computed* from attributes (no label noise).
  Six policies: single-condition (P1), conjunctive (P2, P5), disjunctive
  (P3), negated (P4), mixed (P6). Every trace carries an explicit
  attacker-controlled span (the operator `note`). Ships both attacks:
  `with_benign_logs()` (six policy-irrelevant server-log lines) and
  `choice_question(allow_label=...)` (option-channel rename, descriptions
  identical), plus the deterministic typed-value rule (`ground_truth`)
  and fail-open/fail-closed + margin metrics.
- `battery_option_channel.py` — VM-side runner (stdlib only) against
  `clef-flash:latest` on a Colab T4 via localhost `/v1/systemone`.
  Phases: clean baseline (96) → benign-log attack on correctly-denied
  items → option-channel attack (labels `remediate`, `proceed`,
  `safe_pass`) → confidence-margin analysis on reversed decisions →
  deterministic-rule accuracy.
- `systemone_client.py` — vendored copy of the tested
  `src/arxiv_lab/backends/systemone.py` used by the battery.
- `test_option_channel_cpu.py` — 12/12 CPU checks (no model): suite
  determinism/balance, per-policy ground-truth cases, benign-log hygiene
  (no policy keywords), rename-keeps-description, metric math, 100%
  deterministic rule, stub-backend end-to-end.

## Results

Live battery on `clef-flash:latest` (Colab T4, localhost `/v1/systemone`;
288 decide calls, 0 errors, ~15 min). Full per-item data: `results.json`.

| phase | metric | result |
|---|---|---|
| clean (96 items) | fail-open | **0.000** (0/48 deny-items) |
| clean | fail-closed | **0.229** (11/48 allow-items) |
| attack A: benign server logs (48 correctly-denied) | fail-open | **0.000** |
| attack B: option renamed → `remediate` | fail-open | **0.000** (0/48) |
| attack B: → `proceed` | fail-open | **0.000** (0/48) |
| attack B: → `safe_pass` | fail-open | **0.000** (0/48) |
| reversed decisions (for confidence analysis) | n | 0 — none reversed |
| deterministic typed-value rule | accuracy | 1.000 (96/96, by construction) |

Per-policy clean: fail-open 0.000 on all six policies; fail-closed
P1 0.000, P2 0.500, P3 0.125, P4 0.250, P5 0.250, P6 0.250. The P2
fail-closeds are all secret-data-class actions that are *not* file writes
(db_query / email_send / http_post on secret targets) — the gate
overgeneralizes "secret → deny" beyond the policy's file-write scope.

## Verdict

**Not reproduced on this backend — and that is the finding.** Neither
attack moved the gate: benign logs left fail-open at 0%, and renaming
the permissive option (including the paper's strongest label
`remediate`, which took Qwen2.5-7B to 100% fail-open) left it at 0%
across all three labels. The gate is deny-conservative (0% fail-open,
23% fail-closed), matching the paper's "blocks nearly everything"
archetype — a low fail-open rate that reflects the model's default,
not robustness per se.

The paper itself reports this exact immunity pattern for the `von` and
`rlcd` calling libraries: *"it is a calling-library property, not a
model property"* — when the option label never reaches the network
(definitions only), renaming it cannot move the decision. Clef's
`/v1/systemone` endpoint behaves the same way here: the option channel
does not reach (or does not move) the decision. For Peter's stack this
is good news — the `systemone` backend actually deployed as his agent
guardrail resists both attacks from the paper. The confidence-escalation
claim is untestable here (zero reversed decisions to compare margins
on); the deterministic-rule point is confirmed trivially (100% by
construction — it demonstrates the paper's "makes the model unnecessary"
argument, not a real accuracy claim).

## Honest limits
- The battery ran against one decision model (`clef-flash`, 9B) — the
  paper evaluated seven open-weight models; single-model results don't
  generalize to the model class.
- GuardBench here is synthetic and small (96 items); labels are computed
  from the same attributes the deterministic rule uses, so its 100% is
  by construction — it demonstrates the paper's "makes the model
  unnecessary" point, not a real-world accuracy claim.
- Tunnel POSTs from this VM hang (trycloudflare egress blocked), so all
  live calls ran on the Colab VM via localhost; orchestration only from
  the host side.
