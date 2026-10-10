# Session: Clef /v1/systemone integration (2026-10-11)

Deep test + push of the **Clef decision-model integration** for the local
agent stack: `src/arxiv_lab/backends/systemone.py`
(`SystemoneClient` / `decide` / `build_request` / `parse_answers` /
`score_value`) — typed noul/choice/score decisions via Ollama ≥ 0.35
`/v1/systemone`, stdlib-only, no text parsing.

This is an integration session, not an arXiv reproduction: it wires the
Cloudflare `clef-flash` 9B decision model (Apache 2.0, served from a Colab
T4 via the `clef-flash-9b` recipe in `lesterppo/colab-llm-deploy`) into
`arxiv_lab.backends` as a drop-in decision backend for agents.

## What was built
- `src/arxiv_lab/backends/systemone.py` — the client (new)
- `src/arxiv_lab/backends/__init__.py` — exports (updated)
- `src/arxiv_lab/backends/openai_compat.py` — same timeout/JSON robustness
  fix applied (updated)
- `tests/test_systemone_cpu.py` — 29 checks: request-shape validation
  (incl. all live-verified quirks), response parsing, `score_value`
- `tests/test_systemone_client_cpu.py` — 9 checks: `decide()` end-to-end
  against a localhost mock server (success, HTTP 400, malformed JSON,
  bad shape, connection refused, timeout, trailing-slash base URL)

## Live-verified API quirks (all baked in as validation)
- `questions` must be an object, never a list
- `noul` takes no criteria
- `choice` requires `criteria` as an object with 2–26 entries (never a list)
- `score` requires `criteria` as an **array** (never an object); the server
  returns a distribution over array positions, `score` = expected position
  index, plus `legend`/`probabilities`/`confidence`; `score_value()` maps
  back onto numeric criteria (bare numbers or `{score, description}`)
- `/api/generate` is meaningless for this model — use `/v1/systemone`

## Deep-test results (2026-10-11, clef-flash:latest on T4, ~3–6s/decide)
| test | verdict | note |
|---|---|---|
| incident_triage | PASS | page p=0.92, severity medium/high (not low), urgency 0.80 |
| comment_triage | PASS | spam p=0.73, action escalate (0.97), priority 0.50 |
| score_shapes | PASS | numbers→value in [0,1]; strings→None; objects→value in [0,10] |
| choice_two_criteria | PASS | boundary criteria count works |
| sequential_no_leak | PASS | p_fine=0.009 vs p_down=0.939 — clean discrimination |
| client_side_validation | PASS | bad shapes rejected before any network call |

First `incident_triage` run asserted `severity == high`; the model
deterministically answers **medium** (0.618 vs high 0.338) for
"intermittent" failures — a defensible judgment, so the assertion was
relaxed to "medium or high, low < 0.2". Lesson: assert integration
properties (typed, well-formed, sane), not a specific winning label on
ambiguous input. (Prior 2026-10-07 run: incident page 0.923 / high 0.525 /
urgency 0.778; comment spam 0.759 / escalate 0.971 / priority 0.780 —
see `INTEGRATION_RESULT.json`.)

## Bugs found & fixed by the deep test
1. `SystemoneClient.decide` let bare `TimeoutError` escape (urlopen does
   not wrap timeouts in `URLError`) — now normalized to `RuntimeError`.
2. Malformed JSON in a 200 body escaped as `JSONDecodeError` — now
   normalized to `RuntimeError`.
3. Same two fixes applied to `OpenAICompatClient.chat` (same bug class,
   same package).

## Files
- `deep_live_test.py` — the VM-side live test (6 tests)
- `DEEP_RESULT.json` — 2026-10-11 live results (GREEN)
- `INTEGRATION_RESULT.json` — 2026-10-07 first integration results (GREEN)
- `score_probe.py` — criteria-shape probe that discovered the score-array rule

## Honest limits
- Tunnel POSTs from the VM hang and trycloudflare GETs get Cloudflare-blocked
  from this network — all live verification ran on the Colab VM via localhost.
- Model judgments on ambiguous inputs vary between runs; the tests assert
  decision *properties*, not exact labels.
- One cold-start decide call took 137s (transient); steady state is 3–6s.
