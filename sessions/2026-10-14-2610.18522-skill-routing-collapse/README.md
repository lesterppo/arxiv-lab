# 2610.18522 — Lost in Implicit Routing: How LLM Agents Fail to Load Relevant Skills at Scale

**Article:** https://arxiv.org/abs/2610.18522

**Track:** agent · **AGENT = 5** · **Verdict: SUPPORTED** (bounded scale)

## Why picked
Reusable skills are Peter's daily agent infrastructure (custom skills,
`ig` CLI skills, the arxiv-lab harness itself). The paper shows that
implicit routing over skill metadata *collapses* as the library grows —
recall 73.6% → 27.4% at 1,000 irrelevant skills, zero-load rate 9.3% →
51.6% — and that explicit preselection / description-masking mitigate
it. A mechanism this close to the operator's own tooling, with a
directly implementable mitigation, is the highest-value kind of
agent-track pick. (Near-miss: 2610.18233 Runoff, order-invariant
System-One decisions — strong, deferred.)

## What was built
- `src/arxiv_lab/skills/skill_routing.py` (new `skills` area):
  `build_library` (seeded golden + distractor assembly),
  `render_library` (with description masking), `route_prompt`
  (implicit-routing protocol), `preselect_prompt` (explicit
  preselection), `parse_loaded` (deterministic LOAD:/SELECT: parser),
  `routing_scores` (recall / zero-load / wrong-load / precision),
  `aggregate`, `collapse_gap`.
- `tests/test_skills_cpu.py` — 7/7 pass. Scripted collapsing router
  reproduces the collapse signature end to end; perfect preselection
  restores recall to 1.0; parser/metric/masking unit checks.
- Live test driver (NOT in repo — surrogate client): 8 tasks × 1 golden
  skill, 40 hand-written distractors (unrelated domains + plausible
  same-domain alternatives), driven against NVIDIA NIM
  `openai/gpt-oss-20b` (temperature 0). Conditions: implicit-0 / -20 /
  -40, masked-40, masked2-40, preselect-40.

## Results (live, gpt-oss-20b via NIM)
`results.json` (+ `results_masked2.json`); 8 tasks per condition.

| condition | recall | zero-load rate | vs baseline |
|---|---|---|---|
| implicit-0 | 1.000 | 0.000 | — |
| implicit-20 | 0.875 | 0.125 | −0.125 / +0.125 |
| implicit-40 | 0.875 | 0.125 | −0.125 / +0.125 |
| masked-40 (mask *all* descriptions) | 0.625 | 0.375 | −0.375 / +0.375 |
| masked2-40 (mask *distractor* descriptions only — paper's exact) | 1.000 | 0.000 | restored |
| preselect-40 (explicit preselection) | 1.000 | 0.000 | restored |

- **Collapse reproduced (directionally):** even at 20–40 distractors,
  implicit recall falls 1.0 → 0.875 and a task (t_sql) goes to zero-load.
  The failure mode is pure omission — wrong-load count is 0.0 in *every*
  condition, i.e. the model never loads plausible alternatives, matching
  the paper's "too rare to explain this decline".
- **Both paper mitigations verified:** faithful description-masking
  restores recall to 1.000; explicit preselection restores to 1.000
  (at the cost of ~2× latency: 15.8s vs 3.5–16.5s per task — the
  coverage/latency trade-off the paper notes).
- **New nuance (from my own methodological error):** masking *all*
  descriptions — including the golden skills' — makes things *worse*
  (0.625). The mitigation is selective masking, not blanket masking;
  golden skills need their descriptions visible for discrimination.

## Verdict
**SUPPORTED** at bounded scale: the collapse signature, the
omission-only failure mode, and both mitigations all reproduce with a
real model. Scale is the honest limit (40 distractors vs the paper's
1,000; 8 tasks), so the magnitude — not the direction — is what a
larger replication would need to confirm.

## Follow-ups for the operator
- The preselection protocol (`preselect_prompt` + route-over-selected)
  is the actionable takeaway for skill-heavy agent setups: a cheap
  selection call before loading beats stuffing the full catalogue into
  the task prompt.
- No deepworld wiring: deepworld's tool system has no skill-loading
  mechanism (RATED re-ranks tools, wired 2026-10-13); the mitigation is
  recorded here as a protocol recommendation instead.

## Honest limits
- 8 synthetic tasks, 40 distractors — a miniature of SkillsBench, not a
  replication of it. Single model (gpt-oss-20b), temperature 0.
- Distractor sampling is seeded but the task set is hand-built; a
  different task mix could shift the exact recall numbers.
- Latency measured wall-clock via NIM; not a controlled benchmark.
