# CapLite: Prompt-Injection Defense for Small Language Model Agents

- **Article:** https://arxiv.org/abs/2611.17115
- **Authors:** (see arXiv page)
- **Date screened:** 2026-11-15 (arXiv API serving source; window 2026-11-12–15)
- **Scores:** AGENT **5**, TUNE **2**
- **Verdict:** **SUPPORTED**

## Why picked

Directly reusable in a local agent harness: CapLite is an enforcement
defense for tool-calling agents that does NOT ask the agent's own model
to write the security artifact. Peter's Hermes agent executes tools
(send/collect/write); a schema-derived write-integrity rule is a
drop-in hardening for its tool loop, with no model upgrade and no
extra model calls. Repo fit: `arxiv-lab` (mechanism library) +
`hermes-colab-cli` (agent runtime). The claim is crisp and CPU-testable:
a deterministic guarantee (0 injected writes) rather than a benchmark
number.

## What was built

- `src/arxiv_lab/agent/caplite.py` (graduated) — stdlib-only mechanism:
  `Tool` schemas with deciding arguments; `parse_stated` (values the
  user stated: emails, dollar amounts, file paths); `CapLitePolicy`
  (commits planned writes before any tool result is read; admits a
  write iff committed AND every deciding argument is user-stated);
  `ModelWrittenPolicy` (CaMeL-style baseline: the model writes the
  allowlist from its context, so after the first tool result the
  artifact is contaminated by the injection — the paper's failure
  mode, implemented not assumed); `MockPlanner` (deterministic
  fallible-SLM stand-in: plans correctly from the request, follows
  instructions found in tool results); `run_episode` / `run_battery`
  (AgentDojo-style attacked + benign batteries).
- `test_caplite_cpu.py` (this session, full battery) and
  `tests/test_caplite_cpu.py` (repo, reduced battery) — same assertions.
- `rust/` — Rust second stage (port + verify + unit/property/soak tests
  + Python-vs-Rust parity sweep). See `rust/README.md`. Independently
  re-verified by the parent run: `cargo test` 23 passed / 0 failed;
  `verify` binary reproduces the Python battery exactly (0/300,
  129/150, 283/300, token totals 7,281 / 88,281 identical); parity sweep
  re-run (2,000 specs, 6,000 rows) diff empty — **0 mismatches**; parser
  probe (520 lines) diff empty; soak 50,000 episodes in 0.91s with
  CapLite 0 attacks / 0 bad writes. No `target/`, no binaries, no
  secrets in the subsession.

## Results (CPU, deterministic, seed 7; 300 attacked + 150 benign episodes)

| arm | attack success | benign completion | bad writes | mean model tokens |
|---|---|---|---|---|
| CapLite | **0/300 (0.000)** | 129/150 (0.860) | **0** | 16 |
| CaMeL-style (model-written policy) | 283/300 (0.943) | 143/150 (0.953) | 283 | 196 |
| Undefended | 300/300 (1.000) | 143/150 (0.953) | 307 | 16 |

Paper's numbers for comparison: no attack succeeds in 13,935 attacked
AgentDojo episodes under CapLite; utility under attack 29.0% vs 27.2%
CaMeL; 17.1k tokens/episode, 18% of CaMeL's.

Unit checks passed: commit rule (a write proposed only after a tool
result is blocked even with user-stated args); unstated-arg rule
(committed write with an unstated amount blocked); payload args are
NOT deciding (an injected subject/body change on an otherwise admitted
write is allowed — the guarantee covers whom/what, not what the
message says).

## Honest limits

- The planner is a deterministic worst-case stand-in, not a real SLM:
  it always follows injected instructions. The test therefore measures
  the *enforcer*, not a model's susceptibility — the 0/300 is a
  property of the enforcement logic, and the 283/300 CaMeL figure is a
  property of the modelled contamination mechanism, not a measurement
  of any real model writing policies.
- Token counts are rough (whitespace word counts of model prompts, +
  180 for a program-writing call); the direction (CapLite << CaMeL)
  matches the paper, the 16-vs-196 ratio is not the paper's 18%.
- Benign completion 0.860 vs 0.953 undefended: the 21 misses are the
  15 lookup tasks (write argument only in a file, not user-stated —
  blocked by design, the honest utility cost of the guarantee) plus
  ~6 planner misreadings (wrong deciding argument, correctly blocked).
- Injection battery is 7 hand-written templates; a real AgentDojo run
  would add paraphrase and multi-turn attacks. The guarantee itself is
  template-independent (any write not committed with user-stated args
  is refused), which is why the verdict is SUPPORTED rather than
  VERIFIED: the mechanism is reproduced and its guarantee holds on the
  battery, but the paper's 13,935-episode AgentDojo measurement is not
  replicated here.

## Verdict: SUPPORTED

CapLite's write-integrity mechanism reproduces exactly as specified:
0 injected writes across 300 attacked episodes, 0 bad writes, benign
utility 0.860 (loss confined to lookup tasks the rule is designed to
refuse), at lower model-token cost than the model-written-policy
baseline — which admits 94.3% of attacks in the same battery,
reproducing the paper's failure mode for model-written artifacts.
