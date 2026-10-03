# Model-backed validation

The CPU test suite (`tests/test_*_cpu.py`) validates each mechanism against
scripted failure modes. This note records what happened when the same
mechanisms met **real models** — `openai/gpt-oss-20b` via NVIDIA NIM, 2026-10-03.
(Setup: `backends.OpenAICompatClient.for_preset("nvidia")`; key via
`NVIDIA_API_KEY`. Nothing here ran inside the library — these were external
harness scripts.)

## harness — Mingbird mechanisms vs gpt-oss-20b (toy file tasks)

Same two tasks, harness OFF (bloated 1837 B tool prefill, no gate, no loop
detection) vs ON (439 B budgeted prefill + finish gate + loop detector).

| task | harness | score | steps | prefill | notes |
|------|---------|-------|-------|---------|-------|
| sum  | OFF | 1.00 | 12 | 1837 B | succeeded but wandered 12 steps, never cleanly finished |
| sum  | ON  | 1.00 | 2  | 439 B  | optimal: read → write → finish |
| gate (2 criteria) | OFF | 0.00 | 10 | 1837 B | failed the harder task |
| gate (2 criteria) | ON  | 0.00 | 2  | 439 B  | loop detector broke a 3x-repeated read; saved ~8 wasted steps |

The paper's core claim reproduced with a real 20 B open model: the bloated
harness made the model indecisive (12 steps, no finish); the budgeted harness
made it optimal (2 steps). Prefill cost fell 4.2x with no success loss.
The finish gate did not need to fire here (no premature finish claims) —
it is validated at CPU level. n=1 per condition; treat as directional.

## council — BDA vs real LLM coalitions: boundary found

5-member council (3 honest + 2 adversarial gpt-oss-20b), 4 factual questions,
propose + challenge rounds feeding `bda_fit`:

| run | adversary style | BDA | majority vote |
|-----|----------------|-----|---------------|
| 1 | naive (prompt bug: challenge instruction conflicted) | 1/4 | 4/4 |
| 2 | coherent coalition (push D, challenge freely) | 1/4 | 4/4 |
| 3 | paper-style coalition (push D, challenge D's biggest threat) | 2/4 | 4/4 |

**Finding:** under coherent adversarial coalitions, EM repeatedly converged
to the *flipped* fixed point — adversaries scored rho≈0.98 (reliable),
honest members rho≈0.02 — and that wrong fixed point has strictly *higher*
likelihood than the truth. The typed-move annotator model has a
label-switching symmetry: "3 reliable truth-tellers + 2 D-pushers" vs "2
reliable D-tellers + 3 anti-D contrarians" are likelihood-indistinguishable
without a prior. The paper's synthetic adversaries challenged the truth
noisily (incoherent by construction), which is why its evaluation did not
hit this. **BDA is robust to noisy adversaries, not to coherent strategic
ones.** Remedies (not yet implemented): reliability priors (Beta favoring
rho>0.5), multiple EM restarts, or anchoring on a trusted member.

## text — h_k on real generations

6 real gpt-oss-20b paragraphs (temp 0.9): mean h_k **5.49** (range
5.24–5.75), mean phrase repetition 0.005. Repetitive control ("The bus is
red." ×36): h_k **0.168**. 32x separation — the estimator is well-behaved
on real LLM output.
