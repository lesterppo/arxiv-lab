# LAQ — Enhancing Local and Global Structure in Compressed AI Agent Memory

- **Article:** arXiv:2611.08341 — "Enhancing Local and Global Structure in
  Compressed AI Agent Memory"
- **Link:** https://arxiv.org/abs/2611.08341
- **Track:** agent (CPU)
- **Verdict:** **NEGATIVE**

## Why picked

AGENT 5 / TUNE 2. Directly a local agent-harness mechanism: agents store
text as embedding vectors and compress them at scale, and the paper gives
both a measurement framework (local/global fidelity) and a method (LAQ, a
far-hinge quantizer objective) that could drop into the memory layer of a
local harness (cf. `src/arxiv_lab/memory/` — mempp, cmp, rawrag). The
claims are quantitative and CPU-reproducible: no model, no GPU needed.

## What was built

`laq.py` (session-local; **not** graduated to `src/` — verdict NEGATIVE):
synthetic themed memories (content-group near-duplicates + soft Dirichlet
theme mixtures, unit directions), the paper's L (10-NN survival) and G
(30–50% far-set overlap) fidelities, theme-move rate, far violation rate,
random-projection binary codes (768 bits = 96 bytes), PCA + 8-bit
quantizer (96 dims = 96 bytes), MSE-tuned clips, hinge-tuned clips, and
violation-rate-tuned distance weights. `test_memory_laq_cpu.py` runs the
52-run experiment (13 memory-encoder groups x 4 seeds) plus determinism
and incremental-fit consistency checks. `diag_laq.py` / `diag_g8.py` are
the regime-tuning diagnostics (group_size 16 vs 8).

Bounded adaptations (documented, all in `laq.py`'s docstring):
- Real passage embeddings replaced by synthetic themed memories.
- OPQ index replaced by PCA rotation + per-dim uniform 8-bit quantizer
  (same 96 bytes/passage, rotation, and search).
- LAQ's full codebook optimization restricted to diagonal forms:
  per-dim clip ranges (hinge-clip arm) and per-dim distance weights
  (weights-viol arm). A richer parametrization could in principle
  recover the effect — this is the stated limit of the test.
- The hinge starts from the MSE-OPTIMAL clips (the OPQ-like baseline),
  because OPQ already minimizes reconstruction error; the paper's claim
  is that the hinge adds G *beyond* reconstruction.
- G = far-set overlap (30–50% band); r = far violation rate (outer
  50–70% item invading the far set). Near-vs-far flips are ~0 once
  content groups exist, so they cannot govern G.

Two guards found necessary while building (both in the code):
- The hinge margin is FIXED once from the initial state. A margin
  recomputed from the current state admits a scale degeneracy: shrinking
  all distances shrinks the margin with it, so the loss falls while the
  violation rate rises 8x (measured: hinge loss 0.0497 -> 0.0370 while
  r went 0.034 -> 0.267 and G 0.838 -> 0.388).
- The clip hinge is accepted only within 1% of the starting MSE (the
  reconstruction anchor the paper's joint objective provides). The mean
  hinge alone optimizes the TAIL (worst violations) and can trade many
  small new violations for a few large fixes.

## Results (52 runs, 13 groups x 4 seeds)

| claim | paper | this test |
|---|---|---|
| C1: L binary vs PCA | 0.97 vs 0.77 | rescored binary **0.892**, PCA **0.968** (direction FLIPPED) |
| C1: theme-move binary vs PCA | 10.1% vs 1.7% | **15.4% vs 1.5%** (reproduced, incl. magnitude) |
| C2: bound (1-G) <= 2.18*sqrt(r) | constant 2–2.18 | median ratio **1.00**, max **1.73** (HOLDS) |
| C2: corr(noise, r) | 0.979 | pooled **0.623**; within-memory **-0.058** |
| C3: reconstruction raises G | 0/52 | naive->MSE-optimal raises G **52/52**, mean dG **+0.0423** |
| C3: far hinge raises G | 52/52 | hinge-clip beats MSE-optimal **22/52**, mean dG **-0.0001** (tie); weights-viol **1/52**, mean dG **-0.0063** |

## Verdict: NEGATIVE

The paper's core method claim (C3) did not reproduce under a fair bounded
test: from the MSE-optimal quantizer, the far hinge adds nothing
(22/52 wins, mean dG -0.0001 — a tie) while reconstruction itself raises
G in all 52 runs (+0.0423) — the opposite of the paper's ordering
(hinge 52/52, reconstruction 0/52). A second, differently-aimed hinge
parametrization (violation-rate distance weights, which did cut the
violation rate 0.025 -> 0.007) also failed to move G (1/52). Two
independent hinge variants failing the same way makes this a substantive
bounded-scale negative, not a single-knob artifact.

What DID reproduce: the theme-move half of C1 (binary codes move 10x
more passages across themes than PCA codes, at the paper's magnitudes)
and the C2 bound itself (median ratio 1.00, max 1.73 — inside the
paper's 2–2.18 band). The L half of C1 flipped (PCA keeps higher L than
binary here, because 75 content groups fit inside the top-96 PCA
components), and the noise-tracking half of C2 is weaker than claimed
(pooled corr 0.623 vs 0.979; within-memory ~0 — but the design holds
bytes fixed, so it does not span noise levels; that half is
INCONCLUSIVE rather than refuted).

## Honest limits

- Diagonal restriction: LAQ's full OPQ codebook optimization (256
  centers x 96 dims) was not reproduced; the clip/weight knobs are
  first-order forms. A richer parametrization might recover the effect.
- Synthetic memories, not real passage embeddings (FinDER / Multi-Doc-2025).
- 1,200 passages per memory (paper: up to 1e6); 96 bytes/passage matched.
- The hinge's failure mode is documented, not hidden: the mean hinge
  optimizes violation magnitude, while G counts violations — see "Two
  guards" above.

## Rust second stage

None, per the standing rule: the verdict is NEGATIVE, so there is no Rust
leg for this article today.

## Reproduce

```
python3 test_memory_laq_cpu.py   # ~5 min, stdlib + numpy; prints C1/C2/C3
```
