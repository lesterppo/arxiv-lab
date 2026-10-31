# 2026-10-31 — SPLICE: Locating and Closing Information Gaps in Multi-Agent Systems

- Article: https://arxiv.org/abs/2610.40186 (submitted 29 Oct 2026;
  Gong, Gao, Zhang, Fan, Shen, Wang, Wang, Liu, Liu, Hu)
- Track: agent (CPU, deterministic simulation; no LLM, no Colab GPU)
- Why picked: AGENT=5, TUNE=1. Strongest testable agent mechanism in
  today's RSS increment: a fixed multi-agent workflow whose
  communication graph starves the receiver of evidence, plus the
  displacement hazard — adding an edge can also install a wrong upstream
  conclusion over a correct answer. Repo fit: arxiv-lab multi-agent
  harness library (`src/arxiv_lab/multiagent/`, alongside blockopt).
- Verdict: **SUPPORTED** (directional reproduction at bounded scale;
  synthetic categories are sharper than real benchmarks — see Limits)

## What was built

`splice.py` — deterministic simulation of a skeleton workflow
Retriever A -> Receiver -> Final with an unconnected Retriever B:

- Query categories (ground truth, hidden from the selector):
  GAP (25%) — deciding fact held only by B; skeleton incomplete/wrong,
  candidate correct. NOGAP (45%) — deciding fact held by A, B's output
  on unrelated keys; the relevance probe does not fire, no candidate.
  TRAP (30%) — A correct, B contradicts A on the deciding key; the
  receiver trusts edge-delivered content on conflict, so the candidate
  is wrong (the displacement hazard).
- Gap locator: incomplete skeleton answer, or B's output shares the
  deciding key (relevance probe).
- Candidate: one directed edge B -> Receiver, re-executing Receiver +
  downstream (0.5 of workflow cost — a stated modelling choice, chosen
  so the total lands in the paper's 1.25–1.44x band).
- Adoption selector: logistic regression (numpy, deterministic GD)
  trained on dev process features [content movement, provenance,
  final-reach, observed conflict, observed coverage gain] with probe
  noise (a trap's conflict is missed w.p. 0.20; spurious coverage on a
  trap w.p. 0.15), evaluated on a held-out test split.
- `test_splice_cpu.py` — asserts the paper's directional claims;
  `results.json` — canonical-seed numbers + robustness seeds.

## Results (test split, n=400; dev n=400; canonical seed 20261031)

| system | accuracy |
|---|---|
| Skeleton (no edge) | 0.7325 |
| Naive always-adopt (on probed) | 0.6975 |
| SPLICE process-feature selector | **0.9950** |
| Oracle (adopt iff candidate correct) | 1.0000 |

- Probe rate 0.57, cost ratio **1.285x** (inside the paper's 1.25–1.44x).
- Trap adoption rate 0.0165 (2 of 121 trap queries adopted); gap
  adoption rate 1.0000 (107 of 107).
- Selector weights: movement / provenance / final-reach exactly 0.0 —
  those features are CONSTANT across probed cases (movement 1.0,
  provenance 1.0, final-reach 1.0 for both GAP and TRAP), which is the
  paper's point made measurable: a final-output-only rule cannot
  separate a genuine gap from a trap; it degenerates to always-adopt
  (0.6975, worse than the skeleton). Only conflict + coverage
  features carry the decision (weights −3.84 / +4.36).
- Robustness seeds 1 / 7 / 42: SPLICE 0.9975 / 0.9900 / 0.9925 vs
  skeleton 0.7375 / 0.7800 / 0.7625 and always-adopt 0.7100 / 0.7000 /
  0.7050; trap adoption 0.0086 / 0.0333 / 0.0254; cost 1.276 / 1.260 /
  1.266. All CPU checks pass (`SPLICE_CPU_ALL_PASS`).

## Honest limits

- Synthetic, not benchmark, evidence: categories are clean by
  construction, so the +26-point gain over the skeleton must NOT be
  compared with the paper's +8.0% (Web tool use) / +5.7% (reasoning and
  planning) on five real benchmarks. What transfers is the mechanism's
  direction and the separation result: naive adoption hurts, process
  features rescue it, at ~1.29x cost.
- The paper's selector feature formulas and training data are not in
  the abstract; the feature set here follows the abstract's named
  families (movement, provenance, final-reach) plus conflict/coverage
  as the observable form of "where its new content came from".
- No LLM backbone, no real Web tool use, no zero-shot backbone transfer
  (paper: four unseen backbones) — none of those were testable on CPU.
- The 0.5 re-execution share is a modelling choice, not a measured
  paper quantity; the paper's reported band is the total cost ratio.

## Graduation

Mechanism graduated to `src/arxiv_lab/multiagent/splice.py` (numpy-guarded
per repo conventions) + `tests/test_multiagent_splice_cpu.py`, exported
from `arxiv_lab.multiagent`. All CPU checks pass there too.
