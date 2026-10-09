# Session 2026-10-09 — Curating Always-Loaded Context for LLM Agents (arXiv:2610.11007)

**Article:** "Curating Always-Loaded Context for LLM Agents: A Capacitated
Assortment Model with Censored Feedback"
([abs](https://arxiv.org/abs/2610.11007))

**Why picked:** most prospective *agent* article in the 2026-10-09 cs.AI
window (AGENT 5/5). The paper formalizes exactly the file class this agent
lives with every session — always-loaded context files such as AGENTS.md,
MEMORY.md, SOUL.md: finite attention capacity, per-token per-round cost,
and feedback that is censored (loaded instructions' effects are observable;
missing instructions generate feedback only when their absence causes
harm). Its claims are unusually crisp and CPU-testable: (A) an upper bound
on the optimal file size independent of candidate count; (B) appending
every positive-standalone instruction can be *arbitrarily worse* than the
optimal subset; (C) deleting instructions the agent "ignores" inevitably
removes helpful silent ones; (D) a characterization of how much evidence
to collect before adding an instruction. Direct repo fit: `arxiv-lab` as
the mechanism library (new `context` area), and the takeaway applies
one-to-one to how Peter's agent fleet should curate its own loaded files.

## What was built

- Reusable mechanism: `src/arxiv_lab/context/curated_context.py` (new
  `context` area) — capacitated assortment model faithful to the paper's
  axioms: instruction i has token cost c_i and standalone value v_i;
  attention dilution D(T) = 1/(1+T/kappa) makes every loaded instruction's
  realized value v_i·D(T_S) strictly decrease when anything is added
  ("adding an instruction never raises the compliance of the others");
  per-session setup cost lambda per loaded token. Net value
  V(S) = D(T_S)·sum(v) − lambda·T_S. Exact optimal subset via
  knapsack DP over total tokens; greedy-append baseline (all
  positive-standalone instructions); closed-form worst-case size bound;
  censored-feedback simulator (visible follows vs silent harm-prevention);
  delete-if-ignored policy; explore-then-commit evidence rule with optimal
  trial count m*.
- CPU test: `tests/test_context_cpu.py` — 9 checks, all passing.
- This session: `experiment_curated_context.py` (verbatim snapshot,
  runnable from the repo root) and `results_curated_context.json`.

## Results (deterministic, seeded)

**Claim B — greedy append is arbitrarily worse.** Star-plus-fillers family
(one great instruction + m small-positive-standalone, large-cost fillers):

| m | greedy V | optimal V | gap |
|---|----------|-----------|-----|
| 20 | −55.1 | 97.8 | 153.0 |
| 100 | −382.4 | 97.8 | 480.2 |
| 200 | −783.4 | 97.8 | 881.3 |

The gap grows linearly in m (greedy → −∞) while the optimal file keeps
just the star (index [0]) at every m — "arbitrarily worse" reproduced
exactly.

**Claim A — bounded optimal file size.** Optimal loaded tokens for random
pools: n=25 → 310, n=100 → 734, n=400 → 795. The pool grows 16x; the
optimal file saturates (795 ≤ 1.5×734, ≤ 3×310). Additionally the
n-independent structural property holds on every pool: the optimal file's
value-density sum(v)/T ≥ lambda (follows from V(S*) ≥ V(∅) = 0).

**Claim C — censored feedback.** Delete-if-ignored over K sessions: the
visibly-followed instruction (p=0.3) survives; the purely preventive
silent guardrail (p_follow=0, q_prevent=0.25, harm value 50) is deleted
with probability 1 at K=20 *and* K=200 — no finite evidence can save an
instruction whose value never generates an observable event. The deletion
destroys 12.0 points of per-session value (the unprevented harms).

**Claim D — evidence rule.** Optimal trial count before adding:
m*(sigma0=1)=0, m*(sigma0=10)=3 (more uncertainty → more evidence);
m*(cheap trials)=7 > m*(pricey trials)=1. m* beats fixed m=0/1/50 rules
on expected regret.

**Claim E — budget caps mispricing loss.** Curator underestimates the
token price (0.01 vs true 0.15): the mispriced optimizer overloads
(T=430 vs true-optimal 271) for a realized loss of 6.2; hard-capping it
at the true-optimal budget reduces the loss to 0.0.

## Verdict

**Replicates** claims B, C, D quantitatively and the saturation content
of claim A inside the paper's axiomatic model. The mechanism's practical
takeaway for this agent's own stack: always-loaded files should be
curated by optimal-subset selection under a token budget, never by
appending everything useful-looking, and "delete what the agent ignores"
is an unsafe policy for guardrail-style sections.

## Honest limits

> **Simulation honesty box.** The dilution shape D(T)=1/(1+T/kappa), the
> lognormal value / uniform cost distributions, and the censoring
> probabilities are modeling choices implementing the paper's stated
> axioms — not measurements of any LLM. The module tests the paper's
> *mechanism-level* claims (bound/saturation, arbitrarily-worse greedy,
> inevitable deletion under censoring, evidence scaling) inside that
> model. Real instruction values, attention dilution curves, and
> feedback observability are unmeasured here.
>
> The paper's closed-form size bound is not reproduced tightly:
> `size_upper_bound()` is implemented and valid but vacuous at these
> parameters (worst-case bound ≈ 30000 vs observed optima < 800),
> because a pure parameter-only bound must cover the adversarial
> all-max-value pool. The testable content — saturation of T* and the
> n-independent density floor — is what the test asserts.

- The DP optimum is exact only for integer token costs; fractional costs
  are truncated.
- The evidence rule assumes Gaussian prior/observations and a known
  per-trial cost; real curation trials have unknown, heavy-tailed costs.
- No deepworld wiring: the mechanism is a curation policy, not a
  plug-in component; the takeaway for deepworld/harness work is the
  policy itself (budget + optimal subset + never delete silent
  guardrails on zero-observation evidence).
