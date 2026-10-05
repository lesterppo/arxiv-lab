"""arxiv-lab: portable CPU implementations of validated arXiv-paper findings.

Each subpackage implements one mechanism from a recent arXiv paper, reproduced
as a minimal, dependency-free (stdlib; numpy optional) CPU implementation by
the arxiv-gem-digest daily read-and-test experiments:

- ``arxiv_lab.harness``  — Mingbird agent-harness mechanisms (arXiv:2610.02001)
- ``arxiv_lab.council``  — Bayesian Dialectical Argumentation council
  weighting (arXiv:2610.02005)
- ``arxiv_lab.memory``   — Causal Memory Policy retrieval-intervention
  scoring (arXiv:2610.02070)
- ``arxiv_lab.text``     — Kontoyiannis entropy-rate estimator h_k
  (arXiv:2610.01493)
- ``arxiv_lab.sampling`` — FERPO forward-KL mode-covering sampler
  (arXiv:2610.02198)
- ``arxiv_lab.backends`` — OpenAI-compatible chat client over stdlib urllib
  (no credentials in code; keys only via env vars or explicit args)
- ``arxiv_lab.routing`` — component routing for self-improving agents:
  recurrence/state-conditionality rule routing experience components
  (locators, procedures, state facts, lessons) to weights vs context
  (arXiv:2610.01787)
- ``arxiv_lab.workflow`` — FloWright-style hierarchical structure-aware
  credit assignment and multi-role co-evolution for workflows with a single
  sparse outcome (arXiv:2610.01026)
- ``arxiv_lab.sentry`` — Sentry failure-management layer: conditional
  failure-lesson retrieval, reward-free recovery verification, and a
  store-only-if-verified lesson gate (arXiv:2610.02994)
- ``arxiv_lab.training`` — Prospective Hindsight: surprise-weighted
  advantages for self-calibrating RL via prediction-reality gaps
  (arXiv:2610.02740)

Runs on general Linux with Python >= 3.10. No install needed for tests:
``tests/`` inserts ``src/`` on ``sys.path``.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
