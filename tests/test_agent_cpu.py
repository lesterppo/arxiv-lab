"""CPU tests for arxiv_lab.agent.competence (arXiv:2610.13144).

Paper claims under test:
  1. Stage 1: the task-conditioned competence signal is calibrated for
     the decision-relevant agents (top-ranked per task type).
  2. Stage 2: adaptive orchestration (role activation + communication graph
     driven by the competence signal) beats fixed solo on accuracy
     (paper: 87.22 -> 89.97 on higher-competence backbones, +4.58 pts on
     lower-competence ones) while cutting token cost substantially vs
     fixed full deliberation (paper: -36.26%).

Two scenarios: a "high" competence backbone profile and a "low" one,
each run under three policies (adaptive / full / solo) with fixed seeds.
Seeded and deterministic.

Requires numpy (optional dependency); skipped gracefully without it.

Run:  python3 tests/test_agent_cpu.py
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from arxiv_lab.agent import (
    CompetenceTracker,
    orchestrate,
    select_mode,
    communication_graph,
    simulate_mas,
    make_backbone,
    SOLO,
    PAIR,
    DELIBERATE,
    TASK_TYPES,
)


def _skip():
    print("SKIP: numpy not installed")
    sys.exit(0)


def test_select_mode_thresholds():
    assert select_mode(0.90) == SOLO
    assert select_mode(0.85) == SOLO
    assert select_mode(0.84) == PAIR
    assert select_mode(0.60) == PAIR
    assert select_mode(0.59) == DELIBERATE
    assert select_mode(0.10) == DELIBERATE


def test_communication_graph_shapes():
    assert communication_graph(SOLO, ["a"]) == []
    pair = communication_graph(PAIR, ["a", "b"])
    assert sorted(pair) == [("a", "b"), ("b", "a")]
    delib = communication_graph(DELIBERATE, ["a", "b", "c"])
    assert len(delib) == 6  # ordered pairs, no self-loops
    assert all(x != y for x, y in delib)


def test_cold_start_is_conservative():
    # No observations yet: every estimate is the prior (0.5) -> deliberate.
    tracker = CompetenceTracker()
    agents = ["coder", "reasoner", "researcher"]
    assert all(tracker.estimate(a, "math") == 0.5 for a in agents)
    mode, roles, edges = orchestrate(tracker, agents, "math")
    assert mode == DELIBERATE and len(roles) == 3 and len(edges) == 6


def test_tracker_learns_and_ranks():
    rng = np.random.default_rng(7)
    tracker = CompetenceTracker(alpha=0.3)
    # coder is good at math, bad at qa; researcher the reverse.
    for _ in range(200):
        tracker.observe("coder", "math", rng.random() < 0.9)
        tracker.observe("coder", "qa", rng.random() < 0.3)
        tracker.observe("researcher", "math", rng.random() < 0.3)
        tracker.observe("researcher", "qa", rng.random() < 0.9)
    assert tracker.estimate("coder", "math") > 0.75
    assert tracker.estimate("coder", "qa") < 0.45
    assert tracker.best(["coder", "researcher"], "math") == "coder"
    assert tracker.best(["coder", "researcher"], "qa") == "researcher"


def _run_regime(kind, seed, n=600):
    rng = np.random.default_rng(seed)
    prof = make_backbone(rng, kind)
    res = {}
    for i, policy in enumerate(["adaptive", "full", "solo"]):
        res[policy] = simulate_mas(
            np.random.default_rng(seed + 100 + i), n, prof,
            policy=policy, seed_outcomes=seed + 1000 + i)
    return prof, res


def _top_agent_mae(tracker, prof):
    """Mean |estimate - truth| for the top-estimated agent per task type.

    Decision-relevant calibration: orchestration only ever relies on the
    top-ranked agents' estimates. (Rarely-activated agent x task cells
    stay near the prior -- the standard bandit selection bias, noted as
    a limitation.)
    """
    errs = []
    for t in TASK_TYPES:
        top = tracker.best(list(prof.keys()), t)
        errs.append(abs(tracker.estimate(top, t) - prof[top][t]))
    return sum(errs) / len(errs)


def test_high_competence_backbone():
    prof, res = _run_regime("high", 1234)
    ad, full, solo = res["adaptive"], res["full"], res["solo"]
    print(f"[high] adaptive acc={ad['accuracy']:.3f} tok={ad['tokens']} "
          f"modes={ad['modes']}")
    print(f"[high] full     acc={full['accuracy']:.3f} tok={full['tokens']}")
    print(f"[high] solo     acc={solo['accuracy']:.3f} tok={solo['tokens']}")
    # Claim 2: near-parity with always-deliberate...
    assert ad["accuracy"] >= full["accuracy"] - 0.02, (
        f"adaptive {ad['accuracy']:.3f} < full {full['accuracy']:.3f}")
    # ...while cutting token cost substantially (paper: -36.26%).
    ratio = ad["tokens"] / full["tokens"]
    print(f"[high] token ratio adaptive/full = {ratio:.3f}")
    assert ratio <= 0.75, f"token ratio {ratio:.3f} > 0.75"
    # Claim 2 (paper's 87.22 -> 89.97): adaptive beats fixed solo, because
    # collaboration is added exactly on the hard queries.
    gain = ad["accuracy"] - solo["accuracy"]
    print(f"[high] adaptive-vs-solo gain = {gain:+.3f}")
    assert gain >= 0.01, f"gain {gain:+.3f} < +0.01"
    # Claim 1: the decision-relevant competence estimates are calibrated.
    mae = _top_agent_mae(ad["tracker"], prof)
    print(f"[high] top-agent competence MAE = {mae:.3f}")
    assert mae < 0.10, f"MAE {mae:.3f} >= 0.10"
    # The adaptive policy must actually use the cheap modes.
    assert ad["modes"][SOLO] > 50, ad["modes"]


def test_low_competence_backbone():
    prof, res = _run_regime("low", 999)
    ad, full, solo = res["adaptive"], res["full"], res["solo"]
    print(f"[low] adaptive acc={ad['accuracy']:.3f} tok={ad['tokens']} "
          f"modes={ad['modes']}")
    print(f"[low] full     acc={full['accuracy']:.3f} tok={full['tokens']}")
    print(f"[low] solo     acc={solo['accuracy']:.3f} tok={solo['tokens']}")
    # Non-inferior to fixed full, and better than solo
    # (paper: +4.58 pts over fixed schemes on low-competence backbones).
    assert ad["accuracy"] >= full["accuracy"] - 0.03
    gain = ad["accuracy"] - solo["accuracy"]
    print(f"[low] adaptive-vs-solo gain = {gain:+.3f}")
    assert gain >= 0.005, f"gain {gain:+.3f} < +0.005"
    mae = _top_agent_mae(ad["tracker"], prof)
    print(f"[low] top-agent competence MAE = {mae:.3f}")
    assert mae < 0.10, f"MAE {mae:.3f} >= 0.10"
    # Low competence -> the collaborative modes must dominate.
    assert ad["modes"][DELIBERATE] + ad["modes"][PAIR] > ad["modes"][SOLO], ad["modes"]


if __name__ == "__main__":
    if not HAS_NUMPY:
        _skip()
    test_select_mode_thresholds()
    test_communication_graph_shapes()
    test_cold_start_is_conservative()
    test_tracker_learns_and_ranks()
    test_high_competence_backbone()
    test_low_competence_backbone()
    print("ALL COMPETENCE-ORCHESTRATION CPU TESTS PASSED")
