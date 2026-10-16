"""CPU tests for arxiv_lab.retrieval.tracer (TRACER, arXiv:2610.21267).

Claim under test (bounded, simulated scale): a trajectory-aware retrieval
front-end — a trie over observed tool-call sequences blended with a
contextual semantic prior — improves agent tool-selection accuracy over a
full-registry contextual selector while exposing only k << N tools per
step (the paper: +4.7-10 pts, 5 of 128 tools exposed, ~80% fewer tokens).

Environment: 40 tools (30 regular + 10 near-duplicate "alias" tools),
2 task families with distinct header tools, 8 subroutine templates. One slot
per template is a *variant slot*: family R tasks use the regular tool,
family T tasks use its alias. Alias embeddings are near-identical to their
regular counterpart, and the step context at a variant slot is a 50/50
mixture of the two variants -- genuinely ambiguous for a contextual
selector, but fully disambiguated by the trajectory prefix (the family
header), which is exactly the complementary-priors regime TRACER targets.
Other steps use informative-but-noisy contexts (noise calibrated so the
full-registry selector is imperfect but reasonable, ~0.75-0.85).

Data splits:

* 100 trajectories -> build the trie
*  40 trajectories -> learn the blend weight (held out from the trie)
*  80 trajectories -> final evaluation (held out from both)

Seeded and deterministic. Requires numpy (optional dependency); skipped
gracefully without it.

Run:  python3 tests/test_retrieval_tracer_cpu.py
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy not installed")
    sys.exit(0)

from arxiv_lab.retrieval.tracer import (
    TrajectoryTrie,
    learn_blend_weight,
    tracer_top1_accuracy,
)

SEED = 20261016
N_REGULAR = 30
N_ALIAS = 10
N_TOOLS = N_REGULAR + N_ALIAS
DIM = 16
N_SUBROUTINES = 8
K = 5
NOISE = 0.3
ALIAS_NOISE = 0.03  # alias embedding ~ near-duplicate of its regular twin
HEADER_R, HEADER_T = N_REGULAR - 2, N_REGULAR - 1  # family header tools


def _unit(rng, n, d):
    v = rng.normal(size=(n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def build_env(seed=SEED):
    rng = np.random.default_rng(seed)
    tool_vecs = _unit(rng, N_TOOLS, DIM)
    # Aliases: tool 30+i is a near-duplicate of regular tool i (i < 10).
    for i in range(N_ALIAS):
        twin = tool_vecs[i] + ALIAS_NOISE * rng.normal(size=DIM)
        tool_vecs[N_REGULAR + i] = twin / np.linalg.norm(twin)
    # 8 subroutine templates over tools 0..27 (headers 28, 29 excluded);
    # each template marks one slot as the family-variant slot.
    subroutines = []
    for _ in range(N_SUBROUTINES):
        length = int(rng.integers(2, 4))
        tools = list(rng.choice(N_REGULAR - 2, size=length, replace=False))
        variant_slot = int(rng.integers(length))
        # variant slot must resolve to a tool that has an alias twin
        tools[variant_slot] = int(rng.integers(N_ALIAS))
        subroutines.append((tools, variant_slot))
    return tool_vecs, subroutines


def sample_trajectory(rng, subroutines, shift=False):
    """[family_header] + template with the variant slot family-resolved.

    With shift=True the header and the variant slot come from *different*
    families: the trajectory prior then misleads at the ambiguous step.
    This distribution-shift probe (used only in reporting, not in the main
    assertion) mirrors the paper's own failure-mode analysis.
    """
    family = int(rng.integers(2))  # 0 = regular variant, 1 = alias variant
    template, slot = subroutines[int(rng.integers(len(subroutines)))]
    seq = list(template)
    if family == 1:
        seq[slot] = N_REGULAR + seq[slot]  # use the alias twin
    header_family = 1 - family if shift else family
    header = HEADER_R if header_family == 0 else HEADER_T
    return [header] + seq, slot + 1  # variant position shifted by the header


def steps_from_trajectory(traj_info, tool_vecs, rng):
    """(prefix, context_vec, true_tool); variant slots get ambiguous ctx."""
    traj, variant_pos = traj_info
    steps = []
    for i, true_tool in enumerate(traj):
        prefix = tuple(traj[:i])
        if i == variant_pos:
            # Genuinely ambiguous: 50/50 mixture of the regular tool and
            # its alias twin. Only the trajectory prefix disambiguates.
            twin = (true_tool - N_REGULAR) if true_tool >= N_REGULAR \
                else (true_tool + N_REGULAR)
            context = tool_vecs[true_tool] + tool_vecs[twin]
        else:
            context = 2.0 * tool_vecs[true_tool]
        context = context + NOISE * rng.normal(size=DIM)
        context = context / np.linalg.norm(context)
        steps.append((prefix, context, int(true_tool)))
    return steps


def run_simulation(seed=SEED):
    rng = np.random.default_rng(seed)
    tool_vecs, subroutines = build_env(seed)
    n_build, n_learn, n_test = 100, 40, 80

    trie = TrajectoryTrie()
    for _ in range(n_build):
        trie.insert(sample_trajectory(rng, subroutines)[0])

    learn_steps = []
    for _ in range(n_learn):
        learn_steps.extend(
            steps_from_trajectory(sample_trajectory(rng, subroutines),
                                  tool_vecs, rng))

    w, w_train_acc = learn_blend_weight(learn_steps, tool_vecs, trie)

    test_steps = []
    for _ in range(n_test):
        test_steps.extend(
            steps_from_trajectory(sample_trajectory(rng, subroutines),
                                  tool_vecs, rng))

    # Distribution-shift probe: header/variant family mismatch, reported
    # separately (the trajectory prior misleads here by construction).
    shift_steps = []
    for _ in range(n_test):
        shift_steps.extend(
            steps_from_trajectory(
                sample_trajectory(rng, subroutines, shift=True),
                tool_vecs, rng))

    results = tracer_top1_accuracy(test_steps, tool_vecs, trie, k=K,
                                   w_empirical=w)
    shift = tracer_top1_accuracy(shift_steps, tool_vecs, trie, k=K,
                                 w_empirical=w)
    results["blend_weight"] = w
    results["blend_train_accuracy"] = w_train_acc
    results["n_trajectories"] = (n_build, n_learn, n_test)
    results["shift_tracer_accuracy"] = shift["tracer_accuracy"]
    results["shift_full_accuracy"] = shift["full_accuracy"]
    return results


def test_trie_counts():
    trie = TrajectoryTrie()
    trie.insert([1, 2, 3])
    trie.insert([1, 2, 4])
    trie.insert([1, 2, 3])
    p = trie.continuation_probs((1, 2), 6)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[3] > p[4] > p[0]  # observed 2x vs 1x vs unseen
    # unseen prefix falls back to the marginal (root) distribution
    q = trie.continuation_probs((5, 5), 6)
    assert abs(q.sum() - 1.0) < 1e-9
    assert q[1] > q[2]  # tool 1 starts all three sequences


def test_tracer_improves_accuracy_and_cuts_tokens():
    res = run_simulation()
    print(f"blend w_empirical={res['blend_weight']:.1f} "
          f"(train acc {res['blend_train_accuracy']:.3f})")
    print(f"full-registry accuracy : {res['full_accuracy']:.3f} "
          f"(exposed {res['full_mean_exposed']:.0f} tools/step)")
    print(f"TRACER accuracy        : {res['tracer_accuracy']:.3f} "
          f"(exposed {res['tracer_mean_exposed']:.0f} tools/step)")
    gain = res["tracer_accuracy"] - res["full_accuracy"]
    token_cut = 1.0 - res["tracer_mean_exposed"] / res["full_mean_exposed"]
    print(f"accuracy gain: {gain:+.3f}; token-cost reduction: {token_cut:.1%}")
    print(f"shift probe (mismatched header): TRACER "
          f"{res['shift_tracer_accuracy']:.3f} vs full "
          f"{res['shift_full_accuracy']:.3f}")
    # Paper's directional claim: accuracy up, tokens down.
    assert gain > 0.0, "TRACER should beat the full-registry selector here"
    assert token_cut >= 0.80, "must expose far fewer tools per step"
    assert 0.60 <= res["full_accuracy"] <= 0.95, (
        "sanity: baseline selector must be imperfect but reasonable")


def main():
    test_trie_counts()
    test_tracer_improves_accuracy_and_cuts_tokens()
    print("ALL TRACER CPU TESTS PASSED")


if __name__ == "__main__":
    main()
