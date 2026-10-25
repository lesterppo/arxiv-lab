"""CPU tests for arxiv_lab.training.jevdlc (arXiv:2610.31521).

Model-free: a synthetic frozen scorer that always prefers identity 0,
while the user always prefers identity 5. Checks, deterministically:

  C1. Jev-DLC state learns the user's preference from typed feedback
      and beats the frozen scorer over a 24-event stream.
  C2. The gain is causal: predictions are committed before feedback, so
      event 0 (no prior feedback) cannot benefit from state.
  C3. State stays bounded and history-free: only counts, pair counts and
      two weights are stored; reset() restores the frozen scorer exactly.
  C4. The responsive pair channel alone transfers a correction to a
      later slate sharing the same pair (directed relation, not just a
      count boost): after one correction preferring 5 over 0, a slate
      containing both ranks 5 above 0 in the responsive channel.

Run: python3 tests/test_training_jevdlc_cpu.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arxiv_lab.training.jevdlc import JevDLCState  # noqa: E402

IDENTITIES = ["a0", "a1", "a2", "a3", "a4", "a5"]


def frozen_scores(slate):
    # scorer prefers a0 strongly, others flat -- it knows nothing of the user
    return [2.0 if s == "a0" else 0.0 for s in slate]


def run_stream():
    st = JevDLCState()
    frozen_ok, jev_ok = 0, 0
    n = 24
    for t in range(n):
        # rotating slate of 4, always containing a0 and a5
        slate = ["a0", "a5", IDENTITIES[1 + (t % 4)], IDENTITIES[2 + (t % 3)]]
        slate = list(dict.fromkeys(slate))[:4]
        base = frozen_scores(slate)
        pred, info = st.predict(base, slate)
        frozen_pred = max(range(len(slate)), key=lambda i: base[i])
        target = "a5"
        if slate[frozen_pred] == target:
            frozen_ok += 1
        if slate[pred] == target:
            jev_ok += 1
        st.update(pred, target, slate, info)
    return frozen_ok, jev_ok, st


def main():
    frozen_ok, jev_ok, st = run_stream()
    print(f"frozen {frozen_ok}/24, jevdlc {jev_ok}/24")
    assert jev_ok > frozen_ok, "C1: Jev-DLC must beat the frozen scorer"
    assert jev_ok >= 18, "C1: Jev-DLC must learn the preference (expect >=18/24)"

    # C2: no benefit before any feedback
    st0 = JevDLCState()
    slate = ["a0", "a5", "a1", "a2"]
    pred, info = st0.predict(frozen_scores(slate), slate)
    assert slate[pred] == "a0", "C2: with no feedback, prediction == frozen"

    # C3: bounded, history-free state; reset restores frozen
    assert st.state_size() <= 2 + 6 + 36, "C3: state must stay bounded"
    assert not hasattr(st, "history"), "C3: no raw history stored"
    st.reset()
    pred, _ = st.predict(frozen_scores(slate), slate)
    assert slate[pred] == "a0", "C3: reset must restore the frozen scorer"

    # C4: directed pair transfer, responsive channel only. Weak base gap
    # (0.5) so a single correction can flip the pair; with a 2.0 gap one
    # correction correctly only narrows it, which C1 already covers.
    def weak_scores(slate):
        return [0.5 if s == "a0" else 0.0 for s in slate]

    st1 = JevDLCState()
    slate1 = ["a0", "a5", "a1", "a2"]
    pred, info = st1.predict(weak_scores(slate1), slate1)
    st1.update(pred, "a5", slate1, info)
    slate2 = ["a0", "a5", "a3", "a4"]
    _, info2 = st1.predict(weak_scores(slate2), slate2)
    s_r = info2["s_r"]
    assert s_r[slate2.index("a5")] > s_r[slate2.index("a0")], \
        "C4: responsive channel must rank the corrected pair above the rejected action"
    print("ALL PASS")


if __name__ == "__main__":
    main()
