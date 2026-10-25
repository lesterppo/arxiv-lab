"""CPU check for the AOPD driver math (no model, no GPU, stdlib only).

Verifies, on synthetic values mirroring colab_aopd.py:
  - advantage = reward - per-prompt mean reward (k=2 -> {-0.5, 0, +0.5})
  - advantage factor = clip(1 + advantage, 0, 2) -> {0.5, 1.0, 1.5}
  - importance weight = exp(teacher_seq_logp - student_seq_logp),
    clipped to [0.1, 10]
  - AOPD per-sequence weight = adv_factor * iw; OPD weight = 1
  - answer parsing: first integer in the continuation
Run: python3 test_aopd_logic_cpu.py  (exit 0 on pass)
"""
import math
import re

IW_LO, IW_HI = 0.1, 10.0
_INT = re.compile(r"-?\d+")


def parse_answer(text):
    # LAST integer: a base model may restate the problem before answering.
    ms = _INT.findall(text or "")
    return int(ms[-1]) if ms else None


def advantage(rewards):
    base = sum(rewards) / len(rewards)
    return [r - base for r in rewards]


def adv_factor(a):
    return min(max(1.0 + a, 0.0), 2.0)


def iw(t_logp, s_logp):
    return min(max(math.exp(t_logp - s_logp), IW_LO), IW_HI)


def main():
    # k=2 contrast cases
    assert advantage([1.0, 0.0]) == [0.5, -0.5]
    assert advantage([1.0, 1.0]) == [0.0, 0.0]
    assert advantage([0.0, 0.0]) == [0.0, 0.0]
    assert [adv_factor(a) for a in advantage([1.0, 0.0])] == [1.5, 0.5]
    assert adv_factor(0.0) == 1.0
    # importance weight: identical models -> 1.0; extreme gaps clip
    assert abs(iw(-3.0, -3.0) - 1.0) < 1e-9
    assert iw(0.0, -100.0) == 10.0
    assert iw(-100.0, 0.0) == 0.1
    # AOPD weight ordering: reinforced correct > neutral > downweighted wrong
    w_good = adv_factor(0.5) * iw(-2.0, -2.0)   # 1.5
    w_tie = adv_factor(0.0) * iw(-2.0, -2.0)    # 1.0
    w_bad = adv_factor(-0.5) * iw(-2.0, -2.0)   # 0.5
    assert w_good > w_tie > w_bad == 0.5
    # parsing (last integer wins: restated problem -> final answer)
    assert parse_answer("75\n") == 75
    assert parse_answer("The answer is 42.") == 42
    assert parse_answer("no digits") is None
    assert parse_answer("12 + 7 = 19") == 19
    assert parse_answer("47 + 28 = 75") == 75
    print("AOPD logic CPU check: ALL PASS "
          "(advantage, adv_factor, iw clip, weight ordering, parsing)")


if __name__ == "__main__":
    main()
