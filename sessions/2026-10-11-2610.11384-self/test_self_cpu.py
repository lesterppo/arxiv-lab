"""CPU tests for the SELF session env/teacher/data (arXiv:2610.11384).

Importing colab_self has no side effects (no pip, no torch), so the
env/teacher/hindsight logic is fully testable here.
Run:  python3 test_self_cpu.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colab_self as S  # noqa: E402
import random  # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("apply_action: arithmetic + feedback strings")
def _():
    assert S.apply_action(7, "add 6") == (13, "total=13")
    assert S.apply_action(7, "sub 3") == (4, "total=4")
    assert S.apply_action(7, "mul 3") == (21, "total=21")
    t, fb = S.apply_action(2, "sub 5")
    assert t == 2 and "cannot go below 0" in fb
    t, fb = S.apply_action(7, "div 2")
    assert t == 7 and "illegal action" in fb
    t, fb = S.apply_action(7, "add")
    assert t == 7 and "malformed" in fb
    t, fb = S.apply_action(7, "add 99")
    assert t == 7 and "illegal action" in fb


@check("bfs finds shortest path and respects depth")
def _():
    assert S.bfs(7, 7) == []
    p = S.bfs(0, 10)
    assert p is not None and len(p) <= S.MAX_STEPS
    t = 0
    for a in p:
        t, _ = S.apply_action(t, a)
    assert t == 10
    # unreachable within depth on a far target with tight depth
    assert S.bfs(0, 999, max_depth=1) is None


@check("teacher episodes are valid env rollouts")
def _():
    rng = random.Random(0)
    for _ in range(30):
        s, t = rng.randint(0, 20), rng.randint(10, 120)
        if S.bfs(s, t) is None:
            continue
        steps, reached, final = S.teacher_episode(s, t, rng)
        assert len(steps) <= S.MAX_STEPS
        # replay: every feedback string must match env dynamics
        total = s
        for a, fb in steps:
            total, want = S.apply_action(total, a)
            if total == t and a == steps[-1][0]:
                assert fb.startswith("goal reached"), (a, fb)
            else:
                assert fb == want, (a, fb, want)
        assert reached == (total == t) and final == total


@check("hindsight relabeling: failures become relabeled successes")
def _():
    train, _ = S.make_data(n_train=40, n_eval=8, seed=1)
    assert len(train) == 40
    for s, t, steps in train:
        total = s
        for a, _ in steps:
            total, _ = S.apply_action(total, a)
        assert total == t, (s, t, total, steps)  # relabeled => reached
        assert steps and steps[-1][1].startswith("goal reached")


@check("episode_text renders the full trajectory")
def _():
    txt = S.episode_text(7, 50, [("add 6", "total=13"),
                                 ("mul 4", "goal reached: total=50")])
    assert "Task: reach 50. Start: 7." in txt
    assert "> add 6" in txt and "total=13" in txt
    assert txt.endswith("\n")


@check("mistake episodes contain error or suboptimal feedback")
def _():
    rng = random.Random(0)
    saw_mistake = 0
    for _ in range(200):
        steps, _, _ = S.teacher_episode(5, 60, rng, mistake_p=1.0)
        first_fb = steps[0][1]
        if "error" in first_fb:
            saw_mistake += 1
    assert saw_mistake > 0  # 'div 2' / bad ops surface as error feedback


@check("eval tasks are all solvable within MAX_STEPS")
def _():
    _, eval_tasks = S.make_data(n_train=8, n_eval=20, seed=2)
    for s, t in eval_tasks:
        assert S.bfs(s, t) is not None, (s, t)


def main():
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
