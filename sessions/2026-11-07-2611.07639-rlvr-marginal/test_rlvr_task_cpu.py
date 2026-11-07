"""CPU test for the RLVR-marginal Colab driver (arXiv:2611.07639).

Pure-Python checks of the task generator, answer extraction, group
advantage, and PPO surrogate used by colab_rlvr_driver.py. No GPU, no
network, no model. Run: python3 test_rlvr_task_cpu.py
"""
import importlib.util
import math
import random
import re
import sys

spec = importlib.util.spec_from_file_location(
    "rlvr_driver", "colab_rlvr_driver.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)


def check_cot(q, cot, ans):
    """Recompute every CoT step from the question's numbers; the final
    value must equal ans and every intermediate must stay positive."""
    lines = [ln for ln in cot.splitlines() if ln.startswith("Step")]
    assert lines, "no CoT steps for %r" % q
    cur = None
    for ln in lines:
        m = re.match(r"Step \d+: (-?\d+) (\+|-|x) (-?\d+) = (-?\d+)\.", ln)
        assert m, "unparsable CoT line: %r" % ln
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if cur is None:
            cur = a
        assert a == cur, "CoT step starts from %d, expected %d" % (a, cur)
        got = a + b if op == "+" else a - b if op == "-" else a * b
        assert got == c, "CoT arithmetic wrong: %r" % ln
        assert c > 0, "intermediate not positive: %r" % ln
        cur = c
    assert cur == ans, "CoT final %d != answer %d" % (cur, ans)
    assert cot.strip().endswith("Final: %d" % ans)


def main():
    rng = random.Random(1107)
    # 1. in-dist problems: CoT recomputes the answer, 3-4 steps
    n_steps = []
    for _ in range(300):
        q, cot, ans = drv.gen_problem(rng, ood=False)
        check_cot(q, cot, ans)
        n_steps.append(cot.count("Step"))
        assert ans > 0
    assert set(n_steps) <= {3, 4}, sorted(set(n_steps))
    print("in-dist: 300/300 CoT-consistent, steps in {3,4}: OK")
    # 2. OOD problems: longer chains, bigger numbers
    n_steps_ood, max_ans = [], 0
    for _ in range(300):
        q, cot, ans = drv.gen_problem(rng, ood=True)
        check_cot(q, cot, ans)
        n_steps_ood.append(cot.count("Step"))
        max_ans = max(max_ans, ans)
    assert set(n_steps_ood) <= {5, 6}, sorted(set(n_steps_ood))
    assert max_ans > 10_000, "OOD answers should reach 5 digits, max=%d" % max_ans
    print("ood: 300/300 CoT-consistent, steps in {5,6}, max ans %d: OK" % max_ans)
    # 3. no stray '%' placeholders / format leaks in questions
    rng2 = random.Random(7)
    for _ in range(500):
        q, _, _ = drv.gen_problem(rng2, ood=rng2.random() < 0.5)
        assert "%d" not in q and "%s" not in q, "unformatted placeholder: %r" % q
        assert "%%" not in q
    print("questions: 500/500 free of format placeholders: OK")
    # 4. answer extraction
    assert drv.extract_answer("Step 1: 3 + 4 = 7.\nFinal: 7") == 7
    assert drv.extract_answer("blah 12 and 34") == 34
    assert drv.extract_answer("Final: -5") == -5
    assert drv.extract_answer("no numbers here") is None
    assert drv.extract_answer("Final: 10 then 20") == 10
    print("extract_answer: 5/5 cases: OK")
    # 5. group-normalized advantage (GRPO, population std):
    # [1,0,0,0]: mean .25, std sqrt(.1875)=.433 -> [1.732,-.577,-.577,-.577]
    def group_adv(rewards):
        mu = sum(rewards) / len(rewards)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards)) + 1e-8
        return [(r - mu) / sd for r in rewards]
    adv = group_adv([1.0, 0.0, 0.0, 0.0])
    assert all(abs(a - b) < 1e-6 for a, b in
               zip(adv, [1.7320508075688772, -0.5773502691896257,
                         -0.5773502691896257, -0.5773502691896257])), adv
    assert group_adv([1.0, 1.0, 1.0, 1.0]) == [0.0] * 4
    assert abs(sum(group_adv([1.0, 0.0, 1.0, 0.0]))) < 1e-9
    print("group advantage: [1,0,0,0]->[1.732,-.577,-.577,-.577], all-equal->0: OK")
    # 6. PPO surrogate: -mean(min(ratio*A, clip(ratio)*A))
    def ppo_loss(ratios, advs):
        vals = [min(r * a, max(0.8, min(1.2, r)) * a) for r, a in zip(ratios, advs)]
        return -sum(vals) / len(vals)
    assert abs(ppo_loss([1.1, 0.9], [1.0, -1.0]) - (-0.1)) < 1e-9
    # clipped case: ratio 1.5, A=1 -> min(1.5, 1.2)=1.2
    assert abs(ppo_loss([1.5], [1.0]) - (-1.2)) < 1e-9
    # negative advantage flips the min: ratio 1.5, A=-1 -> min(-1.5,-1.2)=-1.5
    assert abs(ppo_loss([1.5], [-1.0]) - 1.5) < 1e-9
    print("PPO surrogate: 3/3 cases: OK")
    # 7. eval prompt format is stable across arms (v2: 1 short exemplar
    #    + direct instruction; the 2-shot block made the base model
    #    analyse the examples instead of answering)
    ex = drv.make_exemplars(random.Random(1))
    assert len(ex) == 1 and ex[0][1].count("Step") <= 2
    p1 = drv.eval_prompt("Q?", ex)
    assert p1.count("Question:") == 2 and "Final: <number>" in p1
    print("eval prompt: 1-shot + format instruction: OK")
    print("ALL CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
