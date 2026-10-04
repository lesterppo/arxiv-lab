"""Session experiment: FloWright-style role co-evolution (arXiv:2610.01026),
agent track, CPU.

Reproduces the paper's core mechanism synthetically:

 (i)  a 3-role workflow (generator -> verifier -> refiner) solves arithmetic
      tasks with a single sparse outcome (final answer correct?);
 (ii) hierarchical structure-aware credit per role is computed from workflow
      artifacts only (no extra labels/executions): generator <- verifier
      acceptance, verifier <- vindicated decisions, refiner <- correctness
      on handled tasks;
 (iii) roles evolve by (1+1)-ES hill-climbing on their own credit.

Conditions: 'flat' (all roles evolve on the shared sparse reward — no credit
assignment), 'single' (only the generator evolves, on structured credit),
'coev' (all roles co-evolve on structured credits). The paper's direction:
coev > single.

DataWright touch: training on easy tasks (1-op, solvable by a single role)
vs hardened tasks (2-3 ops); the workflow-level gap should open on hardened
tasks while easy tasks sit at ceiling.

Writes results.json next to this file and prints a summary.
Pure stdlib. Deterministic (all RNGs seeded).
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from arxiv_lab.workflow.flowright import (
    CONDITIONS, ROLES, accuracy, accuracy_by_nops, evolve, gen_task,
)

HERE = os.path.dirname(os.path.abspath(__file__))

MASTER_SEED = 20261004
SEEDS = list(range(6))
GENERATIONS = 60
SIGMA = 0.07
TRAIN_N = 192  # eval tasks per generation; 96 left coev's generator
              # noise-limited (accept-rate gradient ~ eval noise floor)
TEST_N = 256  # stratified over n_ops in {1, 2, 3}
COND_SEED = {"flat": 11, "single": 22, "coev": 33}  # fixed ints: never use hash()


def make_tasks(seed, n, n_ops_choices):
    rng = random.Random(seed)
    return [gen_task(rng, rng.choice(n_ops_choices)) for _ in range(n)]


def run_difficulty(difficulty):
    """Run all conditions x seeds for one training difficulty.

    difficulty: 'hardened' (train on 2-3 op tasks) or 'easy' (train on 1-op).
    Test set is always stratified over {1, 2, 3} ops.
    """
    train_choices = (2, 3) if difficulty == "hardened" else (1,)
    out = {"difficulty": difficulty, "seeds": {}}
    for seed in SEEDS:
        train = make_tasks(MASTER_SEED + 1000 * seed + (1 if difficulty == "easy" else 2),
                           TRAIN_N, train_choices)
        test = []
        per = TEST_N // 3
        for i, n_ops in enumerate((1, 2, 3)):
            test.extend(make_tasks(MASTER_SEED + 2000 * seed + 10 + i, per, (n_ops,)))
        base_skills = {r: 0.5 for r in ROLES}
        seed_rec = {
            "baseline_acc": accuracy(base_skills, test, seed=seed),
            "baseline_by_nops": accuracy_by_nops(base_skills, test, seed=seed),
            "conditions": {},
        }
        for cond in CONDITIONS:
            res = evolve(train, condition=cond, generations=GENERATIONS,
                         sigma=SIGMA, seed=MASTER_SEED + 3000 * seed + COND_SEED[cond])
            skills = res["skills"]
            seed_rec["conditions"][cond] = {
                "skills": {r: round(s, 4) for r, s in skills.items()},
                "test_acc": accuracy(skills, test, seed=seed),
                "test_by_nops": accuracy_by_nops(skills, test, seed=seed),
                "final_train_acc": res["train_acc"][-1],
            }
        out["seeds"][str(seed)] = seed_rec
    return out


def summarize(results):
    lines = []
    for difficulty in ("hardened", "easy"):
        d = results[difficulty]
        lines.append("== train difficulty: %s ==" % difficulty)
        for cond in CONDITIONS:
            accs = [d["seeds"][str(s)]["conditions"][cond]["test_acc"] for s in SEEDS]
            mean = sum(accs) / len(accs)
            lines.append("  %-6s test acc: %.3f  (per-seed: %s)"
                         % (cond, mean,
                            " ".join("%.3f" % a for a in accs)))
        # per-n_ops for coev vs single on hardened
        if difficulty == "hardened":
            for n_ops in (1, 2, 3):
                row = []
                for cond in CONDITIONS:
                    vals = [d["seeds"][str(s)]["conditions"][cond]["test_by_nops"][n_ops]
                            for s in SEEDS]
                    row.append("%s=%.3f" % (cond, sum(vals) / len(vals)))
                lines.append("  n_ops=%d: %s" % (n_ops, " ".join(row)))
        # mean final skills
        for cond in CONDITIONS:
            sk = {r: sum(d["seeds"][str(s)]["conditions"][cond]["skills"][r]
                         for s in SEEDS) / len(SEEDS) for r in ROLES}
            lines.append("  %-6s mean skills: %s"
                         % (cond, " ".join("%s=%.2f" % (r[:3], sk[r]) for r in ROLES)))
    return "\n".join(lines)


def main():
    results = {"paper": "arXiv:2610.01026", "master_seed": MASTER_SEED,
               "generations": GENERATIONS, "sigma": SIGMA,
               "train_n": TRAIN_N, "test_n": TEST_N, "seeds": SEEDS}
    for difficulty in ("hardened", "easy"):
        results[difficulty] = run_difficulty(difficulty)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(summarize(results))
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
