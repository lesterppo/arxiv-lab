"""CPU pre-check for the EliteReplay training track (stdlib only).

Verifies, with NO GPU:
  1. synthetic text-to-SQL data generation (150 train / 60 eval, unique SQL)
  2. sqlite execution reward: correct->1, wrong->0, non-SELECT->0
  3. SQL fingerprint: distinct strategies -> distinct fps; paraphrases collapse
  4. elite archive construction with a toy simulated policy
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random

import elite_common as E

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


# 1. data generation (single-pass split => disjoint by construction)
train, evalp = E.generate_split(150, 60, seed=11)
check("gen train n=150", len(train) == 150, f"got {len(train)}")
check("gen eval n=60", len(evalp) == 60, f"got {len(evalp)}")
check("train SQL unique", len({p["sql"] for p in train}) == 150)
check("eval SQL unique", len({p["sql"] for p in evalp}) == 60)
check("train/eval disjoint", not ({p["sql"] for p in train} & {p["sql"] for p in evalp}))
check("nl non-empty", all(p["nl"] and len(p["nl"]) > 10 for p in train))

db = E.build_db(seed=7)
# every reference SQL must execute and return rows
bad = [p["id"] for p in train + evalp if not E.execute_sql(db, p["sql"])[0]]
check("all ref SQL execute", not bad, f"bad={bad[:3]}")

# 2. reward semantics
p0 = train[0]
check("reward exact ref = 1", E.reward_correct(db, p0["sql"], p0["sql"]) == 1)
wrong = "SELECT COUNT(*) FROM staff WHERE dept = 'nope-never'"
check("reward wrong = 0", E.reward_correct(db, wrong, p0["sql"]) == 0)
check("reward non-SELECT = 0", E.reward_correct(db, "DROP TABLE staff;", p0["sql"]) == 0)
check("reward syntax error = 0", E.reward_correct(db, "SELECT FROM WHERE", p0["sql"]) == 0)
# semantically equal variant (different text) must score 1
ref = "SELECT name, salary FROM staff WHERE dept = 'sales' ORDER BY salary DESC LIMIT 3"
var = "select  name,salary  from staff where dept='sales' order by salary desc limit 3;"
check("reward semantic variant = 1", E.reward_correct(db, var, ref) == 1, )

# extract_sql
check("extract fence",
      E.extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1;",
      E.extract_sql("```sql\nSELECT 1;\n```"))
check("extract bare", E.extract_sql("blah SELECT 1; tail") == "SELECT 1;")

# 3. fingerprint diversity
check("fp identical", E.fingerprint(ref) == E.fingerprint(var))
check("fp distinct strategies",
      E.fingerprint("SELECT COUNT(*) FROM staff WHERE dept='sales'") !=
      E.fingerprint("SELECT COUNT(*) FROM staff WHERE dept='eng'"))
check("fp between vs range",
      E.fingerprint("SELECT COUNT(*) FROM staff WHERE age BETWEEN 25 AND 35") !=
      E.fingerprint("SELECT COUNT(*) FROM staff WHERE age >= 25 AND age <= 35"))
strat = [(var, ref), ("SELECT COUNT(*) FROM staff WHERE dept='zzz'", ref)]
check("distinct_correct_strategies=1", E.distinct_correct_strategies(db, strat) == 1)

# 4. elite archive with a toy simulated policy
rng = random.Random(3)


def toy_sampler(nl, ref_sql):
    """Simulate a frozen policy: paraphrases (correct), junk (incorrect)."""
    outs = []
    for _ in range(4):
        outs.append(ref_sql.replace("SELECT", "select").replace("  ", " "))  # correct paraphrase
    for _ in range(2):
        outs.append("SELECT * FROM staff WHERE 1=0")  # wrong (empty result may still be 0 rows)
    outs.append(ref_sql)  # exact
    outs.append("nonsense not sql")
    rng.shuffle(outs)
    return outs


arch = E.build_elite_archive(db, train[:40], toy_sampler, gens_per_prompt=8, cap=200)
check("archive non-empty", len(arch) > 0, f"size={len(arch)}")
# correct paraphrase + exact collapse to 1 fingerprint per prompt -> 40 entries
check("archive dedups fingerprints", len(arch) == 40, f"size={len(arch)}")
check("archive all correct",
      all(E.reward_correct(db, it["sql"], next(p["sql"] for p in train[:40] if p["id"] == it["prompt_id"])) == 1
          for it in arch.items))
ppc = arch.per_prompt_counts()
check("archive per-prompt <= distinct", all(v == 1 for v in ppc.values()), str(set(ppc.values())))
# cap enforcement
big = E.build_elite_archive(db, train, toy_sampler, gens_per_prompt=8, cap=25)
check("archive cap respected", len(big) == 25, f"size={len(big)}")

print()
if fails:
    print("PRECHECK FAILED:", fails)
    sys.exit(1)
print("ALL CPU PRECHECKS PASSED")
