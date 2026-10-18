"""EliteReplay experiment — shared logic (CPU-safe, stdlib only).

Synthetic text-to-SQL task with sqlite-verifiable rewards, SQL fingerprinting
for strategy diversity, and elite-archive construction logic.
"""
import random
import re
import sqlite3

# ---------------------------------------------------------------- schema ----
SCHEMA_SQL = """
CREATE TABLE staff(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER, age INTEGER);
CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price INTEGER, stock INTEGER);
"""

FIRST = ["amy", "bob", "cara", "dan", "erin", "fred", "gina", "hank", "ivy", "jack"]
DEPTS = ["sales", "eng", "hr", "ops"]
CATS = ["books", "toys", "tools", "food"]


def build_db(seed=0):
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA query_only=OFF")
    db.executescript(SCHEMA_SQL)
    for i in range(40):
        db.execute(
            "INSERT INTO staff VALUES (?,?,?,?,?)",
            (i + 1, rng.choice(FIRST) + str(i), rng.choice(DEPTS),
             rng.randint(30, 120) * 1000, rng.randint(22, 60)),
        )
    for i in range(40):
        db.execute(
            "INSERT INTO products VALUES (?,?,?,?,?)",
            (i + 1, "item" + str(i), rng.choice(CATS),
             rng.randint(5, 500), rng.randint(0, 200)),
        )
    db.commit()
    return db


# ------------------------------------------------------------- problems -----
AGGS = ["COUNT(*)", "AVG(salary)", "MAX(salary)", "MIN(salary)", "SUM(salary)",
        "COUNT(*)", "AVG(price)", "MAX(price)", "MIN(price)", "SUM(price)"]

# Each generator returns (nl_question, reference_sql)
def _staff_agg(rng):
    agg = rng.choice([a for a in AGGS if "salary" in a or a == "COUNT(*)"])
    dept = rng.choice(DEPTS)
    col = agg.split("(")[1].rstrip(")") if "(" in agg and "*" not in agg else "*"
    alias = agg.replace("(", "_").replace(")", "").replace("*", "all").lower()
    if rng.random() < 0.5:
        sql = f"SELECT {agg} FROM staff WHERE dept = '{dept}'"
        nl = f"What is the {alias} of staff in the {dept} department?"
    else:
        lo = rng.randint(40, 80) * 1000
        sql = f"SELECT {agg} FROM staff WHERE dept = '{dept}' AND salary > {lo}"
        nl = f"What is the {alias} of {dept} staff earning more than {lo}?"
    return nl, sql


def _staff_list(rng):
    dept = rng.choice(DEPTS)
    k = rng.randint(2, 5)
    order = rng.choice(["salary DESC", "age ASC", "salary ASC"])
    sql = f"SELECT name, salary FROM staff WHERE dept = '{dept}' ORDER BY {order} LIMIT {k}"
    nl = (f"List {k} {dept} staff names and salaries ordered by {order}.")
    return nl, sql


def _prod_agg(rng):
    agg = rng.choice([a for a in AGGS if "price" in a or a == "COUNT(*)"])
    cat = rng.choice(CATS)
    alias = agg.replace("(", "_").replace(")", "").replace("*", "all").lower()
    if rng.random() < 0.5:
        sql = f"SELECT {agg} FROM products WHERE category = '{cat}'"
        nl = f"What is the {alias} of products in the {cat} category?"
    else:
        hi = rng.randint(50, 300)
        sql = f"SELECT {agg} FROM products WHERE category = '{cat}' AND price < {hi}"
        nl = f"What is the {alias} of {cat} products cheaper than {hi}?"
    return nl, sql


def _prod_list(rng):
    cat = rng.choice(CATS)
    k = rng.randint(2, 5)
    order = rng.choice(["price DESC", "stock ASC", "price ASC"])
    sql = f"SELECT name, price FROM products WHERE category = '{cat}' ORDER BY {order} LIMIT {k}"
    nl = f"List {k} {cat} product names and prices ordered by {order}."
    return nl, sql


def _between_variant(rng):
    dept = rng.choice(DEPTS)
    lo = rng.randint(25, 35)
    hi = rng.randint(45, 55)
    sql = f"SELECT COUNT(*) FROM staff WHERE dept = '{dept}' AND age BETWEEN {lo} AND {hi}"
    nl = f"How many {dept} staff are between ages {lo} and {hi}?"
    return nl, sql


GENS = [_staff_agg, _staff_list, _prod_agg, _prod_list, _between_variant]


def generate_problems(n, seed, exclude=None):
    rng = random.Random(seed)
    out = []
    seen = set(exclude or ())
    while len(out) < n:
        nl, sql = rng.choice(GENS)(rng)
        if sql in seen:
            continue
        seen.add(sql)
        out.append({"nl": nl, "sql": sql, "id": f"p{len(out):03d}"})
    return out


def generate_split(n_train, n_eval, seed):
    """Generate train+eval in one pass so the two sets are disjoint."""
    allp = generate_problems(n_train + n_eval, seed)
    train, evalp = allp[:n_train], allp[n_train:]
    for i, p in enumerate(train):
        p["id"] = f"t{i:03d}"
    for i, p in enumerate(evalp):
        p["id"] = f"e{i:03d}"
    return train, evalp


# ---------------------------------------------------------------- reward ----
_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)


def execute_sql(db, sql, timeout_ms=2000):
    """Run a SELECT; return (ok, rows_sorted_as_tuples). Non-SELECT -> (False, None)."""
    if not _SELECT_RE.match(sql or ""):
        return False, None
    try:
        db.execute(f"PRAGMA busy_timeout={timeout_ms}")
        cur = db.execute(sql)
        rows = cur.fetchall()
        norm = sorted(tuple("" if v is None else v for v in r) for r in rows)
        return True, norm
    except Exception:
        return False, None


def reward_correct(db, pred_sql, ref_sql):
    ok_p, rows_p = execute_sql(db, pred_sql)
    ok_r, rows_r = execute_sql(db, ref_sql)
    if not (ok_p and ok_r):
        return 0
    return 1 if rows_p == rows_r else 0


def extract_sql(text):
    """Pull the SQL out of a model completion (code fence or first SELECT..;)."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(select\b.*?;)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(select\b.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ------------------------------------------------------------ fingerprint ---
def fingerprint(sql):
    s = (sql or "").lower().strip().rstrip(";").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*=\s*", "=", s)
    s = re.sub(r"['\"]", "'", s)
    return s


# -------------------------------------------------------- elite archive ----
class EliteArchive:
    """Quality-diversity archive: keep distinct correct solutions (by fingerprint)."""

    def __init__(self, cap=200):
        self.cap = cap
        self.items = []          # list of dicts: prompt_id, nl, sql, fingerprint
        self._fps = set()

    def add(self, prompt_id, nl, sql, correct):
        if not correct:
            return False
        fp = fingerprint(sql)
        if fp in self._fps or len(self.items) >= self.cap:
            return False
        self._fps.add(fp)
        self.items.append({"prompt_id": prompt_id, "nl": nl, "sql": sql, "fp": fp})
        return True

    def __len__(self):
        return len(self.items)

    def per_prompt_counts(self):
        d = {}
        for it in self.items:
            d[it["prompt_id"]] = d.get(it["prompt_id"], 0) + 1
        return d


def distinct_correct_strategies(db, candidates):
    """candidates: list of (pred_sql, ref_sql). Returns # distinct correct fingerprints."""
    fps = set()
    for pred, ref in candidates:
        if reward_correct(db, pred, ref):
            fps.add(fingerprint(pred))
    return len(fps)


def build_elite_archive(db, problems, sampler, gens_per_prompt=8, cap=200):
    """sampler(prompt_nl, ref_sql) -> list of candidate sql strings (frozen policy)."""
    arch = EliteArchive(cap=cap)
    for p in problems:
        cands = sampler(p["nl"], p["sql"])
        for sql in cands[:gens_per_prompt]:
            arch.add(p["id"], p["nl"], sql, reward_correct(db, sql, p["sql"]))
        if len(arch) >= cap:
            break
    return arch
