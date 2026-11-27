#!/usr/bin/env python3
"""CPU test for the frozen-factor driver's PURE HELPERS (arXiv 2610.06621).

Extracts the stdlib+numpy helper block from colab_frozenfactor_driver.py
and exercises it without torch: synthetic format-task data, JSON
scoring, the SVD band-init assembly (numpy), and the arm table.
"""
import pathlib

DRIVER = pathlib.Path(__file__).with_name("colab_frozenfactor_driver.py")
SRC = DRIVER.read_text()
START = SRC.index("# ---------------- PURE HELPERS (stdlib + numpy) ----------------")
END = SRC.index("# ---------------- END PURE HELPERS ----------------")
NS = {}
exec(compile(SRC[START:END], str(DRIVER), "exec"), NS)

make_format_examples = NS["make_format_examples"]
format_prompt = NS["format_prompt"]
format_train_text = NS["format_train_text"]
score_format_json = NS["score_format_json"]
obqa_prompt = NS["obqa_prompt"]
assemble_lora_init = NS["assemble_lora_init"]
arm_table = NS["arm_table"]
ARMS = [
    ("fmt_r2_standard", "format", 2, "none", "default"),
    ("fmt_r2_freezeA_dom", "format", 2, "A", "dominant"),
    ("fmt_r2_freezeA_minor", "format", 2, "A", "minor"),
    ("fmt_r2_freezeB_dom", "format", 2, "B", "dominant"),
    ("fmt_r2_freezeB_minor", "format", 2, "B", "minor"),
    ("fmt_r8_freezeA_dom", "format", 8, "A", "dominant"),
    ("fmt_r8_freezeB_dom", "format", 8, "B", "dominant"),
    ("obqa_r2_freezeA_dom", "obqa", 2, "A", "dominant"),
    ("obqa_r2_freezeB_dom", "obqa", 2, "B", "dominant"),
    ("obqa_r8_freezeA_dom", "obqa", 8, "A", "dominant"),
    ("obqa_r8_freezeB_dom", "obqa", 8, "B", "dominant"),
]

fails = []


def check(name, got, want):
    if got != want:
        fails.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok {name}: {got!r}")


def check_close(name, got, want, tol=1e-9):
    if abs(got - want) > tol:
        fails.append(f"{name}: got {got!r} want {want!r} (tol {tol})")
    else:
        print(f"ok {name}: {got!r}")


# --- synthetic format task: deterministic + self-consistent ---
a = make_format_examples(20261127, 10)
b = make_format_examples(20261127, 10)
check("format examples deterministic", a, b)
check("format examples count", len(a), 10)
import json as _json
for i, e in enumerate(a):
    obj = _json.loads(e["target"])
    assert obj == e["gold"], f"example {i}: target != gold"
    assert list(obj.keys()) == ["ticker", "sentiment", "impact"]
    assert e["target"] in format_train_text(e["news"], e["target"])
    assert format_prompt(e["news"]) in format_train_text(e["news"], e["target"])
print("ok format examples: targets parse, keys ordered, prompts embed")
# all three sentiment/impact values appear across a larger sample
big = make_format_examples(7, 500)
sents = {e["gold"]["sentiment"] for e in big}
imps = {e["gold"]["impact"] for e in big}
check("format sentiments covered", sents, {"positive", "negative", "neutral"})
check("format impacts covered", imps, {"high", "medium", "low"})

# --- JSON scoring ---
gold = {"ticker": "ACME", "sentiment": "positive", "impact": "high"}
check("score perfect",
      score_format_json('{"ticker": "ACME", "sentiment": "positive", "impact": "high"}', gold),
      (1, 1))
check("score wrong value, right format",
      score_format_json('{"ticker": "ACME", "sentiment": "negative", "impact": "high"}', gold),
      (1, 0))
check("score wrong key order",
      score_format_json('{"sentiment": "positive", "ticker": "ACME", "impact": "high"}', gold),
      (0, 0))
check("score extra key",
      score_format_json('{"ticker": "ACME", "sentiment": "positive", "impact": "high", "x": 1}', gold),
      (0, 0))
check("score bad value vocab",
      score_format_json('{"ticker": "ZZZZ", "sentiment": "positive", "impact": "high"}', gold),
      (0, 0))
check("score garbage", score_format_json("I don't know", gold), (0, 0))
check("score empty", score_format_json("", gold), (0, 0))
check("score text around json",
      score_format_json('Here it is:\n{"ticker": "ACME", "sentiment": "positive", "impact": "high"}\nDone.', gold),
      (1, 1))
check("score unparseable json",
      score_format_json('{"ticker": "ACME", "sentiment": }', gold), (0, 0))

# --- OBQA prompt ---
p = obqa_prompt("What is 2+2?", ["1", "2", "3", "4"])
check("obqa prompt",
      p, "Question: What is 2+2?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:")

# --- SVD band-init assembly (numpy) ---
try:
    import numpy as np
except ImportError:
    print("SKIP numpy section: numpy not available")
    np = None

if np is not None:
    rng = np.random.default_rng(0)
    # random tall weight, exact SVD
    W = rng.normal(size=(24, 16))
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    assert np.all(np.diff(S) <= 1e-9), "singular values not descending"
    for r, freeze, band in ((2, "A", "dominant"), (2, "A", "minor"),
                            (2, "B", "dominant"), (2, "B", "minor"),
                            (8, "A", "dominant"), (8, "B", "minor")):
        A, B, fA, fB = assemble_lora_init(U, S, Vh, r, freeze, band)
        check(f"init {freeze}/{band}/r{r} shapes", (A.shape, B.shape),
              ((r, 16), (24, r)))
        check_close(f"init {freeze}/{band}/r{r} zero update",
                    float(np.abs(B @ A).max()), 0.0)
        if freeze == "A":
            check(f"init {freeze}/{band}/r{r} freeze flags", (fA, fB),
                  (True, False))
            want = Vh[:r, :] if band == "dominant" else Vh[-r:, :]
            check(f"init {freeze}/{band}/r{r} A holds singular vectors",
                  bool(np.allclose(A, want)), True)
            check_close(f"init {freeze}/{band}/r{r} A orthonormal rows",
                        float(np.abs(A @ A.T - np.eye(r)).max()), 0.0)
        else:
            check(f"init {freeze}/{band}/r{r} freeze flags", (fA, fB),
                  (False, True))
            want = U[:, :r] if band == "dominant" else U[:, -r:]
            check(f"init {freeze}/{band}/r{r} B holds singular vectors",
                  bool(np.allclose(B, want)), True)
            check_close(f"init {freeze}/{band}/r{r} B orthonormal cols",
                        float(np.abs(B.T @ B - np.eye(r)).max()), 0.0)
    # dominant vs minor really differ
    A_dom, _, _, _ = assemble_lora_init(U, S, Vh, 2, "A", "dominant")
    A_min, _, _, _ = assemble_lora_init(U, S, Vh, 2, "A", "minor")
    check("init dominant != minor", bool(np.allclose(A_dom, A_min)), False)
    # standard arm: zeros, nothing frozen
    A0, B0, fA0, fB0 = assemble_lora_init(U, S, Vh, 2, "none", "default")
    check("init standard zeros",
          (float(np.abs(A0).max()), float(np.abs(B0).max()), fA0, fB0),
          (0.0, 0.0, False, False))
    # invalid band / freeze raise
    for bad in (("A", "bogus"), ("C", "dominant")):
        try:
            assemble_lora_init(U, S, Vh, 2, *bad)
            fails.append(f"init {bad} should have raised")
        except ValueError:
            print(f"ok init {bad} raises ValueError")

# --- arm table ---
check("arm table count", arm_table(ARMS), 11)
try:
    arm_table([("x", "format", 2, "none", "default"),
               ("x", "format", 2, "none", "default")])
    fails.append("arm table should reject duplicate names")
except AssertionError:
    print("ok arm table rejects duplicate names")
try:
    arm_table([("x", "format", 2, "none", "dominant")])
    fails.append("arm table should reject standard arm with band")
except AssertionError:
    print("ok arm table rejects standard arm with band")
try:
    arm_table([("x", "format", 4, "A", "dominant")])
    fails.append("arm table should reject rank 4")
except AssertionError:
    print("ok arm table rejects rank 4")

print()
if fails:
    print(f"FAILED ({len(fails)}):")
    for f in fails:
        print(" -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
