#!/usr/bin/env python3
"""CPU test for the Agentic-OPD naive-distillation control driver
(arXiv:2611.12799).

Verifies, without GPU/torch:
  (1) the pure helpers shared with the run-f driver (extract_number incl.
      the attempt-1 ValueError failure mode, numbers_equal, gold parsing,
      make_batches) behave identically in both drivers;
  (2) build_naive_examples -- the control's only new logic: every pool
      problem with >=1 correct teacher trace is distilled (no LAD
      admission filter); student rollouts affect only the attribution
      split, never selection;
  (3) config identity with run f (model, seed, LoRA, paper) and the
      experiment unchanged (data splits, lrs, epochs, eval sizes, G=4);
  (4) structural guards: no HCN phase executes, no bare
      model.disable_adapter() call (the run-f no-op trap), naive output
      names present, C3naive eval present.
Run: python3 test_agentic_opd_naive_cpu.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = [0]


def check(name, cond):
    CHECKS[0] += 1
    if cond:
        print(f"ok: {name}")
    else:
        print(f"FAIL: {name}")
        sys.exit(1)


def load_driver(fname):
    path = os.path.join(HERE, fname)
    spec = importlib.util.spec_from_file_location(fname[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d8 = load_driver("colab_agentic_opd_driver.py")
dn = load_driver("colab_agentic_opd_driver_naive.py")

# ---------------------------------------------------------------- helpers
ex = dn.extract_number
check("extract_number: plain int", ex("The answer is 42") == 42.0)
check("extract_number: #### form", ex("... #### 18") == 18.0)
check("extract_number: comma thousands", ex("1,000 dollars") == 1000.0)
check("extract_number: no number -> None", ex("no number here") is None)
for weird in ["#### 12,", "x 1,000, y", "total: 5.", "### 9", "1,2,3,", "0.5.6"]:
    try:
        ex(weird)
        check(f"extract_number: no raise on {weird!r}", True)
    except ValueError:
        check(f"extract_number: no raise on {weird!r}", False)

ne = dn.numbers_equal
check("numbers_equal: exact", ne(18.0, 18.0))
check("numbers_equal: mismatch", not ne(18.0, 19.0))
check("numbers_equal: None pred", not ne(None, 18.0))

for fn in ["extract_number", "numbers_equal", "gold_number_gsm8k",
           "clean_gsm8k_answer", "make_batches"]:
    check(f"helper {fn} identical across drivers",
          getattr(d8, fn).__code__.co_code == getattr(dn, fn).__code__.co_code)
check("clean_gsm8k_answer: strips <<>>",
      "<<" not in dn.clean_gsm8k_answer("5 + 6 = <<5+6=11>>11"))
check("make_batches: sizes", [len(b) for b in dn.make_batches(list(range(10)), 4)] == [4, 4, 2])

# ---------------------------------------------------------------- naive selection
P = lambda prompt, gold: {"prompt": prompt, "gold": gold}
rows = [
    # A: student 0/4, teacher 2/4 -> covered, LAD-admitted-equivalent
    (P("qa", 18.0), ["x 5", "x 7", "x 9", "x 11"],
     ["The answer is 18", "x 3", "no number", "The answer is 18"]),
    # B: student 2/4, teacher 1/4 -> covered, LAD-REJECTED-equivalent
    # (this is the key naive-vs-LAD difference: LAD drops it, naive keeps it)
    (P("qb", 3.0), ["The answer is 3", "x 8", "The answer is 3", "x 1"],
     ["x 9", "The answer is 3", "x 2", "x 4"]),
    # C: teacher 0/4 -> excluded entirely
    (P("qc", 7.0), ["The answer is 7", "x 1", "x 2", "x 3"],
     ["x 8", "x 9", "x 0", "no number"]),
    # D: student 4/4, teacher 4/4 -> covered, LAD-rejected-equivalent
    # (q == s, not >; LAD's strict inequality drops it)
    (P("qd", 42.0), ["The answer is 42"] * 4, ["The answer is 42"] * 4),
]
exs, st = dn.build_naive_examples(rows)
check("naive: pool_n", st["pool_n"] == 4)
check("naive: covered_n (A, B, D)", st["covered_n"] == 3)
check("naive: examples_n == covered_n", st["examples_n"] == 3)
check("naive: LAD-admitted-equiv == 1 (only A)",
      st["lad_admitted_equiv_n"] == 1)
check("naive: LAD-rejected-equiv == 2 (B and D)",
      st["lad_rejected_equiv_n"] == 2)
check("naive: mean_s", abs(st["mean_s"] - 0.4375) < 1e-9)  # (0+0.5+0.25+1)/4
check("naive: mean_q", abs(st["mean_q"] - 0.4375) < 1e-9)  # (0.5+0.25+0+1)/4
check("naive: first correct teacher trace kept (A)",
      exs[0]["completion"] == " The answer is 18")
check("naive: B kept despite teacher < student (no filter)",
      any(e["prompt"] == "qb" for e in exs))
check("naive: C excluded (no correct teacher trace)",
      not any(e["prompt"] == "qc" for e in exs))
check("naive: D kept despite student already perfect",
      any(e["prompt"] == "qd" for e in exs))
check("naive: example carries gold", all("gold" in e for e in exs))

exs0, st0 = dn.build_naive_examples([])
check("naive: empty rows -> zeros", st0 == {
    "pool_n": 0, "covered_n": 0, "examples_n": 0, "mean_s": 0.0,
    "mean_q": 0.0, "lad_admitted_equiv_n": 0, "lad_rejected_equiv_n": 0}
    and exs0 == [])

# ---------------------------------------------------------------- config
check("naive driver model id", dn.MODEL_ID == "Qwen/Qwen3-8B")
check("same seed", dn.SEED == d8.SEED == 123)
check("same LoRA r/alpha", dn.RESULTS["lora"] == d8.RESULTS["lora"] == {
    "r": 16, "alpha": 32, "dropout": 0.0, "target": "q,k,v,o,gate,up,down"})
check("same paper id", dn.RESULTS["paper"] == d8.RESULTS["paper"] == "2611.12799")
check("naive RESULTS has naive block, no hcn/lad blocks",
      "naive" in dn.RESULTS and "hcn" not in dn.RESULTS and "lad" not in dn.RESULTS)
check("naive control records run-f reference (C3 43/33/52, C2 24/21/46)",
      dn.RESULTS["control"]["compares_against"] == {
          "C3_hcn_lad": {"A_val": 43, "A_test": 33, "B_test": 52},
          "C2": {"A_val": 24, "A_test": 21, "B_test": 46}})
check("naive teacher fixed to C1", dn.RESULTS["control"]["teacher"].startswith("C1"))

src8 = open(os.path.join(HERE, "colab_agentic_opd_driver.py")).read()
srcn = open(os.path.join(HERE, "colab_agentic_opd_driver_naive.py")).read()

# experiment unchanged: data splits, lrs, epochs, eval sizes, sampling
for token in ["gtr.iloc[0:400]", "str_.iloc[0:400]", "gte.iloc[0:60]",
              "gte.iloc[100:160]", "ste.iloc[0:60]", "gtr.iloc[400:500]",
              "lr=2e-4, epochs=1", "lr=1e-4, epochs=1", "max_new=120",
              "G=4", "temp=0.8", "GRAD_ACCUM = 2", "batch_size=2"]:
    check(f"experiment token unchanged in naive: {token}", token in srcn)

# naive-specific tokens present
for token in ["agentic_opd_metrics_naive.csv", "agentic_opd_results_naive.json",
              "build_naive_examples", "C3naive", "stageC-naive",
              "adapter_c3naive", "agentic_opd_adapter_naive",
              "PHASE NAIVE-POOL"]:
    check(f"naive driver contains {token}", token in srcn)

# structural guards
check("no HCN phase executes (no 'PHASE HCN')", "PHASE HCN" not in srcn)
check("no HCN results block written", 'RESULTS["hcn"]' not in srcn)
check("no LAD results block written", 'RESULTS["lad"]' not in srcn)
check("no cand_solved / HCN gain computation", "cand_solved" not in srcn
      and "weights_after_round1" not in srcn)
# every disable_adapter() call is entered via `with` (the run-f bare-call
# trap): the only code occurrence is the `ctx = ...` assignment, which is
# entered by `with ctx:`; the other mention is a comment.
import re
bad = [ln for ln in srcn.splitlines()
       if re.match(r"^\s*model\.disable_adapter\(\)", ln)]
check("disable_adapter never called as a bare statement (no no-op call)", not bad)
check("eval_all uses the context manager for base",
      "ctx = model.disable_adapter() if adapter_name is None" in srcn)
check("set_adapter refuses None loudly",
      'assert name is not None' in srcn)

print(f"\n{CHECKS[0]} checks passed")
