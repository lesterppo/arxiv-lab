#!/usr/bin/env python3
"""CPU test for the Agentic-OPD automatic-fallback driver (arXiv:2611.12799).

Verifies, without GPU/torch:
  (1) the pure helpers shared by both drivers, incl. the exact failure mode
      that killed attempt agentic-opd-1112 (ValueError in extract_number
      during base eval -- the current driver guards both branches);
  (2) the 4B fallback driver differs from the 8B driver ONLY in the intended
      shrink set (model id, seq cap, micro-batch/accum, eval/sample batch,
      docstring + rationale) -- same data, stages, evals, seeds, LoRA config.
Run: python3 test_agentic_opd_fallback_cpu.py
"""
import difflib
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
d4 = load_driver("colab_agentic_opd_driver_4b.py")

# ---------------------------------------------------------------- helpers
ex = d8.extract_number
check("extract_number: plain int", ex("The answer is 42") == 42.0)
check("extract_number: #### form", ex("... #### 18") == 18.0)
check("extract_number: comma thousands", ex("1,000 dollars") == 1000.0)
check("extract_number: decimal", ex("3.5 kg") == 3.5)
check("extract_number: negative", ex("result -2") == -2.0)
check("extract_number: no number -> None", ex("no number here") is None)
check("extract_number: empty string -> None", ex("") is None)
check("extract_number: bare #### -> None", ex("####") is None)
check("extract_number: trailing comma", ex("Answer: 7,") == 7.0)
check("extract_number: trailing dot", ex("Answer: 7.") == 7.0)
check("extract_number: comma-only token", ex("a, b, c") is None)
# the attempt-1 failure mode: a generation whose last number-like token
# cannot convert to float must not raise (old driver raised ValueError here)
for weird in ["#### 12,", "x 1,000, y", "total: 5.", "### 9", "1,2,3,", "0.5.6"]:
    try:
        ex(weird)
        check(f"extract_number: no raise on {weird!r}", True)
    except ValueError:
        check(f"extract_number: no raise on {weird!r}", False)

ne = d8.numbers_equal
check("numbers_equal: exact", ne(18.0, 18.0))
check("numbers_equal: tolerance", ne(18.0, 18.0000001))
check("numbers_equal: mismatch", not ne(18.0, 19.0))
check("numbers_equal: None pred", not ne(None, 18.0))
check("numbers_equal: None gold", not ne(18.0, None))

check("gold_number_gsm8k: parses ####", d8.gold_number_gsm8k("... #### 1,200") == 1200.0)
check("gold_number_gsm8k: no #### -> None", d8.gold_number_gsm8k("no marker") is None)
check("clean_gsm8k_answer: strips <<>>",
      "<<" not in d8.clean_gsm8k_answer("5 + 6 = <<5+6=11>>11"))
check("make_batches: sizes", [len(b) for b in d8.make_batches(list(range(10)), 4)] == [4, 4, 2])
check("make_batches: empty", list(d8.make_batches([], 4)) == [])

# helpers identical across drivers (same function objects' source behaviour)
for fn in ["extract_number", "numbers_equal", "gold_number_gsm8k",
           "clean_gsm8k_answer", "make_batches"]:
    check(f"helper {fn} identical across drivers",
          getattr(d8, fn).__code__.co_code == getattr(d4, fn).__code__.co_code)

# ---------------------------------------------------------------- config
check("4B driver model id", d4.MODEL_ID == "Qwen/Qwen3-4B")
check("8B driver model id", d8.MODEL_ID == "Qwen/Qwen3-8B")
check("same seed", d4.SEED == d8.SEED == 123)
check("same LoRA r/alpha", d4.RESULTS["lora"] == d8.RESULTS["lora"] == {
    "r": 16, "alpha": 32, "dropout": 0.0, "target": "q,k,v,o,gate,up,down"})
check("same paper id", d4.RESULTS["paper"] == d8.RESULTS["paper"] == "2611.12799")

src8 = open(os.path.join(HERE, "colab_agentic_opd_driver.py")).read()
src4 = open(os.path.join(HERE, "colab_agentic_opd_driver_4b.py")).read()

# intended 4B values present, 8B values present in the original
for token in ['MODEL_ID = "Qwen/Qwen3-4B"', "[:384]", "batch_size=1",
              "GRAD_ACCUM = 4", "bs=8, tag=\"eval\"", "bs=4, temp=0.8"]:
    check(f"4B driver contains {token}", token in src4)
for token in ['MODEL_ID = "Qwen/Qwen3-8B"', "[:512]", "batch_size=2",
              "GRAD_ACCUM = 2", "bs=24, tag=\"eval\"", "bs=8, temp=0.8"]:
    check(f"8B driver contains {token}", token in src8)

# experiment unchanged: data splits, lrs, epochs, eval sizes
for token in ["gtr.iloc[0:400]", "str_.iloc[0:400]", "gte.iloc[0:60]",
              "gte.iloc[100:160]", "ste.iloc[0:60]", "gtr.iloc[400:500]",
              "lr=2e-4, epochs=1", "lr=1e-4, epochs=1", "max_new=120",
              "G=4", "temp=0.8"]:
    check(f"experiment token unchanged in 4B: {token}", token in src4)

# diff scope: every changed line must mention an intended token
ALLOWED = ["Qwen3-4B", "Qwen3-8B", "384", "512", "batch_size", "GRAD_ACCUM",
           "bs=8", "bs=24", "bs=4", "fallback", "Fallback", "OOM", "14.47",
           "micro-batch", "dequant", "Automatic", "Rationale",
           "expandable_segments", "proven fallback", "eval batch",
           "sample batch", "banked", "headroom", "Experiment", "model halves",
           "further headroom", "same experiment"]
changed = []
for line in difflib.unified_diff(src8.splitlines(), src4.splitlines(), lineterm=""):
    if line.startswith(("+++", "---", "@@")):
        continue
    if line.startswith(("+", "-")):
        changed.append(line[1:])
check("diff is non-empty", len(changed) > 0)
bad = [c for c in changed if not any(t in c for t in ALLOWED)]
check(f"diff scope: all {len(changed)} changed lines intended (bad: {bad[:3]})",
      not bad)

print(f"\n{CHECKS[0]} checks passed")
