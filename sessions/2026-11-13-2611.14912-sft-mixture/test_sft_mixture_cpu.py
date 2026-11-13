#!/usr/bin/env python3
"""CPU checks for the SFT-mixture Colab driver (arXiv:2611.14912).

Validates the pure-Python logic without a model or downloads, using a
tiny synthetic GSM8K-shaped dataset:
  - clean_gsm8k_answer / gold_number_gsm8k / equation_format
  - extract_number / numbers_equal
  - build_arms: matched sizes, disjoint sources, all-correct targets,
    arm S is self-gen only, arm M is exactly 1/3+1/3+1/3, short-pool error
Runs offline, stdlib + the driver module (imported with the Colab-only
imports guarded: the driver imports pandas at top level, so this test
stubs it out when pandas is unavailable).
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The driver imports pandas at module level; stub it if absent so the
# pure-logic functions stay testable on a bare stdlib box.
try:
    import pandas  # noqa: F401
except ImportError:
    sys.modules["pandas"] = types.ModuleType("pandas")

import colab_sft_mixture_driver as d

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# --- synthetic GSM8K-shaped rows (dicts, not DataFrames: build_arms takes
# row-iterables with .iterrows-style access, so emulate with a tiny frame)
class MiniFrame:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        yield from enumerate(self._rows)


def synth_human(n, start=0, name="Janet"):
    rows = []
    for i in range(n):
        q = f"Q{start + i}: {name} has {i + 2} apples. She buys {i + 1} more. How many?"
        ans = (f"{name} starts with {i + 2}. She buys {i + 1} more. "
               f"<<{i + 2}+{i + 1}={2 * i + 3}>> The answer is {2 * i + 3}. "
               f"#### {2 * i + 3}")
        rows.append({"question": q, "answer": ans})
    return MiniFrame(rows)


def synth_selfgen(n):
    out = []
    for i in range(n):
        out.append((f"Q{i}: A train goes {i + 10} miles.\nSolution:",
                    f" The train goes {i + 10} miles. #### {i + 10}",
                    float(i + 10)))
    return out


def main():
    print("clean / gold / equation_format:")
    raw = "She has 3. <<3*4=12>> Then 5 more. <<12+5=17>> #### 17"
    check("clean strips <<>> annotations",
          d.clean_gsm8k_answer(raw) == "She has 3. Then 5 more. #### 17",
          repr(d.clean_gsm8k_answer(raw)))
    check("gold extracts 17", d.gold_number_gsm8k(raw) == 17.0)
    check("gold handles commas/decimals",
          d.gold_number_gsm8k("x #### 1,234.5") == 1234.5)
    check("gold returns None without ####",
          d.gold_number_gsm8k("no marker here") is None)
    check("equation_format builds the chain",
          d.equation_format(raw, 17.0) == "3*4=12; 12+5=17\n#### 17",
          repr(d.equation_format(raw, 17.0)))
    check("equation_format falls back when no annotations",
          d.equation_format("plain text #### 5", 5.0) == "plain text #### 5")
    check("equation_format keeps decimal gold",
          d.equation_format("<<1/2=0.5>> #### 0.5", 0.5) == "1/2=0.5\n#### 0.5")

    print("extract_number / numbers_equal:")
    check("extract prefers #### answer",
          d.extract_number("work 3*4=12 then #### 17") == 17.0)
    check("extract falls back to last number",
          d.extract_number("the answer is 42") == 42.0)
    check("extract None on empty", d.extract_number("no numbers") is None)
    check("numbers_equal tolerance",
          d.numbers_equal(17.0, 17.0) and not d.numbers_equal(17.0, 18.0)
          and not d.numbers_equal(None, 17.0))

    print("build_arms:")
    human = synth_human(160, name="Janet")
    eq = synth_human(160, start=1000, name="Maria")
    selfgen = synth_selfgen(700)
    arm_s, arm_m, counts = d.build_arms(human, eq, selfgen)
    check("arm S size == N_ARM", len(arm_s) == d.N_ARM, f"{len(arm_s)}")
    check("arm M size == N_ARM", len(arm_m) == d.N_ARM, f"{len(arm_m)}")
    check("counts add up",
          counts == {"arm_s_selfgen": 480, "arm_m_human": 160,
                     "arm_m_eq": 160, "arm_m_selfgen": 160}, str(counts))
    check("arm S is self-gen only (prompts start with Q, no human rows)",
          all(p["prompt"].startswith("Q") and "Janet" not in p["prompt"]
              and "Maria" not in p["prompt"]
              for p in arm_s))
    check("arm M has exactly 160 human, 160 eq, 160 self-gen",
          sum(1 for p in arm_m if "Janet" in p["prompt"]) == 160
          and sum(1 for p in arm_m if "Maria" in p["prompt"]) == 160
          and sum(1 for p in arm_m if "train goes" in p["prompt"]) == 160)
    check("arm M human and eq prompts are disjoint question sets",
          len({p["prompt"] for p in arm_m if "Janet" in p["prompt"]
               or "Maria" in p["prompt"]}) == 320)
    check("every arm example carries a gold number",
          all("gold" in e and e["gold"] is not None for e in arm_s + arm_m))
    check("every arm completion ends with #### gold",
          all("####" in e["completion"] for e in arm_s + arm_m))
    check("arm S and arm M share no self-gen prompt",
          not ({p["prompt"] for p in arm_s}
               & {p["prompt"] for p in arm_m if "train goes" in p["prompt"]}))
    try:
        d.build_arms(human, eq, synth_selfgen(100))
        check("short self-gen pool raises", False, "no error raised")
    except RuntimeError as e:
        check("short self-gen pool raises", True, str(e)[:60])

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)


if __name__ == "__main__":
    main()
