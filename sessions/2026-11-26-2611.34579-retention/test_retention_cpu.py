#!/usr/bin/env python3
"""CPU test for the retention driver's PURE HELPERS (arXiv 2611.34579).

Extracts the stdlib-only helper block from colab_retention_driver.py and
exercises it without torch: ROUGE-L, GSM8K parsing, MMLU letter parsing,
SFT formatting, MMLU few-shot prompt formatting.
"""
import pathlib

DRIVER = pathlib.Path(__file__).with_name("colab_retention_driver.py")
SRC = DRIVER.read_text()
START = SRC.index("# ---------------- PURE HELPERS (stdlib) ----------------")
END = SRC.index("# ---------------- END PURE HELPERS ----------------")
NS = {}
exec(compile(SRC[START:END], str(DRIVER), "exec"), NS)

rouge_l_f1 = NS["rouge_l_f1"]
gsm8k_correct = NS["gsm8k_correct"]
parse_last_int = NS["parse_last_int"]
parse_mmlu_letter = NS["parse_mmlu_letter"]
mmlu_correct = NS["mmlu_correct"]
format_sft_example = NS["format_sft_example"]
format_sft_prompt = NS["format_sft_prompt"]
format_mmlu_prompt = NS["format_mmlu_prompt"]

fails = []


def check(name, got, want):
    if got != want:
        fails.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok {name}: {got!r}")


# --- ROUGE-L ---
check("rouge identical", round(rouge_l_f1("the cat sat", "the cat sat"), 4), 1.0)
check("rouge disjoint", rouge_l_f1("apple banana", "cat dog"), 0.0)
check("rouge empty gen", rouge_l_f1("", "some reference"), 0.0)
check("rouge empty ref", rouge_l_f1("some text", ""), 0.0)
# LCS "the cat sat" of 3 vs ref 4 tokens: P=3/3, R=3/4 -> F1 = 6/7
check("rouge partial", round(rouge_l_f1("the cat sat", "the cat sat down"), 4),
      round(6 / 7, 4))
# order matters for LCS: "a b c" vs "c b a" -> LCS 1 -> P=R=1/3 -> F1=1/3
check("rouge order", round(rouge_l_f1("a b c", "c b a"), 4), round(1 / 3, 4))

# --- GSM8K parsing ---
check("last int", parse_last_int("work ... #### 72"), 72)
check("last int commas", parse_last_int("total 1,234 dollars"), 1234)
check("last int none", parse_last_int("no numbers here"), None)
check("gsm correct", gsm8k_correct("reasoning... 42", "blah #### 42"), True)
check("gsm wrong", gsm8k_correct("reasoning... 41", "blah #### 42"), False)
check("gsm no int", gsm8k_correct("no answer", "blah #### 42"), False)

# --- MMLU letter parsing ---
check("mmlu letter", parse_mmlu_letter("Answer: B"), "B")
check("mmlu letter sentence", parse_mmlu_letter("The answer is C."), "C")
check("mmlu letter none", parse_mmlu_letter("I do not know"), None)
check("mmlu correct", mmlu_correct("B", "B"), True)
check("mmlu wrong", mmlu_correct("A", "B"), False)

# --- SFT formatting ---
ex = format_sft_example("Explain X", "context here", "Because Y.")
check("sft with input", ex,
      "Instruction: Explain X\nInput: context here\nResponse: Because Y.")
ex2 = format_sft_example("Explain X", "", "Because Y.")
check("sft no input", ex2, "Instruction: Explain X\nResponse: Because Y.")
check("sft prompt", format_sft_prompt("Explain X", "ctx"),
      "Instruction: Explain X\nInput: ctx\nResponse:")

# --- MMLU few-shot prompt ---
few = [("What is 2+2?", ["1", "2", "3", "4"], "D")]
p = format_mmlu_prompt("Capital of France?", ["London", "Paris", "Rome", "Madrid"], few)
check("mmlu few-shot contains answer", "Answer: D" in p, True)
check("mmlu target ends with Answer:", p.rstrip().endswith("Answer:"), True)
check("mmlu target choices", "B. Paris" in p, True)

print()
if fails:
    print(f"FAILED {len(fails)} checks")
    for f in fails:
        print(" -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
