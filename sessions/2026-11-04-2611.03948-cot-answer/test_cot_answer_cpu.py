"""CPU test for the CoT/answer driver (arXiv 2611.03948) pure logic:
segment-weight masks, answer parsing, CoT cleaning. No model, no
network, no GPU: the driver module is imported without running main().
"""

import importlib.util
import os
import sys

DRIVER = os.path.join(os.path.dirname(__file__),
                      "colab_cot_answer_driver.py")
spec = importlib.util.spec_from_file_location("cot_answer_driver", DRIVER)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class FakeTok:
    """Whitespace tokenizer stub: only the segment boundaries matter."""

    def __call__(self, text, add_special_tokens=False):
        ids = [abs(hash(t)) % 100000 for t in text.split()]
        return type("Enc", (), {"input_ids": ids})()


def main():
    tok = FakeTok()
    prompt = "Question: Q\nA. 1\nB. 2\nC. 3\nD. 4\nReasoning:"
    cot = "the patient has fever and rash"
    ids, labels, weights = driver.build_example(tok, prompt, cot, "B",
                                                0.5, 1.5)
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    n_cot = len(tok(" " + cot, add_special_tokens=False).input_ids)
    n_ans = len(tok("\nAnswer: B", add_special_tokens=False).input_ids)
    assert len(ids) == n_prompt + n_cot + n_ans
    assert labels[:n_prompt] == [-100] * n_prompt
    assert weights[:n_prompt] == [0.0] * n_prompt
    assert weights[n_prompt:n_prompt + n_cot] == [0.5] * n_cot
    assert weights[n_prompt + n_cot:] == [1.5] * n_ans
    assert n_ans == 2  # "Answer:" + "B" under whitespace tokenization
    # UaS: every completion token weight 1.0.
    _, _, w_uas = driver.build_example(tok, prompt, cot, "B", 1.0, 1.0)
    assert w_uas[n_prompt:] == [1.0] * (n_cot + n_ans)
    # SwS vs UaS differ only in the segment weights, never the tokens.
    ids_u, _, _ = driver.build_example(tok, prompt, cot, "B", 1.0, 1.0)
    assert ids_u == ids
    # Answer parsing: last match wins, case-insensitive, A-D only.
    assert driver.parse_answer("Reasoning...\nAnswer: B") == "B"
    assert driver.parse_answer("answer: a then Answer: C") == "C"
    assert driver.parse_answer("no answer here") is None
    assert driver.parse_answer("Answer: E") is None
    assert driver.parse_answer("The answer: d.") == "D"
    # CoT cleaning: collapse whitespace, truncate to 600 chars.
    assert driver.clean_cot("  a   b\nc  ") == "a b c"
    assert len(driver.clean_cot("x" * 1000)) == 600
    assert driver.clean_cot("") == ""
    # Prompt format carries the question + all four options.
    item = {"question": "Q?", "opa": "1", "opb": "2", "opc": "3", "opd": "4"}
    p = driver.format_prompt(item)
    assert "Q?" in p and "A. 1" in p and "D. 4" in p
    print("CoT/answer CPU test PASSED")
    print("segment masks: prompt -100/0.0, CoT 0.5, answer 1.5; UaS all 1.0")
    print("parse_answer: last-match, case-insensitive, A-D only")


if __name__ == "__main__":
    main()
