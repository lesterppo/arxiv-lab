"""CPU tests for the SynthSelf session tag language (arXiv:2610.13113).

Tests the ground-truth renderer, prompt generator, and prompt builders
in sessions/2026-10-12-2610.13113-synthself/taglang.py — the task logic
the Colab QLoRA run trains against. No model weights or GPU needed.

Run:  python3 sessions/2026-10-12-2610.13113-synthself/test_synthself_cpu.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taglang import (INSTR, VOCAB, few_shot_block, gen_prompt, random_prompt,
                     render_tags)


def test_render_single_tags():
    assert render_tags("<up>hello</up>") == "HELLO"
    assert render_tags("<lo>HeLLo</lo>") == "hello"
    assert render_tags("<rev>abc</rev>") == "cba"
    assert render_tags("plain text") == "plain text"
    print("  single tags ok")


def test_render_nested_innermost_first():
    # inner <lo> applies before outer <up>
    assert render_tags("Say <up>hi <lo>YO</lo></up>!") == "Say HI YO!"
    assert render_tags("<up>a<rev>bc</rev>d</up>") == "ACBD"
    assert render_tags("<rev><up>ab</up></rev>") == "BA"
    print("  nesting ok")


def test_render_multiple_spans():
    s = "<up>aa</up> mid <lo>BB</lo>"
    assert render_tags(s) == "AA mid bb"
    print("  multi-span ok")


def test_random_prompt_renders_clean():
    rng = random.Random(7)
    for _ in range(200):
        p = random_prompt(rng)
        assert "Convert:" not in p
        out = render_tags(p)
        assert "<up>" not in out and "<lo>" not in out and "<rev>" not in out, \
            "no tag may survive rendering: %r -> %r" % (p, out)
        assert "</" not in out
    print("  200 random prompts render cleanly")


def test_random_prompt_is_deterministic():
    a = [random_prompt(random.Random(3)) for _ in range(10)]
    b = [random_prompt(random.Random(3)) for _ in range(10)]
    assert a == b
    print("  determinism ok")


def test_prompt_builders():
    demos = [("Convert: <up>x</up>", "X")]
    gp = gen_prompt("Convert: <lo>Y</lo>", demos)
    assert gp.startswith(INSTR)
    assert "Input: Convert: <up>x</up>\nOutput: X" in gp
    assert gp.rstrip().endswith("Input: Convert: <lo>Y</lo>\nOutput:")
    blk = few_shot_block([])
    assert blk == INSTR + "\n"
    print("  prompt builders ok")


def test_vocab_coverage():
    # every vocab word should be reachable by the generator
    rng = random.Random(11)
    seen = set()
    for _ in range(300):
        for w in random_prompt(rng).split():
            if w in VOCAB:
                seen.add(w)
    assert seen == set(VOCAB), "missing: %s" % (set(VOCAB) - seen)
    print("  vocab coverage ok")


def main():
    test_render_single_tags()
    test_render_nested_innermost_first()
    test_render_multiple_spans()
    test_random_prompt_renders_clean()
    test_random_prompt_is_deterministic()
    test_prompt_builders()
    test_vocab_coverage()
    print("ALL SYNTHSELF CPU TESTS PASSED")


if __name__ == "__main__":
    main()
