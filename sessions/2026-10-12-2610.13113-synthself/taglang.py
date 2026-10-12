"""Pure-stdlib tag-language task used by the SynthSelf Colab driver.

The tag language (<up>/<lo>/<rev>, nested, innermost-first) is the
*new capability* the base model must acquire in
sessions/2026-10-12-2610.13113-synthself (arXiv:2610.13113). This module
holds the ground-truth renderer, the random prompt generator, and the
prompt builders, so the task logic is CPU-testable without a GPU or
model weights. colab_synthself.py carries its own copy to stay a
single-file VM driver.
"""

import random
import re

VOCAB = ["alpha", "bravo", "cargo", "delta", "ember", "frost", "glimmer",
         "harbor", "ivory", "jungle", "karma", "lumen", "meadow", "north",
         "onyx", "prism", "quartz", "river", "stone", "tiger"]

INSTR = ("You convert tag-language strings. Rules: <up>X</up> makes X "
         "UPPERCASE, <lo>X</lo> makes X lowercase, <rev>X</rev> reverses X. "
         "Apply innermost tags first. Reply with ONLY the converted string, "
         "no explanation.")


def render_tags(s):
    """Ground-truth renderer (innermost tags first)."""
    pat = re.compile(r"<(up|lo|rev)>([^<>]*)</\1>")

    def repl(m):
        tag, inner = m.group(1), m.group(2)
        if tag == "up":
            return inner.upper()
        if tag == "lo":
            return inner.lower()
        return inner[::-1]

    prev = None
    while prev != s:
        prev = s
        s = pat.sub(repl, s)
    return s


def random_prompt(rng):
    """Random tag-language input (no 'Convert: ' prefix)."""
    n_spans = rng.randint(1, 3)
    parts = []
    for _ in range(n_spans):
        depth = rng.randint(1, 3)
        inner = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 3)))
        for _ in range(depth):
            tag = rng.choice(["up", "lo", "rev"])
            inner = "<%s>%s</%s>" % (tag, inner, tag)
        parts.append(inner)
    filler = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(0, 2)))
    return " ".join(parts + ([filler] if filler else [])).strip()


def few_shot_block(examples):
    blk = INSTR + "\n"
    for inp, out in examples:
        blk += "Input: %s\nOutput: %s\n" % (inp, out)
    return blk


def gen_prompt(inp, demos):
    return few_shot_block(demos) + "Input: %s\nOutput:" % inp
