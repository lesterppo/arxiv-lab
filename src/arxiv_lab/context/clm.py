"""arXiv 2609.37725 — Context Language Models.

Core claim implemented here: treat the agent's live context as an editable
file. The model may make unrestricted in-place updates to that file — delete
lines, rewrite spans, append notes — via ordinary edit operations, and every
edit is synchronized to the model's next-turn input. Context management moves
from fixed harness control (append-only + truncation/compaction schedules) to
intrinsic model behavior.

Mechanism (paper Sec. 4.1, from the abstract):
- LiveContextFile: the runtime mirrors the conversation into a
  line-addressed text file. New turns append normally.
- Edit ops: ("delete", start, end), ("replace", start, end, new_text),
  ("append_note", text), ("rewrite", new_text). Batches apply atomically:
  any invalid op aborts the whole batch with no mutation committed.
- sync(): returns the current file content — the model's next-turn input.
  Every mutation bumps a revision; an edit log records what changed.
- Budget accounting uses estimate_tokens (chars // 4), the same convention
  as the other context modules in this package.

What this module does NOT implement (honest limits):
- Suffix Cache Reuse (the paper's SGLang serving patch) — a serving-layer
  optimization, not testable on CPU.
- The paper's online RL / skill-optimization training loops — those need GPU
  training; here the edit policy is scripted, standing in for the model.
- Multi-agent shared files — the single-file case generalizes directly.

The experiment in tests/test_context_clm_cpu.py replays the paper's
experimental design in simulation: a long-horizon tool-use trajectory where
answer-critical facts are scattered among failed/duplicate/verbose tool
outputs, under a tight context budget. A scripted edit policy mirroring the
paper's observed emergent strategies (prune failures/duplicates, compact
verbose outputs preserving refs, maintain a findings table) is compared
against the fixed baseline (append-only + head truncation).
"""

TOKENS_PER_CHAR = 1 / 4.0  # deterministic token estimate for accounting


def estimate_tokens(text):
    """Deterministic token estimate for context accounting."""
    return int(len(text) * TOKENS_PER_CHAR)


class EditError(Exception):
    """Invalid edit batch or protocol error (no partial mutation committed)."""


class LiveContextFile(object):
    """The agent's live context, mirrored as an editable line-addressed file.

    The runtime appends new turns; the model edits in place. sync() yields
    the current file content as the next-turn input.
    """

    def __init__(self, budget_tokens):
        self.lines = []
        self.budget_tokens = budget_tokens
        self.revision = 0
        self.edit_log = []  # list of (revision, op, detail)

    # -- runtime side -------------------------------------------------
    def append(self, text):
        """A new turn / tool output lands at the end of the file."""
        new_lines = text.splitlines() or [""]
        self.lines.extend(new_lines)
        self.revision += 1
        self.edit_log.append(
            (self.revision, "append", "%d lines" % len(new_lines)))

    def text(self):
        return "\n".join(self.lines)

    def tokens(self):
        return estimate_tokens(self.text())

    def over_budget(self):
        return self.tokens() > self.budget_tokens

    def sync(self):
        """Current file content -> the model's next-turn input."""
        return self.text()

    # -- model side: unrestricted in-place updates --------------------
    def apply_batch(self, ops):
        """Apply a batch of edit ops atomically.

        Each op is a tuple:
          ("delete", start, end)              -- drop lines [start, end)
          ("replace", start, end, new_text)   -- swap span for new lines
          ("append_note", text)               -- append note lines at end
          ("rewrite", new_text)               -- replace the whole file

        Ranges are 0-based, end-exclusive, validated against the working
        copy as ops apply in order. Any invalid op raises EditError and
        the file is left exactly as before the batch.
        """
        working = list(self.lines)
        applied = []

        def check_range(start, end, op):
            if not isinstance(start, int) or not isinstance(end, int):
                raise EditError("range bounds must be int: %r" % (op,))
            if not (0 <= start <= end <= len(working)):
                raise EditError(
                    "range out of bounds: %r (n=%d)" % (op, len(working)))

        try:
            for op in ops:
                kind = op[0]
                if kind == "delete":
                    _, start, end = op
                    check_range(start, end, op)
                    del working[start:end]
                    applied.append("delete [%d,%d)" % (start, end))
                elif kind == "replace":
                    _, start, end, new_text = op
                    check_range(start, end, op)
                    if not isinstance(new_text, str):
                        raise EditError(
                            "replace needs new_text str: %r" % (op,))
                    working[start:end] = new_text.splitlines() or [""]
                    applied.append("replace [%d,%d)" % (start, end))
                elif kind == "append_note":
                    _, text = op
                    if not isinstance(text, str):
                        raise EditError(
                            "append_note needs text str: %r" % (op,))
                    working.extend(text.splitlines() or [""])
                    applied.append("append_note")
                elif kind == "rewrite":
                    _, text = op
                    if not isinstance(text, str):
                        raise EditError(
                            "rewrite needs text str: %r" % (op,))
                    working = text.splitlines() or [""]
                    applied.append("rewrite")
                else:
                    raise EditError("unknown op: %r" % (kind,))
        except EditError:
            raise
        except Exception as e:  # malformed op tuple etc.
            raise EditError("malformed op: %r (%s)" % (op, e))

        self.lines = working
        self.revision += 1
        self.edit_log.append((self.revision, "edit_batch",
                              "; ".join(applied) or "no-op"))
        return self.revision
