"""Byte-level net-zero prefill budget (Mingbird, arXiv:2610.02001).

Failure form addressed: under cloud-scale agent harnesses, tool prefill
(system prompt + tool schemas) overflows small models' context windows.

The mechanism: count every prefill byte against a fixed budget. If the
uncompressed prefill exceeds it, tool schemas are progressively compressed
(drop examples, then descriptions) until the prefill fits — the marginal
prefill cost stays at/below the fixed budget ("net-zero" relative to the
budgeted baseline). Pure stdlib.
"""

import copy
import json


class PrefillBudget:
    """Enforce a byte budget on the harness prefill (system text + tool
    schemas)."""

    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes

    @staticmethod
    def _schema_bytes(schema: dict, level: int) -> str:
        """level 0 = full, 1 = drop examples, 2 = drop descriptions+examples.

        Works on a deep copy: the caller's schema dicts are never mutated.
        """
        d = copy.deepcopy(schema)
        if level >= 1:
            d.pop("examples", None)
            params = d.get("parameters")
            if isinstance(params, dict):
                params.pop("examples", None)
        if level >= 2:
            d.pop("description", None)
        return json.dumps(d, separators=(",", ":"))

    def build_prefill(self, system_text: str, tool_schemas: list):
        """Return (prefill_text, bytes_used, compressed).

        Never exceeds ``budget_bytes``: falls back to higher compression
        levels, and as a last resort truncates the system text.
        """
        parts = [system_text]
        for s in tool_schemas:
            parts.append(self._schema_bytes(s, 0))
        full = "\n".join(parts).encode("utf-8")
        if len(full) <= self.budget_bytes:
            return full.decode("utf-8"), len(full), False
        # compress: raise compression level for all tools until it fits
        for level in (1, 2):
            parts = [system_text] + [self._schema_bytes(s, level)
                                     for s in tool_schemas]
            blob = "\n".join(parts).encode("utf-8")
            if len(blob) <= self.budget_bytes:
                return blob.decode("utf-8"), len(blob), True
        # last resort: truncate system text (never exceed budget)
        head = system_text.encode("utf-8")[: self.budget_bytes // 2]
        rest = b"\n".join(
            self._schema_bytes(s, 2).encode("utf-8") for s in tool_schemas
        )[: self.budget_bytes - len(head) - 1]
        blob = head + b"\n" + rest
        return blob.decode("utf-8", "ignore"), len(blob), True
