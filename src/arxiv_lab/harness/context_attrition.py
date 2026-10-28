"""Context attrition in an agent harness: fixed-length slicing of tool-result
JSON deletes end-positioned fields unevenly (arXiv:2610.36114).

The paper's claim: an agent harness that cuts each tool result by slicing
its JSON text at a fixed length does not shorten results evenly — it deletes
whatever data sits near the end. In a real kernel-trace fault-diagnosis
agent (720 runs, largest prompt 5.75% of the context window) a third of
tool results (3,827 of 11,189) were cut, every cut result became broken
JSON, one field reached the model in only 4.7% of calls while another
reached it in 99.3%, and diagnosis accuracy matched the early/late split
of which field held the evidence. A controlled experiment found the cut's
cost appears first as extra work (43% more tool calls, yield 0.61 -> 0.31)
because the agent can still reach the answer another way; whether the cut
truly causes the accuracy drop is left open by the authors.

This module reproduces the mechanism deterministically on synthetic
kernel-trace tool results: a naive fixed-length slice policy, an
evidence-preserving field-aware policy at the same byte budget, and a
no-cut control, scored by a deterministic rule reader (strict JSON parse,
tolerant evidence-field salvage, and a targeted re-query fallback) that
stands in for the model. Pure stdlib.
"""

import json
import random

CANONICAL_SEED = 20261028
ROBUSTNESS_SEEDS = (1, 7, 42)

# Byte budget for the harness cut. Set to the 66th percentile (11,900 chars)
# of the pooled generated-result sizes, so ~1/3 of results exceed it,
# matching the paper's 34.2% cut rate (3,827/11,189). The budget is fixed
# before any policy runs; the achieved cut rate is reported, not assumed.
BUDGET_CHARS = 11900

# Context window used only for the "window almost empty" accounting:
# 128k tokens approximated as chars (paper: largest prompt 5.75% of window).
WINDOW_CHARS = 131072

# Max targeted re-queries after a failed first attempt ("another way to
# reach it" in the paper's controlled experiment).
MAX_REQUERIES = 2


def _record(rng, idx, faulty=False):
    """One synthetic kernel-trace record."""
    return {
        "seq": idx,
        "ts": round(1000.0 + idx * 0.37 + rng.random() * 0.1, 3),
        "pid": rng.choice([101, 202, 303, 404, 505]),
        "syscall": rng.choice(["read", "write", "openat", "futex", "poll"]),
        "dur_us": rng.randint(3, 900),
        "faulty": faulty,
    }


def generate_tool_result(rng, evidence_position):
    """One synthetic tool result + ground truth.

    ``evidence_position``: "early" (fault_evidence is the 2nd serialized
    field), "late" (fault_evidence is the last field), or "records_only"
    (the evidence is 128 faulty records at the tail of the records list,
    mirroring the paper's production trace where the shipped cut deletes
    all 128 records of the fault's evidence).

    Returns (result_dict, truth_dict). Dict insertion order is the
    serialization order, which is what the slicing policy sees.
    """
    faulty_pid = rng.choice([101, 202, 303, 404, 505])
    faulty_signal = rng.choice(["SIGSEGV", "SIGBUS", "EIO", "ETIMEDOUT"])
    first_bad = rng.randint(0, 120)
    # records_only needs a genuine 128-record tail: n_records >= 140 so
    # n_records - 128 is a positive tail offset (earlier 96..160 range
    # wrapped negative and flagged nearly every record faulty).
    n_records = rng.randint(140, 168) if evidence_position == "records_only" else rng.randint(96, 160)
    evidence = {
        "faulty_pid": faulty_pid,
        "signal": faulty_signal,
        "first_bad_seq": first_bad,
    }
    truth = dict(evidence)

    filler = "".join(rng.choice("abcdef0123456789") for _ in range(220))
    if evidence_position == "records_only":
        records = [
            _record(rng, i, faulty=(i >= n_records - 128))
            for i in range(n_records)
        ]
        # Ground truth lives only in the tail records.
        truth = {
            "faulty_pid": records[n_records - 128]["pid"],
            "signal": faulty_signal,
            "first_bad_seq": n_records - 128,
        }
        result = {
            "trace_id": "trace-%06d" % rng.randint(0, 999999),
            "started_at": "2026-10-28T00:00:00Z",
            "process_table": filler,
            "records": records,
            "summary": "trace with %d records; fault in tail" % n_records,
        }
        return result, truth

    records = [_record(rng, i) for i in range(n_records)]
    head = {
        "trace_id": "trace-%06d" % rng.randint(0, 999999),
        "started_at": "2026-10-28T00:00:00Z",
        "process_table": filler,
    }
    tail = {
        "records": records,
        "summary": "trace with %d records" % n_records,
    }
    if evidence_position == "early":
        result = {**head, "fault_evidence": evidence, **tail}
    elif evidence_position == "late":
        result = {**head, **tail, "fault_evidence": evidence}
    else:
        raise ValueError("unknown evidence_position: %r" % evidence_position)
    return result, truth


def serialize(result):
    return json.dumps(result, separators=(",", ":"))


# ---------------------------------------------------------------- policies

def naive_slice(text, budget=BUDGET_CHARS):
    """The shipped cut from the paper: slice the JSON text at fixed length."""
    return text[:budget]


def field_aware_truncate(text, budget=BUDGET_CHARS):
    """Evidence-preserving truncation at the same byte budget.

    Parse the JSON, then rebuild it in priority order — trace id and the
    fault_evidence field verbatim, then faulty records (plus neighbours),
    then a truncated summary, then remaining records until the budget is
    met — and re-serialize. The output is always valid JSON of at most
    ``budget`` chars (evidence alone is far smaller than the budget, so
    the priority content always fits). An explicit ``records_omitted``
    count records how much was dropped, so the cut is auditable instead
    of silent.
    """
    obj = json.loads(text)
    out = {}
    used = 2  # "{}"

    def _cost(key, value):
        frag = json.dumps({key: value}, separators=(",", ":"))
        return len(frag) - 2  # strip the braces; commas handled below

    def _fits(key, value):
        extra = _cost(key, value) + (1 if out else 0)
        return used + extra <= budget

    for key in ("trace_id", "started_at", "fault_evidence"):
        if key in obj and _fits(key, obj[key]):
            out[key] = obj[key]
            used += _cost(key, obj[key]) + (1 if len(out) > 1 else 0)
    records = obj.get("records", [])
    faulty_idx = [i for i, r in enumerate(records) if r.get("faulty")]
    kept = []
    omitted = 0
    if faulty_idx:
        # Keep faulty records (and one neighbour each side) first.
        keep_idx = set()
        for i in faulty_idx:
            keep_idx.update((i - 1, i, i + 1))
        keep_idx = sorted(i for i in keep_idx if 0 <= i < len(records))
        kept = [records[i] for i in keep_idx]
        omitted = len(records) - len(kept)
        # If even the faulty records alone exceed the budget, keep a
        # prefix of them: evidence survives, completeness does not.
        while kept and not _fits("records", kept):
            kept.pop()
            omitted += 1
        if kept:
            out["records"] = kept
            used += _cost("records", kept) + 1
    else:
        # No faulty flags: keep a head sample of records up to budget.
        while records and not _fits("records", kept):
            if not kept:
                break
            records = records[:-1]
        kept = list(records)
        while kept and not _fits("records", kept):
            kept.pop()
        omitted = len(obj.get("records", [])) - len(kept)
        if kept:
            out["records"] = kept
            used += _cost("records", kept) + 1
    if omitted:
        if _fits("records_omitted", omitted):
            out["records_omitted"] = omitted
            used += _cost("records_omitted", omitted) + 1
    for key in ("process_table", "summary"):
        if key in obj and isinstance(obj[key], str):
            val = obj[key][:120]
            if _fits(key, val):
                out[key] = val
                used += _cost(key, val) + 1
    rendered = json.dumps(out, separators=(",", ":"))
    assert len(rendered) <= budget, (len(rendered), budget)
    json.loads(rendered)  # always valid JSON
    return rendered


def no_cut(text, budget=BUDGET_CHARS):
    """Control: the full result reaches the reader."""
    return text


POLICIES = {
    "naive_slice": naive_slice,
    "field_aware": field_aware_truncate,
    "no_cut": no_cut,
}


# ------------------------------------------------------------------- reader

def _salvage_evidence(text):
    """Tolerant extraction of the fault_evidence object from (possibly
    truncated) JSON text. Succeeds iff the whole evidence object lies
    before the cut point — the mechanism behind the paper's early/late
    split. Returns the evidence dict or None."""
    marker = '"fault_evidence":'
    pos = text.find(marker)
    if pos < 0:
        return None
    start = text.find("{", pos + len(marker))
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def diagnose(payload_text, truth, evidence_position, allow_salvage=True):
    """Deterministic rule reader standing in for the model.

    Returns (correct, strict_parse_ok, evidence_found). Strict parse
    first; on failure, tolerant salvage (unless ``allow_salvage`` is
    False, which yields the strict-only accuracy); for records_only
    evidence, a strict parse that contains >=1 faulty record counts as
    found.
    """
    try:
        obj = json.loads(payload_text)
        strict_ok = True
    except json.JSONDecodeError:
        strict_ok = False
        obj = None
    if strict_ok:
        if "fault_evidence" in obj:
            ev = obj["fault_evidence"]
            found = (
                ev.get("faulty_pid") == truth.get("faulty_pid")
                and ev.get("signal") == truth.get("signal")
            )
            return found, True, found
        records = obj.get("records", [])
        faulty = [r for r in records if r.get("faulty")]
        if faulty:
            found = faulty[0]["pid"] == truth.get("faulty_pid")
            return found, True, True
        return False, True, False
    if not allow_salvage:
        return False, False, False
    ev = _salvage_evidence(payload_text)
    if ev is not None:
        found = (
            ev.get("faulty_pid") == truth.get("faulty_pid")
            and ev.get("signal") == truth.get("signal")
        )
        return found, False, found
    return False, False, False


def run_task(payload_text, truth, evidence_position):
    """One task through the reader with the targeted re-query fallback.

    A failed first attempt triggers a targeted ``get_fault_evidence``
    call (small, never cut) — the paper's "another way to reach it".
    Returns (first_correct, final_correct, calls, strict_ok, found).
    """
    first, strict_ok, found = diagnose(payload_text, truth, evidence_position)
    calls = 1
    final = first
    requeries = 0
    while not final and requeries < MAX_REQUERIES:
        # Targeted re-query returns the evidence field alone; it always
        # arrives intact, so the reader succeeds on the first re-query.
        requeries += 1
        calls += 1
        final = True
    return first, final, calls, strict_ok, found


# --------------------------------------------------------------- experiment

def run_condition(policy_name, evidence_position, n_tasks, seed):
    """Run one (policy x evidence-position) cell. Returns metrics dict."""
    rng = random.Random(seed)
    policy = POLICIES[policy_name]
    corpus = [generate_tool_result(rng, evidence_position) for _ in range(n_tasks)]
    sizes = [len(serialize(r)) for r, _ in corpus]
    cut = valid = valid_among_cut = survived = 0
    survived_among_cut = first_among_cut = 0
    first_ok = final_ok = strict_ok_count = calls = 0
    max_prompt = 0
    for result, truth in corpus:
        text = serialize(result)
        payload = policy(text)
        was_cut = len(text) > BUDGET_CHARS and policy_name != "no_cut"
        cut += int(was_cut)
        try:
            json.loads(payload)
            is_valid = True
        except json.JSONDecodeError:
            is_valid = False
        valid += int(is_valid)
        if was_cut:
            valid_among_cut += int(is_valid)
        # Evidence survival: the evidence content the reader can use
        # (strict parse or salvage) is present in the payload.
        _, _, found = diagnose(payload, truth, evidence_position)
        survived += int(found)
        first, final, n_calls, _, _ = run_task(payload, truth, evidence_position)
        strict_first, _, _ = diagnose(
            payload, truth, evidence_position, allow_salvage=False
        )
        strict_ok_count += int(strict_first)
        if was_cut:
            survived_among_cut += int(found)
            first_among_cut += int(first)
        first_ok += int(first)
        final_ok += int(final)
        calls += n_calls
        max_prompt = max(max_prompt, len(payload))
    n = float(n_tasks)
    return {
        "policy": policy_name,
        "evidence_position": evidence_position,
        "n": n_tasks,
        "cut_rate": cut / n,
        "json_valid_rate": valid / n,
        "json_valid_among_cut": (valid_among_cut / cut) if cut else 1.0,
        "evidence_survival": survived / n,
        "evidence_survival_among_cut": (survived_among_cut / cut) if cut else 1.0,
        "first_accuracy": first_ok / n,
        "first_accuracy_among_cut": (first_among_cut / cut) if cut else 1.0,
        "strict_accuracy": strict_ok_count / n,
        "final_accuracy": final_ok / n,
        "calls_per_task": calls / n,
        "yield": final_ok / calls if calls else 0.0,
        "max_prompt_fraction_of_window": max_prompt / WINDOW_CHARS,
        "mean_result_chars": sum(sizes) / n,
    }


def run_experiment(n_tasks=300, seed=CANONICAL_SEED):
    """Full factorial: 3 policies x 3 evidence positions. Returns a list
    of metric dicts in fixed order."""
    rows = []
    for position in ("early", "late", "records_only"):
        for policy in ("naive_slice", "field_aware", "no_cut"):
            rows.append(run_condition(policy, position, n_tasks, seed))
    return rows


def format_table(rows):
    lines = [
        "policy       position     cut    valid  valid|cut  surv   1st-acc  fin-acc  calls  yield",
    ]
    for r in rows:
        lines.append(
            "%-12s %-12s %.3f  %.3f  %.3f      %.3f  %.3f    %.3f    %.2f   %.3f"
            % (
                r["policy"],
                r["evidence_position"],
                r["cut_rate"],
                r["json_valid_rate"],
                r["json_valid_among_cut"],
                r["evidence_survival"],
                r["first_accuracy"],
                r["final_accuracy"],
                r["calls_per_task"],
                r["yield"],
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for row in run_experiment():
        print(row)
    print(format_table(run_experiment()))
