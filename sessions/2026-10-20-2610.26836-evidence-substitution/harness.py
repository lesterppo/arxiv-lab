"""Retrieval-vs-inline harness for the evidence-substitution replication.

Three conditions (mirroring arXiv 2610.26836):
  A inline    — full simulator report in context; agent answers directly.
  B retrieval — report kept in a file; agent has list_sections/read_section
                plus run_simulation (its own recompute, like re-running PHREEQC).
  C named     — same as B, but the prompt names the authoritative output and
                instructs the agent to read it before answering (the paper's fix).

`run_condition(item, condition, call_model)` drives the loop; `call_model`
is injected so the same harness runs against a scripted oracle (CPU tests)
or a real model (NIM). Failure classification is in `classify()`.
"""
import json
import re

import aqsim

SECTIONS = ["INPUTS", "SPECIATION", "SATURATION", "MINERAL_MASSES"]
MAX_STEPS = 6

SYS_BASE = ("You are a geochemistry assistant. Answer the multiple-choice "
            "question with exactly one option letter wrapped as <answer>X</answer>.")
SYS_B = (SYS_BASE + " A simulator report is saved at report.txt. You have "
         "tools: list_sections(), read_section(name), run_simulation(params_json). "
         "Use them as needed, then answer.")
SYS_C = (SYS_BASE + " THE AUTHORITATIVE SOURCE for numeric values is the "
         "simulator report at report.txt. Always read the relevant report "
         "section before answering. Do NOT answer from your own recomputed "
         "values; your own simulation outputs are not authoritative.")
SYS_D = (SYS_BASE + " You previously ran your own simulation; its output is "
         "in own_result.txt. A simulator report is also saved at report.txt. "
         "You have tools: list_sections(), read_section(name), "
         "read_own_result(), run_simulation(params_json). Use them as "
         "needed, then answer.")


def _tools_for(item, log):
    report = item["report"]

    def list_sections():
        log.append(("list_sections", "", "sections listed"))
        return "sections: " + ", ".join(SECTIONS)

    def read_section(name):
        name = name.strip().upper()
        txt = aqsim.get_section(report, name) if name in SECTIONS else ""
        log.append(("read_section", name, txt[:60]))
        return txt if txt else f"unknown section {name!r}"

    def run_simulation(params_json):
        log.append(("run_simulation", (params_json or "")[:40], ""))
        try:
            params = json.loads(params_json)
        except Exception:
            params = {}
        merged = dict(item["params"])
        if isinstance(params, dict):
            merged.update(params)
        sim = aqsim.simulate(merged, item["seed"])
        sis = ", ".join(f"{m}={sim['si'][m]:.6f}" for m in aqsim.MINERALS)
        log[-1] = ("run_simulation", (params_json or "")[:40], sis[:60])
        return ("simulation complete, own output written to own_result.txt. "
                f"SI values: {sis}")

    def read_own_result():
        sim = aqsim.simulate(item["params"], item["seed"])
        sis = ", ".join(f"{m}={sim['si'][m]:.6f}" for m in aqsim.MINERALS)
        log.append(("read_own_result", "", sis[:60]))
        return f"own_result.txt:\nSI values: {sis}"

    return {"list_sections": list_sections,
            "read_section": read_section,
            "run_simulation": run_simulation,
            "read_own_result": read_own_result}


def _tool_schema():
    return [
        {"name": "list_sections", "args": []},
        {"name": "read_section", "args": ["name"]},
        {"name": "run_simulation", "args": ["params_json"]},
        {"name": "read_own_result", "args": []},
    ]


def _extract_answer(text):
    m = re.search(r"<answer>\s*([A-D])\s*</answer>", text or "")
    if m:
        return m.group(1)
    m = re.search(r"\banswer\D{0,10}([A-D])\b", text or "", re.I)
    return m.group(1).upper() if m else None


def run_condition(item, condition, call_model):
    """Run one item under one condition. Returns a result dict."""
    log = []
    transcript_values = []  # every numeric value the agent demonstrably saw
    if condition == "A":
        messages = [
            {"role": "system", "content": SYS_BASE},
            {"role": "user",
             "content": f"Simulator report:\n{item['report']}\n\n{item['question']}"},
        ]
        resp = call_model(messages, tools=None)
        transcript_values.append(item["report"])
        final = resp[0] if isinstance(resp, tuple) else resp
    else:
        sys = {"B": SYS_B, "C": SYS_C, "D": SYS_D}[condition]
        messages = [{"role": "system", "content": sys},
                    {"role": "user", "content": item["question"]}]
        tools = _tools_for(item, log)
        if condition == "D":
            # pre-seed: the agent's own prior simulation output, as in the
            # paper where agents had been running PHREEQC themselves
            own = tools["read_own_result"]()
            transcript_values.append(own)
            messages.append({"role": "user",
                             "content": "Your earlier simulation run produced "
                                        f"this own output:\n{own}"})
        final = None
        for _ in range(MAX_STEPS):
            resp = call_model(messages, tools=_tool_schema())
            text, tcalls = resp
            messages.append({"role": "assistant", "content": text})
            if not tcalls:
                final = text
                break
            for tc in tcalls:
                name, args = tc["name"], tc.get("args", {})
                try:
                    out = tools[name](**args)
                except Exception as e:  # never let a tool crash the loop
                    out = f"tool error: {e}"
                if name == "read_section":
                    transcript_values.append(out)
                if name == "run_simulation":
                    transcript_values.append(out)
                messages.append({"role": "tool", "name": name,
                                 "tool_call_id": tc.get("id"),
                                 "content": out})
        else:
            final = call_model(messages + [
                {"role": "user",
                 "content": "Now output only <answer>X</answer>."}],
                tools=None)
            if isinstance(final, tuple):
                final = final[0]
    pred = _extract_answer(final)
    return {"seed": item["seed"], "condition": condition,
            "pred": pred, "answer": item["answer"],
            "correct": pred == item["answer"],
            "keyed_value": item["keyed_value"],
            "tool_log": [(n, a) for n, a, _ in log],
            "saw_keyed": any(f"{item['keyed_value']:.3f}" in t or
                             f"{item['true_si']:.6f}" in t
                             for t in transcript_values),
            "ran_simulation": any(n == "run_simulation" for n, _ in
                                  [(n, a) for n, a, _ in log]),
            "read_report": any(n == "read_section" for n, _ in
                               [(n, a) for n, a, _ in log]),
            "transcript_tail": final[-600:],
            }


def classify(res, item):
    """Classify a wrong answer. Returns one of:
    substitution (computed/saw the right value, answered a neighbor),
    retrieval_failure (never saw the keyed value), other."""
    if res["correct"]:
        return "correct"
    if not res["saw_keyed"]:
        return "retrieval_failure"
    # saw the right value but picked a neighboring option
    pred_val = dict(item["options"]).get(res["pred"])
    if pred_val is not None and abs(pred_val - item["keyed_value"]) <= 0.0011:
        return "substitution"
    return "other"
