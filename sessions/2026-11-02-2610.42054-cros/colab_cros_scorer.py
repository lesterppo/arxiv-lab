"""CRoS frozen-scorer Colab driver -- arXiv 2610.42054.

Tests the CRoS mechanism on a REAL current-generation model, without an
RLVR training run (shrink the EXPERIMENT, never the model):

- Model: try Qwen3.5-9B first (next-to-validate 9B class), fall back to
  Qwen3-8B (proven T4 model) if the 9B load fails. NF4, bf16, inference
  only -- a frozen scorer, so no peft / no torchao-uninstall needed.
- For 12 embedded arithmetic problems (6 chained, 6 order-independent),
  score the gold answer's logprob under three conditions: original CoT
  order, SCRAMBLED CoT order (same steps, reversed -- content exactly
  preserved, as CRoS requires), and no CoT.
- CRoS prediction: scrambling hurts chained CoTs (causal order matters)
  far more than independent CoTs (order irrelevant), i.e.
  mean(LP_orig - LP_scram) chained >> independent, and the scrambled
  condition still beats no-CoT (content preserved, unlike deletion).

Metrics CSV is written incrementally (flushed per row) to
/content/cros_metrics.csv; a summary JSON goes to
/content/cros_summary.json. CROS_DONE is printed as the VERY LAST line,
only on success. No secrets in this file (public ungated models).
"""

import csv
import json
import math
import subprocess
import sys

MODELS = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"]

# (a, b, c) triples shared with the CPU toy problems.
TRIPLES = [(3, 4, 5), (2, 7, 3), (5, 2, 8), (4, 4, 4), (6, 3, 2), (7, 1, 6)]


def chained_steps(a, b, c):
    gold = (a + b) * c - a
    steps = [
        "Start with %d." % a,
        "Add %d to get %d." % (b, a + b),
        "Multiply by %d to get %d." % (c, (a + b) * c),
        "Subtract the starting value %d to get %d." % (a, gold),
    ]
    return steps, gold


def independent_steps(vals):
    gold = sum(vals)
    steps = ["There are %d red balls." % vals[0],
             "There are %d blue balls." % vals[1],
             "There are %d green balls." % vals[2]]
    return steps, gold


def build_problems():
    problems = []
    for i, (a, b, c) in enumerate(TRIPLES):
        steps, gold = chained_steps(a, b, c)
        problems.append({
            "id": "chain-%d" % i, "kind": "chained",
            "question": ("A number starts at %d. First add %d, then "
                         "multiply by %d, then subtract the starting "
                         "number. What is the final number?" % (a, b, c)),
            "steps": steps, "gold": gold})
    for i, vals in enumerate(TRIPLES):
        steps, gold = independent_steps(vals)
        problems.append({
            "id": "indep-%d" % i, "kind": "independent",
            "question": ("A box holds red, blue, and green balls. How "
                         "many balls are there in total?"),
            "steps": steps, "gold": gold})
    return problems


def score_answer(model, tok, prompt, answer, device):
    """Sum logprob of the answer tokens given the prompt (batch 1)."""
    import torch
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    a_ids = tok(" %d" % answer, add_special_tokens=False,
                return_tensors="pt").input_ids.to(device)
    full = torch.cat([p_ids, a_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    plen = p_ids.shape[1]
    for j in range(a_ids.shape[1]):
        total += logp[plen + j - 1, a_ids[0, j]].item()
    return total


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers", "bitsandbytes", "accelerate"],
                   check=True)
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    print("cuda:", torch.cuda.get_device_name(0),
          "total_gib: %.2f" % (torch.cuda.get_device_properties(0).total_memory
                                / 2**30), flush=True)

    model = None
    model_id = None
    tok = None
    for cand in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                cand, quantization_config=bnb, device_map="auto",
                torch_dtype=torch.bfloat16)
            model.eval()
            model_id = cand
            print("model loaded:", cand, flush=True)
            break
        except Exception as e:  # noqa: BLE001 - fallback to next model
            print("model load failed %s: %r" % (cand, e), flush=True)
            model = None
    if model is None:
        raise RuntimeError("no model loaded from %s" % MODELS)

    def peak_vram():
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)

    csv_path = "/content/cros_metrics.csv"
    fields = ["problem", "kind", "condition", "lp_answer", "peak_vram_gib"]
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    f.flush()

    rows = []
    for prob in build_problems():
        q = prob["question"]
        orig = "\n".join(prob["steps"])
        scram = "\n".join(list(reversed(prob["steps"])))
        conds = [
            ("orig", "Question: %s\nSolution:\n%s\nThe answer is" % (q, orig)),
            ("scrambled", "Question: %s\nSolution:\n%s\nThe answer is" % (q, scram)),
            ("nocot", "Question: %s\nThe answer is" % q),
        ]
        for cond, prompt in conds:
            lp = score_answer(model, tok, prompt, prob["gold"], device)
            r = {"problem": prob["id"], "kind": prob["kind"],
                 "condition": cond, "lp_answer": round(lp, 4),
                 "peak_vram_gib": peak_vram()}
            writer.writerow(r)
            f.flush()
            rows.append(r)
            print("ROW", r, flush=True)

    def mean(kind, cond):
        vals = [r["lp_answer"] for r in rows
                if r["kind"] == kind and r["condition"] == cond]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "model": model_id,
        "n_problems": len(build_problems()),
        "chain_lp_orig": mean("chained", "orig"),
        "chain_lp_scrambled": mean("chained", "scrambled"),
        "chain_lp_nocot": mean("chained", "nocot"),
        "chain_delta": mean("chained", "orig") - mean("chained", "scrambled"),
        "indep_lp_orig": mean("independent", "orig"),
        "indep_lp_scrambled": mean("independent", "scrambled"),
        "indep_lp_nocot": mean("independent", "nocot"),
        "indep_delta": (mean("independent", "orig")
                        - mean("independent", "scrambled")),
        "peak_vram_gib": peak_vram(),
    }
    summary["delta_gap"] = summary["chain_delta"] - summary["indep_delta"]
    summary["cros_prediction_holds"] = bool(summary["delta_gap"] > 0)
    with open("/content/cros_summary.json", "w") as jf:
        json.dump(summary, jf, indent=2)
    print("SUMMARY", json.dumps(summary), flush=True)
    print("CROS_DONE", flush=True)


if __name__ == "__main__":
    main()
