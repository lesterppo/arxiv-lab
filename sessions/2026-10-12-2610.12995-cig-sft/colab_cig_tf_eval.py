#!/usr/bin/env python3
"""Teacher-forced diagnostic for CIG-SFT run.

The free-generation eval failed for both arms (format mismatch), so this
measures what the adapters actually learned: per-token teacher-forced
accuracy on test completions, split into preamble vs substance tokens.
If CIG-SFT works as claimed, the cig adapter should beat plain on
substance (task-informative) tokens.
"""
import os, sys, json, random

def log(m):
    print(m, flush=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "/content/qwen35-9b"
SEED = 7
PREAMBLE = "I'll solve this step by step."

def make_sample(rng, template):
    if template == 2:
        a, b, c, d = (rng.randint(10, 50) for _ in range(4))
        prompt = f"Solve: {a} * {b} + {c} * {d}"
        s1 = a * b; s2 = c * d; ans = s1 + s2
        steps = (f"Step 1: {a} * {b} = {s1}\nStep 2: {c} * {d} = {s2}\n"
                 f"Step 3: {s1} + {s2} = {ans}")
    else:
        a = rng.randint(20, 50); b = rng.randint(1, 19)
        c = rng.randint(20, 50); d = rng.randint(1, 19)
        prompt = f"Solve: ({a} + {b}) * ({c} - {d})"
        s1 = a + b; s2 = c - d; ans = s1 * s2
        steps = (f"Step 1: {a} + {b} = {s1}\nStep 2: {c} - {d} = {s2}\n"
                 f"Step 3: {s1} * {s2} = {ans}")
    return prompt, f"{PREAMBLE}\n{steps}\nAnswer: {ans}\n", ans

rng = random.Random(SEED + 1)
test = [make_sample(rng, [2, 3][i % 2]) for i in range(100)]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16)
base.eval()
n_pre = len(tok(PREAMBLE, add_special_tokens=False)["input_ids"])
log(f"preamble tokens: {n_pre}")

@torch.no_grad()
def tf_acc(model, split):
    pre_ok = pre_n = sub_ok = sub_n = 0
    for prompt, completion, ans in split:
        ids_p = tok(prompt, add_special_tokens=False)["input_ids"]
        ids_nl = tok("\n", add_special_tokens=False)["input_ids"]
        ids_c = tok(completion, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([ids_p + ids_nl + ids_c]).to(base.device)
        n_p, n_nl, Lc = len(ids_p), len(ids_nl), len(ids_c)
        out = model(input_ids=ids)
        pred = out.logits[0].argmax(-1)
        for j in range(1, Lc):
            pos = n_p + n_nl + j
            if pos >= ids.shape[1]:
                break
            ok = int(pred[pos - 1] == ids[0, pos])
            if j < n_pre:
                pre_ok += ok; pre_n += 1
            else:
                sub_ok += ok; sub_n += 1
    return {"preamble_acc": pre_ok / max(1, pre_n),
            "substance_acc": sub_ok / max(1, sub_n),
            "overall_acc": (pre_ok + sub_ok) / max(1, pre_n + sub_n)}

out = {}
for name, adir in [("base_none", None),
                   ("plain", "/content/cig_adapter_plain/plain"),
                   ("cig", "/content/cig_adapter_cig/cig")]:
    if adir is None:
        m = base
    else:
        m = PeftModel.from_pretrained(base, adir)
    m.eval()
    r = tf_acc(m, test)
    out[name] = {k: round(v, 4) for k, v in r.items()}
    log(f"{name}: {out[name]}")
    if adir is not None:
        del m
        torch.cuda.empty_cache()

with open("/content/cig_tf_results.json", "w") as f:
    json.dump(out, f, indent=1)
log("wrote /content/cig_tf_results.json")
print(json.dumps(out, indent=1))
