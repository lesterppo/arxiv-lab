#!/usr/bin/env python3
"""
CoDe-LoRA empirical probe (arXiv 2610.08312, training track, 2026-10-10).

Tests the paper's core diagnosis — the "Orthogonality Dilemma" — at 9B
scale with real model gradients, plus the consolidation mechanism's
stability (Prop. 1), on Qwen/Qwen3.5-9B (NF4 4-bit, T4).

Claim under test: strict orthogonal parameter isolation (O-LoRA: project
each new task's update onto the null space of prior tasks' accumulated
subspace) kills knowledge TRANSFER on semantically related tasks, because
the shared directions are nulled along with the conflicting ones. On
unrelated tasks the same projection is nearly harmless. CoDe-LoRA's
consolidation branch instead accumulates shared directions with dynamic
scaling (W_acc^(t) = c_t W_acc^(t-1) + s_t dW_null^(t),
c_t=sqrt((t-1)/t), s_t=sqrt(1/t)), keeping ||W_acc||_F bounded.

Design (bounded: 6 batched fwd+bwd + 12 SVDs):
- 3 tiny tasks x 16 samples: A=arithmetic (format 1), B=arithmetic word
  problems (format 2; RELATED to A: same math structure, new surface
  form), C=Python code Q&A (UNRELATED).
- For layers 0 and n//2, linears o_proj + down_proj: swap the 4-bit bnb
  modules for dequantized bf16 copies (in-place, no deepcopy — cf.
  2026-10-09 LS-LoRA lesson), batched NLL fwd+bwd per task, collect the
  mean update matrix dW per linear.
- Metrics per (layer, linear):
  * subspace overlap (related A<->B vs unrelated A<->C): mean squared
    canonical cosine between top-8 left singular subspaces.
  * O-LoRA transfer cost: fraction of ||dW_B||_F^2 (resp. dW_C) removed
    by null-space projection onto A's top-8 column space (paper Eq. 3).
    Dilemma => cost_related >> cost_unrelated.
  * consolidation stability: apply the paper's recursive consolidation
    A->B->C and report ||W_acc||_F vs the trivial bound M_t.

Outputs: /content/codelora_<ts>.json   Log: /content/codelora.log
"""
import subprocess, sys, os, time, json, math

TS = time.strftime("%Y%m%d-%H%M%S")
LOG = "/content/codelora.log"
RES_PATH = "/content/codelora_%s.json" % TS

def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

open(LOG, "w").write("")
log("pip installs ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/huggingface/transformers.git",
                "bitsandbytes", "accelerate", "numpy"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, capture_output=True)

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bitsandbytes.functional import dequantize_4bit
import bitsandbytes as bnb

log("torch %s, cuda=%s" % (torch.__version__, torch.cuda.is_available()))
assert torch.cuda.is_available(), "no CUDA"

MODEL_ID = "Qwen/Qwen3.5-9B"
MAX_LEN = 192
SEED = 0
DV = 8          # rank of accumulated column space (paper uses dv = r)
PROBE_LINEARS = ("o_proj", "down_proj")
torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------- tiny datasets
# answers hand-checked below (deterministic)
ARITH = [("What is 17 + 25?", "42"), ("What is 9 + 7?", "16"),
         ("What is 144 - 12?", "132"), ("What is 13 + 13?", "26"),
         ("What is 81 - 9?", "72"), ("What is 2 + 10?", "12"),
         ("What is 7 + 8?", "15"), ("What is 100 - 37?", "63"),
         ("What is 15 + 200?", "215"), ("What comes after 13?", "14"),
         ("What is 12 + 18?", "30"), ("What is 5 + 5?", "10"),
         ("What is 48 - 6?", "42"), ("What is 11 + 11?", "22"),
         ("What is 1000 - 999?", "1"), ("What is 86 - 2?", "84")]
WORD = [("Lena has 17 apples and buys 25 more. How many does she have?", "42"),
        ("A box holds 9 red and 7 blue balls. How many balls?", "16"),
        ("Tom had 144 stickers, gave away 12. How many left?", "132"),
        ("Mia counts 13 shells, finds 13 more. Total?", "26"),
        ("A bus has 81 seats, 9 are taken. How many free?", "72"),
        ("Ben scores 2 goals then 10 more. Total goals?", "12"),
        ("There are 7 cats and 8 dogs. How many pets?", "15"),
        ("A jar had 100 coins, 37 were spent. Left?", "63"),
        ("A shelf has 15 books, 200 more arrive. Total?", "215"),
        ("The number after 13 is?", "14"),
        ("12 boys and 18 girls are in class. How many students?", "30"),
        ("5 birds sit, 5 more land. How many birds?", "10"),
        ("48 pages, 6 read. Pages left?", "42"),
        ("11 rows with 11 seats each. Total seats?", "121"),
        ("1000 meters minus 999 meters equals?", "1"),
        ("86 minus 2 equals?", "84")]
CODE = [("Write a Python function that returns the square of x.",
         "def square(x):\n    return x * x"),
        ("Write a Python function that adds two numbers.",
         "def add(a, b):\n    return a + b"),
        ("Write a Python function that checks if a number is even.",
         "def is_even(n):\n    return n % 2 == 0"),
        ("Write a Python loop that prints numbers 0 to 4.",
         "for i in range(5):\n    print(i)"),
        ("Write a Python function that returns the max of a list.",
         "def list_max(xs):\n    return max(xs)"),
        ("Write a Python function that reverses a string.",
         "def reverse(s):\n    return s[::-1]"),
        ("Write a Python function that counts vowels in a string.",
         "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s.lower())"),
        ("What does 'def' do in Python?", "It defines a function."),
        ("What does 'len()' return for an empty list?", "0"),
        ("Which keyword starts a loop over a sequence in Python?", "for"),
        ("What is a Python dictionary used for?", "Storing key-value pairs."),
        ("What does 'import' do in Python?", "It loads a module."),
        ("What is the output of print(2 ** 3)?", "8"),
        ("What type is the result of 7 / 2 in Python 3?", "float"),
        ("How do you open a file for reading in Python?", "open('f.txt', 'r')"),
        ("What does 'return' do in a function?", "It sends a value back to the caller.")]
TASKS = {"A_arith": ARITH, "B_word": WORD, "C_code": CODE}

# ------------------------------------------------------- model load
log("loading %s (NF4) ..." % MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_cfg, device_map="auto",
    torch_dtype=torch.bfloat16, trust_remote_code=False)
model.config.use_cache = False
model.gradient_checkpointing_enable()  # cut activation memory for fwd+bwd
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
n_layers = model.config.num_hidden_layers
# Qwen3.5 is hybrid: some layers are linear-attention (no o_proj). Probe
# only true attention layers so o_proj exists everywhere we measure.
attn_layers = [i for i, lyr in enumerate(model.model.layers)
               if any(n.split(".")[-1] == "o_proj"
                      for n, _ in lyr.named_modules())]
assert len(attn_layers) >= 2, "need >=2 attention layers, found %d" % len(attn_layers)
LAYERS = [attn_layers[0], attn_layers[len(attn_layers) // 2]]
log("loaded: %d layers (%d attention), probing %s, VRAM=%.2f GB" %
    (n_layers, len(attn_layers), LAYERS, torch.cuda.memory_allocated() / 1e9))

def encode_pair(prompt, answer):
    p_ids = tok("Q: %s\nA:" % prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(" " + answer + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:MAX_LEN]
    labels = ([-100] * min(len(p_ids), MAX_LEN)
              + a_ids[:max(0, MAX_LEN - len(p_ids))])[:len(ids)]
    return ids, labels

ENC = {t: [encode_pair(p, a) for p, a in pairs] for t, pairs in TASKS.items()}

def batch_tensors(tname, idxs):
    ids = [ENC[tname][i][0] for i in idxs]; lbs = [ENC[tname][i][1] for i in idxs]
    L = max(len(x) for x in ids); pad = tok.pad_token_id
    ii = torch.tensor([x + [pad] * (L - len(x)) for x in ids])
    ll = torch.tensor([x + [-100] * (L - len(x)) for x in lbs])
    attn = (ii != pad).long()
    return ii.cuda(), ll.cuda(), attn.cuda()

MICRO = 2  # micro-batch size: keeps logits (b*seq*vocab fp32) in budget

# ------------------------------------------------------- per-(layer,task) mean update
def mean_update(layer_idx, tname):
    """Swap the probed layer's bnb linears for dequantized bf16 copies
    (in-place, no deepcopy), run micro-batched fwd+bwd accumulating the
    mean NLL gradient per task, return
    {linear_name: dW (d_out x d_in, float32, on GPU)}."""
    import torch.nn as nn
    layer = model.model.layers[layer_idx]
    saved = {}
    swapped = {}
    for name, mod in list(layer.named_modules()):
        short = name.split(".")[-1]
        if short in PROBE_LINEARS and isinstance(mod, bnb.nn.Linear4bit):
            w = dequantize_4bit(mod.weight.data, mod.weight.quant_state).to(torch.bfloat16)
            lin = nn.Linear(mod.in_features, mod.out_features, bias=False,
                            dtype=torch.bfloat16).to(mod.weight.device)
            lin.weight.data.copy_(w)
            lin.weight.requires_grad_(True)
            parent = layer
            for part in name.split(".")[:-1]:
                parent = getattr(parent, part)
            setattr(parent, short, lin)
            saved[name] = mod
            swapped[name] = lin
    n = len(ENC[tname])
    n_chunks = math.ceil(n / MICRO)
    for b in range(0, n, MICRO):
        idxs = list(range(b, min(b + MICRO, n)))
        ii, ll, attn = batch_tensors(tname, idxs)
        out = model(input_ids=ii, attention_mask=attn, labels=ll)
        (out.loss / n_chunks).backward()   # accumulate mean gradient
        del ii, ll, attn, out
        torch.cuda.empty_cache()
    grads = {}
    for name, lin in swapped.items():
        short = name.split(".")[-1]
        grads[short] = lin.weight.grad.detach().to(torch.float32).cuda()
        lin.weight.grad = None
    # restore 4-bit modules in place
    for name, mod in saved.items():
        parent = layer
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        setattr(parent, name.split(".")[-1], mod)
    torch.cuda.empty_cache()
    return grads

DW = {}  # (layer, task, linear) -> dW
for li in LAYERS:
    for tname in TASKS:
        log("fwd+bwd layer=%d task=%s ..." % (li, tname))
        g = mean_update(li, tname)
        for lin, dW in g.items():
            DW[(li, tname, lin)] = dW
            log("  %s: ||dW||_F=%.4f shape=%s" % (lin, dW.norm().item(), tuple(dW.shape)))

# ------------------------------------------------------- metrics
def topU(dW, k):
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    return U[:, :k]

def overlap(U1, U2):
    # mean squared canonical cosine between two k-dim subspaces
    return (torch.linalg.svdvals(U1.T @ U2) ** 2).mean().item()

def null_cost(dW_new, dW_old, k=DV):
    # fraction of ||dW_new||_F^2 removed by projecting onto null space of
    # dW_old's top-k column space (paper Eq. 3)
    U = topU(dW_old, k)
    proj = U @ (U.T @ dW_new)
    dW_null = dW_new - proj
    return 1.0 - (dW_null.norm() ** 2 / dW_new.norm().clamp_min(1e-12) ** 2).item()

results = {"model": MODEL_ID, "dv": DV, "layers": LAYERS,
           "linears": list(PROBE_LINEARS), "pairs": {}}
for li in LAYERS:
    for lin in PROBE_LINEARS:
        dA = DW[(li, "A_arith", lin)]; dB = DW[(li, "B_word", lin)]; dC = DW[(li, "C_code", lin)]
        UA, UB, UC = topU(dA, DV), topU(dB, DV), topU(dC, DV)
        ov_rel = overlap(UA, UB); ov_unr = overlap(UA, UC)
        cost_rel = null_cost(dB, dA); cost_unr = null_cost(dC, dA)
        # consolidation stability: A -> B -> C with paper's dynamic scaling
        norms = []
        W_acc = torch.zeros_like(dA)
        for t, dW in enumerate([dA, dB, dC], start=1):
            if t == 1:
                W_acc = dW.clone()
            else:
                ct = math.sqrt((t - 1) / t); st = math.sqrt(1.0 / t)
                U = topU(W_acc, DV)
                dW_null = dW - U @ (U.T @ dW)
                W_acc = ct * W_acc + st * dW_null
            norms.append(W_acc.norm().item())
        Mt = max(d.norm().item() for d in (dA, dB, dC))
        key = "L%d.%s" % (li, lin)
        results["pairs"][key] = {
            "overlap_related_AB": round(ov_rel, 4),
            "overlap_unrelated_AC": round(ov_unr, 4),
            "olorA_transfer_cost_related": round(cost_rel, 4),
            "olorA_transfer_cost_unrelated": round(cost_unr, 4),
            "consolidation_norms": [round(v, 4) for v in norms],
            "Mt_bound": round(Mt, 4),
        }
        log("%s: overlap rel=%.4f unr=%.4f | null-cost rel=%.4f unr=%.4f | "
            "||Wacc||=%s Mt=%.4f" %
            (key, ov_rel, ov_unr, cost_rel, cost_unr,
             [round(v, 2) for v in norms], Mt))

with open(RES_PATH, "w") as f:
    json.dump(results, f, indent=1)
log("wrote %s" % RES_PATH)
print("CODELORA_DONE")
