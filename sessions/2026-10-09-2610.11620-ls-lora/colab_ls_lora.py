#!/usr/bin/env python3
"""
LS-LoRA proxy verification (arXiv 2610.11620, training track, 2026-10-09).

Core claim tested: input-output cosine similarity (forward-only) is a
lightweight proxy for layer sensitivity -- "layers with lower input-output
similarity consistently exhibit higher empirical Fisher scores".

Test: on Qwen/Qwen3.5-9B (NF4 4-bit, T4), compute per-layer mean input->output
cosine similarity (forward-only) and per-layer empirical Fisher diagonal trace
(grad of NLL wrt that layer's params; 4-bit linears are temporarily swapped
for dequantized bf16 copies one layer at a time since frozen 4-bit weights
admit no param grads), on the same tiny instruction dataset. Report
Spearman rho between io_cosine and fisher_trace (expect negative), plus the
top/bottom-quartile overlap used by LS-LoRA's actual selection rule.

Outputs: /content/ls_lora_<ts>.csv, /content/ls_lora_results.json
Log:     /content/ls_lora.log   Status: /content/ls_lora_status.json
"""

import subprocess, sys, os, time, json, math, csv, copy

TS = time.strftime("%Y%m%d-%H%M%S")
LOG = "/content/ls_lora.log"
STATUS = "/content/ls_lora_status.json"
CSV_PATH = f"/content/ls_lora_{TS}.csv"
RES_PATH = "/content/ls_lora_results.json"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def set_status(**kw):
    try:
        with open(STATUS, "w") as f:
            json.dump({"ts": time.time(), "ts_h": time.strftime("%H:%M:%S"), **kw}, f)
    except Exception:
        pass

set_status(stage="starting")
open(LOG, "w").write("")

# ---------------------------------------------------------------- installs
# NOTE: Qwen3.5 needs transformers git main (model_type `qwen3_5` is not in
# any PyPI release as of 2026-10-09, incl. 4.57.6). peft is not used here, so
# the usual <5.0 pin does not apply.
log("pip installs ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/huggingface/transformers.git",
                "bitsandbytes", "accelerate", "scipy", "numpy"], check=True)
log("uninstall torchao (Colab 0.10 breaks peft-style flows) ...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, capture_output=True)
set_status(stage="installed")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bitsandbytes.functional import dequantize_4bit
import bitsandbytes as bnb
from scipy.stats import spearmanr

log(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "no CUDA"

MODEL_ID = "Qwen/Qwen3.5-9B"
N_COS_SAMPLES = 48     # forward-only io-cosine samples
N_FISHER_SAMPLES = 16  # per-sample NLL backprop samples (subset)
MAX_LEN = 192
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------- tiny dataset
# Deterministic, hardcoded: short math/code instruction Q&A.
PROMPTS = [
    ("What is 17 + 25?", "42"),
    ("What is 9 * 7?", "63"),
    ("What is 144 / 12?", "12"),
    ("What is 13 squared?", "169"),
    ("What is the square root of 81?", "9"),
    ("What is 2 to the power of 10?", "1024"),
    ("What is 7 * 8 + 3?", "59"),
    ("What is 100 - 37?", "63"),
    ("What is 15% of 200?", "30"),
    ("What is the next prime after 13?", "17"),
    ("How many sides does a hexagon have?", "6"),
    ("What is 3 factorial?", "6"),
    ("What is the GCD of 12 and 18?", "6"),
    ("What is 5 cubed?", "125"),
    ("What is 48 / 6 * 2?", "16"),
    ("What is the 5th Fibonacci number?", "5"),
    ("What is 11 * 11?", "121"),
    ("What is 1000 - 999?", "1"),
    ("What is half of 86?", "43"),
    ("What is 2 * 3 * 4?", "24"),
    ("Write a Python function that returns the square of x.",
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
    ("Write a Python function that returns the factorial of n.",
     "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"),
    ("Write a Python one-liner that squares each element of a list.",
     "[x * x for x in xs]"),
    ("Write a Python function that checks if a string is a palindrome.",
     "def is_palindrome(s):\n    return s == s[::-1]"),
    ("What does 'def' do in Python?", "It defines a function."),
    ("What does 'len()' return for an empty list?", "0"),
    ("Which keyword starts a loop over a sequence in Python?", "for"),
    ("What is a Python dictionary used for?", "Storing key-value pairs."),
    ("What does 'import' do in Python?", "It loads a module."),
    ("What is the output of print(2 ** 3)?", "8"),
    ("What type is the result of 7 / 2 in Python 3?", "float"),
    ("How do you open a file for reading in Python?", "open('f.txt', 'r')"),
    ("What does 'return' do in a function?", "It sends a value back to the caller."),
    ("Which data type is immutable: list or tuple?", "tuple"),
    ("What is 23 + 41 - 12?", "52"),
    ("What is 8 * 9?", "72"),
    ("What is 256 / 16?", "16"),
    ("What is 6 squared?", "36"),
    ("What is the cube root of 27?", "3"),
    ("What is 12 * 12?", "144"),
    ("What is 99 - 45?", "54"),
    ("What is 25% of 80?", "20"),
]
assert len(PROMPTS) >= N_COS_SAMPLES
SAMPLES = PROMPTS[:N_COS_SAMPLES]
FISHER_SAMPLES = SAMPLES[:N_FISHER_SAMPLES]

# ------------------------------------------------------- model load
log(f"loading tokenizer + model {MODEL_ID} (NF4 4-bit) ...")
set_status(stage="loading_model")
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
model.gradient_checkpointing_enable()
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
n_layers = model.config.num_hidden_layers
log(f"loaded: {n_layers} layers, "
    f"VRAM={(torch.cuda.memory_allocated()/1e9):.2f} GB")
set_status(stage="model_loaded", n_layers=n_layers)

# ------------------------------------------------------- tokenize
def encode_pair(prompt, answer):
    p_ids = tok(f"Q: {prompt}\nA:", add_special_tokens=False)["input_ids"]
    a_ids = tok(" " + answer + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:MAX_LEN]
    labels = ([-100] * min(len(p_ids), MAX_LEN)
              + a_ids[:max(0, MAX_LEN - len(p_ids))])
    labels = labels[:len(ids)]
    return ids, labels

ENC = [encode_pair(p, a) for p, a in SAMPLES]
log(f"tokenized {len(ENC)} samples, avg len "
    f"{sum(len(e[0]) for e in ENC)/len(ENC):.1f}")

def batch_tensors(idxs):
    ids = [ENC[i][0] for i in idxs]
    lbs = [ENC[i][1] for i in idxs]
    L = max(len(x) for x in ids)
    pad = tok.pad_token_id
    ii = torch.tensor([x + [pad] * (L - len(x)) for x in ids])
    ll = torch.tensor([x + [-100] * (L - len(x)) for x in lbs])
    attn = (ii != pad).long()
    return ii.cuda(), ll.cuda(), attn.cuda()

# ------------------------------------------------------- phase 1: io cosine
log("phase 1: forward-only input-output cosine per layer ...")
set_status(stage="io_cosine")
layers = model.model.layers
io_sum = torch.zeros(n_layers, dtype=torch.float64)
io_cnt = torch.zeros(n_layers, dtype=torch.float64)
_handles = []
_capt = {}

def _hook(i):
    def fn(mod, inp, out):
        x = inp[0].detach()
        y = out[0].detach() if isinstance(out, tuple) else out.detach()
        # per-token cosine, mean over (batch, seq)
        xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        yn = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        c = (xn * yn).sum(-1).double().mean()
        io_sum[i] += c.item() * x.shape[0] * x.shape[1]
        io_cnt[i] += x.shape[0] * x.shape[1]
    return fn

for i, lyr in enumerate(layers):
    _handles.append(lyr.register_forward_hook(_hook(i)))

with torch.no_grad():
    for b in range(0, N_COS_SAMPLES, 8):
        idxs = list(range(b, min(b + 8, N_COS_SAMPLES)))
        ii, ll, attn = batch_tensors(idxs)
        model(input_ids=ii, attention_mask=attn)
        del ii, ll, attn
        torch.cuda.empty_cache()
for h in _handles:
    h.remove()
io_cosine = (io_sum / io_cnt.clamp_min(1)).numpy()
log("io_cosine per layer: " + ", ".join(f"{v:.4f}" for v in io_cosine))
set_status(stage="io_cosine_done")

# --------------------------------- phase 2: empirical Fisher diagonal trace
# 4-bit base weights are frozen and admit no param grads, so for each layer we
# temporarily swap its bnb 4-bit linears for plain bf16 linears holding the
# dequantized weights (trainable), run per-sample fwd+bwd (input grads flow
# through the still-frozen 4-bit linears of the other layers), accumulate the
# sum of squared grads over the layer's params, then restore the 4-bit
# modules. Per-sample NLL (teacher-forced on response tokens only),
# micro-batch 1.
log("phase 2: empirical Fisher trace per layer (dequantized bf16 copy, "
    "one layer at a time) ...")
set_status(stage="fisher", done_layers=0)

def swap_layer_to_bf16(layer):
    """Replace every bnb 4-bit Linear in the layer with a plain bf16
    torch.nn.Linear holding the dequantized weights (trainable). Returns
    the saved originals for restore. No deepcopy: deepcopying bnb
    Params4bit modules corrupts their quant_state and crashes forward."""
    saved = {}
    for name, mod in list(layer.named_modules()):
        if isinstance(mod, bnb.nn.Linear4bit):
            lin = torch.nn.Linear(mod.in_features, mod.out_features,
                                  bias=(mod.bias is not None))
            with torch.no_grad():
                lin.weight.copy_(dequantize_4bit(
                    mod.weight.data, mod.weight.quant_state).to(torch.bfloat16))
                if mod.bias is not None and lin.bias is not None:
                    lin.bias.copy_(mod.bias.data.to(torch.bfloat16))
            lin.to(dtype=torch.bfloat16, device=mod.weight.device)
            parent = layer
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], lin)
            saved[name] = mod
    assert saved, "no Linear4bit found in layer!"
    for p in layer.parameters():
        p.requires_grad_(True)
    return saved

def restore_layer(layer, saved):
    for name, mod in saved.items():
        parent = layer
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], mod)
    for p in layer.parameters():
        p.requires_grad_(False)

def nll_loss(logits, labels):
    # per-sample NLL summed over response tokens
    lsm = torch.log_softmax(logits.float(), dim=-1)
    nll = -lsm.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    return (nll * mask).sum() / mask.sum().clamp_min(1)

fisher_trace = np.zeros(n_layers, dtype=np.float64)
fisher_n = N_FISHER_SAMPLES
f_idx = list(range(N_FISHER_SAMPLES))

model.train()  # enable grad flow; dropout is 0.0 in these configs, eval-equivalent
for i in range(n_layers):
    layer = layers[i]
    saved = swap_layer_to_bf16(layer)
    model.zero_grad(set_to_none=True)
    acc = 0.0
    try:
        for j in f_idx:
            ii, ll, attn = batch_tensors([j])
            out = model(input_ids=ii, attention_mask=attn)
            loss = nll_loss(out.logits, ll)
            loss.backward()
            g2 = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    g2 += float((p.grad.detach().double() ** 2).sum())
            acc += g2
            model.zero_grad(set_to_none=True)
            del out, loss, ii, ll, attn
        fisher_trace[i] = acc / fisher_n
    finally:
        restore_layer(layer, saved)
        torch.cuda.empty_cache()
    if (i + 1) % 6 == 0 or i == n_layers - 1:
        log(f"fisher: {i+1}/{n_layers} layers done, "
            f"VRAM={(torch.cuda.memory_allocated()/1e9):.2f} GB")
        set_status(stage="fisher", done_layers=i + 1)

model.eval()
log("fisher_trace per layer (first 6): " +
    ", ".join(f"{v:.3e}" for v in fisher_trace[:6]))

# ------------------------------------------------------- stats + outputs
rho, pval = spearmanr(io_cosine, fisher_trace)
k = max(1, n_layers // 4)
top_f = set(np.argsort(fisher_trace)[-k:])       # highest Fisher
bot_c = set(np.argsort(io_cosine)[:k])            # lowest io-cosine
overlap = len(top_f & bot_c) / k
# also raw correlation io_cosine vs fisher (Pearson) for reference
pear = float(np.corrcoef(io_cosine, fisher_trace)[0, 1])

with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer_idx", "io_cosine", "fisher_trace"])
    for i in range(n_layers):
        w.writerow([i, f"{io_cosine[i]:.6f}", f"{fisher_trace[i]:.6e}"])

results = {
    "model": MODEL_ID, "quant": "nf4-4bit",
    "n_cos_samples": N_COS_SAMPLES, "n_fisher_samples": N_FISHER_SAMPLES,
    "max_len": MAX_LEN, "n_layers": n_layers,
    "spearman_rho_ioCosine_vs_fisherTrace": float(rho),
    "spearman_pvalue": float(pval),
    "pearson_r": pear,
    "quartile_k": k,
    "topFisher_in_bottomCosine_overlap": overlap,
    "csv": CSV_PATH,
}
with open(RES_PATH, "w") as f:
    json.dump(results, f, indent=2)

log("==== RESULTS ====")
log(json.dumps(results, indent=2))
set_status(stage="done", **{k2: v for k2, v in results.items()
                            if isinstance(v, (int, float, str))})
log("DONE")
