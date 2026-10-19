#!/usr/bin/env python3
"""Colab T4 driver — local proxies vs QLoRA vs MeZO at the 8B class.

arXiv 2610.23145 ("Local proxies for memory-efficient training"), training
track. Paper claim under test: fine-tuning LLMs with per-block proxy
networks (distilled to predict the global output; block gradients flow
through the frozen proxy) trains faster and better than memory-efficient
zeroth-order methods at comparable memory cost, without retaining
full-stack activations.

NOTE on model class: the track mandate is Qwen3.5-9B, but Qwen/Qwen3.5-9B
is not loadable under transformers<5.0 ("model type qwen3_5 ... does not
recognize this architecture", established 2026-10-18). The proven
qwen3.5-9b-q4 recipe is Ollama-only (inference). This driver therefore runs
at the 8B class (Qwen3-8B, loadable), same NF4 QLoRA config (r=16/a=32).

Three arms, identical data/steps/seeds (fixed-budget comparison):
  A standard  : NF4 QLoRA SFT, full backprop (no checkpointing)
  B proxy     : NF4 QLoRA, per-block proxies predict final hidden state;
                each block's LoRA grads via frozen proxy + frozen lm_head
  C mezo      : NF4 QLoRA, SPSA zeroth-order on LoRA params (2 fwd/step)

Metrics per arm: train loss curve, eval loss (plain forward, same head),
peak VRAM (torch.cuda.max_memory_allocated, reset per arm), wall time.
Incremental CSV writes -> /content/local_proxies_metrics.csv ; final
/content/local_proxies_results.json.

Self-contained: pip bootstrap at top (incl. `pip uninstall -y torchao`
right after installs — Colab ships torchao 0.10, peft needs >0.16).
Bounded: 16 steps/arm, batch 2, seq 256, 120 train / 40 eval synthetic
finance-domain SFT pairs (toy data; the comparison is relative).
Note: batch 4 / seq 512 was tried first — the standard full-backprop arm
OOMs there (>14.5 GB), which is the memory wall this paper addresses;
all arms therefore run at the reduced config for apples-to-apples.
"""
import csv
import gc
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── 0. pip bootstrap ─────────────────────────────────────────────────────────
def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)

_pip("install", "-q", "transformers>=4.46,<5.0", "peft>=0.14", "bitsandbytes>=0.44",
     "accelerate>=0.34")
_pip("uninstall", "-y", "torchao")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

SEED = 7
MODEL_ID = "Qwen/Qwen3-8B"
# batch 2 / seq 256: full-backprop standard arm must fit in 14.5 GB
# (batch 4 / seq 512 OOMs — the memory wall this paper addresses).
STEPS, BATCH, SEQLEN = 16, 2, 256
N_TRAIN, N_EVAL = 120, 40
CSV_PATH = "/content/local_proxies_metrics.csv"
JSON_PATH = "/content/local_proxies_results.json"

torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
assert torch.cuda.is_available(), "no CUDA"
DEV = "cuda"
print("GPU:", torch.cuda.get_device_name(0), flush=True)

# ── 1. synthetic finance-domain SFT data (seeded, toy; relative comparison) ──
TERMS = {
    "alpha": "excess return over a benchmark, adjusted for risk",
    "beta": "sensitivity of returns to market movements",
    "sharpe ratio": "mean excess return per unit of volatility",
    "drawdown": "peak-to-trough decline in portfolio value",
    "duration": "price sensitivity of a bond to interest-rate moves",
    "convexity": "second-order rate sensitivity of a bond price",
    "var": "maximum expected loss at a confidence level",
    "liquidity": "ease of trading without moving the price",
}
SENTS = [("Margins expanded on strong cloud demand", "positive"),
         ("The firm missed revenue estimates", "negative"),
         ("Guidance was left unchanged", "neutral"),
         ("Buybacks were doubled this quarter", "positive"),
         ("Impairment charges hit the quarter", "negative")]


def make_data(rng, n):
    out = []
    for _ in range(n):
        kind = rng.integers(0, 3)
        if kind == 0:
            t = rng.choice(list(TERMS))
            out.append(f"Q: Define {t} in finance. A: {t} is {TERMS[t]}.")
        elif kind == 1:
            r = float(rng.uniform(2, 15)); v = float(rng.uniform(3, 25))
            out.append(f"Q: A fund returns {r:.1f}% with volatility {v:.1f}%. "
                       f"Compute a simple risk-adjusted score (return/volatility). "
                       f"A: {r/v:.2f}.")
        else:
            s, lab = SENTS[rng.integers(0, len(SENTS))]
            out.append(f"Q: Classify the headline sentiment: '{s}' A: {lab}.")
    return out


tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

rng = __import__("numpy").random.default_rng(SEED)
train_texts = make_data(rng, N_TRAIN)
eval_texts = make_data(rng, N_EVAL)

def encode(texts):
    e = tok(texts, padding="max_length", truncation=True, max_length=SEQLEN,
            return_tensors="pt")
    return e["input_ids"], e["attention_mask"]

train_ids, train_mask = encode(train_texts)
eval_ids, eval_mask = encode(eval_texts)
print(f"data: {N_TRAIN} train / {N_EVAL} eval, seqlen {SEQLEN}", flush=True)


def lm_loss(logits, labels, mask):
    # shift; ignore padded label positions
    l = logits[:, :-1].reshape(-1, logits.size(-1))
    t = labels[:, 1:].reshape(-1)
    m = mask[:, 1:].reshape(-1).bool()
    return torch.nn.functional.cross_entropy(l[m], t[m])


def eval_loss(model):
    model.eval()
    tot, nb = 0.0, 0
    with torch.no_grad():
        for i in range(0, N_EVAL, BATCH):
            ids = eval_ids[i:i+BATCH].to(DEV); m = eval_mask[i:i+BATCH].to(DEV)
            tot += lm_loss(model(input_ids=ids, attention_mask=m,
                                 use_cache=False).logits, ids, m).item()
            nb += 1
    model.train()
    return tot / nb


def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                     task_type=TaskType.CAUSAL_LM)
    m = get_peft_model(m, cfg)
    m.train()
    m.print_trainable_parameters()
    return m


def lora_params(model):
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def log_row(arm, step, train_loss, eval_l, peak_gb, wall_s):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "step", "train_loss", "eval_loss",
                        "peak_vram_gb", "wall_s"])
        w.writerow([arm, step, f"{train_loss:.4f}",
                    f"{eval_l:.4f}" if eval_l is not None else "",
                    f"{peak_gb:.2f}", f"{wall_s:.1f}"])
    print(f"[{arm}] step={step} train={train_loss:.4f} "
          f"eval={eval_l if eval_l is None else f'{eval_l:.4f}'} "
          f"peak={peak_gb:.2f}GB wall={wall_s:.0f}s", flush=True)


def run_arm(arm):
    t0 = time.time()
    model = load_model()
    try:
        return _run_arm_inner(arm, model, t0)
    finally:
        # always release the model, even on failure (a leaked model OOMs
        # the next arm)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _run_arm_inner(arm, model, t0):
    torch.cuda.reset_peak_memory_stats()  # after load: exclude quantize spike
    opt = torch.optim.AdamW(lora_params(model), lr=2e-4)
    text_model = model.model  # underlying Qwen3Model (peft wrapper)
    layers = text_model.model.layers
    n_layers = len(layers)
    print(f"[{arm}] {n_layers} layers", flush=True)
    e0 = eval_loss(model)

    if arm == "proxy":
        # one small proxy per block: block-output -> final hidden state.
        # SGD (no optimizer state) to keep the memory comparison honest.
        hid = text_model.config.hidden_size
        proxies = [torch.nn.Sequential(
            torch.nn.Linear(hid, 512), torch.nn.Tanh(),
            torch.nn.Linear(512, hid)).to(DEV).to(torch.bfloat16)
            for _ in range(n_layers)]
        pxy_opts = [torch.optim.SGD(px.parameters(), lr=0.05)
                    for px in proxies]
        mse = torch.nn.MSELoss()

    for step in range(STEPS):
        idx = torch.randperm(N_TRAIN)[:BATCH]
        ids = train_ids[idx].to(DEV); m = train_mask[idx].to(DEV)

        if arm == "standard":
            out = model(input_ids=ids, attention_mask=m, use_cache=False)
            loss = lm_loss(out.logits, ids, m)
            loss.backward()
            opt.step(); opt.zero_grad()
            train_l = loss.item()

        elif arm == "mezo":
            ps = lora_params(model)
            flats = [p.detach().flatten() for p in ps]
            z = [torch.randn_like(f) for f in flats]
            eps = 1e-3
            with torch.no_grad():
                for p, f, zi in zip(ps, flats, z):
                    p.copy_((f + eps * zi).reshape(p.shape))
                lp = lm_loss(model(input_ids=ids, attention_mask=m,
                                   use_cache=False).logits, ids, m).item()
                for p, f, zi in zip(ps, flats, z):
                    p.copy_((f - eps * zi).reshape(p.shape))
                lm = lm_loss(model(input_ids=ids, attention_mask=m,
                                   use_cache=False).logits, ids, m).item()
                g = (lp - lm) / (2 * eps)
                for p, f, zi in zip(ps, flats, z):
                    p.copy_((f - 5e-4 * g * zi).reshape(p.shape))
            train_l = (lp + lm) / 2

        elif arm == "proxy":
            # (i) no-grad full forward; capture per-block inputs via
            # forward PRE-hooks (with_kwargs=True gives (module, args, kwargs);
            # plain forward hooks receive (module, args, output) instead)
            saved = {}
            handles = []
            for i, layer in enumerate(layers):
                def _hook(mod, args, kwargs, _i=i):
                    saved[_i] = (args[0].detach(),
                                 kwargs.get("attention_mask"),
                                 kwargs.get("position_ids"),
                                 kwargs.get("position_embeddings"))
                handles.append(layer.register_forward_pre_hook(
                    _hook, with_kwargs=True))
            with torch.no_grad():
                full = model(input_ids=ids, attention_mask=m, use_cache=False,
                             output_hidden_states=True)
            for h in handles:
                h.remove()
            target = full.hidden_states[-1].detach()  # global output (final hidden)
            # peft wrapper: hidden_states align with text_model layers
            # (hidden_states[0] = embeddings, [i+1] = output of layer i)

            # (ii) distill: warm-started proxies, 3 steps on this batch
            for i, (px, po) in enumerate(zip(proxies, pxy_opts)):
                h_out = full.hidden_states[i + 1].detach()
                for _ in range(3):
                    po.zero_grad()
                    lpx = mse(px(h_out.reshape(-1, px[0].in_features)),
                              target.reshape(-1, target.size(-1)))
                    lpx.backward()
                    po.step()

            # (iii) local grads, one block at a time through frozen proxy
            opt.zero_grad()
            for i, layer in enumerate(layers):
                h_in, am, pids, pemb = saved[i]
                with torch.enable_grad():
                    lo = layer(hidden_states=h_in, attention_mask=am,
                               position_ids=pids, position_embeddings=pemb)
                    h_out = lo[0] if isinstance(lo, tuple) else lo
                    pred = proxies[i](h_out.reshape(-1, h_out.size(-1)))
                    logits_i = model.lm_head(
                        pred.reshape(h_out.shape[0], h_out.shape[1], -1))
                    loss_i = lm_loss(logits_i, ids, m)
                    loss_i.backward()  # only this block's LoRA params get grads
            opt.step()
            with torch.no_grad():
                train_l = lm_loss(model(input_ids=ids, attention_mask=m,
                                        use_cache=False).logits, ids, m).item()

        ev = eval_loss(model) if (step + 1) % 4 == 0 or step == STEPS - 1 else None
        peak = torch.cuda.max_memory_allocated() / 1e9
        log_row(arm, step + 1, train_l, ev, peak, time.time() - t0)

    final_eval = eval_loss(model)
    peak = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0
    return {"eval_start": e0, "eval_final": final_eval,
            "peak_vram_gb": peak, "wall_s": wall}


results = {}
for arm in ["standard", "proxy", "mezo"]:
    try:
        results[arm] = run_arm(arm)
        results[arm]["status"] = "ok"
    except Exception as ex:  # noqa: BLE001 - keep going; record honestly
        import traceback
        results[arm] = {"status": f"FAILED: {type(ex).__name__}: {ex}",
                        "traceback": traceback.format_exc()[-2000:]}
        print(f"[{arm}] FAILED: {ex}", flush=True)
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

print("RESULTS:", json.dumps(results, indent=2), flush=True)
print("done", flush=True)
