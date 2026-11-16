#!/usr/bin/env python3
"""SPRA-LoRA follow-up v3 (STAGED, NOT RUN): B=0 weight-space SNR probe.

Rationale: v2 (warmed B) measured flat SNR ~0.39 across r=4..32. At
warmed B the gradient G(A) varies with the resampled A, so E[Ghat]
is swamped by mismatch noise. At B=0 the theory is exact: G is
independent of A (BA=0), Ghat = G_0 @ A^T @ A, E[Ghat] = r*sigma^2*G_0,
and A^T A concentrates with r, so SNR should grow ~sqrt(r). v3 is v2
with SNR_WARMUP_STEPS = 0 (B stays at its peft init, i.e. 0) -- the
clean regime for claim (3). Run it as a fresh short Colab session when
the SNR leg is revisited.

v2 fixes the v1 OOM (attempt train-1116-snr2, 2026-11-16): v1 formed
Ghat = grad_B @ A for ALL LoRA layers in fp32 -- a second copy of the
model -- and died at the first resample (14.29 GiB allocated, 192 MiB
allocation failed). v2 computes Ghat only for q_proj of layers
[0, 11, 23, 35] (4 layers), in bf16, with a two-pass seeded resampling
(pass 1 accumulates the mean, pass 2 the deviations; torch.manual_seed
before each resample makes pass 2 reproduce pass 1's A exactly), and
frees out/logits/ce before the Ghat math plus empty_cache between
resamples.

Why this exists: the SNR probe in colab_spra_driver.py measures the SNR
of grad_B across A-resamples. That quantity is degenerate: at B = 0,
E[grad_B] = 0 by symmetry (forward pass independent of A), and even
after a warmup B is small, so E[grad_B] stays ~0 and the measured SNR
is ~1/sqrt(M) flat across ranks (attempt 2: r=4: 0.349, r=16: 0.394,
M=8). It cannot discriminate claim (3).

The paper's claim lives in WEIGHT space. For one layer, y = (W + BA)x,
so with G = dL/d(BA) the full-FT gradient for that layer,
    grad_B = G @ A^T        (chain rule)
and the weight-space gradient implied by a frozen-A step is
    Ghat = grad_B @ A = G @ (A^T A).
E[A^T A] = r*sigma^2 * I for iid A, so E[Ghat] is proportional to the
full gradient G (nonzero signal), while A^T A concentrates as r grows
(law of large numbers over its r rows), so the noise shrinks relative
to the signal: SNR(Ghat) should INCREASE with r (theory: ~sqrt(r)).
This driver measures exactly that, pooled over all LoRA layers:
    SNR(r) = || E[Ghat] ||_F / E|| Ghat - E[Ghat] ||_F
with the Frobenius norms pooled across layers.

Same setup as the main driver (Qwen3.5-9B first, fall back to Qwen3-8B;
ARC-Challenge fixed batch of 8; 4 warmup steps per rank so B != 0;
M=8 resamples per rank). Writes /content/spra_snr2_results.json and
prints TRAIN_DONE.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import math
import random
import subprocess
import sys
import time

RESULTS_JSON = "/content/spra_snr2c_results.json"
SEED = 123
random.seed(SEED)
BASE_LR = 2e-4
MAX_LEN = 256
TRAIN_BS = 8
SNR_RESAMPLES = 8
SNR_RANKS = [4, 8, 16, 32]
SNR_WARMUP_STEPS = 0
MODELS_TRY = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"]
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

RESULTS = {
    "paper": "2611.13331",
    "title": "SPRA-LoRA follow-up v3 (B=0): weight-space SNR-vs-rank probe",
    "method": "Ghat = grad_B @ A per resample, pooled Frobenius SNR over all LoRA layers; "
              "4 warmup steps per rank (frozen A) so B != 0; M=8 resamples per rank.",
    "models_tried": MODELS_TRY,
    "model_used": None,
    "fallback_used": False,
    "snr_probe": {},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flush_results():
    with open(RESULTS_JSON, "w") as f:
        json.dump(RESULTS, f, indent=2)


def install():
    log("installing deps")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.57,<5.0", "bitsandbytes", "accelerate",
                    "peft", "datasets", "hf_transfer"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False)
    log("deps installed")


def load_data():
    from datasets import load_dataset
    tr = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    tr = tr.shuffle(seed=SEED)
    rows = []
    for r in tr:
        texts = r["choices"]["text"]
        labels = r["choices"]["label"]
        if len(texts) != 4 or labels != ["A", "B", "C", "D"]:
            continue
        q = (r["question"] or "").strip()
        if not q:
            continue
        prompt = (f"Question: {q}\nA. {texts[0]}\nB. {texts[1]}\n"
                  f"C. {texts[2]}\nD. {texts[3]}\nAnswer:")
        rows.append({"prompt": prompt, "gold": r["answerKey"],
                     "completion": " " + r["answerKey"]})
        if len(rows) >= TRAIN_BS:
            break
    log(f"probe batch rows: {len(rows)}")
    return rows


def build_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    log(f"loading tokenizer + base model {model_id} (NF4)")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             bias="none", task_type="CAUSAL_LM",
                                             target_modules=TARGET),
                           adapter_name="P16")
    for name, r in [("P4", 4), ("P8", 8), ("P32", 32)]:
        model.add_adapter(name, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                           bias="none", task_type="CAUSAL_LM",
                                           target_modules=TARGET))
    return tok, model


def lora_params(model, adapter, kind):
    return [(n, p) for n, p in model.named_parameters()
            if f"lora_{kind}.{adapter}." in n]


def resample_A(model, adapter):
    import torch
    for _, p in lora_params(model, adapter, "A"):
        torch.nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))


def reinit_adapter(model, adapter):
    import torch
    for _, p in lora_params(model, adapter, "A"):
        torch.nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
    for _, p in lora_params(model, adapter, "B"):
        torch.nn.init.zeros_(p.data)


def encode_batch(tok, rows):
    import torch
    eos_id = tok.eos_token_id
    seqs, labs = [], []
    for e in rows:
        p = tok(e["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(e["completion"], add_special_tokens=False)["input_ids"] + [eos_id]
        ids = (p + c)[:MAX_LEN]
        lab = ([-100] * len(p) + c)[:MAX_LEN]
        seqs.append(ids)
        labs.append(lab)
    mx = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    return (torch.tensor([s + [pad] * (mx - len(s)) for s in seqs]),
            torch.tensor([l + [-100] * (mx - len(l)) for l in labs]),
            torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in seqs]))


def snr_probe(tok, model, rows):
    import torch
    input_ids, labels, attn = (t.to("cuda") for t in encode_batch(tok, rows))
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    model.train()
    # Ghat layers: q_proj of layers 0, 11, 23, 35 (representative subset)
    probe_layers = [0, 11, 23, 35]

    def ghat_for_layer(adapter, layer):
        """Ghat = grad_B @ A for one layer's q_proj (bf16). Caller must
        have run backward already; returns a fresh bf16 tensor."""
        bname = None
        for n, p in model.named_parameters():
            if f"layers.{layer}.self_attn.q_proj.lora_B.{adapter}." in n:
                bname = n
                bp = p
                break
        aname = bname.replace("lora_B", "lora_A")
        ap = dict(model.named_parameters())[aname]
        return bp.grad.detach() @ ap.detach()

    def run_backward():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn)
            ce = torch.nn.functional.cross_entropy(
                out.logits.float()[:, :-1, :].reshape(-1, out.logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        loss = ce.item()
        ce.backward()
        del out, ce
        return loss

    for r in SNR_RANKS:
        adapter = f"P{r}"
        model.set_adapter(adapter)
        for n, p in model.named_parameters():
            p.requires_grad_(False)
        for _, p in lora_params(model, adapter, "B"):
            p.requires_grad_(True)
        reinit_adapter(model, adapter)
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=BASE_LR, weight_decay=0.0)
        for w in range(SNR_WARMUP_STEPS):
            loss = run_backward()
            opt.step()
            torch.cuda.empty_cache()
            log(f"  r={r} warmup {w+1}/{SNR_WARMUP_STEPS} loss {loss:.4f}")
        del opt
        # pass 1: accumulate mean Ghat per probe layer (seeded resamples)
        mean_ghat = {}
        for m in range(SNR_RESAMPLES):
            torch.manual_seed(1000 + m)
            resample_A(model, adapter)
            loss = run_backward()
            for layer in probe_layers:
                g = ghat_for_layer(adapter, layer)
                if layer in mean_ghat:
                    mean_ghat[layer] += g
                else:
                    mean_ghat[layer] = g.clone()
            torch.cuda.empty_cache()
            log(f"  r={r} pass1 resample {m+1}/{SNR_RESAMPLES} loss {loss:.4f}")
        for layer in mean_ghat:
            mean_ghat[layer] /= SNR_RESAMPLES
        # pass 2: same seeds -> same A -> deviations from the mean
        dev_norms = {layer: [] for layer in probe_layers}
        for m in range(SNR_RESAMPLES):
            torch.manual_seed(1000 + m)
            resample_A(model, adapter)
            loss = run_backward()
            for layer in probe_layers:
                g = ghat_for_layer(adapter, layer)
                dev_norms[layer].append((g - mean_ghat[layer]).norm().item())
            torch.cuda.empty_cache()
            log(f"  r={r} pass2 resample {m+1}/{SNR_RESAMPLES} loss {loss:.4f}")
        # pooled Frobenius SNR over probe layers
        sig2 = sum(mean_ghat[l].norm().item() ** 2 for l in probe_layers)
        noi2 = sum((sum(dev_norms[l]) / len(dev_norms[l])) ** 2 for l in probe_layers)
        signal = sig2 ** 0.5
        noise = noi2 ** 0.5
        snr = signal / max(noise, 1e-12)
        per_layer = {f"layer{layer}": {
            "snr": mean_ghat[layer].norm().item() / max(
                sum(dev_norms[layer]) / len(dev_norms[layer]), 1e-12),
            "signal_norm": mean_ghat[layer].norm().item(),
            "noise_norm": sum(dev_norms[layer]) / len(dev_norms[layer])}
            for layer in probe_layers}
        RESULTS["snr_probe"][f"r{r}"] = {
            "snr": snr, "signal_norm": signal, "noise_norm": noise,
            "resamples": SNR_RESAMPLES, "warmup_steps": SNR_WARMUP_STEPS,
            "probe_layers": probe_layers, "per_layer": per_layer}
        log(f"WEIGHT-SPACE SNR r={r}: {snr:.4f} (signal {signal:.4f}, noise {noise:.4f})")
        flush_results()
    xs = [float(r) for r in SNR_RANKS]
    ys = [RESULTS["snr_probe"][f"r{r}"]["snr"] for r in SNR_RANKS]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    RESULTS["snr_probe"]["linear_fit"] = {"slope": slope, "intercept": intercept, "r2": r2}
    log(f"WEIGHT-SPACE SNR linear fit: slope={slope:.4f}/rank, intercept={intercept:.4f}, R2={r2:.4f}")
    flush_results()


def main():
    install()
    rows = load_data()
    for i, model_id in enumerate(MODELS_TRY):
        try:
            tok, model = build_model(model_id)
            RESULTS["model_used"] = model_id
            flush_results()
            snr_probe(tok, model, rows)
            if i > 0:
                RESULTS["fallback_used"] = True
            break
        except Exception as e:
            log(f"MODEL {model_id} FAILED: {type(e).__name__}: {e}")
            flush_results()
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            if i == len(MODELS_TRY) - 1:
                raise
            log("falling back to next model (one automatic fallback)")
    flush_results()
    log("ALL PHASES DONE")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
