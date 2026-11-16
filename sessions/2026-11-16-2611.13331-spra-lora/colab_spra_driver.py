#!/usr/bin/env python3
"""SPRA-LoRA -- Stochastic Projection Rank-Aware LoRA (arXiv:2611.13331).

Paper claims (federated LoRA fine-tuning):
  (1) LoRA with Gaussian init + FROZEN A acts as a stochastic
      approximation of full fine-tuning: batch gradients get a random,
      rank-dependent perturbation.
  (2) Freezing + periodically resampling A decouples A/B gradients,
      cuts compute + uplink communication by 50%, freeing resources
      for higher adaptation ranks.
  (3) In the frozen-A regime the SNR of the perturbed gradients grows
      (linearly) with rank r, so clients' learning rates must scale
      with their rank for higher utility.
  (4) SPRA-LoRA outperforms SOTA baselines with half the overhead.

Bounded single-GPU test of claims (1)-(3) on a current-generation model:
  * SNR probe: fixed batch, warm up B for 4 steps (frozen A) so B != 0,
    then resample A M=8 times per rank,
    SNR(r) = ||E[grad_B]||_F / E||grad_B - E[grad_B]||_F for
    r in {4, 8, 16, 32}. Claim (3) predicts SNR increasing ~linearly.
    The warmup is ESSENTIAL: at init B = 0, so the forward pass does not
    depend on A at all and E[grad_B] = 0 by symmetry -- the probe then
    measures only finite-sample noise (~1/sqrt(M) flat across ranks).
    Attempt 1 measured exactly that (loss identical 11.7816 across all
    resamples; SNR r=4: 0.3808, r=8: 0.3322, both ~= 1/sqrt(8)); those
    numbers are the null control, not evidence about claim (3).
  * Training arms (all QLoRA r as noted, alpha=2r, bf16, NF4, 1 epoch
    over 300 ARC-Easy train rows, lr as noted, 2 simulated clients):
      S16   standard LoRA r=16, lr 2e-4 (baseline)
      F16   frozen-A r=16, lr 2e-4 (rank-scaled reference)
      F8hi  frozen-A r=8,  lr 2e-4 (fixed LR, unscaled)
      F8lo  frozen-A r=8,  lr 1e-4 (rank-scaled: lr = 2e-4 * r/16)
      F32lo frozen-A r=32, lr 4e-4 (rank-scaled; tests reallocating
            resources to higher ranks)
    A is resampled every 10 optimizer steps and at the client switch.
  * Eval: ARC-Challenge validation n=200, LOGPROB-scored MCQA (first-token
    logprob of " A"/" B"/" C"/" D" after "Answer:") -- format
    independent, so no letter-parse artifact (cf. 2026-11-15 PECA).
    Base model scored on the same 200 as reference. ARC-Challenge (not
    Easy): attempt 1 on ARC-Easy hit a ceiling -- base Qwen3-8B scored
    195/200 = 0.975, leaving no headroom for arm differences.
  * Comms proxy: trainable-param counts per arm (frozen-A transmits
    B only) -- claim (2)'s 50% is arithmetic, reported exactly.

Model rule (2026-10-27, still binding): current-generation families
only, weight updates in the 8-9B class under NF4 QLoRA; shrink the
EXPERIMENT, never the model. Try Qwen3.5-9B first (latest 9B-class
Qwen); on load/OOM failure fall back in-driver to the proven
Qwen3-8B carrier (one automatic fallback, 2026-11-05 rule).
Known blocker (attempt 1, 2026-11-16): transformers 4.57.6 -- the newest
<5.0 -- does not recognise model type `qwen3_5`, so the 9B attempt fails
fast (~2s) and the run proceeds on Qwen3-8B. A source install of
transformers was NOT attempted: it risks breaking the proven 8B path.

Honest limits (recorded, not hidden):
- Single-GPU sequential simulation of 2 clients; no FedAvg
  aggregation step (not testable on one GPU without extra machinery).
  The tested claims are the client-local optimization dynamics
  (frozen-A, resampling, rank-aware LR), which is where claims
  (1)-(3) live; claim (4)'s "beats SOTA federated baselines" is
  NOT tested -- no federated baseline is run.
- The paper's abstract does not publish the exact LR-scaling formula;
  lr = 2e-4 * (r/16) is our operationalization of "LR scales with r".
- The paper's experiments are on (unspecified) benchmark datasets;
  ARC-Easy is our transfer task, not the paper's dataset.
- n=200 eval: differences < ~9pp are within noise (95% CI ~+/-7pp
  at p=0.5); report CIs and treat small deltas as ties.
- Metrics CSV + per-item eval rows + results JSON are flushed to
  /content after EVERY phase (reclamation-safe). Final stdout line
  is TRAIN_DONE.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import gc
import json
import math
import random
import subprocess
import sys
import time

METRICS_CSV = "/content/spra_metrics.csv"
EVAL_ROWS_CSV = "/content/spra_eval_rows.csv"
RESULTS_JSON = "/content/spra_results.json"
SEED = 123
random.seed(SEED)

N_TRAIN = 300
EVAL_N = 200
BASE_LR = 2e-4
MAX_LEN = 256
EVAL_BS = 8
TRAIN_BS = 8
RESAMPLE_EVERY = 10
SNR_RESAMPLES = 8
SNR_RANKS = [4, 8, 16, 32]
MODELS_TRY = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"]
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# arm: (adapter_name, rank, lr, freeze_a)
ARMS = [
    ("S16", 16, 2e-4, False),
    ("F16", 16, 2e-4, True),
    ("F8hi", 8, 2e-4, True),
    ("F8lo", 8, 1e-4, True),
    ("F32lo", 32, 4e-4, True),
]

RESULTS = {
    "paper": "2611.13331",
    "title": "Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA",
    "variant": "Bounded single-GPU transfer test: SNR-vs-rank probe + 5 QLoRA arms (standard vs frozen-A vs rank-aware LR) on ARC-Challenge; 300 train rows / 1 epoch / 2 simulated clients; logprob MCQA eval n=200.",
    "models_tried": MODELS_TRY,
    "model_used": None,
    "fallback_used": False,
    "lora": {"alpha_factor": 2, "dropout": 0.0, "target": "q,k,v,o,gate,up,down"},
    "train": {"n": N_TRAIN, "epochs": 1, "batch": TRAIN_BS, "max_len": MAX_LEN,
              "resample_every_steps": RESAMPLE_EVERY, "clients": 2},
    "eval": {"dataset": "allenai/ai2_arc ARC-Challenge validation", "n": EVAL_N,
             "method": "logprob of ' A'/' B'/' C'/' D' after 'Answer:'", "batch": EVAL_BS},
    "arms": {},
    "snr_probe": {},
    "comms": {},
    "phases": [],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flush_results():
    with open(RESULTS_JSON, "w") as f:
        json.dump(RESULTS, f, indent=2)


def record_metric(arm, split, n, correct, notes=""):
    acc = correct / n if n else 0.0
    # Wilson 95% CI half-width approx 1.96*sqrt(p(1-p)/n)
    hw = 1.96 * math.sqrt(max(acc * (1 - acc), 1e-9) / n) if n else 0.0
    exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["arm", "split", "n", "correct", "acc", "ci95_hw", "notes"])
        w.writerow([arm, split, n, correct, f"{acc:.4f}", f"{hw:.4f}", notes])
    log(f"METRIC {arm} {split}: {correct}/{n} = {acc:.4f} +/- {hw:.4f} {notes}")
    return acc, hw


def install():
    log("installing deps")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.57,<5.0", "bitsandbytes", "accelerate",
                    "peft", "datasets", "hf_transfer"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False)
    import transformers
    log(f"transformers {transformers.__version__} installed")
    log("deps installed")


def load_data():
    from datasets import load_dataset
    log("loading ARC-Challenge train + validation")
    tr = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    va = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    tr = tr.shuffle(seed=SEED)
    va = va.shuffle(seed=SEED)

    def to_rows(ds, limit):
        rows = []
        for r in ds:
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
            if len(rows) >= limit:
                break
        return rows

    train_rows = to_rows(tr, N_TRAIN)
    eval_items = to_rows(va, EVAL_N)
    log(f"train rows: {len(train_rows)}, eval items: {len(eval_items)}")
    return train_rows, eval_items


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
    base_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM", target_modules=TARGET)
    model = get_peft_model(model, base_cfg, adapter_name="S16")
    for name, r in [("F16", 16), ("F8hi", 8), ("F8lo", 8), ("F32lo", 32),
                    ("P4", 4), ("P8", 8), ("P16", 16), ("P32", 32)]:
        model.add_adapter(name, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                           bias="none", task_type="CAUSAL_LM",
                                           target_modules=TARGET))
    model.print_trainable_parameters()
    # comms proxy: trainable params per arm type, counted by name
    # (standard LoRA trains A+B; frozen-A arms train B only).
    counts = {}
    for name, _, _, freeze_a in ARMS:
        n_train = 0
        for pname, p in model.named_parameters():
            if f".lora_B.{name}." in pname or (not freeze_a and f".lora_A.{name}." in pname):
                n_train += p.numel()
        counts[name] = n_train
    s16 = counts.get("S16", 0)
    for name in counts:
        ratio = counts[name] / s16 if s16 else 0.0
        log(f"comms {name}: {counts[name]/1e6:.1f}M trainable ({ratio:.2f}x of S16)")
    RESULTS["comms"] = {
        "trainable_params": counts,
        "note": "standard LoRA trains A+B; frozen-A arms train B only. "
                "Uplink per round ~ trainable params (x2 bytes bf16).",
    }
    log(f"VRAM after load: {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated, "
        f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB peak")
    return tok, model


def lora_params(model, adapter, kind):
    """Yield (name, param) for lora_A / lora_B of one adapter."""
    out = []
    for n, p in model.named_parameters():
        if f"lora_{kind}.{adapter}." in n:
            out.append((n, p))
    return out


def activate_arm(model, adapter, freeze_a):
    """Set active adapter and requires_grad flags (A frozen iff freeze_a)."""
    import torch
    model.set_adapter(adapter)
    for n, p in model.named_parameters():
        if ".lora_" in n:
            p.requires_grad_(False)
    for _, p in lora_params(model, adapter, "B"):
        p.requires_grad_(True)
    if not freeze_a:
        for _, p in lora_params(model, adapter, "A"):
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"arm {adapter} active (freeze_a={freeze_a}): {n_train/1e6:.1f}M trainable")
    return n_train


def resample_A(model, adapter):
    """Re-init the active adapter's A matrices (Gaussian-ish, peft default)."""
    import torch
    for _, p in lora_params(model, adapter, "A"):
        torch.nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
    log(f"  resampled A for {adapter}")


def reinit_adapter(model, adapter):
    """Fresh init for one adapter (A kaiming, B zeros) -- not needed across
    arms (adapters are distinct) but used by the SNR probe per rank."""
    import torch
    for _, p in lora_params(model, adapter, "A"):
        torch.nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
    for _, p in lora_params(model, adapter, "B"):
        torch.nn.init.zeros_(p.data)


SNR_WARMUP_STEPS = 4


def snr_probe(tok, model, train_rows):
    """Claim (3): SNR of the frozen-A perturbed B-gradient grows with r.

    Warmup first: at init B = 0, E[grad_B] = 0 by symmetry (the forward
    pass is independent of A), so the probe would measure only
    finite-sample noise. 4 warmup steps (frozen A, B only) make B != 0,
    after which E[grad_B] is meaningful.
    """
    import torch
    log("== SNR probe: grad_B under resampled A, ranks 4/8/16/32 (warmed B) ==")
    # fixed batch of 8 train examples
    batch_rows = train_rows[:TRAIN_BS]
    enc = encode_batch(tok, batch_rows)
    input_ids, labels, attn = (t.to("cuda") for t in enc)
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    model.train()
    for r in SNR_RANKS:
        adapter = f"P{r}"
        activate_arm(model, adapter, freeze_a=True)
        reinit_adapter(model, adapter)
        # warmup: train B only for a few steps on the fixed batch
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=BASE_LR, weight_decay=0.0)
        for w in range(SNR_WARMUP_STEPS):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits.float()
                ce = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1), ignore_index=-100)
            ce.backward()
            opt.step()
            log(f"  r={r} warmup {w+1}/{SNR_WARMUP_STEPS} loss {ce.item():.4f}")
        del opt
        grads = []
        for m in range(SNR_RESAMPLES):
            resample_A(model, adapter)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits.float()
                ce = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1), ignore_index=-100)
            ce.backward()
            g = torch.cat([p.grad.detach().flatten().float()
                           for _, p in lora_params(model, adapter, "B")
                           if p.grad is not None])
            grads.append(g)
            log(f"  r={r} resample {m+1}/{SNR_RESAMPLES} loss {ce.item():.4f}")
        G = torch.stack(grads)  # [M, D]
        mean = G.mean(dim=0)
        noise = (G - mean).norm(dim=1).mean().item()
        signal = mean.norm().item()
        snr = signal / max(noise, 1e-12)
        RESULTS["snr_probe"][f"r{r}"] = {
            "snr": snr, "signal_norm": signal, "noise_norm": noise,
            "resamples": SNR_RESAMPLES, "warmup_steps": SNR_WARMUP_STEPS}
        log(f"SNR r={r}: {snr:.4f} (signal {signal:.4f}, noise {noise:.4f})")
        flush_results()
    # linear fit of SNR vs r
    xs = [float(r) for r in SNR_RANKS]
    ys = [RESULTS["snr_probe"][f"r{r}"]["snr"] for r in SNR_RANKS]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    RESULTS["snr_probe"]["linear_fit"] = {"slope": slope, "intercept": intercept, "r2": r2}
    log(f"SNR linear fit: slope={slope:.4f}/rank, intercept={intercept:.4f}, R2={r2:.4f}")
    flush_results()


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
    input_ids = torch.tensor([s + [pad] * (mx - len(s)) for s in seqs])
    labels = torch.tensor([l + [-100] * (mx - len(l)) for l in labs])
    attn = torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in seqs])
    return input_ids, labels, attn


def train_arm(tok, model, train_rows, adapter, lr, freeze_a, tag):
    """One arm: 1 epoch over 300 rows split into 2 simulated clients;
    A resampled every RESAMPLE_EVERY steps + at the client switch."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    activate_arm(model, adapter, freeze_a)
    torch.cuda.empty_cache()
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    model.train()
    model.config.use_cache = False

    # 2 simulated clients: first half / second half of shuffled rows
    half = len(train_rows) // 2
    clients = [train_rows[:half], train_rows[half:]]
    # batches in client order (client 0 then client 1)
    batches = []
    for c in clients:
        for i in range(0, len(c), TRAIN_BS):
            batches.append(c[i:i + TRAIN_BS])
    client_switch_step = (len(clients[0]) + TRAIN_BS - 1) // TRAIN_BS
    total = len(batches)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, 3, total)
    step = 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for bi, b in enumerate(batches):
        if freeze_a and step > 0 and (step % RESAMPLE_EVERY == 0 or step == client_switch_step):
            resample_A(model, adapter)
        input_ids, labels, attn = (t.cuda() for t in encode_batch(tok, b))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total:
            log(f"{tag} step {step}/{total} loss {loss.item():.4f} "
                f"elapsed {time.time()-t0:.0f}s "
                f"VRAM {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    log(f"{tag} done: {step} steps, {time.time()-t0:.0f}s, "
        f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    return {"steps": step, "seconds": round(time.time() - t0, 1),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}


LETTERS = ["A", "B", "C", "D"]


def run_eval_logprob(tok, model, items, tag):
    """Logprob MCQA eval; per-batch rows flushed to CSV immediately."""
    import torch
    model.eval()
    model.config.use_cache = True
    letter_ids = []
    for L in LETTERS:
        ids = tok(f" {L}", add_special_tokens=False)["input_ids"]
        letter_ids.append(ids[0])
    log(f"{tag} eval letter token ids: {letter_ids}")
    correct, n = 0, 0
    exists = os.path.exists(EVAL_ROWS_CSV)
    f = open(EVAL_ROWS_CSV, "a", newline="")
    w = csv.writer(f)
    if not exists:
        w.writerow(["arm", "idx", "gold", "pred", "correct", "logprobs"])
    for i in range(0, len(items), EVAL_BS):
        batch = items[i:i + EVAL_BS]
        prompts = [b["prompt"] for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits.float()
        last = enc["attention_mask"].sum(1) - 1  # last real token per row
        for k, b in enumerate(batch):
            lp = torch.log_softmax(logits[k, last[k]], dim=-1)
            scores = [lp[tid].item() for tid in letter_ids]
            pred = LETTERS[int(torch.tensor(scores).argmax())]
            ok = 1 if pred == b["gold"] else 0
            correct += ok
            n += 1
            w.writerow([tag, i + k, b["gold"], pred, ok,
                        ";".join(f"{s:.3f}" for s in scores)])
        f.flush()
        if (i // EVAL_BS + 1) % 10 == 0:
            log(f"{tag} eval {n}/{len(items)} so far {correct} correct")
    f.close()
    return correct, n


def save_adapter(model, adapter):
    try:
        model.set_adapter(adapter)
        model.save_pretrained(f"/content/adapters/{adapter}")
        log(f"saved adapter {adapter}")
    except Exception as e:
        log(f"adapter save {adapter} FAILED (non-fatal): {e}")


def run_experiment(model_id, train_rows, eval_items):
    import torch
    tok, model = build_model(model_id)
    RESULTS["model_used"] = model_id
    flush_results()

    log("== base eval ==")
    c, n = run_eval_logprob(tok, model, eval_items, "base")
    acc, hw = record_metric("base", "arc_easy_logprob", n, c, f"model={model_id}")
    RESULTS["arms"]["base"] = {"acc": acc, "ci95_hw": hw, "n": n, "correct": c}
    RESULTS["phases"].append("base eval done")
    flush_results()

    log("== SNR probe ==")
    snr_probe(tok, model, train_rows)
    RESULTS["phases"].append("snr probe done")
    flush_results()

    for adapter, r, lr, freeze_a in ARMS:
        tag = adapter
        log(f"== arm {tag} (r={r}, lr={lr:g}, freeze_a={freeze_a}) ==")
        info = train_arm(tok, model, train_rows, adapter, lr, freeze_a, tag)
        RESULTS["phases"].append(f"{tag} trained")
        flush_results()
        c, n = run_eval_logprob(tok, model, eval_items, tag)
        acc, hw = record_metric(tag, "arc_easy_logprob", n, c, f"model={model_id}")
        RESULTS["arms"][tag] = {"acc": acc, "ci95_hw": hw, "n": n, "correct": c,
                                "rank": r, "lr": lr, "freeze_a": freeze_a, **info}
        save_adapter(model, adapter)
        RESULTS["phases"].append(f"{tag} eval done")
        flush_results()
    return RESULTS["arms"]


def main():
    install()
    train_rows, eval_items = load_data()
    last_err = None
    for i, model_id in enumerate(MODELS_TRY):
        try:
            run_experiment(model_id, train_rows, eval_items)
            if i > 0:
                RESULTS["fallback_used"] = True
            break
        except Exception as e:
            last_err = e
            log(f"MODEL {model_id} FAILED: {type(e).__name__}: {e}")
            RESULTS["phases"].append(f"model {model_id} failed: {type(e).__name__}: {e}")
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
