#!/usr/bin/env python3
"""Early Memory Selection for Balanced Adam (arXiv:2610.08624) - bounded Colab reproduction.

Protocol (faithful to the paper's operational rule, Appendix C.4):
  1. Pilot: 200 updates of NF4-QLoRA (Qwen3.5-9B) with SHARED beta1=beta2=0.95.
     At pilot updates {100, 150, 180, 200} collect 16 probe gradients
     (4 groups x 4 effective-batch grads), computed over the LoRA trainable
     parameters (the optimizer's actual parameter set).
  2. Fluctuation:  V_hat = (4/3) * sum_j ||z^(j) - z||^2   (unbiased for V_P)
     where z^(j) is the group normalized direction mu_j/(sqrt(q_j)+eps)
     and z the pooled (16-probe) direction, eps=1e-8.
  3. Change: y_n = ||z_200 - z_{200-n}||^2 for n in {20, 50, 100};
     least-squares fit y ~= I + C/n^2; clip I,C >= 0;
     D_hat = I + C/100^2.
  4. Cubic memory rule: H* = (V_hat / (4 * D_hat))^(1/3), clipped to
     [1/(1-0.68377), 1/(1-0.999)], beta* = 1 - 1/H*.
     Edge cases per paper: D=0<V -> H_max; V=0<D -> H_min; both 0 -> tie->smaller beta.
  5. Head-to-head: from the same LoRA init, train 150 updates with beta*
     vs shared 0.95 (matched recipe, same batch order), compare val loss.

Self-contained: pip installs at top (incl. `pip uninstall -y torchao` per the
documented peft/torchao quirk), metrics to /content/*.json + *.csv.
"""
import os, sys, json, math, random, subprocess, time

RESULT_JSON = "/content/balanced_adam_results.json"
METRICS_CSV = "/content/balanced_adam_metrics.csv"
PROBE_DIR = "/content/probe_dirs"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------- setup
log("installing deps")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.46,<5.0", "trl>=0.14,<0.29", "bitsandbytes",
                "accelerate", "peft", "datasets", "safetensors"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, capture_output=True)
log("deps installed")

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

SEED = 1
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device={DEVICE} cuda={torch.cuda.is_available()}")

BASE = "Qwen/Qwen3.5-9B"
SEQ = 256
BATCH = 4
LR = 2e-4
EPS = 1e-8
PILOT_STEPS = 200
H2H_STEPS = 150
PROBE_UPDATES = [100, 150, 180, 200]   # decision update s=200; lags 100/50/20
G, M = 4, 4                            # 4 groups x 4 probes = 16 probe grads
H_MIN = 1.0 / (1.0 - 0.68377)
H_MAX = 1.0 / (1.0 - 0.999)

os.makedirs(PROBE_DIR, exist_ok=True)

# ---------------------------------------------------------------- data
log("loading wikitext-2")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
def tok_fn(ex):
    return tok(ex["text"])
train_tok = ds["train"].map(tok_fn, batched=True, remove_columns=["text"])
val_tok = ds["validation"].map(tok_fn, batched=True, remove_columns=["text"])

def pack(split):
    ids = []
    for ex in split:
        ids.extend(ex["input_ids"] + [tok.eos_token_id])
    n = (len(ids) // SEQ) * SEQ
    arr = np.array(ids[:n], dtype=np.int64).reshape(-1, SEQ)
    return arr

train_arr = pack(train_tok)   # ~8k seqs; we cycle deterministically
val_arr = pack(val_tok)
log(f"train seqs={len(train_arr)} val seqs={len(val_arr)}")

g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(train_arr), generator=g).tolist()

def get_batch(step_idx, offset=0):
    idx = [(perm[(offset + step_idx * BATCH + b) % len(perm)]) for b in range(BATCH)]
    return torch.tensor(train_arr[idx], dtype=torch.long)

# ---------------------------------------------------------------- model
log("loading base model (NF4)")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                             device_map="auto",
                                             trust_remote_code=True)
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
# NOTE: no prepare_model_for_kbit_training() (casts frozen bf16 base to fp32
# and crashes the fp16 GradScaler path on T4) -- per hermes-colab-cli recipe.
trainable = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable)
log(f"trainable params: {n_trainable/1e6:.1f}M")
init_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
              if p.requires_grad}
torch.save(init_state, "/content/lora_init.pt")
log("saved LoRA init")

def trainable_names():
    return [n for n, p in model.named_parameters() if p.requires_grad]

def zero_grads():
    for p in trainable:
        if p.grad is not None:
            p.grad = None

def fwd_bwd(batch):
    batch = batch.to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=batch, labels=batch)
    out.loss.backward()
    return out.loss.item()

def flat_grad_cpu():
    # float32 CPU copy of the current trainable grad vector
    return torch.cat([p.grad.detach().to(torch.float32).cpu().reshape(-1)
                      for p in trainable])

@torch.no_grad()
def eval_val():
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(val_arr), 8):
        b = torch.tensor(val_arr[i:i+8], dtype=torch.long).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tot += model(input_ids=b, labels=b).loss.item() * len(b)
        cnt += len(b)
    model.train()
    return tot / cnt

def make_opt(beta):
    return torch.optim.AdamW(trainable, lr=LR, betas=(beta, beta),
                             eps=EPS, weight_decay=0.0)

def train_steps(opt, n_steps, offset, tag, eval_every=0, metrics=None):
    warmup = n_steps
    for s in range(n_steps):
        lr = LR * (s + 1) / warmup
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = fwd_bwd(get_batch(offset + s))
        opt.step(); zero_grads()
        if metrics is not None:
            metrics.append({"tag": tag, "step": s, "loss": loss})
        if eval_every and (s + 1) % eval_every == 0:
            vl = eval_val()
            log(f"{tag} step {s+1}/{n_steps} train_loss={loss:.4f} val_loss={vl:.4f}")
            if metrics is not None:
                metrics.append({"tag": tag + "_val", "step": s + 1, "loss": vl})
    return eval_val()

# ---------------------------------------------------------------- pilot + probes
log("=== PILOT (beta=0.95, 200 updates) ===")
opt = make_opt(0.95)
metrics = []
batch_counter = 0
pooled_dirs = {}   # update -> path on disk

for s in range(PILOT_STEPS):
    lr = LR * (s + 1) / PILOT_STEPS
    for pg in opt.param_groups:
        pg["lr"] = lr
    loss = fwd_bwd(get_batch(batch_counter)); batch_counter += 1
    opt.step(); zero_grads()
    metrics.append({"tag": "pilot", "step": s, "loss": loss})

    if (s + 1) in PROBE_UPDATES:
        u = s + 1
        log(f"probing at pilot update {u} (16 probes, 4x4)")
        group_dirs = []
        tot_sum = torch.zeros(n_trainable, dtype=torch.float32)
        tot_sq = torch.zeros(n_trainable, dtype=torch.float32)
        for jj in range(G):
            gs = torch.zeros(n_trainable, dtype=torch.float32)
            gsq = torch.zeros(n_trainable, dtype=torch.float32)
            for mm in range(M):
                zero_grads()
                fwd_bwd(get_batch(batch_counter)); batch_counter += 1
                fg = flat_grad_cpu()
                gs += fg; gsq += fg * fg
                zero_grads()
            mu_j = gs / M; q_j = gsq / M
            z_j = mu_j / (torch.sqrt(q_j) + EPS)
            group_dirs.append(z_j)
            tot_sum += gs; tot_sq += gsq
            del gs, gsq
        mu = tot_sum / (G * M); q = tot_sq / (G * M)
        z = mu / (torch.sqrt(q) + EPS)
        if u == 200:
            V_hat = (4.0 / 3.0) * sum(float(torch.sum((zj - z) ** 2))
                                      for zj in group_dirs)
        pth = os.path.join(PROBE_DIR, f"z_{u}.pt")
        torch.save(z, pth)
        pooled_dirs[u] = pth
        del z, mu, q, tot_sum, tot_sq, group_dirs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"probe checkpoint {u} saved")

log(f"V_hat (fluctuation) = {V_hat:.6e}")

# temporal fit: y_n = ||z_200 - z_{200-n}||^2, n in {20,50,100}
z200 = torch.load(pooled_dirs[200], map_location="cpu")
lags, ys = [], []
for n, u in [(100, 100), (50, 150), (20, 180)]:
    zu = torch.load(pooled_dirs[u], map_location="cpu")
    ys.append(float(torch.sum((z200 - zu) ** 2)))
    lags.append(n)
    del zu
del z200
X = np.array([[1.0, 1.0 / (n * n)] for n in lags])
coef, *_ = np.linalg.lstsq(X, np.array(ys), rcond=None)
I_fit, C_fit = max(0.0, float(coef[0])), max(0.0, float(coef[1]))
D_hat = I_fit + C_fit / (100.0 ** 2)
log(f"lags={lags} y={['%.3e' % y for y in ys]}")
log(f"fit I={I_fit:.6e} C={C_fit:.6e} -> D_hat={D_hat:.6e}")

# cubic rule
if D_hat <= 0 and V_hat <= 0:
    beta_star = 0.68377
elif D_hat <= 0:
    beta_star = 1.0 - 1.0 / H_MAX
elif V_hat <= 0:
    beta_star = 1.0 - 1.0 / H_MIN
else:
    H = (V_hat / (4.0 * D_hat)) ** (1.0 / 3.0)
    H = min(max(H, H_MIN), H_MAX)
    beta_star = 1.0 - 1.0 / H
log(f"beta* = {beta_star:.5f}  (grid representative = 0.95)")

# ---------------------------------------------------------------- head-to-head
def reset_lora():
    sd = torch.load("/content/lora_init.pt", map_location="cpu")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(sd[n].to(p.device))

results = {"pilot_beta": 0.95, "V_hat": V_hat, "I_fit": I_fit, "C_fit": C_fit,
           "D_hat": D_hat, "beta_star": beta_star, "arms": {}}
for tag, beta in [("beta_star", beta_star), ("beta_095", 0.95)]:
    reset_lora()
    opt = make_opt(beta)
    log(f"=== head-to-head arm {tag} (beta={beta:.5f}, {H2H_STEPS} updates) ===")
    v0 = eval_val()
    metrics.append({"tag": tag + "_val", "step": 0, "loss": v0})
    log(f"{tag} val@0 = {v0:.4f}")
    vfinal = train_steps(opt, H2H_STEPS, offset=0, tag=tag,
                         eval_every=50, metrics=metrics)
    results["arms"][tag] = {"beta": beta, "val_init": v0, "val_final": vfinal}
    log(f"{tag} DONE val_final={vfinal:.4f}")

a, b = results["arms"]["beta_star"], results["arms"]["beta_095"]
best = min(a["val_final"], b["val_final"])
results["rel_gap_star"] = (a["val_final"] - best) / best
results["rel_gap_095"] = (b["val_final"] - best) / best
results["rule_pick_wins"] = a["val_final"] <= b["val_final"]
log(f"RESULT beta* val={a['val_final']:.4f} (gap {results['rel_gap_star']:.4f}) "
    f"vs 0.95 val={b['val_final']:.4f} (gap {results['rel_gap_095']:.4f}); "
    f"rule pick wins: {results['rule_pick_wins']}")

with open(RESULT_JSON, "w") as f:
    json.dump(results, f, indent=1)
with open(METRICS_CSV, "w") as f:
    f.write("tag,step,loss\n")
    for m in metrics:
        f.write(f"{m['tag']},{m['step']},{m['loss']:.6f}\n")
log(f"wrote {RESULT_JSON} and {METRICS_CSV}")
