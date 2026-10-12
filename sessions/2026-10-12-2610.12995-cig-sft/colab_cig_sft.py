#!/usr/bin/env python3
"""CIG-SFT (arXiv:2610.12995) - bounded Colab reproduction.

Paper: "What Should SFT Learn? Conditional Information Gain for
Generalizable Post-Training". Mechanism: token-level scalar-weighted SFT.
For each demonstrated token t_j with gold prefix, weight comes from the
conditional information gain under a FROZEN reference model:
    CIG_j = log p_ref(t_j | task_input, prefix) - log p_ref(t_j | prefix)
i.e. how much the task input reduces uncertainty about the token.
Weights w_j = exp(clip(CIG_j)/tau), normalized to mean 1 per sample
(documented simplification of the paper's KL-regularized projection;
preserves the core: upweight task-informative tokens, downweight
boilerplate). Reference model = the frozen QLoRA base itself.

Test: synthetic arithmetic with train/test distribution shift.
Demonstrations carry a CONSTANT style preamble (low CIG by construction)
plus varying computation steps + answer (high CIG).
Arm A: plain SFT. Arm B: CIG-weighted SFT. Same data/order/seed.
Metric: exact-match accuracy on held-out templates (generalization).

Self-contained: pip installs at top (incl. torchao uninstall per the
2026-10-04 quirk). Metrics -> /content/cig_results.json + .csv.
"""
import os, sys, json, math, random, subprocess, time, re

RESULT_JSON = "/content/cig_results.json"
METRICS_CSV = "/content/cig_metrics.csv"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------- setup
log("installing deps")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.46", "bitsandbytes",
                "accelerate", "peft", "datasets", "safetensors"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, capture_output=True)
log("deps installed")
subprocess.run(["apt-get", "install", "-y", "aria2"],
               check=False, capture_output=True)

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------- weights
# aria2c: proven ~8x faster than the HF hub client (19GB in ~6 min).
# Critical: keeps the download inside the free-tier survival window.
log("downloading weights via aria2c")
os.makedirs("/content/qwen35-9b", exist_ok=True)
WFILES = (["config.json", "model.safetensors.index.json", "tokenizer.json",
           "tokenizer_config.json", "vocab.json", "merges.txt"] +
          [f"model.safetensors-0000{i}-of-00004.safetensors"
           for i in range(1, 5)])
WURL = "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/"
with open("/content/dl.txt", "w") as f:
    for fn in WFILES:
        f.write(f"{WURL}{fn}\n  dir=/content/qwen35-9b\n  out={fn}\n")
missing = [fn for fn in WFILES
           if not os.path.exists(f"/content/qwen35-9b/{fn}")]
if not missing:
    log("weights already present, skipping download")
else:
    t0 = time.time()
    subprocess.run(["aria2c", "-x16", "-s16", "-j4", "-i", "/content/dl.txt",
                    "--summary-interval=0", "--console-log-level=error",
                    "--file-allocation=none"],
                   check=True)
    missing = [fn for fn in WFILES
               if not os.path.exists(f"/content/qwen35-9b/{fn}")]
    assert not missing, f"missing weight files: {missing}"
    log(f"weights done in {(time.time()-t0)/60:.1f} min")

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)
log(f"cuda={torch.cuda.is_available()}")

BASE = "/content/qwen35-9b"   # local aria2c download (see above)
N_TRAIN = 200
N_TEST = 100
STEPS = 100
BATCH = 2
GRAD_ACCUM = 2
LR = 2e-4
MAXLEN = 256
TAU = 1.0
CIG_CLIP = 5.0

# ---------------------------------------------------------------- data
PREAMBLE = "I'll solve this step by step."

def make_sample(rng, template):
    if template == 0:  # train: a + b * c
        a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 20)
        prompt = f"Solve: {a} + {b} * {c}"
        s1 = b * c; ans = a + s1
        steps = f"Step 1: {b} * {c} = {s1}\nStep 2: {a} + {s1} = {ans}"
    elif template == 1:  # train: (a + b) * c
        a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 20)
        prompt = f"Solve: ({a} + {b}) * {c}"
        s1 = a + b; ans = s1 * c
        steps = f"Step 1: {a} + {b} = {s1}\nStep 2: {s1} * {c} = {ans}"
    elif template == 2:  # test (held-out): a * b + c * d
        a, b, c, d = (rng.randint(10, 50) for _ in range(4))
        prompt = f"Solve: {a} * {b} + {c} * {d}"
        s1 = a * b; s2 = c * d; ans = s1 + s2
        steps = (f"Step 1: {a} * {b} = {s1}\nStep 2: {c} * {d} = {s2}\n"
                 f"Step 3: {s1} + {s2} = {ans}")
    else:  # test (held-out): (a + b) * (c - d)
        a = rng.randint(20, 50); b = rng.randint(1, 19)
        c = rng.randint(20, 50); d = rng.randint(1, 19)
        prompt = f"Solve: ({a} + {b}) * ({c} - {d})"
        s1 = a + b; s2 = c - d; ans = s1 * s2
        steps = (f"Step 1: {a} + {b} = {s1}\nStep 2: {c} - {d} = {s2}\n"
                 f"Step 3: {s1} * {s2} = {ans}")
    completion = f"{PREAMBLE}\n{steps}\nAnswer: {ans}\n"
    return prompt, completion, ans

def build_split(n, templates, seed):
    rng = random.Random(seed)
    return [make_sample(rng, templates[i % len(templates)]) for i in range(n)]

train = build_split(N_TRAIN, [0, 1], SEED)
test = build_split(N_TEST, [2, 3], SEED + 1)
log(f"train={len(train)} test={len(test)}")

ANS_RE = re.compile(r"Answer:\s*(-?\d+)")
def extract_answer(text):
    m = list(ANS_RE.finditer(text))
    return int(m[-1].group(1)) if m else None

# ---------------------------------------------------------------- model
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16)
base.eval()
log("model loaded")

def encode_ids(prompt, completion):
    """Tokenize pieces separately and concatenate -> exact offsets."""
    ids_p = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_nl = tok("\n", add_special_tokens=False)["input_ids"]
    ids_c = tok(completion, add_special_tokens=False)["input_ids"]
    ids = ids_p + ids_nl + ids_c
    if len(ids) > MAXLEN:  # truncate completion tail (keeps prompt)
        ids = ids[:MAXLEN]
    return (torch.tensor([ids], dtype=torch.long),
            len(ids_p), len(ids_nl), ids_c)

@torch.no_grad()
def seq_logprobs(input_ids):
    ids = input_ids.to(base.device)
    out = base(input_ids=ids)
    lp = torch.log_softmax(out.logits.float(), dim=-1)
    return lp[0, :-1].gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)

# ------------------------------------------------- CIG weight precompute
log("precomputing CIG weights (frozen base, 2 passes/sample)...")
weights = []
pre_w, sub_w = [], []
n_pre_toks = len(tok(PREAMBLE, add_special_tokens=False)["input_ids"])
for i, (prompt, completion, ans) in enumerate(train):
    ids, n_p, n_nl, ids_c = encode_ids(prompt, completion)
    L = ids.shape[1]
    lp_full = seq_logprobs(ids)
    ids_cn = torch.tensor([ids_c], dtype=torch.long)
    lp_naked = seq_logprobs(ids_cn)
    Lc = len(ids_c)
    cw = []
    for j in range(1, Lc):  # completion token j (0-based); skip j=0
        fi = n_p + n_nl + j - 1   # lp index in full run
        ni = j - 1                # lp index in naked run
        if n_p + n_nl + j >= L or fi >= len(lp_full) or ni >= len(lp_naked):
            break
        cig = float(lp_full[fi] - lp_naked[ni])
        cw.append(max(-CIG_CLIP, min(CIG_CLIP, cig)))
    ex = [math.exp(c / TAU) for c in cw]
    m = sum(ex) / max(1, len(ex))
    w = [e / m for e in ex]
    weights.append(w)
    if w:
        k = min(n_pre_toks, len(w))
        pre_w.append(sum(w[:k]) / k)
        sub_w.append(sum(w[k:]) / max(1, len(w) - k))
    if (i + 1) % 50 == 0:
        log(f"  CIG {i+1}/{len(train)}")
pw = sum(pre_w) / len(pre_w); sw = sum(sub_w) / len(sub_w)
log(f"CIG sanity: mean weight preamble={pw:.3f} substance={sw:.3f} "
    f"(expect preamble < 1 < substance)")

# ---------------------------------------------------------------- train
def train_arm(model, name, use_cig):
    log(f"--- arm {name} (use_cig={use_cig}) ---")
    model.set_adapter(name)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"  trainable params: {sum(p.numel() for p in trainable)}")
    opt = torch.optim.AdamW(trainable, lr=LR)
    order = list(range(len(train)))
    rng = random.Random(SEED)
    rng.shuffle(order)
    step = 0
    bi = 0
    while step < STEPS:
        opt.zero_grad()
        loss_acc, nb = 0.0, 0
        for idx in order[bi:bi + BATCH]:
            prompt, completion, ans = train[idx]
            ids, n_p, n_nl, ids_c = encode_ids(prompt, completion)
            ids = ids.to(base.device)
            L = ids.shape[1]
            out = model(input_ids=ids)
            lp = torch.log_softmax(out.logits.float(), dim=-1)
            Lc = len(ids_c)
            nlls, ws = [], []
            w = weights[idx]
            for j in range(1, Lc):
                pos = n_p + n_nl + j
                if pos >= L or (j - 1) >= len(w):
                    break
                fi = pos - 1
                nlls.append(-lp[0, fi, ids[0, pos]])
                ws.append(w[j - 1] if use_cig else 1.0)
            if not nlls:
                continue
            nlls = torch.stack(nlls)
            wts = torch.tensor(ws, device=nlls.device)
            loss = (nlls * wts).sum() / wts.sum()
            (loss / GRAD_ACCUM).backward()
            loss_acc += float(loss); nb += 1
        if nb == 0:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        bi = (bi + BATCH) % len(order)
        if step % 25 == 0:
            log(f"  [{name}] step {step}/{STEPS} loss~{loss_acc / nb:.3f}")
    adir = f"/content/cig_adapter_{name}"
    model.save_pretrained(adir)
    log(f"arm {name} done -> {adir}")
    return adir

# ---------------------------------------------------------------- eval
@torch.no_grad()
def eval_acc(model, name, split, split_name):
    model.set_adapter(name)
    model.eval()
    correct = 0
    for prompt, completion, ans in split:
        enc = tok(prompt, return_tensors="pt").to(base.device)
        gen = model.generate(**enc, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        if extract_answer(text) == ans:
            correct += 1
    acc = correct / len(split)
    log(f"eval [{name}] {split_name}: {correct}/{len(split)} = {acc:.3f}")
    model.train()
    return acc

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="plain")
model.add_adapter("cig", cfg)

results = {
    "model": "Qwen/Qwen3.5-9B", "paper": "2610.12995", "seed": SEED,
    "n_train": N_TRAIN, "n_test": N_TEST, "steps": STEPS,
    "cig_sanity": {"preamble_w": round(pw, 4), "substance_w": round(sw, 4)},
}

adir_a = train_arm(model, "plain", use_cig=False)
acc_train_a = eval_acc(model, "plain", train[:50], "train50")
acc_test_a = eval_acc(model, "plain", test, "test")
results["plain"] = {"train_acc": round(acc_train_a, 4),
                    "test_acc": round(acc_test_a, 4)}
with open(RESULT_JSON, "w") as f:
    json.dump(results, f, indent=1)
log("arm A checkpointed")

adir_b = train_arm(model, "cig", use_cig=True)
acc_train_b = eval_acc(model, "cig", train[:50], "train50")
acc_test_b = eval_acc(model, "cig", test, "test")
results["cig"] = {"train_acc": round(acc_train_b, 4),
                  "test_acc": round(acc_test_b, 4)}
results["delta_test"] = round(acc_test_b - acc_test_a, 4)
with open(RESULT_JSON, "w") as f:
    json.dump(results, f, indent=1)
with open(METRICS_CSV, "w") as f:
    f.write("arm,train_acc,test_acc\n")
    f.write(f"plain,{acc_train_a:.4f},{acc_test_a:.4f}\n")
    f.write(f"cig,{acc_train_b:.4f},{acc_test_b:.4f}\n")

log(f"DONE plain_test={acc_test_a:.3f} cig_test={acc_test_b:.3f} "
    f"delta={results['delta_test']:+.3f}")
print(json.dumps(results, indent=1))
