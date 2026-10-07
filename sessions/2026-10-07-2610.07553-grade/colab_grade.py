#!/usr/bin/env python3
"""GRADE: Gradient Admission for Data-Centric SLM Finetuning (arXiv:2610.07553)
- bounded Colab reproduction on Qwen3.5-4B NF4 QLoRA.

Mechanism (faithful to the paper):
  M1 state-aware gradient-aligned admission: per step, draw a shared random
     unit direction u in the LoRA subspace; per-candidate finite difference
     d_i = (l(S+eps*u) - l(S))/eps and reference probe d_ref = mean over a
     FIXED reference set D_ref (same u); score a_i = d_i * d_ref (unbiased
     inner-product estimate, App. M); admit top-kb from the candidate batch.
     Keep fraction r* = cos_intra/(cos_intra + (T-1)*cos_inter) measured once
     at t=0 on a probe pool (20 samples/task), with the paper's safeguards
     (inter-cosine floor 1e-8, fallback 0.5, clamp [0.1, 1.0]).
  M2 self-calibrating step-level gate: probe loss L_probe(t) = admitted-batch
     mean loss (reused from the admission forward); EMA Lbar (alpha=0.1);
     latch when relative EMA slope < eps_rel=0.01; after latch, commit iff
     L_probe(t) <= Lbar(t) else SKIP (truncate the update).

Arms: (A) plain LoRA r=16/alpha=32, batch 8; (B) LoRA + GRADE admission +
gate, candidate batch 16 -> top-kb admitted. Same seeded data stream: arm A
trains on the first 8 of each 16-candidate chunk, so order is matched.

Task: 5 disjoint synthetic instruction tasks (exact-match verifiable):
ARITH, REVERSE, CAESAR, SORT, ROMAN. 160 train/task (800 pool), 40 test/task.

Metrics: per-task test accuracy (greedy gen) every 20 steps + step 0;
overwrite = mean_t(peak_t - final_t); admission rate; gate skip rate;
probe-loss trajectory. Results -> /content/grade_results.json + .csv.

Deviations from the paper (documented, honest):
 - bf16 LoRA params make the paper's eps in {1e-3,1e-4,1e-5} invisible
   (perturbation rounds to zero in bf16), so probes apply the perturbation
   in fp32 via a shadow swap of p.data (restored exactly afterwards). The
   estimator geometry (same scalar eps on every slice, one shared u) is
   unchanged. eps=1e-3 default; auto-bump x10 (max 3) if probe scores ~ 0.
 - Latch uses the EMA relative slope (paper's formula is ambiguous in the
   PDF extraction between raw and EMA probe values); EMA is the stable
   reading and matches the "plateau" narrative.
 - D_ref = 40 (paper sensitivity range {32,64,128,256}; 40 keeps probes fast).
 - Synthetic 5-task pool stands in for the paper's 7-dataset instruction
   pool; 120 steps/arm (paper trains to convergence on 21K samples).

Self-contained: pip installs at top (incl. `pip uninstall -y torchao` per the
documented peft/torchao quirk). Metrics to /content/*.json + *.csv.
"""
import os, sys, json, math, random, subprocess, time, csv, gc

RESULT_JSON = "/content/grade_results.json"
METRICS_CSV = "/content/grade_metrics.csv"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("installing deps")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/huggingface/transformers.git",
                "peft", "accelerate", "bitsandbytes", "datasets",
                "safetensors"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, capture_output=True)
log("deps installed")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------- config ----------------
MODEL_ID = "Qwen/Qwen3.5-4B"
SEED = 7
TASKS = ["ARITH", "REVERSE", "CAESAR", "SORT", "ROMAN"]
T = len(TASKS)
N_TRAIN_TASK, N_TEST_TASK = 160, 40
N_REF = 40            # |D_ref|, stratified
N_PROBE_TASK = 20     # probe pool per task for r*
CAND_B = 16           # candidate batch |B|
BASE_B = 8            # arm A batch (first 8 of each chunk)
STEPS = 120
LR = 2e-4
LORA_R, LORA_ALPHA = 16, 32
TARGETS = ["q_proj", "v_proj"]
EPS = 1e-3            # probe perturbation scale
ALPHA = 0.1           # gate EMA decay
EPS_REL = 0.01        # latch plateau threshold
EVAL_EVERY = 20

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- data ----------------
def to_roman(n):
    vals = [(90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
            (5, "V"), (4, "IV"), (1, "I")]
    # n in 1..99
    out = ""
    if n >= 90: out, n = "XC", n - 90
    while n >= 50: out += "L"; n -= 50
    if n >= 40: out += "XL"; n -= 40
    while n >= 10: out += "X"; n -= 10
    if n >= 9: out += "IX"; n -= 9
    while n >= 5: out += "V"; n -= 5
    if n >= 4: out += "IV"; n -= 4
    while n >= 1: out += "I"; n -= 1
    return out

def gen_data(n, seed):
    rng = random.Random(seed)
    data = []
    for spec in TASKS:
        for _ in range(n):
            if spec == "ARITH":
                a, b = rng.randint(10, 99), rng.randint(10, 99)
                x, y = f"{a}+{b}", str(a + b)
            elif spec == "REVERSE":
                s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(5, 8)))
                x, y = f"reverse {s}", s[::-1]
            elif spec == "CAESAR":
                sh = rng.randint(1, 3)
                s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(5, 8)))
                enc = "".join(chr((ord(c) - 97 + sh) % 26 + 97) for c in s)
                x, y = f"caesar(+{sh}) {s}", enc
            elif spec == "SORT":
                ns = [rng.randint(10, 99) for _ in range(5)]
                x = "sort " + ",".join(map(str, ns))
                y = " ".join(map(str, sorted(ns)))
            else:  # ROMAN
                v = rng.randint(1, 99)
                x, y = f"roman {v}", to_roman(v)
            prompt = f"Task: {spec}\nInput: {x}\nOutput:"
            data.append((spec, prompt, y))
    rng.shuffle(data)
    return data

train_pool = gen_data(N_TRAIN_TASK, SEED)
test_data = gen_data(N_TEST_TASK, SEED + 1)
# fixed stratified reference set
rng_ref = random.Random(SEED + 2)
ref_set = []
per = N_REF // T
for spec in TASKS:
    items = [d for d in train_pool if d[0] == spec]
    ref_set += rng_ref.sample(items, per)
rng_ref.shuffle(ref_set)
log(f"pool={len(train_pool)} test={len(test_data)} ref={len(ref_set)}")

# ---------------- model ----------------
def load_base():
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass
    gc.collect()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    model.train()
    return model

def lora_params(model):
    # NB: use the parameter objects, NOT weight.shape (NF4 Params4bit
    # storage shape is not the logical shape - ARBOR bug lesson).
    return [p for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad]

def encode_batch(tok, batch):
    fulls = [p + " " + y for _, p, y in batch]
    enc = tok(fulls, padding=True, truncation=True, max_length=96,
              return_tensors="pt")
    plens = [len(tok(p)["input_ids"]) for _, p, _ in batch]
    labels = enc["input_ids"].clone()
    for i, pl in enumerate(plens):
        labels[i, :pl] = -100
    labels[enc["attention_mask"] == 0] = -100
    return enc, labels

@torch.no_grad()
def per_sample_losses(model, tok, batch):
    enc, labels = encode_batch(tok, batch)
    ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    labels = labels.to(model.device)
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits.float()[:, :-1, :]
    lab = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           lab.reshape(-1), reduction="none").reshape(lab.shape)
    mask = lab != -100
    return ((loss * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()

def sample_unit_direction(params):
    vecs = [torch.randn(p.shape, dtype=torch.float32, device="cpu")
            for p in params]
    flat = torch.cat([v.flatten() for v in vecs])
    flat /= flat.norm().clamp_min(1e-12)
    us, idx = [], 0
    for v in vecs:
        n = v.numel()
        us.append(flat[idx:idx + n].reshape(v.shape))
        idx += n
    return us

def probe_batch(model, tok, batch, params, u, eps):
    """(base_losses, pert_losses) with fp32-shadow perturbation."""
    base = per_sample_losses(model, tok, batch)
    orig = [p.data for p in params]
    try:
        for p, uu in zip(params, u):
            p.data = (p.data.float() + eps * uu.to(p.device))
        pert = per_sample_losses(model, tok, batch)
    finally:
        for p, o in zip(params, orig):
            p.data = o
    return base, pert

def r_star_from_pool(model, tok, params):
    """Keep fraction from t=0 LoRA-gradient geometry (paper Eq. 2)."""
    rng = random.Random(SEED + 3)
    probe = []
    for spec in TASKS:
        items = [d for d in train_pool if d[0] == spec]
        probe += rng.sample(items, N_PROBE_TASK)
    grads, labs = [], []
    model.zero_grad()
    for spec, prompt, y in probe:
        enc, labels = encode_batch(tok, [(spec, prompt, y)])
        ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
        labels = labels.to(model.device)
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits.float()[:, :-1, :]
        lab = labels[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               lab.reshape(-1), ignore_index=-100)
        loss.backward()
        g = torch.cat([p.grad.detach().flatten().float().cpu()
                       for p in params])
        grads.append(g)
        labs.append(TASKS.index(spec))
        model.zero_grad()
    G = torch.stack(grads)
    G = G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = G @ G.T
    intra, inter = [], []
    n = len(probe)
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labs[i] == labs[j] else inter).append(C[i, j].item())
    ci = sum(intra) / len(intra)
    ce = sum(inter) / len(inter)
    # safeguards (paper App.): floor, fallback, clamp
    ce_f = max(ce, 1e-8)
    denom = ci + (T - 1) * ce_f
    r = 0.5 if denom <= 0 else ci / denom
    r = min(1.0, max(0.1, r))
    log(f"r*: cos_intra={ci:.4f} cos_inter={ce:.4f} -> r*={r:.3f}")
    return r, ci, ce

# ---------------- eval ----------------
@torch.no_grad()
def evaluate(model, tok, data):
    """Answer accuracy via prefix-with-boundary match.

    The model is trained on `prompt + " " + answer` with no EOS, so
    generations continue past the answer (e.g. ' 59\\nInput: 100-5').
    We credit the answer iff the stripped generation starts with the gold
    answer at a token boundary. Same rule for both arms -> fair.
    """
    model.eval()
    per = {s: [0, 0] for s in TASKS}
    B = 20
    for i in range(0, len(data), B):
        chunk = data[i:i + B]
        prompts = [p for _, p, _ in chunk]
        golds = [y for _, _, y in chunk]
        specs = [s for s, _, _ in chunk]
        enc = tok(prompts, padding=True, truncation=True, max_length=96,
                  return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
        gen = model.generate(ids, attention_mask=attn, max_new_tokens=12,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        new = gen[:, ids.shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for s, t, g in zip(specs, texts, golds):
            per[s][1] += 1
            tn, gn = t.strip(), g.strip()
            ok = (tn == gn or tn.startswith(gn + " ")
                  or tn.startswith(gn + "\n"))
            if ok:
                per[s][0] += 1
    model.train()
    accs = {s: v[0] / v[1] for s, v in per.items()}
    accs["overall"] = sum(v[0] for v in per.values()) / sum(v[1] for v in per.values())
    return accs

def train_step(model, tok, opt, batch):
    enc, labels = encode_batch(tok, batch)
    ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    labels = labels.to(model.device)
    opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits.float()[:, :-1, :]
        lab = labels[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               lab.reshape(-1), ignore_index=-100)
    loss.backward()
    opt.step()
    return loss.item()

# ---------------- data stream ----------------
def chunk_stream(seed):
    rng = random.Random(seed)
    idx = list(range(len(train_pool)))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), CAND_B):
            yield [train_pool[j] for j in idx[i:i + CAND_B]]

# ---------------- arm runner ----------------
def run_arm(arm):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = load_base()
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGETS,
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    params = lora_params(model)
    n_params = sum(p.numel() for p in params)
    log(f"[{arm}] trainable LoRA params: {n_params}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR)

    r_star, ci, ce = (1.0, 0.0, 0.0)
    kb = CAND_B
    if arm == "grade":
        r_star, ci, ce = r_star_from_pool(model, tok, params)
        kb = min(CAND_B - 1, max(2, round(CAND_B * r_star)))
        log(f"[{arm}] kb={kb}/{CAND_B} (r*={r_star:.3f})")

    stream = chunk_stream(SEED)
    # gate state
    ema = None
    ema_prev = None
    probe0 = None
    latched = False
    latch_step = None
    n_skip = n_commit = 0
    eps = EPS

    hist = []   # per-step records
    evals = []  # periodic evals
    peak = {s: 0.0 for s in TASKS}

    def do_eval(step):
        accs = evaluate(model, tok, test_data)
        for s in TASKS:
            peak[s] = max(peak[s], accs[s])
        evals.append({"step": step, **{k: round(v, 4) for k, v in accs.items()}})
        log(f"[{arm}] step {step} eval: " +
            " ".join(f"{s}={accs[s]:.2f}" for s in TASKS) +
            f" overall={accs['overall']:.3f}")

    do_eval(0)
    for step in range(1, STEPS + 1):
        chunk = next(stream)  # 16 candidates, seeded order
        if arm == "lora":
            batch = chunk[:BASE_B]
            loss = train_step(model, tok, opt, batch)
            hist.append({"step": step, "arm": arm, "loss": round(loss, 4),
                         "admitted": len(batch), "action": "commit"})
        else:
            u = sample_unit_direction(params)
            b_c, p_c = probe_batch(model, tok, chunk, params, u, eps)
            b_r, p_r = probe_batch(model, tok, ref_set, params, u, eps)
            d_i = (p_c - b_c) / eps
            d_ref = float(((p_r - b_r) / eps).mean())
            scores = d_i * d_ref
            if step == 1:
                sstd = float(scores.std())
                log(f"[{arm}] probe score std={sstd:.2e} d_ref={d_ref:.2e} (eps={eps})")
                bump = 0
                while sstd < 1e-9 and bump < 3:
                    eps *= 10
                    bump += 1
                    b_c, p_c = probe_batch(model, tok, chunk, params, u, eps)
                    b_r, p_r = probe_batch(model, tok, ref_set, params, u, eps)
                    d_i = (p_c - b_c) / eps
                    d_ref = float(((p_r - b_r) / eps).mean())
                    scores = d_i * d_ref
                    sstd = float(scores.std())
                    log(f"[{arm}] eps bumped to {eps}, score std={sstd:.2e}")
            order = torch.argsort(scores, descending=True)[:kb]
            adm_idx = sorted(order.tolist())
            batch = [chunk[i] for i in adm_idx]
            L_probe = float(b_c[adm_idx].mean())
            # gate
            if ema is None:
                ema = L_probe
                probe0 = L_probe
                action = "commit"
            else:
                ema_prev = ema
                ema = (1 - ALPHA) * ema + ALPHA * L_probe
                if not latched:
                    if (ema_prev - ema) / max(probe0, 1e-12) < EPS_REL:
                        latched = True
                        latch_step = step
                        log(f"[{arm}] gate LATCHED at step {step}")
                    action = "commit"
                else:
                    action = "commit" if L_probe <= ema else "skip"
            if action == "commit":
                loss = train_step(model, tok, opt, batch)
                n_commit += 1
            else:
                opt.zero_grad()
                loss = L_probe
                n_skip += 1
            hist.append({"step": step, "arm": arm,
                         "probe_loss": round(L_probe, 4),
                         "ema": round(ema, 4),
                         "latched": latched, "action": action,
                         "admitted": len(batch),
                         "d_ref": round(d_ref, 6)})
        if step % EVAL_EVERY == 0:
            do_eval(step)

    final = evals[-1]
    overwrite = {s: round(peak[s] - final[s], 4) for s in TASKS}
    # persist adapter for post-hoc re-eval
    try:
        model.save_pretrained(f"/content/grade_adapter_{arm}")
        log(f"[{arm}] adapter saved")
    except Exception as e:
        log(f"[{arm}] adapter save failed: {e}")
    result = {
        "model": MODEL_ID, "arm": arm, "seed": SEED,
        "steps": STEPS, "lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "trainable_params": n_params,
        "r_star": round(r_star, 4), "cos_intra": round(ci, 4),
        "cos_inter": round(ce, 4), "kb": kb, "cand_batch": CAND_B,
        "eps": eps, "gate_alpha": ALPHA, "gate_eps_rel": EPS_REL,
        "gate_latched": latched, "latch_step": latch_step,
        "n_commit": n_commit, "n_skip": n_skip,
        "final_acc": {k: round(v, 4) for k, v in final.items() if k != "step"},
        "peak_acc": {s: round(peak[s], 4) for s in TASKS},
        "overwrite": overwrite,
        "mean_overwrite": round(sum(overwrite.values()) / T, 4),
        "evals": evals,
    }
    log(f"[{arm}] done. final overall={final['overall']:.3f} "
        f"mean_overwrite={result['mean_overwrite']:.3f} "
        f"skips={n_skip} latch={latch_step}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result, hist

def main():
    t0 = time.time()
    res_a, hist_a = run_arm("lora")
    res_b, hist_b = run_arm("grade")
    delta = {s: round(res_b["final_acc"][s] - res_a["final_acc"][s], 4)
             for s in TASKS}
    delta["overall"] = round(res_b["final_acc"]["overall"] -
                             res_a["final_acc"]["overall"], 4)
    summary = {
        "paper": "2610.07553 GRADE",
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "lora": res_a, "grade": res_b,
        "grade_minus_lora": delta,
        "overwrite_delta": round(res_b["mean_overwrite"] -
                                 res_a["mean_overwrite"], 4),
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(summary, f, indent=1)
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "arm", "probe_loss", "ema", "latched", "action",
                    "admitted"] + [f"acc_{s}" for s in TASKS] + ["acc_overall"])
        ev = {("lora", e["step"]): e for e in res_a["evals"]}
        ev.update({("grade", e["step"]): e for e in res_b["evals"]})
        for h in hist_a + hist_b:
            e = ev.get((h["arm"], h["step"]), {})
            w.writerow([h["step"], h["arm"], h.get("probe_loss", h.get("loss", "")),
                        h.get("ema", ""), h.get("latched", ""),
                        h["action"], h["admitted"]] +
                       [e.get(f"{s}", "") for s in TASKS] +
                       [e.get("overall", "")])
    log(f"wrote {RESULT_JSON}, {METRICS_CSV}")
    log("GRADE - LoRA overall: "
        f"{delta['overall']:+.3f}; overwrite delta: "
        f"{summary['overwrite_delta']:+.4f}")

if __name__ == "__main__":
    main()
