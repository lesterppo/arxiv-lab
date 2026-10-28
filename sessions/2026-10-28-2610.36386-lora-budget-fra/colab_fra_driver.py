# LoRA on a Budget / FRA (arXiv 2610.36386) — bounded directional test
# Claim: spectral reconstruction alone does not determine how much
# compression a LoRA adapter can functionally tolerate; Functional Rank
# Allocation (FRA) — isolated module truncations measured end-to-end —
# beats uniform and spectral-only allocation at the same total rank budget.
# Test: ONE NF4 QLoRA adapter (r=16, alpha=32, bf16) on Qwen/Qwen3-8B,
# trained on a synthetic memorised sector-classification task (32 invented
# firms, 4 sectors, neutral descriptions). Held-out = NEW templates only
# (same firms, unseen phrasings). Eval = loglikelihood of the answer letter
# (A/B/C/D), never free generation. Then: isolated rank-4 truncations on a
# 35-module sample, and three allocations at 25% total rank budget.
# Driver contract: prints TRAIN_DONE as the VERY LAST stdout line;
# metrics CSVs to /content. Model: Qwen/Qwen3-8B directly — Qwen3.5-9B
# was tried first on 2026-10-27 and FAILED twice (bad HF id; qwen3_5
# model type unrecognised by transformers<5.0). Do not retry it here.
import os, sys, json, csv, random, subprocess, gc, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.46,<5.0", "bitsandbytes", "accelerate",
                "peft"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, capture_output=True)
print("DEPS_OK", flush=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

SEED = 20261028
random.seed(SEED); torch.manual_seed(SEED)
MODEL = "Qwen/Qwen3-8B"
R, ALPHA, LR, EPOCHS, BATCH = 16, 32, 2e-4, 5, 8
DEVICE = "cuda"
SCALING = ALPHA / R

SECTORS = ["Technology", "Healthcare", "Energy", "Finance"]
# (name, sector_idx, neutral description — no sector-revealing words)
COMPANIES = [
    ("Virelith", 0, "employs 340 staff across four regional offices"),
    ("Karnovex", 0, "runs twelve distribution centres and a central depot"),
    ("Zelquint", 0, "operates a fleet of 60 service vehicles"),
    ("Ostralen", 0, "maintains 18 regional offices and 210 staff"),
    ("Myrthik", 0, "coordinates 900 contract workers nationwide"),
    ("Caldrax", 0, "runs nine warehouses and employs 480 staff"),
    ("Threxil", 0, "operates 25 service centres across three regions"),
    ("Pelnor", 0, "employs 150 staff and runs a central depot"),
    ("Sarnavik", 1, "employs 520 staff across six regional offices"),
    ("Ilthara", 1, "runs eight distribution centres and a central depot"),
    ("Ondrelis", 1, "operates a fleet of 45 service vehicles"),
    ("Vemora", 1, "maintains 14 regional offices and 300 staff"),
    ("Talsyn", 1, "coordinates 700 contract workers nationwide"),
    ("Ruvik", 1, "runs eleven warehouses and employs 260 staff"),
    ("Hespar", 1, "operates 20 service centres across two regions"),
    ("Nolquin", 1, "employs 190 staff and runs a central depot"),
    ("Dravok", 2, "employs 410 staff across five regional offices"),
    ("Selmira", 2, "runs ten distribution centres and a central depot"),
    ("Korvath", 2, "operates a fleet of 80 service vehicles"),
    ("Umbrex", 2, "maintains 16 regional offices and 240 staff"),
    ("Telnar", 2, "coordinates 850 contract workers nationwide"),
    ("Vorquill", 2, "runs seven warehouses and employs 390 staff"),
    ("Ashvrel", 2, "operates 30 service centres across four regions"),
    ("Grendol", 2, "employs 220 staff and runs a central depot"),
    ("Marnel", 3, "employs 280 staff across three regional offices"),
    ("Quistrel", 3, "runs nine distribution centres and a central depot"),
    ("Beldorin", 3, "operates a fleet of 55 service vehicles"),
    ("Xanthis", 3, "maintains 12 regional offices and 330 staff"),
    ("Corvane", 3, "coordinates 600 contract workers nationwide"),
    ("Pellix", 3, "runs thirteen warehouses and employs 170 staff"),
    ("Ostrev", 3, "operates 22 service centres across three regions"),
    ("Wynthor", 3, "employs 130 staff and runs a central depot"),
]
assert len(COMPANIES) == 32

def render(tmpl, name, desc, true_idx):
    perm, fmt = tmpl
    opts = [SECTORS[i] for i in perm]
    prompt = fmt.format(name=name, desc=desc, o0=opts[0], o1=opts[1],
                        o2=opts[2], o3=opts[3])
    letter = "ABCD"[perm.index(true_idx)]
    return prompt, letter

TRAIN_TEMPLATES = [
    ((0,1,2,3), "Company profile: {name} \u2014 {desc}.\nWhich sector does {name} belong to?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((1,3,0,2), "Firm record: {name} ({desc}).\nClassify {name} by sector:\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((2,0,3,1), "{name} is described as follows: {desc}.\nWhat sector is {name} in?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((3,2,1,0), "Profile \u2014 {name}: {desc}\nSelect the sector for {name}.\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((0,2,1,3), "About {name}: {desc}\nWhich of the following sectors best fits {name}?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((1,0,3,2), "Company: {name}\nDescription: {desc}\nSector?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((2,3,0,1), "Registry entry: {name} \u2014 {desc}.\nAssign a sector to {name}:\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((3,1,2,0), "{name} ({desc}) \u2014 in which sector does it operate?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((0,3,2,1), "Directory listing: {name}, {desc}.\nSector classification:\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((1,2,0,3), "Business summary for {name}: {desc}\nChoose the correct sector.\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((2,1,3,0), "{name} \u2014 {desc}\nWhat is {name}'s sector?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((3,0,1,2), "Entry: {name}. {desc}\nPick the sector:\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((0,1,3,2), "The firm {name} is characterised as: {desc}.\nWhich sector?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((2,0,1,3), "Overview: {name} \u2014 {desc}\nSector label for {name}:\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((1,3,2,0), "Record for {name}: {desc}\nSector (choose one):\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((3,2,0,1), "{name}: {desc}.\nCategorise {name} by sector.\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
]
HELDOUT_TEMPLATES = [
    ((1,0,2,3), "Sector classification task.\nEntity: {name}\nNotes: {desc}\nWhich sector applies?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((2,3,1,0), "Read the profile below and answer.\n{name} \u2014 {desc}\nThe sector of {name} is:\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((0,3,1,2), "Question: given the description of {name} ({desc}), which sector is it?\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((3,1,0,2), "Database extract \u2014 name: {name}; details: {desc}.\nSector code:\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((1,2,3,0), "Consider {name}. {desc}\nDecide its sector.\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
    ((2,0,3,1), "Analyst note: {name} \u2014 {desc}.\nSector assignment:\nA. {o0}\nB. {o1}\nC. {o2}\nD. {o3}\nAnswer:"),
]
TRAIN_ITEMS = [render(t, n, d, s) for t in TRAIN_TEMPLATES for (n, s, d) in COMPANIES]
HELDOUT_ITEMS = [render(t, n, d, s) for t in HELDOUT_TEMPLATES for (n, s, d) in COMPANIES]
assert len(TRAIN_ITEMS) == 512 and len(HELDOUT_ITEMS) == 192

# ---------------- model load ----------------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map="auto")
model.config.use_cache = False
print(f"MODEL_OK {MODEL}", flush=True)
print(f"GPU {torch.cuda.get_device_name(0)} total={torch.cuda.get_device_properties(0).total_memory/2**30:.2f}GiB", flush=True)

CAND_TOKENS = {}
for letter in "ABCD":
    ids = tok(" " + letter, add_special_tokens=False)["input_ids"]
    CAND_TOKENS[letter] = ids
print(f"CAND_TOKENS {CAND_TOKENS}", flush=True)

rows = []
def add(arm, metric, value, n, notes=""):
    rows.append({"arm": arm, "metric": metric, "value": value, "n": n, "notes": notes})
    print(f"RESULT {arm} {metric}={value} n={n} {notes}", flush=True)
def write_csv():
    with open("/content/fra_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "metric", "value", "n", "notes"])
        w.writeheader(); w.writerows(rows)

@torch.no_grad()
def eval_accuracy(mdl):
    mdl.eval()
    # Pre-encode prompts once.
    seqs = []  # (prompt_ids, letter)
    for prompt, letter in HELDOUT_ITEMS:
        seqs.append((tok(prompt, add_special_tokens=False)["input_ids"], letter))
    correct = 0
    B = 32
    for i in range(0, len(seqs), B):
        chunk = seqs[i:i+B]
        # Build 4 candidate sequences per item.
        flat, meta = [], []
        for ci, (pids, letter) in enumerate(chunk):
            for cand in "ABCD":
                cids = CAND_TOKENS[cand]
                flat.append(pids + cids)
                meta.append((ci, cand, len(pids), len(cids)))
        maxlen = max(len(s) for s in flat)
        input_ids = torch.full((len(flat), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(flat), maxlen), dtype=torch.long)
        for j, s in enumerate(flat):
            input_ids[j, :len(s)] = torch.tensor(s)
            attn[j, :len(s)] = 1
        logits = mdl(input_ids=input_ids.to(DEVICE),
                     attention_mask=attn.to(DEVICE)).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        scores = {}
        for j, (ci, cand, plen, clen) in enumerate(meta):
            # completion tokens at positions plen..plen+clen-1, predicted
            # by logits at plen-1..plen+clen-2
            s = 0.0
            for k in range(clen):
                s += float(logp[j, plen + k - 1, flat[j][plen + k]].item())
            scores[(ci, cand)] = s
        for ci, (pids, letter) in enumerate(chunk):
            pred = max("ABCD", key=lambda c: scores[(ci, c)])
            correct += int(pred == letter)
    return correct / len(seqs)

# ---------------- base eval ----------------
print("== BASE EVAL ==", flush=True)
base_acc = eval_accuracy(model)
add("base", "heldout_acc", round(base_acc, 4), len(HELDOUT_ITEMS),
    "chance=0.25; loglikelihood-of-letter, 6 unseen templates")
write_csv()

# ---------------- QLoRA train ----------------
lcfg = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"TRAINABLE {n_trainable} / {n_total}", flush=True)
model.train()

def encode(prompt, letter):
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(" " + letter, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = p_ids + t_ids
    labels = ([-100] * len(p_ids) + t_ids)
    return ids, labels

encoded = [encode(p, l) for p, l in TRAIN_ITEMS]
rng = random.Random(SEED)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
losses = []
n_steps = (len(encoded) * EPOCHS + BATCH - 1) // BATCH
step = 0
for epoch in range(EPOCHS):
    order = list(range(len(encoded)))
    rng.shuffle(order)
    for bi in range(0, len(order), BATCH):
        batch = [encoded[j] for j in order[bi:bi+BATCH]]
        maxlen = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for j, (ids, lab) in enumerate(batch):
            input_ids[j, :len(ids)] = torch.tensor(ids)
            labels[j, :len(lab)] = torch.tensor(lab)
            attn[j, :len(ids)] = 1
        out = model(input_ids=input_ids.to(DEVICE), attention_mask=attn.to(DEVICE),
                    labels=labels.to(DEVICE))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss.item()))
        step += 1
        if step % 40 == 0:
            print(f"  step {step}/{n_steps} loss={losses[-1]:.4f}", flush=True)
model.eval()
peak_gb = torch.cuda.max_memory_allocated() / 2**30
add("full", "train_mean_loss", round(sum(losses)/len(losses), 4), len(losses))
add("full", "trainable_params", n_trainable, 1, f"total_params={n_total}")
add("full", "peak_mem_gb", round(peak_gb, 2), 1, "max_memory_allocated after train")
print("== FULL ADAPTER EVAL ==", flush=True)
full_acc = eval_accuracy(model)
add("full", "heldout_acc", round(full_acc, 4), len(HELDOUT_ITEMS), "r=16 all modules")
write_csv()

# ---------------- LoRA module discovery + truncation machinery ----------------
params = dict(model.named_parameters())
lora = {}
for name, p in params.items():
    if name.endswith("lora_A.default.weight"):
        key = name[:-len("lora_A.default.weight")]
        lora.setdefault(key, {})["A"] = p
    elif name.endswith("lora_B.default.weight"):
        key = name[:-len("lora_B.default.weight")]
        lora.setdefault(key, {})["B"] = p
    elif name.endswith("lora_A.weight"):
        key = name[:-len("lora_A.weight")]
        lora.setdefault(key, {})["A"] = p
    elif name.endswith("lora_B.weight"):
        key = name[:-len("lora_B.weight")]
        lora.setdefault(key, {})["B"] = p
lora = {k: v for k, v in lora.items() if "A" in v and "B" in v}
print(f"LORA_MODULES {len(lora)}", flush=True)
for k in sorted(lora)[:3]:
    print(f"  MODULE {k} A={tuple(lora[k]['A'].shape)} B={tuple(lora[k]['B'].shape)}", flush=True)
if not lora:
    raise RuntimeError("NO_LORA_MODULES " + str([n for n in params][:20]))
ORIG = {k: (v["A"].data.clone(), v["B"].data.clone()) for k, v in lora.items()}

def svd_factors(A, B):
    """Thin SVD of delta = SCALING * B @ A via QR + 16x16 SVD.
    Returns (singular values (r,), U_delta (out,r), Vt_delta (r,in))."""
    Af = A.detach().float()
    Bf = B.detach().float()
    Q1, R1 = torch.linalg.qr(Af.T)   # A.T: (in, r)
    Q2, R2 = torch.linalg.qr(Bf)     # (out, r)
    K = SCALING * (R2 @ R1.T)
    Uk, Sk, Vhk = torch.linalg.svd(K)
    U_delta = Q2 @ Uk
    Vt_delta = (Q1 @ Vhk.T).T
    return Sk, U_delta, Vt_delta

def truncate_pair(A0, B0, k):
    """Rank-k truncated (A,B) with the SAME shapes (zero-padded) and the
    same alpha/r scaling, so SCALING*B'@A' = rank-k delta."""
    r = A0.shape[0]
    if k >= r:
        return A0.clone(), B0.clone()
    if k <= 0:
        return torch.zeros_like(A0), torch.zeros_like(B0)
    Sk, U_delta, Vt_delta = svd_factors(A0, B0)
    scale = torch.sqrt(Sk[:k] / SCALING)
    A_new = torch.zeros_like(A0, dtype=torch.float32)
    B_new = torch.zeros_like(B0, dtype=torch.float32)
    A_new[:k] = scale[:, None] * Vt_delta[:k].float()
    B_new[:, :k] = U_delta[:, :k].float() * scale[None, :]
    return A_new.to(A0.dtype), B_new.to(B0.dtype)

def apply_ranks(rank_map):
    for key, k in rank_map.items():
        A0, B0 = ORIG[key]
        A_new, B_new = truncate_pair(A0, B0, k)
        lora[key]["A"].data.copy_(A_new)
        lora[key]["B"].data.copy_(B_new)

def restore_full():
    for key, (A0, B0) in ORIG.items():
        lora[key]["A"].data.copy_(A0)
        lora[key]["B"].data.copy_(B0)

# Verify truncation math on the first module: reconstruction error at
# rank 4 must equal the discarded singular-value energy.
_first = sorted(lora)[0]
A0, B0 = ORIG[_first]
Sk, _, _ = svd_factors(A0, B0)
A4, B4 = truncate_pair(A0, B0, 4)
delta = SCALING * (B0.float() @ A0.float())
delta4 = SCALING * (B4.float() @ A4.float())
err2 = float(((delta - delta4) ** 2).sum())
expect2 = float((Sk[4:] ** 2).sum())
print(f"TRUNC_CHECK module={_first} err2={err2:.4f} expect2={expect2:.4f} "
      f"rel={abs(err2-expect2)/max(expect2,1e-9):.4f}", flush=True)
add("check", "trunc_recon_rel_err", round(abs(err2-expect2)/max(expect2,1e-9), 4), 1,
    "QR+16x16 SVD truncation identity, first module")
write_csv()

# Singular values for every module (cheap 16x16 route).
SPECTRAL = {}
for key in sorted(lora):
    Sk, _, _ = svd_factors(ORIG[key][0], ORIG[key][1])
    SPECTRAL[key] = Sk.cpu()
    tot = float((Sk ** 2).sum())
    e4 = float((Sk[:4] ** 2).sum()) / tot if tot > 0 else 0.0
    add("spectral", f"energy_at_rank4::{key}", round(e4, 4), 1, "share of delta energy in top-4 singular values")

# ---------------- isolated sensitivity (FRA measurements) ----------------
def layer_of(key):
    import re
    m = re.search(r"layers\.(\d+)\.", key)
    return int(m.group(1)) if m else -1
def short_name(key):
    for nm in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if nm in key:
            return nm
    return key
SAMPLE_LAYERS = [0, 9, 18, 27, 35]
sample_keys = [k for k in sorted(lora) if layer_of(k) in SAMPLE_LAYERS]
print(f"SAMPLE_MODULES {len(sample_keys)} layers={SAMPLE_LAYERS}", flush=True)
sensitivity = {}  # key -> (trunc4_acc, drop)
for key in sample_keys:
    apply_ranks({key: 4})
    acc4 = eval_accuracy(model)
    restore_full()
    drop = full_acc - acc4
    sensitivity[key] = (acc4, drop)
    add("sensitivity", f"trunc4_acc::{layer_of(key)}/{short_name(key)}",
        round(acc4, 4), len(HELDOUT_ITEMS), f"drop={drop:.4f}; spectral_energy4={float((SPECTRAL[key][:4]**2).sum()/ (SPECTRAL[key]**2).sum()):.4f}")
    write_csv()
print("== SENSITIVITY DONE ==", flush=True)

# ---------------- three allocations at 25% total rank budget ----------------
M = len(lora)
BUDGET = 4 * M  # 25% of 16*M
print(f"BUDGET total_rank={BUDGET} modules={M} avg_rank=4.0", flush=True)

# (a) Uniform: every module rank 4.
uniform_map = {k: 4 for k in lora}
apply_ranks(uniform_map); uniform_acc = eval_accuracy(model); restore_full()
add("uniform", "heldout_acc", round(uniform_acc, 4), len(HELDOUT_ITEMS),
    f"all modules rank 4; total_rank={BUDGET}")
write_csv()

# (b) FRA/sensitivity: sampled modules tiered by measured drop
# (top 12 -> 8, next 11 -> 4, bottom 12 -> 0; mean 4, same sub-budget);
# unsampled modules rank 4.
ordered = sorted(sample_keys, key=lambda k: sensitivity[k][1], reverse=True)
fra_map = {k: 4 for k in lora}
for j, k in enumerate(ordered):
    fra_map[k] = 8 if j < 12 else (4 if j < 23 else 0)
assert sum(fra_map[k] for k in sample_keys) == 4 * len(sample_keys)
fra_acc = None
apply_ranks(fra_map); fra_acc = eval_accuracy(model); restore_full()
add("fra", "heldout_acc", round(fra_acc, 4), len(HELDOUT_ITEMS),
    f"sampled tiered by sensitivity drop; total_rank={sum(fra_map.values())}")
for j, k in enumerate(ordered):
    add("fra", f"alloc::{layer_of(k)}/{short_name(k)}", fra_map[k], 1,
        f"drop={sensitivity[k][1]:.4f} trunc4_acc={sensitivity[k][0]:.4f}")
write_csv()

# (c) Spectral-only: greedy by next singular value until budget spent.
ranks = {k: 0 for k in lora}
remaining = BUDGET
while remaining > 0:
    best, best_val = None, -1.0
    for k in lora:
        if ranks[k] < R:
            v = float(SPECTRAL[k][ranks[k]])
            if v > best_val:
                best, best_val = k, v
    ranks[best] += 1
    remaining -= 1
assert sum(ranks.values()) == BUDGET
apply_ranks(ranks); spec_acc = eval_accuracy(model); restore_full()
add("spectral", "heldout_acc", round(spec_acc, 4), len(HELDOUT_ITEMS),
    f"greedy by singular value; total_rank={BUDGET}")
for k in sorted(lora):
    add("spectral", f"alloc::{layer_of(k)}/{short_name(k)}", ranks[k], 1, "")
write_csv()

peak_gb2 = torch.cuda.max_memory_allocated() / 2**30
add("full", "peak_mem_gb_final", round(peak_gb2, 2), 1, "after all evals")

summary = {
    "paper": "2610.36386", "model": MODEL, "model_note":
        "Qwen3.5-9B tried first 2026-10-27, FAILED twice (bad id / qwen3_5 "
        "unrecognised by transformers<5.0); Qwen3-8B used directly per task brief",
    "r": R, "alpha": ALPHA, "lr": LR, "epochs": EPOCHS, "batch": BATCH,
    "seed": SEED, "n_train": len(TRAIN_ITEMS), "n_heldout": len(HELDOUT_ITEMS),
    "n_train_templates": len(TRAIN_TEMPLATES), "n_heldout_templates": len(HELDOUT_TEMPLATES),
    "trainable_params": n_trainable, "total_params": n_total,
    "peak_mem_gb": round(peak_gb2, 2),
    "base_acc": round(base_acc, 4), "full_acc": round(full_acc, 4),
    "uniform_acc": round(uniform_acc, 4), "fra_acc": round(fra_acc, 4),
    "spectral_acc": round(spec_acc, 4),
    "budget_total_rank": BUDGET, "n_modules": M, "avg_rank": 4.0,
    "sample_layers": SAMPLE_LAYERS, "n_sample_modules": len(sample_keys),
    "sensitivity": {f"{layer_of(k)}/{short_name(k)}": {
        "trunc4_acc": round(sensitivity[k][0], 4), "drop": round(sensitivity[k][1], 4),
        "spectral_energy4": round(float((SPECTRAL[k][:4]**2).sum()/(SPECTRAL[k]**2).sum()), 4)}
        for k in sample_keys},
    "fra_alloc": {f"{layer_of(k)}/{short_name(k)}": fra_map[k] for k in sample_keys},
    "spectral_alloc_summary": {
        "rank_hist": {str(r): sum(1 for v in ranks.values() if v == r) for r in range(R+1)}},
    "predicted_ordering": "FRA >= uniform, FRA >= spectral (paper)",
    "observed_ordering": "FRA vs uniform vs spectral per accuracies above",
}
with open("/content/fra_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
write_csv()
print("WROTE /content/fra_metrics.csv + /content/fra_summary.json", flush=True)
print("TRAIN_DONE", flush=True)
