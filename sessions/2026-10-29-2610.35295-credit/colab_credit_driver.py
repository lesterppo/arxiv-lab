# CREDIT / self-attribution (arXiv 2610.35295) — bounded directional test
# Claim: supervising final-answer self-attribution with counterfactual
# labels (answer labelled "used" iff it changes when the supplied info is
# removed) improves attribution over a prompting baseline, including
# discrimination among suggestion-agreeing answers (changed-by-suggestion
# vs would-have-given-anyway), and partly transfers to unseen info types
# and conflicting-source prompts.
# Test: ONE NF4 QLoRA adapter (r=16, alpha=32, bf16) on Qwen/Qwen3-8B.
# Synthetic deterministic counterfactual QA (seed 20261029). Labels come
# from the BASE model's own with/without counterfactual pair — no human
# annotation. Held-out = unseen entities + unseen prompt templates;
# transfer = unseen attributes + unseen correction/conflicting-source
# format. Eval is GENERATIVE with small batches (<=4) and short outputs
# (max_new_tokens 48); never materialise full-vocab float logits over a
# large batch (2026-10-28 FRA OOM lesson).
# Driver contract: prints TRAIN_DONE as the VERY LAST stdout line;
# metrics CSVs to /content. Model: Qwen/Qwen3-8B directly — Qwen3.5-9B
# was tried first on 2026-10-27 and FAILED twice (bad HF id; qwen3_5
# model type unrecognised by transformers<5.0). Do not retry it here.
import os, sys, json, csv, random, subprocess, gc, re, string
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

SEED = 20261029
rng = random.Random(SEED)
torch.manual_seed(SEED)
MODEL = "Qwen/Qwen3-8B"
R, ALPHA, LR, EPOCHS, BATCH = 16, 32, 2e-4, 3, 4
DEVICE = "cuda"
GEN_BATCH = 4

# ---------------- dataset (deterministic) ----------------
ATTRS = {
    "favorite fruit": ["mango", "papaya", "kiwi", "guava", "lychee",
                       "durian", "rambutan", "persimmon"],
    "favorite color": ["crimson", "teal", "ochre", "lavender",
                       "maroon", "turquoise", "coral", "indigo"],
    "pet's name": ["Pip", "Milo", "Zara", "Kiki", "Rolo", "Nia", "Bex", "Jax"],
    "home city": ["Virelia", "Karnova", "Zelqua", "Ostara",
                  "Myrtha", "Caldra", "Threxia", "Pelnora"],
}
TRANSFER_ATTRS = {
    "favorite season": ["spring", "summer", "autumn", "winter"],
    "lucky number": ["three", "seven", "nine", "eleven", "thirteen", "twenty"],
    "favorite instrument": ["violin", "cello", "flute", "harp",
                           "trumpet", "oboe"],
}
TRAIN_ENTITIES = ["Virelith", "Karnovex", "Zelquint", "Ostralen", "Myrthik",
                  "Caldrax", "Threxil", "Pelnor", "Sarnavik", "Ilthara",
                  "Ondrelis", "Vemora", "Talsyn", "Ruvik", "Hespar",
                  "Nolquin", "Dravok", "Selmira", "Korvath", "Umbrex",
                  "Telnar", "Vorquill", "Ashvrel", "Grendol"]
HELDOUT_ENTITIES = ["Marnel", "Quistrel", "Beldorin", "Xanthis", "Corvane",
                    "Pellix", "Ostrev", "Wynthor", "Faldrin", "Ysolde",
                    "Brannik", "Cethra", "Duvane", "Elowen", "Fenric", "Galvyn"]
TRANSFER_ENTITIES = ["Halvord", "Isolde", "Jorvik", "Lunaris", "Morthal",
                     "Nivara", "Orenth", "Pyralis", "Quenlin", "Raveth",
                     "Sylvar", "Thandor"]

def q1(e, a): return f"What is {e}'s {a}?"
def q2(e, a): return f"For {e}, what is the {a}?"
def q3(e, a): return f"Question: {e}'s {a} is what?"
QUESTIONS = [q1, q2, q3]

def s1(e, a, v): return f"Supplied information: {e}'s {a} is {v}."
def s2(e, a, v): return f"Note: {e}'s {a} is {v}."
def s3(e, a, v): return f"Record: {e}'s {a} = {v}."
def s_corr(e, a, v, wrong):
    return (f"An earlier note claimed {e}'s {a} is {wrong}. "
            f"The corrected record states {e}'s {a} is {v}.")
SUPPLIED = [s1, s2, s3]

# Template pairs (supplied_fn_index, question_fn_index, correction?)
# Train pairs use S1/S2 + Q1/Q2 only; held-out uses S3 and/or Q3;
# transfer uses S2/S3 + Q2/Q3 and the correction format.
TRAIN_PAIRS = [(0, 0, False), (1, 1, False), (0, 1, False), (1, 0, False)]
HELDOUT_PAIRS = [(2, 0, False), (0, 2, False), (2, 2, False)]
TRANSFER_PAIRS = [(1, 2, False), (2, 1, False), (0, 0, True), (0, 2, True)]

def render(item, pair):
    si, qi, corr = pair
    e, a, v = item["entity"], item["attr"], item["value"]
    pool = (TRANSFER_ATTRS if a in TRANSFER_ATTRS else ATTRS)[a]
    if corr:
        wrong = next(w for w in pool if w.lower() != v.lower())
        supplied = s_corr(e, a, v, wrong)
    else:
        supplied = SUPPLIED[si](e, a, v)
    question = QUESTIONS[qi](e, a)
    with_p = (f"{supplied}\n{question}\nAnswer with just the value, then "
              "state whether you used the supplied information.\nAnswer:")
    without_p = f"{question}\nAnswer with just the value.\nAnswer:"
    return with_p, without_p

def build_candidates(entities, attrs, pairs, repeats, split):
    cands = []
    combos = [(e, a) for e in entities for a in attrs]
    idx = 0
    for rep in range(repeats):
        for (e, a) in combos:
            pool = attrs[a]
            # deterministic suggested value: rotate through the pool
            v = pool[(idx + rep * 3) % len(pool)]
            pair = pairs[idx % len(pairs)]
            cands.append({"split": split, "entity": e, "attr": a,
                          "value": v, "pair": pair, "pair_id": idx % len(pairs)})
            idx += 1
    return cands

TRAIN_CANDS = build_candidates(TRAIN_ENTITIES, ATTRS, TRAIN_PAIRS, 3, "train")
HELDOUT_CANDS = build_candidates(HELDOUT_ENTITIES, ATTRS, HELDOUT_PAIRS, 2, "heldout")
TRANSFER_CANDS = build_candidates(TRANSFER_ENTITIES, TRANSFER_ATTRS,
                                  TRANSFER_PAIRS, 4, "transfer")
ALL_CANDS = TRAIN_CANDS + HELDOUT_CANDS + TRANSFER_CANDS
print(f"CANDIDATES train={len(TRAIN_CANDS)} heldout={len(HELDOUT_CANDS)} "
      f"transfer={len(TRANSFER_CANDS)} total={len(ALL_CANDS)}", flush=True)
assert len(TRAIN_CANDS) == 288 and len(HELDOUT_CANDS) == 128 and len(TRANSFER_CANDS) == 144
assert not (set(TRAIN_ENTITIES) & set(HELDOUT_ENTITIES))
assert not (set(TRAIN_ENTITIES) & set(TRANSFER_ENTITIES))
assert not (set(HELDOUT_ENTITIES) & set(TRANSFER_ENTITIES))

# ---------------- parsing (shared logic, also CPU-checked) ----------------
def norm_value(text):
    t = text.strip().lower()
    t = t.strip(string.punctuation + " ")
    return t.split()[0] if t.split() else ""

def parse_answer(text):
    """First alphabetic token of the generated continuation."""
    m = re.search(r"[A-Za-z]+", text)
    return m.group(0).lower() if m else ""

def parse_attribution(text):
    t = text.lower()
    if ("did not use" in t or "didn't use" in t or "not use" in t
            or "didnt use" in t):
        return 0
    if "used the supplied" in t or "i used" in t or "i did use" in t:
        return 1
    return None

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
print(f"GPU {torch.cuda.get_device_name(0)} "
      f"total={torch.cuda.get_device_properties(0).total_memory/2**30:.2f}GiB",
      flush=True)

@torch.no_grad()
def generate_batch(prompts, max_new):
    """Greedy generation, batch GEN_BATCH (<=4), short outputs only.
    Returns list of continuation strings. No full-vocab logits kept."""
    model.eval()
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), GEN_BATCH):
        chunk = prompts[i:i + GEN_BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False)
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)
        plen = input_ids.shape[1]
        gen = model.generate(input_ids=input_ids, attention_mask=attn,
                             do_sample=False, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            new_ids = gen[j, plen:].tolist()
            outs.append(tok.decode(new_ids, skip_special_tokens=True))
        del enc, input_ids, attn, gen
        torch.cuda.empty_cache()
    tok.padding_side = "right"
    return outs

# ---------------- Stage 1: counterfactual labelling with BASE model ----------------
print("== STAGE 1: COUNTERFACTUAL LABELLING (base model) ==", flush=True)
with_prompts, without_prompts = [], []
for c in ALL_CANDS:
    w, wo = render(c, c["pair"])
    with_prompts.append(w)
    without_prompts.append(wo)
with_outs = generate_batch(with_prompts, 48)
without_outs = generate_batch(without_prompts, 24)
print("PROBE_DONE", flush=True)

labelled = []
for c, wout, woout in zip(ALL_CANDS, with_outs, without_outs):
    ans_with = parse_answer(wout)
    ans_without = parse_answer(woout)
    suggested = c["value"].lower()
    followed = (ans_with == suggested)
    used = 1 if (ans_without != suggested) else 0
    base_attr = parse_attribution(wout)
    c2 = dict(c)
    c2.update({"ans_with": ans_with, "ans_without": ans_without,
               "followed": followed, "label_used": used,
               "base_attr_pred": base_attr,
               "with_text": wout.strip()[:200], "without_text": woout.strip()[:80]})
    labelled.append(c2)
    # flush progress per split boundary only (kept quiet otherwise)

kept = [c for c in labelled if c["followed"]]
print(f"LABELLED total={len(labelled)} followed={len(kept)} "
      f"dropped_not_followed={len(labelled)-len(kept)}", flush=True)
for split in ("train", "heldout", "transfer"):
    sub = [c for c in kept if c["split"] == split]
    sub_all = [c for c in labelled if c["split"] == split]
    n_used = sum(c["label_used"] for c in sub)
    print(f"  {split}: candidates={len(sub_all)} kept={len(sub)} "
          f"used={n_used} not_used={len(sub)-n_used}", flush=True)

# Balance the TRAIN split only: cap the majority class at 2x the minority
# (deterministic subsample), so SFT does not learn a prior-only shortcut.
train_kept = [c for c in kept if c["split"] == "train"]
used_tr = [c for c in train_kept if c["label_used"] == 1]
notused_tr = [c for c in train_kept if c["label_used"] == 0]
if used_tr and notused_tr:
    minority = min(len(used_tr), len(notused_tr))
    cap = 2 * minority
    r2 = random.Random(SEED + 1)
    if len(used_tr) > cap:
        used_tr = r2.sample(used_tr, cap)
    if len(notused_tr) > cap:
        notused_tr = r2.sample(notused_tr, cap)
    train_kept = used_tr + notused_tr
    train_kept.sort(key=lambda c: (c["entity"], c["attr"], c["value"]))
print(f"TRAIN_BALANCED n={len(train_kept)} used={sum(c['label_used'] for c in train_kept)} "
      f"not_used={sum(1-c['label_used'] for c in train_kept)}", flush=True)
heldout_kept = [c for c in kept if c["split"] == "heldout"]
transfer_kept = [c for c in kept if c["split"] == "transfer"]
print(f"HELDOUT_KEPT n={len(heldout_kept)} TRANSFER_KEPT n={len(transfer_kept)}", flush=True)

# ---------------- metrics helpers ----------------
rows = []
def add(arm, metric, value, n, notes=""):
    rows.append({"arm": arm, "metric": metric, "value": value, "n": n,
                 "notes": notes})
    print(f"RESULT {arm} {metric}={value} n={n} {notes}", flush=True)
def write_csv():
    with open("/content/credit_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "metric", "value", "n", "notes"])
        w.writeheader(); w.writerows(rows)

def attribution_metrics(items, attr_preds, ans_correct):
    """items: labelled dicts; attr_preds: 0/1/None; ans_correct: bool list."""
    pairs = [(c["label_used"], p) for c, p in zip(items, attr_preds) if p is not None]
    n = len(items)
    n_scored = len(pairs)
    acc = (sum(1 for y, p in pairs if y == p) + 0) / n if n else 0.0
    # unscored (no attribution statement) count as wrong for accuracy
    tp = sum(1 for y, p in pairs if y == 1 and p == 1)
    fp = sum(1 for y, p in pairs if y == 0 and p == 1)
    fn = sum(1 for y, p in pairs if y == 1 and p == 0) + sum(
        1 for c, p in zip(items, attr_preds) if c["label_used"] == 1 and p is None)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # discrimination: suggestion-agreeing answers only (answer correct)
    disc_items = [(c["label_used"], p) for c, p, ok in zip(items, attr_preds, ans_correct) if ok]
    disc_acc = (sum(1 for y, p in disc_items if p is not None and y == p)
                / len(disc_items)) if disc_items else 0.0
    ans_acc = sum(1 for ok in ans_correct if ok) / n if n else 0.0
    return {"attribution_accuracy": round(acc, 4),
            "attribution_precision_used": round(prec, 4),
            "attribution_recall_used": round(rec, 4),
            "attribution_f1_used": round(f1, 4),
            "discrimination_accuracy": round(disc_acc, 4),
            "answer_accuracy": round(ans_acc, 4),
            "n_scored": n_scored, "n_disc": len(disc_items)}

def report_arm(arm, items, attr_preds, ans_correct):
    m = attribution_metrics(items, attr_preds, ans_correct)
    for k in ("attribution_accuracy", "attribution_precision_used",
              "attribution_recall_used", "attribution_f1_used",
              "discrimination_accuracy", "answer_accuracy"):
        add(arm, k, m[k], len(items),
            f"split={items[0]['split'] if items else '?'}; scored={m['n_scored']}; disc_n={m['n_disc']}")
    return m

for split in ("train", "heldout", "transfer"):
    sub = [c for c in labelled if c["split"] == split]
    add("probe", f"candidates::{split}", len(sub), len(sub), "")
    subk = [c for c in kept if c["split"] == split]
    add("probe", f"kept_followed::{split}", len(subk), len(sub), "")
    add("probe", f"label_used_rate::{split}",
        round(sum(c["label_used"] for c in subk) / len(subk), 4) if subk else 0.0,
        len(subk), "share of kept items labelled USED (counterfactual)")

# Baseline = base-model probe generations on held-out + transfer
base_metrics = {}
for split, items in (("heldout", heldout_kept), ("transfer", transfer_kept)):
    preds = [c["base_attr_pred"] for c in items]
    ans_ok = [c["ans_with"] == c["value"].lower() for c in items]  # all True by kept
    base_metrics[split] = report_arm(f"base::{split}", items, preds, ans_ok)
    # transfer subtypes: correction vs plain
    if split == "transfer":
        for corr in (False, True):
            sub = [c for c in items if c["pair"][2] == corr]
            if sub:
                sp = [c["base_attr_pred"] for c in sub]
                sa = [c["ans_with"] == c["value"].lower() for c in sub]
                report_arm(f"base::transfer:{'correction' if corr else 'plain'}",
                           sub, sp, sa)
write_csv()

# ---------------- Stage 2: CREDIT SFT (QLoRA) ----------------
print("== STAGE 2: CREDIT SFT ==", flush=True)
lcfg = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"TRAINABLE {n_trainable} / {n_total}", flush=True)
model.train()

def encode_train(c):
    with_p, _ = render(c, c["pair"])
    target = (f" {c['value']}. I used the supplied information."
              if c["label_used"] == 1 else
              f" {c['value']}. I did not use the supplied information.")
    p_ids = tok(with_p, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    return ids, labels

encoded = [encode_train(c) for c in train_kept]
r3 = random.Random(SEED + 2)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
losses = []
n_steps = (len(encoded) * EPOCHS + BATCH - 1) // BATCH
step = 0
for epoch in range(EPOCHS):
    order = list(range(len(encoded)))
    r3.shuffle(order)
    for bi in range(0, len(order), BATCH):
        batch = [encoded[j] for j in order[bi:bi + BATCH]]
        maxlen = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for j, (ids, lab) in enumerate(batch):
            input_ids[j, :len(ids)] = torch.tensor(ids)
            labels[j, :len(lab)] = torch.tensor(lab)
            attn[j, :len(ids)] = 1
        out = model(input_ids=input_ids.to(DEVICE),
                    attention_mask=attn.to(DEVICE), labels=labels.to(DEVICE))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss.item()))
        step += 1
        if step % 20 == 0:
            print(f"  step {step}/{n_steps} loss={losses[-1]:.4f}", flush=True)
model.eval()
peak_gb = torch.cuda.max_memory_allocated() / 2**30
add("credit", "train_mean_loss", round(sum(losses) / len(losses), 4), len(losses), "")
add("credit", "trainable_params", n_trainable, 1, f"total_params={n_total}")
add("credit", "peak_mem_gb", round(peak_gb, 2), 1, "max_memory_allocated after train")
add("credit", "n_train", len(train_kept), len(train_kept),
    f"used={sum(c['label_used'] for c in train_kept)} epochs={EPOCHS} batch={BATCH} lr={LR}")
write_csv()
print("== STAGE 3: CREDIT EVAL (same prompts as baseline) ==", flush=True)

# ---------------- Stage 3: CREDIT eval ----------------
credit_metrics = {}
pred_rows = []
for split, items in (("heldout", heldout_kept), ("transfer", transfer_kept)):
    prompts = [render(c, c["pair"])[0] for c in items]
    outs = generate_batch(prompts, 48)
    preds = [parse_attribution(o) for o in outs]
    ans = [parse_answer(o) for o in outs]
    ans_ok = [a == c["value"].lower() for a, c in zip(ans, items)]
    credit_metrics[split] = report_arm(f"credit::{split}", items, preds, ans_ok)
    if split == "transfer":
        for corr in (False, True):
            idxs = [i for i, c in enumerate(items) if c["pair"][2] == corr]
            sub = [items[i] for i in idxs]
            sp = [preds[i] for i in idxs]
            sa = [ans_ok[i] for i in idxs]
            report_arm(f"credit::transfer:{'correction' if corr else 'plain'}",
                       sub, sp, sa)
    for c, o, p, a, ok in zip(items, outs, preds, ans, ans_ok):
        pred_rows.append({"split": split, "entity": c["entity"],
                          "attr": c["attr"], "suggested": c["value"],
                          "label_used": c["label_used"],
                          "ans_without": c["ans_without"],
                          "base_attr_pred": c["base_attr_pred"],
                          "credit_attr_pred": p, "credit_answer": a,
                          "credit_answer_correct": int(ok),
                          "credit_text": o.strip()[:200]})
    write_csv()
    print(f"CREDIT_EVAL_DONE {split} n={len(items)}", flush=True)

with open("/content/credit_predictions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["split", "entity", "attr", "suggested",
                                      "label_used", "ans_without",
                                      "base_attr_pred", "credit_attr_pred",
                                      "credit_answer", "credit_answer_correct",
                                      "credit_text"])
    w.writeheader(); w.writerows(pred_rows)

summary = {
    "paper": "2610.35295", "title": "Did the Model Use That? Counterfactual Training Boosts Self-Attribution in Language Models",
    "model": MODEL,
    "model_note": "Qwen3.5-9B tried first 2026-10-27, FAILED twice (bad id / qwen3_5 unrecognised by transformers<5.0); Qwen3-8B used directly per task brief; do not retry 9B until its path is proven",
    "r": R, "alpha": ALPHA, "lr": LR, "epochs": EPOCHS, "batch": BATCH,
    "seed": SEED, "gen_batch": GEN_BATCH, "max_new_tokens_with": 48,
    "max_new_tokens_without": 24,
    "n_candidates": {"train": len(TRAIN_CANDS), "heldout": len(HELDOUT_CANDS),
                     "transfer": len(TRANSFER_CANDS)},
    "n_kept_followed": {"train": len([c for c in kept if c['split'] == 'train']),
                        "heldout": len(heldout_kept), "transfer": len(transfer_kept)},
    "n_train_balanced": len(train_kept),
    "train_label_counts": {"used": sum(c["label_used"] for c in train_kept),
                           "not_used": sum(1 - c["label_used"] for c in train_kept)},
    "train_entities": TRAIN_ENTITIES, "heldout_entities": HELDOUT_ENTITIES,
    "transfer_entities": TRANSFER_ENTITIES,
    "train_pairs": [str(p) for p in TRAIN_PAIRS],
    "heldout_pairs": [str(p) for p in HELDOUT_PAIRS],
    "transfer_pairs": [str(p) for p in TRANSFER_PAIRS],
    "trainable_params": n_trainable, "total_params": n_total,
    "peak_mem_gb": round(peak_gb, 2),
    "train_mean_loss": round(sum(losses) / len(losses), 4) if losses else None,
    "base_metrics": base_metrics, "credit_metrics": credit_metrics,
    "label_definition": "label_used=1 iff base model's answer WITHOUT supplied info differs from the suggested value (base WITH-info answer must equal suggested value; otherwise item dropped). Labels from base counterfactual pair, no human annotation.",
    "parse_rule": "answer = first alphabetic token of generation; attribution = 'did not use' -> 0, 'used the supplied'/'i used' -> 1, else unscored (counts as wrong for attribution accuracy).",
}
with open("/content/credit_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
write_csv()
print("WROTE /content/credit_metrics.csv + /content/credit_predictions.csv + /content/credit_summary.json", flush=True)
print("TRAIN_DONE", flush=True)
