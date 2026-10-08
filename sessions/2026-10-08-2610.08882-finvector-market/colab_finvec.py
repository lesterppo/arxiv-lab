#!/usr/bin/env python3
"""FinVector-Market-4B replication (arXiv:2610.08882)
- bounded Colab reproduction on Qwen3.5-4B NF4 QLoRA (paper's backbone).

Paper's crispest claim (TUNE 5/5): rank-16 LoRA on a 22k-example corpus for
structured financial tasks gives gains BEYOND output-format learning, under
matched explicit prompting. Supplying the JSON schema alone raises base-model
JSON validity 0% -> 91.3% (format), but task accuracy stays low; the LoRA
adapter raises task accuracy (FinQA EM 14.7 -> 40.0%, calculator-expression
48.0 -> 82.7%, branch-label 20.1 -> 89.5%, implication-direction
52.4 -> 87.2%).

Arms (same explicit JSON-schema prompt at eval for B and C):
  A base-noschema  : frozen base, plain prompt (replicates validity ~0%)
  B base+schema    : frozen base, explicit JSON-schema prompt (format-only)
  C lora+schema    : LoRA r=16/alpha=32 trained on corpus, same schema prompt
Dissociation to test: validity(B) ~= validity(C) >> validity(A), while
accuracy(C) > accuracy(B) substantially.

Tasks (synthetic, seeded, exactly verifiable; stand in for the paper corpus):
  CALC   : arithmetic expression -> value (calculator-expression correctness)
  FINQA  : small financial table + question -> numeric answer (exact match)
  BRANCH : scenario features -> APPROVE/REVIEW/REJECT (branch-label agreement)
  IMPLY  : statement pair -> SUPPORTS/CONTRADICTS/NEUTRAL (direction agreement)
600 train/task (2400 pool), 60 test/task (240 benchmark; paper uses 600).
150 steps, batch 4, lr 2e-4 (memory-lean: grad checkpointing, seq 128), bf16, transformers latest, torchao uninstalled
(the documented peft/torchao quirk). Metrics -> /content/*.json + *.csv.
"""
import os, sys, json, math, random, subprocess, time, csv, gc, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

RESULT_JSON = "/content/finvec_results.json"
METRICS_CSV = "/content/finvec_metrics.csv"

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
TASKS = ["CALC", "FINQA", "BRANCH", "IMPLY"]
T = len(TASKS)
N_TRAIN_TASK, N_TEST_TASK = 600, 60
STEPS = 150
BATCH = 4
LR = 2e-4
LORA_R, LORA_ALPHA = 16, 32
TARGETS = ["q_proj", "v_proj"]
EVAL_EVERY = 50
MAX_NEW = 16

SCHEMA_PROMPT = 'Output only JSON {"answer": "<string>"}. Q: __Q__\nA:'
PLAIN_PROMPT = "Q: __Q__\nA:"

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- data ----------------
def gen_calc(rng):
    n = rng.randint(2, 3)
    parts, val = [], rng.uniform(10, 500)
    parts.append(f"{val:.2f}")
    expr = f"{val:.2f}"
    for _ in range(n - 1):
        op = rng.choice(["+", "-", "*", "/"])
        b = rng.uniform(2, 99)
        if op == "+":
            val += b
        elif op == "-":
            val -= b
        elif op == "*":
            val *= b
        else:
            val /= b
        expr = f"({expr} {op} {b:.2f})"
    q = f"Compute the value of the expression: {expr}"
    return q, f"{val:.2f}"

def gen_finqa(rng):
    rev23, cogs23 = rng.uniform(800, 2000), rng.uniform(300, 900)
    rev24, cogs24 = rng.uniform(800, 2000), rng.uniform(300, 900)
    kind = rng.choice(["gp", "margin", "growth"])
    if kind == "gp":
        q = (f"Financials 2024: revenue {rev24:.1f}M, COGS {cogs24:.1f}M. "
             f"What was gross profit in 2024 (in M)?")
        a = f"{rev24 - cogs24:.1f}"
    elif kind == "margin":
        q = (f"Financials 2023: revenue {rev23:.1f}M, COGS {cogs23:.1f}M. "
             f"What was the 2023 gross margin in percent?")
        a = f"{(rev23 - cogs23) / rev23 * 100:.1f}"
    else:
        q = (f"Revenue was {rev23:.1f}M in 2023 and {rev24:.1f}M in 2024. "
             f"What was revenue growth in percent from 2023 to 2024?")
        a = f"{(rev24 - rev23) / rev23 * 100:.1f}"
    return q, a

def gen_branch(rng):
    amt = rng.uniform(5, 500)          # k$
    risk = rng.choice(["low", "medium", "high"])
    flag = rng.choice(["none", "sanctions-hit", "pep"])
    # deterministic policy rule the model must learn:
    if flag != "none" or (risk == "high" and amt > 100):
        lab = "REJECT"
    elif risk == "high" or amt > 250 or flag == "pep":
        lab = "REVIEW"
    elif risk == "medium" and amt > 100:
        lab = "REVIEW"
    else:
        lab = "APPROVE"
    q = (f"Transaction review: amount ${amt:.1f}k, customer risk tier {risk}, "
         f"compliance flag {flag}. Decide: APPROVE, REVIEW, or REJECT?")
    return q, lab

def gen_imply(rng):
    a = rng.uniform(100, 1000)
    b = rng.uniform(100, 1000)
    rel = rng.choice(["SUPPORTS", "CONTRADICTS", "NEUTRAL"])
    if rel == "SUPPORTS":
        s1 = f"Q3 operating cash flow was ${a:.1f}M."
        s2 = f"Management reported operating cash flow of ${a:.1f}M in Q3."
    elif rel == "CONTRADICTS":
        s1 = f"Q3 operating cash flow was ${a:.1f}M."
        s2 = f"Management reported operating cash flow of ${b:.1f}M in Q3."
    else:
        s1 = f"Q3 operating cash flow was ${a:.1f}M."
        s2 = "The board approved a new dividend policy in Q3."
    q = (f'Statement 1: "{s1}" Statement 2: "{s2}" '
         f'Does statement 2 SUPPORT, CONTRADICT, or is it NEUTRAL to statement 1?')
    return q, rel

GEN = {"CALC": gen_calc, "FINQA": gen_finqa, "BRANCH": gen_branch,
       "IMPLY": gen_imply}

def gen_data(n, seed):
    rng = random.Random(seed)
    data = []
    for spec in TASKS:
        for _ in range(n):
            q, a = GEN[spec](rng)
            data.append((spec, q, a))
    rng.shuffle(data)
    return data

train_pool = gen_data(N_TRAIN_TASK, SEED)
test_data = gen_data(N_TEST_TASK, SEED + 1)
log(f"pool={len(train_pool)} test={len(test_data)}")

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

def encode_batch(tok, batch):
    fulls = [p + " " + a for _, p, a in batch]
    enc = tok(fulls, padding=True, truncation=True, max_length=128,
              return_tensors="pt")
    plens = [len(tok(p)["input_ids"]) for _, p, _ in batch]
    labels = enc["input_ids"].clone()
    for i, pl in enumerate(plens):
        labels[i, :pl] = -100
    labels[enc["attention_mask"] == 0] = -100
    return enc, labels

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

# ---------------- eval ----------------
JSON_RE = re.compile(r"\{[^{}]*\}")

def parse_answer(text):
    m = JSON_RE.search(text)
    if not m:
        return None, False
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, False
    if isinstance(obj, dict) and "answer" in obj:
        return str(obj["answer"]), True
    return None, False

def norm_num(s):
    try:
        return float(str(s).replace(",", "").strip().rstrip("%"))
    except Exception:
        return None

def correct(spec, pred, gold):
    if pred is None:
        return False
    if spec in ("CALC", "FINQA"):
        pn, gn = norm_num(pred), norm_num(gold)
        if pn is None or gn is None:
            return pred.strip() == gold.strip()
        return abs(pn - gn) <= max(0.05, 0.005 * abs(gn))
    return pred.strip().upper() == gold.strip().upper()

@torch.no_grad()
def evaluate(model, tok, data, prompt_fn):
    model.eval()
    stats = {s: {"n": 0, "valid": 0, "acc": 0, "acc_given_valid": 0}
             for s in TASKS}
    B = 32
    for i in range(0, len(data), B):
        chunk = data[i:i + B]
        prompts = [prompt_fn(q) for _, q, _ in chunk]
        enc = tok(prompts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
        gen = model.generate(ids, attention_mask=attn,
                             max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen[:, ids.shape[1]:],
                                 skip_special_tokens=True)
        for (spec, _, gold), t in zip(chunk, texts):
            st = stats[spec]
            st["n"] += 1
            pred, valid = parse_answer(t)
            if valid:
                st["valid"] += 1
                if correct(spec, pred, gold):
                    st["acc"] += 1
                    st["acc_given_valid"] += 1
    model.train()
    out = {}
    for s, st in stats.items():
        n = st["n"]
        out[s] = {
            "n": n,
            "json_validity": round(st["valid"] / n, 4),
            "acc": round(st["acc"] / n, 4),
            "acc_given_valid": round(st["acc_given_valid"] / max(st["valid"], 1), 4),
        }
    tn = sum(st["n"] for st in stats.values())
    tv = sum(st["valid"] for st in stats.values())
    ta = sum(st["acc"] for st in stats.values())
    out["overall"] = {
        "json_validity": round(tv / tn, 4),
        "acc": round(ta / tn, 4),
        "acc_given_valid": round(sum(out[s]["acc_given_valid"] * stats[s]["valid"]
                                     for s in TASKS) / max(tv, 1), 4),
    }
    return out

def prompt_schema(q):
    return SCHEMA_PROMPT.replace("__Q__", q)

def prompt_plain(q):
    return PLAIN_PROMPT.replace("__Q__", q)

# ---------------- frozen arm eval ----------------
def eval_frozen(arm, prompt_fn):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_base()
    res = evaluate(model, tok, test_data, prompt_fn)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return res

# ---------------- LoRA arm ----------------
def run_lora():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = load_base()
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                     target_modules=TARGETS, lora_dropout=0.0,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    try:
        model.gradient_checkpointing_enable()
        log("[lora] gradient checkpointing enabled")
    except Exception as e:
        log(f"[lora] grad checkpoint n/a: {e}")
    n_params = sum(p.numel() for n, p in model.named_parameters()
                   if "lora_" in n and p.requires_grad)
    log(f"[lora] trainable LoRA params: {n_params}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR)
    # schema-formatted training examples (paper: adapter trained on corpus,
    # eval under matched explicit prompting)
    train_fmt = [(s, prompt_schema(q), '{"answer": "' + a + '"}')
                 for s, q, a in train_pool]
    rng = random.Random(SEED)
    hist, evals = [], []

    def do_eval(step):
        tok.padding_side = "left"
        e = evaluate(model, tok, test_data, prompt_schema)
        tok.padding_side = "right"
        evals.append({"step": step, **{k: v for k, v in e.items()}})
        save_partial("armC_evals", evals)  # incremental: survives reclamation
        o = e["overall"]
        log(f"[lora] step {step}: validity={o['json_validity']:.3f} "
            f"acc={o['acc']:.3f} acc|valid={o['acc_given_valid']:.3f} | " +
            " ".join(f"{s}={e[s]['acc']:.2f}" for s in TASKS))

    do_eval(0)
    for step in range(1, STEPS + 1):
        batch = rng.sample(train_fmt, BATCH)
        loss = train_step(model, tok, opt, batch)
        hist.append({"step": step, "loss": round(loss, 4)})
        if step % EVAL_EVERY == 0:
            do_eval(step)
    final = evals[-1]
    try:
        model.save_pretrained("/content/finvec_adapter")
        log("[lora] adapter saved")
    except Exception as e:
        log(f"[lora] adapter save failed: {e}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"trainable_params": n_params, "evals": evals,
            "final": final}, hist

def save_partial(tag, obj):
    try:
        with open(f"/content/finvec_partial_{tag}.json", "w") as f:
            json.dump(obj, f, indent=1)
    except Exception as e:
        log(f"partial save failed: {e}")

def fix_overall(r):
    # recompute overall acc|valid from per-task ratios (repairs old partials)
    tv = sum(round(r[s]["json_validity"] * r[s]["n"]) for s in TASKS)
    num = sum(r[s]["acc_given_valid"] * round(r[s]["json_validity"] * r[s]["n"])
              for s in TASKS)
    r["overall"]["acc_given_valid"] = round(num / max(tv, 1), 4)
    return r

def load_partial(tag):
    try:
        with open(f"/content/finvec_partial_{tag}.json") as f:
            return fix_overall(json.load(f))
    except Exception:
        return None

def main():
    t0 = time.time()

    res_a = load_partial("armA")
    if res_a is None:
        log("arm A: base frozen, plain prompt")
        res_a = eval_frozen("base_noschema", prompt_plain)
        log(f"arm A overall: {res_a['overall']}")
        save_partial("armA", res_a)
    else:
        log(f"arm A resumed from partial: {res_a['overall']}")
    res_b = load_partial("armB")
    if res_b is None:
        log("arm B: base frozen, explicit schema prompt")
        res_b = eval_frozen("base_schema", prompt_schema)
        log(f"arm B overall: {res_b['overall']}")
        save_partial("armB", res_b)
    else:
        log(f"arm B resumed from partial: {res_b['overall']}")
    log("arm C: LoRA r=16 on corpus, explicit schema prompt")
    res_c, hist_c = run_lora()
    log(f"arm C overall: {res_c['final']['overall']}")

    def pick(r, task, metric):
        return r["overall"][metric] if task == "overall" else r[task][metric]

    tasks = TASKS + ["overall"]
    delta_cb = {s: {m: round(pick(res_c['final'], s, m) - pick(res_b, s, m), 4)
                    for m in ("json_validity", "acc", "acc_given_valid")}
                for s in tasks}
    delta_ba = {s: {m: round(pick(res_b, s, m) - pick(res_a, s, m), 4)
                    for m in ("json_validity", "acc", "acc_given_valid")}
                for s in tasks}
    summary = {
        "paper": "2610.08882 FinVector-Market-4B",
        "model": MODEL_ID, "seed": SEED,
        "train_per_task": N_TRAIN_TASK, "test_per_task": N_TEST_TASK,
        "steps": STEPS, "batch": BATCH, "lr": LR,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "base_noschema": res_a,
        "base_schema": res_b,
        "lora_schema": res_c,
        "delta_schema_minus_noschema": delta_ba,
        "delta_lora_minus_baseschema": delta_cb,
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(summary, f, indent=1)
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "arm", "loss"] +
                   [f"{s}_{m}" for s in tasks
                    for m in ("validity", "acc", "acc_given_valid")])
        ev = {("lora", e["step"]): e for e in res_c["evals"]}
        for h in hist_c:
            e = ev.get(("lora", h["step"]), {})
            w.writerow([h["step"], "lora_schema", h["loss"]] +
                       [e.get(s, {}).get(m, "") for s in tasks
                        for m in ("json_validity", "acc", "acc_given_valid")])
    log(f"wrote {RESULT_JSON}, {METRICS_CSV}")
    o = res_c["final"]["overall"]
    log(f"LoRA-schema vs base-schema overall acc: "
        f"{delta_cb['overall']['acc']:+.3f} "
        f"(validity {delta_cb['overall']['json_validity']:+.3f})")

if __name__ == "__main__":
    main()
