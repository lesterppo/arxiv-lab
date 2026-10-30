"""SST (Subspace Tuning, arXiv:2610.22293) at 9B scale on Colab T4.

Sequential LoRA fine-tune of Qwen3.5-9B (NF4, bf16, r=16/a=32):
  Phase A: synthetic finance glossary (240 pairs) -- both arms share it.
  Phase B: synthetic medical glossary (240 pairs) --
      arm "plain": vanilla LoRA continued training.
      arm "sst":   LoRA with Subspace Tuning: after every optimizer step,
                  each SST target module's adapters are reparameterized
                  B <- P_left B,  A <- A P_right
              with P_left  = I - U_r U_r^T,  P_right = I - V_r V_r^T from
              the SVD of the FROZEN base weight W0 (protect_frac=0.9).
              Then the effective update s*BA satisfies
              BA = P_left (BA) P_right -- the faithful two-sided reading
              of "no change along principal directions" (matches the CPU
              toy's project_grad_to_bottom_subspace; the one-sided
              top-top-block reading showed no effect in the toy).
Metrics: exact-match accuracy on 48 held-out prompts per domain.
  retention = task-A acc after phase B; learning = task-B acc after B.
Writes /content/sst_run/metrics.json + metrics.csv. Resumable: completed
phases are skipped on re-run (marker files under /content/sst_run/).

Bounded simplifications (documented): SST targets q_proj/v_proj of every
layer (SVD of W0 computed once, cached; no refresh as weights drift --
LoRA drift is small vs W0 scale); synthetic templated data; 3 epochs.

Run on the VM:  python3 /content/colab_sst_9b.py   (see run log tail)
"""

import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
RUN = "/content/sst_run"
os.makedirs(RUN, exist_ok=True)
SEED = 20261016
random.seed(SEED)
SST_SESSION = "2026-10-16-2610.22293-sst"


def log(msg):
    print("[sst9b] %s" % msg, flush=True)


# Drive-persistent checkpoints (survive Colab VM reclamation). drive_ckpt.py
# is uploaded to /content alongside this driver; all failures are non-fatal
# and the driver falls back to /content-only checkpoints.
try:
    import drive_ckpt
    _DRIVE_OK = drive_ckpt.restore(RUN, SST_SESSION)
    log("drive restore: %s" % ("restored" if _DRIVE_OK else "nothing to restore"))
except Exception as e:
    _DRIVE_OK = False
    print("[sst9b] drive restore skipped: %s" % e, flush=True)


def marker(name):
    return os.path.join(RUN, name + ".done")


def done(name):
    return os.path.exists(marker(name))


def mark(name):
    open(marker(name), "w").write("ok")
    try:
        import drive_ckpt
        drive_ckpt.sync(RUN, SST_SESSION)
    except Exception as e:
        log("drive sync skipped: %s" % e)


# ---------------------------------------------------------------- deps
if not done("deps"):
    log("installing deps...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/huggingface/transformers.git",
                    "peft", "bitsandbytes", "accelerate", "safetensors"],
                   check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False)
    mark("deps")
    log("deps done")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------------------------------------------------------- data
FINANCE = [
    ("contango", "futures above spot"), ("backwardation", "futures below spot"),
    ("alpha", "excess return measure"), ("beta", "market sensitivity measure"),
    ("arbitrage", "risk-free price profit"), ("liquidity", "ease of trading"),
    ("volatility", "price fluctuation degree"), ("dividend", "shareholder profit payout"),
    ("equity", "ownership stake"), ("bond", "fixed income debt"),
    ("option", "right not obligation"), ("future", "agreed future trade"),
    ("hedge", "risk offset position"), ("leverage", "borrowed capital use"),
    ("margin", "collateral for trading"), ("short", "sell borrowed asset"),
    ("ETF", "exchange traded basket"), ("IPO", "first public offering"),
    ("bull market", "rising prices trend"), ("bear market", "falling prices trend"),
    ("portfolio", "asset collection"), ("diversification", "risk spreading strategy"),
    ("yield", "return on investment"), ("spread", "price difference gap"),
]
MEDICAL = [
    ("tachycardia", "fast heart rate"), ("bradycardia", "slow heart rate"),
    ("hypertension", "high blood pressure"), ("hypotension", "low blood pressure"),
    ("anemia", "low red cells"), ("biopsy", "tissue sample test"),
    ("vaccine", "immunity injection"), ("antibiotic", "bacteria killer drug"),
    ("fracture", "broken bone"), ("sprain", "ligament stretch injury"),
    ("diagnosis", "disease identification"), ("prognosis", "outcome prediction"),
    ("symptom", "disease sign"), ("syndrome", "symptom cluster"),
    ("acute", "sudden severe onset"), ("chronic", "long term condition"),
    ("benign", "non cancerous growth"), ("malignant", "cancerous growth"),
    ("inflammation", "swelling immune response"), ("edema", "fluid swelling"),
    ("nausea", "urge to vomit"), ("placebo", "inactive treatment"),
    ("dosage", "drug amount"), ("allergy", "immune overreaction"),
]
TRAIN_TMPL = [
    "Q: What does '{t}' mean in {d}? A: {a}",
    "Define '{t}' ({d}): {a}",
    "{D} term '{t}' means {a}.",
    "Q: Explain {t} briefly ({d}). A: {a}",
    "'{t}' in {d} refers to {a}.",
    "Q: {t} ({d} definition)? A: {a}",
    "In {d}, {t} is {a}.",
    "What is {t} in {d}? {a}.",
    "Glossary - {t} ({d}): {a}",
    "Q: Give the {d} meaning of '{t}'. A: {a}",
]
EVAL_TMPL = [
    "Q: Define the {d} term '{t}'. A:",
    "{D}: what is '{t}'? A:",
]


def build_pairs(terms, domain, Domain):
    train, ev = [], []
    for t, a in terms:
        for i, tp in enumerate(TRAIN_TMPL):
            train.append(tp.format(t=t, a=a, d=domain, D=Domain))
        for tp in EVAL_TMPL:
            ev.append((tp.format(t=t, d=domain, D=Domain), a))
    rng = random.Random(SEED)
    rng.shuffle(train)
    return train, ev


trainA, evalA = build_pairs(FINANCE, "finance", "Finance")
trainB, evalB = build_pairs(MEDICAL, "medicine", "Medicine")
log("data: %d/%d train, %d/%d eval" % (len(trainA), len(trainB), len(evalA), len(evalB)))

# ---------------------------------------------------------------- weights
WFILES = (["config.json", "model.safetensors.index.json", "tokenizer.json",
           "tokenizer_config.json", "vocab.json", "merges.txt"] +
          ["model.safetensors-0000%d-of-00004.safetensors" % i for i in range(1, 5)])
WURL = "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/"
WDIR = "/content/qwen35-9b"
if not done("weights"):
    subprocess.run(["apt-get", "install", "-y", "aria2"],
                   check=False, capture_output=True)
    os.makedirs(WDIR, exist_ok=True)
    with open("/content/dl.txt", "w") as f:
        for fn in WFILES:
            f.write("%s%s\n  dir=%s\n  out=%s\n" % (WURL, fn, WDIR, fn))
    missing = [fn for fn in WFILES if not os.path.exists(WDIR + "/" + fn)]
    if missing:
        log("downloading weights...")
        subprocess.run(["aria2c", "-x16", "-s16", "-j4", "-i", "/content/dl.txt",
                        "--summary-interval=0", "--console-log-level=error",
                        "--file-allocation=none"], check=True)
    mark("weights")
    log("weights ready")


def load_base():
    tok = AutoTokenizer.from_pretrained(WDIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left padding: correct batched generation for eval
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        WDIR, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    return tok, model


LORA_CFG = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")


def train_phase(model, tok, texts, epochs, lr, tag, sst_projectors=None,
                sst_adapter="b"):
    model.train()
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    texts = list(texts)
    bs, max_len = 4, 160
    step = 0
    for ep in range(epochs):
        rng = random.Random(SEED + ep)
        rng.shuffle(texts)
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(dev)
            out = model(**enc, labels=enc["input_ids"])
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if sst_projectors:
                with torch.no_grad():
                    for mod, (u_r, v_r) in sst_projectors.items():
                        # B_b <- P_left B_b ; A_b <- A_b P_right (exact
                        # two-sided SST on the phase-B adapter ONLY; the
                        # frozen phase-A adapter ("default") is never
                        # touched -- projecting it would amputate prior
                        # knowledge, the flaw of the first 9B attempt)
                        lb = mod.lora_B[sst_adapter]
                        la = mod.lora_A[sst_adapter]
                        b, a = lb.weight, la.weight
                        ur, vr = u_r.to(b.dtype), v_r.to(a.dtype)
                        b.copy_(b - ur @ (ur.T @ b))
                        a.copy_(a - (a @ vr) @ vr.T)
            step += 1
            if step % 50 == 0:
                log("%s ep%d step%d loss=%.3f" % (tag, ep, step, loss.item()))
    return model


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


@torch.no_grad()
def evaluate(model, tok, eval_pairs):
    model.eval()
    correct = 0
    for j in range(0, len(eval_pairs), 8):
        chunk = eval_pairs[j:j + 8]
        prompts = [p for p, _ in chunk]
        dev = next(model.parameters()).device
        enc = tok(prompts, padding=True, return_tensors="pt").to(dev)
        gen = model.generate(**enc, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for k, (_, ans) in enumerate(chunk):
            txt = tok.decode(gen[k][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            if norm(ans) and norm(ans) in norm(txt):
                correct += 1
    return correct / len(eval_pairs)


def compute_sst_projectors(model, protect_frac=0.9):
    """SVD of frozen W0 for q_proj/v_proj of every layer; cache to disk."""
    cache = os.path.join(RUN, "sst_projectors.pt")
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu", weights_only=True)
        log("loaded cached SST projectors (%d modules)" % len(d))
        dev = next(model.parameters()).device
        return {k: (u.to(dev), v.to(dev)) for k, (u, v) in d.items()}
    import bitsandbytes.functional as F
    proj = {}
    raw = {}
    mods = [(n, m) for n, m in model.named_modules()
            if n.endswith(("q_proj", "v_proj")) and hasattr(m, "weight")]
    log("SVD over %d modules..." % len(mods))
    for n, m in mods:
        w = m.weight
        dq = F.dequantize_4bit(w.data, w.quant_state).float().cuda()
        u, _s, vh = torch.linalg.svd(dq, full_matrices=False)
        del dq
        r = max(1, int(round(protect_frac * min(u.shape[1], vh.shape[0]))))
        r = min(r, min(u.shape[1], vh.shape[0]) - 1)
        u_r = u[:, :r].to(torch.bfloat16).cpu()
        v_r = vh[:r, :].T.contiguous().to(torch.bfloat16).cpu()
        raw[n] = (u_r, v_r)
        torch.cuda.empty_cache()
    torch.save(raw, cache)
    log("SVD done, cached")
    dev = next(model.parameters()).device
    return {k: (u.to(dev), v.to(dev)) for k, (u, v) in raw.items()}


def resolve_lora_modules(peft_model, projectors):
    """Map SVD projector name -> the peft LoRA linear module.

    SVD names come from the base model (e.g. ``model.layers.0...q_proj``);
    after peft wrapping the same module lives under a longer name ending
    with the base name and carries ``lora_A``/``lora_B``.
    """
    out = {}
    for n, m in peft_model.named_modules():
        if not hasattr(m, "lora_A"):
            continue
        for pname, (u_r, v_r) in projectors.items():
            if n.endswith(pname):
                out[m] = (u_r, v_r)
                break
    return out


def save_metrics(row):
    p = os.path.join(RUN, "metrics.json")
    rows = json.load(open(p)) if os.path.exists(p) else []
    rows = [r for r in rows if r["arm"] != row["arm"]] + [row]
    json.dump(rows, open(p, "w"), indent=1)
    with open(os.path.join(RUN, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)
    log("metrics: %s" % json.dumps(row))


# ---------------------------------------------------------------- phase A
tok, model = load_base()
if done("phaseA"):
    model = PeftModel.from_pretrained(model, os.path.join(RUN, "adapter_A"))
    log("phase A: loaded saved adapter")
else:
    model = get_peft_model(model, LORA_CFG)
    train_phase(model, tok, trainA, epochs=3, lr=2e-4, tag="phaseA")
    model.save_pretrained(os.path.join(RUN, "adapter_A"))
    mark("phaseA")
accA = evaluate(model, tok, evalA)
log("phase A done: task-A acc=%.3f" % accA)
del model
torch.cuda.empty_cache()


def set_lora_trainable(model):
    n = 0
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name)
        n += p.requires_grad
    log("trainable params: %d" % n)


def build_combined_adapter(model):
    """Add a rank-32 'combined' adapter = exact sum of 'default' + 'b'.

    B_C = [B_A, B_B], A_C = [A_A; A_B]  =>  B_C @ A_C = B_A@A_A + B_B@A_B.
    With lora_alpha=64, r=32 the peft scaling (2.0) matches the r=16/a=32
    adapters, so 'combined' reproduces both adapters' joint forward
    exactly. (This peft version's set_adapter() takes one adapter only.)
    """
    cfg_c = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                       task_type="CAUSAL_LM")
    model.add_adapter("combined", cfg_c)
    with torch.no_grad():
        for n, m in model.named_modules():
            if not hasattr(m, "lora_A"):
                continue
            Ba = m.lora_B["default"].weight.data
            Aa = m.lora_A["default"].weight.data
            Bb = m.lora_B["b"].weight.data
            Ab = m.lora_A["b"].weight.data
            m.lora_B["combined"].weight.data.copy_(
                torch.cat([Ba, Bb], dim=1))
            m.lora_A["combined"].weight.data.copy_(
                torch.cat([Aa, Ab], dim=0))
    model.set_adapter("combined")
    log("combined adapter built (rank 32 = A + B)")


def run_arm(arm):
    """Phase-B arms. Faithful LoRA port of SST (fixed 2026-10-16):

    * "plain"  (done): keep training the phase-A adapter ("default").
    * "frozen": freeze phase-A adapter, train a fresh adapter "b"
      unrestricted -- isolates the freezing effect.
    * "sst":    freeze phase-A adapter, train fresh adapter "b" with the
      two-sided SST projection on "b" only -- isolates SST's
      interference reduction. The phase-A adapter is NEVER projected
      (projecting it amputated prior knowledge: first attempt's sst
      retention was 0.417 vs plain 0.729).

    At eval, "default"+"b" are combined exactly via rank-concatenation
    (build_combined_adapter) since this peft version activates one
    adapter at a time.
    """
    tag = "arm_%s" % arm
    if done(tag):
        log("%s already done, skipping" % tag)
        return
    assert arm in ("plain", "frozen", "sst")
    tok3, base2 = load_base()
    sst = None
    if arm == "sst":
        proj = compute_sst_projectors(base2)  # base model, before wrapping
    model = PeftModel.from_pretrained(base2, os.path.join(RUN, "adapter_A"))
    if arm in ("frozen", "sst"):
        torch.manual_seed(SEED + 123)  # identical adapter-B init, both arms
        model.add_adapter("b", LORA_CFG)
        model.set_adapter("b")  # single active adapter (this peft version)
        for n, p in model.named_parameters():
            p.requires_grad = (".lora_A.b." in n or ".lora_B.b." in n)
        ntrain = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        log("trainable params (adapter b): %d tensors" % ntrain)
        assert ntrain > 0
    else:
        set_lora_trainable(model)
        model.set_adapter("default")
    if arm == "sst":
        sst = resolve_lora_modules(model, proj)
        log("SST active on %d modules" % len(sst))
        assert len(sst) > 0, "no SST modules resolved!"
    train_phase(model, tok3, trainB, epochs=3, lr=2e-4, tag="phaseB-" + arm,
                sst_projectors=sst, sst_adapter="b")
    if arm in ("frozen", "sst"):
        build_combined_adapter(model)
    ret = evaluate(model, tok3, evalA)
    lrn = evaluate(model, tok3, evalB)
    save_metrics({"arm": arm, "accA_after_A": round(accA, 4),
                  "retention": round(ret, 4), "learning": round(lrn, 4)})
    mark(tag)
    del model
    torch.cuda.empty_cache()


run_arm("plain")
run_arm("frozen")
run_arm("sst")
log("ALL DONE")
