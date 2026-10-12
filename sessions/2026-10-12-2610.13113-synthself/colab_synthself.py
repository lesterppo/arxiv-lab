"""SynthSelf on Colab T4 (arXiv:2610.13113).

Continual post-training via Synthetic Self-Distillation, bounded for a
free Colab T4 at the 9B class (Qwen/Qwen3.5-9B, NF4 QLoRA r=16/a=32,
bf16) per the arxiv read-and-test training-track protocol.

Paper claims under test:
  1. In-distribution self-generated supervision (SynthSelf) gives better
     ACQUISITION of a new capability than higher-shift supervision.
  2. It also gives better RETENTION (less forgetting) of prior abilities.
  3. Root cause: acquisition/forgetting correlate with distribution shift
     (measured here as mean response perplexity under the base model).

Design (bounded):
  * New capability: a synthetic tag language (<up>/<lo>/<rev>, nested,
    rule-verifiable) the base model has never seen.
  * Arm A (SynthSelf): self-generate with in-context demos, keep only
    verifier-correct outputs (rejection sampling), SFT on those.
  * Arm B (shifted): SAME prompts and SAME correct answers, but wrapped
    in a chatty style -> higher PPL under the base model (distribution
    shift), SFT on those.
  * Retention: arithmetic + short factual items the BASE answers
    correctly (filtered), re-measured after each arm.

Bounded deviations from the paper (documented): the "minimally larger
same-family aligner" is the 9B model itself (single-T4 constraint);
two arms only; tiny data / few steps.

Self-contained: pip installs at top (incl. `pip uninstall -y torchao`
for the peft/torchao-0.10 conflict), writes metrics incrementally to
/content/synthself_metrics.json and /content/synthself_metrics.csv.

Run on the VM:  python3 /content/colab_synthself.py
"""

import json
import math
import os
import random
import re
import subprocess
import sys
import time

# Reduce CUDA fragmentation (the loss's fp32 cast is spiky on 15GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------- deps
print("[deps] installing...", flush=True)
# NOTE: Qwen3.5's `qwen3_5` arch only exists in transformers main
# (overrides the usual transformers<5.0 training-track pin).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/huggingface/transformers.git",
                "peft", "bitsandbytes", "accelerate", "safetensors"],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False)
print("[deps] done", flush=True)

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------------------------------------------------------- config
BASE = "Qwen/Qwen3.5-9B"
OUT = "/content/synthself_metrics.json"
OUTCSV = "/content/synthself_metrics.csv"
ADIR = "/content/synthself_adapters"
STAGE1 = "/content/synthself_stage1.json"
STAGE2 = "/content/synthself_stage2.json"
SEED = 7
random.seed(SEED)

N_TRAIN_PROMPTS = 150
N_TEST = 60
N_RETAIN_ARITH = 60   # pre-filter pool; keep base-correct up to 40
N_RETAIN_FACT = 40    # pre-filter pool; keep base-correct up to 30
EPOCHS = 4
BATCH = 1            # micro-batch (OOM safety on 15GB); effective 4 via accum
GRAD_ACCUM = 4
MAXLEN = 320
LR = 2e-4
MAX_NEW = 96

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[env] device=%s cuda=%s" % (DEVICE, torch.cuda.is_available()),
      flush=True)
if DEVICE == "cuda":
    print("[env] gpu=%s mem=%.1fGB" % (
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 1e9), flush=True)

# ---------------------------------------------------------------- task
VOCAB = ["alpha", "bravo", "cargo", "delta", "ember", "frost", "glimmer",
         "harbor", "ivory", "jungle", "karma", "lumen", "meadow", "north",
         "onyx", "prism", "quartz", "river", "stone", "tiger"]


def render_tags(s):
    """Ground-truth renderer for the tag language (innermost-first)."""
    pat = re.compile(r"<(up|lo|rev)>([^<>]*)</\1>")

    def repl(m):
        tag, inner = m.group(1), m.group(2)
        if tag == "up":
            return inner.upper()
        if tag == "lo":
            return inner.lower()
        return inner[::-1]

    prev = None
    while prev != s:
        prev = s
        s = pat.sub(repl, s)
    return s


def random_prompt(rng):
    """Random tag-language input string."""
    n_spans = rng.randint(1, 3)
    parts = []
    for _ in range(n_spans):
        depth = rng.randint(1, 3)
        inner = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 3)))
        for _ in range(depth):
            tag = rng.choice(["up", "lo", "rev"])
            inner = "<%s>%s</%s>" % (tag, inner, tag)
        parts.append(inner)
    filler = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(0, 2)))
    # NOTE: no "Convert: " prefix here; callers add it.
    return " ".join(parts + ([filler] if filler else [])).strip()


INSTR = ("You convert tag-language strings. Rules: <up>X</up> makes X "
         "UPPERCASE, <lo>X</lo> makes X lowercase, <rev>X</rev> reverses X. "
         "Apply innermost tags first. Reply with ONLY the converted string, "
         "no explanation.")


def few_shot_block(examples):
    blk = INSTR + "\n"
    for inp, out in examples:
        blk += "Input: %s\nOutput: %s\n" % (inp, out)
    return blk


def gen_prompt(inp, demos):
    return few_shot_block(demos) + "Input: %s\nOutput:" % inp


def test_prompt(inp):
    # 1-shot eval prompt (harder than train's 3-shot -> real acquisition test)
    return few_shot_block([DEMO_POOL[0]]) + "Input: %s\nOutput:" % inp


# fixed demo pool (deterministic)
_rng0 = random.Random(SEED)
DEMO_POOL = []
for _ in range(6):
    p = random_prompt(_rng0)
    DEMO_POOL.append(("Convert: " + p, render_tags(p)))

T0 = time.time()

# ---------------------------------------------------------------- model
print("[model] loading %s (NF4)..." % BASE, flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16)
base_model.eval()
print("[model] loaded", flush=True)


@torch.no_grad()
def generate(model, prompts, max_new=MAX_NEW, batch=4):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j, g in enumerate(gen):
            inp_len = enc["input_ids"].shape[1]
            text = tok.decode(g[inp_len:], skip_special_tokens=True)
            outs.append(text.strip().split("\n")[0].strip())
    return outs


@torch.no_grad()
def mean_ppl(model, prompt, response):
    """Mean token perplexity of `response` given `prompt` under model."""
    full = prompt + " " + response
    enc = tok(full, return_tensors="pt").to(DEVICE)
    plen = len(tok(prompt)["input_ids"])
    labels = enc["input_ids"].clone()
    labels[0, :plen] = -100
    out = model(**enc, labels=labels)
    # shift for causal LM
    logits = out.logits[0, :-1]
    labs = labels[0, 1:]
    mask = labs != -100
    if mask.sum() == 0:
        return float("inf")
    nll = torch.nn.functional.cross_entropy(
        logits[mask], labs[mask], reduction="mean")
    return math.exp(nll.item())


# ---------------------------------------------------------------- phase 1: self-generation + rejection sampling
if os.path.exists(STAGE1):
    print("[stage1] resuming from %s" % STAGE1, flush=True)
    s1 = json.load(open(STAGE1))
    kept = s1["kept"]
    armB = s1["armB"]
    armA_ppl = s1["armA_ppl"]
    armB_ppl = s1["armB_ppl"]
else:
    print("[gen] generating %d train candidates (3-shot ICL)..."
          % N_TRAIN_PROMPTS, flush=True)
    rng = random.Random(SEED + 1)
    train_prompts = [random_prompt(rng) for _ in range(N_TRAIN_PROMPTS)]
    gen_inputs = [gen_prompt("Convert: " + p, DEMO_POOL[1:4])
                  for p in train_prompts]
    cands = generate(base_model, gen_inputs)
    kept = []   # (prompt_text, response)
    for p, c in zip(train_prompts, cands):
        want = render_tags(p)
        if c.strip() == want:
            kept.append((gen_prompt("Convert: " + p, DEMO_POOL[1:4]),
                         c.strip()))
    print("[gen] kept %d/%d after rejection sampling"
          % (len(kept), N_TRAIN_PROMPTS), flush=True)

    print("[ppl] measuring in-distribution PPL of kept responses...",
          flush=True)
    ppls = [mean_ppl(base_model, pr, rs) for pr, rs in kept[:60]]
    armA_ppl = float(sum(ppls) / max(len(ppls), 1))
    print("[ppl] Arm A (SynthSelf) mean response PPL = %.2f" % armA_ppl,
          flush=True)

    # ---------------------------------------------------------- phase 2: shifted arm
    def shift_style(response):
        return ("Sure! Here is the converted result:\n%s\nHope that helps!"
                % response)

    armB = [(pr, shift_style(rs)) for pr, rs in kept]
    ppls_b = [mean_ppl(base_model, pr, rs) for pr, rs in armB[:60]]
    armB_ppl = float(sum(ppls_b) / max(len(ppls_b), 1))
    print("[ppl] Arm B (shifted)  mean response PPL = %.2f" % armB_ppl,
          flush=True)
    json.dump({"kept": kept, "armB": armB, "armA_ppl": armA_ppl,
               "armB_ppl": armB_ppl}, open(STAGE1, "w"))
    print("[stage1] saved %s" % STAGE1, flush=True)
    del cands

# ---------------------------------------------------------------- phase 3: retention set (base-correct only)
def arith_items(rng, n):
    items = []
    for _ in range(n):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        op = rng.choice(["+", "-", "*"])
        q = "Compute %d %s %d. Reply with ONLY the integer, no explanation." % (a, op, b)
        ans = str(eval("%d%s%d" % (a, op, b)))
        items.append((q, ans))
    return items


FACTS = [("The capital of Peru is?", "Lima"),
         ("The chemical symbol for gold is?", "Au"),
         ("How many days are in a leap year?", "366"),
         ("The largest planet in the solar system is?", "Jupiter"),
         ("Water boils at 100 degrees on which scale?", "Celsius"),
         ("The currency of Japan is?", "Yen"),
         ("How many continents are there?", "7"),
         ("The author of '1984' is?", "Orwell"),
         ("What is the square root of 144?", "12"),
         ("The fastest land animal is?", "Cheetah"),
         ("HCl is the formula for?", "Hydrochloric acid"),
         ("How many players on a soccer team?", "11"),
         ("The longest river in the world is?", "Nile"),
         ("What gas do plants absorb?", "Carbon dioxide"),
         ("The first element on the periodic table is?", "Hydrogen"),
         ("How many seconds in an hour?", "3600"),
         ("The capital of Canada is?", "Ottawa"),
         ("What is H2O commonly known as?", "Water"),
         ("The red planet is?", "Mars"),
         ("How many sides does a hexagon have?", "6"),
         ("The currency of the UK is?", "Pound"),
         ("What is 15 percent of 200?", "30"),
         ("The tallest mountain in the world is?", "Everest"),
         ("How many letters in the alphabet?", "26"),
         ("The capital of Australia is?", "Canberra"),
         ("What do bees make?", "Honey"),
         ("The freezing point of water in Fahrenheit is?", "32"),
         ("How many wheels does a tricycle have?", "3"),
         ("The largest ocean is?", "Pacific"),
         ("What is the main ingredient in guacamole?", "Avocado")]


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def accuracy(model, prompts, golds):
    outs = generate(model, prompts, batch=4)
    return sum(1 for o, g in zip(outs, golds) if norm(o) == norm(g)) / len(golds)


def retention_acc(model):
    if not retain:
        return 0.0
    outs = generate(model, [q for q, _ in retain], max_new=24, batch=8)
    return sum(1 for o, (_, a) in zip(outs, retain)
               if norm(o) == norm(a)) / len(retain)


if os.path.exists(STAGE2):
    print("[stage2] resuming from %s" % STAGE2, flush=True)
    s2 = json.load(open(STAGE2))
    retain = s2["retain"]
    base_acq = s2["base_acq"]
    base_ret = s2["base_ret"]
    test_prompts = s2["test_prompts"]
    test_gold = s2["test_gold"]
else:
    print("[retain] filtering retention set to base-correct items...",
          flush=True)
    rng2 = random.Random(SEED + 2)
    arith_pool = arith_items(rng2, N_RETAIN_ARITH)
    fact_pool = FACTS[:N_RETAIN_FACT]
    pool_q = [q for q, _ in arith_pool] + [q for q, _ in fact_pool]
    pool_a = [a for _, a in arith_pool] + [a for _, a in fact_pool]
    pool_out = generate(base_model, pool_q, max_new=24, batch=8)
    retain = [(q, a) for q, a, o in zip(pool_q, pool_a, pool_out)
              if norm(o) == norm(a)][:70]
    print("[retain] kept %d base-correct retention items" % len(retain),
          flush=True)

    # ------------------------------------------------------ phase 4: test set
    test_inputs = [random_prompt(random.Random(SEED + 3))
                   for _ in range(N_TEST)]
    test_prompts = [test_prompt("Convert: " + p) for p in test_inputs]
    test_gold = [render_tags(p) for p in test_inputs]

    print("[eval] base acquisition accuracy...", flush=True)
    base_acq = accuracy(base_model, test_prompts, test_gold)
    print("[eval] base acquisition = %.3f" % base_acq, flush=True)

    base_ret = retention_acc(base_model)
    print("[eval] base retention = %.3f (n=%d)" % (base_ret, len(retain)),
          flush=True)

    json.dump({"retain": retain, "base_acq": base_acq, "base_ret": base_ret,
               "test_prompts": test_prompts, "test_gold": test_gold},
              open(STAGE2, "w"))
    print("[stage2] saved %s" % STAGE2, flush=True)

# ---------------------------------------------------------------- phase 5: QLoRA SFT
def train_arm(name, data):
    print("[train:%s] n=%d ..." % (name, len(data)), flush=True)
    torch.cuda.empty_cache()
    import bitsandbytes as bnb_opt
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(base_model, cfg)
    # gradient checkpointing: trade compute for activation memory
    # (essential on 15GB with a 152k-vocab fp32 loss cast)
    model.gradient_checkpointing_enable()
    model.train()
    # 8-bit AdamW: halves optimizer state vs fp32 AdamW (OOM safety)
    opt = bnb_opt.optim.AdamW8bit(model.parameters(), lr=LR)

    # tokenize once
    texts = [pr + " " + rs for pr, rs in data]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAXLEN)
    plen = [len(tok(pr)["input_ids"]) for pr, _ in data]
    input_ids = enc["input_ids"].to(DEVICE)
    attn = enc["attention_mask"].to(DEVICE)
    labels = input_ids.clone()
    for i, pl in enumerate(plen):
        labels[i, :min(pl, labels.shape[1])] = -100

    n = len(data)
    idx = list(range(n))
    step = 0
    for ep in range(EPOCHS):
        random.Random(SEED + ep).shuffle(idx)
        for s in range(0, n, BATCH):
            bi = idx[s:s + BATCH]
            out = model(input_ids=input_ids[bi], attention_mask=attn[bi],
                        labels=labels[bi])
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if step % 20 == 0:
                print("[train:%s] ep=%d step=%d loss=%.4f" %
                      (name, ep, step, loss.item() * GRAD_ACCUM), flush=True)
    # flush remaining grads
    if step % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    apath = os.path.join(ADIR, name)
    model.save_pretrained(apath)
    print("[train:%s] saved %s (%d steps)" % (name, apath, step), flush=True)

    # eval with adapter active
    model.eval()
    try:
        model.config.use_cache = True  # restore for generate()
    except Exception:
        pass
    acq = accuracy(model, test_prompts, test_gold)
    ret = retention_acc(model)
    print("[eval:%s] acquisition=%.3f retention=%.3f" % (name, acq, ret),
          flush=True)
    # unload adapter -> back to base
    model = model.unload()
    del model
    torch.cuda.empty_cache()
    return acq, ret


os.makedirs(ADIR, exist_ok=True)
resA = train_arm("armA_synthself", kept)
resB = train_arm("armB_shifted", armB)

# ---------------------------------------------------------------- metrics
metrics = {
    "base_model": BASE,
    "n_train_kept": len(kept),
    "n_train_prompts": N_TRAIN_PROMPTS,
    "armA_mean_response_ppl": armA_ppl,
    "armB_mean_response_ppl": armB_ppl,
    "base_acquisition_acc": base_acq,
    "base_retention_acc": base_ret,
    "n_retention": len(retain),
    "n_test": N_TEST,
    "armA_acquisition_acc": resA[0],
    "armA_retention_acc": resA[1],
    "armB_acquisition_acc": resB[0],
    "armB_retention_acc": resB[1],
    "epochs": EPOCHS, "micro_batch": BATCH, "grad_accum": GRAD_ACCUM,
    "lr": LR, "optimizer": "bnb-8bit-AdamW",
    "elapsed_s": time.time() - T0,
}
with open(OUT, "w") as f:
    json.dump(metrics, f, indent=1)
with open(OUTCSV, "w") as f:
    f.write("metric,value\n")
    for k, v in metrics.items():
        f.write("%s,%s\n" % (k, v))
print("[done] metrics -> %s" % OUT, flush=True)
print(json.dumps(metrics, indent=1), flush=True)
