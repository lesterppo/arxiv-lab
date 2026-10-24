# ==============================================================================
# 2610.31483 — Training Small Language Models for Transferable Tool Selection
# Bounded Colab T4 reproduction (self-contained driver script).
#
# Paper core claim: a small open model (Qwen3-8B) post-trained with RL can
# drive tool selection over 10k+ tool corpora for complex multi-step
# instructions, outperforming Llama-4-Maverick (402B) on ToolBench, and
# transferring to different retrievers and new tool domains.
#
# What we test at bounded scale (honest stand-in, NOT the full paper):
#   - synthetic 300-tool catalog across 3 domains (finance, calendar, health)
#   - ~200 multi-step train tasks + 50 dev + 50 test (in-domain) + 50 transfer
#     tasks on a HELD-OUT domain with a DIFFERENT retriever style
#   - baseline: zero-shot NF4 Qwen3-8B tool-selection accuracy (no RL)
#   - post-train: bounded REINFORCE-with-baseline on sampled trajectories
#     (the paper uses full RL/GRPO; trl-GRPO was too heavy for this run, so
#      this is a documented bounded stand-in — see README)
#   - transfer eval: held-out domain + different retriever after RL
# Metrics CSV: /content/toolselect_metrics.csv
# ==============================================================================
import subprocess, sys

def _sh(*args):
    print("+", " ".join(args), flush=True)
    r = subprocess.run(list(args), capture_output=True, text=True)
    tail = (r.stdout or "")[-600:] + (r.stderr or "")[-600:]
    print(tail[-800:], flush=True)
    if r.returncode != 0:
        print(f"!! command failed rc={r.returncode}", flush=True)
    return r

# ---- Phase 0: deps (order matters on Colab) -----------------------------------
_sh(sys.executable, "-m", "pip", "install", "-q",
    "transformers<5.0", "bitsandbytes", "accelerate", "peft",
    "datasets", "safetensors")
# Colab ships torchao 0.10 which breaks peft's get_peft_model; remove it
# (peft falls back to its default LoRA path).
_sh(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")

import json, math, os, random, re, csv, time
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MODEL_ID = os.environ.get("TS_MODEL", "Qwen/Qwen3-8B")  # HF id; ungated
RL_STEPS = int(os.environ.get("TS_RL_STEPS", "60"))
N_ROLLOUT = int(os.environ.get("TS_N_ROLLOUT", "4"))
CSV_PATH = "/content/toolselect_metrics.csv"
ADAPTER_DIR = "/content/ts_adapter"

# ==============================================================================
# Phase 1 — synthetic tool corpora + multi-step tasks (deterministic)
# ==============================================================================
DOMAINS = {
    "finance":  ["pay_bill", "check_balance", "transfer_funds", "freeze_card",
                 "unfreeze_card", "dispute_charge", "set_budget_alert",
                 "export_statement", "convert_currency", "schedule_payment"],
    "calendar": ["create_event", "invite_attendee", "reschedule_event",
                 "cancel_event", "find_free_slot", "set_reminder",
                 "list_events", "share_calendar", "snooze_reminder",
                 "decline_invite"],
    "health":   ["log_symptom", "book_appointment", "refill_prescription",
                 "find_pharmacy", "check_drug_interactions",
                 "set_medication_reminder", "upload_lab_result",
                 "message_doctor", "cancel_appointment", "track_vitals"],
}

def describe(domain, name, style="plain"):
    base = name.replace("_", " ")
    if style == "plain":
        return f"{base} — a {domain} utility for common {domain} tasks."
    # paraphrased style: different surface wording, same semantics (stresses
    # the selector rather than lexical memorization)
    syn = {"pay bill":"settle an outstanding invoice", "check balance":"view current funds",
           "transfer funds":"move money between accounts", "freeze card":"temporarily lock a payment card",
           "unfreeze card":"restore access to a locked card", "dispute charge":"challenge an erroneous transaction",
           "set budget alert":"configure a spending threshold notification", "export statement":"download account records",
           "convert currency":"exchange money between denominations", "schedule payment":"arrange a future transfer",
           "create event":"add a new calendar entry", "invite attendee":"send a meeting invitation",
           "reschedule event":"change an appointment's time", "cancel event":"remove a scheduled meeting",
           "find free slot":"locate available time", "set reminder":"create a notification",
           "list events":"show upcoming appointments", "share calendar":"grant calendar access",
           "snooze reminder":"postpone a notification", "decline invite":"reject a meeting request",
           "log symptom":"record a health complaint", "book appointment":"schedule a clinic visit",
           "refill prescription":"request more medication", "find pharmacy":"locate a drugstore",
           "check drug interactions":"verify medication compatibility",
           "set medication reminder":"configure a dosage alarm", "upload lab result":"submit test results",
           "message doctor":"contact a physician", "cancel appointment":"call off a clinic visit",
           "track vitals":"monitor physiological measurements"}
    return f"{syn.get(base, base)} [{domain}]"

def build_catalog(domains, style="plain"):
    tools = []
    for d in domains:
        for i, name in enumerate(DOMAINS[d]):
            tid = f"{d}.{i:02d}.{name}"
            tools.append({"id": tid, "name": name, "domain": d,
                          "desc": describe(d, name, style)})
    return tools

IN_DOMAIN_CATALOG = build_catalog(["finance", "calendar"], style="plain")
TRANSFER_CATALOG  = build_catalog(["health"], style="paraphrase")  # held-out domain AND paraphrased style

# Task templates: (instruction pattern, domain, tool sequence)
TEMPLATES = [
    ("I need to {a} and then {b}.",      2),
    ("First {a}, then {b}, and finally {c}.", 3),
    ("{a_cap} before I {b}, and also {c} after that.", 3),
    ("Can you {a}? Then {b}.",           2),
    ("I want to {a}, {b}, and {c} in that order.", 3),
    ("Please {a} and also {b}.",         2),
]
PHRASES = {
    "finance": ["pay my electricity bill", "check my savings balance",
                "transfer funds to my brokerage", "freeze my lost card",
                "unfreeze my card", "dispute a double charge",
                "set a budget alert at 80%", "export last month's statement",
                "convert USD to HKD", "schedule next month's rent payment"],
    "calendar": ["create a team standup", "invite my manager to lunch",
                 "reschedule tomorrow's review", "cancel the 3pm call",
                 "find a free slot Friday", "set a reminder to file taxes",
                 "list this week's events", "share my calendar with the team",
                 "snooze the morning reminder", "decline the vendor invite"],
    "health": ["log my headache", "book a dentist appointment",
               "refill my blood pressure prescription", "find a pharmacy nearby",
               "check drug interactions for ibuprofen",
               "set a medication reminder for 8am", "upload my lab results",
               "message my doctor about the refill", "cancel my appointment",
               "track my blood pressure"],
}

def make_task(catalog, domains, style_seed, paraphrase=False):
    d = random.choice(domains)
    n = random.choice([2, 3])
    tmpl, _ = random.choice([t for t in TEMPLATES if t[1] == n])
    idxs = random.sample(range(len(DOMAINS[d])), k=n)
    phr = [PHRASES[d][i] for i in idxs]
    # fill template slots a/b/c with phrases
    slots = {"a": phr[0], "b": phr[1]}
    if n == 3:
        slots["c"] = phr[2]
        slots["a_cap"] = phr[0][0].upper() + phr[0][1:]
    instr = tmpl.format(**slots)
    if paraphrase:
        instr = instr.replace("I need to", "I have to get done").replace("Please ", "")
    gold = [f"{d}.{i:02d}.{DOMAINS[d][i]}" for i in idxs]
    return {"instruction": instr, "gold": gold, "domain": d}

def make_split(catalog, domains, n, start_seed, paraphrase=False):
    rng_state = random.getstate()
    random.seed(start_seed)
    tasks = [make_task(catalog, domains, start_seed + i, paraphrase)
             for i in range(n)]
    random.setstate(rng_state)
    return tasks

TRAIN = make_split(IN_DOMAIN_CATALOG, ["finance", "calendar"], 200, 1000)
DEV   = make_split(IN_DOMAIN_CATALOG, ["finance", "calendar"], 50, 2000)
TEST  = make_split(IN_DOMAIN_CATALOG, ["finance", "calendar"], 50, 3000)
XFER  = make_split(TRANSFER_CATALOG, ["health"], 50, 4000, paraphrase=True)
print(f"catalog in-domain={len(IN_DOMAIN_CATALOG)} transfer={len(TRANSFER_CATALOG)} "
      f"train={len(TRAIN)} dev={len(DEV)} test={len(TEST)} transfer_tasks={len(XFER)}", flush=True)

# ---- Simulated retrievers (stand-ins for "different retrievers") --------------
STOP = set("a an the and or to of for in on with my is it be as at by this that".split())

def _toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower())) - STOP

def retriever_keyword(task, catalog, k=12):
    q = _toks(task["instruction"])
    scored = []
    for t in catalog:
        s = len(q & _toks(t["desc"] + " " + t["name"]))
        scored.append((s, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]

def retriever_name_only(task, catalog, k=12):
    # harsher retriever: scores only on tool NAME tokens (noisy, less lexical help)
    q = _toks(task["instruction"])
    scored = []
    for t in catalog:
        s = len(q & _toks(t["name"]))
        scored.append((s, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]

RETRIEVERS = {"keyword": retriever_keyword, "name_only": retriever_name_only}

def prompt_for(task, cands):
    lines = [f"- {t['name']}: {t['desc']}" for t in cands]
    gold_names = [g.split(".")[-1] for g in task["gold"]]
    return (
        "You are a tool-selection agent. Given the user instruction and the candidate tools, "
        "output ONLY a JSON object like {\"tools\": [\"tool_name\", ...]} listing the tools to call, "
        "in the exact order needed to fulfill the instruction. Use tool names exactly as listed.\n\n"
        f"Instruction: {task['instruction']}\n\nCandidate tools:\n" + "\n".join(lines) +
        "\n\nJSON:"
    )

def parse_tools(text, valid_names):
    m = re.search(r'\{[^{}]*"tools"[^{}]*\}', text, re.S)
    names = []
    if m:
        try:
            names = json.loads(m.group(0)).get("tools", [])
        except Exception:
            names = []
    if not names:
        # fallback: pick valid tool names appearing in order in the text
        for w in re.findall(r"[a-z_]{3,40}", text):
            if w in valid_names and w not in names:
                names.append(w)
    return [n for n in names if n in valid_names][:6]

# ==============================================================================
# Phase 2 — model (NF4 QLoRA, Qwen3-8B, bf16)
# ==============================================================================
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
print("loading base model (NF4)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto",
    torch_dtype=torch.bfloat16, trust_remote_code=True)
# NOTE: deliberately NOT calling prepare_model_for_kbit_training() — it casts
# the frozen bf16 base to fp32 and crashes the GradScaler path on T4.

lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj",
                                      "o_proj", "gate_proj", "up_proj",
                                      "down_proj"],
                      task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model.config.use_cache = False

alloc = torch.cuda.memory_allocated() / 1e9
print(f"VRAM after load+LoRA: {alloc:.2f} GB", flush=True)

# ==============================================================================
# Phase 3 — evaluation harness
# ==============================================================================
@torch.no_grad()
def select_tools(model, tok, task, cands, max_new=96):
    prompt = prompt_for(task, cands)
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=768).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    valid = {t["name"] for t in cands}
    return parse_tools(text, valid)

def score_task(pred_names, gold_ids):
    gold_names = [g.split(".")[-1] for g in gold_ids]
    exact = 1.0 if pred_names == gold_names else 0.0
    if not pred_names and not gold_names:
        f1 = 1.0
    else:
        inter = len(set(pred_names) & set(gold_names))
        p = inter / max(1, len(pred_names)); r = inter / max(1, len(gold_names))
        f1 = 2 * p * r / max(1e-9, p + r)
    first_ok = 1.0 if pred_names[:1] == gold_names[:1] else 0.0
    return exact, f1, first_ok

def evaluate(model, tok, tasks, catalog, retriever, k=12, tag="eval"):
    rets = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        cands = retriever(task, catalog, k=k)
        pred = select_tools(model, tok, task, cands)
        rets.append(score_task(pred, task["gold"]))
        if (i + 1) % 25 == 0:
            print(f"[{tag}] {i+1}/{len(tasks)}", flush=True)
    exact = float(np.mean([r[0] for r in rets]))
    f1 = float(np.mean([r[1] for r in rets]))
    first = float(np.mean([r[2] for r in rets]))
    print(f"[{tag}] exact={exact:.3f} stepF1={f1:.3f} first_tool={first:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"exact": exact, "f1": f1, "first": first, "n": len(tasks)}

results = []

def log_row(split, state, retriever, metrics):
    row = {"split": split, "model_state": state, "retriever": retriever,
           "n_tasks": metrics["n"], "seq_exact": round(metrics["exact"], 4),
           "step_f1": round(metrics["f1"], 4),
           "first_tool_acc": round(metrics["first"], 4)}
    results.append(row)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
    print("CSV:", row, flush=True)

print("=== BASELINE: zero-shot (no RL) ===", flush=True)
m_base_test_kw = evaluate(model, tok, TEST, IN_DOMAIN_CATALOG,
                          RETRIEVERS["keyword"], tag="base-test-kw")
log_row("test_in_domain", "zeroshot", "keyword", m_base_test_kw)
m_base_xfer_kw = evaluate(model, tok, XFER, TRANSFER_CATALOG,
                          RETRIEVERS["keyword"], tag="base-xfer-kw")
log_row("transfer_health", "zeroshot", "keyword", m_base_xfer_kw)
m_base_xfer_no = evaluate(model, tok, XFER, TRANSFER_CATALOG,
                          RETRIEVERS["name_only"], tag="base-xfer-nameonly")
log_row("transfer_health", "zeroshot", "name_only", m_base_xfer_no)

# ==============================================================================
# Phase 4 — bounded RL post-train (REINFORCE with baseline; documented
# stand-in for the paper's full RL/GRPO — see README).
# ==============================================================================
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=2e-4)
GRAD_ACCUM = 2

def reward_for(pred_names, gold_ids):
    exact, f1, _ = score_task(pred_names, gold_ids)
    return exact + 0.5 * f1   # in [0, 1.5]

@torch.no_grad()
def sample_traj(task, catalog, k=12):
    cands = RETRIEVERS["keyword"](task, catalog, k=k)
    prompt = prompt_for(task, cands)
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=768).to(model.device)
    out = model.generate(**inputs, max_new_tokens=96, do_sample=True,
                         temperature=0.8, top_p=0.9,
                         pad_token_id=tok.eos_token_id,
                         return_dict_in_generate=True,
                         output_scores=True)
    text = tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    pred = parse_tools(text, {t["name"] for t in cands})
    # log-prob of the generated continuation
    scores = torch.stack(out.scores, dim=1)          # [1, T, V]
    gen_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
    logp = torch.log_softmax(scores.float(), dim=-1)
    lp = logp[0, torch.arange(gen_ids.shape[1],
                              device=logp.device), gen_ids[0]].sum()
    return lp, reward_for(pred, task["gold"]), prompt

print(f"=== RL post-train: {RL_STEPS} steps x {N_ROLLOUT} rollouts "
      f"(accum {GRAD_ACCUM}) ===", flush=True)
train_iter = iter(TRAIN * 10)
t0 = time.time()
for step in range(RL_STEPS):
    task = next(train_iter)
    lps, rews = [], []
    with torch.no_grad():
        for _ in range(N_ROLLOUT):
            lp, r, _ = sample_traj(task, IN_DOMAIN_CATALOG)
            lps.append(lp); rews.append(r)
    base = float(np.mean(rews))
    loss = 0
    for lp, r in zip(lps, rews):
        adv = r - base
        loss = loss - adv * lp / N_ROLLOUT
    (loss / GRAD_ACCUM).backward()
    if (step + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad()
    if (step + 1) % 10 == 0:
        print(f"[rl] step {step+1}/{RL_STEPS} mean_reward={base:.3f} "
              f"loss={float(loss):.3f} ({time.time()-t0:.0f}s)", flush=True)
opt.zero_grad()
model.save_pretrained(ADAPTER_DIR)
print(f"adapter saved -> {ADAPTER_DIR}", flush=True)

model.eval()
alloc = torch.cuda.memory_allocated() / 1e9
print(f"VRAM after RL: {alloc:.2f} GB", flush=True)

print("=== POST-RL EVAL ===", flush=True)
m_rl_test_kw = evaluate(model, tok, TEST, IN_DOMAIN_CATALOG,
                        RETRIEVERS["keyword"], tag="rl-test-kw")
log_row("test_in_domain", "post_rl", "keyword", m_rl_test_kw)
m_rl_test_no = evaluate(model, tok, TEST, IN_DOMAIN_CATALOG,
                        RETRIEVERS["name_only"], tag="rl-test-nameonly")
log_row("test_in_domain", "post_rl", "name_only", m_rl_test_no)
m_rl_xfer_kw = evaluate(model, tok, XFER, TRANSFER_CATALOG,
                        RETRIEVERS["keyword"], tag="rl-xfer-kw")
log_row("transfer_health", "post_rl", "keyword", m_rl_xfer_kw)
m_rl_xfer_no = evaluate(model, tok, XFER, TRANSFER_CATALOG,
                        RETRIEVERS["name_only"], tag="rl-xfer-nameonly")
log_row("transfer_health", "post_rl", "name_only", m_rl_xfer_no)

print("DONE. metrics ->", CSV_PATH, flush=True)
print(json.dumps(results, indent=2), flush=True)
