# ==============================================================================
# 2610.31563 — "When to Follow the Teacher: Advantage-Based On-Policy
# Distillation" (AOPD). Bounded Colab T4 reproduction (self-contained driver).
#
# Paper core claim: in on-policy distillation, weight teacher supervision by
# counterfactual advantage under the student — teacher actions that improve
# the student's expected outcome are reinforced, others downweighted — and
# this can be estimated using ONLY standard student rollouts + terminal
# rewards (teacher importance weights on student-sampled actions), with NO
# additional rollouts or reward evaluations. Gains over distillation and RL
# baselines are reported on math/coding/MedQA in external-teacher and
# self-distillation settings.
#
# What we test at bounded scale (honest stand-in, NOT the full paper):
#   - SELF-DISTILLATION setting only: teacher = frozen base model
#     (Qwen3-8B weights, adapter disabled); student = same base + LoRA.
#     No external teacher, no MedQA/coding, no full GRPO pipeline.
#   - Model: Qwen/Qwen3-8B base (8B class; closest HF base to the 9B-class
#     budget used by this track — stated explicitly, not hidden).
#     NF4 QLoRA r=16/alpha=32, bf16, transformers<5.0.
#   - Data: ~200 synthetic verifiable arithmetic prompts (train) + 100
#     held-out prompts (disjoint number pairs), deterministic seed.
#     Terminal reward 1/0 by exact-answer match on the generated continuation.
#   - Student rollouts: k=2 per prompt at temperature 0.8. Per-prompt
#     baseline = mean reward; advantage = reward - baseline.
#   - Arm (a) OPD baseline: unweighted teacher-supervision loss on student
#     rollouts = mean over rollouts of mean-token KL(teacher || student).
#   - Arm (b) AOPD: same KL loss, per-sequence weight =
#         clip(1 + advantage, 0, 2) * clip(importance_weight, 0.1, 10),
#     where importance_weight = exp(sum teacher logp - sum detached
#     student logp) over the rollout's continuation tokens.
#     (Advantage in {-0.5, 0, +0.5} for k=2, so the advantage factor is
#     0.5 / 1.0 / 1.5: downweight / neutral / reinforce. Importance-weight
#     clip [0.1, 10] chosen and stated here.)
#   - Two arms on IDENTICAL data/seed/order; few optimizer steps only.
#   - Eval BEFORE vs AFTER per arm on the held-out set: greedy accuracy
#     and sampled accuracy (k=2, temp 0.8), n=100.
#
# Attempt-1 lesson (2026-10-25, session train-1025-0424, DIED, 0 steps):
# a single batched full-vocab float32 KL forward over 16 sequences OOM'd
# the T4 (13.91 GiB allocated). Fix in this revision: KL computed in
# micro-batches of 4 sequences with per-chunk backward (gradient
# accumulation), generation chunked to 32 prompts, expandable segments
# on. Answer parsing also changed from first-integer to LAST integer in
# the continuation: a base model may restate the problem ("47 + 28 = 75")
# before the answer, and the final number is the answer. Attempt 1's
# BEFORE numbers (0.09/0.09, first-integer parse) are superseded and are
# NOT comparable to this revision's numbers.
#
# Metrics CSV: /content/aopd_metrics.csv (written incrementally, flushed
# per row, so a reclaimed run still leaves partial evidence).
# Completion marker: TRAIN_DONE printed as the VERY LAST stdout line.
# ==============================================================================
import subprocess, sys

def _sh(*args):
    print("+", " ".join(args), flush=True)
    r = subprocess.run(list(args), capture_output=True, text=True)
    tail = (r.stdout or "")[-500:] + (r.stderr or "")[-500:]
    print(tail[-800:], flush=True)
    if r.returncode != 0:
        print(f"!! command failed rc={r.returncode}", flush=True)
    return r

# ---- Phase 0: deps (order matters on Colab) -----------------------------------
_sh(sys.executable, "-m", "pip", "install", "-q",
    "transformers<5.0", "bitsandbytes", "accelerate", "peft", "safetensors")
# Colab ships torchao 0.10 which breaks peft's get_peft_model; remove it.
_sh(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")

import csv, gc, json, os, random, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

SEED = 20261025
random.seed(SEED); torch.manual_seed(SEED)
MODEL_ID = os.environ.get("AOPD_MODEL", "Qwen/Qwen3-8B")
STEPS = int(os.environ.get("AOPD_STEPS", "12"))            # per arm
BATCH_PROMPTS = int(os.environ.get("AOPD_BATCH_PROMPTS", "8"))
K = int(os.environ.get("AOPD_K", "2"))
N_TRAIN = int(os.environ.get("AOPD_N_TRAIN", "200"))
N_TEST = int(os.environ.get("AOPD_N_TEST", "100"))
LR = float(os.environ.get("AOPD_LR", "5e-5"))
CSV_PATH = "/content/aopd_metrics.csv"
IW_LO, IW_HI = 0.1, 10.0

if not torch.cuda.is_available():
    print("NO_GPU: this driver requires a CUDA GPU; refusing to fake a run",
          flush=True)
    sys.exit(2)

# ---- Phase 1: deterministic synthetic arithmetic data -------------------------
FEWSHOT = (
    "Q: What is 12 + 7?\nA: 19\n"
    "Q: What is 9 times 6?\nA: 54\n"
    "Q: What is 41 - 18?\nA: 23\n"
)

def gen_prompts(rng, n, used):
    out = []
    while len(out) < n:
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            a, b = rng.randint(11, 98), rng.randint(11, 98)
            q, gold = f"What is {a} + {b}?", a + b
            if rng.random() < 0.3:
                q, gold = (f"A farmer has {a} apples and buys {b} more. "
                           f"How many apples in total?", a + b)
        elif op == "-":
            a, b = rng.randint(21, 99), rng.randint(11, 20)
            q, gold = f"What is {a} - {b}?", a - b
        else:
            a, b = rng.randint(3, 19), rng.randint(3, 12)
            q, gold = f"What is {a} times {b}?", a * b
        if q in used:
            continue
        used.add(q)
        out.append({"q": q, "gold": gold})
    return out

_rng = random.Random(SEED)
_used = set()
TRAIN = gen_prompts(_rng, N_TRAIN, _used)
TEST = gen_prompts(_rng, N_TEST, _used)
assert not ({p["q"] for p in TRAIN} & {p["q"] for p in TEST}), "split overlap"
print(f"data: train={len(TRAIN)} test={len(TEST)} disjoint", flush=True)

def full_prompt(q):
    return FEWSHOT + f"Q: {q}\nA:"

_INT = re.compile(r"-?\d+")

def parse_answer(text):
    # LAST integer: the model may restate the problem before answering
    # ("47 + 28 = 75"); the final number is the answer.
    ms = _INT.findall(text or "")
    return int(ms[-1]) if ms else None

def reward_of(text, gold):
    return 1.0 if parse_answer(text) == gold else 0.0

# ---- Phase 2: model helpers ----------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def load_base():
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda:0")
    m.config.use_cache = False
    return m

def batch_generate(model, prompts, do_sample, temp=0.8, n_return=1,
                   max_new=12):
    """Greedy (do_sample=False) or sampled generation, chunked to 32
    prompts per generate call to bound KV/logits memory."""
    texts = []
    for start in range(0, len(prompts), 32):
        chunk = prompts[start:start + 32]
        tok.padding_side = "left"
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda:0")
        plen = enc["attention_mask"].sum(dim=1)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new,
                                 do_sample=do_sample,
                                 temperature=temp if do_sample else None,
                                 top_p=0.95 if do_sample else None,
                                 num_return_sequences=n_return,
                                 pad_token_id=tok.pad_token_id)
        for i in range(len(chunk)):
            for r in range(n_return):
                row = out[i * n_return + r]
                p = int(plen[i])
                texts.append(tok.decode(row[p:], skip_special_tokens=True))
    return texts

@torch.no_grad()
def evaluate(model, split_name):
    prompts = [full_prompt(p["q"]) for p in TEST]
    golds = [p["gold"] for p in TEST]
    greedy = batch_generate(model, prompts, do_sample=False)
    gacc = sum(reward_of(t, g) for t, g in zip(greedy, golds)) / len(golds)
    sampled = batch_generate(model, prompts, do_sample=True, n_return=K)
    sacc = sum(reward_of(sampled[i * K + r], golds[i])
               for i in range(len(golds)) for r in range(K)) / (len(golds) * K)
    print(f"eval {split_name}: greedy={gacc:.3f} sampled={sacc:.3f} n={len(golds)}",
          flush=True)
    return gacc, sacc

def csv_open():
    f = open(CSV_PATH, "w", newline="")
    w = csv.writer(f)
    w.writerow(["phase", "arm", "step", "greedy_acc", "sampled_acc", "n",
                "mean_reward", "mean_loss", "mean_iw", "mean_adv_factor"])
    return f, w

def train_arm(model, arm):
    """One bounded training arm on the student-rollout KL objective.
    Returns (mean_reward_over_steps, mean_loss, mean_iw, mean_adv_factor)."""
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    tot_r, tot_l, tot_iw, tot_af, cnt = 0.0, 0.0, 0.0, 0.0, 0
    order = list(range(len(TRAIN)))
    for step in range(STEPS):
        idx = order[(step * BATCH_PROMPTS) % len(order):
                    (step * BATCH_PROMPTS) % len(order) + BATCH_PROMPTS]
        if len(idx) < BATCH_PROMPTS:  # wrap-around, keep batch full
            idx = idx + order[:BATCH_PROMPTS - len(idx)]
        batch = [TRAIN[i] for i in idx]
        prompts = [full_prompt(p["q"]) for p in batch]
        golds = [p["gold"] for p in batch]
        model.eval()
        rollouts = batch_generate(model, prompts, do_sample=True, n_return=K)
        model.train()
        # rewards / advantages
        rewards = [reward_of(rollouts[i * K + r], golds[i])
                   for i in range(BATCH_PROMPTS) for r in range(K)]
        advs, iw_list, af_list = [], [], []
        seq_prompts, seq_conts = [], []
        for i in range(BATCH_PROMPTS):
            base_r = sum(rewards[i * K:(i + 1) * K]) / K
            for r in range(K):
                advs.append(rewards[i * K + r] - base_r)
                seq_prompts.append(prompts[i])
                seq_conts.append(rollouts[i * K + r])
        # tokenize prompt+continuation (right padding) for KL forward
        tok.padding_side = "right"
        full = [p + c for p, c in zip(seq_prompts, seq_conts)]
        enc_p = tok(seq_prompts, add_special_tokens=False)["input_ids"]
        enc_f = tok(full, return_tensors="pt", padding=True,
                    add_special_tokens=False).to("cuda:0")
        input_ids, attn = enc_f["input_ids"], enc_f["attention_mask"]
        cont_lens = [len(tok(c, add_special_tokens=False)["input_ids"])
                     for c in seq_conts]
        keep = [cl > 0 for cl in cont_lens]
        if not any(keep):
            print(f"step {step} arm {arm}: no continuations, skipped",
                  flush=True)
            continue
        input_ids, attn = input_ids[keep], attn[keep]
        enc_p = [e for e, k in zip(enc_p, keep) if k]
        cont_lens = [c for c, k in zip(cont_lens, keep) if k]
        advs = [a for a, k in zip(advs, keep) if k]
        rewards = [r for r, k in zip(rewards, keep) if k]
        # Micro-batched KL (attempt-1 OOM fix): process 4 sequences at a
        # time; teacher forward under no_grad + disabled adapter, student
        # forward with grad; per-chunk backward scaled by 1/N so the
        # accumulated gradient equals the full-batch mean loss. This
        # bounds peak memory to one chunk's full-vocab logits instead of
        # all 16 sequences' float32 logits at once.
        n_seq = len(cont_lens)
        iws, afs = [], []
        opt.zero_grad()
        loss_sum = 0.0
        for cs in range(0, n_seq, 4):
            ce = min(cs + 4, n_seq)
            c_ids, c_attn = input_ids[cs:ce], attn[cs:ce]
            with torch.no_grad():
                with model.disable_adapter():
                    t_logits = model(input_ids=c_ids,
                                     attention_mask=c_attn).logits
            s_logits = model(input_ids=c_ids, attention_mask=c_attn).logits
            chunk_losses = []
            for jj in range(ce - cs):
                j = cs + jj
                pl, cl = len(enc_p[j]), cont_lens[j]
                # positions predicting continuation tokens pl..pl+cl-1
                # are logits pl-1..pl+cl-2
                tl = t_logits[jj, pl - 1:pl + cl - 1].float()
                sl = s_logits[jj, pl - 1:pl + cl - 1].float()
                tgt = c_ids[jj, pl:pl + cl]
                logp_t = torch.log_softmax(tl, dim=-1)
                logp_s = torch.log_softmax(sl, dim=-1)
                p_t = logp_t.exp()
                kl_tok = (p_t * (logp_t - logp_s)).sum(dim=-1)  # [cl]
                kl_seq = kl_tok.mean()
                s_seq = logp_s.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()
                t_seq = logp_t.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()
                iw = torch.exp(t_seq - s_seq.detach()).clamp(IW_LO, IW_HI)
                af = min(max(1.0 + advs[j], 0.0), 2.0)
                iws.append(float(iw)); afs.append(af)
                if arm == "aopd":
                    chunk_losses.append(af * iw * kl_seq)
                else:
                    chunk_losses.append(kl_seq)
            chunk_loss = torch.stack(chunk_losses).sum() / n_seq
            loss_sum += float(chunk_loss)
            chunk_loss.backward()
            del t_logits, s_logits
            torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        loss = torch.tensor(loss_sum)
        mr = sum(rewards) / len(rewards)
        ml, mi, ma = float(loss), sum(iws) / len(iws), sum(afs) / len(afs)
        tot_r += mr; tot_l += ml; tot_iw += mi; tot_af += ma; cnt += 1
        print(f"step {step} arm {arm}: loss={ml:.4f} mean_reward={mr:.3f} "
              f"mean_iw={mi:.3f} mean_adv_factor={ma:.3f}", flush=True)
        train_writer.writerow(["train", arm, step, "", "", "",
                               f"{mr:.4f}", f"{ml:.4f}", f"{mi:.4f}",
                               f"{ma:.4f}"])
        train_csv.flush()
    return (tot_r / max(cnt, 1), tot_l / max(cnt, 1),
            tot_iw / max(cnt, 1), tot_af / max(cnt, 1))

# ---- Phase 3: run ---------------------------------------------------------------
print(f"model={MODEL_ID} steps/arm={STEPS} batch_prompts={BATCH_PROMPTS} "
      f"k={K} lr={LR} iw_clip=[{IW_LO},{IW_HI}]", flush=True)
train_csv, train_writer = csv_open()

base = load_base()
print("loaded base; BEFORE eval", flush=True)
g0, s0 = evaluate(base, "before")
train_writer.writerow(["eval", "before", "", f"{g0:.4f}", f"{s0:.4f}",
                       len(TEST), "", "", "", ""])
train_csv.flush()

# arm (a): OPD baseline (fresh LoRA on the same base instance)
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj",
                                      "o_proj", "gate_proj", "up_proj",
                                      "down_proj"],
                      task_type="CAUSAL_LM")
peft_model = get_peft_model(base, lora_cfg)
peft_model.print_trainable_parameters()
mr_a, ml_a, mi_a, ma_a = train_arm(peft_model, "opd")
g_a, s_a = evaluate(peft_model, "after_opd")
train_writer.writerow(["eval", "opd", "", f"{g_a:.4f}", f"{s_a:.4f}",
                       len(TEST), f"{mr_a:.4f}", f"{ml_a:.4f}",
                       f"{mi_a:.4f}", f"{ma_a:.4f}"])
train_csv.flush()
del peft_model, base
gc.collect(); torch.cuda.empty_cache()

# arm (b): AOPD (fresh base + fresh LoRA, identical data/seed/order)
base = load_base()
peft_model = get_peft_model(base, lora_cfg)
mr_b, ml_b, mi_b, ma_b = train_arm(peft_model, "aopd")
g_b, s_b = evaluate(peft_model, "after_aopd")
train_writer.writerow(["eval", "aopd", "", f"{g_b:.4f}", f"{s_b:.4f}",
                       len(TEST), f"{mr_b:.4f}", f"{ml_b:.4f}",
                       f"{mi_b:.4f}", f"{ma_b:.4f}"])
train_csv.flush()
train_csv.close()
print(f"SUMMARY before greedy={g0:.3f} sampled={s0:.3f} | "
      f"opd greedy={g_a:.3f} sampled={s_a:.3f} | "
      f"aopd greedy={g_b:.3f} sampled={s_b:.3f}", flush=True)
# Marker LAST: nothing is printed after this line.
print("TRAIN_DONE", flush=True)
