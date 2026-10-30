# WISD (arXiv 2610.38681) — Weakening-Induced On-Policy Self-Distillation.
# Shrunk reproduction on ONE free Colab T4, per the 2026-10-30 brief.
# Model: Qwen/Qwen3-8B ONLY (proven training default; Qwen3.5-9B QLoRA is
# NOT proven end-to-end — 2026-10-30 probe OOM'd at the first backward).
# NF4 QLoRA r=16/alpha=32, bf16, transformers<5.0. ONE weakening arm:
# magnitude pruning (paper's stronger arm, +5.5 avg), tau=1/6,
# group-relative (groups of 128 input columns), per-output-row L2 restore,
# attention + FFN projections only (embeddings/heads/norms untouched).
# Arms share cached trajectories + steps + LR: (1) BASE eval only,
# (2) WISD student: KL to tilted target q_w, alpha=2,
# (3) CONTROL: KL to pi0 itself (alpha -> inf), same cache/steps.
# Data: GSM8K TRAIN questions only for trajectories (NO correctness
# filtering — no privileged feedback); GSM8K TEST, greedy, exact match.
# Loss: reverse KL over a documented top-K truncation (K=64 union set =
# top-64 of q_w per position, sampled token force-included); the frozen
# pair (pi0, pi_w) never sees student logits. Loss masked to response
# tokens only. Driver contract: TRAIN_DONE is the VERY LAST stdout line;
# metrics CSVs to /content. Deviations from the paper (full-param LR 1e-6
# -> LoRA LR 5e-5; 3200 prompts x 4 -> 350 x 2; top-K truncation) are
# recorded in the session README and in /content/wisd_summary.json.
import os, sys, json, csv, math, random, re, subprocess, time, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.46,<5.0", "bitsandbytes", "accelerate",
                "peft", "datasets"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, capture_output=True)
print("DEPS_OK", flush=True)

import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model

SEED = 20261030
MODEL = "Qwen/Qwen3-8B"
ALPHA = 2.0
TAU = 1.0 / 6.0
GROUP = 128
TOPK = 64
N_PROMPTS = 350
N_RESP = 2
GEN_TEMP, GEN_TOP_P, GEN_MAX_NEW = 0.8, 0.95, 192
EVAL_N = 200
EVAL_MAX_NEW = 192
LR = 5e-5
R, LORA_ALPHA = 16, 32
MICRO_BATCH, GRAD_ACCUM = 1, 8
GEN_BATCH, EVAL_BATCH = 8, 8
MAX_SEQ = 384
DEVICE = "cuda"
CSV_PATH = "/content/wisd_metrics.csv"
LOSS_PATH = "/content/wisd_train_loss.csv"
SUMMARY_PATH = "/content/wisd_summary.json"
CACHE_PATH = "/content/wisd_cache.pt"

assert torch.cuda.is_available(), "no GPU: refusing to fake a GPU run"
print(f"GPU {torch.cuda.get_device_name(0)} "
      f"total={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB "
      f"torch={torch.__version__}", flush=True)
import transformers
print(f"transformers={transformers.__version__}", flush=True)
torch.manual_seed(SEED)
random.seed(SEED)

# ---------------- metrics plumbing (write early, write often) -------------
rows, loss_rows = [], []


def add(arm, metric, value, n, notes=""):
    rows.append({"arm": arm, "metric": metric, "value": value, "n": n,
                 "notes": notes})
    print(f"RESULT {arm} {metric}={value} n={n} {notes}", flush=True)
    write_csv()


def write_csv():
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "metric", "value", "n", "notes"])
        w.writeheader()
        w.writerows(rows)
    with open(LOSS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "step", "loss", "kl"])
        w.writeheader()
        w.writerows(loss_rows)


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


# ---------------- prompt / parsing (identical to CPU check) ---------------
PROMPT_TEMPLATE = ("Solve the following math problem step by step. "
                   "Give the final answer after '####'.\n\nProblem: {q}\n\nSolution:")


def make_prompt(question):
    return PROMPT_TEMPLATE.format(q=question)


def parse_answer(text):
    if text is None:
        return None
    m = None
    if "####" in text:
        tail = text.split("####")[-1].replace("$", "")
        m = re.search(r"(-?\d[\d,]*(?:\.\d+)?)", tail)
    if m is None:
        nums = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", text.replace("$", ""))
        if not nums:
            return None
        return float(nums[-1].replace(",", ""))
    return float(m.group(1).replace(",", ""))


def gold_answer(answer_field):
    """GSM8K answer field ends with '#### <number>'."""
    return parse_answer(answer_field)


# ---------------- data -------------------------------------------------------
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main")
print(f"GSM8K train={len(ds['train'])} test={len(ds['test'])} "
      f"columns={ds['train'].column_names}", flush=True)
assert "question" in ds["train"].column_names and "answer" in ds["train"].column_names
rng = random.Random(SEED)
train_idx = list(range(len(ds["train"])))
rng.shuffle(train_idx)
prompt_rows = [ds["train"][i] for i in train_idx[:N_PROMPTS]]
rng2 = random.Random(SEED + 1)
test_idx = list(range(len(ds["test"])))
rng2.shuffle(test_idx)
eval_rows = [ds["test"][i] for i in test_idx[:EVAL_N]]
print(f"PROMPTS n={len(prompt_rows)} EVAL n={len(eval_rows)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"tokenizer vocab={len(tok)} pad={tok.pad_token!r} eos_id={tok.eos_token_id}",
      flush=True)


def bnb_config(double_quant):
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=double_quant,
                              bnb_4bit_compute_dtype=torch.bfloat16)


def load_base(double_quant=True):
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb_config(double_quant), device_map="auto")
    m.config.use_cache = False
    return m


@torch.no_grad()
def generate_batch(model, prompts, max_new, do_sample, temp=None, top_p=None):
    """Returns list of (continuation_ids, continuation_text)."""
    model.eval()
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), GEN_BATCH if do_sample else EVAL_BATCH):
        bs = GEN_BATCH if do_sample else EVAL_BATCH
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False)
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)
        plen = input_ids.shape[1]
        kwargs = dict(input_ids=input_ids, attention_mask=attn,
                      do_sample=do_sample, max_new_tokens=max_new,
                      pad_token_id=tok.pad_token_id,
                      eos_token_id=tok.eos_token_id)
        if do_sample:
            kwargs.update(temperature=temp, top_p=top_p)
        gen = model.generate(**kwargs)
        for j in range(len(chunk)):
            new_ids = gen[j, plen:].tolist()
            # cut at first eos (inclusive), drop pads
            cut = []
            for t in new_ids:
                if t == tok.pad_token_id:
                    break
                cut.append(t)
                if t == tok.eos_token_id:
                    break
            outs.append((cut, tok.decode(cut, skip_special_tokens=True)))
        del enc, input_ids, attn, gen
        torch.cuda.empty_cache()
    tok.padding_side = "right"
    return outs


def eval_greedy(model, tag):
    """Greedy eval on held-out GSM8K test. Returns (metrics, predictions)."""
    prompts = [make_prompt(r["question"]) for r in eval_rows]
    outs = generate_batch(model, prompts, EVAL_MAX_NEW, do_sample=False)
    preds, correct, has_hash, ntok = [], 0, 0, []
    for r, (ids, text) in zip(eval_rows, outs):
        pred = parse_answer(text)
        gold = gold_answer(r["answer"])
        ok = pred is not None and gold is not None and abs(pred - gold) < 1e-6
        correct += int(ok)
        has_hash += int("####" in text)
        ntok.append(len(ids))
        preds.append({"question": r["question"], "gold": gold, "pred": pred,
                      "correct": bool(ok), "text": text[:500]})
    acc = correct / len(eval_rows)
    metrics = {"accuracy": round(acc, 4), "n": len(eval_rows),
               "correct": correct, "hash_rate": round(has_hash / len(eval_rows), 4),
               "mean_gen_tokens": round(float(np.mean(ntok)), 1)}
    print(f"[{tag}] EVAL acc={acc:.4f} ({correct}/{len(eval_rows)}) "
          f"hash_rate={metrics['hash_rate']:.3f} mean_tokens={metrics['mean_gen_tokens']}",
          flush=True)
    return metrics, preds

# ---------------- Phase B: pi0 — trajectories + base eval --------------------
t_phase = time.time()
print("== PHASE B: load pi0 (NF4), generate trajectories, base eval ==", flush=True)
pi0 = load_base(double_quant=True)
print(f"PI0_OK params={sum(p.numel() for p in pi0.parameters()) / 1e9:.2f}B "
      f"VRAM={torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

trajectories = []
prompt_texts = [make_prompt(r["question"]) for r in prompt_rows]
for rep in range(N_RESP):
    outs = generate_batch(pi0, prompt_texts, GEN_MAX_NEW, do_sample=True,
                          temp=GEN_TEMP, top_p=GEN_TOP_P)
    for pi, (ids, text) in enumerate(outs):
        if len(ids) < 5:
            continue  # degenerate empty response only; NEVER filter by correctness
        p_ids = tok(prompt_texts[pi], add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(ids) > MAX_SEQ:
            ids = ids[:MAX_SEQ - len(p_ids)]
        trajectories.append({"prompt_idx": pi, "prompt_ids": p_ids,
                             "resp_ids": ids, "text": text})
    print(f"  rep {rep + 1}/{N_RESP} trajectories so far={len(trajectories)} "
          f"({time.time() - t_phase:.0f}s)", flush=True)
resp_lens = [len(t["resp_ids"]) for t in trajectories]
add("probe", "n_trajectories", len(trajectories), len(trajectories),
    f"prompts={N_PROMPTS} responses_per_prompt={N_RESP} dropped_empty={N_PROMPTS * N_RESP - len(trajectories)}")
add("probe", "mean_response_tokens", round(float(np.mean(resp_lens)), 1),
    len(trajectories), f"min={min(resp_lens)} max={max(resp_lens)}")
add("probe", "peak_mem_gb_after_generation", peak_gb(), 1, "")

base_metrics, base_preds = eval_greedy(pi0, "base")
for k, v in base_metrics.items():
    add("base", f"eval::{k}", v, base_metrics["n"], "greedy, GSM8K test, same prompt as training")
with open("/content/wisd_eval_base.jsonl", "w") as f:
    for p in base_preds:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

# ---------------- Phase C: pi_w — pruning + frozen-pair scoring -------------
print("== PHASE C: load pi_w, prune, score frozen pair ==", flush=True)
t_phase = time.time()
from bitsandbytes.nn import Linear4bit
from bitsandbytes.functional import dequantize_4bit, quantize_4bit

pi_w = load_base(double_quant=False)  # simple quant_state so weights can be re-quantized
print(f"PIW_LOADED VRAM={torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def dequant(mod):
    try:
        return dequantize_4bit(mod.weight.data, mod.weight.quant_state)
    except TypeError:
        return dequantize_4bit(mod.weight.data, quant_state=mod.weight.quant_state)


def prune_module(mod):
    """Dequantize -> group-relative magnitude prune -> row-norm restore ->
    re-quantize NF4 in place. Returns pruned fraction for this module."""
    deq = dequant(mod).float()  # [out, in], on GPU
    out_dim, in_dim = deq.shape
    assert in_dim % GROUP == 0, f"in_dim {in_dim} not divisible by {GROUP}"
    G = deq.reshape(out_dim, in_dim // GROUP, GROUP)
    gmax = G.abs().amax(dim=2, keepdim=True)
    mask = (G.abs() >= TAU * gmax).reshape(out_dim, in_dim)
    frac = float(1.0 - mask.float().mean())
    P = deq * mask
    orig_norm = deq.norm(dim=1, keepdim=True)
    new_norm = P.norm(dim=1, keepdim=True)
    P = P * (orig_norm / new_norm.clamp(min=1e-12))
    res = quantize_4bit(P, blocksize=64, compress_statistics=False,
                        quant_type="nf4", quant_storage=torch.uint8)
    qdata, qstate = res if isinstance(res, tuple) else (res[0], res[1])
    mod.weight.data = qdata
    mod.weight.quant_state = qstate
    del deq, G, mask, P
    return frac


edited, fracs = [], []
for name, mod in pi_w.named_modules():
    if isinstance(mod, Linear4bit) and any(name.endswith(p) for p in PROJ):
        fracs.append(prune_module(mod))
        edited.append(name)
n_layers = pi_w.config.num_hidden_layers
expected = n_layers * len(PROJ)
print(f"PRUNED modules={len(edited)} expected={expected} "
      f"mean_frac={float(np.mean(fracs)):.3f} min={min(fracs):.3f} max={max(fracs):.3f}",
      flush=True)
assert len(edited) == expected, f"edited {len(edited)} != expected {expected} projection modules"
add("probe", "pruned_modules", len(edited), 1, f"expected={expected} layers={n_layers}")
add("probe", "pruned_fraction_mean", round(float(np.mean(fracs)), 4), len(edited),
    f"tau={TAU} group={GROUP} min={min(fracs):.3f} max={max(fracs):.3f}")
torch.cuda.empty_cache()

# verify: re-dequantize 3 edited modules — zeros present, row norms restored
verify_errs, verify_fracs = [], []
for name, mod in pi_w.named_modules():
    if name in edited[:3]:
        deq = dequant(mod).float()
        # row-norm check needs the ORIGINAL norms: recompute from pi0 twin
        twin = dict(pi0.named_modules())[name]
        orig = dequant(twin).float()
        rel = ((deq.norm(dim=1) - orig.norm(dim=1)).abs()
               / orig.norm(dim=1).clamp(min=1e-12))
        verify_errs.append(float(rel.max()))
        verify_fracs.append(float((deq.abs() < 1e-6).float().mean()))
        del deq, orig
print(f"PRUNE_VERIFY max_row_norm_rel_err={max(verify_errs):.4f} "
      f"zero_frac={np.mean(verify_fracs):.3f} (3 modules)", flush=True)
assert max(verify_errs) < 0.05, f"row-norm restore failed: {verify_errs}"
assert np.mean(verify_fracs) > 0.05, "pruning produced almost no zeros"
add("probe", "prune_verify_max_row_norm_rel_err", round(max(verify_errs), 4), 3, "")
add("probe", "peak_mem_gb_after_prune", peak_gb(), 1, "")

# Scoring: per trajectory, full-vocab frozen log-probs -> top-K tilted cache.
# log q_w = 1.5*log pi0 - 0.5*log pi_w - logZ (alpha=2). Cache per position:
# top-64 indices of q_w (sampled token force-included), normalised log q_w,
# and raw log pi0 at the same indices (control target). Chunked over
# positions (16) so no full-vocab float tensor over the whole sequence.
@torch.no_grad()
def frozen_resp_logprobs(model, input_ids):
    """Full-vocab log-probs for response positions only, [T, V] float32."""
    out = model(input_ids=input_ids).logits  # [1, L, V] bf16
    return out


cache = {"indices": [], "logq": [], "logp0": [], "prompt_ids": [],
         "resp_ids": []}
kl_full_sum, kl_full_n, topk_mass_sum, topk_mass_n = 0.0, 0, 0.0, 0
pi0.eval()
pi_w.eval()
CH = 16
for ti, traj in enumerate(trajectories):
    p_ids, r_ids = traj["prompt_ids"], traj["resp_ids"]
    ids = torch.tensor([p_ids + r_ids], device=DEVICE)
    T = len(r_ids)
    pos = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + T, device=DEVICE)
    lp0_full = frozen_resp_logprobs(pi0, ids)[0, pos, :].float()
    lp0_full = torch.log_softmax(lp0_full, dim=-1)
    lpw_full = frozen_resp_logprobs(pi_w, ids)[0, pos, :].float()
    lpw_full = torch.log_softmax(lpw_full, dim=-1)
    idx_list, lq_list, lp_list = [], [], []
    for s in range(0, T, CH):
        e = min(T, s + CH)
        raw = 1.5 * lp0_full[s:e] - 0.5 * lpw_full[s:e]
        lq = torch.log_softmax(raw, dim=-1)
        kl_full_sum += float((torch.exp(lp0_full[s:e]) * (lp0_full[s:e] - lq)).sum())
        kl_full_n += (e - s)
        top_vals, top_idx = torch.topk(lq, k=TOPK, dim=-1)
        topk_mass_sum += float(torch.exp(top_vals).sum())
        topk_mass_n += (e - s)
        # force-include the sampled token in the kept set
        sampled = torch.tensor(r_ids[s:e], device=DEVICE)
        for j in range(e - s):
            if sampled[j].item() not in top_idx[j].tolist():
                worst = int(torch.argmin(top_vals[j]).item())
                top_idx[j, worst] = sampled[j]
                top_vals[j, worst] = lq[j, sampled[j]]
        idx_list.append(top_idx.cpu())
        lq_list.append(torch.log_softmax(
            torch.gather(lq, 1, top_idx), dim=-1).cpu())
        lp_list.append(torch.gather(lp0_full[s:e], 1, top_idx).cpu())
    cache["indices"].append(torch.cat(idx_list, dim=0).to(torch.int32))
    cache["logq"].append(torch.cat(lq_list, dim=0).to(torch.float16))
    cache["logp0"].append(torch.cat(lp_list, dim=0).to(torch.float16))
    cache["prompt_ids"].append(p_ids)
    cache["resp_ids"].append(r_ids)
    del ids, lp0_full, lpw_full
    torch.cuda.empty_cache()
    if (ti + 1) % 100 == 0:
        print(f"  scored {ti + 1}/{len(trajectories)} ({time.time() - t_phase:.0f}s)",
              flush=True)
mean_kl_full = kl_full_sum / max(1, kl_full_n)
# topk_mass_sum accumulated the top-K mass per position; mean over positions:
mean_topk_mass = topk_mass_sum / max(1, topk_mass_n)
print(f"SCORE_DONE mean_KL(pi0||q_w)_full={mean_kl_full:.4f} "
      f"mean_top{TOPK}_mass={mean_topk_mass:.4f} ({time.time() - t_phase:.0f}s)", flush=True)
add("probe", "mean_KL_pi0_to_qw_full_vocab", round(mean_kl_full, 4),
    kl_full_n, "per response position, full vocab, alpha=2")
add("probe", f"mean_top{TOPK}_mass_qw", round(float(mean_topk_mass), 4),
    kl_full_n, "mass of cached top-K set under q_w per position")
add("probe", "peak_mem_gb_after_scoring", peak_gb(), 1, "")
torch.save(cache, CACHE_PATH)
print(f"CACHE_SAVED {CACHE_PATH} trajectories={len(cache['indices'])}", flush=True)
del pi0, pi_w
gc.collect()
torch.cuda.empty_cache()

# ---------------- training: one student per arm, same cache/steps/LR --------
order = list(range(len(trajectories)))
random.Random(SEED + 7).shuffle(order)


def train_student(arm, target):
    """target='wisd' -> KL to tilted q_w; 'control' -> KL to pi0.
    Returns (eval_metrics, predictions)."""
    t0 = time.time()
    print(f"== TRAIN {arm} (target={target}) ==", flush=True)
    base = load_base(double_quant=True)
    lcfg = LoraConfig(r=R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"TRAINABLE {n_trainable} / {n_total}", flush=True)
    model.train()
    # No gradient checkpointing: batch 1 / seq <= 384 activations fit
    # without it, and checkpointing + PEFT needs enable_input_require_grads
    # to keep gradients flowing to the adapters.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.0)
    n_opt_steps = (len(order) + GRAD_ACCUM - 1) // GRAD_ACCUM
    sched = get_cosine_schedule_with_warmup(opt, 10, n_opt_steps)
    step, micro, losses = 0, 0, []
    opt.zero_grad(set_to_none=True)
    for start in range(0, len(order), GRAD_ACCUM):
        chunk_idx = order[start:start + GRAD_ACCUM]
        for ti in chunk_idx:
            p_ids = cache["prompt_ids"][ti]
            r_ids = cache["resp_ids"][ti]
            ids = torch.tensor([p_ids + r_ids], device=DEVICE)
            attn = torch.ones_like(ids)
            logits = model(input_ids=ids, attention_mask=attn).logits
            T = len(r_ids)
            pos = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + T, device=DEVICE)
            idx = cache["indices"][ti].to(DEVICE)
            if target == "wisd":
                tgt = cache["logq"][ti].to(DEVICE).float()
            else:
                tgt = cache["logp0"][ti].to(DEVICE).float()
            tgt = tgt - torch.logsumexp(tgt, dim=-1, keepdim=True)
            kl_sum = None
            for s in range(0, T, 32):
                e = min(T, s + 32)
                lg = torch.log_softmax(logits[0, pos[s:e], :].float(), dim=-1)
                lp_k = torch.gather(lg, 1, idx[s:e])
                lp_k = lp_k - torch.logsumexp(lp_k, dim=-1, keepdim=True)
                kl_c = (torch.exp(lp_k) * (lp_k - tgt[s:e])).sum(dim=-1).mean()
                kl_sum = kl_c * (e - s) if kl_sum is None else kl_sum + kl_c * (e - s)
            loss = ALPHA * kl_sum / T / GRAD_ACCUM
            loss.backward()
            losses.append(float(kl_sum.item() / T))
            del ids, attn, logits
            micro += 1
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0 or step == n_opt_steps:
            print(f"  [{arm}] step {step}/{n_opt_steps} "
                  f"mean_KL_last={float(np.mean(losses[-20:])):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    for j, kl in enumerate(losses):
        loss_rows.append({"arm": arm, "step": j + 1, "loss": round(ALPHA * kl, 4),
                          "kl": round(kl, 4)})
    write_csv()
    print(f"[{arm}] TRAINED steps={step} mean_KL={float(np.mean(losses)):.4f} "
          f"peak={peak_gb()} GiB ({time.time() - t0:.0f}s)", flush=True)
    add(arm, "train_mean_KL", round(float(np.mean(losses)), 4), len(losses),
        f"reverse KL over top-{TOPK} set, alpha={ALPHA}, lr={LR}, steps={step}")
    add(arm, "trainable_params", n_trainable, 1, f"total_params={n_total}")
    add(arm, "peak_mem_gb_after_train", peak_gb(), 1, "")
    model.eval()
    metrics, preds = eval_greedy(model, arm)
    for k, v in metrics.items():
        add(arm, f"eval::{k}", v, metrics["n"],
            "greedy, GSM8K test, same prompt as base eval")
    with open(f"/content/wisd_eval_{arm}.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    model.save_pretrained(f"/content/adapter_{arm}")
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


wisd_metrics = train_student("wisd", "wisd")
control_metrics = train_student("control", "control")

# ---------------- summary + done marker (LAST line) -------------------------
summary = {
    "paper": "2610.38681",
    "title": "Weaken to Strengthen: Weakening-Induced On-Policy Self-Distillation for Large Reasoning Models",
    "model": MODEL,
    "model_note": "Qwen3-8B only: proven training default. Qwen3.5-9B QLoRA NOT proven end-to-end (2026-10-30 probe OOM at first backward); not retried here.",
    "method": "WISD: frozen pi0 + pruned pi_w, tilted target alpha=2, reverse KL, LoRA student from theta0",
    "weakening": {"arm": "magnitude pruning only", "tau": TAU, "group": GROUP,
                  "scope": "attention + FFN projections (incl. MoE if present; Qwen3-8B is dense)",
                  "preserved": "embeddings, output head, norms, routing",
                  "pruned_fraction_mean": round(float(np.mean(fracs)), 4)},
    "hyperparams": {"alpha": ALPHA, "lr": LR, "lr_note": "paper LR 1e-6 is full-parameter; LoRA adapter LR 5e-5 used for both students",
                    "r": R, "lora_alpha": LORA_ALPHA, "epochs": 1,
                    "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
                    "effective_batch": MICRO_BATCH * GRAD_ACCUM,
                    "topk": TOPK, "max_seq": MAX_SEQ},
    "data": {"train_prompts": N_PROMPTS, "responses_per_prompt": N_RESP,
             "n_trajectories": len(trajectories),
             "generation": {"temperature": GEN_TEMP, "top_p": GEN_TOP_P,
                            "max_new_tokens": GEN_MAX_NEW,
                            "correctness_filter": "none (no privileged feedback)"},
             "eval": {"split": "GSM8K test", "n": EVAL_N, "decoding": "greedy",
                      "max_new_tokens": EVAL_MAX_NEW,
                      "parse": "number after last ####, fallback last number"}},
    "truncation": f"top-{TOPK} of q_w per position, sampled token force-included; KL over renormalised restricted distributions for student and target",
    "probe": {"mean_KL_pi0_to_qw_full_vocab": round(mean_kl_full, 4),
              f"mean_top{TOPK}_mass_qw": round(float(mean_topk_mass), 4),
              "base_eval": base_metrics},
    "arms": {"wisd": wisd_metrics, "control": control_metrics},
    "accuracy_delta_wisd_minus_base_pp": round(
        (wisd_metrics["accuracy"] - base_metrics["accuracy"]) * 100, 2),
    "accuracy_delta_control_minus_base_pp": round(
        (control_metrics["accuracy"] - base_metrics["accuracy"]) * 100, 2),
    "accuracy_delta_wisd_minus_control_pp": round(
        (wisd_metrics["accuracy"] - control_metrics["accuracy"]) * 100, 2),
    "peak_mem_gb": peak_gb(),
    "seed": SEED,
    "transformers": transformers.__version__, "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=1)
write_csv()
print(json.dumps(summary, indent=1), flush=True)
print("TRAIN_DONE", flush=True)
