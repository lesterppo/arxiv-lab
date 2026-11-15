#!/usr/bin/env python3
"""PECA -- Parameter-Efficient Conciseness Adaptation (arXiv:2611.17128).

Paper claim: adding a single differentiable conciseness term to the QLoRA
SFT objective cuts mean response length by 72.8% while improving
multiple-choice diagnostic accuracy (40.8% -> 44.0%) on a 1.7B model
fine-tuned on medical-o1-reasoning-SFT. The term raises the EOS
probability once generation exceeds a token budget (token count itself
is discrete, so EOS logprob is the smooth surrogate).

Bounded test of that claim on a current-generation model:
  Arm SFT  -- plain QLoRA SFT (baseline)
  Arm PECA -- same data/epochs/lr + conciseness term:
              mean over response positions j > BUDGET of
              -log p(eos | prefix up to j-1), weight LAMBDA.
  Both arms matched: same 400 train rows, same prompt, 1 epoch,
  lr 2e-4, r=16/alpha=32 NF4 QLoRA, bf16, transformers<5.0.
  Eval: MedQA-USMLE 4-option test subset (n=100), greedy generation,
  accuracy (letter parse) + mean generated tokens. Base model scored
  on the same 100 as reference.

Model rule (2026-10-27, still binding): current-generation families
only, weight updates in the 8-9B class under NF4 QLoRA; shrink the
EXPERIMENT, never the model. Try Qwen3.5-9B first (latest 9B-class
Qwen); on load/OOM failure fall back in-driver to the proven
Qwen3-8B carrier (one automatic fallback, 2026-11-05 rule).
transformers>=4.57 is required: earlier pins do not recognise the
'qwen3_5' architecture (that killed the 2026-11-13 SFT-mixture run).

Operationalization notes (honest limits):
- The paper's abstract does not publish LAMBDA or the token budget;
  LAMBDA=1.0 and BUDGET=96 response tokens are our choices, recorded
  here, not the paper's values.
- The paper fine-tuned a 1.7B model; we test the 8-9B class per the
  standing model rule, so this is a transfer test, not a replication
  of the paper's exact numbers.
- Target = Complex_CoT + "Final answer: " + Response (the corpus's
  reasoning trace + short answer), truncated to 640 tokens.
- Metrics CSV + results JSON are flushed to /content after EVERY
  phase (reclamation-safe). Final stdout line is TRAIN_DONE.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import gc
import json
import random
import re
import subprocess
import sys
import time

METRICS_CSV = "/content/peca_metrics.csv"
RESULTS_JSON = "/content/peca_results.json"
SEED = 123
random.seed(SEED)

N_TRAIN = 400
EPOCHS = 1
LR = 2e-4
MAX_LEN = 640
LAMBDA = 1.0
BUDGET = 96
EVAL_N = 100
EVAL_MAX_NEW = 160
EVAL_BS = 4

MODELS_TRY = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"]

RESULTS = {
    "paper": "2611.17128",
    "title": "Parameter-Efficient Conciseness Adaptation: Differentiable Length Control for Concise Clinical Reasoning in Smaller Language Models",
    "variant": "Bounded transfer test: plain QLoRA SFT vs SFT+PECA conciseness term (lambda=1.0, budget=96 resp tokens), matched 400 rows / 1 epoch / lr 2e-4; MedQA n=100 greedy eval (accuracy + mean gen tokens). Base model as reference.",
    "models_tried": MODELS_TRY,
    "model_used": None,
    "fallback_used": False,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target": "q,k,v,o,gate,up,down"},
    "peca": {"lambda": LAMBDA, "budget_tokens": BUDGET,
             "note": "lambda/budget are our operationalization choices; paper abstract does not publish them"},
    "train": {"n": N_TRAIN, "epochs": EPOCHS, "lr": LR, "max_len": MAX_LEN},
    "eval": {"dataset": "GBaker/MedQA-USMLE-4-options test", "n": EVAL_N,
             "max_new_tokens": EVAL_MAX_NEW, "batch": EVAL_BS},
    "arms": {},
    "phases": [],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flush_results():
    with open(RESULTS_JSON, "w") as f:
        json.dump(RESULTS, f, indent=2)


def record_metric(arm, split, n, correct, mean_tokens, notes=""):
    acc = correct / n if n else 0.0
    exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["arm", "split", "n", "correct", "acc", "mean_gen_tokens", "notes"])
        w.writerow([arm, split, n, correct, f"{acc:.4f}", f"{mean_tokens:.1f}", notes])
    log(f"METRIC {arm} {split}: {correct}/{n} = {acc:.4f} mean_tokens={mean_tokens:.1f} {notes}")
    return acc


def install():
    log("installing deps")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.57,<5.0", "bitsandbytes", "accelerate",
                    "peft", "datasets", "hf_transfer"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False)
    log("deps installed")


def load_data():
    from datasets import load_dataset
    log("loading medical-o1-reasoning-SFT (en) + MedQA test")
    sft = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    sft = sft.shuffle(seed=SEED)
    rows = []
    for r in sft.select(range(N_TRAIN)):
        q = (r["Question"] or "").strip()
        cot = (r["Complex_CoT"] or "").strip()
        resp = (r["Response"] or "").strip()
        if not q or not resp:
            continue
        target = f"{cot}\n\nFinal answer: {resp}" if cot else resp
        rows.append({"prompt": f"Question: {q}\nAnswer:", "completion": " " + target})
    log(f"train rows kept: {len(rows)}")
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    medqa = medqa.shuffle(seed=SEED).select(range(EVAL_N))
    eval_items = []
    for r in medqa:
        opts = r["options"]
        prompt = ("Answer the following medical question. Reply with the letter "
                  "of the correct option, then a brief explanation.\n\n"
                  f"Question: {r['question']}\n"
                  f"A. {opts['A']}\nB. {opts['B']}\nC. {opts['C']}\nD. {opts['D']}\nAnswer:")
        eval_items.append({"prompt": prompt, "gold": r["answer_idx"]})
    log(f"eval items: {len(eval_items)}")
    return rows, eval_items


def build_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    log(f"loading tokenizer + base model {model_id} (NF4)")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    log(f"VRAM after load: {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated, "
        f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB peak")
    return tok, model


def train_arm(tok, model, examples, use_peca, tag):
    """Manual SFT loop with optional PECA conciseness term."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_cosine_schedule_with_warmup

    eos_id = tok.eos_token_id

    class SFTData(Dataset):
        def __init__(self, exs):
            self.rows = []
            for e in exs:
                p = tok(e["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(e["completion"], add_special_tokens=False)["input_ids"] + [eos_id]
                ids = (p + c)[:MAX_LEN]
                labels = ([-100] * len(p) + c)[:MAX_LEN]
                # resp_len = number of completion tokens kept (for budget math)
                resp_len = max(0, len(ids) - len(p))
                self.rows.append((ids, labels, len(p), resp_len))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        mx = max(len(ids) for ids, _, _, _ in batch)
        pad = tok.pad_token_id
        input_ids, labels, attn, prompt_lens, resp_lens = [], [], [], [], []
        for ids, lab, plen, rlen in batch:
            n = mx - len(ids)
            input_ids.append(ids + [pad] * n)
            labels.append(lab + [-100] * n)
            attn.append([1] * len(ids) + [0] * n)
            prompt_lens.append(plen)
            resp_lens.append(rlen)
        return (torch.tensor(input_ids), torch.tensor(labels),
                torch.tensor(attn), prompt_lens, resp_lens)

    torch.cuda.empty_cache()
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    model.train()
    model.config.use_cache = False
    GRAD_ACCUM = 2  # micro-batch 2 x accum 2 = effective batch 4
    dl = DataLoader(SFTData(examples), batch_size=2, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=LR, weight_decay=0.0)
    total = (len(dl) * EPOCHS + GRAD_ACCUM - 1) // GRAD_ACCUM
    sched = get_cosine_schedule_with_warmup(opt, 3, total)
    step = 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for ep in range(EPOCHS):
        for mbi, (input_ids, labels, attn, prompt_lens, resp_lens) in enumerate(dl):
            input_ids, labels, attn = (input_ids.cuda(), labels.cuda(), attn.cuda())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits.float()
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                ce = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1), ignore_index=-100)
                loss = ce
                peca_term = None
                if use_peca:
                    # positions j (0-indexed within response) with j >= BUDGET:
                    # encourage EOS at the prediction slot for that token.
                    terms = []
                    for b in range(input_ids.size(0)):
                        plen = prompt_lens[b]
                        rlen = resp_lens[b]
                        # response token j sits at seq pos plen+j; its
                        # prediction slot is logits[plen+j-1]
                        for j in range(BUDGET, rlen):
                            slot = plen + j - 1
                            if 0 <= slot < shift_logits.size(1):
                                lp = torch.log_softmax(shift_logits[b, slot], dim=-1)
                                terms.append(-lp[eos_id])
                    if terms:
                        peca_term = torch.stack(terms).mean()
                        loss = ce + LAMBDA * peca_term
            (loss / GRAD_ACCUM).backward()
            if (mbi + 1) % GRAD_ACCUM == 0 or mbi == len(dl) - 1:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or step == total:
                    extra = f" peca={peca_term.item():.4f}" if peca_term is not None else ""
                    log(f"{tag} step {step}/{total} loss {loss.item():.4f}{extra} "
                        f"elapsed {time.time()-t0:.0f}s "
                        f"VRAM {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    log(f"{tag} done: {step} steps, {time.time()-t0:.0f}s, "
        f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    return step


LETTER_RE = re.compile(r"Answer:\s*([A-D])\b")
ANY_LETTER_RE = re.compile(r"\b([A-D])\b")


def parse_letter(text):
    m = LETTER_RE.search(text)
    if m:
        return m.group(1)
    m = ANY_LETTER_RE.search(text)
    return m.group(1) if m else None


def run_eval(tok, model, items, tag):
    import torch
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    correct, n_tok, n = 0, 0, 0
    for i in range(0, len(items), EVAL_BS):
        batch = items[i:i + EVAL_BS]
        prompts = [b["prompt"] for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=EVAL_MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for k, b in enumerate(batch):
            gen_ids = gen[k, plen:].tolist()
            # strip trailing pad/eos for length accounting
            while gen_ids and gen_ids[-1] in (tok.pad_token_id, tok.eos_token_id):
                gen_ids.pop()
            text = tok.decode(gen_ids, skip_special_tokens=True)
            pred = parse_letter(text)
            n += 1
            n_tok += len(gen_ids)
            if pred == b["gold"]:
                correct += 1
        if (i // EVAL_BS + 1) % 5 == 0:
            log(f"{tag} eval {n}/{len(items)} so far {correct} correct")
    tok.padding_side = "right"
    mean_tokens = n_tok / n if n else 0.0
    return correct, n, mean_tokens


def run_experiment(model_id, train_rows, eval_items):
    """Full experiment on one model: base eval, SFT arm, PECA arm. Returns dict."""
    import torch
    tok, model = build_model(model_id)
    RESULTS["model_used"] = model_id
    flush_results()

    # base reference eval
    log("== base eval ==")
    c, n, mt = run_eval(tok, model, eval_items, "base")
    acc = record_metric("base", "medqa", n, c, mt, f"model={model_id}")
    RESULTS["arms"]["base"] = {"acc": acc, "mean_gen_tokens": mt, "n": n, "correct": c}
    RESULTS["phases"].append("base eval done")
    flush_results()

    # arm 1: plain SFT
    log("== arm SFT (plain QLoRA) ==")
    train_arm(tok, model, train_rows, use_peca=False, tag="SFT")
    RESULTS["phases"].append("SFT arm trained")
    flush_results()
    c, n, mt = run_eval(tok, model, eval_items, "SFT")
    acc = record_metric("SFT", "medqa", n, c, mt, f"model={model_id}")
    RESULTS["arms"]["SFT"] = {"acc": acc, "mean_gen_tokens": mt, "n": n, "correct": c}
    RESULTS["phases"].append("SFT eval done")
    flush_results()

    # arm 2: PECA -- fresh LoRA on the same base: unload peft, rebuild
    log("== arm PECA (SFT + conciseness term) ==")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    tok2, model2 = build_model(model_id)
    train_arm(tok2, model2, train_rows, use_peca=True, tag="PECA")
    RESULTS["phases"].append("PECA arm trained")
    flush_results()
    c, n, mt = run_eval(tok2, model2, eval_items, "PECA")
    acc = record_metric("PECA", "medqa", n, c, mt, f"model={model_id}")
    RESULTS["arms"]["PECA"] = {"acc": acc, "mean_gen_tokens": mt, "n": n, "correct": c}
    RESULTS["phases"].append("PECA eval done")
    flush_results()
    return RESULTS["arms"]


def main():
    install()
    train_rows, eval_items = load_data()
    last_err = None
    for i, model_id in enumerate(MODELS_TRY):
        try:
            run_experiment(model_id, train_rows, eval_items)
            if i > 0:
                RESULTS["fallback_used"] = True
            break
        except Exception as e:
            last_err = e
            log(f"MODEL {model_id} FAILED: {type(e).__name__}: {e}")
            RESULTS["phases"].append(f"model {model_id} failed: {type(e).__name__}: {e}")
            flush_results()
            import torch
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if i == len(MODELS_TRY) - 1:
                raise
            log("falling back to next model (one automatic fallback)")
    flush_results()
    log("ALL PHASES DONE")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
