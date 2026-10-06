#!/usr/bin/env python3
"""ARBOR vs LoRA on Colab T4 — arXiv 2610.06765.

Compares, on Qwen3-4B in NF4 (8B SIGKILLed 5/5 at load on 12GB RAM), a fixed LoRA r=16 update
against an ARBOR-style conditional adapter: a shared basis of R rank-one
atoms per target linear, routed per question by the paper's additive gate
(question repr + specialty tag + interaction).

Synthetic 3-specialty task (disjoint input->output mappings, verifiable by
exact match): ARITH (a+b), REVERSE (string), CAESAR (shift cipher).
Bounded: 450 train / 150 test, 100 steps/arm, batch 4.

Self-contained: pip installs at top (incl. torchao uninstall per the
documented Colab quirk). Writes /content/arbor_results.json and
/content/arbor_metrics.csv. No secrets.
"""
import subprocess, sys

# --- stage 0: deps (idempotent) ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers<5.0", "peft", "bitsandbytes", "accelerate"],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               capture_output=True)

import csv, json, math, os, random, time, gc, argparse

import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

OUT_JSON = "/content/arbor_4b_results.json"
OUT_CSV = "/content/arbor_4b_metrics.csv"
MODEL_ID = "Qwen/Qwen3-4B"
SEED = 7
N_TRAIN, N_TEST = 150, 50
STEPS, BATCH = 100, 4
LR = 2e-4
TARGETS = ["q_proj", "v_proj"]
LORA_R, LORA_ALPHA = 16, 32
ARBOR_ATOMS = 32  # shared rank-one atoms per target linear

random.seed(SEED)

SPECIALTIES = ["ARITH", "REVERSE", "CAESAR"]


def gen_data(n, seed):
    rng = random.Random(seed)
    data = []
    for spec in SPECIALTIES:
        for _ in range(n):
            if spec == "ARITH":
                a, b = rng.randint(10, 99), rng.randint(10, 99)
                x, y = f"{a}+{b}", str(a + b)
            elif spec == "REVERSE":
                s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(5, 8)))
                x, y = f"reverse {s}", s[::-1]
            else:
                sh = rng.randint(1, 3)
                s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(5, 8)))
                enc = "".join(chr((ord(c) - 97 + sh) % 26 + 97) for c in s)
                x, y = f"caesar(+{sh}) {s}", enc
            prompt = f"Task: {spec}\nInput: {x}\nOutput:"
            data.append((spec, prompt, y))
    rng.shuffle(data)
    return data


def load_base(tok):
    # The Colab T4 VM has only ~12GB system RAM; from_pretrained can peak
    # above that while materializing shards. Drop page cache first (we run
    # as root on the VM) to maximize headroom. Best effort.
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass
    gc.collect()
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
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
    prompts = [p for _, p, _ in batch]
    fulls = [p + " " + y for _, p, y in batch]
    enc = tok(fulls, padding=True, truncation=True, max_length=96,
              return_tensors="pt")
    plens = [len(tok(p)["input_ids"]) for _, p, _ in batch]
    tags = [SPECIALTIES.index(s) for s, _, _ in batch]
    labels = enc["input_ids"].clone()
    for i, pl in enumerate(plens):
        labels[i, :pl] = -100
    labels[enc["attention_mask"] == 0] = -100
    return enc, labels, plens, tags


def train_loss(model, tok, batch):
    enc, labels, plens, tags = encode_batch(tok, batch)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    labels = labels.to(model.device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits.float()
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1), ignore_index=-100)
    return loss


@torch.no_grad()
def evaluate(model, tok, data, arbor=None):
    model.eval()
    per, tot = {s: [0, 0] for s in SPECIALTIES}, [0, 0]
    for spec, prompt, y in data:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
        if arbor is not None:
            arbor.set_scores_for(model, tok, [(spec, prompt, y)])
        gen = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
        ok = text.split()[0] == y if text else False
        per[spec][1] += 1
        tot[1] += 1
        if ok:
            per[spec][0] += 1
            tot[0] += 1
    model.train()
    return {s: v[0] / v[1] for s, v in per.items()}, tot[0] / tot[1]


# ---------------- ARBOR adapter ----------------
class ArborAdapter:
    """Shared rank-one basis + additive gate, patched onto target linears."""

    def __init__(self, model, tok, targets, n_atoms, n_tags, d_model):
        self.n_atoms = n_atoms
        self.scores = None  # (B, R) set per forward
        self.layers = []
        self.dev = next(model.parameters()).device
        for name, mod in model.named_modules():
            if any(name.endswith("." + t) or name == t or t in name.split(".")
                   for t in targets) and isinstance(mod, nn.Linear):
                d_out, d_in = mod.out_features, mod.in_features  # NB: NOT mod.weight.shape (Params4bit storage shape!)
                U = nn.Parameter(torch.zeros(d_out, n_atoms, device=self.dev, dtype=torch.bfloat16))
                V = nn.Parameter((torch.randn(n_atoms, d_in, device=self.dev) * 0.02).to(torch.bfloat16))
                U.requires_grad_(True)
                V.requires_grad_(True)
                mod._arbor_U, mod._arbor_V = U, V
                orig = mod.forward
                ad = self

                def fwd(x, _orig=orig, _mod=mod, _ad=ad):
                    out = _orig(x)
                    s = _ad.scores
                    if s is None:
                        return out
                    v = x @ _mod._arbor_V.t()                      # (B,S,R)
                    v = v * s.unsqueeze(1)                        # gate
                    return out + v @ _mod._arbor_U.t()            # (B,S,d)
                mod.forward = fwd
                self.layers.append(mod)
        # gate: Wq h + Wt t + Wx vec(h (x) t)
        d_q = d_model
        self.Wq = nn.Parameter((torch.randn(n_atoms, d_q, device=self.dev) * 0.02).to(torch.bfloat16))
        self.Wt = nn.Parameter(torch.zeros(n_atoms, n_tags, device=self.dev, dtype=torch.bfloat16))
        self.Wx = nn.Parameter((torch.randn(n_atoms, d_q * n_tags, device=self.dev) * 0.01).to(torch.bfloat16))
        for p in (self.Wq, self.Wt, self.Wx):
            p.requires_grad_(True)

    def parameters(self):
        ps = [self.Wq, self.Wt, self.Wx]
        for m in self.layers:
            ps += [m._arbor_U, m._arbor_V]
        return ps

    @torch.no_grad()
    def question_repr(self, model, tok, prompts, plens):
        ids = tok(prompts, padding=True, return_tensors="pt")["input_ids"]
        ids = ids.to(model.device)
        emb = model.model.embed_tokens(ids).to(torch.bfloat16)  # (B,S,d)
        hs = []
        for i, pl in enumerate(plens):
            hs.append(emb[i, :pl].mean(0))
        return torch.stack(hs)  # (B,d)

    def set_scores_for(self, model, tok, batch):
        prompts = [p for _, p, _ in batch]
        plens = [len(tok(p)["input_ids"]) for p in prompts]
        h = self.question_repr(model, tok, prompts, plens)
        t = torch.zeros(len(batch), len(SPECIALTIES), dtype=torch.bfloat16)
        for i, (s, _, _) in enumerate(batch):
            t[i, SPECIALTIES.index(s)] = 1.0
        t = t.to(h.device)
        inter = (h.unsqueeze(2) * t.unsqueeze(1)).reshape(len(batch), -1)
        logits = h @ self.Wq.t() + t @ self.Wt.t() + inter @ self.Wx.t()
        self.scores = torch.softmax(logits, dim=-1).to(torch.bfloat16)

    def clear(self):
        self.scores = None


def run_arm(arm, train_data, test_data):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = load_base(tok)
    arbor = None
    if arm == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                         target_modules=TARGETS, bias="none",
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        d_model = model.config.hidden_size
        arbor = ArborAdapter(model, tok, TARGETS, ARBOR_ATOMS,
                             len(SPECIALTIES), d_model)
        opt = torch.optim.AdamW(arbor.parameters(), lr=LR)
        n_params = sum(p.numel() for p in arbor.parameters())
    print(f"[{arm}] trainable params: {n_params}", flush=True)

    order = list(range(len(train_data)))
    losses = []
    t0 = time.time()
    for step in range(STEPS):
        if step % (len(train_data) // BATCH) == 0:
            random.Random(SEED).shuffle(order)
        batch = [train_data[order[(step * BATCH + j) % len(train_data)]]
                 for j in range(BATCH)]
        if arbor is not None:
            arbor.set_scores_for(model, tok, batch)
        opt.zero_grad()
        loss = train_loss(model, tok, batch)
        loss.backward()
        opt.step()
        if arbor is not None:
            arbor.clear()
        losses.append(float(loss.detach()))
        if (step + 1) % 20 == 0:
            print(f"[{arm}] step {step+1}/{STEPS} loss {float(loss.detach()):.4f} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
    per, overall = evaluate(model, tok, test_data, arbor)
    print(f"[{arm}] test EM per-specialty: {per} overall: {overall:.3f}",
          flush=True)
    result = {"per_specialty": per, "overall": overall,
              "final_loss": losses[-1], "trainable_params": n_params,
              "seconds": time.time() - t0}
    # release the model before the next arm runs in a fresh process
    del model, tok, opt
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["lora", "arbor", "both"], default="both")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    t_start = time.time()
    train_data = gen_data(N_TRAIN, SEED)
    test_data = gen_data(N_TEST, SEED + 1)
    results = {"model": MODEL_ID, "seed": SEED, "n_train": len(train_data),
               "n_test": len(test_data), "steps": STEPS, "batch": BATCH,
               "lr": LR, "specialties": SPECIALTIES}
    arms = ("lora", "arbor") if args.arm == "both" else (args.arm,)
    for arm in arms:
        r = run_arm(arm, train_data, test_data)
        results[arm] = r
        del r
        gc.collect()
        torch.cuda.empty_cache()
    results["total_seconds"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    if args.arm == "both":
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "specialty", "em_accuracy"])
            for arm in ("lora", "arbor"):
                for s, v in results[arm]["per_specialty"].items():
                    w.writerow([arm, s, f"{v:.4f}"])
                w.writerow([arm, "OVERALL", f"{results[arm]['overall']:.4f}"])
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
