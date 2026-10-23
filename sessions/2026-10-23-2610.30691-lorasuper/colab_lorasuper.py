#!/usr/bin/env python3
"""Colab T4 test of arXiv:2610.30691 — LoRA Superposition across Pretrained Models.

Paper claim: LoRAs trained independently for the same task on different
pretrained backbones can be summed and applied unchanged to all contributing
backbones. On 7 ViT-B/16 CLIP backbones x 8 image tasks: exact sum improves
ALL 56 backbone-task pairs (+30.21pp mean over unadapted), within 0.28pp of
solo LoRAs; leave-one-model-out sum gives only +0.06pp; solo LoRAs are
nearly orthogonal in parameter space (frozen backbone = implicit selector).

Bounded replication on one T4: 2 distinct ViT-B/16 CLIP backbones
  (openai/clip-vit-base-patch16, laion/CLIP-ViT-B-16-laion2B-s12B-b42K)
x 1 task (CIFAR-10 4-class subset: 300 train / 100 test per class).
LoRA r=8/alpha=16 on q,k,v, 3 epochs; per-(backbone, adapter) linear probe
(5 epochs, frozen encoder) isolates adapter feature quality.

Arms per backbone: none (unadapted), solo (own LoRA), sum (exact sum of
both LoRAs), lomo (the OTHER backbone's LoRA only).
Metrics -> /content/lorasuper/metrics.csv + summary.json.

Self-contained: pip installs at top, torchao uninstalled after installs
(Colab torchao 0.10 breaks peft).
"""
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"] + list(args),
                   check=True)

pip("transformers<5.0", "peft", "accelerate")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, capture_output=True)

import csv, json, math, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model

OUT = "/content/lorasuper"
os.makedirs(OUT, exist_ok=True)
SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, flush=True)

BACKBONES = {
    "openai": "openai/clip-vit-base-patch16",
    "laion": "laion/CLIP-ViT-B-16-laion2B-s12B-b42K",
}
CLASSES = [0, 1, 2, 3]  # airplane, automobile, bird, cat
N_TRAIN, N_TEST = 300, 100
ADAPT_SUBSTRINGS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")


def make_loader(backbone_id, train, n_per_class, batch):
    proc = CLIPImageProcessor.from_pretrained(backbone_id)
    ds = CIFAR10(root="/content/data", train=train, download=True)
    idx = []
    per = {c: 0 for c in CLASSES}
    rng = random.Random(SEED)
    order = list(range(len(ds)))
    rng.shuffle(order)
    for i in order:
        _, y = ds[i]
        if y in CLASSES and per[y] < n_per_class:
            idx.append(i); per[y] += 1
        if all(per[c] >= n_per_class for c in CLASSES):
            break
    label_map = {c: k for k, c in enumerate(CLASSES)}

    class Wrap(torch.utils.data.Dataset):
        def __len__(self): return len(idx)
        def __getitem__(self, j):
            img, y = ds[idx[j]]
            pix = proc(images=img, return_tensors="pt")["pixel_values"][0]
            return pix, label_map[y]

    return DataLoader(Wrap(), batch_size=batch, shuffle=train,
                      num_workers=2, pin_memory=True)


class Head(nn.Module):
    def __init__(self, d=768, n=4):
        super().__init__()
        self.fc = nn.Linear(d, n)
    def forward(self, x):
        return self.fc(x)


def train_classifier(vision, loader, epochs, lr, lora=False):
    """Train head (+LoRA if lora) on frozen/fine-tuned encoder. Returns acc fn."""
    vision = vision.to(DEVICE)
    head = Head().to(DEVICE)
    params = list(head.parameters())
    if lora:
        params += [p for p in vision.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    vision.train(); head.train()
    for ep in range(epochs):
        tot, nl = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                feat = vision(pixel_values=x).pooler_output
                logits = head(feat)
                loss = nn.functional.cross_entropy(logits, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(x); nl += len(x)
        print(f"  epoch {ep+1}/{epochs} loss={tot/nl:.4f}", flush=True)
    return head


@torch.no_grad()
def evaluate(vision, head, loader):
    vision.eval(); head.eval()
    ok, n = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        feat = vision(pixel_values=x).pooler_output
        ok += (head(feat).argmax(1) == y).sum().item(); n += len(x)
    return ok / n


def base_state(backbone_id):
    m = CLIPVisionModel.from_pretrained(backbone_id,
                                        torch_dtype=torch.float16).to(DEVICE)
    return m, {k: v.detach().clone() for k, v in m.named_parameters()}


def delta_of(peft_model, base_params):
    """Delta matrices per adapted layer: W_merged - W_base."""
    merged = peft_model.merge_and_unload()
    delta = {}
    for k, v in merged.named_parameters():
        if not k.endswith(".weight"):
            continue  # biases untouched by LoRA (bias="none"); their zero
                      # deltas would drag the orthogonality mean toward 0
        for sub in ADAPT_SUBSTRINGS:
            if sub in k and k in base_params:
                delta[k] = (v.detach().float().cpu()
                            - base_params[k].float().cpu())
    del merged
    torch.cuda.empty_cache()
    return delta


def apply_delta(model, delta, scale=1.0):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, d in delta.items():
            params[k].data += (scale * d).to(params[k].device,
                                             dtype=params[k].dtype)
    return model


def main():
    train_loaders, test_loaders = {}, {}
    for name, bid in BACKBONES.items():
        train_loaders[name] = make_loader(bid, True, N_TRAIN, 32)
        test_loaders[name] = make_loader(bid, False, N_TEST, 64)

    # ---- Phase 1: train solo LoRA per backbone, extract deltas ----
    deltas, names = {}, list(BACKBONES)
    for name in names:
        bid = BACKBONES[name]
        print(f"=== solo LoRA train: {name} ===", flush=True)
        model, base_params = base_state(bid)
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj"])
        peft_model = get_peft_model(model, cfg)
        peft_model.print_trainable_parameters()
        train_classifier(peft_model, train_loaders[name], epochs=3,
                         lr=2e-4, lora=True)
        deltas[name] = delta_of(peft_model, base_params)
        print(f"solo delta layers: {len(deltas[name])}", flush=True)
        del peft_model, model, base_params
        torch.cuda.empty_cache()

    # ---- orthogonality: mean cosine sim between the two solo deltas ----
    keys = sorted(set(deltas[names[0]]) & set(deltas[names[1]]))
    cos = [float(torch.nn.functional.cosine_similarity(
        deltas[names[0]][k].flatten(), deltas[names[1]][k].flatten(), dim=0))
        for k in keys]
    mean_cos = float(np.mean(cos))
    print(f"orthogonality: {len(keys)} layers, mean cosine={mean_cos:.4f}",
          flush=True)
    sum_delta = {k: deltas[names[0]][k] + deltas[names[1]][k] for k in keys}

    # ---- Phase 2: per (backbone, arm) linear probe ----
    rows = []
    for name in names:
        bid = BACKBONES[name]
        arms = {
            "none": {},
            "solo": {k: deltas[name][k] for k in keys},
            "sum": sum_delta,
            "lomo": {k: deltas[names[1] if name == names[0] else names[0]][k]
                     for k in keys},
        }
        for arm, delta in arms.items():
            print(f"=== probe: {name}/{arm} ===", flush=True)
            model, _ = base_state(bid)
            apply_delta(model, delta)
            for p in model.parameters():
                p.requires_grad = False
            head = train_classifier(model, train_loaders[name], epochs=5,
                                    lr=1e-3, lora=False)
            acc = evaluate(model, head, test_loaders[name])
            print(f"  test acc={acc:.4f}", flush=True)
            rows.append({"backbone": name, "arm": arm, "test_acc": acc})
            del model, head
            torch.cuda.empty_cache()

    with open(f"{OUT}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "arm", "test_acc"])
        w.writeheader(); w.writerows(rows)

    # ---- summary: paper's headline numbers ----
    s = {"mean_cosine": mean_cos, "pairs": []}
    for name in names:
        a = {r["arm"]: r["test_acc"] for r in rows if r["backbone"] == name}
        s["pairs"].append({
            "backbone": name,
            "gain_sum_over_unadapted_pp": (a["sum"] - a["none"]) * 100,
            "gap_solo_minus_sum_pp": (a["solo"] - a["sum"]) * 100,
            "gain_lomo_over_unadapted_pp": (a["lomo"] - a["none"]) * 100,
            "acc": {k: round(v, 4) for k, v in a.items()},
        })
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
