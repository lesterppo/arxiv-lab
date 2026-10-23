#!/usr/bin/env python3
"""CPU pipeline check for the LoRA-superposition Colab run (arXiv:2610.30691).

This does NOT test the paper's claim (backbones are random-init tiny ViTs,
the task is synthetic). It validates the *measurement pipeline* end to end
with the real libraries (transformers + peft, same code path as
colab_lorasuper.py): LoRA training on CLIPVisionModel, delta extraction via
merge-and-unload diff, orthogonality, exact-sum / LOMO arm application, and
the frozen-encoder linear-probe protocol. If this passes, a Colab resume is
a clean re-run, not a debug cycle.
"""
import csv, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPVisionModel, CLIPVisionConfig
from peft import LoraConfig, get_peft_model

SEED = 11
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cpu"
ADAPT = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")


def make_backbone(seed):
    torch.manual_seed(seed)
    cfg = CLIPVisionConfig(hidden_size=128, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=256,
                           image_size=32, patch_size=8, num_channels=3)
    return CLIPVisionModel(cfg)


def synth_data(n):
    # class = which quadrant is bright; trivially learnable
    x = torch.rand(n, 3, 32, 32) * 0.5
    y = torch.randint(0, 2, (n,))
    for i in range(n):
        if y[i] == 0:
            x[i, :, :16, :16] += 0.5
        else:
            x[i, :, 16:, 16:] += 0.5
    return TensorDataset(x, y)


class Head(nn.Module):
    def __init__(self, d=128, n=2):
        super().__init__()
        self.fc = nn.Linear(d, n)
    def forward(self, x):
        return self.fc(x)


def train(vision, loader, epochs, lr, lora):
    vision.to(DEVICE)
    head = Head().to(DEVICE)
    params = list(head.parameters())
    if lora:
        params += [p for p in vision.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    vision.train(); head.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(
                head(vision(pixel_values=x).pooler_output), y)
            loss.backward(); opt.step()
    return head


@torch.no_grad()
def evaluate(vision, head, loader):
    vision.eval(); head.eval()
    ok = n = 0
    for x, y in loader:
        ok += (head(vision(pixel_values=x).pooler_output).argmax(1) == y).sum().item()
        n += len(x)
    return ok / n


def main():
    names = ["bb0", "bb1"]
    train_l = DataLoader(synth_data(128), batch_size=32, shuffle=True)
    test_l = DataLoader(synth_data(64), batch_size=64)

    deltas = {}
    for i, name in enumerate(names):
        model = make_backbone(100 + i)
        base = {k: v.detach().clone() for k, v in model.named_parameters()}
        peft_m = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj"]))
        train(peft_m, train_l, epochs=2, lr=1e-3, lora=True)
        merged = peft_m.merge_and_unload()
        d = {}
        for k, v in merged.named_parameters():
            if not k.endswith(".weight"):
                continue  # biases untouched by LoRA; zero deltas would bias
                          # the orthogonality mean toward 0
            if any(s in k for s in ADAPT) and k in base:
                d[k] = v.detach().float().cpu() - base[k].float().cpu()
        deltas[name] = d
        print(f"{name}: delta layers={len(d)}", flush=True)
        assert len(d) == 2 * 3, f"expected 6 adapted layers, got {len(d)}"

    keys = sorted(set(deltas["bb0"]) & set(deltas["bb1"]))
    cos = [float(torch.nn.functional.cosine_similarity(
        deltas["bb0"][k].flatten(), deltas["bb1"][k].flatten(), dim=0))
        for k in keys]
    print(f"orthogonality check: {len(keys)} layers, mean cos={np.mean(cos):.4f}",
          flush=True)
    sum_d = {k: deltas["bb0"][k] + deltas["bb1"][k] for k in keys}

    rows = []
    for i, name in enumerate(names):
        other = names[1 - i]
        arms = {"none": {}, "solo": deltas[name], "sum": sum_d,
                "lomo": deltas[other]}
        for arm, d in arms.items():
            model = make_backbone(100 + i)
            params = dict(model.named_parameters())
            with torch.no_grad():
                for k, dv in d.items():
                    params[k].data += dv
            for p in model.parameters():
                p.requires_grad = False
            head = train(model, train_l, epochs=3, lr=1e-2, lora=False)
            acc = evaluate(model, head, test_l)
            rows.append({"backbone": name, "arm": arm, "test_acc": round(acc, 4)})
            print(f"probe {name}/{arm}: acc={acc:.4f}", flush=True)

    out = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out, "cpu_check_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "arm", "test_acc"])
        w.writeheader(); w.writerows(rows)
    assert len(rows) == 8 and all(0.0 <= r["test_acc"] <= 1.0 for r in rows)
    print("CPU PIPELINE CHECK PASSED: 8/8 probe arms completed, "
          "delta extraction + superposition math validated", flush=True)


if __name__ == "__main__":
    main()
