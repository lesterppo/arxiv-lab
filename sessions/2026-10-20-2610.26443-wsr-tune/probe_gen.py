#!/usr/bin/env python3
"""Self-contained probe: does short SFT break generation? Prints 6
generations before/after 30 SFT steps. Run on the Colab T4 VM."""
import random, subprocess, sys, time
subprocess.run("pip install -q 'transformers>=4.56,<5.0' peft bitsandbytes accelerate",
               shell=True, check=True)
subprocess.run("pip uninstall -y torchao", shell=True, check=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

UNITS = [("km","m",1000.0),("m","cm",100.0),("kg","g",1000.0),("hr","min",60.0)]
def gen(n, seed):
    rng = random.Random(seed); out=[]
    for _ in range(n):
        u1,u2,f = rng.choice(UNITS); x = round(rng.uniform(0.5,999.5),2)
        out.append((f"Convert {x} {u1} to {u2}. Reply with only the number.",
                    f"{round(x*f,2)}"))
    return out

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", quantization_config=bnb, device_map="auto",
    torch_dtype=torch.bfloat16, trust_remote_code=True)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(); model.config.use_cache=False
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                        target_modules=["q_proj","v_proj"],
                                        task_type="CAUSAL_LM"))
def fmt(t):
    return tok.apply_chat_template([{"role":"user","content":t}], tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
@torch.no_grad()
def gen_text(t):
    ids = tok(fmt(t), return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=24, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

data = gen(120, seed=11); probe = gen(6, seed=999)
print("BEFORE:", flush=True)
for p,a in probe: print(f"  {p!r} -> {gen_text(p)!r} (key {a})", flush=True)
model.train()
opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=2e-4)
for step in range(30):
    rng = random.Random(step); batch = rng.sample(data, 4)
    pr = [fmt(p) for p,_ in batch]
    enc = tok([x+y for x,y in zip(pr,[a for _,a in batch])], padding=True,
              return_tensors="pt", truncation=True, max_length=256)
    labels = enc["input_ids"].clone()
    for j,pp in enumerate(pr): labels[j,:len(tok(pp)["input_ids"])] = -100
    enc = {k:v.to(model.device) for k,v in enc.items()}
    opt.zero_grad(); loss = model(**enc, labels=labels.to(model.device)).loss
    loss.backward(); opt.step()
    if step % 10 == 0: print(f"step {step} loss={loss.item():.4f}", flush=True)
print("AFTER:", flush=True)
for p,a in probe: print(f"  {p!r} -> {gen_text(p)!r} (key {a})", flush=True)
print("PROBE DONE", flush=True)
