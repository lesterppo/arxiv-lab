"""CARM ultra-short drift test on Qwen2.5-7B-Instruct (Colab T4).

Paper arXiv:2610.02039: standard mask (|mean log r|<=band) accepts
sequences with large CANCELING drift; CARM (mean|log r|<=band) rejects.

Ultra-short protocol (~12 min):
 1. Load 7B 4-bit + LoRA. Generate 8 rollouts (GSM8K, eval mode).
 2. Reference log-probs.
 3. Synthetic drift: Gaussian noise on LoRA weights, 3 scales.
 4. Per scale: mask decisions, disagreement rate, canceling structure.

Output: /content/carm_7b_ultra.json + printed summary.
"""
import json, math, random, re, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED, N_PROBS, G, TOKENS = 7, 4, 2, 192
EPS, BAND = 0.2, math.log1p(0.2)
NOISE_SCALES = [0.01, 0.03, 0.1]

def load_gsm8k(n, seed):
    url = ("https://raw.githubusercontent.com/openai/grade-school-math/"
           "master/grade_school_math/data/test.jsonl")
    rows = [json.loads(l) for l in urllib.request.urlopen(url, timeout=120)
            .read().decode().splitlines() if l.strip()]
    out = []
    for r in random.Random(seed).sample(rows, n):
        m = re.search(r"####\s*(-?[\d,]+)", r["answer"])
        out.append((r["question"] + "\nSolve in at most 5 short steps. "
                    "Put the final numeric answer in \\boxed{}.",
                    m.group(1).replace(",", "")))
    return out

def reward_fn(text, ans):
    m = re.findall(r"\\boxed\{\s*(-?[\d,]+)\s*\}", text)
    return 1.0 if m and m[-1].replace(",", "") == ans else 0.0

def seq_logps(model, pids, cids):
    inp = torch.cat([pids, cids], dim=1)
    with torch.no_grad():
        lp = torch.log_softmax(model(inp).logits[0].float(), dim=-1)
    off = pids.shape[1] - 1
    idx = torch.arange(cids.shape[1], device=inp.device)
    return lp[off + idx, cids[0]]

torch.manual_seed(SEED); random.seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
                                         target_modules=["q_proj", "v_proj"],
                                         task_type="CAUSAL_LM"))
model.eval()
lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
orig = [p.detach().clone() for p in lora_params]

problems = load_gsm8k(N_PROBS, SEED)
rollouts = []
with torch.no_grad():
    for q, ans in problems:
        pids = tok(q, return_tensors="pt").input_ids.cuda()
        out = model.generate(pids, max_new_tokens=TOKENS, do_sample=True,
                             temperature=0.8, top_p=0.95,
                             pad_token_id=tok.eos_token_id,
                             num_return_sequences=G)
        for gi in range(G):
            cids = out[gi:gi + 1, pids.shape[1]:]
            text = tok.decode(cids[0], skip_special_tokens=True)
            rlp = seq_logps(model, pids, cids)
            rollouts.append((pids, cids, reward_fn(text, ans), rlp))
print(f"rollouts={len(rollouts)} "
      f"mean_reward={sum(r[2] for r in rollouts)/len(rollouts):.2f}", flush=True)

report = {}
for ns in NOISE_SCALES:
    with torch.no_grad():
        for p, o in zip(lora_params, orig):
            p.copy_(o + ns * torch.randn_like(o))
    n_dis, n_std, n_carm, cancel_cases = 0, 0, 0, 0
    mean_abs_all, max_abs_dis = [], []
    for pids, cids, r, rlp in rollouts:
        logr = (seq_logps(model, pids, cids) - rlp).tolist()
        mlr = sum(logr) / len(logr)
        mal = sum(abs(x) for x in logr) / len(logr)
        sk = abs(mlr) <= BAND
        ck = mal <= BAND
        n_std += sk; n_carm += ck
        mean_abs_all.append(mal)
        if sk and not ck:
            n_dis += 1
            max_abs_dis.append(max(abs(x) for x in logr))
            # canceling? both signs present with |x|>0.1
            pos = any(x > 0.1 for x in logr)
            neg = any(x < -0.1 for x in logr)
            cancel_cases += (pos and neg)
    report[str(ns)] = {
        "mean_abs_logratio": round(sum(mean_abs_all)/len(mean_abs_all), 4),
        "std_keep_rate": round(n_std/len(rollouts), 3),
        "carm_keep_rate": round(n_carm/len(rollouts), 3),
        "disagreements": n_dis,
        "disagreement_rate": round(n_dis/len(rollouts), 3),
        "canceling_among_disagreements": f"{cancel_cases}/{n_dis}",
        "mean_max_abs_if_disagree": round(
            sum(max_abs_dis)/max(len(max_abs_dis),1), 3),
    }
    print(f"noise={ns}: " + json.dumps(report[str(ns)]), flush=True)

with torch.no_grad():
    for p, o in zip(lora_params, orig):
        p.copy_(o)
json.dump(report, open("/content/carm_7b_ultra.json", "w"), indent=1)
print("ULTRA TEST DONE", flush=True)
