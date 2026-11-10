"""Colab driver — Audit Before You Merge: Provenance, Probing, and Continual
LoRA Composition (arXiv:2611.10684, 2026-11-10 daily training track).

Paper claims tested (bounded scale, one T4 day, stated honestly):
  C1  Public adapter hubs contain modules that fail basic provenance checks
      (silent base-version mismatch, near-zero delta norm); including them
      inflates measured interference by ~10pp. We test the two-check audit
      (base-model match, non-trivial delta norm) against a deliberately
      wrong-base adapter and an untrained (zero-delta) adapter.
  C2  On an audited adapter set, naive merging causes forgetting (-10.5pp
      GSM8K, -18.3pp HumanEval on Llama-3.1-8B-Instruct, 4 adapters) while
      Task Arithmetic at lambda=0.5 sits within a standard error of the
      unmerged baseline. We test with 2 adapters (math + code) on Qwen3.5-9B
      (attempt 1) / Qwen3-8B (fallback): arms base / single-adapter /
      merged TA lambda=1.0 ("naive sum") / TA lambda=0.5 / math lambda=-0.5.
      With 2 adapters, averaging the two merged models equals TA-0.5, so
      the naive arm is the unscaled sum (lambda=1.0) — the paper's naive
      merge on 4 adapters is the unscaled average of 4 merged models.
  C3  A coordinate search including negative coefficients finds the math
      adapter probe-optimal at lambda=-0.5. We test the math-neg arm
      (math -0.5, code +0.5) on the GSM8K probe.

Bounded adaptation: GSM8K train 800 / test 60; MBPP train 374 / test 40
(MBPP stands in for HumanEval — same code-generation shape, runnable
without the human-eval package). QLoRA r=16/alpha=32, NF4, bf16, lr 2e-4,
micro-batch 2 / accum 2, seq 512, grad checkpointing on, 1 epoch math /
2 epochs code (MBPP train is small). Model choice: standing 9B-first rule
— Qwen3.5-9B needs transformers==5.20.2 (4.x has no qwen3_5 modeling;
FPRT v3 lesson, 2026-11-09). If this attempt BLOCKED, the automatic
fallback (2026-11-05 rule) reruns the same driver with Qwen/Qwen3-8B.

Phases (incremental writes to /content so a mid-run download sees partials):
  0 installs, 1 data, 2 train math LoRA, 3 train code LoRA, 4 audit,
  5 merge + eval arms. Prints AUDITMERGE_DONE as the very last line on
  full completion; on phase failure prints AUDITMERGE_FAILED:<phase> and
  exits 1 (wrapper reports DIED; partials still downloadable).

Run via: python3 colab.py train --driver <this file> --rotate
  --session auditmerge-1110 --monitor-timeout 14400
  --env MODEL_ID=Qwen/Qwen3.5-9B
  --results /content/auditmerge_metrics.csv /content/auditmerge_eval.jsonl
    /content/adapters/math.zip /content/adapters/code.zip
  --out-dir <session dir> --done-marker AUDITMERGE_DONE
  --registry-file <isolated registry>
"""
import csv
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

METRICS = "/content/auditmerge_metrics.csv"
EVAL_DETAILS = "/content/auditmerge_eval.jsonl"
ADAPTER_DIR = "/content/adapters"

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
N_MATH_TRAIN = int(os.environ.get("N_MATH_TRAIN", "800"))
N_MATH_TEST = int(os.environ.get("N_MATH_TEST", "60"))
N_CODE_TEST = int(os.environ.get("N_CODE_TEST", "40"))
CODE_EPOCHS = int(os.environ.get("CODE_EPOCHS", "2"))
SEED = 20261110

FEWSHOT = """Solve each problem step by step. End your solution with "The answer is: <number>".

Problem: A baker made 4 trays of 12 muffins. He sold 30 muffins in the morning and 9 in the afternoon. How many muffins are left?
Solution: The baker made 4 * 12 = 48 muffins. He sold 30 + 9 = 39 muffins. So 48 - 39 = 9 muffins are left.
The answer is: 9

Problem: Sara had 15 stickers. She bought 3 packs with 8 stickers each and gave 10 to her brother. How many stickers does she have now?
Solution: She bought 3 * 8 = 24 stickers. She had 15 + 24 = 39 stickers. After giving away 10, she has 39 - 10 = 29 stickers.
The answer is: 29

Problem: A rectangle is 8 meters long and 5 meters wide. What is its perimeter?
Solution: The perimeter is 2 * (8 + 5) = 2 * 13 = 26 meters.
The answer is: 26

Problem: Tom reads 20 pages every day. A book has 140 pages. How many days will it take him to finish the book?
Solution: 140 / 20 = 7 days.
The answer is: 7
"""

CODE_FEW = """Write a Python function to solve the following problem.

Problem: Write a function to find the longest chain of letters in a string.
```python
def longest_chain(s):
"""

CODE_INSTR = "Write a Python function to solve the following problem.\n\nProblem: {text}\n```python\n"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_METRIC_ROWS = []
_METRIC_COLS = ["phase"]


def metric(phase, **kw):
    row = {"phase": phase}
    row.update({k: v for k, v in kw.items()})
    _METRIC_ROWS.append(row)
    for k in row:
        if k not in _METRIC_COLS:
            _METRIC_COLS.append(k)
    with open(METRICS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METRIC_COLS)
        w.writeheader()
        for r in _METRIC_ROWS:
            w.writerow({c: r.get(c, "") for c in _METRIC_COLS})
    log("METRIC " + json.dumps(row))


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def extract_answer(text):
    if text is None:
        return None
    m = re.search(r"[Tt]he answer is:?\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def answers_equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def gold_answer(ans_field):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans_field)
    return m.group(1).replace(",", "") if m else None


def peak_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0


def install():
    log("PHASE 0: installs")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if "3.5" in MODEL_ID:
        # Qwen3.5 needs transformers 5.x (qwen3_5 modeling absent from 4.x;
        # FPRT v3 lesson 2026-11-09). Fail fast if it is not registered.
        tf_pin = "transformers==5.20.2"
    else:
        tf_pin = "transformers>=4.46,<5.0"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", tf_pin,
                    "peft", "accelerate", "bitsandbytes", "datasets",
                    "hf_transfer"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False, capture_output=True)
    import transformers
    log(f"transformers {transformers.__version__} model={MODEL_ID}")
    if "3.5" in MODEL_ID:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
        assert "qwen3_5" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, (
            "transformers has no qwen3_5 model registered — "
            "Qwen3.5-9B cannot load")
    import torch
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_data():
    from datasets import load_dataset
    log("PHASE 1: data")
    gsm = load_dataset("openai/gsm8k", "main")
    math_train = [{"q": r["question"], "a": r["answer"]}
                  for r in gsm["train"].select(range(N_MATH_TRAIN))]
    math_test = [{"question": r["question"], "answer": r["answer"]}
                 for r in gsm["test"].select(range(N_MATH_TEST))]
    # MBPP: `google/mbpp` 401s (removed/gated) and the bare `mbpp` id
    # breaks datasets' URI parsing (HfUriError, needs namespace/name).
    # Load the canonical parquet files directly instead — schema verified
    # 2026-11-10: train 374 / test 500, cols task_id/text/code/test_list/
    # test_setup_code/challenge_test_list.
    import urllib.request
    import pyarrow.parquet as pq
    mbpp_files = {}
    for split, fn in (("train", "train-00000-of-00001.parquet"),
                      ("test", "test-00000-of-00001.parquet")):
        url = ("https://huggingface.co/datasets/mbpp/resolve/main/full/" + fn)
        local = f"/content/mbpp_{split}.parquet"
        urllib.request.urlretrieve(url, local)
        mbpp_files[split] = pq.read_table(local).to_pylist()
        log(f"mbpp {split}: {len(mbpp_files[split])} rows from {url}")
    code_train = [{"text": r["text"], "code": r["code"]}
                  for r in mbpp_files["train"]]
    code_test = [{"task_id": r["task_id"], "text": r["text"],
                  "test_list": list(r["test_list"]),
                  "test_setup_code": r.get("test_setup_code") or ""}
                 for r in mbpp_files["test"][:N_CODE_TEST]]
    log(f"data: math_train={len(math_train)} math_test={len(math_test)} "
        f"code_train={len(code_train)} code_test={len(code_test)}")
    metric("data", math_train=len(math_train), math_test=len(math_test),
           code_train=len(code_train), code_test=len(code_test))
    return math_train, math_test, code_train, code_test


def load_causal(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    model.config.use_cache = True
    model.eval()
    log(f"LOADED {model_id} peak={peak_gb()} GB")
    return tok, model


def free_mem():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


STOP_FEW_SHOT = ["\nProblem:", "Problem:"]


def batch_generate(tok, model, prompts, max_new_tokens, temperature=0.0,
                   batch_size=8, stop_strings=None):
    import torch
    outs = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        gkw = dict(max_new_tokens=max_new_tokens,
                   do_sample=(temperature > 0),
                   temperature=max(temperature, 1e-6),
                   top_p=0.95, pad_token_id=tok.pad_token_id)
        if stop_strings:
            gkw["stop_strings"] = stop_strings
            gkw["tokenizer"] = tok
        with torch.no_grad():
            gen = model.generate(**enc, **gkw)
        plen = enc["input_ids"].shape[1]
        for g in gen:
            outs.append(tok.decode(g[plen:], skip_special_tokens=True))
        log(f"  gen {min(s + batch_size, len(prompts))}/{len(prompts)}")
    return outs


def sft_train(tok, base, examples, tag, epochs):
    """Manual QLoRA SFT loop (FPRT-proven pattern): micro-batch 2, accum 2,
    completion-only loss, seq cap 512, lr 2e-4, bf16, grad checkpointing."""
    import torch
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    # Gradient checkpointing needs the checkpointed segments' inputs to
    # require grad; under transformers 4.x the manual loop dies at the
    # first backward ("element 0 of tensors does not require grad") without
    # this. (trl's SFTTrainer does it internally; the 9B run under
    # transformers 5.20.2 never hit it.) This is NOT
    # prepare_model_for_kbit_training() — no fp32 cast of the frozen base.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    SEQ_CAP = 512

    def batchify(chunk):
        prompts, comps = [], []
        for ex in chunk:
            if ex["kind"] == "math":
                prompts.append(FEWSHOT + f"Problem: {ex['q'].strip()}\nSolution:")
                sol = ex["a"].split("####")[0].strip()
                comps.append("\n" + sol + f"\nThe answer is: {gold_answer(ex['a'])}"
                             + tok.eos_token)
            else:  # code
                prompts.append(CODE_INSTR.format(text=ex["text"].strip()))
                comps.append(ex["code"].strip() + tok.eos_token)
        pids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        cids = [tok(c, add_special_tokens=False)["input_ids"] for c in comps]
        maxlen = min(max(len(a) + len(b) for a, b in zip(pids, cids)), SEQ_CAP)
        ids, labs, att = [], [], []
        for a, b in zip(pids, cids):
            full = (a + b)[:maxlen]
            lab = ([-100] * len(a) + b)[:maxlen]
            pad = maxlen - len(full)
            ids.append(full + [tok.pad_token_id] * pad)
            labs.append(lab + [-100] * pad)
            att.append([1] * len(full) + [0] * pad)
        return (torch.tensor(ids, device=model.device),
                torch.tensor(labs, device=model.device),
                torch.tensor(att, device=model.device))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    model.train()
    losses, steps = [], 0
    for epoch in range(epochs):
        random.shuffle(examples)
        for s in range(0, len(examples), 2):
            chunk = examples[s:s + 2]
            ids, labs, att = batchify(chunk)
            out = model(input_ids=ids, labels=labs, attention_mask=att)
            (out.loss / 2).backward()
            losses.append(float(out.loss))
            if (s // 2 + 1) % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
        log(f"  {tag} epoch {epoch + 1}/{epochs} done, "
            f"mean_loss={sum(losses) / len(losses):.4f} peak={peak_gb()} GB")
    if (len(examples) // 2) % 2 == 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps += 1
    mean_loss = sum(losses) / max(len(losses), 1)
    log(f"SFT {tag} done: {len(examples)} examples, {steps} steps, "
        f"mean_loss={mean_loss:.4f} peak={peak_gb()} GB")
    metric(f"sft_{tag}", n=len(examples), steps=steps,
           mean_loss=round(mean_loss, 4), peak_gb=peak_gb())
    return model


def save_adapter(peft_model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    log(f"saved adapter -> {out_dir}")
    return out_dir


def zip_adapter(adapter_dir):
    z = adapter_dir + ".zip"
    shutil.make_archive(adapter_dir, "zip", adapter_dir)
    log(f"zipped {z} ({os.path.getsize(z) / 1e6:.1f} MB)")
    return z


def delta_norm(adapter_dir):
    """Frobenius norm of the effective delta sum over all LoRA pairs:
    ||(alpha/r) * B @ A||_F, the paper's 'delta norm' for the audit."""
    import torch
    from safetensors.torch import load_file
    import glob
    sd = {}
    for f in glob.glob(os.path.join(adapter_dir, "*.safetensors")):
        sd.update(load_file(f))
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = cfg.get("lora_alpha", 32) / cfg.get("r", 16)
    total = 0.0
    for k in list(sd):
        if k.endswith("lora_A.weight"):
            base = k[:-len("lora_A.weight")]
            b = sd.get(base + "lora_B.weight")
            if b is None:
                continue
            a = sd[k].float()
            d = scale * (b.float() @ a)
            total += float((d * d).sum())
    return math.sqrt(total)


def audit_adapter(adapter_dir, expected_base):
    """Paper's two-check audit. Returns (passed, detail)."""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return False, "missing adapter_config.json"
    cfg = json.load(open(cfg_path))
    recorded = cfg.get("base_model_name_or_path", "")
    if recorded != expected_base:
        return False, (f"BASE MISMATCH: adapter records base "
                       f"'{recorded}', expected '{expected_base}'")
    dn = delta_norm(adapter_dir)
    if dn < 1e-3:
        return False, f"NEAR-ZERO DELTA: ||delta||_F = {dn:.6f} < 1e-3"
    return True, f"ok: base match, ||delta||_F = {dn:.2f}"


def phase_audit(math_dir, code_dir):
    log("PHASE 4: provenance audit (paper's two checks)")
    results = {}
    for tag, d in (("math", math_dir), ("code", code_dir)):
        ok, detail = audit_adapter(d, MODEL_ID)
        results[tag] = {"passed": ok, "detail": detail}
        log(f"  audit {tag}: {'PASS' if ok else 'FAIL'} — {detail}")
        metric(f"audit_{tag}", passed=int(ok), detail=detail[:200])
    # Deliberately bad adapters: the audit must catch both.
    bad_base = "/content/adapters/bad_base"
    shutil.copytree(math_dir, bad_base, dirs_exist_ok=True)
    cfgp = os.path.join(bad_base, "adapter_config.json")
    cfg = json.load(open(cfgp))
    cfg["base_model_name_or_path"] = "Qwen/Qwen3-4B"
    json.dump(cfg, open(cfgp, "w"), indent=2)
    ok, detail = audit_adapter(bad_base, MODEL_ID)
    results["bad_base"] = {"passed": ok, "detail": detail}
    log(f"  audit bad_base (wrong base): {'PASS' if ok else 'FAIL'} — {detail}")
    metric("audit_bad_base", passed=int(ok), detail=detail[:200])
    # Untrained adapter: copy the math adapter and zero every lora_B matrix
    # (peft zero-inits B, so a fresh LoRA has delta norm exactly 0) — no
    # second model load needed.
    from safetensors.torch import load_file, save_file
    import glob
    import torch
    bad_zero = "/content/adapters/bad_zero"
    shutil.copytree(math_dir, bad_zero, dirs_exist_ok=True)
    for f in glob.glob(os.path.join(bad_zero, "*.safetensors")):
        sd = load_file(f)
        for k in list(sd):
            if k.endswith("lora_B.weight"):
                sd[k] = torch.zeros_like(sd[k])
        save_file(sd, f, metadata={"format": "pt"})
    ok, detail = audit_adapter(bad_zero, MODEL_ID)
    results["bad_zero"] = {"passed": ok, "detail": detail}
    log(f"  audit bad_zero (untrained): {'PASS' if ok else 'FAIL'} — {detail}")
    metric("audit_bad_zero", passed=int(ok), detail=detail[:200])
    shutil.rmtree(bad_base, ignore_errors=True)
    shutil.rmtree(bad_zero, ignore_errors=True)
    return results


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_mbpp_tests(code, test_list, task_id, test_setup_code=""):
    """Run the generated function against MBPP's asserts in a subprocess.
    test_setup_code (e.g. helper class defs the asserts need) is prepended."""
    prog = (test_setup_code or "") + "\n\n" + code + "\n\n" + "\n".join(test_list) + "\n"
    with open("/tmp/_mbpp_test.py", "w") as f:
        f.write(prog)
    try:
        r = subprocess.run([sys.executable, "/tmp/_mbpp_test.py"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0, (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:300]


def evaluate_math(tok, model, problems, arm):
    log(f"EVAL {arm} math: {len(problems)} problems, greedy")
    prompts = [FEWSHOT + f"Problem: {p['question'].strip()}\nSolution:"
               for p in problems]
    texts = batch_generate(tok, model, prompts, max_new_tokens=200,
                           temperature=0.0, batch_size=8,
                           stop_strings=STOP_FEW_SHOT)
    n_ok, rows = 0, []
    for p, t in zip(problems, texts):
        gold = gold_answer(p["answer"])
        got = extract_answer(t)
        ok = got is not None and answers_equal(got, gold)
        n_ok += int(ok)
        rows.append({"arm": arm, "bench": "gsm8k", "gold": gold, "got": got,
                     "correct": ok, "gen": t[:600]})
    for r in rows:
        append_jsonl(EVAL_DETAILS, r)
    acc = n_ok / len(problems)
    lo = acc - 1.96 * math.sqrt(acc * (1 - acc) / len(problems))
    hi = acc + 1.96 * math.sqrt(acc * (1 - acc) / len(problems))
    log(f"EVAL {arm} math: acc={acc:.4f} ({n_ok}/{len(problems)}) "
        f"95% CI [{max(0, lo):.3f}, {min(1, hi):.3f}] peak={peak_gb()} GB")
    metric(f"eval_{arm}_gsm8k", acc=round(acc, 4), n=len(problems),
           ci_lo=round(max(0, lo), 4), ci_hi=round(min(1, hi), 4),
           peak_gb=peak_gb())
    return acc, rows


def evaluate_code(tok, model, problems, arm):
    log(f"EVAL {arm} code: {len(problems)} problems, greedy")
    prompts = [CODE_INSTR.format(text=p["text"].strip()) for p in problems]
    texts = batch_generate(tok, model, prompts, max_new_tokens=256,
                           temperature=0.0, batch_size=4)
    n_ok, rows = 0, []
    for p, t in zip(problems, texts):
        code = extract_code(t)
        ok, err = run_mbpp_tests(code, p["test_list"], p["task_id"],
                                 p.get("test_setup_code", ""))
        n_ok += int(ok)
        rows.append({"arm": arm, "bench": "mbpp", "task_id": p["task_id"],
                     "correct": ok, "err": err, "code": code[:600]})
    for r in rows:
        append_jsonl(EVAL_DETAILS, r)
    acc = n_ok / len(problems)
    lo = acc - 1.96 * math.sqrt(acc * (1 - acc) / len(problems))
    hi = acc + 1.96 * math.sqrt(acc * (1 - acc) / len(problems))
    log(f"EVAL {arm} code: acc={acc:.4f} ({n_ok}/{len(problems)}) "
        f"95% CI [{max(0, lo):.3f}, {min(1, hi):.3f}] peak={peak_gb()} GB")
    metric(f"eval_{arm}_mbpp", acc=round(acc, 4), n=len(problems),
           ci_lo=round(max(0, lo), 4), ci_hi=round(min(1, hi), 4),
           peak_gb=peak_gb())
    return acc, rows


def mcnemar(a_rows, b_rows):
    b = sum(1 for a, c in zip(a_rows, b_rows) if a["correct"] and not c["correct"])
    c = sum(1 for a, c2 in zip(a_rows, b_rows) if not a["correct"] and c2["correct"])
    if b + c == 0:
        return None
    z = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(z / 2))
    return {"b_only_a": b, "c_only_b": c, "chi2": round(z, 3), "p": round(p, 4)}


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(SEED)
    try:
        install()
        math_train, math_test, code_train, code_test = load_data()
        tok, base = load_causal(MODEL_ID)

        # Phase 2: math LoRA
        log("PHASE 2: train math LoRA")
        math_ex = [{"kind": "math", "q": r["q"], "a": r["a"]} for r in math_train]
        peft_math = sft_train(tok, base, math_ex, "math", epochs=1)
        math_dir = save_adapter(peft_math, os.path.join(ADAPTER_DIR, "math"))
        base = peft_math.unload()
        del peft_math
        free_mem()

        # Phase 3: code LoRA
        log("PHASE 3: train code LoRA")
        code_ex = [{"kind": "code", "text": r["text"], "code": r["code"]}
                   for r in code_train]
        peft_code = sft_train(tok, base, code_ex, "code", epochs=CODE_EPOCHS)
        code_dir = save_adapter(peft_code, os.path.join(ADAPTER_DIR, "code"))
        base = peft_code.unload()
        del peft_code
        free_mem()
        zip_adapter(math_dir)
        zip_adapter(code_dir)

        # Phase 4: audit
        audit = phase_audit(math_dir, code_dir)
        assert audit["math"]["passed"] and audit["code"]["passed"], \
            "genuine adapters must pass the audit"
        assert not audit["bad_base"]["passed"], \
            "audit must reject the wrong-base adapter"
        assert not audit["bad_zero"]["passed"], \
            "audit must reject the near-zero-delta adapter"
        log("AUDIT: genuine PASS, both planted bad adapters REJECTED")

        # Phase 5: merge + eval. Load both adapters into one PeftModel.
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, math_dir, adapter_name="math")
        model.load_adapter(code_dir, adapter_name="code")
        model.eval()
        model.config.use_cache = True

        results = {}
        # base arm (adapters disabled via context manager)
        with model.disable_adapter():
            b_m, b_m_rows = evaluate_math(tok, model, math_test, "base")
            b_c, b_c_rows = evaluate_code(tok, model, code_test, "base")
        results["base"] = (b_m, b_c)
        # single-adapter arms (own benchmark only)
        model.set_adapter("math")
        m_m, _ = evaluate_math(tok, model, math_test, "math_only")
        results["math_only"] = (m_m, None)
        model.set_adapter("code")
        _, c_c = evaluate_code(tok, model, code_test, "code_only")
        results["code_only"] = (None, c_c)
        # merged arms
        arms = [("ta05", [0.5, 0.5]), ("ta10", [1.0, 1.0]),
                ("mathneg", [-0.5, 0.5])]
        for name, weights in arms:
            model.add_weighted_adapter(
                adapters=["math", "code"], weights=weights,
                adapter_name=name, combination_type="linear")
            model.set_adapter(name)
            a_m, a_m_rows = evaluate_math(tok, model, math_test, name)
            a_c, a_c_rows = evaluate_code(tok, model, code_test, name)
            results[name] = (a_m, a_c)
            metric(f"mcnemar_{name}_vs_base_gsm8k",
                   **(mcnemar(b_m_rows, a_m_rows) or {}))
            metric(f"mcnemar_{name}_vs_base_mbpp",
                   **(mcnemar(b_c_rows, a_c_rows) or {}))
            model.delete_adapter(name)
            free_mem()

        log(f"SUMMARY model={MODEL_ID} base GSM8K={results['base'][0]:.3f} "
            f"MBPP={results['base'][1]:.3f} | math_only GSM8K="
            f"{results['math_only'][0]:.3f} | code_only MBPP="
            f"{results['code_only'][1]:.3f} | TA-0.5 GSM8K="
            f"{results['ta05'][0]:.3f} MBPP={results['ta05'][1]:.3f} | "
            f"TA-1.0 GSM8K={results['ta10'][0]:.3f} MBPP="
            f"{results['ta10'][1]:.3f} | math-neg GSM8K="
            f"{results['mathneg'][0]:.3f} MBPP={results['mathneg'][1]:.3f}")
        print("AUDITMERGE_DONE", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"AUDITMERGE_FAILED:{type(e).__name__}:{e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
