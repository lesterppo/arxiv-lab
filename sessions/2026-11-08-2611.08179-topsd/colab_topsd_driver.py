"""Colab driver — Teachable On-Policy Self-Distillation (TOPSD)
(arXiv:2611.08179, 2026-11-08 daily training track).

Paper: "Knowing When to Teach: Learning from Teachable Prefixes in
Multilingual Reasoning Distillation" (Cho et al.). Core claims:
  (1) Teacher Drift: in On-Policy Self-Distillation (OPSD) the teacher
      (same model conditioned on reference solutions) starts FOLLOWING
      the student's incorrect steps instead of correcting them, and the
      shift happens EARLIER in lower-resource languages.
  (2) TOPSD: use teacher-student likelihood to estimate where drift
      begins, select a teachable prefix, verify the teacher can recover
      the correct answer from that prefix, and distill ONLY verified
      prefixes. TOPSD beats conventional OPSD, especially in low- and
      medium-resource languages (10 languages, 3 model scales, multilingual
      mathematical reasoning).

Bounded adaptation (one T4 day, stated honestly):
  - MGSM (juletxara/mgsm), 3 languages: en (high), zh (medium), sw (low).
    Train idx 0-119, test idx 150-209 per language (same index split in
    all languages, so no problem appears in both train and test in any
    language). Reference solution = MGSM native-language `answer` CoT.
  - Arms (same base, same train pool): base; OPSD (SFT on full
    teacher traces conditioned on the reference solution, verified final
    answer); TOPSD (SFT on verified teachable prefixes only: student
    trace cut at the likelihood-detected drift onset, teacher completes
    from the prefix, kept only if the completion recovers the gold
    answer; correct student traces kept whole).
  - Drift measurement: for up to 30 wrong student traces per language,
    teacher completion (question + student trace + reference solution)
    either recovers the gold answer or repeats the student's wrong
    answer (drift); onset position = first step whose teacher-conditioned
    mean token logprob drops >1.0 nat below the best-so-far step (or
    below -3.0 absolute). Report recovery/drift rate + mean onset
    fraction per language.
  - Eval: greedy, zero-shot + fixed instruction, 60 held-out problems
    per language, last-integer exact match vs MGSM `answer_number`.
  - Model: Qwen/Qwen3.5-9B NF4-QLoRA tried first per standing picks
    (env MODEL_ID overrides, e.g. Qwen/Qwen3-8B for the fallback run).
    QLoRA r=16/alpha=32, lr 2e-4, 1 epoch, batch 2 x accum 2,
    TRAIN_SEQ_CAP 384, gradient checkpointing, sdpa attention.
  - Deviations from the paper, recorded: (a) 3 languages instead of 10,
    120 train / 60 test per language instead of the full MGSM; (b) one
    SFT epoch on ~360 examples per arm (paper: larger-scale OPSD);
    (c) drift onset by per-step teacher logprob drop (paper: teacher-
    student likelihood) with the thresholds above; (d) teacher trace
    verified by final-answer match only; (e) 9B/8B base instead of the
    paper's three scales.
  - Incremental writes: every phase appends to /content/topsd_metrics.csv
    and per-item rows to /content/topsd_details.csv; adapter zips saved
    after each arm (reclamation lesson).
  - Prints TOPSD_DONE as its very last line on completion.

Run via: python3 colab.py train --driver <this file> --rotate
  --session topsd-1108 --monitor-timeout 14400
  --results /content/topsd_metrics.csv /content/topsd_details.csv
  /content/adapters/opsd.zip /content/adapters/topsd.zip
  --out-dir <session dir> --done-marker TOPSD_DONE
  --registry-file <isolated registry>
"""

import csv
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys

METRICS = "/content/topsd_metrics.csv"
DETAILS = "/content/topsd_details.csv"
ADAPTER_DIR = "/content/adapters"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
LANGS = ["en", "zh", "sw"]
N_TRAIN = 120
TEST_START = 150
N_TEST = 60
N_DRIFT = 30
MAX_NEW_EVAL = 192
MAX_NEW_TRACE = 256
SEQ_CAP = 512
TRAIN_SEQ_CAP = 384
SEED = 1108
SFT_LR = 2e-4

INSTR = ("Solve step by step. End your response with a line: "
         "The answer is <number>.")


def _pip():
    pkgs = ["transformers==5.20.2", "peft", "bitsandbytes", "accelerate",
            "safetensors", "datasets"]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                       check=True, timeout=1200)
    except Exception as e:  # noqa: BLE001
        print("pip install failed (continuing with existing stack): %r" % e,
              flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   capture_output=True)


METRIC_FIELDS = ["phase", "model", "lang", "arm", "n", "acc",
                 "acc_en", "acc_zh", "acc_sw", "mean_loss", "peak_gb",
                 "recovery_rate", "drift_rate", "onset_frac_mean",
                 "onset_detected", "opsd_n", "topsd_n"]


def _append_metrics(row):
    new = not os.path.exists(METRICS)
    with open(METRICS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_FIELDS})
    print("METRIC %s" % json.dumps(row), flush=True)


def _append_detail(row):
    new = not os.path.exists(DETAILS)
    with open(DETAILS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def extract_answer(text):
    """Last integer in the text, preferring the number after 'answer is'."""
    if not text:
        return None
    m = re.findall(r"[Aa]nswer is[:\s]*(-?\d[\d,]*)", text)
    if m:
        return int(m[-1].replace(",", ""))
    nums = re.findall(r"-?\d[\d,]*", text)
    if not nums:
        return None
    return int(nums[-1].replace(",", ""))


def split_steps(trace):
    """Split a reasoning trace into steps (non-empty lines; fall back to
    sentence split when the trace is a single line)."""
    lines = [l.strip() for l in (trace or "").split("\n") if l.strip()]
    if len(lines) >= 2:
        return lines
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", trace or "")
             if p.strip()]
    return parts if parts else [trace.strip()]


def drift_onset(step_lps, drop=1.0, floor=-3.0):
    """First step index where the teacher-conditioned mean token logprob
    drops > `drop` nat below the best-so-far step, or below `floor`.
    Returns None when no onset is detected (no drift)."""
    best = None
    for i, lp in enumerate(step_lps):
        if lp < floor:
            return i
        if best is not None and lp < best - drop:
            return i
        best = lp if best is None else max(best, lp)
    return None


def eval_prompt(question):
    return question.strip() + "\n\n" + INSTR


def sft_text(question, trace):
    return eval_prompt(question), "\n" + trace.strip()


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    _pip()
    import torch
    import transformers
    print("transformers=%s" % transformers.__version__, flush=True)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    assert torch.cuda.is_available(), "no GPU: refusing to fake a GPU run"
    print("GPU: %s total=%.1f GB torch=%s"
          % (torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9,
             torch.__version__), flush=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # ---- data: MGSM, disjoint train/test index split per language ----
    # NOTE: juletxara/mgsm puts the 250 problems in split="test"; the
    # "train" split holds only 8 few-shot exemplars. We split the 250
    # test problems into our own train/test pools by index.
    data = {}  # lang -> {"train": [...], "test": [...]}
    for lang in LANGS:
        ds = load_dataset("juletxara/mgsm", lang, split="test")
        recs = [{"q": r["question"], "cot": r["answer"],
                 "gold": int(r["answer_number"])} for r in ds]
        assert len(recs) == 250, "MGSM %s has %d rows, expected 250" % (
            lang, len(recs))
        data[lang] = {"train": recs[:N_TRAIN],
                      "test": recs[TEST_START:TEST_START + N_TEST]}
        print("data %s train=%d test=%d" % (lang, len(data[lang]["train"]),
                                            len(data[lang]["test"])),
              flush=True)

    # ---- model ----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    base.config.use_cache = False
    print("LOADED %s peak=%.2f GB"
          % (MODEL_ID, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    _append_metrics({"phase": "probe", "model": MODEL_ID,
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                     "acc": "", "n": ""})

    @torch.no_grad()
    def batch_generate(model, prompts, max_new, batch=8):
        """Greedy batched generation; returns list of completion strings."""
        model.eval()
        model.config.use_cache = True
        outs = []
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=SEQ_CAP).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                ids = gen[j][plen:].tolist()
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id)]
                outs.append(tok.decode(ids, skip_special_tokens=True))
        return outs

    @torch.no_grad()
    def evaluate(model, arm):
        accs = {}
        for lang in LANGS:
            items = data[lang]["test"]
            prompts = [eval_prompt(r["q"]) for r in items]
            gens = batch_generate(model, prompts, MAX_NEW_EVAL)
            correct, fails = 0, 0
            for r, g in zip(items, gens):
                pred = extract_answer(g)
                if pred is None:
                    fails += 1
                elif pred == r["gold"]:
                    correct += 1
                _append_detail({"phase": "eval", "arm": arm, "lang": lang,
                                "gold": r["gold"],
                                "pred": pred if pred is not None else "",
                                "gen": g[:300].replace("\n", " ")})
            accs[lang] = correct / len(items)
            print("EVAL arm=%s lang=%s n=%d acc=%.4f parse_fail=%d"
                  % (arm, lang, len(items), accs[lang], fails), flush=True)
        _append_metrics({"phase": "eval_" + arm, "model": MODEL_ID,
                         **{("acc_" + l): round(a, 4)
                            for l, a in accs.items()},
                         "n": N_TEST,
                         "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        return accs

    @torch.no_grad()
    def step_logprobs(context, trace):
        """Mean per-token logprob of each step of `trace` under the model
        conditioned on `context`. One forward pass; returns list of floats
        aligned with split_steps(trace)."""
        steps = split_steps(trace)
        ctx_ids = tok(context, add_special_tokens=False)["input_ids"]
        step_ids = [tok(" " + s, add_special_tokens=False)["input_ids"]
                    for s in steps]
        ids = ctx_ids + [t for s in step_ids for t in s]
        ids = ids[-SEQ_CAP:]
        # step token spans in the (possibly truncated) sequence
        spans, pos = [], len(ids)
        for s in reversed(step_ids):
            pos -= len(s)
            spans.append((pos, pos + len(s)))
        spans.reverse()
        # drop spans fully truncated away (only possible at the front)
        spans = [(max(a, 0), b) for a, b in spans if b > 0 and a < len(ids)]
        inp = torch.tensor([ids], device=base.device)
        out = base(input_ids=inp)
        logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
        tgt = torch.tensor(ids[1:], device=base.device)
        tok_lp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        lps = []
        for (a, b) in spans:
            # token at position p predicts token p+1; step span [a,b) is
            # predicted by logits [a-1, b-1)
            seg = tok_lp[max(a - 1, 0):b - 1]
            lps.append(float(seg.mean()) if seg.numel() else -99.0)
        return lps

    # ---- phase 1: base eval ----
    base_accs = evaluate(base, "base")

    # ---- phase 2: student traces on the train pool + drift measurement ----
    train_prompts, train_meta = [], []
    for lang in LANGS:
        for r in data[lang]["train"]:
            train_prompts.append(eval_prompt(r["q"]))
            train_meta.append((lang, r))
    student_traces = batch_generate(base, train_prompts, MAX_NEW_TRACE)
    print("student traces: %d" % len(student_traces), flush=True)

    drift_rows = []  # (lang, onset_frac or None, recovered bool)
    for lang in LANGS:
        wrong = [(m, t) for m, t in zip(train_meta, student_traces)
                 if m[0] == lang and extract_answer(t) != m[1]["gold"]]
        print("drift pool %s: %d wrong of %d"
              % (lang, len(wrong), N_TRAIN), flush=True)
        for (l2, r), trace in wrong[:N_DRIFT]:
            ctx = (r["q"].strip() + "\n\nReference solution:\n"
                   + r["cot"].strip())
            lps = step_logprobs(ctx, trace)
            onset = drift_onset(lps)
            # teacher completion from the student trace + reference:
            # does the teacher recover the gold answer or repeat the error?
            comp = batch_generate(
                base, [r["q"].strip() + "\n\nYour previous attempt:\n"
                       + trace.strip() + "\n\nReference solution:\n"
                       + r["cot"].strip() + "\n\nNow write the corrected "
                       "step-by-step solution. End your response with a "
                       "line: The answer is <number>."],
                MAX_NEW_TRACE, batch=1)[0]
            recovered = extract_answer(comp) == r["gold"]
            onset_frac = (onset / max(1, len(lps))) if onset is not None else None
            drift_rows.append((lang, onset_frac, recovered))
            _append_detail({"phase": "drift", "lang": lang,
                            "gold": r["gold"],
                            "student_pred": extract_answer(trace) or "",
                            "onset": onset if onset is not None else "",
                            "n_steps": len(lps),
                            "recovered": int(recovered)})
    for lang in LANGS:
        rows = [d for d in drift_rows if d[0] == lang]
        if not rows:
            continue
        rec = sum(d[2] for d in rows) / len(rows)
        onsets = [d[1] for d in rows if d[1] is not None]
        _append_metrics({"phase": "drift", "model": MODEL_ID, "lang": lang,
                         "n": len(rows),
                         "recovery_rate": round(rec, 4),
                         "drift_rate": round(1 - rec, 4),
                         "onset_frac_mean": (round(sum(onsets) / len(onsets), 4)
                                             if onsets else ""),
                         "onset_detected": len(onsets)})
        print("DRIFT lang=%s n=%d recovery=%.3f drift=%.3f onset_frac=%s "
              "(detected %d/%d)"
              % (lang, len(rows), rec, 1 - rec,
                 (round(sum(onsets) / len(onsets), 3) if onsets else "n/a"),
                 len(onsets), len(rows)), flush=True)

    # ---- phase 3: build OPSD + TOPSD SFT datasets ----
    def teacher_trace(r):
        """Teacher = base model conditioned on the reference solution.
        Returns a verified-correct trace, else the reference CoT itself."""
        comp = batch_generate(
            base, [r["q"].strip() + "\n\nReference solution:\n"
                   + r["cot"].strip() + "\n\nNow write your own "
                   "step-by-step solution in the same language as the "
                   "question. End your response with a line: "
                   "The answer is <number>."],
            MAX_NEW_TRACE, batch=1)[0]
        if extract_answer(comp) == r["gold"] and len(comp.strip()) > 20:
            return comp.strip()
        return r["cot"].strip()

    opsd_data, topsd_data = [], []  # (prompt, completion) pairs
    n_teacher_calls = 0
    for (lang, r), trace in zip(train_meta, student_traces):
        spred = extract_answer(trace)
        # OPSD arm: full teacher trace (verified correct).
        tt = teacher_trace(r)
        n_teacher_calls += 1
        opsd_data.append(sft_text(r["q"], tt))
        # TOPSD arm: teachable prefix only.
        if spred == r["gold"]:
            topsd_data.append(sft_text(r["q"], trace.strip()))
            continue
        ctx = (r["q"].strip() + "\n\nReference solution:\n"
               + r["cot"].strip())
        lps = step_logprobs(ctx, trace)
        onset = drift_onset(lps)
        steps = split_steps(trace)
        prefix = "\n".join(steps[:onset]) if onset is not None else ""
        if not prefix.strip():
            # drift at step 0: no teachable prefix; drop the example.
            continue
        comp = batch_generate(
            base, [r["q"].strip() + "\n\nPartial solution so far:\n"
                   + prefix.strip() + "\n\nContinue the step-by-step "
                   "solution from here. End your response with a line: "
                   "The answer is <number>."],
            MAX_NEW_TRACE, batch=1)[0]
        if extract_answer(comp) == r["gold"]:
            topsd_data.append(
                sft_text(r["q"], prefix.strip() + "\n" + comp.strip()))
    print("SFT data: opsd=%d topsd=%d teacher_calls=%d"
          % (len(opsd_data), len(topsd_data), n_teacher_calls), flush=True)
    _append_metrics({"phase": "data_built", "model": MODEL_ID,
                     "opsd_n": len(opsd_data), "topsd_n": len(topsd_data),
                     "acc": "", "n": ""})

    # ---- phase 4: SFT arm helper (completion-only CE, proven scaffold) ----
    def train_arm(examples, tag):
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj", "gate_proj", "up_proj",
                                          "down_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg)
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        def sft_batch(chunk):
            pids = [tok(p, add_special_tokens=False)["input_ids"]
                    for p, _ in chunk]
            cids = [tok("\n" + c + tok.eos_token,
                        add_special_tokens=False)["input_ids"]
                    for _, c in chunk]
            maxlen = min(max(len(a) + len(b) for a, b in zip(pids, cids)),
                         TRAIN_SEQ_CAP)
            input_ids, labels, attn = [], [], []
            for a, b in zip(pids, cids):
                ids = (a + b)[:maxlen]
                lab = ([-100] * len(a) + b)[:maxlen]
                pad = maxlen - len(ids)
                input_ids.append(ids + [tok.pad_token_id] * pad)
                labels.append(lab + [-100] * pad)
                attn.append([1] * len(ids) + [0] * pad)
            return (torch.tensor(input_ids, device=model.device),
                    torch.tensor(labels, device=model.device),
                    torch.tensor(attn, device=model.device))

        opt = torch.optim.AdamW(model.parameters(), lr=SFT_LR,
                                weight_decay=0.0)
        model.train()
        losses, steps = [], 0
        for step in range(0, len(examples), 2):
            chunk = examples[step:step + 2]
            ids, labs, att = sft_batch(chunk)
            out = model(input_ids=ids, labels=labs, attention_mask=att)
            (out.loss / 2).backward()
            losses.append(float(out.loss))
            if (step // 2 + 1) % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
            if (step // 2 + 1) % 50 == 0:
                print("SFT %s micro-step %d loss=%.4f peak=%.2f GB"
                      % (tag, step // 2 + 1, losses[-1],
                         torch.cuda.max_memory_allocated() / 1e9), flush=True)
        if (len(examples) // 2) % 2 == 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
        mean_loss = sum(losses) / max(1, len(losses))
        print("SFT %s done mean_loss=%.4f steps=%d peak=%.2f GB"
              % (tag, mean_loss, steps,
                 torch.cuda.max_memory_allocated() / 1e9), flush=True)
        model.save_pretrained("/content/adapters/" + tag)
        shutil.make_archive("/content/adapters/" + tag, "zip",
                            "/content/adapters", tag)
        _append_metrics({"phase": "sft_done_" + tag, "model": MODEL_ID,
                         "mean_loss": round(mean_loss, 4), "n": len(examples),
                         "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        accs = evaluate(model, tag)
        return model

    # ---- phase 5: OPSD arm, eval, save; reload base; TOPSD arm, eval ----
    model = train_arm(opsd_data, "opsd")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    base.config.use_cache = False
    print("RELOADED base for TOPSD arm peak=%.2f GB"
          % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    model = train_arm(topsd_data, "topsd")
    del model
    gc.collect()

    print("TOPSD_DONE", flush=True)


if __name__ == "__main__":
    main()
