"""Colab driver BODY for EliteReplay (2610.22290) — appended after elite_common.py.

Run on a Colab T4. Expects the INSTALL_HEADER (pip) to have run first.
Writes: /content/elitereplay_status.txt, /content/elitereplay_metrics.csv,
        /content/elitereplay.log (via detach), /content/elitereplay_results.json,
        /content/elitereplay_archive.json, /content/elitereplay_data.json
"""
import csv
import json
import os
import random
import time

import torch

T0 = time.time()
OUT = "/content"
STATUS = os.path.join(OUT, "elitereplay_status.txt")
METRICS = os.path.join(OUT, "elitereplay_metrics.csv")
RESULTS = os.path.join(OUT, "elitereplay_results.json")

CFG = {
    "model_primary": os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B"),
    "model_fallback": "Qwen/Qwen3-8B",
    "n_train": 96,
    "n_eval": 60,
    "eval_subset": 30,
    "k64_subset": 8,
    "archive_prompts": 48,
    "archive_g": 8,
    "archive_cap": 200,
    "grpo_steps": 24,
    "g": 4,
    "replay_coef": 0.3,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_completion": 128,
    "max_prompt": 256,
    "wall_cap_s": 5.5 * 3600,
    "seed": 11,
}
notes = []


def log(msg):
    line = f"[{time.time()-T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def elapsed():
    return time.time() - T0


def wall_left(minutes=20):
    return elapsed() < CFG["wall_cap_s"] - minutes * 60


PROMPT_TMPL = (
    "You are a text-to-SQL assistant. Given the schema and question, "
    "write ONLY the SQL query, no explanation.\n\nSchema:\n"
    "CREATE TABLE staff(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER, age INTEGER);\n"
    "CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price INTEGER, stock INTEGER);\n\n"
    "Question: {nl}\nSQL:"
)

# ------------------------------------------------------------ data ----------
random.seed(CFG["seed"])
db = build_db(seed=7)
train, evalp = generate_split(CFG["n_train"], CFG["n_eval"], seed=CFG["seed"])
with open(os.path.join(OUT, "elitereplay_data.json"), "w") as f:
    json.dump({"train": train, "eval": evalp}, f)
prompt_of = {PROMPT_TMPL.format(nl=p["nl"]): p for p in train + evalp}
eval_sub = evalp[: CFG["eval_subset"]]
log(f"data ready: {len(train)} train / {len(evalp)} eval; db tables ok")

# ------------------------------------------------------------ model ---------
from huggingface_hub import login as hf_login  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN") or None
if HF_TOKEN:
    try:
        hf_login(HF_TOKEN)
        log("HF login ok (token from env)")
    except Exception as e:  # noqa: BLE001
        notes.append(f"hf_login failed: {e}")
        log(f"HF login failed: {e}")
else:
    notes.append("no HF_TOKEN in env; trying anonymous download")

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model  # noqa: E402

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
MODEL_USED = CFG["model_primary"]


def load_base(model_id):
    # HF token via env var only (HF_TOKEN=os.environ.get("HF_TOKEN")); never hardcoded.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, token=HF_TOKEN, trust_remote_code=True,  # via env var, see above
        quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    return tok, mdl


try:
    tok, base_model = load_base(MODEL_USED)
    log(f"loaded primary model {MODEL_USED}")
except Exception as e:  # noqa: BLE001
    notes.append(f"primary model {MODEL_USED} failed: {type(e).__name__}: {e}")
    log(f"primary failed ({e}); falling back to {CFG['model_fallback']}")
    MODEL_USED = CFG["model_fallback"]
    tok, base_model = load_base(MODEL_USED)
    log(f"loaded fallback model {MODEL_USED}")

log(f"cuda: {torch.cuda.get_device_name(0)}; "
    f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB")


@torch.no_grad()
def gen_sql(model, prompt, n=1, temp=0.8, max_new=None):
    enc = tok([prompt] * n, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new or CFG["max_completion"],
        do_sample=(temp > 0), temperature=max(temp, 1e-6), top_p=0.95,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return [extract_sql(tok.decode(g, skip_special_tokens=True)) for g in gen]


def eval_pass(model, problems, tag):
    """greedy pass@1 + sampled pass@8 / distinct-correct-strategies."""
    n1_ok = 0
    for p in problems:
        pr = PROMPT_TMPL.format(nl=p["nl"])
        s = gen_sql(model, pr, n=1, temp=0.0)[0]
        n1_ok += reward_correct(db, s, p["sql"])
    pass1 = n1_ok / len(problems)
    # sampled diversity
    cands, ok_flags = [], []
    for p in problems:
        pr = PROMPT_TMPL.format(nl=p["nl"])
        for s in gen_sql(model, pr, n=8, temp=0.8):
            c = reward_correct(db, s, p["sql"])
            cands.append((s, p["sql"]))
            ok_flags.append(c)
    pass8 = sum(1 for p_i in range(len(problems))
                if any(ok_flags[p_i * 8 + j] for j in range(8))) / len(problems)
    # per-problem distinct correct fingerprints
    strat_total, strat_hit = 0, 0
    for p_i, p in enumerate(problems):
        pc = [(cands[p_i * 8 + j][0], p["sql"]) for j in range(8)]
        d = distinct_correct_strategies(db, pc)
        strat_total += d
        strat_hit += 1 if d > 0 else 0
    res = {"pass@1": round(pass1, 4), "pass@8": round(pass8, 4),
           "distinct_strategies@8_total": strat_total,
           "distinct_strategies@8_problems_hit": strat_hit,
           "n": len(problems)}
    log(f"eval[{tag}]: " + json.dumps(res))
    return res


def eval_pass64(model, problems, tag):
    res = {}
    per = []
    for p in problems:
        pr = PROMPT_TMPL.format(nl=p["nl"])
        oks = [reward_correct(db, s, p["sql"]) for s in gen_sql(model, pr, n=64, temp=0.8)]
        per.append(1 if any(oks) else 0)
        if not wall_left(15):
            notes.append(f"pass@64[{tag}] aborted early at {len(per)}/{len(problems)} (wall clock)")
            log(f"pass@64[{tag}] early stop at {len(per)} problems")
            break
    res = {"pass@64": round(sum(per) / max(len(per), 1), 4), "n": len(per)}
    log(f"eval[{tag}]: " + json.dumps(res))
    return res


# ------------------------------------------------------------ base eval ----
log("== base (frozen) eval ==")
base_eval = eval_pass(base_model, eval_sub, "base")
base_k64 = eval_pass64(base_model, evalp[: CFG["k64_subset"]], "base") if wall_left(90) else {"skipped": True}

# ------------------------------------------------------------ LoRA ---------
lora_cfg = LoraConfig(
    r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()
log("LoRA attached (r=16/a=32, all linears)")

# ------------------------------------------------------------ elite archive
log("== elite archive build (frozen base, quality-diversity sampling) ==")


def frozen_sampler(nl, ref_sql):
    pr = PROMPT_TMPL.format(nl=nl)
    return gen_sql(model, pr, n=CFG["archive_g"], temp=1.0)


archive = EliteArchive(cap=CFG["archive_cap"])
arch_problems = train[: CFG["archive_prompts"]]
for i, p in enumerate(arch_problems):
    if not wall_left(60):
        notes.append(f"archive build stopped early at prompt {i}/{len(arch_problems)}")
        log("archive early stop (wall clock)")
        break
    for s in frozen_sampler(p["nl"], p["sql"]):
        archive.add(p["id"], p["nl"], s, reward_correct(db, s, p["sql"]))
    if (i + 1) % 12 == 0:
        log(f"archive: {len(archive)} entries after {i+1} prompts")
log(f"archive done: {len(archive)} elite entries "
    f"({len(archive.per_prompt_counts())} prompts covered)")
with open(os.path.join(OUT, "elitereplay_archive.json"), "w") as f:
    json.dump(archive.items, f)

# ------------------------------------------------------------ GRPO ---------
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

train_ds = [{"prompt": PROMPT_TMPL.format(nl=p["nl"])} for p in train]
random.Random(CFG["seed"]).shuffle(train_ds)


def sqlite_reward(prompts, completions, **kwargs):
    out = []
    for pr, co in zip(prompts, completions):
        p = prompt_of.get(pr)
        sql = extract_sql(co)
        out.append(float(reward_correct(db, sql, p["sql"])) if p else 0.0)
    return out


class CsvLogger(TrainerCallback):
    def __init__(self, arm):
        self.arm = arm
        if not os.path.exists(METRICS):
            with open(METRICS, "w", newline="") as f:
                csv.writer(f).writerow(["arm", "step", "loss", "reward", "kl", "replay_loss"])

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        with open(METRICS, "a", newline="") as f:
            csv.writer(f).writerow([
                self.arm, state.global_step,
                round(logs.get("loss", float("nan")), 5),
                round(logs.get("reward", float("nan")), 5),
                round(logs.get("kl", float("nan")), 5),
                round(getattr(kw.get("model"), "_last_replay_loss", float("nan"))
                      if kw.get("model") is not None else float("nan"), 5),
            ])


class EliteGRPOTrainer(GRPOTrainer):
    """GRPOTrainer + auxiliary elite-archive replay loss (the paper's pairing)."""

    def __init__(self, *args, elite=None, replay_coef=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.elite = elite or []
        self.replay_coef = replay_coef
        self._rrng = random.Random(1234)
        self._last_replay_loss = float("nan")

    def _replay_loss(self, model):
        if not self.elite or self.replay_coef <= 0:
            self._last_replay_loss = 0.0
            return 0.0
        k = min(8, len(self.elite))
        batch = self._rrng.sample(self.elite, k)
        texts = [PROMPT_TMPL.format(nl=b["nl"]) + " " + b["sql"] for b in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=CFG["max_prompt"] + CFG["max_completion"])
        plen = [len(tok(PROMPT_TMPL.format(nl=b["nl"]),
                        add_special_tokens=False)["input_ids"]) for b in batch]
        labels = enc["input_ids"].clone()
        for i, pl in enumerate(plen):
            labels[i, :pl] = -100
        labels[enc["attention_mask"] == 0] = -100
        dev = next(model.parameters()).device
        outm = model(input_ids=enc["input_ids"].to(dev),
                     attention_mask=enc["attention_mask"].to(dev),
                     labels=labels.to(dev))
        self._last_replay_loss = float(outm.loss.detach().cpu())
        return outm.loss

    def _compute_loss(self, model, inputs, *a, **k):
        out = super()._compute_loss(model, inputs, *a, **k)
        loss = out[0] if isinstance(out, tuple) else out
        try:
            rl = self._replay_loss(model)
            loss = loss + self.replay_coef * rl
        except Exception as e:  # noqa: BLE001
            notes.append(f"replay term failed at step: {type(e).__name__}: {e}")
        return (loss, out[1]) if isinstance(out, tuple) else loss


def grpo_config(steps, outdir):
    kw = dict(
        output_dir=outdir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=CFG["g"],
        max_prompt_length=CFG["max_prompt"],
        max_completion_length=CFG["max_completion"],
        max_steps=steps,
        learning_rate=CFG["lr"],
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        temperature=0.9,
        top_p=0.95,
        seed=CFG["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    # defensive: drop kwargs unknown to the installed TRL version
    for _ in range(8):
        try:
            return GRPOConfig(**kw)
        except TypeError as e:
            import re as _re
            m = _re.search(r"'([^']+)'", str(e))
            if not m or m.group(1) not in kw:
                raise
            notes.append(f"GRPOConfig dropped unknown kwarg {m.group(1)}")
            del kw[m.group(1)]
    raise RuntimeError("GRPOConfig construction failed")


def run_arm(arm, replay_coef):
    tag = f"arm-{arm}"
    log(f"== {tag}: GRPO{' + elite replay' if replay_coef else ' only'} ==")
    cfg = grpo_config(CFG["grpo_steps"], os.path.join(OUT, f"ckpt-{tag}"))
    trainer = EliteGRPOTrainer(
        model=model, args=cfg, train_dataset=train_ds,
        processing_class=tok, reward_funcs=[sqlite_reward],
        callbacks=[CsvLogger(arm)], elite=archive.items, replay_coef=replay_coef,
    )
    trainer.train()
    log(f"{tag} training done")
    ev = eval_pass(model, eval_sub, tag)
    k64 = eval_pass64(model, evalp[: CFG["k64_subset"]], tag) if wall_left(45) else {"skipped": True}
    return {"eval": ev, "pass64": k64, "replay_coef": replay_coef}


results = {
    "paper": "2610.22290 EliteReplay",
    "model_requested": CFG["model_primary"],
    "model_used": MODEL_USED,
    "config": CFG,
    "archive": {"size": len(archive),
                "prompts_covered": len(archive.per_prompt_counts())},
    "base": {"eval": base_eval, "pass64": base_k64},
    "notes": notes,
}

ok_a = True
try:
    results["arm_a"] = run_arm("a", replay_coef=0.0)
except Exception as e:  # noqa: BLE001
    ok_a = False
    notes.append(f"arm (a) failed: {type(e).__name__}: {e}")
    log(f"arm (a) FAILED: {e}")

# fresh LoRA for arm (b) so the two arms are comparable
if wall_left(60):
    try:
        log("reloading fresh base+LoRA for arm (b)")
        del model
        torch.cuda.empty_cache()
        _, base2 = load_base(MODEL_USED)
        model = get_peft_model(base2, lora_cfg)
        results["arm_b"] = run_arm("b", replay_coef=CFG["replay_coef"])
    except Exception as e:  # noqa: BLE001
        notes.append(f"arm (b) failed: {type(e).__name__}: {e}")
        log(f"arm (b) FAILED: {e}")
else:
    notes.append("arm (b) skipped: wall-clock budget")

results["elapsed_s"] = round(elapsed(), 1)
with open(RESULTS, "w") as f:
    json.dump(results, f, indent=1)
log("RESULTS WRITTEN to " + RESULTS)
print("ELITE_DONE " + json.dumps({
    "base": results["base"]["eval"],
    "arm_a": results.get("arm_a", {}).get("eval"),
    "arm_b": results.get("arm_b", {}).get("eval"),
    "archive_size": results["archive"]["size"],
    "elapsed_s": results["elapsed_s"],
}, indent=1), flush=True)
