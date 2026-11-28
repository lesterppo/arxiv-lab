#!/usr/bin/env python3
"""Skill composition repro — arXiv 2611.34398 (When Skills Fail to Compose).

FALLBACK driver (attempt 3). The Qwen3-8B NF4 arm was BLOCKED: bitsandbytes
fell back to the slow _dequant_linear_fallback path (minutes/step) and OOMed
on the first training step. This driver runs the same protocol on Qwen3-4B
in plain bf16 (no quantization, the proven 0.6B-arm code path).

Self-contained Colab driver. Trains soft-prompt skills (20 trainable embedding
rows, frozen base) for two digit operations from the paper — SHIFT_LEFT and
SWAP_PAIRS (the paper's worked example pair) — on Qwen3-0.6B (paper-scale arm)
and Qwen3-8B (mandatory current-gen 8-9B arm, NF4), then measures:

  * single-skill accuracy (n=200 held-out, diagnostic subset)
  * ONE-call composition accuracy (both blocks in a single call)
  * TWO-call composition accuracy (sequential calls, explicit intermediate)
  * partner-replacement check (0.6B arm always; 8B arm if time permits)

Paper recipe (App. A.2): skill = 20 new embedding rows, cross-entropy, all
other params frozen; LR 0.005, 99,999 examples, 3 epochs (we shrink the
experiment to 2-3k examples/skill, 2 epochs — never the model — to fit a
~90 min T4 budget; loss reaches ~0 within ~100 steps, and a top-up rule
re-trains any skill whose quick single accuracy is < 0.90); eval = greedy
decoding, exact match on first |y| chars, diagnostic subset where
x, A(x), B(A(x)) are pairwise distinct. Training loss is computed only on
the 8 target positions (hidden states sliced before lm_head — the 152k
vocab makes full-sequence logits the step-time bottleneck).

Writes /content/skillcomp_metrics.csv incrementally (one row per condition),
/content/prompt_blocks.pt (trained blocks, updated after each skill),
/content/progress.json, /content/skillcomp_summary.json. Prints TRAIN_DONE
as the very last stdout line.
"""

import csv
import json
import math
import os
import random
import subprocess
import sys
import time

START = time.time()
METRICS_CSV = "/content/skillcomp_metrics_4b.csv"
SUMMARY_JSON = "/content/skillcomp_summary_4b.json"
BLOCKS_PT = "/content/prompt_blocks_4b.pt"
PROGRESS_JSON = "/content/progress_4b.json"

# ---------------------------------------------------------------- pure logic
def op_sl(s):
    """SHIFT_LEFT: move first digit to end. 26647378 -> 66473782 (paper)."""
    return s[1:] + s[:1]


def op_swap(s):
    """SWAP_PAIRS: exchange adjacent digits. 66473782 -> 66747328 (paper)."""
    return "".join(s[i + 1] + s[i] for i in range(0, len(s) - 1, 2))


OPS = {"SL": op_sl, "SWAP": op_swap}


def is_diagnostic(x, a_fn, b_fn):
    ax = a_fn(x)
    bax = b_fn(ax)
    return len({x, ax, bax}) == 3


def rand_digits(rng, length=8):
    return "".join(rng.choice("0123456789") for _ in range(length))


def gen_items(rng, n, a_fn, b_fn, length=8):
    items = []
    while len(items) < n:
        x = rand_digits(rng, length)
        if is_diagnostic(x, a_fn, b_fn):
            items.append(x)
    return items


def gen_train(rng, n, length=8):
    return [rand_digits(rng, length) for _ in range(n)]


def score_exact(outputs, golds):
    return sum(1 for o, g in zip(outputs, golds) if o == g)


def sanitize_digits(s, length=8):
    """Coerce a generated string to an 8-digit string for the 2nd call."""
    s = "".join(c if c.isdigit() else "0" for c in s)[:length]
    return s + "0" * (length - len(s))


# ---------------------------------------------------------------- driver
def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def elapsed():
    return time.time() - START


def write_progress(state):
    try:
        with open(PROGRESS_JSON, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"progress write failed: {e}")


def append_metric(row):
    header = ["arm", "model", "condition", "order", "n", "correct", "accuracy",
              "a_present", "notes"]
    exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})
    log(f"metric: {row['arm']} {row['condition']} {row.get('order','')} "
        f"n={row['n']} acc={row['accuracy']:.3f}")


def main():
    import torch

    # ---- installs (Colab ships transformers 5.x; we need <5.0; peft not used
    # ---- but torchao uninstall is the standing driver rule) ----
    log("installing deps...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.57.1", "bitsandbytes", "accelerate",
                    "hf_transfer"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda"
    assert torch.cuda.is_available(), "no CUDA on this VM"
    log(f"GPU: {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")

    rng = random.Random(20261128)
    torch.manual_seed(20261128)

    N_TRAIN = {"4B": 3000}
    EPOCHS = 2
    BATCH = {"4B": 16}
    LR = 0.03
    PROMPT_LEN = 20
    EVAL_N = 200
    QUICK_N = 100

    blocks = {}  # name -> fp32 cpu tensor (20, hidden)

    def save_blocks():
        torch.save({k: v.cpu() for k, v in blocks.items()}, BLOCKS_PT)

    def digit_ids(tok, s):
        ids = []
        for ch in s:
            ids.extend(tok(ch, add_special_tokens=False)["input_ids"])
        return ids

    def load_model(model_id, quant4=False):
        log(f"loading {model_id} (4bit={quant4})...")
        tok = AutoTokenizer.from_pretrained(model_id)
        if quant4:
            bnb = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb, device_map="auto",
                torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        model.config.use_cache = False
        model.eval()
        log(f"loaded {model_id}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
        return model, tok

    def init_mean(model, tok):
        emb = model.get_input_embeddings().weight.detach()
        ids = []
        for d in "0123456789":
            ids.extend(tok(d, add_special_tokens=False)["input_ids"])
        # gather the 10 digit rows first, then cast (avoids a 2.5GB fp32
        # copy of the full 152k-vocab embedding matrix)
        mean = emb[torch.tensor(ids, device=emb.device)].float().mean(dim=0)
        return mean.to("cpu")

    def train_skill(model, tok, op_fn, arm, skill_name, epochs, lr, n_train,
                    init_vec, topup_from=None):
        """Train one 20-token soft prompt. Returns fp32 cpu tensor (20, d)."""
        dtype = torch.bfloat16
        emb_layer = model.get_input_embeddings()
        d = model.config.hidden_size
        if topup_from is not None:
            prompt = torch.nn.Parameter(topup_from.clone().to(device))
            log(f"top-up {skill_name} from existing prompt")
        else:
            prompt = torch.nn.Parameter(
                init_vec.to(device) + 0.02 * torch.randn(PROMPT_LEN, d, device=device))
        opt = torch.optim.AdamW([prompt], lr=lr, weight_decay=0.0)
        steps_per_epoch = math.ceil(n_train / BATCH[arm])
        total_steps = steps_per_epoch * epochs
        warmup = max(5, total_steps // 20)

        def lr_at(step):
            if step < warmup:
                return lr * (step + 1) / warmup
            t = (step - warmup) / max(1, total_steps - warmup)
            return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

        model.train()
        step = 0
        for ep in range(epochs):
            xs = gen_train(rng, n_train)
            ys = [op_fn(x) for x in xs]
            for i in range(0, n_train, BATCH[arm]):
                chunk_x, chunk_y = xs[i:i + BATCH[arm]], ys[i:i + BATCH[arm]]
                seqs, labs = [], []
                for x, y in zip(chunk_x, chunk_y):
                    xin = digit_ids(tok, x)
                    yin = digit_ids(tok, y)
                    assert len(xin) == 8 and len(yin) == 8, (xin, yin)
                    e_in = emb_layer(torch.tensor(xin, device=device)).to(dtype)
                    e_tg = emb_layer(torch.tensor(yin, device=device)).to(dtype)
                    seq = torch.cat([prompt.to(dtype), e_in, e_tg], dim=0)
                    seqs.append(seq)
                    labs.append([-100] * (PROMPT_LEN + 8) + yin)
                inp = torch.stack(seqs)
                tgt = torch.tensor([l[PROMPT_LEN + 8:] for l in labs],
                                   device=device)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                # Loss only on the 8 target positions: slice hidden states
                # before lm_head (positions PROMPT_LEN+7 .. PROMPT_LEN+14
                # predict targets at PROMPT_LEN+8 .. PROMPT_LEN+15).
                hidden = model.model(inputs_embeds=inp).last_hidden_state
                h = hidden[:, PROMPT_LEN + 7:PROMPT_LEN + 15, :]
                logits = model.lm_head(h).float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_([prompt], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    log(f"  {skill_name} ep{ep+1}/{epochs} step {step}/{total_steps} "
                        f"loss {loss.item():.4f}")
        model.eval()
        log(f"trained {skill_name} ({arm}): {epochs} epochs, {step} steps")
        return prompt.detach().float().cpu()

    @torch.no_grad()
    def batch_greedy(model, tok, prompt_blocks, digit_strs, batch_size, max_new=8):
        dtype = torch.bfloat16
        emb_layer = model.get_input_embeddings()
        outs = []
        for i in range(0, len(digit_strs), batch_size):
            chunk = digit_strs[i:i + batch_size]
            seqs = []
            for s in chunk:
                ids = digit_ids(tok, s)
                assert len(ids) == 8, (s, ids)
                parts = [p.to(device=device, dtype=dtype) for p in prompt_blocks]
                parts.append(emb_layer(torch.tensor(ids, device=device)).to(dtype))
                seqs.append(torch.cat(parts, dim=0))
            cur = torch.stack(seqs)
            gen = []
            for _ in range(max_new):
                out = model(inputs_embeds=cur)
                nxt = out.logits[:, -1, :].argmax(dim=-1)
                gen.append(nxt)
                cur = torch.cat([cur, emb_layer(nxt).to(dtype).unsqueeze(1)], dim=1)
            gen_ids = torch.stack(gen, dim=1).tolist()
            for row in gen_ids:
                outs.append(tok.decode(row).replace(" ", ""))
        return outs

    def eval_single(model, tok, prompt, items, op_fn, arm, model_name,
                    skill_name, batch_size):
        outs = batch_greedy(model, tok, [prompt], items, batch_size)
        golds = [op_fn(x) for x in items]
        correct = score_exact(outs, golds)
        append_metric({"arm": arm, "model": model_name, "condition": "single",
                       "order": skill_name, "n": len(items), "correct": correct,
                       "accuracy": correct / len(items), "a_present": "",
                       "notes": f"skill {skill_name} alone"})
        return correct / len(items), outs

    def run_arm(model_id, arm, quant4, do_replacement):
        model, tok = load_model(model_id, quant4)
        # sanity: digits tokenise 1 token each
        n_tok = len(digit_ids(tok, "0123456789"))
        log(f"digit tokenisation check: '0123456789' -> {n_tok} tokens")
        assert n_tok == 10, "digits are not single tokens on this model"
        init_vec = init_mean(model, tok)
        batch_size = BATCH[arm]

        # ---- train SL and SWAP-A, with top-up if single accuracy < 0.90 ----
        trained = {}
        for skill_name in ["SL", "SWAP"]:
            key = f"{skill_name}_{arm}"
            p = train_skill(model, tok, OPS[skill_name], arm, skill_name,
                            EPOCHS, LR, N_TRAIN[arm], init_vec)
            blocks[key] = p
            save_blocks()
            # quick check on fresh items
            q_items = gen_items(rng, QUICK_N, OPS["SL"], OPS["SWAP"])
            acc, _ = eval_single(model, tok, p, q_items, OPS[skill_name],
                                 arm, model_id, skill_name + "-quick", batch_size)
            topups = 0
            while acc < 0.90 and topups < 2 and elapsed() < 85 * 60:
                topups += 1
                log(f"quick acc {acc:.2f} < 0.90 for {skill_name}, top-up {topups}")
                p = train_skill(model, tok, OPS[skill_name], arm, skill_name,
                                2, 0.01, N_TRAIN[arm], init_vec, topup_from=p)
                blocks[key] = p
                save_blocks()
                acc, _ = eval_single(model, tok, p, q_items, OPS[skill_name],
                                     arm, model_id, skill_name + "-quick", batch_size)
            trained[skill_name] = p
            log(f"{skill_name} ({arm}) quick single acc: {acc:.3f} "
                f"(topups={topups})")

        # ---- diagnostic eval set (shared across conditions, paired) ----
        items = gen_items(rng, EVAL_N, OPS["SL"], OPS["SWAP"])
        log(f"diagnostic eval set: n={len(items)}")

        p_sl, p_sw = trained["SL"], trained["SWAP"]
        acc_sl, _ = eval_single(model, tok, p_sl, items, OPS["SL"], arm, model_id,
                                "SL", batch_size)
        acc_sw, _ = eval_single(model, tok, p_sw, items, OPS["SWAP"], arm, model_id,
                                "SWAP", batch_size)

        for first, second, pf, ps in [("SL", "SWAP", p_sl, p_sw),
                                      ("SWAP", "SL", p_sw, p_sl)]:
            golds = [OPS[second](OPS[first](x)) for x in items]
            a_golds = [OPS[first](x) for x in items]
            # one-call: both blocks, single call
            outs1 = batch_greedy(model, tok, [pf, ps], items, batch_size)
            c1 = score_exact(outs1, golds)
            a_present = sum(1 for o, a in zip(outs1, a_golds) if o == a) / len(items)
            append_metric({"arm": arm, "model": model_id,
                           "condition": "one-call", "order": f"{first}->{second}",
                           "n": len(items), "correct": c1,
                           "accuracy": c1 / len(items), "a_present": round(a_present, 3),
                           "notes": "both blocks in one call"})
            # two-call: sequential, explicit intermediate
            mid = batch_greedy(model, tok, [pf], items, batch_size)
            mid_clean = [sanitize_digits(m) for m in mid]
            outs2 = batch_greedy(model, tok, [ps], mid_clean, batch_size)
            c2 = score_exact(outs2, golds)
            append_metric({"arm": arm, "model": model_id,
                           "condition": "two-call", "order": f"{first}->{second}",
                           "n": len(items), "correct": c2,
                           "accuracy": c2 / len(items), "a_present": "",
                           "notes": "sequential calls, explicit intermediate"})
            log(f"{arm} {first}->{second}: one-call {c1/len(items):.3f} "
                f"(A-present {a_present:.3f}) vs two-call {c2/len(items):.3f}")

        # ---- partner replacement (independent SWAP-B block) ----
        if do_replacement:
            log("partner-replacement check: training independent SWAP-B")
            rng2 = random.Random(777)
            old_rng = rng
            # fresh data order / init noise via a different rng stream
            p_b = train_skill(model, tok, OPS["SWAP"], arm, "SWAP-B",
                              EPOCHS, LR, N_TRAIN[arm], init_vec)
            blocks[f"SWAP_B_{arm}"] = p_b
            save_blocks()
            q_items = gen_items(rng, QUICK_N, OPS["SL"], OPS["SWAP"])
            acc_b, _ = eval_single(model, tok, p_b, q_items, OPS["SWAP"], arm,
                                   model_id, "SWAP-B-quick", batch_size)
            log(f"SWAP-B ({arm}) quick single acc: {acc_b:.3f}")
            for first, second, pf, ps in [("SL", "SWAP-B", p_sl, p_b)]:
                golds = [OPS["SWAP"](OPS["SL"](x)) for x in items]
                outs1 = batch_greedy(model, tok, [pf, ps], items, batch_size)
                c1 = score_exact(outs1, golds)
                append_metric({"arm": arm, "model": model_id,
                               "condition": "one-call-replacement",
                               "order": "SL->SWAP-B", "n": len(items), "correct": c1,
                               "accuracy": c1 / len(items), "a_present": "",
                               "notes": "SL + independently learned SWAP-B"})
            log(f"replacement done ({arm})")

        del model, tok
        torch.cuda.empty_cache()
        write_progress({"arm_done": arm, "elapsed_s": elapsed()})
        return {"single_SL": acc_sl, "single_SWAP": acc_sw}

    results = {}
    try:
        results["4B"] = run_arm("Qwen/Qwen3-4B", "4B", quant4=False,
                                do_replacement=True)
        log("4B arm complete")
    except Exception as e:
        log(f"4B arm FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    summary = {"paper": "2611.34398", "elapsed_s": elapsed(), "arms": results}
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"summary written; total elapsed {elapsed()/60:.1f} min")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
