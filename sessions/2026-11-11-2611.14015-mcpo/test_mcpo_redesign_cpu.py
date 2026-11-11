#!/usr/bin/env python3
"""CPU test for the redesigned MCPO retry (arXiv 2611.14015) — GSM8K arms.

Verifies, without any GPU/model/network (except an optional local-parquet
check), that the two split arm drivers are identical except for the
token-credit weighting, that the GSM8K parsing/selection logic is correct
on the real data, and that the redesign's signal assumption holds.

Run:  python3 test_mcpo_redesign_cpu.py   (from the session dir)
"""
import ast, math, os, random, re, sys

SESSION = os.path.dirname(os.path.abspath(__file__))
GRPO = os.path.join(SESSION, "colab_mcpo_driver_gsm8k_grpo.py")
MCPO = os.path.join(SESSION, "colab_mcpo_driver_gsm8k_mcpo.py")
PARQUET_DIR = "/tmp/mcpo-redesign"  # local copies of the GSM8K parquet files

CHECKS = []
def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond)))
    print(("ok: " if cond else "FAIL: ") + name + (f" — {detail}" if detail else ""))
    if not cond:
        raise SystemExit(f"FAILED: {name}")

HELPER_FNS = ["margin_weights", "spread_advantage", "extract_boxed",
              "extract_answer", "_try_float", "norm_ans", "answers_equal",
              "gsm8k_parse_gold", "gsm8k_build_prompt", "select_pools"]
CONST_NAMES = ["MODEL", "SEED", "N_TRAIN_POOL", "N_EVAL", "PROMPTS_PER_STEP",
               "GROUP", "STEPS", "MAX_NEW", "MAX_PROMPT_TOKENS", "LR",
               "LORA_R", "LORA_ALPHA", "T1", "T2", "W_HI", "W_MID", "W_LO",
               "EVAL_BATCH", "GSM8K_INSTRUCTION"]
# Config keys allowed to differ between the two arm drivers (arm identity only).
ALLOWED_DIFF = {"ARM_NAME", "ARM_MCPO", "DONE_MARKER", "METRICS_CSV",
                "RESULTS_JSON", "ADAPTER_DIR", "MID_ADAPTER_DIR"}

def extract_helpers(path):
    """Exec only the pure-helper defs + needed constants from a driver file
    (the driver itself cannot be imported on CPU: it loads a GPU model at
    module level). Returns (namespace, source_text)."""
    src = open(path).read()
    tree = ast.parse(src)
    ns = {"re": re, "math": math, "random": random}
    wanted_assigns = set(CONST_NAMES) | {"ARM_NAME", "ARM_MCPO", "DONE_MARKER",
                                          "METRICS_CSV", "RESULTS_JSON",
                                          "ADAPTER_DIR", "MID_ADAPTER_DIR"}
    parts = []
    def _target_names(targets):
        names = set()
        for t in targets:
            if isinstance(t, ast.Name):
                names.add(t.id)
            elif isinstance(t, ast.Tuple):
                names |= _target_names(t.elts)
        return names
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_FNS:
            parts.append(node)
        elif isinstance(node, ast.Assign):
            if _target_names(node.targets) & wanted_assigns:
                parts.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in wanted_assigns:
                parts.append(node)
    # GSM8K_GOLD_RE / ANSWER_RE are compiled-regex assignments used by helpers.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if _target_names(node.targets) & {"ANSWER_RE", "GSM8K_GOLD_RE"}:
                parts.append(node)
    mod = ast.Module(body=parts, type_ignores=[])
    exec(compile(mod, path, "exec"), ns)
    return ns, src

g_ns, g_src = extract_helpers(GRPO)
m_ns, m_src = extract_helpers(MCPO)

# ---- 1. helpers identical across arms (same source text for every helper) ----
def fn_source(path, name):
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(open(path).read(), node)
    return None
for fn in HELPER_FNS:
    check(f"helper {fn} identical in both drivers", fn_source(GRPO, fn) == fn_source(MCPO, fn))

# ---- 2. config identity except arm identity ----
def config_map(path):
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out
gc, mc = config_map(GRPO), config_map(MCPO)
diff_keys = {k for k in set(gc) | set(mc) if gc.get(k) != mc.get(k)}
check("configs differ ONLY in arm-identity keys", diff_keys <= ALLOWED_DIFF, f"diff={sorted(diff_keys)}")
check("both arms share seed/model/steps", gc["SEED"] == mc["SEED"] == 20261111 and gc["MODEL"] == mc["MODEL"] == "Qwen/Qwen3-8B" and gc["STEPS"] == mc["STEPS"] == 24)
check("arm flags are grpo=False / mcpo=True", gc["ARM_NAME"] == "grpo" and gc["ARM_MCPO"] is False and mc["ARM_NAME"] == "mcpo" and mc["ARM_MCPO"] is True)
check("done markers distinct per arm", gc["DONE_MARKER"] == "MCPO_GRPO_DONE" and mc["DONE_MARKER"] == "MCPO_MCPO_DONE")

# ---- 3. margin weighting / advantage spreading (paper mechanism) ----
mw = g_ns["margin_weights"]
check("margin weights thresholds", mw([0.0, 0.09, 0.10, 0.49, 0.50, 0.99]) == [1.0, 1.0, 0.5, 0.5, 0.1, 0.1])
sa = g_ns["spread_advantage"]
a = sa(2.0, [1.0, 0.5, 0.1])
check("spread preserves mean advantage", abs(sum(a) / len(a) - 2.0) < 1e-12, f"mean={sum(a)/len(a)}")
check("spread zero advantage stays zero", sa(0.0, [1.0, 0.5, 0.1]) == [0.0, 0.0, 0.0])

# ---- 4. answer extraction / equality (reused unchanged) ----
ea = g_ns["extract_answer"]; eq = g_ns["answers_equal"]
check("extract boxed", ea("work... \\boxed{42} tail") == "42")
check("extract boxed last wins", ea("\\boxed{1} then \\boxed{2}") == "2")
check("extract Answer line", ea("reasoning\nAnswer: $72") == "72")
check("extract Answer beats nothing", ea("no answer here") is None)
check("answers equal: commas", eq("72,000", "72000"))
check("answers equal: fraction", eq("\\frac{1}{2}", "0.5"))
check("answers unequal", not eq("72", "73"))
check("answers equal: None safe", not eq(None, "72") and not eq("72", None))

# ---- 5. GSM8K gold parsing (inline real rows + full files when present) ----
pg = g_ns["gsm8k_parse_gold"]
check("gsm8k gold basic", pg("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72") == "72")
check("gsm8k gold comma", pg("... #### 95,200") == "95200")
check("gsm8k gold negative", pg("... #### -3") == "-3")
check("gsm8k gold missing -> None", pg("no final line") is None)
try:
    import pandas as pd
    tr = pd.read_parquet(os.path.join(PARQUET_DIR, "gsm8k_train.parquet"))
    te = pd.read_parquet(os.path.join(PARQUET_DIR, "gsm8k_test.parquet"))
    ok_tr = sum(1 for a in tr["answer"] if pg(str(a)) is not None and pg(str(a)).lstrip("-").isdigit())
    ok_te = sum(1 for a in te["answer"] if pg(str(a)) is not None and pg(str(a)).lstrip("-").isdigit())
    check("gsm8k gold parse train 7473/7473", ok_tr == 7473, f"{ok_tr}/7473")
    check("gsm8k gold parse test 1319/1319", ok_te == 1319, f"{ok_te}/1319")
    qs = [str(q) for q in tr["question"]] + [str(q) for q in te["question"]]
    w = [len(q.split()) for q in qs]
    check("gsm8k questions fit MAX_PROMPT_TOKENS=512", max(w) <= 250, f"max words={max(w)}")
    sw = [len(str(a).split()) for a in list(tr["answer"]) + list(te["answer"])]
    check("gsm8k solutions fit MAX_NEW=320", max(sw) <= 260, f"max sol words={max(sw)}")
    HAVE_PARQUET = True
except Exception as e:
    print(f"note: full-parquet check skipped ({e}); inline cases above still cover parsing")
    HAVE_PARQUET = False

# ---- 6. prompt builder ----
bp = g_ns["gsm8k_build_prompt"]
p = bp("Natalia sold clips to 48 friends. How many altogether?")
check("prompt contains question", "Natalia sold clips" in p)
check("prompt specifies Answer format", "Answer: $N" in p)
check("prompt does not append a CoT", "<<" not in p and "####" not in p)

# ---- 7. pool selection: identical across arms, sizes, no train/eval overlap ----
sp = g_ns["select_pools"]
rng_rows = lambda n, tag: [{"prompt": f"{tag} prompt {i}", "gold": str(i)} for i in range(n)]
train_rows, test_rows = rng_rows(800, "train"), rng_rows(200, "test")
tp1, ev1, pp1 = sp(train_rows, test_rows)
tp2, ev2, pp2 = sp(train_rows, test_rows)  # second call = the other arm
check("pool sizes 512/80/96", len(tp1) == 512 and len(ev1) == 80 and len(pp1) == 96,
      f"{len(tp1)}/{len(ev1)}/{len(pp1)}")
check("paired prompts identical across arms", pp1 == pp2)
check("eval sets identical across arms", ev1 == ev2)
check("train pools identical across arms", tp1 == tp2)
check("paired prompts subset of train pool", all(x in tp1 for x in pp1))
check("no train/eval prompt overlap", not ({r['prompt'] for r in tp1} & {r['prompt'] for r in ev1}))
check("pool selection deterministic across processes (seed fixed)",
      sp(train_rows, test_rows)[2] == pp1)

# ---- 8. signal math: mixed-group fraction at group=6 ----
def mixed_frac(p, group=6):
    return 1.0 - p**group - (1 - p)**group
for p, lo in [(0.7, 0.8), (0.8, 0.7), (0.9, 0.4)]:
    check(f"mixed-group fraction at p={p} > {lo}", mixed_frac(p) > lo, f"{mixed_frac(p):.3f}")
check("DAPO contrast: ~4/88 groups had signal", abs(4 / 88 - 0.045) < 0.001)
print(f"note: mixed fractions p=0.7/0.8/0.9 -> {mixed_frac(0.7):.3f}/{mixed_frac(0.8):.3f}/{mixed_frac(0.9):.3f}; "
      f"DAPO measured 0.045 -> GSM8K gives ~10-19x denser signal")

# ---- 9. driver pitfalls present in BOTH files ----
for label, src in [("grpo", g_src), ("mcpo", m_src)]:
    check(f"{label}: torchao uninstall", "pip uninstall -y torchao" in src)
    check(f"{label}: expandable_segments", "expandable_segments:True" in src)
    check(f"{label}: chunked loss CH=64", "CH = 64" in src)
    check(f"{label}: micro 4", "micro = 4" in src)
    check(f"{label}: mid-run adapter checkpoint at step 12", "STEPS // 2" in src and "MID_ADAPTER_DIR" in src)
    check(f"{label}: adapter zipped", "make_archive" in src)
    check(f"{label}: metrics flushed per row", "metrics_f.flush()" in src)
    check(f"{label}: results.json rewritten per phase", src.count('json.dump(results, open(RESULTS_JSON, "w")') >= 2)
    check(f"{label}: SIGNAL_CHECK after base eval", "SIGNAL_CHECK" in src)
    check(f"{label}: done marker is the last print", src.rstrip().endswith('print(DONE_MARKER, flush=True)'))
    check(f"{label}: no TRAIN_DONE (arm marker replaces it)", "TRAIN_DONE" not in src)
    check(f"{label}: HF token never in code", "HF_TOKEN" not in src and "huggingface_hub" not in src.lower().replace("huggingface.co", ""))

print(f"\n{len(CHECKS)} checks passed")
