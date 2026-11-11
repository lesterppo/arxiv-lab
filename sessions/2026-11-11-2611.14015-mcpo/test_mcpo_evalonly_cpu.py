#!/usr/bin/env python3
"""CPU test for the MCPO GRPO eval-only resume driver (arXiv 2611.14015).

Verifies, without any GPU/model/network, that the eval-only driver runs
the SAME seeded 80-item GSM8K eval as the GRPO arm driver (identical
helpers, identical constants, identical pool selection), that the banked
step-12 adapter it resumes from is present and matches the training
config, and that it fails loudly instead of silently evaling the base
model if the adapter is missing.

Run:  python3 test_mcpo_evalonly_cpu.py   (from the session dir)
"""
import ast, json, math, os, random, re, sys, zipfile

SESSION = os.path.dirname(os.path.abspath(__file__))
GRPO = os.path.join(SESSION, "colab_mcpo_driver_gsm8k_grpo.py")
EVALONLY = os.path.join(SESSION, "colab_mcpo_driver_gsm8k_grpo_evalonly.py")
HIDDEN = os.path.expanduser(
    "~/workspace/goals/arxiv-daily-read-and-test/hidden_files/mcpo-1111-redesign/grpo")
ADAPTER_ZIP = os.path.join(HIDDEN, "midrun_mcpo_gsm8k_adapter_grpo_mid.zip")
METRICS_BANKED = os.path.join(SESSION, "mcpo_gsm8k_metrics_grpo.csv")

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
# Config keys allowed to differ (arm identity + eval-only keys). MID_ADAPTER_DIR
# is GRPO-arm-only (the eval-only driver resumes from the uploaded zip).
ALLOWED_DIFF = {"ARM_NAME", "ARM_MCPO", "DONE_MARKER", "METRICS_CSV",
                "RESULTS_JSON", "ADAPTER_DIR", "ADAPTER_ZIP",
                "STEP12_ADAPTER_DIR", "BASE_EVAL_ACC_BANKED", "MID_ADAPTER_DIR"}

def extract_helpers(path):
    """Exec only the pure-helper defs + needed constants from a driver file
    (the driver itself cannot be imported on CPU: it loads a GPU model at
    module level). Returns (namespace, source_text)."""
    src = open(path).read()
    tree = ast.parse(src)
    ns = {"re": re, "math": math, "random": random}
    wanted_assigns = set(CONST_NAMES) | {"ARM_NAME", "ARM_MCPO", "DONE_MARKER",
                                          "METRICS_CSV", "RESULTS_JSON",
                                          "ADAPTER_DIR", "ADAPTER_ZIP",
                                          "STEP12_ADAPTER_DIR",
                                          "BASE_EVAL_ACC_BANKED"}
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
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if _target_names(node.targets) & {"ANSWER_RE", "GSM8K_GOLD_RE"}:
                parts.append(node)
    mod = ast.Module(body=parts, type_ignores=[])
    exec(compile(mod, path, "exec"), ns)
    return ns, src

g_ns, g_src = extract_helpers(GRPO)
e_ns, e_src = extract_helpers(EVALONLY)

# ---- 1. eval helpers identical to the GRPO driver ----
def fn_source(path, name):
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(open(path).read(), node)
    return None
for fn in HELPER_FNS:
    check(f"helper {fn} identical to GRPO driver", fn_source(GRPO, fn) == fn_source(EVALONLY, fn))

# ---- 2. config identity except arm-identity / eval-only keys ----
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
gc, ec = config_map(GRPO), config_map(EVALONLY)
diff_keys = {k for k in set(gc) | set(ec) if gc.get(k) != ec.get(k)}
check("configs differ ONLY in arm-identity / eval-only keys", diff_keys <= ALLOWED_DIFF, f"diff={sorted(diff_keys)}")
check("same seed/model/eval size", gc["SEED"] == ec["SEED"] == 20261111 and gc["MODEL"] == ec["MODEL"] == "Qwen/Qwen3-8B" and ec["N_EVAL"] == 80)
check("same LoRA config", e_ns["LORA_R"] == 16 and e_ns["LORA_ALPHA"] == 32)
check("banked base acc recorded", ec["BASE_EVAL_ACC_BANKED"] == 0.2375)
check("done marker distinct", ec["DONE_MARKER"] == "MCPO_GRPO_EVAL_DONE" and gc["DONE_MARKER"] == "MCPO_GRPO_DONE")

# ---- 3. pool selection: same eval set as the GRPO arm ----
sp_g, sp_e = g_ns["select_pools"], e_ns["select_pools"]
rng_rows = lambda n, tag: [{"prompt": f"{tag} prompt {i}", "gold": str(i)} for i in range(n)]
train_rows, test_rows = rng_rows(800, "train"), rng_rows(200, "test")
tp_g, ev_g, pp_g = sp_g(train_rows, test_rows)
tp_e, ev_e, pp_e = sp_e(train_rows, test_rows)
check("pool sizes 512/80/96", len(tp_e) == 512 and len(ev_e) == 80 and len(pp_e) == 96,
      f"{len(tp_e)}/{len(ev_e)}/{len(pp_e)}")
check("eval set identical to GRPO arm", ev_e == ev_g)
check("train pool identical to GRPO arm", tp_e == tp_g)
check("paired prompts identical to GRPO arm", pp_e == pp_g)

# ---- 4. answer extraction spot checks (same eval, same answers) ----
ea = e_ns["extract_answer"]; eq = e_ns["answers_equal"]
check("extract boxed", ea("work... \\boxed{42} tail") == "42")
check("extract Answer line", ea("reasoning\nAnswer: $72") == "72")
check("answers equal: commas", eq("72,000", "72000"))
check("answers unequal", not eq("72", "73"))
pg = e_ns["gsm8k_parse_gold"]
check("gsm8k gold basic", pg("... <<48/2=24>>24 clips.\n#### 72") == "72")

# ---- 5. banked step-12 adapter: present, valid, matches training config ----
check("adapter zip exists locally", os.path.isfile(ADAPTER_ZIP), ADAPTER_ZIP)
size = os.path.getsize(ADAPTER_ZIP) if os.path.isfile(ADAPTER_ZIP) else -1
check("adapter zip size 161,310,709 bytes", size == 161310709, f"{size}")
if os.path.isfile(ADAPTER_ZIP):
    with zipfile.ZipFile(ADAPTER_ZIP) as z:
        names = z.namelist()
        check("adapter zip valid (testzip clean)", z.testzip() is None)
        check("adapter zip has adapter_config.json", "adapter_config.json" in names)
        check("adapter zip has adapter_model.safetensors", "adapter_model.safetensors" in names)
        acfg = json.loads(z.read("adapter_config.json"))
        check("adapter config r=16 alpha=32", acfg.get("r") == 16 and acfg.get("lora_alpha") == 32,
              f"r={acfg.get('r')} alpha={acfg.get('lora_alpha')}")
        check("adapter config base is Qwen3-8B", "Qwen3-8B" in str(acfg.get("base_model_name_or_path", "")),
              str(acfg.get("base_model_name_or_path", "")))

# ---- 6. banked metrics: base + 24 train rows survived ----
check("banked metrics file exists", os.path.isfile(METRICS_BANKED))
if os.path.isfile(METRICS_BANKED):
    lines = open(METRICS_BANKED).read().strip().splitlines()
    check("banked metrics: header + base + 24 rows", len(lines) == 26, f"{len(lines)} lines")
    check("banked base eval 0.2375", lines[1].startswith("eval,base,-1,0.2375"), lines[1] if len(lines) > 1 else "")
    check("banked train rows 24", sum(1 for l in lines[1:] if l.startswith("train,grpo,")) == 24)

# ---- 7. eval-only driver safety / pitfalls ----
check("torchao uninstall", "pip uninstall -y torchao" in e_src)
check("expandable_segments", "expandable_segments:True" in e_src)
check("loads adapter via PeftModel.from_pretrained", "PeftModel.from_pretrained" in e_src)
check("fails loudly if adapter missing", "ADAPTER MISSING" in e_src and "sys.exit(4)" in e_src)
check("asserts adapter config r/alpha", 'cfg.get("r") != LORA_R' in e_src)
check("asserts adapter attached (peft_config)", "peft_config" in e_src and "ADAPTER NOT ATTACHED" in e_src)
check("no training loop", "train_arm" not in e_src and "opt.step()" not in e_src)
check("no optimizer", "AdamW" not in e_src)
check("outputs do not overwrite banked files",
      ec["METRICS_CSV"] != gc["METRICS_CSV"] and ec["RESULTS_JSON"] != gc["RESULTS_JSON"],
      f"{ec['METRICS_CSV']} / {ec['RESULTS_JSON']}")
check("re-saves step-12 adapter + zip", "STEP12_ADAPTER_DIR" in e_src and "make_archive" in e_src)
check("step-12 limit stated in docstring",
      "NOT the lost step-24 final" in e_src and "slightly understates" in e_src)
check("done marker is the last print", e_src.rstrip().endswith('print(DONE_MARKER, flush=True)'))
check("no TRAIN_DONE", "TRAIN_DONE" not in e_src)
check("HF token never in code", "HF_TOKEN" not in e_src and "huggingface_hub" not in e_src.lower().replace("huggingface.co", ""))
check("results note names the trade-off", '"step-12 post-training accuracy, NOT the lost step-24 final' in e_src)

print(f"\n{len(CHECKS)} checks passed")
