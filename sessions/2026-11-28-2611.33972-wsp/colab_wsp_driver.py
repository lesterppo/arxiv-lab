#!/usr/bin/env python3
"""Fine-Tuning Without Gradients? -- arXiv 2611.33972
("Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers
on a Frozen Encoder").

Paper claim: closed-form weight-space programming (WSP) on a frozen
RoBERTa-large encoder matches gradient fine-tuning. An LLM writes
programs that read the frozen encoder through cloze-style prompt
templates at several layers, fit closed-form heads (shrinkage
discriminant analysis, kernel ridge regression, and the encoder's
label-word prior) on labelled data, and select among candidate heads
by cross-fitting. No gradient is computed, no encoder weight changes.
On 11 dev datasets WSP is ahead of full fine-tuning (FT) and LoRA at
every label budget (0.3-0.6 points on average) and within 0.5 points
of prompt-based fine-tuning; with fewer labels its lead grows: 1.2-3.5
points at 64 and 128 labels, 3.8-5.3 points at 16 labels per class.
Paper numbers for our two datasets at 512 labels (Table 2):
  SST-2:  WSP 93.65 / FT 92.85 / LoRA 92.97
  AG News: WSP 90.10 / FT 90.11 / LoRA 89.63

This run (bounded scale, Colab T4):
  model: env MODEL_ID, default FacebookAI/roberta-large (355M, bf16,
    NOT quantized -- it fits on a T4). Fallback on load failure:
    FacebookAI/roberta-base, recorded in results.
  datasets: SST-2 (binary, HF glue/sst2) and AG News (4-class).
    Train pool: 2000 stratified examples per dataset (encoded once,
    frozen encoder -- the closed-form fits then reuse the features
    at every budget). Dev: 200 examples carved from the pool,
    excluded from train subsets. Eval: SST-2 validation (872),
    AG News test subsample (1000, seeded). If a dataset fails to
    load, record it and continue with the other -- never fake numbers.
  label budgets (stratified, seeded, total labels): 16, 64, 128, 512.
  templates + label words: the paper's round-1 programs (Appendix K,
    Table 27), copied verbatim -- 4 templates per dataset. The
    templates here are HUMAN-written from the paper (the paper's were
    LLM-written); that is a stated scope difference, not a claim
    that LLM authorship helps (the paper itself makes no such claim).
  views (frozen encoder, top 6 layers, not the paper's 12):
    mask        -- mean over templates of the <mask> hidden state
    plain_first -- mean over layers of the plain input's first token
    mask_concat -- concatenated <mask> states per template per layer
    (for the kernel head only). Prior: MLM label-word mean logit
    per class, averaged over the 4 templates.
  heads (pool, per dataset x budget): SDA on mask, SDA on
    plain_first, their average, RBF kernel ridge on mask_concat,
    the label-word prior, and prior+SDA mixture. Selection: 5-fold
    stratified cross-fitting on the train subset (+ dev rows),
    Occam's window (heads within 1 SE of the best are averaged).
    Final heads refit on all train data.
  arms per dataset x budget:
    (a) WSP closed-form (as above),
    (b) full fine-tuning of RoBERTa-large (lr 1e-5, batch 16,
        epochs {16:8, 64:6, 128:4, 512:3}),
    (c) LoRA fine-tuning (peft, r=8 on query/value, alpha=16,
        lr 1e-4, same epochs/batch).
    FT/LoRA use one fixed learning rate each (the paper uses a
    2-point dev-selected grid); single seed.
  The paper's discriminating prediction is at the LOW budgets
  (WSP beats FT/LoRA by 1.2-5.3 points at 16-128 labels).
  If every arm saturates or floors on a dataset, that dataset is
  INCONCLUSIVE (ceiling/floor), not evidence for or against.

Stated limits: 2 of the paper's 11 dev datasets; 4 templates (paper:
8 view + 16 prior templates); top 6 layers (paper: 12); one seed;
one fixed LR per trained arm; templates human-copied from the paper,
not LLM-written; no PFT arm (out of scope for one session); dev 200
(paper: 1024). If WSP does not beat FT/LoRA at low budgets here, the
verdict follows the numbers -- the paper's claim is then untested
at full recipe scale, not refuted.

Outputs (written to /content, flushed after every phase):
  /content/wsp_metrics.csv  - per-phase rows (encode / wsp / ft / lora)
  /content/wsp_results.json - summary + config + per-arm accuracies
Prints WSP_DONE as its last line on completion.

Attempt 1 (2026-11-28, session wsp-1128, cyc236es): BLOCKED before
any arm -- datasets>=3 no longer resolves the legacy script ids
"glue"/"sst2"/"ag_news", and the fancyzhx/ag_news parquet mirror
has a single 'text' column (no 'title'). Fixed in this driver:
canonical nyu-mll/glue for SST-2, fancyzhx/ag_news first for
AG News, and a schema-robust ag_texts() builder. Attempt-1
artifacts are kept as wsp_results_attempt1.json /
wsp_metrics_attempt1.csv / train-run-1128a.json in the session dir.

Follow-up (2026-11-28, session wsp-1128c, cyc236es): the FT/LoRA
arms of attempt 2 collapsed on a bf16 optimizer bug (see the
fp32 note in train_classifier). This run re-runs ONLY those 16
arms (WSP_ONLY_ARMS=ft,lora) with the classifier loaded in fp32;
WSP arms are skipped, their banked numbers are not re-run.
"""
import os, sys, json, math, re, csv, random, time, subprocess, gc

# ---------------- install (self-contained) ----------------
def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"INSTALL FAILED rc={r.returncode}: {cmd}", flush=True)
        sys.exit(3)

MODEL_ID = os.environ.get("MODEL_ID", "FacebookAI/roberta-large")
FALLBACK_MODEL_ID = "FacebookAI/roberta-base"
sh("pip install -q 'transformers==5.20.2' bitsandbytes accelerate peft datasets")
sh("pip uninstall -y torchao")  # Colab ships torchao 0.10; peft needs > 0.16

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn.functional as F
from transformers import (AutoModelForMaskedLM,
                          AutoModelForSequenceClassification, AutoTokenizer)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ---------------- config ----------------
SEED = 20261128
BUDGETS = [16, 64, 128, 512]
POOL_N = 2000
DEV_N = 200
EVAL_AG_N = 1000
MAX_LEN = 128
ENC_BATCH = 32
FT_BATCH = 16
TOP_LAYERS = 6
N_TEMPLATES = 4
EPOCHS_BY_BUDGET = {16: 8, 64: 6, 128: 4, 512: 3}
FT_LR = 1e-5
LORA_LR = 1e-4
TIME_GATE_MIN = 90    # past this, drop remaining FT/LoRA arms
HARD_STOP_MIN = 140   # skip remaining arms past this (record partial)
METRICS_CSV = "/content/wsp_metrics.csv"
RESULTS_JSON = "/content/wsp_results.json"
T0 = time.time()

torch.manual_seed(SEED); random.seed(SEED)
DEVICE = "cuda"
print(f"CUDA: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)
print(f"model={MODEL_ID} fallback={FALLBACK_MODEL_ID}", flush=True)

# ---------------- PURE HELPERS (stdlib + numpy) ----------------
# Self-contained: this block is extracted and exec'd by
# tests/test_wsp_cpu.py, so every name it uses must be defined here.
import math as _math
import re as _re
import random as _random

N_TEMPLATES = 4

# Paper Appendix K, Table 27 (round-1 programs). "{x}" is the input
# text placeholder; "<mask>" is the RoBERTa mask token.
TEMPLATES = {
    "sst2": [
        "{x} It was <mask> .",
        "{x} All in all , it was <mask> .",
        "{x} In summary , the film is <mask> .",
        "{x} Overall , I thought it was <mask> .",
    ],
    "agnews": [
        "{x} This article is about <mask> .",
        "<mask> news : {x}",
        "{x} The topic is <mask> .",
        "{x} Category : <mask> .",
    ],
}
# Label words, one group per class, in dataset label order.
# SST-2: 0=negative, 1=positive. AG News: 0=World, 1=Sports,
# 2=Business, 3=Sci/Tech.
LABEL_WORDS = {
    "sst2": [
        ["terrible", "bad", "awful", "boring", "poor", "horrible",
         "dull", "disappointing"],
        ["great", "good", "wonderful", "excellent", "fantastic",
         "amazing", "fun", "enjoyable"],
    ],
    "agnews": [
        ["world", "politics", "international", "war"],
        ["sports", "football", "baseball", "soccer"],
        ["business", "economy", "finance", "markets"],
        ["technology", "science", "computers", "internet"],
    ],
}


def build_template(template, text):
    """Fill one cloze template. Exactly one {x} and one <mask>."""
    if template.count("{x}") != 1:
        raise ValueError("template must contain exactly one {x}: %r" % template)
    out = template.replace("{x}", text)
    if out.count("<mask>") != 1:
        raise ValueError("template must contain exactly one <mask>: %r" % template)
    return out


def check_template_set(dataset):
    """Validate a dataset's program: 4 templates, each with exactly
    one {x} and one <mask>; label words are single words."""
    ts = TEMPLATES[dataset]
    assert len(ts) == N_TEMPLATES, (dataset, len(ts))
    for t in ts:
        assert t.count("{x}") == 1 and t.count("<mask>") == 1, t
    for group in LABEL_WORDS[dataset]:
        for w in group:
            assert " " not in w.strip() and w.strip(), w
    return len(ts)


def softmax_np(x, axis=-1):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def label_word_scores(mask_logits, label_word_ids):
    """Per-class mean MLM logit at <mask>.

    mask_logits: (V,) or (n, V) numpy array of MLM logits.
    label_word_ids: list per class of token ids (single-token words).
    Returns (C,) or (n, C) mean logits per class."""
    import numpy as np
    lg = np.asarray(mask_logits, dtype=np.float64)
    single = lg.ndim == 1
    if single:
        lg = lg[None, :]
    out = np.zeros((lg.shape[0], len(label_word_ids)), dtype=np.float64)
    for c, ids in enumerate(label_word_ids):
        if not ids:
            raise ValueError("empty label-word group for class %d" % c)
        out[:, c] = lg[:, list(ids)].mean(axis=1)
    return out[0] if single else out


def prior_probs(mask_logits, label_word_ids):
    """Label-word prior: softmax over per-class mean logits."""
    return softmax_np(label_word_scores(mask_logits, label_word_ids))


def accuracy(preds, golds):
    """Fraction correct; empty input -> 0.0 (never NaN)."""
    preds = list(preds); golds = list(golds)
    if not golds:
        return 0.0
    return sum(int(p == g) for p, g in zip(preds, golds)) / len(golds)


def stratified_subsample(labels, n_total, seed):
    """Seeded stratified subsample indices: per-class counts as even
    as possible (differ by at most 1), shuffled within class."""
    rng = _random.Random(seed)
    labels = list(labels)
    n_classes = len(set(labels))
    if n_total > len(labels):
        raise ValueError("n_total %d > pool %d" % (n_total, len(labels)))
    base = n_total // n_classes
    rem = n_total % n_classes
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    out = []
    for k, cls in enumerate(sorted(by_class)):
        idxs = by_class[cls][:]
        rng.shuffle(idxs)
        take = base + (1 if k < rem else 0)
        take = min(take, len(idxs))
        out.extend(idxs[:take])
    rng.shuffle(out)
    return out


def stratified_kfold(labels, n_splits, seed):
    """Stratified K-fold splits: list of (train_idx, test_idx).
    n_splits is capped at the smallest class count."""
    rng = _random.Random(seed)
    labels = list(labels)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    min_count = min(len(v) for v in by_class.values())
    k = max(2, min(n_splits, min_count))
    folds = [[] for _ in range(k)]
    for cls in sorted(by_class):
        idxs = by_class[cls][:]
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % k].append(idx)
    splits = []
    for f in range(k):
        test = sorted(folds[f])
        train = sorted(i for g in range(k) if g != f for i in folds[g])
        splits.append((train, test))
    return splits


def sda_fit(X, y, alpha=0.1):
    """Shrinkage LDA: pooled within-class covariance shrunk toward
    its diagonal, plus a tiny ridge for rank-deficient fits.
    Returns a model dict."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    n, d = X.shape
    means = np.zeros((len(classes), d))
    priors = np.zeros(len(classes))
    S = np.zeros((d, d))
    for ci, c in enumerate(classes):
        Xc = X[y == c]
        means[ci] = Xc.mean(axis=0)
        priors[ci] = len(Xc) / n
        Xd = Xc - means[ci]
        S += Xd.T @ Xd
    S /= max(n - len(classes), 1)
    diag = np.diag(np.diag(S))
    S = (1.0 - alpha) * S + alpha * diag
    S += 1e-4 * (np.trace(S) / max(d, 1) + 1e-8) * np.eye(d)
    inv = np.linalg.pinv(S)
    return {"classes": classes, "means": means, "inv": inv,
            "priors": priors, "alpha": alpha}


def sda_predict_proba(model, X):
    """Gaussian log-posteriors (shared covariance) -> probs."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    means, inv = model["means"], model["inv"]
    # -0.5 (x-mu)' inv (x-mu) + log prior, per class
    Xm = X @ inv  # (n, d)
    cross = Xm @ means.T  # (n, C)
    mu_quad = np.einsum("ij,ij->i", means @ inv, means)
    logp = cross - 0.5 * mu_quad[None, :] + np.log(model["priors"])[None, :]
    # The x' inv x term is identical for every class, so it cancels
    # in the softmax and is omitted.
    return softmax_np(logp)


def krr_fit(X, y, alpha=1.0, gamma=None):
    """Exact RBF kernel ridge, one-vs-rest in dual form.
    Standardisation is fit on the training rows only."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    if gamma is None:
        gamma = 1.0 / Xs.shape[1]
    sq = (Xs ** 2).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)
    np.maximum(D2, 0.0, out=D2)
    K = np.exp(-gamma * D2)
    Y = np.zeros((len(y), len(classes)))
    for ci, c in enumerate(classes):
        Y[y == c, ci] = 1.0
    A = np.linalg.solve(K + alpha * np.eye(len(y)), Y)
    return {"classes": classes, "mu": mu, "sd": sd, "gamma": gamma,
            "alpha": alpha, "X_train": Xs, "coef": A}


def krr_predict_proba(model, X):
    """RBF kernel against the stored training rows -> softmax probs."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - model["mu"]) / model["sd"]
    Xt = model["X_train"]
    sq_te = (Xs ** 2).sum(axis=1)
    sq_tr = (Xt ** 2).sum(axis=1)
    D2 = sq_te[:, None] + sq_tr[None, :] - 2.0 * (Xs @ Xt.T)
    np.maximum(D2, 0.0, out=D2)
    K = np.exp(-model["gamma"] * D2)
    scores = K @ model["coef"]
    return softmax_np(scores)


def crossfit_select(heads, feats_tr, y_tr, feats_te, y_te, seed=0,
                    n_splits=5):
    """Select heads by stratified cross-fitting on train (+ dev rows).

    heads: dict name -> callable (feats_tr_dict, y_tr, feats_te_dict)
        returning (n_te, C) predicted probabilities.
    feats_tr / feats_te: dicts of feature arrays keyed by view name.
    Returns (selected_names, scores, best_name): scores are the
    accuracy on cross-fitted train rows + dev rows; selected is the
    Occam window -- every head within 1 SE of the best, where SE is
    the binomial SE of the best head's score."""
    import numpy as np
    y_tr = np.asarray(y_tr)
    y_te = np.asarray(y_te)
    splits = stratified_kfold(y_tr.tolist(), n_splits, seed)
    scores = {}
    for name, head in heads.items():
        cv_pred = np.zeros(len(y_tr), dtype=int)
        for tr_idx, te_idx in splits:
            sub_tr = {k: v[tr_idx] for k, v in feats_tr.items()}
            sub_te = {k: v[te_idx] for k, v in feats_tr.items()}
            p = head(sub_tr, y_tr[tr_idx], sub_te)
            cv_pred[te_idx] = np.argmax(p, axis=1)
        dev_pred = np.argmax(head(feats_tr, y_tr, feats_te), axis=1)
        all_pred = np.concatenate([cv_pred, dev_pred])
        all_gold = np.concatenate([y_tr, y_te])
        scores[name] = accuracy(all_pred.tolist(), all_gold.tolist())
    best = max(scores, key=lambda k: scores[k])
    n_rows = len(y_tr) + len(y_te)
    p_best = scores[best]
    se = _math.sqrt(max(p_best * (1.0 - p_best), 0.0) / max(n_rows, 1))
    selected = [n for n, s in scores.items() if s >= p_best - se]
    return selected, scores, best


def fit_selected_probs(heads, selected, feats_tr, y_tr, feats_te):
    """Mean predicted probabilities of the selected heads, each fit
    on all train rows."""
    import numpy as np
    probs = [heads[n](feats_tr, y_tr, feats_te) for n in selected]
    return np.mean(np.stack(probs, axis=0), axis=0)


def arm_table(arms):
    """Validate the arm list: distinct names, dataset in (sst2,
    agnews), budget in (16, 64, 128, 512), arm in (wsp, ft, lora)."""
    seen = set()
    for name, dataset, budget, arm in arms:
        assert name not in seen, "duplicate arm %s" % name
        seen.add(name)
        assert dataset in ("sst2", "agnews"), name
        assert budget in (16, 64, 128, 512), name
        assert arm in ("wsp", "ft", "lora"), name
    return len(arms)
# ---------------- END PURE HELPERS ----------------

# ---------------- metrics ----------------
metrics_f = open(METRICS_CSV, "w", newline="")
mw = csv.writer(metrics_f)
mw.writerow(["phase", "dataset", "budget", "arm", "step", "loss",
             "metric", "value", "n", "note"])
metrics_f.flush()

def log_row(*vals):
    mw.writerow(list(vals)); metrics_f.flush()
    print("METRIC", vals, flush=True)

results = {"paper": "arXiv:2611.33972", "seed": SEED,
           "model_requested": MODEL_ID,
           "config": {"budgets": BUDGETS, "pool_n": POOL_N,
                      "dev_n": DEV_N, "eval_ag_n": EVAL_AG_N,
                      "max_len": MAX_LEN, "top_layers": TOP_LAYERS,
                      "n_templates": N_TEMPLATES,
                      "epochs_by_budget": EPOCHS_BY_BUDGET,
                      "ft_lr": FT_LR, "lora_lr": LORA_LR,
                      "templates": TEMPLATES,
                      "label_words": LABEL_WORDS},
           "datasets": {}, "arms": {}, "means": {}}

def save_results():
    results["elapsed_min"] = round((time.time() - T0) / 60, 1)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=1)

# ---------------- data ----------------
def load_sst2():
    # datasets>=3 removed script-based ids ("glue", "sst2"): the
    # canonical repo is nyu-mll/glue. Try it first, then the legacy
    # ids, then stanfordnlp/sst2.
    for args in (("nyu-mll/glue", "sst2"), ("glue", "sst2"),
                 ("stanfordnlp/sst2",)):
        try:
            return load_dataset(*args)
        except Exception as e:
            print(f"sst2 load {args} failed: {e}", flush=True)
    return None


def load_agnews():
    # fancyzhx/ag_news is the parquet mirror that loads on
    # datasets>=3; the legacy "ag_news" script id no longer resolves.
    for args in (("fancyzhx/ag_news",), ("ag_news",)):
        try:
            return load_dataset(*args)
        except Exception as e:
            print(f"ag_news load {args} failed: {e}", flush=True)
    return None


def ag_texts(split):
    """AG News text, robust to the mirror's schema: the old script
    version had title+description columns; the parquet mirror has a
    single 'text' column (title and description already joined)."""
    cols = split.column_names
    if "title" in cols and "text" in cols:
        return [t + " " + d for t, d in zip(split["title"], split["text"])]
    if "title" in cols and "description" in cols:
        return [t + " " + d
                for t, d in zip(split["title"], split["description"])]
    if "text" in cols:
        return [t for t in split["text"]]
    raise ValueError("ag_news split has no usable text column: %s" % cols)

def pool_split(texts, labels, seed):
    """Stratified pool (POOL_N) + dev (DEV_N) carved from it.
    Returns (pool_texts, pool_labels, dev_texts, dev_labels)."""
    rng = random.Random(seed)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    pool_idx, dev_idx = [], []
    for cls in sorted(by_class):
        idxs = by_class[cls][:]
        rng.shuffle(idxs)
        n_pool = max(1, round(POOL_N * len(idxs) / len(labels)))
        pool_idx.extend(idxs[:n_pool])
        rest = idxs[n_pool:]
        n_dev = max(1, round(DEV_N * len(idxs) / len(labels)))
        dev_idx.extend(rest[:n_dev])
    rng.shuffle(pool_idx); rng.shuffle(dev_idx)
    return ([texts[i] for i in pool_idx], [labels[i] for i in pool_idx],
            [texts[i] for i in dev_idx], [labels[i] for i in dev_idx])

DATASETS = {}
sst2 = load_sst2()
if sst2 is not None:
    try:
        tr = sst2["train"]; va = sst2["validation"]
        texts = [t for t in tr["sentence"]]
        labels = [int(y) for y in tr["label"]]
        p_t, p_y, d_t, d_y = pool_split(texts, labels, SEED)
        ev_t = [t for t in va["sentence"]]
        ev_y = [int(y) for y in va["label"]]
        DATASETS["sst2"] = {"pool_texts": p_t, "pool_labels": p_y,
                            "dev_texts": d_t, "dev_labels": d_y,
                            "eval_texts": ev_t, "eval_labels": ev_y,
                            "n_classes": 2}
        print(f"sst2: pool={len(p_t)} dev={len(d_t)} eval={len(ev_t)}", flush=True)
    except Exception as e:
        print(f"sst2 split failed: {e}", flush=True)
ag = load_agnews()
if ag is not None:
    try:
        tr = ag["train"]; te = ag["test"]
        texts = ag_texts(tr)
        labels = [int(y) for y in tr["label"]]
        p_t, p_y, d_t, d_y = pool_split(texts, labels, SEED)
        rng = random.Random(SEED)
        eidx = list(range(len(te))); rng.shuffle(eidx)
        eidx = eidx[:EVAL_AG_N]
        te_texts = ag_texts(te)
        ev_t = [te_texts[i] for i in eidx]
        ev_y = [int(te["label"][i]) for i in eidx]
        DATASETS["agnews"] = {"pool_texts": p_t, "pool_labels": p_y,
                              "dev_texts": d_t, "dev_labels": d_y,
                              "eval_texts": ev_t, "eval_labels": ev_y,
                              "n_classes": 4}
        print(f"agnews: pool={len(p_t)} dev={len(d_t)} eval={len(ev_t)}", flush=True)
    except Exception as e:
        print(f"agnews split failed: {e}", flush=True)

if not DATASETS:
    print("NO DATASET LOADED -- recording BLOCKED and exiting", flush=True)
    results["datasets"] = {}
    results["blocked"] = "no dataset loaded (sst2 + agnews both failed)"
    save_results()
    print("WSP_DONE", flush=True)
    sys.exit(0)

results["datasets"] = {k: {"pool": len(v["pool_texts"]),
                           "dev": len(v["dev_texts"]),
                           "eval": len(v["eval_texts"]),
                           "n_classes": v["n_classes"]}
                       for k, v in DATASETS.items()}
save_results()

# ---------------- model (frozen encoder) ----------------
tok = None
enc_model = None
MODEL_ACTUAL = MODEL_ID
try:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    enc_model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    enc_model.eval()
    print(f"encoder loaded: {MODEL_ID}", flush=True)
except Exception as e:
    print(f"encoder load failed for {MODEL_ID}: {e}", flush=True)
    print(f"falling back to {FALLBACK_MODEL_ID}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID)
        enc_model = AutoModelForMaskedLM.from_pretrained(
            FALLBACK_MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
        enc_model.eval()
        MODEL_ACTUAL = FALLBACK_MODEL_ID
        print(f"encoder loaded (fallback): {FALLBACK_MODEL_ID}", flush=True)
    except Exception as e2:
        print(f"fallback encoder load failed: {e2}", flush=True)
        results["blocked"] = ("encoder load failed for both %s and %s: %s"
                              % (MODEL_ID, FALLBACK_MODEL_ID, e2))
        save_results()
        print("WSP_DONE", flush=True)
        sys.exit(0)

results["model_actual"] = MODEL_ACTUAL
MASK_ID = tok.mask_token_id
if MASK_ID is None:
    print("tokenizer has no mask token -- BLOCKED", flush=True)
    results["blocked"] = "tokenizer has no mask token"
    save_results()
    print("WSP_DONE", flush=True)
    sys.exit(0)

# Label-word token ids (single-token words only; multi-token words
# are dropped at run time, as in the paper).
def label_word_ids(dataset):
    out = []
    dropped = []
    for group in LABEL_WORDS[dataset]:
        ids = []
        for w in group:
            toks = tok.encode(" " + w, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
            else:
                dropped.append(w)
        out.append(ids)
    if dropped:
        print(f"{dataset}: dropped multi-token label words: {dropped}", flush=True)
    return out

LW_IDS = {d: label_word_ids(d) for d in DATASETS}
results["label_word_ids"] = LW_IDS
results["label_word_dropped"] = {}
save_results()

# ---------------- encoding (frozen, once per dataset) ----------------
@torch.no_grad()
def encode_split(dataset, texts):
    """One encoder pass per template (+ plain input) per text.

    Returns dict of float16 numpy arrays:
      mask        (n, 1024)  mean over templates+top layers, <mask> state
      plain_first (n, 1024)  mean over top layers, first-token state
      mask_concat (n, 4*6*1024) concatenated <mask> states
      prior       (n, C)     label-word prior probs (mean over templates)
    """
    import numpy as np
    templates = TEMPLATES[dataset]
    lw_ids = LW_IDS[dataset]
    n = len(texts)
    mask_acc = np.zeros((n, 1024), dtype=np.float64)
    plain_acc = np.zeros((n, 1024), dtype=np.float64)
    concat = np.zeros((n, N_TEMPLATES * TOP_LAYERS * 1024), dtype=np.float16)
    prior_acc = np.zeros((n, len(lw_ids)), dtype=np.float64)
    # plain input first
    for start in range(0, n, ENC_BATCH):
        batch = texts[start:start + ENC_BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        out = enc_model(**enc, output_hidden_states=True)
        hs = out.hidden_states[-TOP_LAYERS:]  # 6 x (B, T, 1024)
        stacked = torch.stack(hs, dim=0).float()  # (6, B, T, 1024)
        plain_acc[start:start + len(batch)] = (
            stacked[:, :, 0, :].mean(dim=0).cpu().numpy())
        del enc, out, hs, stacked
        torch.cuda.empty_cache()
    # templates: mask state + MLM prior
    for ti, tmpl in enumerate(templates):
        wrapped = [build_template(tmpl, t) for t in texts]
        for start in range(0, n, ENC_BATCH):
            batch = wrapped[start:start + ENC_BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(DEVICE)
            mask_pos = (enc["input_ids"] == MASK_ID).nonzero(as_tuple=False)
            out = enc_model(**enc, output_hidden_states=True)
            hs = out.hidden_states[-TOP_LAYERS:]
            stacked = torch.stack(hs, dim=0).float()
            B = len(batch)
            # mask state per layer: gather at mask positions
            for b in range(B):
                rows = mask_pos[mask_pos[:, 0] == b]
                if len(rows) == 0:
                    continue
                pos = int(rows[0, 1].item())
                vec = stacked[:, b, pos, :]  # (6, 1024)
                mask_acc[start + b] += vec.mean(dim=0).cpu().numpy() / N_TEMPLATES
                concat[start + b, ti * TOP_LAYERS * 1024:
                       (ti + 1) * TOP_LAYERS * 1024] = (
                    vec.reshape(-1).cpu().numpy().astype(np.float16))
                logits = out.logits[b, pos, :].float().cpu().numpy()
                prior_acc[start + b] += (
                    label_word_scores(logits, lw_ids) / N_TEMPLATES)
            del enc, out, hs, stacked
            torch.cuda.empty_cache()
    prior = softmax_np(prior_acc)
    return {"mask": mask_acc.astype(np.float16),
            "plain_first": plain_acc.astype(np.float16),
            "mask_concat": concat,
            "prior": prior.astype(np.float16)}

FEATURES = {}
for dataset, d in DATASETS.items():
    print(f"== encode {dataset} ==", flush=True)
    feats = {}
    for split in ("pool", "dev", "eval"):
        texts = d["%s_texts" % split]
        t_enc = time.time()
        feats[split] = encode_split(dataset, texts)
        dt = time.time() - t_enc
        print(f"encoded {dataset}/{split}: n={len(texts)} in {dt:.1f}s",
              flush=True)
        log_row("encode", dataset, "", "", "", "", "encode_seconds",
                round(dt, 1), len(texts), split)
    FEATURES[dataset] = feats
    save_results()

# Free the encoder before gradient training (memory headroom).
del enc_model
gc.collect(); torch.cuda.empty_cache()
print("encoder freed; features cached in RAM", flush=True)

# ---------------- WSP fit ----------------
def make_heads():
    """Head closures over the feature dict.

    Each head: (feats_tr, y_tr, feats_te) -> (n_te, C) probs.
    Features are float16 on entry; heads cast to float64."""
    import numpy as np

    def sda_head(view):
        def head(ftr, ytr, fte):
            m = sda_fit(np.asarray(ftr[view], dtype=np.float64), ytr, alpha=0.1)
            return sda_predict_proba(m, np.asarray(fte[view], dtype=np.float64))
        return head

    def sda_avg_head(ftr, ytr, fte):
        ps = []
        for view in ("mask", "plain_first"):
            m = sda_fit(np.asarray(ftr[view], dtype=np.float64), ytr, alpha=0.1)
            ps.append(sda_predict_proba(m, np.asarray(fte[view], dtype=np.float64)))
        return np.mean(np.stack(ps, axis=0), axis=0)

    def krr_head(ftr, ytr, fte):
        m = krr_fit(np.asarray(ftr["mask_concat"], dtype=np.float64), ytr,
                    alpha=1.0)
        return krr_predict_proba(m, np.asarray(fte["mask_concat"], dtype=np.float64))

    def prior_head(ftr, ytr, fte):
        return np.asarray(fte["prior"], dtype=np.float64)

    def prior_sda_head(ftr, ytr, fte):
        m = sda_fit(np.asarray(ftr["mask"], dtype=np.float64), ytr, alpha=0.1)
        sda = sda_predict_proba(m, np.asarray(fte["mask"], dtype=np.float64))
        return 0.5 * sda + 0.5 * np.asarray(fte["prior"], dtype=np.float64)

    return {"sda_mask": sda_head("mask"),
            "sda_plain": sda_head("plain_first"),
            "sda_avg": sda_avg_head,
            "krr_rbf": krr_head,
            "prior": prior_head,
            "prior_sda": prior_sda_head}


def run_wsp(dataset, budget):
    """Closed-form WSP for one dataset x budget. Returns eval
    accuracy + selection trace."""
    import numpy as np
    d = DATASETS[dataset]
    feats = FEATURES[dataset]
    y_pool = np.asarray(d["pool_labels"])
    idx = stratified_subsample(y_pool.tolist(), budget, SEED + budget)
    idx = np.asarray(idx)
    y_tr = y_pool[idx]
    ftr = {k: v[idx] for k, v in feats["pool"].items()}
    fdev = feats["dev"]
    y_dev = np.asarray(d["dev_labels"])
    fev = feats["eval"]
    y_ev = np.asarray(d["eval_labels"])
    heads = make_heads()
    selected, scores, best = crossfit_select(
        heads, ftr, y_tr, fdev, y_dev, seed=SEED)
    probs = fit_selected_probs(heads, selected, ftr, y_tr, fev)
    pred = np.argmax(probs, axis=1)
    acc = accuracy(pred.tolist(), y_ev.tolist())
    # also score the single best head alone, for the trace
    probs_best = heads[best](ftr, y_tr, fev)
    acc_best = accuracy(np.argmax(probs_best, axis=1).tolist(),
                        y_ev.tolist())
    print(f"WSP {dataset} budget={budget}: acc={acc:.4f} "
          f"(best-head alone {acc_best:.4f}) selected={selected} "
          f"scores={ {k: round(v, 3) for k, v in scores.items()} }",
          flush=True)
    log_row("eval", dataset, budget, "wsp", "", "", "accuracy",
            round(acc, 4), len(y_ev), "selected=" + ",".join(selected))
    return {"accuracy": round(acc, 4), "n_eval": len(y_ev),
            "selected": selected, "best_head": best,
            "cv_scores": {k: round(v, 4) for k, v in scores.items()},
            "best_head_alone_acc": round(acc_best, 4),
            "n_train": int(len(idx))}

# ---------------- FT / LoRA arms ----------------
def train_classifier(dataset, budget, arm):
    """Gradient arm: full FT (arm='ft') or LoRA (arm='lora').
    Returns eval accuracy + dev accuracy + collapse flag."""
    import numpy as np
    d = DATASETS[dataset]
    y_pool = np.asarray(d["pool_labels"])
    idx = stratified_subsample(y_pool.tolist(), budget, SEED + budget)
    train_texts = [d["pool_texts"][i] for i in idx]
    train_labels = [int(y_pool[i]) for i in idx]
    dev_texts, dev_labels = d["dev_texts"], d["dev_labels"]
    eval_texts, eval_labels = d["eval_texts"], d["eval_labels"]
    n_classes = d["n_classes"]
    epochs = EPOCHS_BY_BUDGET[budget]
    lr = FT_LR if arm == "ft" else LORA_LR

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ACTUAL, num_labels=n_classes,
        torch_dtype=torch.float32).to(DEVICE)
    # fp32, NOT bf16: the 2026-11-28 run loaded this classifier in
    # pure bf16 with a raw AdamW loop and no fp32 master weights, so
    # lr-1e-5 updates to ~0.01-magnitude weights rounded to zero and
    # the weights never moved (loss flat at ln 2 / ln 4, all arms
    # collapsed to majority rate). fp32 weights + fp32 AdamW states
    # make the updates representable; the frozen encoder above stays
    # bf16, only this gradient-trained classifier is fp32.
    if arm == "lora":
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                         target_modules=["query", "value"],
                         task_type="SEQ_CLS")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    tok.padding_side = "right"
    encoded = [tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
               for t in train_texts]
    order = list(range(len(encoded)))
    arm_rng = random.Random(SEED)
    step = 0
    losses = []
    for epoch in range(epochs):
        arm_rng.shuffle(order)
        for ms in range(0, len(order), FT_BATCH):
            bidx = order[ms:ms + FT_BATCH]
            bids = [encoded[i] for i in bidx]
            maxlen = max(len(x) for x in bids)
            pad = tok.pad_token_id
            input_ids = torch.tensor(
                [x + [pad] * (maxlen - len(x)) for x in bids], device=DEVICE)
            attn = torch.tensor(
                [[1] * len(x) + [0] * (maxlen - len(x)) for x in bids],
                device=DEVICE)
            labs = torch.tensor([train_labels[i] for i in bidx], device=DEVICE)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labs)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            lr_now = lr * min(1.0, (step + 1) / 3.0)
            for g in opt.param_groups:
                g["lr"] = lr_now
            opt.step()
            opt.zero_grad()
            step += 1
            losses.append(loss.item())
            log_row("train", dataset, budget, arm, step,
                    round(loss.item(), 4), "", "", "", "")
            torch.cuda.empty_cache()
    print(f"TRAIN {dataset}/{arm}/b{budget} done: {step} steps, "
          f"mean loss {sum(losses)/len(losses):.4f}", flush=True)

    @torch.no_grad()
    def predict(texts):
        model.eval()
        tok.padding_side = "left"
        preds = []
        for i in range(0, len(texts), FT_BATCH * 2):
            batch = texts[i:i + FT_BATCH * 2]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(DEVICE)
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            del enc, logits
            torch.cuda.empty_cache()
        model.train()
        return preds

    dev_pred = predict(dev_texts)
    dev_acc = accuracy(dev_pred, dev_labels)
    ev_pred = predict(eval_texts)
    ev_acc = accuracy(ev_pred, eval_labels)
    # collapse rule (paper): best dev accuracy within 1pt of majority rate
    from collections import Counter
    maj = max(Counter(dev_labels).values()) / len(dev_labels)
    collapsed = bool(dev_acc <= maj + 0.01)
    print(f"EVAL {dataset}/{arm}/b{budget}: eval_acc={ev_acc:.4f} "
          f"dev_acc={dev_acc:.4f} majority={maj:.4f} "
          f"collapsed={collapsed}", flush=True)
    log_row("eval", dataset, budget, arm, "", "", "accuracy",
            round(ev_acc, 4), len(eval_labels),
            "dev=%.4f collapsed=%s" % (dev_acc, collapsed))
    del model
    gc.collect(); torch.cuda.empty_cache()
    return {"accuracy": round(ev_acc, 4), "n_eval": len(eval_labels),
            "dev_accuracy": round(dev_acc, 4),
            "majority_rate": round(maj, 4), "collapsed": collapsed,
            "steps": step,
            "mean_loss": round(sum(losses) / len(losses), 4),
            "n_train": int(len(idx))}

# ---------------- run ----------------
# WSP_ONLY_ARMS (env, comma-separated, e.g. "ft,lora"): run only those
# arms. Default is the full arm list. The 2026-11-28 follow-up run
# sets WSP_ONLY_ARMS=ft,lora to re-run only the gradient arms with
# the fp32 fix; the WSP numbers are already banked and must not be
# re-run.
_only = os.environ.get("WSP_ONLY_ARMS", "")
_only_set = {a.strip() for a in _only.split(",") if a.strip()}
ARMS = []
for dataset in DATASETS:
    for budget in BUDGETS:
        for arm in ("wsp", "ft", "lora"):
            if _only_set and arm not in _only_set:
                continue
            ARMS.append(("%s_b%d_%s" % (dataset, budget, arm),
                         dataset, budget, arm))
arm_table(ARMS)
results["arms"] = {}
save_results()

for name, dataset, budget, arm in ARMS:
    elapsed = (time.time() - T0) / 60
    if elapsed > HARD_STOP_MIN:
        print(f"HARD STOP: elapsed {elapsed:.1f} min > {HARD_STOP_MIN}, "
              f"skipping {name} and later arms", flush=True)
        results["arms"][name] = {"accuracy": None, "skipped": "hard-stop time gate"}
        continue
    if elapsed > TIME_GATE_MIN and arm in ("ft", "lora"):
        print(f"TIME-GATE: elapsed {elapsed:.1f} min > {TIME_GATE_MIN}, "
              f"skipping {name}", flush=True)
        results["arms"][name] = {"accuracy": None, "skipped": "time-gate (ft/lora)"}
        continue
    print(f"== arm {name} ==", flush=True)
    try:
        if arm == "wsp":
            res = run_wsp(dataset, budget)
        else:
            res = train_classifier(dataset, budget, arm)
        results["arms"][name] = res
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ARM FAILED {name}: {type(e).__name__}: {e}", flush=True)
        results["arms"][name] = {"accuracy": None,
                                 "failed": "%s: %s" % (type(e).__name__, e)}
    save_results()

# ---------------- means / contrasts ----------------
def mean_acc(arm, budget):
    vals = []
    for dataset in DATASETS:
        r = results["arms"].get("%s_b%d_%s" % (dataset, budget, arm), {})
        if r.get("accuracy") is not None:
            vals.append(r["accuracy"])
    return round(sum(vals) / len(vals), 4) if vals else None

for budget in BUDGETS:
    row = {}
    for arm in ("wsp", "ft", "lora"):
        row[arm] = mean_acc(arm, budget)
    if row["wsp"] is not None and row["ft"] is not None:
        row["wsp_minus_ft"] = round(row["wsp"] - row["ft"], 4)
    if row["wsp"] is not None and row["lora"] is not None:
        row["wsp_minus_lora"] = round(row["wsp"] - row["lora"], 4)
    results["means"]["budget_%d" % budget] = row
    print(f"MEAN budget={budget}: {row}", flush=True)

save_results()
metrics_f.close()
print("RESULTS", json.dumps(results, indent=1), flush=True)
print("WSP_DONE", flush=True)
