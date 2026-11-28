#!/usr/bin/env python3
"""CPU test for the WSP driver's PURE HELPERS (arXiv 2611.33972).

Extracts the stdlib+numpy helper block from
sessions/2026-11-28-2611.33972-wsp/colab_wsp_driver.py and exercises
it without torch: cloze template building, label-word scoring,
shrinkage-LDA and kernel-ridge closed-form fits on synthetic frozen
features, cross-fitting head selection, accuracy scoring, and
stratified label-budget subsampling.

E1  Templates: exactly one {x} and one <mask>; paper programs valid.
E2  Label-word scores: per-class mean logit; prior is a distribution.
E3  Accuracy scoring, incl. empty input.
E4  Stratified subsampling: per-class counts even, seeded, deterministic.
E5  Stratified K-fold: every fold stratified, union covers all rows.
E6  Shrinkage LDA separates two Gaussian blobs (>0.9 accuracy).
E7  Kernel ridge separates the same blobs (>0.9 accuracy).
E8  Cross-fitting selection picks the informative head over noise.
E9  Selected-head averaging: probs sum to 1, accuracy stays high.
E10 Arm table validation.
"""
import pathlib
import sys

DRIVER = (pathlib.Path(__file__).resolve().parents[1] / "sessions" /
          "2026-11-28-2611.33972-wsp" / "colab_wsp_driver.py")
SRC = DRIVER.read_text()
START = SRC.index("# ---------------- PURE HELPERS (stdlib + numpy) ----------------")
END = SRC.index("# ---------------- END PURE HELPERS ----------------")
NS = {}
exec(compile(SRC[START:END], str(DRIVER), "exec"), NS)

build_template = NS["build_template"]
check_template_set = NS["check_template_set"]
label_word_scores = NS["label_word_scores"]
prior_probs = NS["prior_probs"]
accuracy = NS["accuracy"]
stratified_subsample = NS["stratified_subsample"]
stratified_kfold = NS["stratified_kfold"]
sda_fit = NS["sda_fit"]
sda_predict_proba = NS["sda_predict_proba"]
krr_fit = NS["krr_fit"]
krr_predict_proba = NS["krr_predict_proba"]
crossfit_select = NS["crossfit_select"]
fit_selected_probs = NS["fit_selected_probs"]
arm_table = NS["arm_table"]
TEMPLATES = NS["TEMPLATES"]
LABEL_WORDS = NS["LABEL_WORDS"]

fails = []


def check(name, got, want):
    if got != want:
        fails.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok {name}: {got!r}")


def check_true(name, cond, detail=""):
    if not cond:
        fails.append(f"{name}: {detail}")
    else:
        print(f"ok {name} {detail}")


# ---------------------------------------------------------------- E1
t = build_template("{x} It was <mask> .", "a great film")
check("E1 template fill", t, "a great film It was <mask> .")
check("E1 template mask-first",
      build_template("<mask> news : {x}", "markets rally"),
      "<mask> news : markets rally")
for bad in ("{x} no mask here", "no placeholder <mask>",
            "{x} {x} two <mask>", "{x} two <mask> <mask>"):
    try:
        build_template(bad, "text")
        fails.append(f"E1 bad template should raise: {bad!r}")
    except ValueError:
        print(f"ok E1 bad template raises: {bad!r}")
check("E1 sst2 program valid", check_template_set("sst2"), 4)
check("E1 agnews program valid", check_template_set("agnews"), 4)
check("E1 sst2 label groups", len(LABEL_WORDS["sst2"]), 2)
check("E1 agnews label groups", len(LABEL_WORDS["agnews"]), 4)
check("E1 sst2 neg words", LABEL_WORDS["sst2"][0][:2], ["terrible", "bad"])
check("E1 agnews world words", LABEL_WORDS["agnews"][0][:2],
      ["world", "politics"])

# ---------------------------------------------------------------- E2/E3 (stdlib)
check("E3 accuracy perfect", accuracy([0, 1, 1], [0, 1, 1]), 1.0)
check("E3 accuracy half", accuracy([0, 0], [0, 1]), 0.5)
check("E3 accuracy empty", accuracy([], []), 0.0)

# ---------------------------------------------------------------- E4/E5 (stdlib)
labels4 = [0] * 40 + [1] * 40 + [2] * 40 + [3] * 40
sub = stratified_subsample(labels4, 16, seed=7)
check("E4 subsample size", len(sub), 16)
from collections import Counter
check("E4 subsample even per class",
      sorted(Counter(labels4[i] for i in sub).values()), [4, 4, 4, 4])
check("E4 subsample deterministic",
      stratified_subsample(labels4, 16, seed=7), sub)
check("E4 subsample odd total",
      sorted(Counter(labels4[i] for i in
                     stratified_subsample(labels4, 10, seed=7)).values()),
      [2, 2, 3, 3])
try:
    stratified_subsample(labels4, 161, seed=7)
    fails.append("E4 oversample should raise")
except ValueError:
    print("ok E4 oversample raises ValueError")
folds = stratified_kfold(labels4, 5, seed=7)
check("E5 kfold count", len(folds), 5)
all_test = sorted(i for _, te in folds for i in te)
check("E5 kfold union covers all", all_test, list(range(160)))
for fi, (tr, te) in enumerate(folds):
    check(f"E5 fold {fi} test size", len(te), 32)
    check(f"E5 fold {fi} stratified",
          sorted(Counter(labels4[i] for i in te).values()), [8, 8, 8, 8])
    assert not set(tr) & set(te), f"fold {fi} train/test overlap"
print("ok E5 folds disjoint")

# ---------------------------------------------------------------- numpy sections
try:
    import numpy as np
except ImportError:
    print("SKIP numpy sections: numpy not available")
    np = None

if np is not None:
    # E2: label-word scores on synthetic logits
    rng = np.random.default_rng(0)
    V = 50
    logits = rng.normal(size=V)
    logits[3] = 5.0; logits[7] = 4.0   # class 0 words
    logits[11] = -2.0; logits[13] = -3.0  # class 1 words
    sc = label_word_scores(logits, [[3, 7], [11, 13]])
    check("E2 label-word mean logits", [round(float(v), 3) for v in sc],
          [4.5, -2.5])
    pp = prior_probs(logits, [[3, 7], [11, 13]])
    check_true("E2 prior sums to 1", abs(float(pp.sum()) - 1.0) < 1e-12,
               f"sum={float(pp.sum()):.12f}")
    check_true("E2 prior picks class 0", int(np.argmax(pp)) == 0,
               f"prior={pp}")
    # batched logits
    lg2 = np.stack([logits, -logits])
    sc2 = label_word_scores(lg2, [[3, 7], [11, 13]])
    check("E2 batched scores shape", sc2.shape, (2, 2))
    check_true("E2 batched second row flips",
               int(np.argmax(sc2[1])) == 1)

    # Synthetic frozen features: two Gaussian blobs, 8-dim.
    def make_blobs(seed=0, n_per=60, d=8, shift=3.0):
        r = np.random.default_rng(seed)
        X0 = r.normal(size=(n_per, d))
        X1 = r.normal(size=(n_per, d)) + shift
        X = np.vstack([X0, X1])
        y = np.array([0] * n_per + [1] * n_per)
        return X, y

    X, y = make_blobs()
    # E6: shrinkage LDA
    m = sda_fit(X, y, alpha=0.1)
    p_sda = sda_predict_proba(m, X)
    check("E6 sda probs shape", p_sda.shape, (120, 2))
    check_true("E6 sda probs sum to 1",
               bool(np.allclose(p_sda.sum(axis=1), 1.0)))
    acc_sda = accuracy(np.argmax(p_sda, axis=1).tolist(), y.tolist())
    print(f"  sda train acc={acc_sda:.3f}")
    check_true("E6 sda separates blobs", acc_sda > 0.9, f"acc={acc_sda:.3f}")
    # held-out split: SDA must generalise, not just memorise
    Xtr, ytr = X[:100], y[:100]
    Xte, yte = X[100:], y[100:]
    m2 = sda_fit(Xtr, ytr, alpha=0.1)
    acc_sda_te = accuracy(np.argmax(sda_predict_proba(m2, Xte), axis=1).tolist(),
                          yte.tolist())
    check_true("E6 sda held-out", acc_sda_te > 0.85, f"acc={acc_sda_te:.3f}")

    # E7: kernel ridge
    km = krr_fit(X, y, alpha=1.0)
    p_krr = krr_predict_proba(km, X)
    acc_krr = accuracy(np.argmax(p_krr, axis=1).tolist(), y.tolist())
    print(f"  krr train acc={acc_krr:.3f}")
    check_true("E7 krr separates blobs", acc_krr > 0.9, f"acc={acc_krr:.3f}")
    km2 = krr_fit(Xtr, ytr, alpha=1.0)
    acc_krr_te = accuracy(np.argmax(krr_predict_proba(km2, Xte), axis=1).tolist(),
                          yte.tolist())
    check_true("E7 krr held-out", acc_krr_te > 0.85, f"acc={acc_krr_te:.3f}")

    # E8/E9: cross-fitting selection, good vs noise head
    Xn = rng.normal(size=X.shape)  # pure noise features
    feats = {"good": X, "noise": Xn}
    dev_X = make_blobs(seed=99)[0]
    dev_y = make_blobs(seed=99)[1]

    def sda_on(view):
        def head(ftr, ytr_, fte):
            mm = sda_fit(ftr[view], ytr_, alpha=0.1)
            return sda_predict_proba(mm, fte[view])
        return head

    heads = {"sda_good": sda_on("good"), "sda_noise": sda_on("noise")}
    selected, scores, best = crossfit_select(
        heads, feats, y, {"good": dev_X, "noise": dev_X}, dev_y, seed=0)
    print(f"  crossfit scores={ {k: round(v, 3) for k, v in scores.items()} } "
          f"selected={selected}")
    check_true("E8 selection picks good head", best == "sda_good",
               f"best={best} scores={scores}")
    check_true("E8 good head scores higher",
               scores["sda_good"] > scores["sda_noise"] + 0.3)
    check_true("E8 noise head not selected", "sda_noise" not in selected)
    probs = fit_selected_probs(heads, selected, feats, y,
                               {"good": dev_X, "noise": dev_X})
    check("E9 selected probs shape", probs.shape, (120, 2))
    check_true("E9 selected probs sum to 1",
               bool(np.allclose(probs.sum(axis=1), 1.0)))
    acc_sel = accuracy(np.argmax(probs, axis=1).tolist(), dev_y.tolist())
    check_true("E9 selected heads accurate on dev", acc_sel > 0.85,
               f"acc={acc_sel:.3f}")

# ---------------------------------------------------------------- E10
ARMS = [("sst2_b16_wsp", "sst2", 16, "wsp"),
        ("sst2_b512_ft", "sst2", 512, "ft"),
        ("agnews_b128_lora", "agnews", 128, "lora")]
check("E10 arm table count", arm_table(ARMS), 3)
for bad in ([("x", "sst2", 16, "wsp"), ("x", "sst2", 16, "wsp")],
            [("x", "sst2", 32, "wsp")],
            [("x", "mr", 16, "wsp")],
            [("x", "sst2", 16, "pft")]):
    try:
        arm_table(bad)
        fails.append(f"E10 arm table should reject {bad}")
    except AssertionError:
        print(f"ok E10 arm table rejects {bad}")

print()
if fails:
    print(f"FAILED ({len(fails)}):")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
