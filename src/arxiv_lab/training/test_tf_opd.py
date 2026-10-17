"""CPU test for TF-OPD (arXiv:2610.22236), stdlib + numpy only.

Deterministic synthetic reproduction of the paper's two core claims:

1. EXACT layer-wise solution: specialists sharing a base are fused by the
   closed-form least-squares solve (Eq. 2), which needs only the second
   moments of each specialist's own rollout inputs. Checks:
   (a) the solve satisfies the normal equations to solver precision;
   (b) the fused weight strictly beats the base and task-arithmetic
       baselines on the distillation objective (Eq. 1);
   (c) when specialists' input ranges are mutually orthogonal, the fused
       layer reproduces each specialist's output on that specialist's own
       inputs exactly (Prop. 3.1(iii)).
2. ROLLOUT COMPOSITION matters more than the solver (§3.4): answers
   truncated in repetition loops are few among answers but a large share
   of tokens, and their second moments concentrate on a few directions.
   Checks:
   (a) the DEFLATE-compressibility filter flags looping text and passes
       ordinary reasoning text;
   (b) ablating (dropping) loop-contaminated rollout rows improves the
       fused specialist's own-task accuracy vs the contaminated fusion.

Plus an end-to-end check: the fused 2-layer MLP keeps each specialist's
own-task accuracy (no catastrophic loss) and is on average better than
either specialist alone (the paper's headline result).

Writes measured numbers to TF_OPD_RESULTS (JSON) if the env var is set.

Run:  python3 src/arxiv_lab/training/test_tf_opd.py   (from repo root)
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from arxiv_lab.training.tf_opd import (
    accumulate_gram,
    collect_layer_inputs,
    compression_ratio,
    detect_repetition_loop,
    fuse_layer_dict,
    fuse_linear_layer,
    mlp_forward,
    train_specialist_sgd,
    unreproduced_share,
    weighted_objective,
)

SEED = 20261017
D_IN, H, D_OUT = 8, 16, 1
N_TRAIN, N_TEST = 400, 1000
ROLLOUTS, ROWS_PER_ANSWER = 600, 24

results = {"checks": []}


def check(name, cond, **nums):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} " + " ".join(f"{k}={v:.6g}" for k, v in nums.items()))
    results["checks"].append({"name": name, "pass": bool(cond), **nums})
    if not cond:
        raise AssertionError(f"check failed: {name} {nums}")


def mse(yhat, y):
    return float(np.mean((yhat - y) ** 2))


def task_data(rng, n, mu, scales, kind):
    X = rng.normal(0, 1, (n, D_IN)) * scales + mu
    if kind == "A":  # "mathematics": quadratic bowl in (x0, x1)
        y = (X[:, [0]] ** 2 + 0.5 * X[:, [1]] ** 2 - 1.0)
    else:            # "code": sinusoid in x4 plus linear x5
        y = (np.sin(2.0 * X[:, [4]]) + 0.5 * X[:, [5]])
    return X, y


def make_rollouts(rng, mu, scales, n_answers, rows):
    return [rng.normal(0, 1, (rows, D_IN)) * scales + mu
            for _ in range(n_answers)]


def main():
    if np is None:
        print("SKIP: numpy not available")
        return
    rng = np.random.default_rng(SEED)

    # ---- base model + two specialists on different toy tasks ----
    # Informative subspaces are nearly disjoint (dims 0-1 for A, dims 4-5
    # for B): the regime where input-weighted fusion avoids interference
    # (cf. Prop. 3.1(iii)); the remaining dims carry small shared noise.
    muA = np.array([2.0, 2.0, 0, 0, 0, 0, 0, 0])
    muB = np.array([0, 0, 0, 0, -2.0, -2.0, 0, 0])
    scA = np.array([1.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    scB = np.array([0.3, 0.3, 0.3, 0.3, 1.0, 1.0, 0.3, 0.3])
    init = {
        "W1": rng.normal(0, 0.5, (D_IN, H)),
        "b1": np.zeros(H),
        "W2": rng.normal(0, 0.5, (H, D_OUT)),
        "b2": np.zeros(D_OUT),
    }
    XA_tr, yA_tr = task_data(rng, N_TRAIN, muA, scA, "A")
    XB_tr, yB_tr = task_data(rng, N_TRAIN, muB, scB, "B")
    XA_te, yA_te = task_data(rng, N_TEST, muA, scA, "A")
    XB_te, yB_te = task_data(rng, N_TEST, muB, scB, "B")

    # Standardize each task's targets (zero mean, unit variance, stats
    # from train). The two toy tasks have very different raw output
    # scales; without this the large-scale task's update dominates the
    # shared W2 least-squares and the test would measure target-scale
    # dominance rather than the fusion mechanism. The exactness claims
    # and the loop ablation are scale-invariant.
    def _std(y_tr, y_te):
        m, s = float(y_tr.mean()), float(y_tr.std())
        return (y_tr - m) / s, (y_te - m) / s
    yA_tr, yA_te = _std(yA_tr, yA_te)
    yB_tr, yB_te = _std(yB_tr, yB_te)

    # Phase 1: a capable shared base, trained on the mixture (the
    # "pretrained" model both specialists start from).
    Xmix = np.vstack([XA_tr, XB_tr])
    ymix = np.vstack([yA_tr, yB_tr])
    base = train_specialist_sgd(init, Xmix, ymix, steps=500, lr=0.05)

    # Phase 2: brief specialist tuning from the shared base (small weight
    # movement, as in the paper's RL specialists — §3.5 notes RL "barely
    # moves weights").
    specA = train_specialist_sgd(base, XA_tr, yA_tr, steps=80, lr=0.01)
    specB = train_specialist_sgd(base, XB_tr, yB_tr, steps=80, lr=0.01)
    mseA_A = mse(mlp_forward(XA_te, specA)[0], yA_te)
    mseB_B = mse(mlp_forward(XB_te, specB)[0], yB_te)
    mseA_B = mse(mlp_forward(XB_te, specA)[0], yB_te)
    mseB_A = mse(mlp_forward(XA_te, specB)[0], yA_te)
    mseBase_A = mse(mlp_forward(XA_te, base)[0], yA_te)
    mseBase_B = mse(mlp_forward(XB_te, base)[0], yB_te)
    check("each specialist is the best on its own task",
          mseA_A < mseBase_A and mseB_B < mseBase_B
          and mseA_A < mseB_A and mseB_B < mseA_B,
          mseA_A=mseA_A, mseB_B=mseB_B,
          mseBase_A=mseBase_A, mseBase_B=mseBase_B,
          mseA_B=mseA_B, mseB_A=mseB_A)
    results["specialist_mse"] = {"A_on_A": mseA_A, "B_on_B": mseB_B,
                                 "A_on_B": mseA_B, "B_on_A": mseB_A,
                                 "base_on_A": mseBase_A, "base_on_B": mseBase_B}

    deltas = [{k: specA[k] - base[k] for k in specA},
              {k: specB[k] - base[k] for k in specB}]

    # ---- rollouts: each specialist reads its own answers ----
    ansA = make_rollouts(rng, muA, scA, ROLLOUTS, ROWS_PER_ANSWER)
    ansB = make_rollouts(rng, muB, scB, ROLLOUTS, ROWS_PER_ANSWER)
    inpA = collect_layer_inputs(specA, ansA)
    inpB = collect_layer_inputs(specB, ansB)
    grams = [{n: accumulate_gram(inpA[n]) for n in ("W1", "W2")},
             {n: accumulate_gram(inpB[n]) for n in ("W1", "W2")}]

    # ---- claim 1a: the closed-form solve is exact (normal equations) ----
    for name in ("W1", "W2"):
        W_star, Ds, info = fuse_linear_layer(
            base[name], [deltas[0][name], deltas[1][name]],
            [grams[0][name], grams[1][name]])
        check(f"normal-equation exactness [{name}]",
              info["normal_eq_rel_residual"] < 1e-12,
              rel_resid=info["normal_eq_rel_residual"])
        results[f"normal_eq_rel_resid_{name}"] = info["normal_eq_rel_residual"]

    # ---- claim 1b: fused beats base + task arithmetic on Eq. 1 ----
    for name in ("W1", "W2"):
        Gt = [w * (0.9 * g + 0.1 * np.diag(np.diag(g)))
              for w, g in zip([1.0, 1.0],
                              [grams[0][name], grams[1][name]])]
        W_star, _, _ = fuse_linear_layer(
            base[name], [deltas[0][name], deltas[1][name]],
            [grams[0][name], grams[1][name]])
        W_ta = base[name] + deltas[0][name] + deltas[1][name]
        J_star = weighted_objective(W_star, base[name],
                                    [deltas[0][name], deltas[1][name]], Gt)
        J_base = weighted_objective(base[name], base[name],
                                    [deltas[0][name], deltas[1][name]], Gt)
        J_ta = weighted_objective(W_ta, base[name],
                                  [deltas[0][name], deltas[1][name]], Gt)
        check(f"fused minimizes distillation objective [{name}]",
              J_star < J_base and J_star < J_ta,
              J_star=J_star, J_base=J_base, J_task_arith=J_ta)
        results[f"objective_{name}"] = {"fused": J_star, "base": J_base,
                                        "task_arithmetic": J_ta}

    # ---- claim 1c: orthogonal input ranges -> exact per-teacher output ----
    rng2 = np.random.default_rng(SEED + 1)
    XA_o = np.hstack([rng2.normal(0, 1, (200, 4)),
                      np.zeros((200, 4))])          # support dims 0..3
    XB_o = np.hstack([np.zeros((200, 4)),
                      rng2.normal(0, 1, (200, 4))])  # support dims 4..7
    Go = [accumulate_gram(XA_o), accumulate_gram(XB_o)]
    W_star_o, D_star_o, _ = fuse_linear_layer(
        base["W1"], [deltas[0]["W1"], deltas[1]["W1"]], Go)
    relA = (np.linalg.norm(XA_o @ D_star_o - XA_o @ deltas[0]["W1"])
            / np.linalg.norm(XA_o @ deltas[0]["W1"]))
    relB = (np.linalg.norm(XB_o @ D_star_o - XB_o @ deltas[1]["W1"])
            / np.linalg.norm(XB_o @ deltas[1]["W1"]))
    Gt_o = [0.9 * g + 0.1 * np.diag(np.diag(g)) for g in Go]
    epsA = unreproduced_share(D_star_o, deltas[0]["W1"], Gt_o[0])
    epsB = unreproduced_share(D_star_o, deltas[1]["W1"], Gt_o[1])
    check("orthogonal ranges: exact per-teacher reproduction",
          relA < 1e-8 and relB < 1e-8 and epsA < 1e-9 and epsB < 1e-9,
          rel_resid_A=relA, rel_resid_B=relB, eps_A=epsA, eps_B=epsB)
    results["orthogonal_exact"] = {"rel_resid_A": float(relA),
                                   "rel_resid_B": float(relB),
                                   "eps_A": float(epsA), "eps_B": float(epsB)}

    # ---- end-to-end: fused MLP keeps each specialist's own-task skill ----
    fused = fuse_layer_dict(base, deltas, grams)
    mseF_A = mse(mlp_forward(XA_te, fused)[0], yA_te)
    mseF_B = mse(mlp_forward(XB_te, fused)[0], yB_te)
    avg_fused = 0.5 * (mseF_A + mseF_B)
    avg_A = 0.5 * (mseA_A + mseA_B)
    avg_B = 0.5 * (mseB_A + mseB_B)
    check("fused keeps own-task accuracy (no catastrophic loss)",
          mseF_A <= 1.30 * mseA_A and mseF_B <= 1.30 * mseB_B,
          fused_A=mseF_A, specA_A=mseA_A, fused_B=mseF_B, specB_B=mseB_B)
    check("fused on average better than either specialist alone",
          avg_fused < avg_A and avg_fused < avg_B,
          avg_fused=avg_fused, avg_specA=avg_A, avg_specB=avg_B)
    results["end_to_end_mse"] = {"fused_A": mseF_A, "fused_B": mseF_B,
                                 "specA_A": mseA_A, "specB_B": mseB_B,
                                 "specA_B": mseA_B, "specB_A": mseB_A}

    # ---- claim 2a: the DEFLATE loop filter ----
    # Genuinely varied pseudo-reasoning text (seeded): random words and
    # numbers, so it is NOT self-repetitive like a naive `para * 90`.
    vrng = np.random.default_rng(SEED + 9)
    vocab = ["consider", "quadratic", "discriminant", "roots", "square",
             "compute", "verify", "hence", "therefore", "let", "define",
             "suppose", "then", "gives", "equals", "plus", "minus",
             "coefficient", "equation", "solve", "check", "result",
             "step", "next", "first", "second", "value", "obtain",
             "substitute", "simplify", "factor", "expand", "derive"]
    words = []
    while len(" ".join(words)) < 26000:
        n = int(vrng.integers(6, 14))
        sent = [vocab[int(vrng.integers(0, len(vocab)))] for _ in range(n)]
        if vrng.random() < 0.5:
            sent.insert(int(vrng.integers(0, n)), str(int(vrng.integers(0, 9999))))
        words.extend(sent)
        words.append(".")
    normal_text = " ".join(words)
    loop_text = ("the answer is 42 and the answer is 42 and the answer is "
                 "42 and ") * 1200      # repetition loop, > 20000 chars
    r_normal = compression_ratio(normal_text)
    r_loop = compression_ratio(loop_text)
    check("loop filter flags repetition, passes normal text",
          detect_repetition_loop(loop_text) and
          not detect_repetition_loop(normal_text),
          ratio_loop=r_loop, ratio_normal=r_normal)
    results["loop_filter"] = {"ratio_loop": r_loop, "ratio_normal": r_normal,
                              "flagged_loop": True, "flagged_normal": False}

    # ---- claim 2b: loop-contaminated rollouts degrade the fusion ----
    rng3 = np.random.default_rng(SEED + 2)
    loop_dir = np.ones(D_IN) / math.sqrt(D_IN)
    n_loop_answers, loop_rows = 15, 160       # few answers...
    clean_rows = ROLLOUTS * ROWS_PER_ANSWER   # 4800 token rows
    loop_rows_tot = n_loop_answers * loop_rows  # 2400 token rows (33%)
    loop_ans = [np.outer(np.ones(loop_rows), 5.0 * loop_dir)
                + rng3.normal(0, 0.01, (loop_rows, D_IN))
                for _ in range(n_loop_answers)]
    G_loop = accumulate_gram(np.vstack(loop_ans))
    eig = np.linalg.eigvalsh(G_loop)
    eig_share = float(eig[-1] / eig.sum())
    # fused WITH loop contamination (A's grams only; B unchanged)
    grams_dirty = [{n: (clean_rows * grams[0][n] + loop_rows_tot *
                       accumulate_gram(np.vstack(
                           collect_layer_inputs(specA, loop_ans)[n])))
                       / (clean_rows + loop_rows_tot)
                    for n in ("W1", "W2")}, grams[1]]
    fused_dirty = fuse_layer_dict(base, deltas, grams_dirty)
    fused_clean = fused  # from above: loop rows excluded (filtered)
    mseF_clean_A = mse(mlp_forward(XA_te, fused_clean)[0], yA_te)
    mseF_dirty_A = mse(mlp_forward(XA_te, fused_dirty)[0], yA_te)
    # Mechanism-level view: the contaminated solve is worse even on the
    # CLEAN input distribution — capacity spent on degenerate inputs
    # (paper §3.4: "spends its capacity reproducing the teacher on
    # degenerate input"). J = Eq.1 on A's clean shrunk W1 gram.
    GtA_clean = 0.9 * grams[0]["W1"] + 0.1 * np.diag(np.diag(grams[0]["W1"]))
    def _J_clean_W1(fused):
        Wd = (fused["W1"] - base["W1"]) - deltas[0]["W1"]
        return float(np.trace(Wd.T @ GtA_clean @ Wd))
    J_clean = _J_clean_W1(fused_clean)
    J_dirty = _J_clean_W1(fused_dirty)
    check("loop contamination concentrates second moments",
          eig_share > 0.90, top_eig_share=eig_share,
          loop_token_share=loop_rows_tot / (clean_rows + loop_rows_tot))
    check("dropping loop rollouts improves fused own-task accuracy",
          mseF_clean_A < mseF_dirty_A,
          mse_filtered=mseF_clean_A, mse_contaminated=mseF_dirty_A)
    check("contaminated solve is worse on the clean input distribution",
          J_dirty > J_clean,
          J_filtered=J_clean, J_contaminated=J_dirty)
    results["ablation"] = {"top_eig_share_loop_gram": eig_share,
                           "loop_token_share": loop_rows_tot /
                           (clean_rows + loop_rows_tot),
                           "mse_filtered_A": mseF_clean_A,
                           "mse_contaminated_A": mseF_dirty_A,
                           "improvement": mseF_dirty_A - mseF_clean_A,
                           "J_clean_dist_filtered": J_clean,
                           "J_clean_dist_contaminated": J_dirty}

    n_pass = sum(c["pass"] for c in results["checks"])
    print(f"\n{n_pass}/{len(results['checks'])} checks passed")
    results["summary"] = {"passed": n_pass, "total": len(results["checks"])}

    out = os.environ.get("TF_OPD_RESULTS")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"results written to {out}")


if __name__ == "__main__":
    main()
