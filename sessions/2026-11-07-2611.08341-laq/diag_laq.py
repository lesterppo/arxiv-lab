"""Diagnostic for the LAQ regime: 6 runs, all arms, no assertions.
Tune the synthetic memory until L/G sit in the paper's regime."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from arxiv_lab.memory import laq  # noqa: E402

N, D, N_KEEP = 1200, 768, 96


def run_one(seed, params, rng):
    X, themes = laq.make_memory(seed, **params)
    D_orig = laq.pairwise_sq(X)
    codes, Q_proj = laq.binary_codes(X, seed=seed)
    D_bin = laq.binary_distance(Q_proj, codes)
    D_ham = laq.pairwise_sq(codes)
    mu, R, zm, zs = laq.fit_pca(X, n_keep=N_KEEP)
    Z = laq.project(X, mu, R)
    clip0 = np.full(N_KEEP, 3.0)
    Zhat0 = laq.quantize_dequantize(Z, zm, zs, clip0)
    D_pca = laq.pairwise_sq(Zhat0)
    triples = laq.sample_far_triples(D_orig, rng, n_triples=2000)
    out = {
        "L_ham": laq.local_fidelity(D_orig, D_ham),
        "L_bin": laq.local_fidelity(D_orig, D_bin),
        "L_pca": laq.local_fidelity(D_orig, D_pca),
        "mv_bin": laq.theme_move_rate(X, codes, themes),
        "mv_pca": laq.theme_move_rate(X, Zhat0, themes),
        "G_pca0": laq.global_fidelity(D_orig, D_pca),
        "r_pca": laq.far_violation_rate(D_orig, D_pca, triples),
    }
    clip_rec, _ = laq.fit_clip_mse(Z, zm, zs)
    Zhat_rec = laq.quantize_dequantize(Z, zm, zs, clip_rec)
    out["G_rec"] = laq.global_fidelity(D_orig, laq.pairwise_sq(Zhat_rec))
    out["L_rec"] = laq.local_fidelity(D_orig, laq.pairwise_sq(Zhat_rec))
    # hinge starts where reconstruction stopped (OPQ-like baseline)
    clip_h, _ = laq.fit_clip_hinge(Z, zm, zs, triples, clip_start=clip_rec)
    Zhat_h = laq.quantize_dequantize(Z, zm, zs, clip_h)
    out["G_hclip"] = laq.global_fidelity(D_orig, laq.pairwise_sq(Zhat_h))
    w, _ = laq.fit_weights_viol(Z, zm, zs, clip_rec, triples)
    Zhat_w = laq.apply_weights(Zhat_rec, w)
    out["G_hw"] = laq.global_fidelity(D_orig, laq.pairwise_sq(Zhat_w))
    out["L_hw"] = laq.local_fidelity(D_orig, laq.pairwise_sq(Zhat_w))
    out["r_hw"] = laq.far_violation_rate(
        D_orig, laq.pairwise_sq(Zhat_w), triples)
    return out


def main():
    rng = np.random.default_rng(20261107)
    rows = []
    for g in range(3):
        params = dict(n=N, d=D, n_themes=int(rng.integers(4, 13)),
                      spread=float(rng.uniform(0.7, 1.6)),
                      noise=float(rng.uniform(0.2, 0.5)),
                      alpha=float(rng.uniform(0.5, 1.2)),
                      group_size=16)
        for s in range(2):
            rows.append(run_one(5000 + 17 * g + s, params, rng))
    keys = ["L_ham", "L_bin", "L_pca", "L_rec", "L_hw", "mv_bin", "mv_pca",
            "G_pca0", "G_rec", "G_hclip", "G_hw", "r_pca", "r_hw"]
    n_hclip = sum(r["G_hclip"] > r["G_rec"] + 1e-9 for r in rows)
    n_hw = sum(r["G_hw"] > r["G_rec"] + 1e-9 for r in rows)
    print("hinge-clip beats MSE-baseline in %d/%d | weights-viol beats it in %d/%d"
          % (n_hclip, len(rows), n_hw, len(rows)))
    for k in keys:
        vals = [r[k] for r in rows]
        print("%-8s mean=%.3f min=%.3f max=%.3f" % (k, np.mean(vals),
                                                    np.min(vals), np.max(vals)))


if __name__ == "__main__":
    main()
