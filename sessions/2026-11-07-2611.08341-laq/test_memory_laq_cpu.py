"""CPU test for LAQ (arXiv:2611.08341) — compressed agent memory.

Reproduces the paper's three claims at bounded scale (synthetic themed
memories, 13 groups x 4 seeds = 52 runs for the hinge experiment):
  C1  binary codes keep higher local fidelity L but move more passages
      across themes than PCA+8-bit codes (paper: L 0.97 vs 0.77,
      theme-move 10.1% vs 1.7%).
  C2  bound: 1-G <= 2.18 * sqrt(r), r = far violation rate; r tracks
      the distance noise of the compression (paper: correlation 0.979).
  C3  the far hinge raises G where reconstruction alone does not
      (paper: reconstruction raises G in 0/52 held-out runs, the far
      hinge in all 52). Fair framing used here: the baseline is the
      MSE-OPTIMAL quantizer (the OPQ-like baseline — OPQ already
      minimizes reconstruction error), and the hinge must beat THAT;
      the naive clip-3.0 quantizer is reported separately so the
      reconstruction headroom is visible instead of hidden.

What the test asserts (measurement validity, set before the full run):
  determinism, incremental-fit consistency, both compressors keep
  L > 0.85, G has headroom (mean G at the naive quantizer < 0.95),
  median bound ratio <= 2.18, corr(noise, r) > 0.5 (pooled; the
  paper's 0.979 is reported alongside, not assumed).
What it REPORTS without prejudging (the paper's claims, exact counts):
  L binary vs PCA, theme-move binary vs PCA, hinge-vs-MSE win counts.
The session README assigns the verdict from those numbers.

Run: python3 tests/test_memory_laq_cpu.py  (~5 min)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laq  # noqa: E402  (session-local: not graduated to src/ — verdict NEGATIVE)

N_GROUPS = 13
N_SEEDS = 4
N = 1200  # passages per memory (bounded scale; paper: up to 1e6)
D = 768   # embedding dim; 96 bytes/passage budget:
#   binary = 768 bits; PCA = 96 dims x 8-bit (n_keep=96)
N_KEEP = 96


def group_params(g):
    """13 memory-encoder groups: vary size, themes, spread, noise, blend."""
    rng = np.random.default_rng(1000 + g)
    return dict(n=N, d=D,
                n_themes=int(rng.integers(4, 13)),
                spread=float(rng.uniform(0.7, 1.6)),
                noise=float(rng.uniform(0.2, 0.5)),
                alpha=float(rng.uniform(0.5, 1.2)),
                group_size=16)


def run_one(seed, params, rng):
    X, themes = laq.make_memory(seed, **params)
    D_orig = laq.pairwise_sq(X)
    # --- C1: binary codes vs PCA+8-bit (equal 96 bytes/passage) ---
    codes, Q_proj = laq.binary_codes(X, seed=seed)
    D_bin_ham = laq.pairwise_sq(codes)          # plain Hamming
    D_bin = laq.binary_distance(Q_proj, codes)  # rescored (asymmetric)
    mu, R, z_mean, z_std = laq.fit_pca(X, n_keep=N_KEEP)
    Z = laq.project(X, mu, R)
    clip0 = np.full(N_KEEP, 3.0)
    Zhat0 = laq.quantize_dequantize(Z, z_mean, z_std, clip0)
    D_pca = laq.pairwise_sq(Zhat0)
    out = {
        "L_bin_ham": laq.local_fidelity(D_orig, D_bin_ham),
        "L_bin": laq.local_fidelity(D_orig, D_bin),
        "L_pca": laq.local_fidelity(D_orig, D_pca),
        "move_bin": laq.theme_move_rate(X, codes, themes),
        "move_pca": laq.theme_move_rate(X, Zhat0, themes),
        "noise_bin": laq.distance_noise(D_orig, D_bin, rng),
        "noise_pca": laq.distance_noise(D_orig, D_pca, rng),
        "G_pca0": laq.global_fidelity(D_orig, D_pca),
    }
    triples = laq.sample_far_triples(D_orig, rng, n_triples=2000)
    out["r_bin"] = laq.far_violation_rate(D_orig, D_bin, triples)
    out["r_pca"] = laq.far_violation_rate(D_orig, D_pca, triples)
    # --- C3: MSE-optimal baseline, then the hinge from there ---
    clip_rec, _ = laq.fit_clip_mse(Z, z_mean, z_std)
    Zhat_rec = laq.quantize_dequantize(Z, z_mean, z_std, clip_rec)
    D_rec = laq.pairwise_sq(Zhat_rec)
    out["G_rec"] = laq.global_fidelity(D_orig, D_rec)
    out["L_rec"] = laq.local_fidelity(D_orig, D_rec)
    out["r_rec"] = laq.far_violation_rate(D_orig, D_rec, triples)
    out["noise_rec"] = laq.distance_noise(D_orig, D_rec, rng)
    clip_h, _ = laq.fit_clip_hinge(Z, z_mean, z_std, triples,
                                    clip_start=clip_rec)
    Zhat_h = laq.quantize_dequantize(Z, z_mean, z_std, clip_h)
    D_h = laq.pairwise_sq(Zhat_h)
    out["G_hclip"] = laq.global_fidelity(D_orig, D_h)
    out["r_hclip"] = laq.far_violation_rate(D_orig, D_h, triples)
    out["noise_hclip"] = laq.distance_noise(D_orig, D_h, rng)
    w, _ = laq.fit_weights_viol(Z, z_mean, z_std, clip_rec, triples)
    Zhat_w = laq.apply_weights(Zhat_rec, w)
    D_w = laq.pairwise_sq(Zhat_w)
    out["G_hw"] = laq.global_fidelity(D_orig, D_w)
    out["L_hw"] = laq.local_fidelity(D_orig, D_w)
    out["r_hw"] = laq.far_violation_rate(D_orig, D_w, triples)
    out["noise_hw"] = laq.distance_noise(D_orig, D_w, rng)
    return out


def main():
    rng = np.random.default_rng(20261107)
    rows = []
    for g in range(N_GROUPS):
        params = group_params(g)
        for s in range(N_SEEDS):
            rows.append(run_one(5000 + 17 * g + s, params, rng))
    L_bin_ham = np.mean([r["L_bin_ham"] for r in rows])
    L_bin = np.mean([r["L_bin"] for r in rows])
    L_pca = np.mean([r["L_pca"] for r in rows])
    mv_bin = np.mean([r["move_bin"] for r in rows])
    mv_pca = np.mean([r["move_pca"] for r in rows])
    print("C1  L binary-ham=%.3f binary-rescored=%.3f PCA=%.3f | "
          "theme-move binary=%.3f PCA=%.3f"
          % (L_bin_ham, L_bin, L_pca, mv_bin, mv_pca))
    rec_wins = sum(r["G_rec"] > r["G_pca0"] + 1e-9 for r in rows)
    hclip_wins = sum(r["G_hclip"] > r["G_rec"] + 1e-9 for r in rows)
    hw_wins = sum(r["G_hw"] > r["G_rec"] + 1e-9 for r in rows)
    dG_rec = np.mean([r["G_rec"] - r["G_pca0"] for r in rows])
    dG_hclip = np.mean([r["G_hclip"] - r["G_rec"] for r in rows])
    dG_hw = np.mean([r["G_hw"] - r["G_rec"] for r in rows])
    print("C3  MSE tuning (naive->optimal) raises G in %d/52 (mean dG=%+.4f)"
          % (rec_wins, dG_rec))
    print("C3  hinge-clip beats MSE-optimal in %d/52 (mean dG=%+.4f) | "
          "weights-viol beats MSE-optimal in %d/52 (mean dG=%+.4f)"
          % (hclip_wins, dG_hclip, hw_wins, dG_hw))
    # --- C2: bound ratio and noise correlation over all runs/compressors ---
    ratios, noises, viols = [], [], []
    for r in rows:
        for key_g, key_r, key_n in (("G_pca0", "r_pca", "noise_pca"),
                                    ("G_rec", "r_rec", "noise_rec"),
                                    ("G_hclip", "r_hclip", "noise_hclip"),
                                    ("G_hw", "r_hw", "noise_hw")):
            ratios.append((1.0 - r[key_g]) / np.sqrt(max(r[key_r], 1e-12)))
            noises.append(r[key_n])
            viols.append(r[key_r])
    ratios = np.array(ratios)
    corr = float(np.corrcoef(noises, viols)[0, 1])
    # Within-memory correlation: the paper varies the compression on
    # fixed memories, so demean noise/r within each memory (across its
    # 4 compressors) and correlate the pooled deviations.
    dn, dr = [], []
    for r in rows:
        ns = np.array([r["noise_pca"], r["noise_rec"],
                       r["noise_hclip"], r["noise_hw"]])
        rs = np.array([r["r_pca"], r["r_rec"],
                       r["r_hclip"], r["r_hw"]])
        dn.extend((ns - ns.mean()).tolist())
        dr.extend((rs - rs.mean()).tolist())
    corr_within = float(np.corrcoef(dn, dr)[0, 1])
    print("C2  (1-G)/sqrt(r): median=%.2f max=%.2f | corr(noise, r)=%.3f "
          "(within-memory %.3f)" % (np.median(ratios), ratios.max(),
                                    corr, corr_within))
    # --- assertions (measurement validity) ---
    assert L_bin > 0.85 and L_pca > 0.85, "L too low: %.3f / %.3f" % (L_bin, L_pca)
    assert mv_bin > mv_pca, "binary should move more themes than PCA"
    assert np.mean([r["G_pca0"] for r in rows]) < 0.95, "G has no headroom"
    assert np.median(ratios) <= 2.18, "median bound ratio %.2f" % np.median(ratios)
    # Correlation is positive and substantial, but weaker than the
    # paper's 0.979 (measured 0.623 pooled); reported as-is in the
    # session README, not hidden behind a passing bar.
    assert corr > 0.5, "corr(noise, violation) %.3f" % corr
    # determinism: one group twice gives identical numbers
    p0 = group_params(0)
    a = run_one(5000, p0, np.random.default_rng(1))
    b = run_one(5000, p0, np.random.default_rng(1))
    assert a == b, "run is not deterministic"
    # incremental-fit consistency: fast hinge fit's returned loss must
    # equal the reference hinge_loss evaluated at the returned clip
    # with the SAME fixed margin (0.05 x median far-leg distance under
    # the initial clip 3.0)
    X, _ = laq.make_memory(777, n=400, d=32, n_themes=4)
    mu, R, zm, zs = laq.fit_pca(X)
    Z = laq.project(X, mu, R)
    tr = laq.sample_far_triples(laq.pairwise_sq(X),
                                np.random.default_rng(3), n_triples=500)
    clip0 = np.full(Z.shape[1], 3.0)
    Zhat0 = laq.quantize_dequantize(Z, zm, zs, clip0)
    ii, kk, mm = tr
    m_fix = 0.05 * float(np.median(np.sqrt(
        ((Zhat0[ii] - Zhat0[kk]) ** 2).sum(axis=1))))
    clip_h, best_h = laq.fit_clip_hinge(Z, zm, zs, tr, sweeps=1)
    ref = laq.hinge_loss(Z, zm, zs, clip_h, tr, margin_abs=m_fix)
    assert abs(ref - best_h) < 1e-9, "hinge fit inconsistent: %r vs %r" % (ref, best_h)
    clip_m, best_m = laq.fit_clip_mse(Z, zm, zs, sweeps=1)
    assert abs(laq.mse_of(Z, zm, zs, clip_m) - best_m) < 1e-9, "mse fit inconsistent"
    print("incremental-fit consistency: OK")
    print("ALL CPU CHECKS PASSED")
    import json as _json
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        _json.dump({"n_runs": len(rows),
                    "means": {k: float(np.mean([r[k] for r in rows]))
                              for k in rows[0]},
                    "corr_pooled": corr, "corr_within": corr_within,
                    "bound_ratio_median": float(np.median(ratios)),
                    "bound_ratio_max": float(ratios.max()),
                    "rows": rows}, f, indent=1)
    print("results written to %s" % out_path)
    return rows


if __name__ == "__main__":
    main()
