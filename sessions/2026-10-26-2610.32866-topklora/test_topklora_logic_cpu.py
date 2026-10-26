"""CPU logic test for the TopK-LoRA gating math used in colab_topklora.py.

Verifies (numpy, no model, no GPU):
1. TopK mask keeps exactly k latents per token, chosen by |activation|.
2. Dense mode (k == r) keeps all latents.
3. Ablating a latent zeroes it in both modes.
4. First-line truncation + normalisation used by the eval parser.
"""
import numpy as np

R, K = 16, 4


def topk_gate(h, k):
    if k >= h.shape[-1]:
        return h.copy()
    idx = np.argpartition(-np.abs(h), k, axis=-1)[..., :k]
    m = np.zeros_like(h)
    np.put_along_axis(m, idx, 1.0, axis=-1)
    return h * m


def ablate(h, idxs):
    out = h.copy()
    out[..., list(idxs)] = 0.0
    return out


def first_line(text):
    return text.split("\n")[0].strip()


def norm(s):
    return s.strip().lower().rstrip(".").strip()


rng = np.random.default_rng(20261026)
fails = []

# 1. TopK keeps exactly k per token, the largest-|.| ones
h = rng.normal(size=(64, R))
g = topk_gate(h, K)
active = (g != 0).sum(axis=-1)
if not np.all(active == K):
    fails.append(f"active counts not all {K}: {sorted(set(active.tolist()))}")
top_idx = np.argpartition(-np.abs(h), K, axis=-1)[:, :K]
for i in range(64):
    if set(np.nonzero(g[i])[0].tolist()) != set(top_idx[i].tolist()):
        fails.append(f"row {i}: kept set != top-|.| set"); break
# values of kept latents unchanged
if not np.allclose(g[g != 0], h[g != 0]):
    fails.append("kept latent values changed by gating")

# 2. dense keeps everything
if not np.array_equal(topk_gate(h, R), h):
    fails.append("dense mode altered activations")

# 3. ablation zeroes the chosen latent in both modes
for k in (K, R):
    a = ablate(topk_gate(h, k), {3, 7})
    if not np.all(a[:, [3, 7]] == 0):
        fails.append(f"ablation failed for k={k}")

# 4. parser: first line only, case/punctuation-insensitive match
assert first_line("MEADOW\nand more text") == "MEADOW"
assert norm("  MEADOW. ") == "meadow"
assert norm(first_line("Paris\nFrance")) == "paris"
assert first_line("") == ""

if fails:
    print("FAIL"); [print(" -", f) for f in fails]; raise SystemExit(1)
print("PASS: topk_gate exact-k + top-|.| selection, dense identity, "
      "ablation zeroing, first-line parser (6 checks)")
