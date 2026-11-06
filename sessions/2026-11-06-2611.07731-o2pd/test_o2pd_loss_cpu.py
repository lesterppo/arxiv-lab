"""CPU test for the O2PD loss math (arXiv:2611.07731).

Pure-Python mirror of the formulas in colab_o2pd_driver.py (which needs
torch on the Colab VM). Checks the mathematical properties the paper's
method relies on — no GPU, no model, deterministic.

  * reset support: teacher/student distributions sum to 1 over
    top-k + reset atom;
  * alpha-divergence limits: D_alpha -> KL(q||p) as alpha -> 0 (RKL,
    mode-seeking) and -> KL(p||q) as alpha -> 1 (FKL, mode-covering);
  * overlap gate: alpha = clip(1 - overlap, 0.05, 0.95) decreases as
    overlap grows; support converges (reset atom dropped) when
    overlap > 0.5;
  * prefix gate: only the first 20% of positions contribute;
  * top-k RKL (opd_full arm): zero when student matches teacher,
    positive otherwise.
"""
import math
import sys

TOPK = 10
PREFIX_FRAC = 0.20
EPS = 1e-8


def _norm(xs):
    s = sum(xs)
    return [x / s for x in xs]


def reset_support(p_topk, q_topk):
    """Append the reset atom pooling mass outside teacher top-k."""
    p_reset = max(0.0, 1.0 - sum(p_topk))
    q_reset = max(0.0, 1.0 - sum(q_topk))
    return p_topk + [p_reset], q_topk + [q_reset]


def overlap_of(p, q):
    return sum(min(a, b) for a, b in zip(p, q))


def alpha_of(overlap):
    return min(0.95, max(0.05, 1.0 - overlap))


def d_alpha(p, q, alpha):
    inner = sum((a ** alpha) * (b ** (1 - alpha))
                for a, b in zip(p, q) if a > 0 and b > 0)
    return (1.0 - inner) / (alpha * (1 - alpha))


def kl(a, b):
    return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0)


def rkl_topk(p_topk, q_topk):
    pn, qn = _norm(p_topk), _norm(q_topk)
    return kl(qn, pn)


def o2pd_token_loss(p_topk, q_topk):
    """One position: returns (loss, overlap, alpha, reset_kept)."""
    p, q = reset_support(p_topk, q_topk)
    o = overlap_of(p, q)
    a = alpha_of(o)
    if o > 0.5:  # support convergence: drop the reset atom
        p, q = _norm(p[:-1]), _norm(q[:-1])
        kept = False
    else:
        kept = True
    return d_alpha(p, q, a), o, a, kept


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" | " + detail if detail else ""))
    if not cond:
        check.failures += 1


check.failures = 0

# --- 1. reset support sums to 1 -------------------------------------------
p_topk = [0.40, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002]
q_topk = [0.10, 0.30, 0.05, 0.20, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
p, q = reset_support(p_topk, q_topk)
check("reset-support sums to 1",
      abs(sum(p) - 1.0) < 1e-9 and abs(sum(q) - 1.0) < 1e-9,
      "sum(p)=%.6f sum(q)=%.6f" % (sum(p), sum(q)))
check("reset atom holds the missing mass",
      abs(p[-1] - (1 - sum(p_topk))) < 1e-9
      and abs(q[-1] - (1 - sum(q_topk))) < 1e-9,
      "p_reset=%.4f q_reset=%.4f" % (p[-1], q[-1]))

# --- 2. alpha-divergence limits -------------------------------------------
loss, o, a, kept = o2pd_token_loss(p_topk, q_topk)
p_full, q_full = reset_support(p_topk, q_topk)
d_small = d_alpha(p_full, q_full, 0.01)   # ~ KL(q||p)
d_large = d_alpha(p_full, q_full, 0.99)   # ~ KL(p||q)
check("alpha->0 limit is RKL KL(q||p)",
      abs(d_small - kl(q_full, p_full)) < 0.02,
      "D_0.01=%.4f KL(q||p)=%.4f" % (d_small, kl(q_full, p_full)))
check("alpha->1 limit is FKL KL(p||q)",
      abs(d_large - kl(p_full, q_full)) < 0.02,
      "D_0.99=%.4f KL(p||q)=%.4f" % (d_large, kl(p_full, q_full)))
check("alpha-divergence is non-negative", d_alpha(p_full, q_full, 0.5) >= -1e-9)

# --- 3. overlap gate monotonicity + support convergence -------------------
o_low = overlap_of(*reset_support(p_topk, q_topk))
o_high = overlap_of(*reset_support(p_topk, [x * 0.9 + y * 0.1 for x, y in
                                            zip(p_topk, q_topk)]))
check("alpha decreases as overlap grows",
      alpha_of(o_high) < alpha_of(o_low),
      "overlap %.3f->%.3f alpha %.3f->%.3f"
      % (o_low, o_high, alpha_of(o_low), alpha_of(o_high)))
loss_hi, o_hi, a_hi, kept_hi = o2pd_token_loss(
    p_topk, [x * 0.95 + y * 0.05 for x, y in zip(p_topk, q_topk)])
check("support converges (reset dropped) at high overlap",
      o_hi > 0.5 and not kept_hi,
      "overlap=%.3f kept=%s" % (o_hi, kept_hi))
check("reset kept at low overlap",
      o2pd_token_loss(p_topk,
                      [0.01] * 8 + [0.30, 0.50])[3],
      "overlap=%.3f kept=%s"
      % (o2pd_token_loss(p_topk, [0.01] * 8 + [0.30, 0.50])[1],
         o2pd_token_loss(p_topk, [0.01] * 8 + [0.30, 0.50])[3]))

# --- 4. prefix gate --------------------------------------------------------
L = 50
losses = [0.1 * (i + 1) for i in range(L)]
n_pre = max(1, int(PREFIX_FRAC * L))
prefix_mean = sum(losses[:n_pre]) / n_pre
full_mean = sum(losses) / L
check("prefix gate keeps only the first 20%",
      n_pre == 10 and abs(prefix_mean - 0.55) < 1e-9
      and abs(full_mean - 2.55) < 1e-9,
      "n_pre=%d prefix_mean=%.2f full_mean=%.2f" % (n_pre, prefix_mean, full_mean))

# --- 5. top-k RKL (opd_full arm) ------------------------------------------
check("RKL is 0 when student matches teacher",
      abs(rkl_topk(p_topk, list(p_topk))) < 1e-9)
check("RKL is positive on mismatch",
      rkl_topk(p_topk, q_topk) > 0,
      "RKL=%.4f" % rkl_topk(p_topk, q_topk))
check("RKL differs from FKL (mode-seeking vs covering)",
      abs(rkl_topk(p_topk, q_topk) - kl(_norm(p_topk), _norm(q_topk))) > 1e-6,
      "RKL=%.4f FKL=%.4f"
      % (rkl_topk(p_topk, q_topk), kl(_norm(p_topk), _norm(q_topk))))

print("failures: %d" % check.failures)
sys.exit(1 if check.failures else 0)
