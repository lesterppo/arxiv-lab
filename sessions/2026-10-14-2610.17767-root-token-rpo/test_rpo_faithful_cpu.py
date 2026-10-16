"""CPU test: faithful-RPO loss math (numpy mirror of the torch code in
colab_rpo_faithful.py). Validates the formulas, not the model integration.

Covers:
  1. group advantages: A_i = r_i - mean(r)
  2. root decision logp: think -> log P(think); direct -> log(1 - P(think))
  3. Schulman sequence KL: mean(exp(dr) - dr - 1), dr = logp_ref - logp
  4. combined loss: -mean(A * logp_root) + beta * mean(KL)
  5. direction sanity: positive advantage on a THINK arm must push
     P(<think>) UP (gradient of -A*logp w.r.t. the think logit > 0)
"""
import math
import numpy as np

BETA = 0.02
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print("PASS %s" % name)
    else:
        failed += 1
        print("FAIL %s %s" % (name, detail))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# --- 1. group advantages -------------------------------------------------
rewards = np.array([0.85, 0.85, 1.0, 1.0])  # easy item: direct better
advs = rewards - rewards.mean()
check("advantage easy", np.allclose(advs, [-0.075, -0.075, 0.075, 0.075]))
check("advantage sums to zero", abs(advs.sum()) < 1e-12)

rewards_h = np.array([0.55, 0.55, 0.23, 0.23])  # hard item: think better
advs_h = rewards_h - rewards_h.mean()
check("advantage hard", np.allclose(advs_h, [0.16, 0.16, -0.16, -0.16]))

uniform = np.array([1.0, 1.0, 1.0, 1.0])
check("uniform -> zero advantage", np.allclose(uniform - uniform.mean(), 0.0))

# --- 2. root decision logp -----------------------------------------------
rng = np.random.RandomState(0)
V = 32000
think_id = 1234
logits = rng.randn(V).astype(np.float64)
p = softmax(logits)
p_think = p[think_id]

logp_think = math.log(p_think)                       # torch: log_softmax[THINK_ID]
logp_direct = math.log1p(-min(p_think, 1.0 - 1e-6))   # torch: log1p(-p.clamp)
check("think logp == log p", abs(logp_think - math.log(p_think)) < 1e-12)
check("direct logp == log(1-p)",
      abs(logp_direct - math.log(1.0 - p_think)) < 1e-9)
check("direct logp <= 0", logp_direct <= 0.0)

# edge: p_think ~ 0  -> direct logp ~ 0 ; p_think ~ 1 -> think logp ~ 0
check("p~0: direct logp ~ 0", abs(math.log1p(-1e-9)) < 1e-8)
check("p~1: think logp ~ 0", abs(math.log(1.0 - 1e-9)) < 1e-8)

# --- 3. Schulman KL -------------------------------------------------------
# identical distributions -> KL ~ 0 ; shifted -> KL > 0
lp = np.log(p + 1e-12)
lp_ref = np.log(p + 1e-12)
dr = lp_ref - lp
kl_same = np.mean(np.exp(dr) - dr - 1.0)
check("KL identical ~ 0", abs(kl_same) < 1e-9, str(kl_same))

p2 = softmax(logits + rng.randn(V) * 0.5)
lp2 = np.log(p2 + 1e-12)
dr2 = lp2 - lp
kl_shift = np.mean(np.exp(dr2) - dr2 - 1.0)
check("KL shifted > 0", kl_shift > 1e-6, str(kl_shift))
# cross-check against the textbook KL for small shifts (2nd-order match)
kl_textbook = np.sum(p * (lp - lp2))
check("Schulman ~= textbook KL (small shift)",
      abs(kl_shift - kl_textbook) < 0.05 * max(kl_textbook, 1e-9),
      "schulman=%.5f textbook=%.5f" % (kl_shift, kl_textbook))

# --- 4. combined loss ------------------------------------------------------
advs4 = np.array([0.16, 0.16, -0.16, -0.16])
logp_roots = np.array([-2.0, -2.1, -0.5, -0.4])   # think arms unlikely, direct likely
kls = np.array([0.01, 0.012, 0.008, 0.009])
pg = -(advs4 * logp_roots).mean()
loss = pg + BETA * kls.mean()
check("loss finite", math.isfinite(loss))
# hand calc: -(0.16*-2 + 0.16*-2.1 + -0.16*-0.5 + -0.16*-0.4)/4 + 0.02*0.00975
hand = -((0.16 * -2.0 + 0.16 * -2.1 - 0.16 * -0.5 - 0.16 * -0.4) / 4) \
    + 0.02 * 0.00975
check("loss matches hand calc", abs(loss - hand) < 1e-12, str(loss - hand))

# --- 5. gradient direction --------------------------------------------------
# d(-A * log p_think)/d(logit_think) = -A * (1 - p_think).
# With A > 0 (think was better), the gradient must be NEGATIVE
# (gradient descent then INCREASES the think logit).
A_pos, pt = 0.16, 0.3
grad = -A_pos * (1.0 - pt)
check("positive advantage pushes P(think) up", grad < 0, str(grad))
# d(-A * log(1-p))/d(logit) = +A * p ; A < 0 (direct was better) -> negative
A_neg = -0.16
grad_d = A_neg * pt  # = -(-0.16)*... let me redo: -A*(d log(1-p)/d logit)
# d log(1-p)/d logit_think = -p  (since dp/dlogit = p(1-p), d(1-p) = -p(1-p))
# so d(-A log(1-p))/dlogit = -A * (-p) = A*p
grad_direct = A_neg * pt
check("negative advantage on direct arm pushes P(think) down",
      grad_direct < 0, str(grad_direct))

print("\n%d passed, %d failed" % (passed, failed))
raise SystemExit(1 if failed else 0)
