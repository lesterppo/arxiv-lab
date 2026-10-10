"""CPU test for arXiv 2610.08312 — CoDe-LoRA mechanism miniature.

Reproduces the paper's core claims on a synthetic continual-learning setup
(d_in=64, d_out=16, rank-6 linear tasks, fixed seed):

- task A = shared input subspace (rank 4: s1..s4) + A-specific (rank 2)
- task B = semantically related: shares 3 of 4 shared input directions and
  the transferable input->output mapping, with its own specific directions.
  A and B deliberately share the output space (the high-similarity regime
  where the paper's Q1 says the orthogonality dilemma is most severe).
- task C = unrelated: different input subspace AND output space.

Per-task updates are least-squares fits (min-norm). Three arms on A->B:
(1) naive sequential fine-tuning (= unconstrained update),
(2) O-LoRA-style strict null projection of B's update onto the null space
    of A's accumulated column space,
(3) CoDe: consolidation branch (Eq. 3-4) + per-task experts with
    prototype routing and confidence fallback (tau=0.75).

Deterministic (fixed seed). numpy-guarded: skips cleanly without numpy.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

try:
    import numpy as np
    from arxiv_lab.training.code_lora import (
        DEFAULT_TAU,
        dynamic_scaling,
        top_column_basis,
        consolidate_null_project,
        update_overlap_rho,
        code_lora_step,
        build_prototype,
        route_prototype,
    )
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if not HAS_NUMPY:
    print("SKIP: numpy not available")
    sys.exit(0)

passed = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok: {name}")


SEED = 20261010
D_IN, D_OUT, RANK = 64, 16, 6
N_TRAIN, N_TEST = 300, 200
SIGMA = 0.2
MU_NORM = 10.0
DV = 6  # null-space basis rank (= LoRA rank r in the paper)

rng = np.random.default_rng(SEED)

# ---- ground truth geometry ----
B64, _ = np.linalg.qr(rng.standard_normal((D_IN, D_IN)))
S = B64[:, :4]                 # shared input subspace (rank 4)
A_in = B64[:, 4:6]             # A-specific (rank 2)
B_in = B64[:, 6:9]             # B-specific (rank 3)
C_in = B64[:, 9:15]            # C subspace, unrelated (rank 6)
muA, muB, muC = MU_NORM * B64[:, 15], MU_NORM * B64[:, 16], MU_NORM * B64[:, 17]
V_A = np.hstack([S, A_in])
V_B = np.hstack([S[:, :3], B_in])   # shares 3 of 4 shared directions
V_C = C_in

Q16, _ = np.linalg.qr(rng.standard_normal((D_OUT, D_OUT)))
Q = Q16[:, :RANK]              # output space shared by A and B
Q_C = Q16[:, RANK:2 * RANK]    # C's own output space (orthogonal)
C_AB = rng.standard_normal((RANK, RANK))   # transferable mapping, same for A/B
C_C = rng.standard_normal((RANK, RANK))
WAs = Q @ C_AB @ V_A.T
WBs = Q @ C_AB @ V_B.T
WCs = Q_C @ C_C @ V_C.T


def gen(V, mu, Wstar, n):
    X = mu[None, :] + rng.standard_normal((n, RANK)) @ V.T
    Ys = X @ Wstar.T                      # noiseless targets
    Y = Ys + SIGMA * rng.standard_normal((n, D_OUT))
    return X, Y, Ys


XtrA, YtrA, _ = gen(V_A, muA, WAs, N_TRAIN)
XtrB, YtrB, _ = gen(V_B, muB, WBs, N_TRAIN)
XtrC, YtrC, _ = gen(V_C, muC, WCs, N_TRAIN)
XteA, _, YsA = gen(V_A, muA, WAs, N_TEST)
XteB, _, YsB = gen(V_B, muB, WBs, N_TEST)
XteC, _, YsC = gen(V_C, muC, WCs, N_TEST)
Xpr, _, YsPr = gen(V_B, muB, WBs, N_TEST)   # related probe: fresh B-region inputs


def ls_fit(X, Y):
    """Min-norm least-squares task solution (the 'learned update' primitive)."""
    return np.linalg.lstsq(X, Y, rcond=None)[0].T


WhatA, WhatB, WhatC = ls_fit(XtrA, YtrA), ls_fit(XtrB, YtrB), ls_fit(XtrC, YtrC)


def accuracy(W, X, Ys):
    pred = X @ W.T
    return max(0.0, 1.0 - float(np.sum((pred - Ys) ** 2)) / float(np.sum(Ys ** 2)))


# ---- arm 1: naive sequential (= unconstrained update) ----
W_after_A = WhatA.copy()
dW_B = WhatB - W_after_A
W_naive = W_after_A + dW_B          # == WhatB

# ---- arm 2: O-LoRA-style strict null projection ----
rho_B = update_overlap_rho(dW_B, W_after_A, dv=DV)
U_A = top_column_basis(W_after_A, DV)
dW_B_orth = dW_B - U_A @ (U_A.T @ dW_B)
W_olora = W_after_A + dW_B_orth

# ---- arm 3: CoDe (consolidation branch + routed experts) ----
W_acc_B, dnull_B, (c2, s2) = code_lora_step(W_after_A, dW_B, t=2, dv=DV)
dW_C = WhatC - W_acc_B
W_acc_C, dnull_C, (c3, s3) = code_lora_step(W_acc_B, dW_C, t=3, dv=DV)
experts = [WhatA, WhatB]   # De-LoRA pool after seeing A, B
protos = [build_prototype(XtrA, n_proto=10, rng=rng),
          build_prototype(XtrB, n_proto=10, rng=rng)]


def routed_predictions(X, W_shared):
    """Per-sample prototype routing; falls back to the shared branch."""
    P = np.zeros((X.shape[0], D_OUT))
    routes, confs = [], []
    for i, x in enumerate(X):
        k, cf, use = route_prototype(x, protos, tau=DEFAULT_TAU)
        routes.append((k, use))
        confs.append(cf)
        P[i] = (experts[k] if use else W_shared) @ x
    return P, routes, confs


def routed_accuracy(X, Ys, W_shared, true_k):
    P, routes, confs = routed_predictions(X, W_shared)
    acc = max(0.0, 1.0 - float(np.sum((P - Ys) ** 2)) / float(np.sum(Ys ** 2)))
    route_ok = sum(1 for k, use in routes if k == true_k and use) / len(routes)
    return acc, route_ok, confs


accB_naive = accuracy(W_naive, XteB, YsB)
accB_olora = accuracy(W_olora, XteB, YsB)
accA_naive = accuracy(W_naive, XteA, YsA)
accA_olora = accuracy(W_olora, XteA, YsA)
accPr_naive = accuracy(W_naive, Xpr, YsPr)
accPr_olora = accuracy(W_olora, Xpr, YsPr)
accA_code, routeA, _ = routed_accuracy(XteA, YsA, W_acc_B, true_k=0)
accB_code, routeB, _ = routed_accuracy(XteB, YsB, W_acc_B, true_k=1)
accPr_code, routePr, _ = routed_accuracy(Xpr, YsPr, W_acc_B, true_k=1)

# OOD probe: task-C region inputs while only A/B experts exist
_, ood_routes, ood_confs = routed_predictions(XteC, W_acc_B)
ood_fallback = sum(1 for _, use in ood_routes if not use) / len(ood_routes)

print("== (a) orthogonality dilemma: strict null projection on related A->B ==")
print(f"  rho_B (B-update energy killed by projection): {rho_B:.4f}")
check("large fraction of B's update energy killed (rho_B >= 0.8)", rho_B >= 0.8)
print(f"  B accuracy: naive/unconstrained {accB_naive:.4f} vs O-LoRA {accB_olora:.4f}")
check("unconstrained B update learns B (>= 0.95)", accB_naive >= 0.95)
check("O-LoRA B accuracy materially worse (gap >= 0.25)",
      accB_olora <= accB_naive - 0.25)
print(f"  A retention: naive {accA_naive:.4f} vs O-LoRA {accA_olora:.4f}")
check("naive sequential forgets A (<= 0.80)", accA_naive <= 0.80)
check("O-LoRA A-retention ~perfect (>= 0.95)", accA_olora >= 0.95)
print(f"  related probe: naive {accPr_naive:.4f} vs O-LoRA {accPr_olora:.4f}")
check("O-LoRA related-probe accuracy materially worse (gap >= 0.25)",
      accPr_olora <= accPr_naive - 0.25)

print("== (b) CoDe fix: consolidation + routed experts ==")
print(f"  A retention: CoDe {accA_code:.4f} vs O-LoRA {accA_olora:.4f}")
check("CoDe A-retention matches O-LoRA (within 0.03)",
      abs(accA_code - accA_olora) <= 0.03)
print(f"  B accuracy: CoDe {accB_code:.4f} vs unconstrained {accB_naive:.4f}")
check("CoDe B accuracy recovers to near unconstrained (within 0.03)",
      abs(accB_code - accB_naive) <= 0.03 and accB_code >= 0.95)
print(f"  routing accuracy: A {routeA:.4f}, B {routeB:.4f}")
check("routing accuracy high (>= 0.95)", routeA >= 0.95 and routeB >= 0.95)
print(f"  related probe: CoDe {accPr_code:.4f} (routed to B expert: {routePr:.4f})")
check("CoDe related-probe recovers (within 0.03 of unconstrained)",
      abs(accPr_code - accPr_naive) <= 0.03)
print(f"  OOD probe (C region, A/B experts only): fallback frac {ood_fallback:.4f}, "
      f"mean max-confidence {np.mean(ood_confs):.4f} (tau={DEFAULT_TAU})")
check("OOD probe falls back to shared branch (>= 0.95)", ood_fallback >= 0.95)
check("OOD confidence below threshold", max(ood_confs) <= DEFAULT_TAU)
# diagnostic mirroring paper Table 5: the shared branch alone still carries A
print(f"  diagnostic: shared-branch-only A accuracy {accuracy(W_acc_B, XteA, YsA):.4f}")

print("== (c) stability (Prop. 1): ||W_acc^(t)||_F bounded by M_t ==")
M1 = float(np.linalg.norm(WhatA))
M2 = max(M1, float(np.linalg.norm(dW_B)))
M3 = max(M2, float(np.linalg.norm(dW_C)))
for t, W, M in [(1, W_after_A, M1), (2, W_acc_B, M2), (3, W_acc_C, M3)]:
    ratio = float(np.linalg.norm(W)) / M
    print(f"  t={t}: ||W_acc||_F={float(np.linalg.norm(W)):.4f}  "
          f"M_t={M:.4f}  ratio={ratio:.4f}")
    check(f"t={t}: ||W_acc||_F <= M_t (Prop. 1)", ratio <= 1.0 + 1e-9)
    if t >= 2:
        check(f"t={t}: well under the bound (ratio <= 0.9)", ratio <= 0.9)

print("== (d) dynamic scaling sanity ==")
for t in range(1, 6):
    c, s = dynamic_scaling(t)
    err = abs(c * c + s * s - 1.0)
    check(f"t={t}: c_t^2 + s_t^2 == 1 (err {err:.1e})", err <= 1e-12)
for name, dnull, dshared in [("B", dnull_B, dW_B), ("C", dnull_C, dW_C)]:
    ratio = float(np.linalg.norm(dnull)) / float(np.linalg.norm(dshared))
    print(f"  ||dW_null||/||dW_shared|| ({name}): {ratio:.4f}")
    check(f"projection never increases norm ({name})", ratio <= 1.0 + 1e-12)
# Pythagorean split from orthogonality <W_prev, dW_null> = 0 (proof of Prop. 1)
lhs = float(np.linalg.norm(W_acc_B) ** 2)
rhs = c2 ** 2 * float(np.linalg.norm(W_after_A) ** 2) + s2 ** 2 * float(np.linalg.norm(dnull_B) ** 2)
perr = abs(lhs - rhs) / lhs
print(f"  Pythagorean split rel. err: {perr:.2e}")
check("orthogonality Pythagorean identity holds", perr <= 1e-6)

print("== control: unrelated task C is (almost) untouched by the projection ==")
U_B = top_column_basis(W_acc_B, DV)
own_C = float(np.sum(((np.eye(D_OUT) - U_B @ U_B.T) @ WhatC) ** 2)) / float(np.sum(WhatC ** 2))
print(f"  fraction of C's own update energy surviving projection: {own_C:.4f}")
check("unrelated task's own update survives (>= 0.9)", own_C >= 0.9)
check("rho in [0,1]",
      0.0 <= rho_B <= 1.0 and 0.0 <= update_overlap_rho(dW_C, W_acc_B, dv=DV) <= 1.0)

print(f"\nALL {len(passed)} CHECKS PASSED")
