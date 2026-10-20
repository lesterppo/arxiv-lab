"""Deterministic CPU test of the WSR-Tune core mechanism (arXiv 2610.26443).

Paper claim: reparameterize so safety-relevant information concentrates in a
few directions, freeze those directions, update complementary directions ->
safety preserved, task learned.

This test operationalizes the claim on a toy linear adapter with numpy:
  * safety gradient G_s -> top-k SVD directions = "safety-critical dirs"
  * WSR update: project the adapter delta orthogonal to those directions
  * assert: (1) projection is idempotent and exactly orthogonal to the
    safety directions; (2) a plain task step moves the weight ALONG the
    safety direction while the projected step does NOT (safety component
    unchanged to 1e-12); (3) both steps reduce the synthetic task loss.
"""
import numpy as np

rng = np.random.RandomState(0)
IN, R, OUT, K = 32, 8, 32, 2


def safety_directions(Gs, k=K):
    u, s, vh = np.linalg.svd(Gs, full_matrices=False)
    return u[:, :k], vh[:k, :].T  # U: out->k, V: in->k


def project_A(A, V):
    P = np.eye(V.shape[0]) - V @ V.T
    return A @ P


def project_B(B, U):
    P = np.eye(U.shape[0]) - U @ U.T
    return P @ B


def main():
    # toy adapter delta = B @ A ; safety gradient matrices (fixed)
    A = rng.randn(R, IN)
    B = rng.randn(OUT, R)
    Gs_A = rng.randn(R, IN)
    Gs_B = rng.randn(OUT, R)
    Ua, Va = safety_directions(Gs_A)
    Ub, Vb = safety_directions(Gs_B)

    # (1) idempotence + orthogonality
    A2 = project_A(project_A(A, Va), Va)
    assert np.allclose(A2, project_A(A, Va)), "A projection not idempotent"
    assert np.allclose(project_A(A, Va) @ Va, 0, atol=1e-12), \
        "A projection not orthogonal to safety dirs"
    assert np.allclose(Ub.T @ project_B(B, Ub), 0, atol=1e-12), \
        "B projection not orthogonal to safety dirs"

    # (2)/(3): REAL gradient step on a synthetic task; plain leaks into
    # safety dirs, projected step does not; both must reduce task loss.
    X = rng.randn(64, IN)
    W_star = rng.randn(IN, OUT)  # task target
    N = X.shape[0]
    Y = X @ A.T @ B.T
    E = Y - X @ W_star
    G = 2.0 * E / N                      # dL/dY
    G_A = B.T @ G.T @ X                  # dL/dA, R x IN
    G_B = G.T @ X @ A.T                  # dL/dB, OUT x R
    lr = 1e-3
    dA_task, dB_task = -lr * G_A, -lr * G_B
    def loss(Aw, Bw):
        return float(np.mean((X @ (Aw.T @ Bw.T) - X @ W_star) ** 2))
    l0 = loss(A, B)
    l_plain = loss(A + dA_task, B + dB_task)
    proj_dA = project_A(dA_task, Va)
    proj_dB = project_B(dB_task, Ub)
    l_wsr = loss(A + proj_dA, B + proj_dB)
    assert l_plain < l0, f"plain step must reduce loss: {l0:.4f} -> {l_plain:.4f}"
    assert l_wsr < l0, f"wsr step must reduce loss: {l0:.4f} -> {l_wsr:.4f}"

    plain_align_A = float(np.abs(Va.T @ dA_task.T).sum())
    wsr_align_A = float(np.abs(Va.T @ proj_dA.T).sum())
    plain_align_B = float(np.abs(Ub.T @ dB_task).sum())
    wsr_align_B = float(np.abs(Ub.T @ proj_dB).sum())
    assert plain_align_A > 1e-9, "test degenerate: task grad already orthogonal"
    assert wsr_align_A < 1e-10, f"WSR A-update leaks into safety dirs: {wsr_align_A}"
    assert wsr_align_B < 1e-10, f"WSR B-update leaks into safety dirs: {wsr_align_B}"

    print(f"PASS wsr-core: idempotent+orthogonal | "
          f"safety-leak plain=({plain_align_A:.2e},{plain_align_B:.2e}) "
          f"wsr=({wsr_align_A:.2e},{wsr_align_B:.2e}) | "
          f"task loss {l0:.4f} -> plain {l_plain:.4f}, wsr {l_wsr:.4f}")


if __name__ == "__main__":
    main()
