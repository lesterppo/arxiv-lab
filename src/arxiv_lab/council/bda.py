"""Bayesian Dialectical Argumentation council weighting (arXiv:2610.02005).

Paper claim implemented: a multi-LLM council's *typed* deliberation moves
(who proposed / challenged / conceded which answer) are treated as
observations of a classical annotator model with *per-agent* reliabilities
(Dawid-Skene style). Reliability-weighted evidence yields calibrated
posterior probabilities over candidate answers, and persistently unreliable
agents are *inverted* (reliability < 0.5) rather than merely outvoted.

Provides the typed-move annotator model + EM inference (``bda_fit``), a
synthetic council generator (``simulate_council``), the zero-cost
majority-vote baseline (``majority_vote``), and ``brier_score`` for
calibration comparison.

Requires numpy (optional dependency): ``pip install numpy``.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def _need_numpy():
    if np is None:
        raise ImportError(
            "arxiv_lab.council requires numpy (optional dependency); "
            "install it with `pip install numpy`"
        )


def simulate_council(rng, n_cases, n_answers, true_rho, collude_answer=None,
                     challenge_p=1.0, concede_p=1.0):
    """Generate synthetic council deliberation traces.

    Each agent, per case, emits typed moves:
      - propose(a): proposes answer a. P(a | truth y) = rho if a==y else
        (1-rho)/(K-1). A colluding adversarial coalition proposes a single
        fixed wrong answer with prob 0.9.
      - challenge(a): challenges a proposal of a. Reliable agents challenge a
        wrong answer (uniformly); unreliable agents wrongly challenge the
        truth: P(challenge a | y) = rho/(K-1) if a != y else (1-rho).
      - concede(a): concedes to a proposal of a; same likelihood as propose.

    ``rng``: a ``numpy.random.Generator``. Returns ``(truths, traces)``.
    """
    _need_numpy()
    n_agents = len(true_rho)
    adv = set(np.where(np.asarray(true_rho) < 0.5)[0])
    traces = []
    truths = rng.integers(n_answers, size=n_cases)
    for y in truths:
        obs = []
        for i in range(n_agents):
            rho = true_rho[i]
            wrong = [a for a in range(n_answers) if a != y]
            if i in adv and collude_answer is not None:
                prop = collude_answer if rng.random() < 0.9 else rng.choice(wrong)
            else:
                prop = y if rng.random() < rho else rng.choice(wrong)
            obs.append((i, prop, 'propose'))
            if rng.random() < challenge_p:
                tgt = rng.choice(wrong) if rng.random() < rho else y
                obs.append((i, int(tgt), 'challenge'))
            if rng.random() < concede_p:
                cg = y if rng.random() < rho else rng.choice(wrong)
                obs.append((i, int(cg), 'concede'))
        traces.append(obs)
    return truths, traces


def bda_fit(n_answers, obs, max_iter=200, tol=1e-8):
    """EM inference of per-agent reliabilities + posterior over answers.

    Likelihoods (agent i, reliability rho_i):
      propose/concede a:  rho_i            if a == y
                          (1-rho_i)/(K-1)  otherwise
      challenge a:        (1-rho_i)        if a == y   (wrongly challenges truth)
                          rho_i/(K-1)      otherwise   (rightly challenges wrong)

    ``obs``: list of ``(agent_id, answer, move_type)`` with move_type in
    {'propose', 'challenge', 'concede'}. Returns ``(posterior[K], rho[N])``.
    """
    _need_numpy()
    n_agents = max(i for (i, _, _) in obs) + 1
    K = n_answers
    rho = np.full(n_agents, 0.6)
    logK = np.log(K)
    for _ in range(max_iter):
        logp = np.zeros(K)
        for (i, a, t) in obs:
            r = rho[i]
            if t in ('propose', 'concede'):
                logp += np.where(np.arange(K) == a, np.log(r),
                                 np.log((1 - r) / (K - 1)))
            else:  # challenge
                logp += np.where(np.arange(K) == a, np.log(1 - r),
                                 np.log(r / (K - 1)) if K > 1 else -logK)
        logp -= logp.max()
        post = np.exp(logp)
        post /= post.sum()
        num = np.zeros(n_agents)
        den = np.zeros(n_agents)
        for (i, a, t) in obs:
            den[i] += 1.0
            num[i] += post[a] if t in ('propose', 'concede') else (1.0 - post[a])
        new = np.clip(num / np.maximum(den, 1e-9), 0.02, 0.98)
        if np.max(np.abs(new - rho)) < tol:
            rho = new
            break
        rho = new
    return post, rho


def majority_vote(n_answers, obs):
    """Zero-cost baseline: majority over proposals, agreement fraction as
    confidence. Returns ``(answer, confidence)``."""
    _need_numpy()
    props = [a for (_, a, t) in obs if t == 'propose']
    counts = np.bincount(props, minlength=n_answers)
    y = int(counts.argmax())
    return y, float(counts[y] / len(props))


def brier_score(items):
    """Brier score over ``items``: list of (confidence, correct_bool)."""
    _need_numpy()
    return float(np.mean([(p - (1.0 if ok else 0.0)) ** 2 for p, ok in items]))
