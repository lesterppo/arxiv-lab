"""Curated always-loaded context as a capacitated assortment problem.

Paper: arXiv 2610.11007 "Curating Always-Loaded Context for LLM Agents:
A Capacitated Assortment Model with Censored Feedback".

One-line claim: an always-loaded context file (e.g. AGENTS.md) is a
capacitated assortment — appending every positive-standalone instruction
can be arbitrarily worse than the optimal subset, and under censored
feedback, deleting instructions the agent "ignores" inevitably deletes
helpful silent-guardrail instructions too.

What this module implements (the mechanism, not an LLM harness):

  * Instruction model. Instruction i has token cost c_i and standalone
    value v_i. Attention dilution D(T) = 1/(1+T/kappa) captures the
    paper's axioms: (a) adding an instruction never raises any other's
    realized compliance — every loaded instruction's realized value is
    v_i * D(T_S), which strictly decreases as total tokens T_S grow;
    (b) retained instructions incur a per-session setup cost lambda per
    loaded token (each token is charged again in every later round).
    Net value: V(S) = D(T_S) * sum_{i in S} v_i - lambda * T_S.
  * Exact optimal subset via knapsack-style DP over the total-cost
    dimension (the objective depends on S only through (sum c, sum v)).
  * Greedy-append baseline: load every instruction with positive
    standalone value (the "curators grow files by appending" policy).
  * Closed-form worst-case upper bound on the optimal file's instruction
    count, independent of pool size (size_upper_bound). It is valid but
    loose at small token prices — the testable content of the paper's
    bound claim is the saturation of the optimal file size plus the
    n-independent value-density floor (sum v / T >= lambda).
  * Censored-feedback simulator: instruction benefits are observable
    only when the agent visibly follows them; missing instructions
    generate feedback only when their absence causes harm. A
    delete-if-ignored policy provably removes purely preventive
    (silent-guardrail) instructions no matter how much evidence is
    collected.
  * Evidence rule: optimal number of trial sessions before adding a
    candidate instruction, minimizing expected regret
    (exploration cost + wrong-decision cost).

Simulation-honesty note: the dilution shape D(T), the cost/value
distributions and the censoring probabilities are modeling choices that
implement the paper's stated axioms (monotone compliance dilution,
per-token setup cost, censored observability). The module tests the
paper's *mechanism-level* claims (bound, arbitrarily-worse greedy,
inevitable deletion under censoring, evidence scaling) inside that
model — it is not a measurement of any real LLM agent.

numpy-guarded per repo conventions.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.context.curated_context")


# ---------------------------------------------------------------------------
# Module-level constants (numpy-free)
# ---------------------------------------------------------------------------

KAPPA = 500.0        # attention half-saturation constant (tokens)
LAMBDA = 0.02        # per-session setup cost per loaded token
CAPACITY = 4000      # hard attention-capacity budget (tokens)


# ---------------------------------------------------------------------------
# Core assortment model
# ---------------------------------------------------------------------------

def dilution(total_tokens, kappa=KAPPA):
    """Attention-dilution factor D(T) = 1/(1+T/kappa), decreasing in T."""
    return 1.0 / (1.0 + total_tokens / kappa)


def net_value(selected, costs, values, kappa=KAPPA, lam=LAMBDA):
    """Net value V(S) = D(T_S)*sum(v) - lam*T_S of a loaded subset S."""
    t = sum(costs[i] for i in selected)
    v = sum(values[i] for i in selected)
    return dilution(t, kappa) * v - lam * t


def standalone_value(i, costs, values, kappa=KAPPA, lam=LAMBDA):
    """Net value of instruction i loaded alone (the curator's append test)."""
    return dilution(costs[i], kappa) * values[i] - lam * costs[i]


def greedy_append(costs, values, kappa=KAPPA, lam=LAMBDA, capacity=None):
    """Append every instruction with positive standalone value.

    The paper's observed curator behavior ("grow these files by
    appending"). With capacity=None the file grows unboundedly; with a
    capacity the appends stop when the budget fills (in index order).
    """
    chosen = []
    for i in range(len(costs)):
        if standalone_value(i, costs, values, kappa, lam) > 0:
            if capacity is not None and \
                    sum(costs[j] for j in chosen) + costs[i] > capacity:
                break
            chosen.append(i)
    return chosen


def optimal_subset(costs, values, capacity=CAPACITY, kappa=KAPPA, lam=LAMBDA):
    """Exact optimal file: DP over total-token cost.

    best_v[t] = max sum of standalone values achievable with exactly t
    tokens; then V(t) = D(t)*best_v[t] - lam*t is maximized over t.
    Returns (chosen_indices, value, total_tokens).
    """
    n = len(costs)
    cap = int(capacity)
    neg = float("-inf")
    best_v = [neg] * (cap + 1)
    best_v[0] = 0.0
    keep = [[False] * (cap + 1) for _ in range(n)]
    for i in range(n):
        c = int(costs[i])
        v = values[i]
        for t in range(cap, c - 1, -1):
            if best_v[t - c] != neg and best_v[t - c] + v > best_v[t]:
                best_v[t] = best_v[t - c] + v
                keep[i][t] = True
    best_t, best_val = 0, 0.0  # empty file is always feasible
    for t in range(cap + 1):
        if best_v[t] != neg:
            val = dilution(t, kappa) * best_v[t] - lam * t
            if val > best_val:
                best_val, best_t = val, t
    chosen, t = [], best_t
    for i in range(n - 1, -1, -1):
        if t >= 0 and keep[i][t]:
            chosen.append(i)
            t -= int(costs[i])
    chosen.reverse()
    return chosen, best_val, best_t


def size_upper_bound(v_max, c_min, kappa=KAPPA, lam=LAMBDA):
    """Closed-form upper bound on the optimal file's instruction count.

    For any file S with |S| = k: V(S) <= H(k) where
    H(k) = k*v_max/(1+k*c_min/kappa) - lam*k*c_min, because every
    instruction costs at least c_min tokens, is worth at most v_max,
    and dilution is decreasing. H(k) -> -inf as k -> inf (the value term
    saturates at v_max*kappa/c_min while the setup cost grows linearly),
    so beyond K* no file beats the empty file; K* does not depend on the
    number of candidate instructions. Returns ceil(K*).
    """
    import math
    a = c_min / kappa          # dilution slope scale
    b = lam * c_min            # per-instruction setup cost floor
    # k1: beyond this, H(k) < 0 (saturated value can't cover setup cost)
    k1 = v_max * kappa / (lam * c_min * c_min)
    # k2: beyond this, H is strictly decreasing (dH/dk < 0)
    k2 = (math.sqrt(v_max / b) - 1.0) / a
    return int(math.ceil(max(k1, k2))) + 1


def arbitrarily_worse_family(m, kappa=KAPPA, lam=LAMBDA):
    """Instance family where greedy-append is arbitrarily worse than optimal.

    One star instruction (high value, cheap) plus m filler instructions,
    each with small positive standalone value but large token cost.
    Greedy appends all m+1: dilution -> 0 while the token tax grows
    linearly in m, so V(greedy) -> -inf in m while the optimal file
    keeps just the star. Returns (costs, values).
    """
    star_c, star_v = 10, 100.0
    fill_c = 200
    # standalone value exactly +0.5: D(fill_c)*v - lam*fill_c = 0.5
    fill_v = (0.5 + lam * fill_c) / dilution(fill_c, kappa)
    costs = [star_c] + [fill_c] * m
    values = [star_v] + [fill_v] * m
    return costs, values


# ---------------------------------------------------------------------------
# Censored feedback
# ---------------------------------------------------------------------------

def simulate_censored_feedback(rng, instructions, n_sessions):
    """Simulate K sessions of observable feedback under censoring.

    Each instruction dict carries:
      p_follow: per-session probability the agent visibly follows it
                (observable "follow" event),
      q_prevent: per-session probability it silently prevents a harm of
                value harm_value (invisible — a prevented harm produces
                no observable event),
      harm_value: value of each prevented harm.
    Returns per-instruction dicts with observed follow counts and the
    true per-session value delivered.
    """
    out = []
    for ins in instructions:
        follows = int(rng.binomial(n_sessions, ins["p_follow"]))
        prevented = float(rng.binomial(n_sessions, ins["q_prevent"]))
        true_value = prevented * ins["harm_value"] / n_sessions
        out.append({"follows": follows, "prevented": prevented,
                    "true_value_per_session": true_value})
    return out


def delete_if_ignored(feedback):
    """Deletion policy: drop every instruction with zero observed follows.

    The paper's censored-feedback hazard: an instruction that is never
    visibly followed is treated as useless, even when its value is
    purely preventive (p_follow = 0, q_prevent > 0) and hence can never
    generate a follow event no matter how many sessions are observed.
    Returns the indices kept.
    """
    return [i for i, fb in enumerate(feedback) if fb["follows"] > 0]


# ---------------------------------------------------------------------------
# Evidence rule: how many trial sessions before adding an instruction
# ---------------------------------------------------------------------------

def expected_regret(m, mu0, sigma0, sigma_obs, explore_cost, n_mc=20000,
                    seed=0):
    """Expected regret of an explore-then-commit rule with m trial sessions.

    Prior value v ~ N(mu0, sigma0^2); each trial costs explore_cost and
    yields y ~ N(v, sigma_obs^2). After m trials add iff posterior mean
    > 0. Regret = m*explore_cost + |v| on wrong-sign decisions.
    Monte-Carlo over the prior predictive (seeded).
    """
    _need_numpy()
    rng = np.random.default_rng(seed)
    v = rng.normal(mu0, sigma0, size=n_mc)
    if m == 0:
        mu_m = np.full(n_mc, mu0)
    else:
        ybar = v + rng.normal(0.0, sigma_obs / np.sqrt(m), size=n_mc)
        # posterior mean under N(mu0,sigma0^2) prior, N(v,sigma_obs^2/m) data
        prec0, precd = 1.0 / sigma0**2, m / sigma_obs**2
        mu_m = (prec0 * mu0 + precd * ybar) / (prec0 + precd)
    add = mu_m > 0
    wrong = (add & (v < 0)) | (~add & (v > 0))
    return float(m * explore_cost + np.mean(wrong * np.abs(v)))


def optimal_trial_count(mu0, sigma0, sigma_obs, explore_cost, m_max=60,
                       seed=0):
    """m* = argmin_m expected_regret(m): the paper's evidence rule.

    How much evidence to collect before adding a candidate instruction:
    more prior uncertainty (sigma0) pushes m* up, higher per-trial cost
    pushes it down. Returns (m_star, regret_star).
    """
    regrets = [expected_regret(m, mu0, sigma0, sigma_obs, explore_cost,
                               seed=seed) for m in range(m_max + 1)]
    m_star = int(np.argmin(regrets))
    return m_star, regrets[m_star]
