"""Self-consistency certification under correlated reasoning.

Reproduces the core mechanism of arXiv:2611.26592 ("Realized versus
Population Reliability: Anytime-Valid Self-Consistency Certification
under Correlated Reasoning", Agbeyegbe & Agbeyegbe, 2026):

* Self-consistency certification treats repeated sampling as sequential
  evidence accumulation and stops when an anytime-valid (betting)
  e-process crosses 1/eps, with Type I error control by Ville's
  inequality -- PROVIDED the rollouts are conditionally independent.
* In practice reasoning streams are positively correlated through a
  per-unit latent state (random-effects / intraclass-correlation
  structure): exchangeable, equi-correlated increments with
  correlation rho. Correlation splits the single null into a
  *realized* null (about the drawn belief mu(Z)) and a *population*
  null (about average model behaviour E[mu(Z)] = theta), which
  coincide only when rho = 0.
* The paper's claims, implemented here:
  (i)   the standard betting e-process stays anytime-valid for the
        realized null, not the population null;
  (ii)  read as a population-null test, a suitable mixture has
        asymptotic crossing probability at least pi = P(mu(Z) > 0),
        potentially exceeding nominal eps;
  (iii) the within-unit mean's effective sample size saturates at
        1/rho, so extra within-unit rollouts cannot restore
        population validity;
  (iv)  a prompt-level e-process (one rollout per prompt) restores
        population validity, with optimal allocation at one rollout
        per prompt under a pure rollout budget.
* Empirical reference points from the paper (Claude Haiku 4.5
  rollouts): GSM8K rho-hat = 0.435, theta-hat = +0.96 (gap dormant);
  AIME rho-hat = 0.480, theta-hat = -0.15, single-unit rule crosses
  on 24% of prompts at nominal eps = 0.05, prompt-level 0.00.

Model (exactly the paper's random-effects structure, latent scale
normalised so tau**2 + sigma**2 = 1):
  mu_j ~ N(theta, tau**2)            per-prompt latent belief
  x_ij = +1 if mu_j + sigma * z_ij > 0 else -1,  z_ij ~ N(0, 1)
  rho = tau**2 / (tau**2 + sigma**2)  (intraclass correlation)
  population null  <=> theta <= 0;  realized null <=> mu_j <= 0.
  pi = P(mu(Z) > 0) = Phi(theta / tau).

Betting e-process: a mixture over a fixed lambda grid of
E_n(lambda) = prod_i (1 + lambda * x_i), x_i in {-1, +1}.
The mixture (average over the grid) is an e-process under the
null that E[x_i | past] <= 0, i.e. under the realized null when
conditioning on the drawn unit, and under the population null
for the prompt-level construction (one independent increment per
prompt). The finite grid is a practical approximation of the
paper's continuous mixture; the asymptotic lower bound pi is
approached as the grid refines and N grows.

Stdlib only. Deterministic: all randomness flows through
SplitMix64 (same arithmetic as arxiv_lab.agent.anytime_stopping),
with Gaussian draws via Acklam's inverse-CDF approximation on a
single uniform -- no cached state, so the Rust port can reproduce
the stream bit-for-bit.
"""

import math

__all__ = [
    "SplitMix64",
    "LAMBDA_GRID",
    "normal_cdf",
    "normal_ppf",
    "rho_from_tau_sigma",
    "tau_sigma_from_rho",
    "pi_positive",
    "betting_eprocess",
    "crossing_step",
    "simulate_unit",
    "simulate_prompt_level",
    "ess_within_unit",
    "within_unit_mean_var",
    "icc_anova",
    "theta_hat_from_accuracy",
    "run_single_unit_experiment",
    "run_prompt_level_experiment",
]

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15

# Finite lambda grid approximating the paper's mixture over (0, 1).
# Includes a small lambda (0.05) so units only slightly above chance
# still accumulate evidence, and large lambdas for fast growth when
# the unit is clearly above chance.
LAMBDA_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

_EPS_DEFAULT = 0.05


class SplitMix64:
    """SplitMix64 PRNG (Stafford). Same arithmetic as the sibling
    module arxiv_lab.agent.anytime_stopping, and as the Rust port.

    ``uniform()`` returns a float in [0, 1) with 53 bits of
    randomness: ``(next_u64() >> 11) * 2**-53``.
    """

    def __init__(self, seed):
        self.state = int(seed) & _MASK64

    def next_u64(self):
        self.state = (self.state + _GOLDEN) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def normal(self):
        """Standard normal via Acklam's inverse CDF on one uniform."""
        u = self.uniform()
        if u <= 0.0:
            u = 2.0 ** -53
        elif u >= 1.0:
            u = 1.0 - 2.0 ** -53
        return normal_ppf(u)


# ---------------------------------------------------------------------------
# Normal distribution helpers (Acklam's rational approximation).
# ---------------------------------------------------------------------------

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p):
    """Inverse standard normal CDF (Acklam 2003), |err| < 1.15e-9."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                                + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


# ---------------------------------------------------------------------------
# Random-effects parameterisation.
# ---------------------------------------------------------------------------

def rho_from_tau_sigma(tau, sigma):
    return (tau * tau) / (tau * tau + sigma * sigma)


def tau_sigma_from_rho(rho):
    """tau, sigma on the normalised scale tau**2 + sigma**2 = 1."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    return math.sqrt(rho), math.sqrt(1.0 - rho)


def pi_positive(theta, tau):
    """pi = P(mu(Z) > 0) for mu ~ N(theta, tau**2)."""
    if tau <= 0.0:
        return 1.0 if theta > 0.0 else 0.0
    return normal_cdf(theta / tau)


# ---------------------------------------------------------------------------
# Betting e-process.
# ---------------------------------------------------------------------------

def betting_eprocess(increments, lambdas=LAMBDA_GRID):
    """Wealth path of the mixture betting e-process.

    Returns [E_0=1.0, E_1, ..., E_n] where E_n is the average over
    the lambda grid of prod_i (1 + lambda * x_i). Increments must
    lie in [-1, 1]; lambda < 1 keeps every factor positive.
    """
    log_w = [0.0] * len(lambdas)
    path = [1.0]
    log_plus = [math.log1p(lam) for lam in lambdas]
    log_minus = [math.log1p(-lam) for lam in lambdas]
    for x in increments:
        if x > 0.0:
            for j in range(len(lambdas)):
                log_w[j] += log_plus[j] * x
        elif x < 0.0:
            for j in range(len(lambdas)):
                log_w[j] += log_minus[j] * (-x)
        # x == 0 contributes nothing.
        m = max(log_w)
        path.append(sum(math.exp(lw - m) for lw in log_w)
                    / len(lambdas) * math.exp(m) if m > -745.0 else 0.0)
    return path


def crossing_step(increments, eps=_EPS_DEFAULT, lambdas=LAMBDA_GRID):
    """First 1-indexed step where the mixture e-process reaches
    1/eps, or None if it never crosses within the increments."""
    threshold = 1.0 / eps
    log_w = [0.0] * len(lambdas)
    log_plus = [math.log1p(lam) for lam in lambdas]
    log_minus = [math.log1p(-lam) for lam in lambdas]
    for i, x in enumerate(increments, start=1):
        if x > 0.0:
            for j in range(len(lambdas)):
                log_w[j] += log_plus[j] * x
        elif x < 0.0:
            for j in range(len(lambdas)):
                log_w[j] += log_minus[j] * (-x)
        m = max(log_w)
        if m > -745.0:
            wealth = (sum(math.exp(lw - m) for lw in log_w)
                      / len(lambdas) * math.exp(m))
            if wealth >= threshold:
                return i
    return None


# ---------------------------------------------------------------------------
# Simulation of the paper's random-effects rollout model.
# ---------------------------------------------------------------------------

def simulate_unit(rng, theta, tau, sigma, n, binary=True):
    """One unit (prompt): draw its latent belief mu, then n rollouts.

    Returns (mu, increments). With binary=True increments are +-1
    via the threshold rule x = +1 iff mu + sigma*z > 0; with
    binary=False they are the continuous latent values mu + sigma*z
    (used for the ESS / variance calculation, claim (iii)).
    """
    mu = theta + tau * rng.normal() if tau > 0.0 else theta
    incs = []
    for _ in range(n):
        z = rng.normal()
        if binary:
            incs.append(1.0 if mu + sigma * z > 0.0 else -1.0)
        else:
            incs.append(mu + sigma * z)
    return mu, incs


def simulate_prompt_level(rng, theta, tau, sigma, n_prompts):
    """One rollout per fresh prompt: n_prompts independent increments
    (the prompt-level e-process, claim (iv))."""
    out = []
    for _ in range(n_prompts):
        _, incs = simulate_unit(rng, theta, tau, sigma, 1)
        out.append(incs[0])
    return out


def ess_within_unit(n, rho):
    """Effective sample size of n equi-correlated increments:
    ESS = n / (1 + (n-1) rho) -> 1/rho as n -> infinity."""
    if n <= 0:
        return 0.0
    return n / (1.0 + (n - 1) * rho)


def within_unit_mean_var(n, rho):
    """Var of the within-unit mean of n equi-correlated unit-variance
    increments: rho + (1 - rho)/n. Saturates at rho."""
    if n <= 0:
        raise ValueError("n must be positive")
    return rho + (1.0 - rho) / n


# ---------------------------------------------------------------------------
# Estimators for the model-backed leg (real rollouts).
# ---------------------------------------------------------------------------

def icc_anova(outcomes):
    """Intraclass correlation of binary outcomes via one-way random
    effects ANOVA. ``outcomes`` is a list of per-prompt lists of 0/1.
    Returns rho-hat; None if degenerate (fewer than 2 prompts, or
    no within-prompt replication)."""
    groups = [g for g in outcomes if len(g) >= 1]
    k = len(groups)
    if k < 2:
        return None
    ns = [len(g) for g in groups]
    n_total = sum(ns)
    if n_total <= k:
        return None
    grand = sum(sum(g) for g in groups) / n_total
    ss_between = sum(n * ((sum(g) / n) - grand) ** 2
                     for g, n in zip(groups, ns))
    ss_within = sum(sum((y - sum(g) / len(g)) ** 2 for y in g)
                    for g in groups)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)
    n0 = (n_total - sum(n * n for n in ns) / n_total) / (k - 1)
    if ms_between + (n0 - 1.0) * ms_within == 0.0:
        return 0.0
    rho = (ms_between - ms_within) / (ms_between + (n0 - 1.0) * ms_within)
    return max(0.0, min(1.0, rho))


def theta_hat_from_accuracy(p_bar):
    """Latent-scale theta-hat from mean correctness p_bar, on the
    normalised scale tau**2 + sigma**2 = 1: p_bar = Phi(theta), so
    theta-hat = Phi^{-1}(p_bar)."""
    p = min(max(p_bar, 1e-6), 1.0 - 1e-6)
    return normal_ppf(p)


# ---------------------------------------------------------------------------
# Experiments.
# ---------------------------------------------------------------------------

def run_single_unit_experiment(seed, theta, rho, n_rollouts, n_units,
                               eps=_EPS_DEFAULT, lambdas=LAMBDA_GRID):
    """Single-unit rule: for each unit, run up to n_rollouts rollouts
    of ONE prompt and check whether the betting e-process crosses.

    Returns crossing rates overall and split by whether the realized
    null holds (mu <= 0) or not, plus pi = P(mu > 0)."""
    tau, sigma = tau_sigma_from_rho(rho)
    rng = SplitMix64(seed)
    crossed = 0
    crossed_real_null = 0
    n_real_null = 0
    crossed_real_alt = 0
    n_real_alt = 0
    for _ in range(n_units):
        mu, incs = simulate_unit(rng, theta, tau, sigma, n_rollouts)
        hit = crossing_step(incs, eps=eps, lambdas=lambdas) is not None
        crossed += 1 if hit else 0
        if mu <= 0.0:
            n_real_null += 1
            crossed_real_null += 1 if hit else 0
        else:
            n_real_alt += 1
            crossed_real_alt += 1 if hit else 0
    return {
        "theta": theta,
        "rho": rho,
        "n_rollouts": n_rollouts,
        "n_units": n_units,
        "eps": eps,
        "pi": pi_positive(theta, tau),
        "crossing_rate": crossed / n_units,
        "n_realized_null": n_real_null,
        "crossing_rate_realized_null": (
            crossed_real_null / n_real_null if n_real_null else None),
        "n_realized_alt": n_real_alt,
        "crossing_rate_realized_alt": (
            crossed_real_alt / n_real_alt if n_real_alt else None),
    }


def run_prompt_level_experiment(seed, theta, rho, n_prompts, n_reps,
                                eps=_EPS_DEFAULT, lambdas=LAMBDA_GRID):
    """Prompt-level rule: each rep accumulates one rollout per fresh
    prompt; check whether the betting e-process over the n_prompts
    independent increments crosses."""
    tau, sigma = tau_sigma_from_rho(rho)
    rng = SplitMix64(seed)
    crossed = 0
    for _ in range(n_reps):
        incs = simulate_prompt_level(rng, theta, tau, sigma, n_prompts)
        if crossing_step(incs, eps=eps, lambdas=lambdas) is not None:
            crossed += 1
    return {
        "theta": theta,
        "rho": rho,
        "n_prompts": n_prompts,
        "n_reps": n_reps,
        "eps": eps,
        "crossing_rate": crossed / n_reps,
    }
