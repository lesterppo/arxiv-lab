//! Self-consistency certification under correlated reasoning.
//!
//! Rust port of `arxiv_lab.agent.selfconsistency` (arXiv:2611.26592,
//! "Realized versus Population Reliability: Anytime-Valid
//! Self-Consistency Certification under Correlated Reasoning",
//! Agbeyegbe & Agbeyegbe, 2026). Python stays the primary verdict
//! carrier; this port cross-checks it. A failed Rust leg would never
//! overturn the Python verdict.
//!
//! Ported faithfully, in the same order of operations, so that the
//! seeded simulation stream reproduces the Python track bit-for-bit:
//! SplitMix64, Acklam's inverse normal CDF on one uniform per
//! Gaussian draw, the mixture betting e-process over the same lambda
//! grid, and the random-effects rollout model (latent belief mu per
//! prompt, equi-correlated increments with correlation rho).
//!
//! Std only — no crates. The one function Rust's std lacks is `erf`
//! (needed by `normal_cdf`, which Python implements as
//! `0.5 * (1 + math.erf(x / sqrt(2)))`); it is called from the
//! system libm through a tiny `extern "C"` declaration, i.e. the
//! exact function CPython's `math.erf` calls. `ln`, `ln_1p`, `exp`
//! and `sqrt` likewise resolve to the same libm routines CPython
//! uses, which is what makes bit-exact parity possible.

pub const LAMBDA_GRID: [f64; 9] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

pub const EPS_DEFAULT: f64 = 0.05;

const MASK64: u64 = u64::MAX;
const GOLDEN: u64 = 0x9E3779B97F4A7C15;

// libm `erf`, the same routine CPython's `math.erf` calls.
// Rust's std has no `erf`; linking it here (no crates involved)
// keeps `normal_cdf` bit-identical to Python's.
#[link(name = "m")]
extern "C" {
    fn erf(x: f64) -> f64;
}

// ---------------------------------------------------------------------------
// SplitMix64 PRNG (Stafford). Same arithmetic as the Python module.
// ``uniform()`` returns a float in [0, 1) with 53 bits of randomness:
// ``(next_u64() >> 11) * 2**-53``.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 {
            state: seed & MASK64,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }

    /// Standard normal via Acklam's inverse CDF on one uniform.
    pub fn normal(&mut self) -> f64 {
        let mut u = self.uniform();
        if u <= 0.0 {
            u = 2f64.powi(-53);
        } else if u >= 1.0 {
            u = 1.0 - 2f64.powi(-53);
        }
        normal_ppf(u)
    }
}

// ---------------------------------------------------------------------------
// Normal distribution helpers (Acklam's rational approximation).
// ---------------------------------------------------------------------------

/// CPython 3.12's `sum()` over floats, bit-for-bit: Neumaier
/// compensated summation. Rust's naive `.sum()` differs by 1 ulp
/// often enough to break exact parity (verified: this exact
/// formulation matches CPython's `sum()` on 20,000 random float
/// lists with 0 mismatches, and on the e-process term list).
/// Use this wherever the Python source calls `sum()` on floats.
pub fn py_sum<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for x in xs {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// `0.5 * (1 + erf(x / sqrt(2)))`, exactly as Python's `normal_cdf`.
pub fn normal_cdf(x: f64) -> f64 {
    // SAFETY: erf is a pure function from the system libm.
    let e = unsafe { erf(x / 2.0f64.sqrt()) };
    0.5 * (1.0 + e)
}

/// Inverse standard normal CDF (Acklam 2003), |err| < 1.15e-9.
/// Operation order matches the Python source term for term, so the
/// result is bit-identical to CPython's `math.log`/`math.sqrt` path.
pub fn normal_ppf(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "p must be in (0, 1)");
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
        / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
}

// ---------------------------------------------------------------------------
// Random-effects parameterisation.
// ---------------------------------------------------------------------------

pub fn rho_from_tau_sigma(tau: f64, sigma: f64) -> f64 {
    (tau * tau) / (tau * tau + sigma * sigma)
}

/// tau, sigma on the normalised scale tau**2 + sigma**2 = 1.
pub fn tau_sigma_from_rho(rho: f64) -> (f64, f64) {
    assert!((0.0..=1.0).contains(&rho), "rho must be in [0, 1]");
    (rho.sqrt(), (1.0 - rho).sqrt())
}

/// pi = P(mu(Z) > 0) for mu ~ N(theta, tau**2).
pub fn pi_positive(theta: f64, tau: f64) -> f64 {
    if tau <= 0.0 {
        return if theta > 0.0 { 1.0 } else { 0.0 };
    }
    normal_cdf(theta / tau)
}

// ---------------------------------------------------------------------------
// Betting e-process.
// ---------------------------------------------------------------------------

/// Wealth path of the mixture betting e-process.
///
/// Returns [E_0=1.0, E_1, ..., E_n] where E_n is the average over
/// the lambda grid of prod_i (1 + lambda * x_i). Computed in
/// log-space with the same max-shift as Python, in the same order.
pub fn betting_eprocess(increments: &[f64], lambdas: &[f64]) -> Vec<f64> {
    let mut log_w = vec![0.0f64; lambdas.len()];
    let log_plus: Vec<f64> = lambdas.iter().map(|lam| lam.ln_1p()).collect();
    let log_minus: Vec<f64> = lambdas.iter().map(|lam| (-lam).ln_1p()).collect();
    let mut path = vec![1.0f64];
    for &x in increments {
        if x > 0.0 {
            for j in 0..lambdas.len() {
                log_w[j] += log_plus[j] * x;
            }
        } else if x < 0.0 {
            for j in 0..lambdas.len() {
                log_w[j] += log_minus[j] * (-x);
            }
        }
        // x == 0 contributes nothing.
        let m = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let v = if m > -745.0 {
            // Python's sum() is Neumaier-compensated (CPython 3.12);
            // see py_sum. A naive sum differs by 1 ulp here.
            let s = py_sum(log_w.iter().map(|lw| (lw - m).exp()));
            s / lambdas.len() as f64 * m.exp()
        } else {
            0.0
        };
        path.push(v);
    }
    path
}

/// First 1-indexed step where the mixture e-process reaches 1/eps,
/// or None if it never crosses within the increments.
pub fn crossing_step(increments: &[f64], eps: f64, lambdas: &[f64]) -> Option<usize> {
    let threshold = 1.0 / eps;
    let mut log_w = vec![0.0f64; lambdas.len()];
    let log_plus: Vec<f64> = lambdas.iter().map(|lam| lam.ln_1p()).collect();
    let log_minus: Vec<f64> = lambdas.iter().map(|lam| (-lam).ln_1p()).collect();
    for (i, &x) in increments.iter().enumerate() {
        if x > 0.0 {
            for j in 0..lambdas.len() {
                log_w[j] += log_plus[j] * x;
            }
        } else if x < 0.0 {
            for j in 0..lambdas.len() {
                log_w[j] += log_minus[j] * (-x);
            }
        }
        let m = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if m > -745.0 {
            let s = py_sum(log_w.iter().map(|lw| (lw - m).exp()));
            let wealth = s / lambdas.len() as f64 * m.exp();
            if wealth >= threshold {
                return Some(i + 1);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Simulation of the paper's random-effects rollout model.
// ---------------------------------------------------------------------------

/// One unit (prompt): draw its latent belief mu, then n rollouts.
///
/// Returns (mu, increments). With binary=true increments are +-1
/// via the threshold rule x = +1 iff mu + sigma*z > 0; with
/// binary=false they are the continuous latent values mu + sigma*z.
/// NOTE: when tau == 0 no Gaussian draw is consumed for mu, exactly
/// as in Python — the RNG stream alignment depends on this.
pub fn simulate_unit(
    rng: &mut SplitMix64,
    theta: f64,
    tau: f64,
    sigma: f64,
    n: usize,
    binary: bool,
) -> (f64, Vec<f64>) {
    let mu = if tau > 0.0 {
        theta + tau * rng.normal()
    } else {
        theta
    };
    let mut incs = Vec::with_capacity(n);
    for _ in 0..n {
        let z = rng.normal();
        if binary {
            incs.push(if mu + sigma * z > 0.0 { 1.0 } else { -1.0 });
        } else {
            incs.push(mu + sigma * z);
        }
    }
    (mu, incs)
}

/// One rollout per fresh prompt: n_prompts independent increments
/// (the prompt-level e-process, claim (iv)).
pub fn simulate_prompt_level(
    rng: &mut SplitMix64,
    theta: f64,
    tau: f64,
    sigma: f64,
    n_prompts: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_prompts);
    for _ in 0..n_prompts {
        let (_, incs) = simulate_unit(rng, theta, tau, sigma, 1, true);
        out.push(incs[0]);
    }
    out
}

/// Effective sample size of n equi-correlated increments:
/// ESS = n / (1 + (n-1) rho) -> 1/rho as n -> infinity.
pub fn ess_within_unit(n: usize, rho: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    n as f64 / (1.0 + (n - 1) as f64 * rho)
}

/// Var of the within-unit mean of n equi-correlated unit-variance
/// increments: rho + (1 - rho)/n. Saturates at rho.
pub fn within_unit_mean_var(n: usize, rho: f64) -> f64 {
    assert!(n > 0, "n must be positive");
    rho + (1.0 - rho) / n as f64
}

// ---------------------------------------------------------------------------
// Estimators for the model-backed leg (real rollouts).
// ---------------------------------------------------------------------------

/// Intraclass correlation of binary outcomes via one-way random
/// effects ANOVA. ``outcomes`` is a list of per-prompt lists of 0/1.
/// Returns rho-hat; None if degenerate (fewer than 2 prompts, or
/// no within-prompt replication).
pub fn icc_anova(outcomes: &[Vec<f64>]) -> Option<f64> {
    let groups: Vec<&Vec<f64>> = outcomes.iter().filter(|g| !g.is_empty()).collect();
    let k = groups.len();
    if k < 2 {
        return None;
    }
    let ns: Vec<usize> = groups.iter().map(|g| g.len()).collect();
    let n_total: usize = ns.iter().sum();
    if n_total <= k {
        return None;
    }
    let n_total_f = n_total as f64;
    let mut total = 0.0f64;
    for g in &groups {
        for y in g.iter() {
            total += *y;
        }
    }
    let grand = total / n_total_f;
    // Python: sum(n * ((sum(g)/n) - grand) ** 2 ...) — a float
    // sum, so Neumaier (py_sum), not a naive accumulation.
    let ss_between = py_sum(groups.iter().zip(ns.iter()).map(|(g, n)| {
        let mut s = 0.0f64;
        for y in g.iter() {
            s += *y;
        }
        let d = s / *n as f64 - grand;
        *n as f64 * d * d
    }));
    let ss_within = py_sum(groups.iter().map(|g| {
        let mut s = 0.0f64;
        for y in g.iter() {
            s += *y;
        }
        let mean = s / g.len() as f64;
        py_sum(g.iter().map(|y| {
            let d = y - mean;
            d * d
        }))
    }));
    let ms_between = ss_between / (k - 1) as f64;
    let ms_within = ss_within / (n_total_f - k as f64);
    let mut sum_n2 = 0.0f64;
    for n in &ns {
        sum_n2 += *n as f64 * *n as f64;
    }
    let n0 = (n_total_f - sum_n2 / n_total_f) / (k - 1) as f64;
    if ms_between + (n0 - 1.0) * ms_within == 0.0 {
        return Some(0.0);
    }
    let rho = (ms_between - ms_within) / (ms_between + (n0 - 1.0) * ms_within);
    Some(rho.max(0.0).min(1.0))
}

/// Latent-scale theta-hat from mean correctness p_bar, on the
/// normalised scale tau**2 + sigma**2 = 1: p_bar = Phi(theta), so
/// theta-hat = Phi^{-1}(p_bar).
pub fn theta_hat_from_accuracy(p_bar: f64) -> f64 {
    let p = p_bar.max(1e-6).min(1.0 - 1e-6);
    normal_ppf(p)
}

// ---------------------------------------------------------------------------
// Experiments.
// ---------------------------------------------------------------------------

/// Result of `run_single_unit_experiment`, field-for-field the
/// Python dict (``crossing_rate_realized_null`` / ``_alt`` are None
/// when the corresponding group is empty).
#[derive(Clone, Debug, PartialEq)]
pub struct SingleUnitResult {
    pub theta: f64,
    pub rho: f64,
    pub n_rollouts: usize,
    pub n_units: usize,
    pub eps: f64,
    pub pi: f64,
    pub crossing_rate: f64,
    pub n_realized_null: usize,
    pub crossing_rate_realized_null: Option<f64>,
    pub n_realized_alt: usize,
    pub crossing_rate_realized_alt: Option<f64>,
}

/// Single-unit rule: for each unit, run up to n_rollouts rollouts
/// of ONE prompt and check whether the betting e-process crosses.
pub fn run_single_unit_experiment(
    seed: u64,
    theta: f64,
    rho: f64,
    n_rollouts: usize,
    n_units: usize,
    eps: f64,
    lambdas: &[f64],
) -> SingleUnitResult {
    let (tau, sigma) = tau_sigma_from_rho(rho);
    let mut rng = SplitMix64::new(seed);
    let mut crossed = 0usize;
    let mut crossed_real_null = 0usize;
    let mut n_real_null = 0usize;
    let mut crossed_real_alt = 0usize;
    let mut n_real_alt = 0usize;
    for _ in 0..n_units {
        let (mu, incs) = simulate_unit(&mut rng, theta, tau, sigma, n_rollouts, true);
        let hit = crossing_step(&incs, eps, lambdas).is_some();
        if hit {
            crossed += 1;
        }
        if mu <= 0.0 {
            n_real_null += 1;
            if hit {
                crossed_real_null += 1;
            }
        } else {
            n_real_alt += 1;
            if hit {
                crossed_real_alt += 1;
            }
        }
    }
    SingleUnitResult {
        theta,
        rho,
        n_rollouts,
        n_units,
        eps,
        pi: pi_positive(theta, tau),
        crossing_rate: crossed as f64 / n_units as f64,
        n_realized_null: n_real_null,
        crossing_rate_realized_null: if n_real_null > 0 {
            Some(crossed_real_null as f64 / n_real_null as f64)
        } else {
            None
        },
        n_realized_alt: n_real_alt,
        crossing_rate_realized_alt: if n_real_alt > 0 {
            Some(crossed_real_alt as f64 / n_real_alt as f64)
        } else {
            None
        },
    }
}

/// Result of `run_prompt_level_experiment`.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptLevelResult {
    pub theta: f64,
    pub rho: f64,
    pub n_prompts: usize,
    pub n_reps: usize,
    pub eps: f64,
    pub crossing_rate: f64,
}

/// Prompt-level rule: each rep accumulates one rollout per fresh
/// prompt; check whether the betting e-process over the n_prompts
/// independent increments crosses.
pub fn run_prompt_level_experiment(
    seed: u64,
    theta: f64,
    rho: f64,
    n_prompts: usize,
    n_reps: usize,
    eps: f64,
    lambdas: &[f64],
) -> PromptLevelResult {
    let (tau, sigma) = tau_sigma_from_rho(rho);
    let mut rng = SplitMix64::new(seed);
    let mut crossed = 0usize;
    for _ in 0..n_reps {
        let incs = simulate_prompt_level(&mut rng, theta, tau, sigma, n_prompts);
        if crossing_step(&incs, eps, lambdas).is_some() {
            crossed += 1;
        }
    }
    PromptLevelResult {
        theta,
        rho,
        n_prompts,
        n_reps,
        eps,
        crossing_rate: crossed as f64 / n_reps as f64,
    }
}

// ---------------------------------------------------------------------------
// Python-`repr`-compatible float formatting (for JSON output and the
// parity sweep: parsed values round-trip exactly; integral floats
// print as `1.0`, never `1`).
// ---------------------------------------------------------------------------

/// Format an f64 the way CPython's `repr` does, closely enough that
/// parsing the result recovers the exact same bits. Rust's `{}`
/// already prints the shortest string that round-trips; the only
/// adjustments are the `.0` suffix for integral values and
/// scientific notation for very small / very large magnitudes
/// (Python switches at 1e-4 / 1e16).
pub fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let ax = x.abs();
    if ax < 1e-4 || ax >= 1e16 {
        // Rust {:e} gives e.g. "1.23e-140"; Python gives "1.23e-140"
        // (two exponent digits when |exp| < 10, e.g. "1e-05").
        let s = format!("{:e}", x);
        let (mant, exp) = s.split_once('e').unwrap();
        let e: i32 = exp.parse().unwrap();
        return format!("{}e{}{:02}", mant, if e < 0 { "-" } else { "+" }, e.abs());
    }
    let s = format!("{}", x);
    if s.contains('.') {
        s
    } else {
        format!("{}.0", s)
    }
}

/// JSON for a `SingleUnitResult`, key order matching the Python dict.
pub fn single_unit_json(r: &SingleUnitResult) -> String {
    let null_rate = match r.crossing_rate_realized_null {
        Some(v) => py_repr(v),
        None => "null".to_string(),
    };
    let alt_rate = match r.crossing_rate_realized_alt {
        Some(v) => py_repr(v),
        None => "null".to_string(),
    };
    format!(
        "{{\"theta\": {}, \"rho\": {}, \"n_rollouts\": {}, \"n_units\": {}, \
         \"eps\": {}, \"pi\": {}, \"crossing_rate\": {}, \
         \"n_realized_null\": {}, \"crossing_rate_realized_null\": {}, \
         \"n_realized_alt\": {}, \"crossing_rate_realized_alt\": {}}}",
        py_repr(r.theta),
        py_repr(r.rho),
        r.n_rollouts,
        r.n_units,
        py_repr(r.eps),
        py_repr(r.pi),
        py_repr(r.crossing_rate),
        r.n_realized_null,
        null_rate,
        r.n_realized_alt,
        alt_rate
    )
}

/// JSON for a `PromptLevelResult`, key order matching the Python dict.
pub fn prompt_level_json(r: &PromptLevelResult) -> String {
    format!(
        "{{\"theta\": {}, \"rho\": {}, \"n_prompts\": {}, \"n_reps\": {}, \
         \"eps\": {}, \"crossing_rate\": {}}}",
        py_repr(r.theta),
        py_repr(r.rho),
        r.n_prompts,
        r.n_reps,
        py_repr(r.eps),
        py_repr(r.crossing_rate)
    )
}
