//! Static same-task retry economics (The Retry Tax).
//!
//! Rust port of `arxiv_lab.agent.retry_tax` (arXiv:2611.29133): a
//! static retry policy that stops at verified success or an attempt
//! cap, resets the prompt and workspace, and passes no feedback
//! between attempts. Under stationary per-task success probabilities,
//! attempts independent conditional on the task, and a common attempt
//! cost, cost per completion at cap K is at least the one-shot cost
//! and marginal cost per additional completion is nondecreasing in K.
//!
//! Ported — the pure, deterministic core:
//!
//! * [`completion_prob`] / [`expected_attempts`] — closed-form
//!   per-task quantities.
//! * [`population_metrics`] / [`marginal_cost`] — closed-form
//!   population expectations.
//! * [`simulate_population`] — Monte-Carlo run of the static policy.
//! * [`replay_runs`] — the paper's replay procedure over recorded
//!   per-task attempt-outcome sequences.
//! * [`unresolved_mean_p`] — survivor-selection diagnostic.
//! * [`calibrated_mixture`] — the two-point population calibrated to
//!   the paper's replay endpoints.
//!
//! NOT ported — nothing: the Python module is already pure logic.
//! The only deliberate divergence from Python is the RNG: Python
//! uses Mersenne Twister (`random.Random`), Rust uses xorshift64*
//! ([`Rng`]) below. Simulation draws therefore differ draw-for-draw,
//! but every closed-form quantity is bit-identical (both sides are
//! IEEE-754 f64) and every simulated quantity agrees with the closed
//! form within Monte-Carlo noise on both sides; the parity sweep
//! checks the closed forms exactly and the simulations within
//! tolerance.

/// Paper's replay estimates for one OpenHands configuration.
pub const PAPER_COMPLETION_K1: f64 = 0.597;
pub const PAPER_COMPLETION_K4: f64 = 0.700;
pub const PAPER_CPC_K1_USD: f64 = 0.70;
pub const PAPER_CPC_K4_USD: f64 = 1.32;

/// Deterministic xorshift64* RNG (std-only; no crates).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Zero state would freeze xorshift; force a nonzero state.
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform double in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Fisher-Yates shuffle, in place.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() % ((i + 1) as u64)) as usize;
            v.swap(i, j);
        }
    }
}

/// P(task succeeds within `cap` attempts) = 1 - (1 - p)^cap.
///
/// Computed as -expm1(cap * ln_1p(-p)): for p < ~2^-53 the naive
/// (1 - p) rounds to 1.0 in f64 and the naive form returns 0, while
/// the ln_1p form stays exact. Same formula as the Python module.
pub fn completion_prob(p: f64, cap: usize) -> f64 {
    assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
    assert!(cap >= 1, "cap must be >= 1");
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    -((cap as f64) * (-p).ln_1p()).exp_m1()
}

/// Expected attempts spent on one task under the static policy.
pub fn expected_attempts(p: f64, cap: usize) -> f64 {
    assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
    assert!(cap >= 1, "cap must be >= 1");
    if p == 0.0 {
        return cap as f64;
    }
    completion_prob(p, cap) / p
}

/// Closed-form population expectations at one cap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationMetrics {
    pub cap: usize,
    pub completion: f64,
    pub spend_per_task: f64,
    pub cost_per_completion: f64,
}

pub fn population_metrics(ps: &[f64], cap: usize, cost: f64) -> PopulationMetrics {
    assert!(!ps.is_empty(), "empty population");
    assert!(cost > 0.0, "cost must be positive");
    let n = ps.len() as f64;
    let completion: f64 = ps.iter().map(|&p| completion_prob(p, cap)).sum::<f64>() / n;
    let spend: f64 = cost * ps.iter().map(|&p| expected_attempts(p, cap)).sum::<f64>() / n;
    let cpc = if completion > 0.0 {
        spend / completion
    } else {
        f64::INFINITY
    };
    PopulationMetrics {
        cap,
        completion,
        spend_per_task: spend,
        cost_per_completion: cpc,
    }
}

/// Marginal cost of the completions added by raising the cap to `cap`.
/// `None` when the cap adds no completions (the paper's "wherever the
/// completion increments are positive" qualifier).
///
/// The increments are computed directly per task — ΔS = E[(1-p)^(K-1)]
/// and ΔC = E[p (1-p)^(K-1)] — not by differencing the population
/// means: for p near 1 both C(K) and C(K-1) are ~1 and their
/// difference loses ~6 digits to cancellation.
pub fn marginal_cost(ps: &[f64], cap: usize, cost: f64) -> Option<f64> {
    assert!(cap >= 2, "marginal cost needs cap >= 2");
    assert!(!ps.is_empty(), "empty population");
    assert!(cost > 0.0, "cost must be positive");
    let n = ps.len() as f64;
    let d_spend = cost * ps.iter().map(|&p| survival(p, cap - 1)).sum::<f64>() / n;
    let d_comp = ps.iter().map(|&p| p * survival(p, cap - 1)).sum::<f64>() / n;
    if d_comp <= 0.0 {
        None
    } else {
        Some(d_spend / d_comp)
    }
}

/// (1 - p)^k, accurate for p near 0 and near 1.
pub fn survival(p: f64, k: usize) -> f64 {
    if p == 0.0 {
        return 1.0;
    }
    if p == 1.0 {
        return 0.0;
    }
    ((k as f64) * (-p).ln_1p()).exp()
}

/// Monte-Carlo run of the static policy: (completion, spend/task, CPC).
pub fn simulate_population(
    ps: &[f64],
    cap: usize,
    cost: f64,
    rng: &mut Rng,
) -> (f64, f64, f64) {
    let mut done = 0usize;
    let mut attempts = 0usize;
    for &p in ps {
        for _ in 0..cap {
            attempts += 1;
            if rng.next_f64() < p {
                done += 1;
                break;
            }
        }
    }
    let n = ps.len() as f64;
    let completion = done as f64 / n;
    let spend = cost * attempts as f64 / n;
    let cpc = if done > 0 { spend / completion } else { f64::INFINITY };
    (completion, spend, cpc)
}

/// The paper's replay procedure over recorded per-task attempt
/// outcomes. `sequences` are per-task outcome lists (true = verified
/// success on that attempt); each is arranged in `n_reorderings`
/// random orders, stopped at verified success or the cap, and
/// completion/spending are averaged. Sequences are assumed at least
/// `cap` long (as in the Python test).
pub fn replay_runs(
    sequences: &[Vec<bool>],
    cap: usize,
    cost: f64,
    n_reorderings: usize,
    rng: &mut Rng,
) -> (f64, f64, f64) {
    assert!(!sequences.is_empty(), "empty run set");
    assert!(cap >= 1, "cap must be >= 1");
    let mut total_done = 0usize;
    let mut total_attempts = 0usize;
    let mut total_tasks = 0usize;
    for _ in 0..n_reorderings {
        for seq in sequences {
            let mut order = seq.clone();
            rng.shuffle(&mut order);
            total_tasks += 1;
            let mut finished = false;
            for ok in order.iter().take(cap) {
                total_attempts += 1;
                if *ok {
                    total_done += 1;
                    finished = true;
                    break;
                }
            }
            if !finished {
                // Ran out of recorded attempts before the cap without
                // success (only fires for sequences shorter than cap).
                total_attempts += cap.saturating_sub(order.len());
            }
        }
    }
    let completion = total_done as f64 / total_tasks as f64;
    let spend = cost * total_attempts as f64 / total_tasks as f64;
    let cpc = if total_done > 0 {
        spend / completion
    } else {
        f64::INFINITY
    };
    (completion, spend, cpc)
}

/// Mean per-task p among tasks still unresolved after `k` attempts.
/// Returns NaN when nothing remains unresolved.
pub fn unresolved_mean_p(ps: &[f64], k: usize, rng: &mut Rng) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &p in ps {
        let mut survived = true;
        for _ in 0..k {
            if rng.next_f64() < p {
                survived = false;
                break;
            }
        }
        if survived {
            sum += p;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Two-point task population calibrated to the paper's replay:
/// 64.3% of tasks with p = 0.90, 35.7% with p = 0.05.
/// Closed form: C(1) = 0.597 (paper 0.597), C(4) = 0.709 (paper 0.700).
pub fn calibrated_mixture(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            if rng.next_f64() < 0.643 {
                0.90
            } else {
                0.05
            }
        })
        .collect()
}
