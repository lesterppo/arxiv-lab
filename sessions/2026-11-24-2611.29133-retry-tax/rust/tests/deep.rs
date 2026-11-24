//! Deep tests: property sweeps, exhaustive small cases against a
//! brute-force oracle, a soak at a scale the Python test does not
//! reach, and thread identity (the code is stateless).

use retrytax_rs::*;
use std::sync::Arc;
use std::thread;

/// Brute-force oracle: enumerate every attempt-outcome pattern for a
/// small population and compute completion / expected attempts by
/// exhaustive enumeration (no closed form used).
fn brute_force(ps: &[f64], cap: usize) -> (f64, f64) {
    // Each task contributes cap Bernoulli attempts; enumerate the
    // joint outcome space (2^(n*cap) patterns) for tiny n/cap only.
    let n = ps.len();
    let total_bits = n * cap;
    assert!(total_bits <= 20, "oracle only for tiny cases");
    let total = 1u64 << total_bits;
    let mut e_completion = 0.0;
    let mut e_attempts = 0.0;
    for pat in 0..total {
        // Probability of the full bit pattern: EVERY bit contributes
        // its Bernoulli factor, including bits after a task's first
        // success. (Breaking out of the probability product early
        // double-counts the post-success bit combinations.)
        let mut prob = 1.0;
        for (t, &p) in ps.iter().enumerate() {
            for a in 0..cap {
                let bit = (pat >> (t * cap + a)) & 1;
                prob *= if bit == 1 { p } else { 1.0 - p };
            }
        }
        // Outcomes derive from the first success bit per task.
        let mut done = 0usize;
        let mut attempts = 0usize;
        for (t, &_p) in ps.iter().enumerate() {
            let mut succeeded = false;
            for a in 0..cap {
                let bit = (pat >> (t * cap + a)) & 1;
                attempts += 1;
                if bit == 1 {
                    succeeded = true;
                    break;
                }
            }
            if succeeded {
                done += 1;
            }
        }
        e_completion += prob * done as f64 / n as f64;
        e_attempts += prob * attempts as f64 / n as f64;
    }
    (e_completion, e_attempts)
}

#[test]
fn exhaustive_small_cases_match_oracle() {
    // All 3-task populations over p in {0, 0.25, 0.5, 0.75, 1},
    // caps 1..=3, checked against the brute-force oracle.
    let grid = [0.0, 0.25, 0.5, 0.75, 1.0];
    let mut cases = 0;
    for &p1 in &grid {
        for &p2 in &grid {
            for &p3 in &grid {
                let ps = [p1, p2, p3];
                for cap in 1..=3usize {
                    let m = population_metrics(&ps, cap, 1.0);
                    let (bc, ba) = brute_force(&ps, cap);
                    assert!(
                        (m.completion - bc).abs() < 1e-12,
                        "completion mismatch {ps:?} cap {cap}"
                    );
                    assert!(
                        (m.spend_per_task - ba).abs() < 1e-12,
                        "spend mismatch {ps:?} cap {cap}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 5 * 5 * 5 * 3);
}

#[test]
fn property_sweep_random_populations() {
    // 2,000 random populations: CPC monotone in cap, marginal cost
    // monotone in cap, CPC >= one-shot, completion in [0, 1].
    let mut rng = Rng::new(0xBEEF);
    for _ in 0..2000 {
        let n = 1 + (rng.next_u64() % 40) as usize;
        let ps: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let mut prev_cpc = 0.0f64;
        let mut prev_mc: Option<f64> = None;
        for k in 1..=6usize {
            let m = population_metrics(&ps, k, 1.7);
            assert!((0.0..=1.0).contains(&m.completion));
            // Relative tolerance: with near-zero p in the population
            // the marginal cost can be ~1e17, where absolute 1e-12
            // is below float noise. The theorem holds exactly (moment
            // log-convexity); the check is against rounding.
            assert!(
                m.cost_per_completion >= prev_cpc - 1e-9 * prev_cpc.max(1.0),
                "CPC k={k}"
            );
            if k >= 2 {
                if let Some(mc) = marginal_cost(&ps, k, 1.7) {
                    if let Some(p) = prev_mc {
                        assert!(mc >= p - 1e-9 * p.max(1.0), "marginal k={k}");
                    }
                    prev_mc = Some(mc);
                }
            }
            prev_cpc = m.cost_per_completion;
        }
    }
}

#[test]
fn property_sweep_random_costs_and_caps() {
    // Cost scaling: CPC scales linearly with the attempt cost.
    let mut rng = Rng::new(1234);
    for _ in 0..500 {
        let n = 5 + (rng.next_u64() % 20) as usize;
        let ps: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let cap = 1 + (rng.next_u64() % 8) as usize;
        let c1 = population_metrics(&ps, cap, 1.0).cost_per_completion;
        let c3 = population_metrics(&ps, cap, 3.0).cost_per_completion;
        assert!((c3 - 3.0 * c1).abs() < 1e-9 * c3.max(1.0));
    }
}

#[test]
fn soak_100k_tasks() {
    // Same inequalities at 100k tasks / caps 1..=12 (also exercised
    // by the `soak` subcommand; here as a test with a heavy tail).
    let mut rng = Rng::new(0xC0FFEE);
    let ps: Vec<f64> = (0..100_000)
        .map(|_| {
            let u = rng.next_f64();
            if u < 0.5 {
                0.02
            } else if u < 0.9 {
                rng.next_f64() * 0.5
            } else {
                0.95
            }
        })
        .collect();
    let mut prev_cpc = 0.0f64;
    let mut prev_mc: Option<f64> = None;
    for k in 1..=12usize {
        let m = population_metrics(&ps, k, 1.0);
        assert!(m.cost_per_completion >= prev_cpc - 1e-12);
        if k >= 2 {
            let mc = marginal_cost(&ps, k, 1.0).unwrap();
            if let Some(p) = prev_mc {
                assert!(mc >= p - 1e-12);
            }
            prev_mc = Some(mc);
        }
        prev_cpc = m.cost_per_completion;
    }
}

#[test]
fn thread_identity_stateless() {
    // The pure functions carry no shared state: the same computation
    // on 8 threads returns bit-identical results.
    let ps = Arc::new(calibrated_mixture(5000, 77));
    let mut handles: Vec<_> = (0..8)
        .map(|_| {
            let ps = Arc::clone(&ps);
            thread::spawn(move || {
                let mut out = Vec::new();
                for k in 1..=8usize {
                    out.push(population_metrics(&ps, k, 0.418));
                }
                out
            })
        })
        .collect();
    let first = handles.remove(0).join().unwrap();
    for h in handles {
        assert_eq!(h.join().unwrap(), first);
    }
}
