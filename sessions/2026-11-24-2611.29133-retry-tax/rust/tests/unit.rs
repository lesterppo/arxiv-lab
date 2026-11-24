//! Unit tests for each mechanism in `retrytax_rs`.

use retrytax_rs::*;

const EPS: f64 = 1e-12;

#[test]
fn completion_prob_known_values() {
    assert!((completion_prob(0.5, 1) - 0.5).abs() < EPS);
    assert!((completion_prob(0.5, 2) - 0.75).abs() < EPS);
    assert!((completion_prob(0.25, 3) - (1.0 - 0.75f64.powi(3))).abs() < EPS);
    assert_eq!(completion_prob(0.0, 5), 0.0);
    assert_eq!(completion_prob(1.0, 5), 1.0);
}

#[test]
fn expected_attempts_known_values() {
    // p=0.5, cap=3: 1 + 0.5 + 0.25 = 1.75.
    assert!((expected_attempts(0.5, 3) - 1.75).abs() < EPS);
    assert_eq!(expected_attempts(0.0, 4), 4.0);
    assert_eq!(expected_attempts(1.0, 4), 1.0);
    // p=0.25, cap=2: 1 + 0.75 = 1.75.
    assert!((expected_attempts(0.25, 2) - 1.75).abs() < EPS);
}

#[test]
fn population_metrics_single_task() {
    let m = population_metrics(&[0.5], 2, 2.0);
    assert!((m.completion - 0.75).abs() < EPS);
    assert!((m.spend_per_task - 3.0).abs() < EPS); // 2.0 * 1.5 attempts
    assert!((m.cost_per_completion - 4.0).abs() < EPS);
}

#[test]
fn marginal_cost_two_point_population() {
    // Easy task p=1 (completes at once) + hard task p=0 (never):
    // cap 1 -> cap 2 adds no completions, so marginal cost is None.
    assert!(marginal_cost(&[1.0, 0.0], 2, 1.0).is_none());
    // Two easy tasks p=0.5: S(1)=1, C(1)=0.5; S(2)=1.5, C(2)=0.75;
    // marginal = 0.5/0.25 = 2.0.
    let mc = marginal_cost(&[0.5, 0.5], 2, 1.0).unwrap();
    assert!((mc - 2.0).abs() < EPS);
}

#[test]
fn simulate_deterministic_same_seed() {
    let ps = calibrated_mixture(1000, 42);
    let a = simulate_population(&ps, 3, 1.0, &mut Rng::new(7));
    let b = simulate_population(&ps, 3, 1.0, &mut Rng::new(7));
    assert_eq!(a, b);
}

#[test]
fn replay_all_success_first_attempt() {
    // Every recorded attempt succeeds: any cap completes everything
    // with exactly one attempt spent per task.
    let seqs = vec![vec![true; 4]; 50];
    let (c, s, cpc) = replay_runs(&seqs, 3, 2.0, 10, &mut Rng::new(1));
    assert_eq!(c, 1.0);
    assert_eq!(s, 2.0);
    assert_eq!(cpc, 2.0);
}

#[test]
fn replay_all_fail() {
    let seqs = vec![vec![false; 4]; 50];
    let (c, s, cpc) = replay_runs(&seqs, 3, 2.0, 10, &mut Rng::new(1));
    assert_eq!(c, 0.0);
    assert_eq!(s, 6.0);
    assert!(cpc.is_infinite());
}

#[test]
fn unresolved_mean_p_monotone_on_mixture() {
    let ps = calibrated_mixture(5000, 99);
    let mut rng = Rng::new(5);
    let m0 = ps.iter().sum::<f64>() / ps.len() as f64;
    let m1 = unresolved_mean_p(&ps, 1, &mut rng);
    let m2 = unresolved_mean_p(&ps, 2, &mut rng);
    assert!(m1 < m0 && m2 < m1);
}
