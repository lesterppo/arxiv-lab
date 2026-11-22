//! Unit tests for each mechanism in `selfconsistency_rs`.
//!
//! Reference values were produced by the Python module
//! (`arxiv_lab.agent.selfconsistency`, CPython 3.12 on this VM)
//! and are asserted bit-exactly where the computation is a pure
//! function of libm routines (`ppf`, `cdf`, `pi`, e-process wealth
//! paths, crossing steps, ESS) — the parity sweep in
//! `tests/parity_sweep.py` re-checks the same claim over thousands
//! of random inputs.

use selfconsistency_rs::*;

fn bits(x: f64) -> u64 {
    x.to_bits()
}

// ---------------------------------------------------------------------------
// SplitMix64 stream (Python: SplitMix64(0) / SplitMix64(42)).
// ---------------------------------------------------------------------------

#[test]
fn splitmix64_first_u64s_seed0() {
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next_u64(), 0xe220a8397b1dcdaf);
    assert_eq!(r.next_u64(), 0x6e789e6aa1b965f4);
    assert_eq!(r.next_u64(), 0x6c45d188009454f);
}

#[test]
fn splitmix64_uniforms_seed0() {
    let mut r = SplitMix64::new(0);
    assert_eq!(bits(r.uniform()), bits(0.8833108082136426));
    assert_eq!(bits(r.uniform()), bits(0.43152799704850997));
    assert_eq!(bits(r.uniform()), bits(0.026433771592597743));
}

#[test]
fn splitmix64_normals_seed42() {
    let mut r = SplitMix64::new(42);
    assert_eq!(bits(r.normal()), bits(0.6481773615835266));
    assert_eq!(bits(r.normal()), bits(-0.9948262326814028));
    assert_eq!(bits(r.normal()), bits(-0.5870021539269944));
}

#[test]
fn splitmix64_same_seed_same_stream() {
    let mut a = SplitMix64::new(12345);
    let mut b = SplitMix64::new(12345);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(a.uniform().to_bits(), b.uniform().to_bits());
    }
}

// ---------------------------------------------------------------------------
// normal_ppf / normal_cdf — bit-exact against CPython values.
// ---------------------------------------------------------------------------

#[test]
fn ppf_known_values_bit_exact() {
    assert_eq!(bits(normal_ppf(0.5)), bits(0.0));
    assert_eq!(bits(normal_ppf(0.975)), 0x3fff5c03325b95a6);
    assert_eq!(bits(normal_ppf(0.025)), 0xbfff5c03325b95a6);
    assert_eq!(bits(normal_ppf(0.001)), 0xc008b8cbb6ee2822);
    assert_eq!(bits(normal_ppf(0.999)), 0x4008b8cbb6ee2822);
    assert_eq!(bits(normal_ppf(1e-6)), 0xc0130381a9cfc25e);
    assert_eq!(bits(normal_ppf(0.999999)), 0x40130381a9cfa8cd);
    assert_eq!(bits(normal_ppf(0.1)), 0xbff4813c3681e9f4);
}

#[test]
fn ppf_matches_reference_to_1e5() {
    // The unit test above is bit-exact; this is the paper-level
    // sanity check the Python E0 also makes.
    assert!((normal_ppf(0.975) - 1.959964).abs() < 1e-5);
}

#[test]
fn cdf_known_values_bit_exact() {
    assert_eq!(bits(normal_cdf(0.0)), 0x3fe0000000000000);
    assert_eq!(bits(normal_cdf(1.0)), 0x3feaec4bd120d37d);
    assert_eq!(bits(normal_cdf(-1.0)), 0x3fc44ed0bb7cb20c);
    assert_eq!(bits(normal_cdf(1.959963984540054)), 0x3fef333333333333);
    assert_eq!(bits(normal_cdf(0.5)), 0x3fe62075e232ac77);
}

#[test]
fn cdf_ppf_round_trip() {
    for p in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999] {
        let back = normal_cdf(normal_ppf(p));
        assert!((back - p).abs() < 1e-8, "p={} back={}", p, back);
    }
}

#[test]
fn ppf_is_monotone() {
    let mut prev = f64::NEG_INFINITY;
    let mut p = 0.001f64;
    while p < 1.0 {
        let v = normal_ppf(p);
        assert!(v > prev, "ppf not monotone at p={}", p);
        prev = v;
        p += 0.001;
    }
}

// ---------------------------------------------------------------------------
// Random-effects parameterisation.
// ---------------------------------------------------------------------------

#[test]
fn rho_tau_sigma_round_trip() {
    for rho in [0.0, 0.1, 0.435, 0.48, 0.9, 1.0] {
        let (tau, sigma) = tau_sigma_from_rho(rho);
        assert!((tau * tau + sigma * sigma - 1.0).abs() < 1e-15);
        assert!((rho_from_tau_sigma(tau, sigma) - rho).abs() < 1e-12);
    }
}

#[test]
fn tau_sigma_048_bit_exact() {
    let (tau, sigma) = tau_sigma_from_rho(0.48);
    assert_eq!(bits(tau), bits(0.6928203230275509));
    assert_eq!(bits(sigma), bits(0.7211102550927979));
}

#[test]
fn pi_positive_values() {
    let (tau, _) = tau_sigma_from_rho(0.48);
    assert_eq!(bits(pi_positive(-0.15, tau)), bits(0.4142965376093233));
    assert!((pi_positive(0.96, tau_sigma_from_rho(0.435).0) - 0.9272412277642623).abs() < 1e-15);
    // tau == 0: degenerate step function.
    assert_eq!(pi_positive(0.5, 0.0), 1.0);
    assert_eq!(pi_positive(-0.5, 0.0), 0.0);
    assert_eq!(pi_positive(0.0, 0.0), 0.0);
}

// ---------------------------------------------------------------------------
// Betting e-process.
// ---------------------------------------------------------------------------

#[test]
fn eprocess_wealth_path_bit_exact() {
    let path = betting_eprocess(&[1.0, -1.0, 1.0, 1.0, -1.0], &LAMBDA_GRID);
    let expected = [
        1.0,
        1.4055555555555557,
        0.7730555555555556,
        1.0345972222222222,
        1.4256159722222221,
        0.8297462847222222,
    ];
    assert_eq!(path.len(), expected.len());
    for (got, want) in path.iter().zip(expected.iter()) {
        assert_eq!(bits(*got), bits(*want), "got {} want {}", got, want);
    }
}

#[test]
fn eprocess_crosses_on_all_plus_one_at_step_8() {
    assert_eq!(crossing_step(&vec![1.0; 20], 0.05, &LAMBDA_GRID), Some(8));
    assert_eq!(crossing_step(&vec![1.0; 10], 0.05, &LAMBDA_GRID), Some(8));
}

#[test]
fn eprocess_never_crosses_on_all_minus_one() {
    assert_eq!(crossing_step(&vec![-1.0; 200], 0.05, &LAMBDA_GRID), None);
    assert_eq!(crossing_step(&vec![-1.0; 200], 0.001, &LAMBDA_GRID), None);
}

#[test]
fn eprocess_mixed_sequence_does_not_cross() {
    assert_eq!(
        crossing_step(&[1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.05, &LAMBDA_GRID),
        None
    );
}

#[test]
fn eprocess_zero_increments_are_neutral() {
    // A zero increment appends the unchanged wealth and consumes
    // no step: [1,1] and [1,0,1] agree at the corresponding points.
    let a = betting_eprocess(&[1.0, 1.0], &LAMBDA_GRID);
    let b = betting_eprocess(&[1.0, 0.0, 1.0], &LAMBDA_GRID);
    assert_eq!(bits(a[1]), bits(b[1]));
    assert_eq!(bits(a[2]), bits(b[3]));
}

#[test]
fn eprocess_single_lambda_matches_direct_product() {
    // With a one-element grid the mixture is a plain product.
    let lambdas = [0.5];
    let incs = [1.0, 1.0, -1.0, 1.0];
    let path = betting_eprocess(&incs, &lambdas);
    let mut w = 1.0f64;
    assert_eq!(path[0], 1.0);
    for (i, &x) in incs.iter().enumerate() {
        w *= 1.0 + 0.5 * x;
        assert_eq!(bits(path[i + 1]), bits(w));
    }
}

// ---------------------------------------------------------------------------
// ESS / within-unit variance.
// ---------------------------------------------------------------------------

#[test]
fn ess_formula_values() {
    assert_eq!(bits(ess_within_unit(256, 0.48)), bits(2.0745542949756888));
    assert_eq!(ess_within_unit(1, 0.48), 1.0);
    assert_eq!(ess_within_unit(0, 0.48), 0.0);
    // ESS -> 1/rho as n -> infinity.
    assert!((ess_within_unit(1_000_000, 0.48) - 1.0 / 0.48).abs() < 1e-5);
    // rho = 0: no saturation, ESS = n.
    assert_eq!(ess_within_unit(256, 0.0), 256.0);
}

#[test]
fn within_unit_mean_var_values() {
    assert_eq!(
        bits(within_unit_mean_var(256, 0.48)),
        bits(0.48203124999999997)
    );
    assert_eq!(within_unit_mean_var(1, 0.48), 1.0);
    assert!((within_unit_mean_var(1_000_000, 0.48) - 0.48).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// ICC ANOVA.
// ---------------------------------------------------------------------------

#[test]
fn icc_perfectly_split_groups_is_one() {
    let rho = icc_anova(&[
        vec![1.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0],
    ])
    .unwrap();
    assert_eq!(bits(rho), bits(1.0));
}

#[test]
fn icc_identical_groups_is_zero() {
    let rho = icc_anova(&[
        vec![1.0, 0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0, 1.0],
    ])
    .unwrap();
    assert_eq!(bits(rho), bits(0.0));
}

#[test]
fn icc_degenerate_returns_none() {
    assert_eq!(icc_anova(&[vec![1.0, 1.0, 1.0]]), None); // < 2 prompts
    assert_eq!(icc_anova(&[vec![1.0], vec![0.0], vec![1.0]]), None); // no replication
    assert_eq!(icc_anova(&[]), None);
    assert_eq!(icc_anova(&[vec![], vec![1.0, 0.0]]), None); // one usable group
}

#[test]
fn icc_unequal_group_sizes_in_range() {
    let rho = icc_anova(&[
        vec![1.0, 1.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0, 0.0],
    ])
    .unwrap();
    assert!((0.0..=1.0).contains(&rho), "rho={}", rho);
}

#[test]
fn icc_mixed_groups_bit_exact() {
    // Non-degenerate groups: exercises the Neumaier sums and the
    // x*x-vs-pow(x, 2) equivalence in ss_between / ss_within.
    let rho = icc_anova(&[
        vec![1.0, 1.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0, 0.0],
    ])
    .unwrap();
    assert_eq!(bits(rho), 0x3fd226b90226b903);
    let rho2 = icc_anova(&[
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ])
    .unwrap();
    assert_eq!(bits(rho2), 0x3fba2a1fb391b23f);
    // Negative raw estimate clamps to 0.
    let rho3 = icc_anova(&[
        vec![1.0, 1.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    .unwrap();
    assert_eq!(bits(rho3), 0x0);
}

#[test]
fn py_sum_matches_cpython_sum() {
    // The e-process term list that first exposed the 1-ulp gap.
    let terms = [
        0.5833333333333334,
        0.611111111111111,
        0.6666666666666666,
        0.7222222222222222,
        0.7777777777777778,
        0.8333333333333333,
        0.8888888888888888,
        0.9444444444444444,
        1.0,
    ];
    assert_eq!(bits(py_sum(terms)), bits(7.027777777777778));
    // Naive sum differs on this list (that was the bug).
    let naive: f64 = terms.iter().sum();
    assert_ne!(bits(naive), bits(7.027777777777778));
}

// ---------------------------------------------------------------------------
// theta_hat.
// ---------------------------------------------------------------------------

#[test]
fn theta_hat_values() {
    assert_eq!(bits(theta_hat_from_accuracy(0.5)), bits(0.0));
    assert_eq!(
        bits(theta_hat_from_accuracy(0.9)),
        bits(1.2815515641401563)
    );
    // Clamping: 0 and 1 map to the ppf of 1e-6 / 1-1e-6, finite.
    assert!(theta_hat_from_accuracy(0.0).is_finite());
    assert!(theta_hat_from_accuracy(1.0).is_finite());
    assert_eq!(
        bits(theta_hat_from_accuracy(0.0)),
        bits(normal_ppf(1e-6))
    );
}

// ---------------------------------------------------------------------------
// simulate_unit stream behaviour.
// ---------------------------------------------------------------------------

#[test]
fn simulate_unit_tau_zero_consumes_no_mu_draw() {
    // tau = 0: mu == theta exactly, and the first increment's z is
    // the FIRST normal draw of the stream (no mu draw consumed).
    let mut rng = SplitMix64::new(5);
    let (mu, incs) = simulate_unit(&mut rng, -0.15, 0.0, 1.0, 3, true);
    assert_eq!(mu, -0.15);
    let mut rng2 = SplitMix64::new(5);
    let z0 = rng2.normal();
    let expect = if -0.15 + 1.0 * z0 > 0.0 { 1.0 } else { -1.0 };
    assert_eq!(incs[0], expect);
}

#[test]
fn simulate_unit_binary_vs_continuous_share_latent() {
    // Same seed: binary and continuous runs draw the same mu and the
    // same z sequence, so signs must agree.
    let (tau, sigma) = tau_sigma_from_rho(0.48);
    let mut r1 = SplitMix64::new(99);
    let (mu_b, bin) = simulate_unit(&mut r1, -0.15, tau, sigma, 50, true);
    let mut r2 = SplitMix64::new(99);
    let (mu_c, cont) = simulate_unit(&mut r2, -0.15, tau, sigma, 50, false);
    assert_eq!(bits(mu_b), bits(mu_c));
    for (b, c) in bin.iter().zip(cont.iter()) {
        assert_eq!(*b, if *c > 0.0 { 1.0 } else { -1.0 });
    }
}
