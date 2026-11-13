//! Unit tests for the termination-lag mechanism: one test per mechanism,
//! each with hand-computed or directly asserted behaviour.

use termination_lag::*;

// ---------------------------------------------------------------- RNG
#[test]
fn splitmix64_first_draws_match_python() {
    // Values from the frozen Python module (RNG(0).next_u64() x8).
    let mut r = Rng::new(0);
    let expected: [u64; 8] = [
        16294208416658607535,
        7960286522194355700,
        487617019471545679,
        17909611376780542444,
        1961750202426094747,
        6038094601263162090,
        3207296026000306913,
        14232521865600346940,
    ];
    for e in expected {
        assert_eq!(r.next_u64(), e);
    }
}

#[test]
fn uniform_stays_in_unit_interval_and_matches_python() {
    let mut r = Rng::new(0);
    let first = r.uniform();
    assert!((first - 0.8833108082136426).abs() < 1e-15, "{}", first);
    for _ in 0..10_000 {
        let u = r.uniform();
        assert!(u >= 0.0 && u < 1.0, "uniform out of range: {}", u);
    }
}

#[test]
fn randint_bounds() {
    let mut r = Rng::new(123);
    for _ in 0..10_000 {
        let d = r.randint(0, 3);
        assert!(d < 3, "randint(0,3) = {}", d);
    }
    // degenerate range: always lo
    let mut r2 = Rng::new(5);
    for _ in 0..100 {
        assert_eq!(r2.randint(7, 8), 7);
    }
}

#[test]
fn poisson_small_lambda_is_usually_zero() {
    let mut r = Rng::new(9);
    let zeros = (0..1000).filter(|_| r.poisson(0.1) == 0).count();
    assert!(zeros > 850, "poisson(0.1) zeros = {}", zeros);
    // poisson(0) is always 0 (limit = 1.0, first product < 1)
    let mut r2 = Rng::new(9);
    for _ in 0..100 {
        assert_eq!(r2.poisson(0.0), 0);
    }
}

// ---------------------------------------------------------------- tasks
#[test]
fn gen_tasks_difficulties_are_balanced() {
    let tasks = gen_tasks(3000, 20261113);
    let mut counts = [0usize; 3];
    for t in &tasks {
        counts[t.difficulty as usize] += 1;
    }
    // randint(0,3) is uniform; each difficulty within ~3.5 sigma of 1000.
    for (i, c) in counts.iter().enumerate() {
        assert!((900..1100).contains(c), "difficulty {} count {}", i, c);
    }
    assert_eq!(tasks[0].idx, 0);
    assert_eq!(tasks[2999].idx, 2999);
}

#[test]
fn difficulty_first_edit_probs() {
    assert_eq!(Difficulty::Easy.p_first(), 0.82);
    assert_eq!(Difficulty::Medium.p_first(), 0.60);
    assert_eq!(Difficulty::Hard.p_first(), 0.38);
}

// ---------------------------------------------------------------- verifier
#[test]
fn verifier_passes_correct_fails_wrong() {
    assert!(verify(true));
    assert!(!verify(false));
}

// ---------------------------------------------------------------- run_task
#[test]
fn stop_on_pass_never_reedits_and_lag_is_one() {
    for seed in [0u64, 1, 7, 12345, 20261113] {
        for idx in 0..20u64 {
            let mut rng = Rng::new(seed ^ idx.wrapping_mul(0x9E3779B97F4A7C15));
            let r = run_task(
                Task { idx, difficulty: Difficulty::Medium },
                Policy::StopOnPass,
                &mut rng,
                3.0,
                P_UNDO,
                10,
                1.0,
            );
            if r.solved {
                assert_eq!(r.lag, 1, "seed {} idx {}", seed, idx);
                assert_eq!(r.n_reedit, 0);
                assert_eq!(r.n_undos, 0);
                assert_eq!(r.n_verify, 1);
                assert!(r.final_correct);
            }
        }
    }
}

#[test]
fn p_undo_zero_means_no_undos_ever() {
    for seed in [0u64, 1, 7] {
        let tasks = gen_tasks(200, seed);
        for t in &tasks {
            let mut rng = Rng::new(seed ^ t.idx.wrapping_mul(0x9E3779B97F4A7C15));
            let r = run_task(*t, Policy::Continue, &mut rng, 6.0, 0.0, 10, 1.0);
            assert_eq!(r.n_undos, 0);
            assert!(!r.undo_to_incorrect);
            if r.solved {
                assert!(r.final_correct);
            }
        }
    }
}

#[test]
fn p_undo_one_undoes_every_reedit() {
    // With p_undo = 1.0, every post-solution re-edit undoes the fix, so any
    // solved run that re-edits at least once must record an undo.
    let mut saw_reedit = false;
    for seed in [0u64, 1, 7, 42] {
        let tasks = gen_tasks(50, seed);
        for t in &tasks {
            let mut rng = Rng::new(seed ^ t.idx.wrapping_mul(0x9E3779B97F4A7C15));
            let r = run_task(*t, Policy::Continue, &mut rng, 6.0, 1.0, 10, 1.0);
            if r.n_reedit > 0 {
                saw_reedit = true;
                assert!(r.n_undos >= 1, "re-edited but no undo recorded");
            }
        }
    }
    assert!(saw_reedit, "expected at least one re-edit across the sweep");
}

#[test]
fn fixed_budget_solved_runs_end_exactly_at_budget() {
    for seed in [0u64, 1, 7] {
        let tasks = gen_tasks(200, seed);
        for t in &tasks {
            let mut rng = Rng::new(seed ^ t.idx.wrapping_mul(0x9E3779B97F4A7C15));
            let r = run_task(*t, Policy::FixedBudget, &mut rng, 3.0, P_UNDO, 10, 1.0);
            if r.solved {
                assert_eq!(r.steps, 10, "seed {} idx {}", seed, t.idx);
            } else {
                assert!(r.steps <= 10);
            }
        }
    }
}

#[test]
fn invariants_hold_for_every_run() {
    for seed in [0u64, 1, 7, 42, 20261113] {
        let tasks = gen_tasks(100, seed);
        for t in &tasks {
            for pol in [Policy::Continue, Policy::StopOnPass, Policy::FixedBudget] {
                let mut rng = Rng::new(seed ^ t.idx.wrapping_mul(0x9E3779B97F4A7C15));
                let r = run_task(*t, pol, &mut rng, 3.0, P_UNDO, 10, 1.0);
                assert!(r.steps <= 20, "steps {} > MAX_STEPS", r.steps);
                if r.solved {
                    let fcs = r.first_correct_step.unwrap();
                    assert!(fcs >= 1 && fcs <= r.steps);
                    assert_eq!(r.lag, r.steps - fcs);
                    assert!((0.0..=1.0).contains(&r.post_share));
                    assert_eq!(
                        r.n_verify + r.n_explore + r.n_reedit,
                        r.lag,
                        "post-solution steps must sum to lag"
                    );
                    assert_eq!(r.undo_to_incorrect, !r.final_correct);
                    assert!(r.n_undos <= r.n_reedit);
                } else {
                    assert!(r.first_correct_step.is_none());
                    assert_eq!(r.lag, 0);
                }
            }
        }
    }
}

#[test]
fn first_edit_solve_share_matches_difficulty_mix() {
    // Easy tasks solve on the first edit far more often than hard ones.
    let mut rate = [0.0f64; 3];
    for (i, d) in [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard]
        .iter()
        .enumerate()
    {
        let tasks: Vec<Task> = (0..500u64)
            .map(|idx| Task { idx, difficulty: *d })
            .collect();
        let m = run_experiment(Policy::Continue, &tasks, 99, 3.0, P_UNDO, 10, 1.0);
        rate[i] = m.first_edit_solve_share;
    }
    assert!(rate[0] > rate[1] && rate[1] > rate[2], "{:?}", rate);
    assert!((0.75..0.90).contains(&rate[0]), "easy {}", rate[0]);
    assert!((0.50..0.70).contains(&rate[1]), "medium {}", rate[1]);
    assert!((0.28..0.48).contains(&rate[2]), "hard {}", rate[2]);
}
