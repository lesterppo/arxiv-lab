//! Deeper tests for the termination-lag mechanism: property sweeps over
//! random inputs/budgets, exhaustive small-scale sequences checked against
//! brute-force invariants, a soak at a scale the Python test does not
//! reach, and thread-identity (the code is stateless, so parallel runs
//! must agree exactly).

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;
use termination_lag::*;

// ---------------------------------------------------------------- property sweeps
#[test]
fn mean_lag_grows_with_mean_extra() {
    // 2000 tasks: sampling noise on the mean lag is ~0.1, so the ordering
    // across the sweep is a real property, not luck.
    let tasks = gen_tasks(2000, 20261113);
    let mut prev = f64::NEG_INFINITY;
    for me in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0] {
        let m = run_experiment(Policy::Continue, &tasks, 20261113, me, P_UNDO, 10, 1.0);
        assert!(
            m.mean_lag > prev,
            "mean lag not monotone at mean_extra={}: {} <= {}",
            me,
            m.mean_lag,
            prev
        );
        prev = m.mean_lag;
    }
}

#[test]
fn undo_rate_grows_with_p_undo() {
    let tasks = gen_tasks(2000, 7);
    let mut prev = f64::NEG_INFINITY;
    for pu in [0.0, 0.05, 0.15, 0.30, 0.50, 1.0] {
        let m = run_experiment(Policy::Continue, &tasks, 7, 3.0, pu, 10, 1.0);
        assert!(
            m.undo_to_incorrect_rate >= prev,
            "undo rate not monotone at p_undo={}: {} < {}",
            pu,
            m.undo_to_incorrect_rate,
            prev
        );
        prev = m.undo_to_incorrect_rate;
    }
    // p_undo = 0 eliminates undos entirely (also covered in unit tests).
    let m0 = run_experiment(Policy::Continue, &tasks, 7, 3.0, 0.0, 10, 1.0);
    assert_eq!(m0.undo_to_incorrect_rate, 0.0);
}

#[test]
fn stop_on_pass_dominates_continue_on_final_correctness_everywhere() {
    // Across a grid of budgets and undo probabilities, stopping on the
    // first pass never ends with fewer correct final patches than
    // continuing — continuing can only destroy a correct patch.
    for seed in [1u64, 7, 42, 20261113] {
        let tasks = gen_tasks(300, seed);
        for me in [1.0, 3.0, 6.0] {
            for pu in [0.05, 0.15, 0.30] {
                let c = run_experiment(Policy::Continue, &tasks, seed, me, pu, 10, 1.0);
                let s = run_experiment(Policy::StopOnPass, &tasks, seed, me, pu, 10, 1.0);
                assert!(
                    s.final_correct_rate >= c.final_correct_rate,
                    "seed {} me {} pu {}: stop {} < continue {}",
                    seed, me, pu, s.final_correct_rate, c.final_correct_rate
                );
                assert!(
                    s.mean_lag <= c.mean_lag,
                    "seed {} me {} pu {}: stop lag {} > continue lag {}",
                    seed, me, pu, s.mean_lag, c.mean_lag
                );
            }
        }
    }
}

#[test]
fn harder_tasks_lag_no_more_than_easier_ones_only_via_solve_rate() {
    // Difficulty changes the solve probability, not the post-solution
    // machinery: conditional on solving, mean lag is the same across
    // difficulties up to sampling noise.
    let mut lags = Vec::new();
    for d in [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard] {
        let tasks: Vec<Task> = (0..800u64)
            .map(|idx| Task { idx, difficulty: d })
            .collect();
        let m = run_experiment(Policy::Continue, &tasks, 5, 3.0, P_UNDO, 10, 1.0);
        lags.push(m.mean_lag);
    }
    let spread = lags.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - lags.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(spread < 0.6, "conditional mean lags differ too much: {:?}", lags);
}

// ---------------------------------------------------------------- exhaustive small sequences
#[test]
fn all_policy_seed_combinations_satisfy_invariants() {
    // 3 policies x 200 seeds x 1 task: every run must satisfy the
    // structural invariants (lag accounting, step cap, composition sum).
    let task = Task { idx: 0, difficulty: Difficulty::Medium };
    let mut seen_first_steps = HashSet::new();
    for seed in 0..200u64 {
        for pol in [Policy::Continue, Policy::StopOnPass, Policy::FixedBudget] {
            let mut rng = Rng::new(seed);
            let r = run_task(task, pol, &mut rng, 3.0, P_UNDO, 10, 1.0);
            assert!(r.steps <= 20);
            if r.solved {
                seen_first_steps.insert(r.first_correct_step.unwrap());
                assert_eq!(r.lag, r.steps - r.first_correct_step.unwrap());
                assert_eq!(r.n_verify + r.n_explore + r.n_reedit, r.lag);
                assert!((0.0..=1.0).contains(&r.post_share));
            }
        }
    }
    // The first correct patch happens at several different steps across
    // seeds (the sweep actually exercises the space).
    assert!(seen_first_steps.len() >= 3, "{:?}", seen_first_steps);
}

#[test]
fn poisson_draws_cover_the_support() {
    // 5000 draws at lambda 3: the sampler must produce 0..~10, with the
    // mode near 3 (sanity against a degenerate sampler).
    let mut rng = Rng::new(20261113);
    let mut counts = [0u32; 16];
    for _ in 0..5000 {
        let k = rng.poisson(3.0) as usize;
        if k < 16 {
            counts[k] += 1;
        }
    }
    assert!(counts[0] > 100, "poisson(3) zeros: {}", counts[0]);
    assert!(counts[3] > counts[0], "mode should be near 3: {:?}", counts);
    assert!(counts[3] > counts[6], "tail should fall: {:?}", counts);
    assert!(counts.iter().sum::<u32>() == 5000);
}

// ---------------------------------------------------------------- soak
#[test]
fn soak_20000_tasks_all_policies() {
    // 20k tasks x 3 policies: invariants at a scale the Python test
    // (600 tasks) does not reach, plus a runtime sanity bound.
    let tasks = gen_tasks(20_000, 20261113);
    let t0 = std::time::Instant::now();
    let results = compare_policies(&tasks, 20261113, 3.0, P_UNDO, 10, 1.0);
    assert!(t0.elapsed().as_secs() < 30, "soak too slow: {:?}", t0.elapsed());
    let cont = &results[0];
    let stop = &results[1];
    assert!((0.55..0.65).contains(&cont.first_edit_solve_share));
    assert!((0.45..0.55).contains(&cont.median_post_share));
    assert!(stop.final_correct_rate >= cont.final_correct_rate);
    assert_eq!(stop.median_lag, 1.0);
    assert!(cont.undo_to_incorrect_rate > 0.0 && cont.undo_to_incorrect_rate < 0.15);
}

// ---------------------------------------------------------------- thread identity
#[test]
fn parallel_runs_agree_exactly() {
    // The mechanism is stateless (all state in the per-task RNG), so the
    // same experiment on four threads must produce bit-identical metrics.
    let tasks = Arc::new(gen_tasks(600, 20261113));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let t = Arc::clone(&tasks);
            thread::spawn(move || compare_policies(&t, 20261113, 3.0, P_UNDO, 10, 1.0))
        })
        .collect();
    let first = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .fold(None, |acc: Option<[ExperimentMetrics; 3]>, cur| {
            match acc {
                None => Some(cur),
                Some(prev) => {
                    for i in 0..3 {
                        assert_eq!(prev[i].n_solved, cur[i].n_solved);
                        assert_eq!(prev[i].mean_lag, cur[i].mean_lag);
                        assert_eq!(prev[i].median_lag, cur[i].median_lag);
                        assert_eq!(prev[i].median_post_share, cur[i].median_post_share);
                        assert_eq!(prev[i].mean_post_share, cur[i].mean_post_share);
                        assert_eq!(prev[i].final_correct_rate, cur[i].final_correct_rate);
                        assert_eq!(
                            prev[i].undo_to_incorrect_rate,
                            cur[i].undo_to_incorrect_rate
                        );
                    }
                    Some(prev)
                }
            }
        });
    assert!(first.is_some());
}
