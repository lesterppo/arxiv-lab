//! Unit tests for the anytime-stopping mechanism (one per mechanism).
//!
//! Run: cargo test --release --offline

use anytime_stopping_rs::*;

fn step(step: usize, patch_id: usize, changed: bool, hidden: bool, self_p: bool) -> Step {
    Step {
        step,
        patch_id,
        changed,
        hidden_pass: hidden,
        self_pass: self_p,
    }
}

#[test]
fn splitmix64_reference_vector() {
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next_u64(), 0xE220A8397B1DCDAF);
    // Second output of the reference stream (seed 0), cross-checked
    // against the Python port.
    assert_eq!(r.next_u64(), 0x6E789E6AA1B965F4);
}

#[test]
fn splitmix64_deterministic_and_spread() {
    let mut a = SplitMix64::new(42);
    let mut b = SplitMix64::new(42);
    for _ in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    let mut c = SplitMix64::new(43);
    assert_ne!(a.next_u64(), c.next_u64());
    // uniform() stays in [0, 1).
    let mut r = SplitMix64::new(7);
    for _ in 0..1000 {
        let u = r.uniform();
        assert!((0.0..1.0).contains(&u));
    }
    // bernoulli edges: p=0 never fires, p=1 always fires.
    let mut r = SplitMix64::new(1);
    for _ in 0..100 {
        assert!(!r.bernoulli(0.0));
        assert!(r.bernoulli(1.0));
    }
}

#[test]
fn default_params_are_calibrated() {
    let p = TrajectoryParams::default();
    assert_eq!(p.budget, 30);
    assert_eq!(p.p_solvable, 0.65);
    assert_eq!(p.p_fix, 0.03);
    assert_eq!(p.p_change_after_self_pass, 0.15);
    assert_eq!(p.p_change_after_self_fail, 0.90);
    assert_eq!(p.p_break, 0.08);
    assert_eq!(p.p_self_false_pass, 0.01);
    assert_eq!(p.p_self_false_fail, 0.15);
}

#[test]
fn trajectory_shape_and_stickiness() {
    let params = TrajectoryParams::default();
    let mut rng = SplitMix64::new(20261118);
    for _ in 0..200 {
        let t = generate_trajectory(&mut rng, &params);
        assert_eq!(t.len(), params.budget);
        assert_eq!(t[0].step, 1);
        assert!(t[0].changed);
        for w in t.windows(2) {
            assert_eq!(w[1].step, w[0].step + 1);
            assert!(w[1].patch_id >= w[0].patch_id);
            if w[1].changed {
                assert_eq!(w[1].patch_id, w[0].patch_id + 1);
            } else {
                // Unchanged patch: same id, same hidden and self states.
                assert_eq!(w[1].patch_id, w[0].patch_id);
                assert_eq!(w[1].hidden_pass, w[0].hidden_pass);
                assert_eq!(w[1].self_pass, w[0].self_pass);
            }
        }
    }
}

#[test]
fn trajectory_determinism() {
    let params = TrajectoryParams::default();
    let mk = |seed: u64| {
        let mut rng = SplitMix64::new(seed);
        (0..10)
            .map(|_| trajectory_digest(&generate_trajectory(&mut rng, &params)))
            .collect::<Vec<_>>()
    };
    assert_eq!(mk(7), mk(7));
    assert_ne!(mk(7), mk(8));
}

#[test]
fn policies_on_hand_built_trajectories() {
    let t1 = vec![
        step(1, 0, true, true, true),
        step(2, 0, false, true, true),
        step(3, 0, false, true, true),
    ];
    assert_eq!(stop_double_pass(&t1), 2);
    assert_eq!(stop_first_self_pass(&t1), 1);
    assert_eq!(stop_oracle(&t1), 1);
    assert_eq!(stop_run_to_budget(&t1), 3);
    let t2 = vec![
        step(1, 0, true, false, true),
        step(2, 1, true, false, true),
        step(3, 1, false, false, true),
    ];
    assert_eq!(stop_double_pass(&t2), 3);
    assert_eq!(stop_double_pass_cumulative(&t2), 3);
    let t3: Vec<Step> = (1..=5).map(|i| step(i, i - 1, true, false, false)).collect();
    assert_eq!(stop_run_to_budget(&t3), 5);
    assert_eq!(stop_first_self_pass(&t3), 5);
    assert_eq!(stop_double_pass(&t3), 5);
    assert_eq!(stop_oracle(&t3), 5);
}

#[test]
fn digest_counts() {
    let t = vec![
        step(1, 0, true, false, false),
        step(2, 1, true, true, true),
        step(3, 1, false, true, true),
        step(4, 2, true, false, false),
    ];
    // [steps, hidden passes, self passes, final patch, changes, first hidden]
    assert_eq!(trajectory_digest(&t), [4, 2, 2, 2, 3, 2]);
}

#[test]
fn paired_quality_delta_known_case() {
    // A passes where B fails once, B never passes where A fails.
    let a = [true, false, true];
    let b = [false, false, true];
    let (delta, wins, losses, ties, _ci) = paired_quality_delta(&a, &b);
    assert_eq!(wins, 1);
    assert_eq!(losses, 0);
    assert_eq!(ties, 2);
    assert!((delta - 1.0 / 3.0).abs() < 1e-12);
    // Identical finals: delta 0, all ties.
    let (delta, wins, losses, ties, _) = paired_quality_delta(&a, &a);
    assert_eq!(delta, 0.0);
    assert_eq!((wins, losses, ties), (0, 0, 3));
}

#[test]
fn evaluate_all_invariants() {
    let params = TrajectoryParams::default();
    let mut rng = SplitMix64::new(5);
    let trajs: Vec<Vec<Step>> =
        (0..500).map(|_| generate_trajectory(&mut rng, &params)).collect();
    let ev = evaluate_all(&trajs, &params);
    assert_eq!(ev.n_runs, 500);
    assert_eq!(ev.budget, 30);
    assert_eq!(ev.baseline.saving_pct, 0.0);
    assert_eq!(ev.baseline.fire_rate, 0.0);
    assert!(ev.double_pass.saving_pct > 0.0);
    assert!(ev.double_pass.saving_pct < 100.0);
    assert!(ev.oracle.saving_pct >= 0.0);
    assert!(ev.double_pass.fire_rate > 0.0);
    assert!(ev.double_pass.fire_rate < 1.0);
    // Cumulative and consecutive double-pass coincide under stickiness.
    assert_eq!(
        ev.double_pass.sum_stop_steps,
        ev.double_pass_cumulative.sum_stop_steps
    );
    assert_eq!(ev.double_pass.n_final_pass, ev.double_pass_cumulative.n_final_pass);
    assert_eq!(ev.post_fire.n_fired, ev.double_pass.n_fired);
    assert!(ev.post_fire.changed_after_fire <= ev.post_fire.n_fired);
    assert!(ev.quality_wins + ev.quality_losses + ev.quality_ties == 500);
}
