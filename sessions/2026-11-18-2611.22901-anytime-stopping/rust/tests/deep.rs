//! Deep tests for the anytime-stopping mechanism.
//!
//! Property sweeps over random inputs/budgets, exhaustive + random
//! sequences checked against independent oracle implementations, a soak
//! at a scale the Python test does not reach, and thread-identity for
//! the stateless code.
//!
//! Run: cargo test --release --offline --test deep

use anytime_stopping_rs::*;
use std::collections::HashSet;

fn default_params() -> TrajectoryParams {
    TrajectoryParams::default()
}

/// Independent brute-force reference for the double-pass rule: scan
/// for the first step that is the second consecutive self-pass of the
/// same patch id.
fn oracle_double_pass(traj: &[Step]) -> usize {
    let mut prev: Option<(usize, bool, usize)> = None; // (patch, self, run)
    for s in traj {
        let run = match prev {
            Some((p, sp, r)) if p == s.patch_id && sp && s.self_pass => r + 1,
            _ => {
                if s.self_pass {
                    1
                } else {
                    0
                }
            }
        };
        if run >= 2 {
            return s.step;
        }
        prev = Some((s.patch_id, s.self_pass, run));
    }
    traj.len()
}

/// Independent brute-force reference for the oracle policy.
fn oracle_first_hidden(traj: &[Step]) -> usize {
    traj.iter()
        .find(|s| s.hidden_pass)
        .map(|s| s.step)
        .unwrap_or(traj.len())
}

#[test]
fn policies_match_brute_force_oracles() {
    let params = default_params();
    let mut rng = SplitMix64::new(20261118);
    for _ in 0..5000 {
        let t = generate_trajectory(&mut rng, &params);
        assert_eq!(stop_double_pass(&t), oracle_double_pass(&t));
        assert_eq!(stop_oracle(&t), oracle_first_hidden(&t));
        // First-self-pass is a direct scan.
        let expect = t
            .iter()
            .find(|s| s.self_pass)
            .map(|s| s.step)
            .unwrap_or(t.len());
        assert_eq!(stop_first_self_pass(&t), expect);
    }
}

#[test]
fn exhaustive_short_sequences() {
    // Every length-4 sequence over a small alphabet of step shapes,
    // checked against the brute-force oracles. Alphabet: (changed,
    // hidden, self) with patch ids assigned by the changed flags.
    let shapes = [
        (true, false, false),
        (true, false, true),
        (true, true, false),
        (true, true, true),
        (false, false, false),
        (false, true, true),
    ];
    let mut checked = 0usize;
    for a in &shapes {
        for b in &shapes {
            for c in &shapes {
                for d in &shapes {
                    let seq = [a, b, c, d];
                    let mut patch = 0usize;
                    let mut traj: Vec<Step> = Vec::new();
                    // Step 1 always changes (emits the first patch).
                    for (i, (changed, hidden, self_p)) in seq.iter().enumerate() {
                        let changed = if i == 0 { true } else { *changed };
                        if changed {
                            patch += 1;
                        }
                        traj.push(Step {
                            step: i + 1,
                            patch_id: patch,
                            changed,
                            hidden_pass: *hidden,
                            self_pass: *self_p,
                        });
                    }
                    // Coherence: an unchanged step must keep the
                    // previous hidden/self states (stickiness); skip
                    // incoherent sequences.
                    let coherent = traj.windows(2).all(|w| {
                        w[1].changed
                            || (w[0].hidden_pass == w[1].hidden_pass
                                && w[0].self_pass == w[1].self_pass)
                    });
                    if !coherent {
                        continue;
                    }
                    assert_eq!(stop_double_pass(&traj), oracle_double_pass(&traj));
                    assert_eq!(stop_oracle(&traj), oracle_first_hidden(&traj));
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 500, "checked {} coherent sequences", checked);
}

#[test]
fn property_sweep_random_params_and_budgets() {
    // Random regimes, including extremes: unsolvable, always solvable,
    // never/always repairing, never/always breaking, tiny budgets.
    let mut rng = SplitMix64::new(777);
    let mut seen_savings = HashSet::new();
    for _ in 0..200 {
        let p = TrajectoryParams {
            budget: 1 + (rng.next_u64() % 40) as usize,
            p_solvable: rng.uniform(),
            p_fix: rng.uniform(),
            p_change_after_self_pass: rng.uniform(),
            p_change_after_self_fail: rng.uniform(),
            p_break: rng.uniform(),
            p_self_false_pass: rng.uniform(),
            p_self_false_fail: rng.uniform(),
        };
        let trajs: Vec<Vec<Step>> = (0..200)
            .map(|_| generate_trajectory(&mut rng, &p))
            .collect();
        let ev = evaluate_all(&trajs, &p);
        assert!((0.0..=100.0).contains(&ev.double_pass.saving_pct));
        assert!((0.0..=1.0).contains(&ev.baseline.final_hidden_pass_rate));
        assert!((0.0..=1.0).contains(&ev.double_pass.fire_rate));
        assert!(ev.post_fire.changed_after_fire <= ev.post_fire.n_fired);
        assert_eq!(
            ev.quality_wins + ev.quality_losses + ev.quality_ties,
            200
        );
        // Monotone ordering that must hold in every regime: the
        // earlier-stopping first-self-pass rule never saves less than
        // the double-pass rule, and never beats it on final quality
        // by more than the double-pass rule's own margin... (only the
        // saving ordering is asserted: first pass stops no later.)
        assert!(
            ev.first_self_pass.saving_pct >= ev.double_pass.saving_pct - 1e-9,
            "first-self-pass must save >= double-pass: {:?}",
            p
        );
        seen_savings.insert((ev.double_pass.saving_pct * 10.0) as u64);
    }
    assert!(seen_savings.len() > 20, "sweep covered distinct regimes");
}

#[test]
fn extreme_regimes() {
    // p_solvable = 0: nothing ever passes hidden; oracle never fires,
    // double-pass fires only on false self-passes.
    let p = TrajectoryParams {
        p_solvable: 0.0,
        ..default_params()
    };
    let mut rng = SplitMix64::new(3);
    let trajs: Vec<Vec<Step>> =
        (0..500).map(|_| generate_trajectory(&mut rng, &p)).collect();
    let ev = evaluate_all(&trajs, &p);
    assert_eq!(ev.baseline.final_hidden_pass_rate, 0.0);
    assert_eq!(ev.oracle.fire_rate, 0.0);
    assert_eq!(ev.oracle.saving_pct, 0.0);
    assert_eq!(ev.double_pass.final_hidden_pass_rate, 0.0);
    // p_solvable = 1, p_fix = 1: every changed failing patch repairs.
    let p = TrajectoryParams {
        p_solvable: 1.0,
        p_fix: 1.0,
        p_break: 0.0,
        ..default_params()
    };
    let mut rng = SplitMix64::new(4);
    let trajs: Vec<Vec<Step>> =
        (0..200).map(|_| generate_trajectory(&mut rng, &p)).collect();
    let ev = evaluate_all(&trajs, &p);
    // Once a patch passes it never breaks, so the baseline ends passing
    // whenever a pass was ever reached (nearly always).
    assert!(ev.baseline.final_hidden_pass_rate > 0.9);
    assert!(ev.oracle.final_hidden_pass_rate > 0.9);
    // budget = 1: every policy stops at step 1, saving 0.
    let p = TrajectoryParams {
        budget: 1,
        ..default_params()
    };
    let mut rng = SplitMix64::new(5);
    let trajs: Vec<Vec<Step>> =
        (0..100).map(|_| generate_trajectory(&mut rng, &p)).collect();
    let ev = evaluate_all(&trajs, &p);
    for agg in [
        &ev.baseline,
        &ev.first_self_pass,
        &ev.double_pass,
        &ev.oracle,
    ] {
        assert_eq!(agg.saving_pct, 0.0);
        assert_eq!(agg.fire_rate, 0.0);
    }
}

#[test]
fn soak_200k_runs() {
    // A scale the Python test does not reach (Python: 10k main + 44k
    // sweep). Same seeds, same assertions, plus stability across two
    // independent passes over the same streams.
    let params = default_params();
    let seeds = [20261118u64, 7, 123, 2026, 99];
    let run_once = || {
        let mut trajs = Vec::new();
        for seed in seeds {
            let mut rng = SplitMix64::new(seed);
            for _ in 0..40_000 {
                trajs.push(generate_trajectory(&mut rng, &params));
            }
        }
        evaluate_all(&trajs, &params)
    };
    let ev1 = run_once();
    let ev2 = run_once();
    assert_eq!(ev1.n_runs, 200_000);
    assert_eq!(ev1.double_pass.sum_stop_steps, ev2.double_pass.sum_stop_steps);
    assert_eq!(ev1.baseline.n_final_pass, ev2.baseline.n_final_pass);
    assert!((15.0..=35.0).contains(&ev1.double_pass.saving_pct));
    assert!((0.28..=0.50).contains(&ev1.baseline.final_hidden_pass_rate));
    assert!((1.0..=57.0).contains(&ev1.hindsight_all_runs_pct));
    assert!(ev1.post_fire.repairs_after_fire > 0);
    assert!(ev1.post_fire.breaks_after_fire > 0);
    assert!(ev1.quality_losses > 0);
    // Record the soak numbers for results.json (printed, not asserted
    // beyond the bands above).
    println!(
        "SOAK 200k runs: double-pass saving={:.2}% baseline pass={:.4} \
         quality delta={:+.2}pp wins={} losses={}",
        ev1.double_pass.saving_pct,
        ev1.baseline.final_hidden_pass_rate,
        ev1.quality_delta_pp,
        ev1.quality_wins,
        ev1.quality_losses
    );
}

#[test]
fn thread_identity() {
    // The code is stateless pure functions: four threads generating
    // from the same seeds must produce bit-identical digests.
    use std::thread;
    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let params = default_params();
                let mut rng = SplitMix64::new(20261118);
                (0..2000)
                    .map(|_| trajectory_digest(&generate_trajectory(&mut rng, &params)))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for r in &results[1..] {
        assert_eq!(r, &results[0]);
    }
}
