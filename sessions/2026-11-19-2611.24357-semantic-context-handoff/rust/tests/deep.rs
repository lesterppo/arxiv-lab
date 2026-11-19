//! Deep tests for the semantic-context-handoff mechanism.
//!
//! Property sweeps over random inputs/budgets, exhaustive + random
//! small cases checked against an independent oracle implementation, a
//! soak at a scale the Python test does not reach, and thread-identity
//! for the stateless code.
//!
//! Run: cargo test --release --offline --test deep

use semantic_context_handoff_rs::*;
use std::collections::HashSet;

fn default_params() -> TaskParams {
    TaskParams::default()
}

/// Independent brute-force reference for `evaluate_policy`.
///
/// Structurally different from the library implementation: it first
/// walks the task once, precomputing the recovery outcome table
/// (recall hit / false-positive hit / distractor size per missing
/// delayed fact, in need order, driving its own recovery stream), then
/// computes step success and tokens from that table with a running
/// retained-token sum instead of re-filtering the facts each step.
/// It shares only the documented draw-order contract with the library.
fn reference_evaluate(
    task: &Task,
    policy: Policy,
    params: &TaskParams,
    seed: u64,
    task_index: u64,
) -> (usize, bool, u64, u64, u64) {
    // (steps_ok, task_pass, total_tokens, recoveries, recovery_hits)
    let recall = if policy == Policy::Oracle {
        1.0
    } else {
        params.recall
    };
    let fp = if policy == Policy::Oracle {
        0.0
    } else {
        params.fp_per_lookup
    };
    // Precompute the recovery table: per step, one entry per missing
    // delayed fact, in need order.
    let mut table: Vec<Vec<(bool, bool, u64)>> = vec![Vec::new(); task.steps()];
    if policy == Policy::Sch || policy == Policy::Oracle {
        let mut rng = recovery_rng(seed, task_index);
        for t in 0..task.steps() {
            for &k in &task.needs[t] {
                let f = &task.facts[k];
                if f.produced_step < t && !f.retained {
                    let hit = rng.bernoulli(recall);
                    let fp_hit = rng.bernoulli(fp);
                    let distractor = if fp_hit {
                        rng.randint(params.min_fact_tokens, params.max_fact_tokens)
                    } else {
                        0
                    };
                    table[t].push((hit, fp_hit, distractor));
                }
            }
        }
    }
    let mut steps_ok = 0usize;
    let mut total_tokens: u64 = 0;
    let mut recoveries: u64 = 0;
    let mut recovery_hits: u64 = 0;
    // Running sums instead of per-step filters.
    let mut produced_tokens: u64 = 0;
    let mut retained_compressed: u64 = 0;
    for t in 0..task.steps() {
        for f in task.facts.iter().filter(|f| f.produced_step == t) {
            produced_tokens += f.tokens;
            if f.retained {
                retained_compressed += (f.tokens as f64 * params.compression) as u64;
            }
        }
        if policy == Policy::Full {
            total_tokens += produced_tokens;
            steps_ok += 1;
            continue;
        }
        let mut step_tokens = params.state_overhead + retained_compressed;
        let mut step_ok = true;
        let mut outcome_i = 0usize;
        for &k in &task.needs[t] {
            let f = &task.facts[k];
            if f.produced_step >= t {
                continue; // current observation, always in context
            }
            if f.retained {
                continue;
            }
            match policy {
                Policy::Summary => {
                    step_ok = false;
                    break;
                }
                Policy::Sch | Policy::Oracle => {
                    let (hit, fp_hit, distractor) = table[t][outcome_i];
                    outcome_i += 1;
                    recoveries += 1;
                    if hit {
                        recovery_hits += 1;
                        step_tokens += f.tokens + params.lookup_overhead;
                    } else {
                        step_ok = false;
                        step_tokens += params.lookup_overhead;
                    }
                    if fp_hit {
                        step_tokens += distractor;
                    }
                }
                Policy::Full => unreachable!(),
            }
        }
        total_tokens += step_tokens;
        if step_ok {
            steps_ok += 1;
        }
    }
    let threshold = (params.pass_step_fraction * task.steps() as f64).ceil() as usize;
    (
        steps_ok,
        steps_ok >= threshold,
        total_tokens,
        recoveries,
        recovery_hits,
    )
}

fn check_against_reference(
    task: &Task,
    params: &TaskParams,
    seed: u64,
    task_index: u64,
) {
    for p in Policy::all() {
        let r = evaluate_policy(task, p, params, seed, task_index);
        let (steps_ok, task_pass, tokens, recoveries, hits) =
            reference_evaluate(task, p, params, seed, task_index);
        assert_eq!(r.steps_ok, steps_ok, "policy {} steps_ok", p.name());
        assert_eq!(r.task_pass, task_pass, "policy {} task_pass", p.name());
        assert_eq!(r.total_tokens, tokens, "policy {} tokens", p.name());
        assert_eq!(r.recoveries, recoveries, "policy {} recoveries", p.name());
        assert_eq!(r.recovery_hits, hits, "policy {} hits", p.name());
    }
}

#[test]
fn policies_match_brute_force_reference() {
    // Random generated tasks across random regimes.
    let mut rng = SplitMix64::new(20261119);
    for i in 0..2000 {
        let p = TaskParams {
            steps: 1 + (rng.next_u64() % 12) as usize,
            facts_per_step: 1 + (rng.next_u64() % 4) as usize,
            delayed_frac: rng.uniform(),
            max_delay: 1 + (rng.next_u64() % 6) as usize,
            min_fact_tokens: 1 + rng.next_u64() % 20,
            max_fact_tokens: 30 + rng.next_u64() % 30,
            state_retention: rng.uniform(),
            compression: rng.uniform(),
            state_overhead: rng.next_u64() % 100,
            recall: rng.uniform(),
            lookup_overhead: rng.next_u64() % 10,
            fp_per_lookup: rng.uniform(),
            pass_step_fraction: 0.5 + rng.uniform() * 0.5,
        };
        // max_fact_tokens must stay >= min_fact_tokens.
        let p = TaskParams {
            max_fact_tokens: p.max_fact_tokens.max(p.min_fact_tokens),
            ..p
        };
        let seed = rng.next_u64();
        let task = generate_task(&mut SplitMix64::new(seed), &p);
        check_against_reference(&task, &p, seed, i);
    }
    // Calibrated regime, many tasks.
    let params = default_params();
    let mut rng = SplitMix64::new(99);
    for i in 0..1000 {
        let seed = rng.next_u64();
        let task = generate_task(&mut SplitMix64::new(seed), &params);
        check_against_reference(&task, &params, seed, i);
    }
}

#[test]
fn exhaustive_small_tasks() {
    // Every 2-step, 1-fact-per-step task: delay of f0 in {0, 1},
    // retained flags, token sizes in {10, 30}. f1 is always produced
    // at step 1 and needed at step 1 (delay 0). Checked against the
    // brute-force reference for every policy and several recall/fp
    // settings, with several recovery seeds.
    let params = default_params();
    let mut checked = 0usize;
    for delay0 in [0usize, 1] {
        for ret0 in [false, true] {
            for ret1 in [false, true] {
                for tok0 in [10u64, 30] {
                    for tok1 in [10u64, 30] {
                        let task = Task {
                            facts: vec![
                                Fact {
                                    key: 0,
                                    produced_step: 0,
                                    needed_step: delay0,
                                    tokens: tok0,
                                    retained: ret0,
                                },
                                Fact {
                                    key: 1,
                                    produced_step: 1,
                                    needed_step: 1,
                                    tokens: tok1,
                                    retained: ret1,
                                },
                            ],
                            needs: {
                                let mut n: Vec<Vec<usize>> = vec![vec![], vec![]];
                                if delay0 == 0 {
                                    n[0].push(0);
                                } else {
                                    n[1].push(0);
                                }
                                n[1].push(1);
                                n
                            },
                        };
                        for rc in [0.0, 0.5, 0.9, 1.0] {
                            for fp in [0.0, 0.2, 1.0] {
                                let p = TaskParams {
                                    recall: rc,
                                    fp_per_lookup: fp,
                                    ..params
                                };
                                for seed in [0u64, 7, 12345] {
                                    check_against_reference(&task, &p, seed, 3);
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 1000, "checked {} small cases", checked);
}

#[test]
fn property_sweep_random_params_and_budgets() {
    // Random regimes, including extremes: no delay, always delayed,
    // no retention, full retention, no recall, perfect recall, tiny
    // budgets, single fact per step, wide token ranges.
    let mut rng = SplitMix64::new(777);
    let mut seen_savings = HashSet::new();
    for _ in 0..200 {
        let p = TaskParams {
            steps: 1 + (rng.next_u64() % 20) as usize,
            facts_per_step: 1 + (rng.next_u64() % 5) as usize,
            delayed_frac: rng.uniform(),
            max_delay: 1 + (rng.next_u64() % 6) as usize,
            min_fact_tokens: 1 + rng.next_u64() % 20,
            max_fact_tokens: 30 + rng.next_u64() % 50,
            state_retention: rng.uniform(),
            compression: rng.uniform(),
            state_overhead: rng.next_u64() % 100,
            recall: rng.uniform(),
            lookup_overhead: rng.next_u64() % 10,
            fp_per_lookup: rng.uniform(),
            pass_step_fraction: 0.5 + rng.uniform() * 0.5,
        };
        let p = TaskParams {
            max_fact_tokens: p.max_fact_tokens.max(p.min_fact_tokens),
            ..p
        };
        let tasks: Vec<Task> = (0..200)
            .map(|_| generate_task(&mut rng, &p))
            .collect();
        let ev = evaluate_all(&tasks, &p, 777);
        for pol in Policy::all() {
            let a = ev.get(pol);
            assert!((0.0..=1.0).contains(&a.task_pass_rate));
            assert!((0.0..=1.0).contains(&a.step_success_rate));
        }
        // FULL always pays full fact tokens (facts exist, tokens >= 1).
        // SUMMARY/SCH/ORACLE may legitimately total 0 in the degenerate
        // state_overhead=0, compression=0 regime -- Python agrees.
        assert!(ev.full.tokens > 0);
        // Invariants that must hold in every regime.
        assert_eq!(ev.full.task_pass_rate, 1.0, "FULL always passes");
        assert_eq!(ev.oracle.task_pass_rate, 1.0, "ORACLE always passes");
        assert_eq!(ev.full.step_success_rate, 1.0);
        assert_eq!(ev.oracle.step_success_rate, 1.0);
        assert!(
            ev.summary.task_pass_rate <= ev.sch.task_pass_rate + 1e-12,
            "SCH never below SUMMARY"
        );
        assert!(
            ev.sch.task_pass_rate <= ev.oracle.task_pass_rate + 1e-12,
            "ORACLE never below SCH"
        );
        assert_eq!(
            ev.sch.recoveries, ev.oracle.recoveries,
            "SCH and ORACLE attempt the same recoveries"
        );
        assert!(ev.sch.recovery_hits <= ev.sch.recoveries);
        assert_eq!(ev.oracle.recovery_hits, ev.oracle.recoveries);
        assert!(ev.sch.tokens >= ev.summary.tokens, "recovery only adds tokens");
        let saving = 100.0 * (1.0 - ev.sch.mean_tokens_per_task / ev.full.mean_tokens_per_task);
        seen_savings.insert((saving * 10.0) as i64);
    }
    assert!(seen_savings.len() > 20, "sweep covered distinct regimes");
}

#[test]
fn recall_monotonicity_and_ablations() {
    // Fixed tasks: SCH step success is pointwise non-decreasing in
    // recall (same recovery stream, same draw positions), recall=0
    // collapses to SUMMARY per task, recall=1 matches ORACLE per task.
    let params = default_params();
    let mut rng = SplitMix64::new(4242);
    let tasks: Vec<Task> = (0..300)
        .map(|_| generate_task(&mut rng, &params))
        .collect();
    for (i, t) in tasks.iter().enumerate() {
        let s = evaluate_policy(t, Policy::Summary, &params, 4242, i as u64);
        let o = evaluate_policy(t, Policy::Oracle, &params, 4242, i as u64);
        let mut prev = 0usize;
        for rc in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let p = TaskParams {
                recall: rc,
                ..params
            };
            let r = evaluate_policy(t, Policy::Sch, &p, 4242, i as u64);
            assert!(
                r.steps_ok >= prev,
                "step success monotone in recall (task {}, rc {})",
                i,
                rc
            );
            prev = r.steps_ok;
            if rc == 0.0 {
                assert_eq!(r.steps_ok, s.steps_ok, "recall=0 == SUMMARY steps");
                assert_eq!(r.task_pass, s.task_pass);
            }
            if rc == 1.0 {
                assert_eq!(r.steps_ok, o.steps_ok, "recall=1 == ORACLE steps");
                assert_eq!(r.task_pass, o.task_pass);
            }
        }
    }
}

#[test]
fn extreme_regimes() {
    // delayed_frac = 0: nothing ever needed later; all pass, no recovery.
    let p = TaskParams {
        delayed_frac: 0.0,
        ..default_params()
    };
    let mut rng = SplitMix64::new(3);
    let tasks: Vec<Task> = (0..200)
        .map(|_| generate_task(&mut rng, &p))
        .collect();
    let ev = evaluate_all(&tasks, &p, 3);
    for pol in Policy::all() {
        assert_eq!(ev.get(pol).task_pass_rate, 1.0);
    }
    assert_eq!(ev.sch.recoveries, 0);
    // state_retention = 1: everything is in the state, so SUMMARY
    // passes every step exactly like FULL (tokens differ: compressed).
    let p = TaskParams {
        state_retention: 1.0,
        ..default_params()
    };
    let mut rng = SplitMix64::new(4);
    let tasks: Vec<Task> = (0..200)
        .map(|_| generate_task(&mut rng, &p))
        .collect();
    let ev = evaluate_all(&tasks, &p, 4);
    assert_eq!(ev.summary.step_success_rate, 1.0);
    assert_eq!(ev.summary.task_pass_rate, 1.0);
    assert_eq!(ev.sch.recoveries, 0);
    assert!(ev.summary.tokens < ev.full.tokens);
    // state_retention = 0 + recall = 0: SCH == SUMMARY on every task,
    // and every delayed fact fails its step.
    let p = TaskParams {
        state_retention: 0.0,
        recall: 0.0,
        ..default_params()
    };
    let mut rng = SplitMix64::new(5);
    let tasks: Vec<Task> = (0..200)
        .map(|_| generate_task(&mut rng, &p))
        .collect();
    let ev = evaluate_all(&tasks, &p, 5);
    assert_eq!(ev.sch.steps_ok, ev.summary.steps_ok);
    assert_eq!(ev.sch.passes, ev.summary.passes);
    assert!(ev.summary.task_pass_rate < 1.0);
    // steps = 1, facts_per_step = 1: no fact can be delayed past the
    // only step, so every policy passes; exact token formulas.
    let p = TaskParams {
        steps: 1,
        facts_per_step: 1,
        ..default_params()
    };
    let task = generate_task(&mut SplitMix64::new(11), &p);
    let f = task.facts[0];
    assert_eq!(task.needs, vec![vec![0]]);
    let full = evaluate_policy(&task, Policy::Full, &p, 11, 0);
    assert_eq!(full.total_tokens, f.tokens);
    let sum = evaluate_policy(&task, Policy::Summary, &p, 11, 0);
    let expect_state = 40 + if f.retained {
        (f.tokens as f64 * 0.3) as u64
    } else {
        0
    };
    assert_eq!(sum.total_tokens, expect_state);
    assert_eq!(sum.steps_ok, 1);
    // compression = 0: retained facts cost nothing in the state.
    let p = TaskParams {
        compression: 0.0,
        ..default_params()
    };
    let mut rng = SplitMix64::new(6);
    let tasks: Vec<Task> = (0..100)
        .map(|_| generate_task(&mut rng, &p))
        .collect();
    let ev = evaluate_all(&tasks, &p, 6);
    // Every step's state is exactly the 40-token overhead.
    assert_eq!(ev.summary.tokens, 100 * 12 * 40);
}

#[test]
fn soak_100k_tasks() {
    // A scale the Python test does not reach (Python: 10k main + 10.8k
    // sweep + 400 no-delay). Same seeds, same assertions, plus
    // stability across two independent passes over the same streams.
    let params = default_params();
    let seeds = [20261119u64, 7, 123, 2026, 99];
    let run_once = || {
        let mut all = Vec::new();
        for seed in seeds {
            let mut rng = SplitMix64::new(seed);
            for _ in 0..20_000 {
                all.push(generate_task(&mut rng, &params));
            }
        }
        evaluate_all(&all, &params, seeds[0])
    };
    let ev1 = run_once();
    let ev2 = run_once();
    assert_eq!(ev1.full.tasks, 100_000);
    assert_eq!(ev1.sch.tokens, ev2.sch.tokens);
    assert_eq!(ev1.summary.passes, ev2.summary.passes);
    assert_eq!(ev1.sch.passes, ev2.sch.passes);
    assert_eq!(ev1.full.tokens, ev2.full.tokens);
    assert!((0.99..=1.0).contains(&ev1.sch.task_pass_rate));
    assert!((0.90..=0.95).contains(&ev1.summary.task_pass_rate));
    assert!((0.98..=1.0).contains(&ev1.sch.step_success_rate));
    assert!((0.90..=0.93).contains(&ev1.summary.step_success_rate));
    let saving = 100.0 * (1.0 - ev1.sch.mean_tokens_per_task / ev1.full.mean_tokens_per_task);
    assert!((60.0..=70.0).contains(&saving));
    // Record the soak numbers for results.json (printed, not asserted
    // beyond the bands above).
    println!(
        "SOAK 100k tasks: summary pass={:.4} step={:.4} | sch pass={:.4} step={:.4} tokens/task={:.1} | full tokens/task={:.1} | saving={:.2}% | hit_rate={:.4}",
        ev1.summary.task_pass_rate,
        ev1.summary.step_success_rate,
        ev1.sch.task_pass_rate,
        ev1.sch.step_success_rate,
        ev1.sch.mean_tokens_per_task,
        ev1.full.mean_tokens_per_task,
        saving,
        ev1.sch.recovery_hit_rate.unwrap()
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
                let mut rng = SplitMix64::new(20261119);
                (0..2000)
                    .map(|_| task_digest(&generate_task(&mut rng, &params)))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for r in &results[1..] {
        assert_eq!(r, &results[0]);
    }
    // Same for evaluation: identical integer aggregates across threads.
    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let params = default_params();
                let mut rng = SplitMix64::new(7);
                let tasks: Vec<Task> = (0..500)
                    .map(|_| generate_task(&mut rng, &params))
                    .collect();
                let ev = evaluate_all(&tasks, &params, 7);
                (
                    ev.sch.tokens,
                    ev.sch.passes,
                    ev.summary.tokens,
                    ev.full.tokens,
                )
            })
        })
        .collect();
    let evals: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for e in &evals[1..] {
        assert_eq!(e, &evals[0]);
    }
}
