//! Unit tests for the semantic-context-handoff mechanism (one per
//! mechanism).
//!
//! Run: cargo test --release --offline

use semantic_context_handoff_rs::*;

fn fact(key: usize, produced: usize, needed: usize, tokens: u64, retained: bool) -> Fact {
    Fact {
        key,
        produced_step: produced,
        needed_step: needed,
        tokens,
        retained,
    }
}

#[test]
fn splitmix64_reference_vector() {
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next_u64(), 0xE220A8397B1DCDAF);
    assert_eq!(r.next_u64(), 0x6E789E6AA1B965F4);
    assert_eq!(r.next_u64(), 0x6C45D188009454F);
    let mut r = SplitMix64::new(42);
    assert_eq!(r.next_u64(), 0xBDD732262FEB6E95);
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
    // The 53-bit mantissa behind uniform() matches Python.
    assert_eq!(SplitMix64::new(1).mantissa(), 5103132997656651);
    // randint stays in range, endpoints reachable over many draws.
    let mut r = SplitMix64::new(5);
    let mut seen_lo = false;
    let mut seen_hi = false;
    for _ in 0..10000 {
        let v = r.randint(10, 30);
        assert!((10..=30).contains(&v));
        if v == 10 {
            seen_lo = true;
        }
        if v == 30 {
            seen_hi = true;
        }
    }
    assert!(seen_lo && seen_hi, "randint reaches both endpoints");
}

#[test]
fn default_params_are_calibrated() {
    let p = TaskParams::default();
    assert_eq!(p.steps, 12);
    assert_eq!(p.facts_per_step, 3);
    assert_eq!(p.delayed_frac, 0.25);
    assert_eq!(p.max_delay, 4);
    assert_eq!(p.min_fact_tokens, 10);
    assert_eq!(p.max_fact_tokens, 30);
    assert_eq!(p.state_retention, 0.85);
    assert_eq!(p.compression, 0.3);
    assert_eq!(p.state_overhead, 40);
    assert_eq!(p.recall, 0.9);
    assert_eq!(p.lookup_overhead, 5);
    assert_eq!(p.fp_per_lookup, 0.2);
    assert_eq!(p.pass_step_fraction, 0.8);
}

#[test]
fn task_generation_invariants() {
    let params = TaskParams::default();
    let mut rng = SplitMix64::new(20261119);
    for _ in 0..200 {
        let t = generate_task(&mut rng, &params);
        assert_eq!(t.facts.len(), params.steps * params.facts_per_step);
        assert_eq!(t.needs.len(), params.steps);
        for f in &t.facts {
            assert!(f.needed_step - f.produced_step <= params.max_delay);
            assert!((params.min_fact_tokens..=params.max_fact_tokens).contains(&f.tokens));
        }
        // Needs are consistent: every needed key exists and is needed
        // at exactly its needed_step.
        for (s, keys) in t.needs.iter().enumerate() {
            for &k in keys {
                assert_eq!(t.facts[k].needed_step, s);
            }
        }
        // Keys are the production-order indices.
        for (i, f) in t.facts.iter().enumerate() {
            assert_eq!(f.key, i);
        }
    }
}

#[test]
fn task_generation_determinism_and_digest() {
    let params = TaskParams::default();
    let mk = |seed: u64| task_digest(&generate_task(&mut SplitMix64::new(seed), &params));
    assert_eq!(mk(999), mk(999));
    assert_ne!(mk(999), mk(1000));
    // Known digest + facts + needs, cross-checked against Python.
    assert_eq!(mk(123), 16768509106337255366);
    let t = generate_task(&mut SplitMix64::new(123), &params);
    assert_eq!(
        t.facts[..3].to_vec(),
        vec![
            fact(0, 0, 0, 24, true),
            fact(1, 0, 0, 28, true),
            fact(2, 0, 0, 24, true),
        ]
    );
    assert_eq!(
        t.needs,
        vec![
            vec![0, 1, 2],
            vec![3, 5],
            vec![6, 7],
            vec![9, 10, 11],
            vec![4, 12, 13, 14],
            vec![8, 15, 16, 17],
            vec![18, 19, 20],
            vec![21, 23],
            vec![25, 26],
            vec![27, 28, 29],
            vec![22, 24, 30, 32],
            vec![33, 34, 35],
        ]
    );
    // Delayed facts exist in the calibrated regime (the crux).
    let delayed = t
        .facts
        .iter()
        .filter(|f| f.needed_step > f.produced_step)
        .count();
    assert!(delayed > 0);
}

#[test]
fn policies_on_hand_built_task() {
    // 2 steps, 1 fact per step. f0 is delayed (needed at step 1) and
    // NOT retained; f1 is current at step 1 and retained.
    // pass threshold: ceil(0.8 * 2) = 2, so a task passes only if both
    // steps succeed.
    let params = TaskParams::default();
    let task = Task {
        facts: vec![fact(0, 0, 1, 20, false), fact(1, 1, 1, 10, true)],
        needs: vec![vec![], vec![0, 1]],
    };
    // FULL: step 0 sees f0 (20 tokens), step 1 sees both (30).
    let r = evaluate_policy(&task, Policy::Full, &params, 0, 0);
    assert_eq!(r.steps_ok, 2);
    assert!(r.task_pass);
    assert!(r.strict_task_pass);
    assert_eq!(r.total_tokens, 50);
    assert_eq!(r.recoveries, 0);
    // SUMMARY: step 0 state = 40 (f0 not retained); step 1 state =
    // 40 + int(10 * 0.3) = 43, but f0 is missing -> step fails.
    let r = evaluate_policy(&task, Policy::Summary, &params, 0, 0);
    assert_eq!(r.steps_ok, 1);
    assert!(!r.task_pass);
    assert!(!r.strict_task_pass);
    assert_eq!(r.total_tokens, 83);
    assert_eq!(r.recoveries, 0);
    // ORACLE: recovers f0 at step 1 (20 tokens + 5 lookup overhead),
    // no false positives.
    let r = evaluate_policy(&task, Policy::Oracle, &params, 0, 0);
    assert_eq!(r.steps_ok, 2);
    assert!(r.task_pass);
    assert_eq!(r.total_tokens, 40 + (43 + 25));
    assert_eq!(r.recoveries, 1);
    assert_eq!(r.recovery_hits, 1);
    // SCH with recall=1, fp=0: same tokens as ORACLE.
    let p = TaskParams {
        recall: 1.0,
        fp_per_lookup: 0.0,
        ..params
    };
    let r = evaluate_policy(&task, Policy::Sch, &p, 0, 0);
    assert_eq!(r.steps_ok, 2);
    assert!(r.task_pass);
    assert_eq!(r.total_tokens, 108);
    assert_eq!((r.recoveries, r.recovery_hits), (1, 1));
    // SCH with recall=0, fp=0: lookup misses (5 overhead), step fails.
    let p = TaskParams {
        recall: 0.0,
        fp_per_lookup: 0.0,
        ..params
    };
    let r = evaluate_policy(&task, Policy::Sch, &p, 0, 0);
    assert_eq!(r.steps_ok, 1);
    assert!(!r.task_pass);
    assert_eq!(r.total_tokens, 40 + (43 + 5));
    assert_eq!((r.recoveries, r.recovery_hits), (1, 0));
}

#[test]
fn ablation_recall_zero_collapses_to_summary() {
    let params = TaskParams::default();
    let p0 = TaskParams {
        recall: 0.0,
        ..params
    };
    let mut rng = SplitMix64::new(20261119);
    for i in 0..200 {
        let t = generate_task(&mut rng, &params);
        let s = evaluate_policy(&t, Policy::Summary, &params, 20261119, i);
        let r = evaluate_policy(&t, Policy::Sch, &p0, 20261119, i);
        assert_eq!(r.steps_ok, s.steps_ok, "recall=0: same steps_ok");
        assert_eq!(r.task_pass, s.task_pass, "recall=0: same task_pass");
        // Recovery is attempted exactly for the missing delayed facts.
        let missing = t
            .needs
            .iter()
            .enumerate()
            .map(|(step, keys)| {
                keys.iter()
                    .filter(|&&k| {
                        let f = &t.facts[k];
                        f.produced_step < step && !f.retained
                    })
                    .count()
            })
            .sum::<usize>();
        assert_eq!(r.recoveries as usize, missing);
        assert_eq!(r.recovery_hits, 0);
    }
}

#[test]
fn oracle_is_upper_bound() {
    let params = TaskParams::default();
    let mut rng = SplitMix64::new(77);
    for i in 0..500 {
        let t = generate_task(&mut rng, &params);
        let o = evaluate_policy(&t, Policy::Oracle, &params, 77, i);
        assert_eq!(o.steps_ok, t.steps());
        assert!(o.task_pass);
        assert!(o.strict_task_pass);
        assert_eq!(o.recovery_hits, o.recoveries);
        let s = evaluate_policy(&t, Policy::Summary, &params, 77, i);
        let r = evaluate_policy(&t, Policy::Sch, &params, 77, i);
        assert!(s.steps_ok <= r.steps_ok, "SCH never loses a SUMMARY step");
        assert!(r.steps_ok <= o.steps_ok, "ORACLE never loses an SCH step");
        // SCH and ORACLE attempt recovery for exactly the same facts.
        assert_eq!(r.recoveries, o.recoveries);
    }
}

#[test]
fn evaluate_all_invariants() {
    let params = TaskParams::default();
    let mut rng = SplitMix64::new(5);
    let tasks: Vec<Task> = (0..500)
        .map(|_| generate_task(&mut rng, &params))
        .collect();
    let ev = evaluate_all(&tasks, &params, 5);
    assert_eq!(ev.full.tasks, 500);
    assert_eq!(ev.full.passes, 500);
    assert_eq!(ev.full.strict_passes, 500);
    assert_eq!(ev.full.step_success_rate, 1.0);
    assert_eq!(ev.full.recoveries, 0);
    assert!(ev.summary.task_pass_rate < 1.0);
    assert!(ev.sch.task_pass_rate > ev.summary.task_pass_rate);
    assert!(ev.oracle.task_pass_rate == 1.0);
    assert_eq!(ev.sch.recoveries, ev.oracle.recoveries);
    assert!(ev.sch.recovery_hit_rate.unwrap() > 0.8);
    assert!(ev.sch.recovery_hit_rate.unwrap() < 1.0);
    // Token ordering in the calibrated regime: recovery only adds to
    // the state, and full history dwarfs both.
    assert!(ev.sch.tokens >= ev.summary.tokens);
    assert!(ev.full.tokens > ev.sch.tokens);
    assert_eq!(ev.sch.steps, 500 * 12);
    // Integer sums are consistent with the rates.
    assert_eq!(
        ev.summary.passes as f64 / 500.0,
        ev.summary.task_pass_rate
    );
}

#[test]
fn no_delay_sanity() {
    // Nothing ever needed later: every policy passes everything.
    let params = TaskParams {
        delayed_frac: 0.0,
        ..TaskParams::default()
    };
    let mut rng = SplitMix64::new(999983);
    let tasks: Vec<Task> = (0..400)
        .map(|_| generate_task(&mut rng, &params))
        .collect();
    let ev = evaluate_all(&tasks, &params, 999983);
    for p in Policy::all() {
        assert_eq!(ev.get(p).task_pass_rate, 1.0, "{} passes all", p.name());
        assert_eq!(ev.get(p).step_success_rate, 1.0);
    }
    // With no delayed facts, SUMMARY pays only the state: no step can
    // fail, and SCH attempts no recovery.
    assert_eq!(ev.sch.recoveries, 0);
    assert_eq!(ev.summary.steps_ok, 400 * 12);
}
