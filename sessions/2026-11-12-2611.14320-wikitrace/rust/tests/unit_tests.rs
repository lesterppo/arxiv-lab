//! Unit tests for the WikiTrace Rust port: one test per mechanism,
//! with vectors taken from the frozen Python module.

use std::collections::HashMap;
use wikitrace::*;

// ---------------------------------------------------------------- RNG

#[test]
fn splitmix64_vectors_match_python() {
    // First 8 outputs per seed, from the frozen Python module.
    let cases: &[(u64, [u64; 8])] = &[
        (
            0,
            [
                16294208416658607535,
                7960286522194355700,
                487617019471545679,
                17909611376780542444,
                1961750202426094747,
                6038094601263162090,
                3207296026000306913,
                14232521865600346940,
            ],
        ),
        (
            1,
            [
                10451216379200822465,
                13757245211066428519,
                17911839290282890590,
                8196980753821780235,
                8195237237126968761,
                14072917602864530048,
                16184226688143867045,
                9648886400068060533,
            ],
        ),
        (
            20261112,
            [
                1049095683001091839,
                2901343246710161558,
                4068396439025931088,
                18401376878894613872,
                7199368500981904300,
                5472753310247400052,
                476443444721587946,
                2913769880479557746,
            ],
        ),
        (
            u64::MAX,
            [
                16490336266968443936,
                16834447057089888969,
                4048727598324417001,
                7862637804313477842,
                13015481187462834606,
                15212506146343009075,
                17388166129998380965,
                4638043754431676516,
            ],
        ),
    ];
    for (seed, expected) in cases {
        let mut rng = SplitMix64::new(*seed);
        for (i, e) in expected.iter().enumerate() {
            assert_eq!(rng.next_u64(), *e, "seed={} output {}", seed, i);
        }
    }
}

#[test]
fn splitmix64_random_matches_python() {
    // First 4 random() values per seed, from Python repr().
    let cases: &[(u64, [f64; 4])] = &[
        (0, [0.8833108082136427, 0.43152799704851, 0.026433771592597816, 0.9708819781538285]),
        (1, [0.566561575172281, 0.7457817572627012, 0.9710027535867963, 0.4443592170557721]),
        (
            20261112,
            [0.05687159093274739, 0.1572821325604652, 0.22054821288621024, 0.9975406394411036],
        ),
    ];
    for (seed, expected) in cases {
        let mut rng = SplitMix64::new(*seed);
        for (i, e) in expected.iter().enumerate() {
            assert_eq!(rng.random(), *e, "seed={} random {}", seed, i);
        }
    }
}

#[test]
fn randbelow_edge_cases() {
    // n == 1 must terminate (the u128-limit trap) and always return 0.
    let mut rng = SplitMix64::new(42);
    for _ in 0..100 {
        assert_eq!(rng.randbelow(1), 0);
    }
    // randbelow sequences for n in [1,2,3,4,5,7,100,2^63], from Python.
    let cases: &[(u64, [u64; 8])] = &[
        (0, [0, 0, 1, 0, 2, 2, 13, 5009149828745571132]),
        (1, [0, 1, 0, 3, 1, 2, 45, 425514363213284725]),
        (20261112, [0, 0, 1, 0, 0, 1, 46, 2913769880479557746]),
        (u64::MAX, [0, 1, 1, 2, 1, 5, 65, 4638043754431676516]),
        (123456789, [0, 0, 1, 1, 3, 5, 70, 5002838425050760028]),
    ];
    for (seed, expected) in cases {
        let mut rng = SplitMix64::new(*seed);
        for (n, e) in [1u64, 2, 3, 4, 5, 7, 100, 1u64 << 63].iter().zip(expected.iter()) {
            assert_eq!(rng.randbelow(*n), *e, "seed={} randbelow({})", seed, n);
        }
    }
    // Range invariant for many n.
    let mut rng = SplitMix64::new(7);
    for n in 1..=1000u64 {
        assert!(rng.randbelow(n) < n);
    }
}

#[test]
fn randbelow_unbiased_spot_check() {
    // randbelow(3) over a large sample is near-uniform (chi-square-ish
    // spot check: each bucket within 2% of 1/3).
    let mut rng = SplitMix64::new(20261112);
    let mut counts = [0u64; 3];
    for _ in 0..90_000 {
        counts[rng.randbelow(3) as usize] += 1;
    }
    for c in counts {
        assert!((c as f64 - 30_000.0).abs() < 600.0, "bucket {}", c);
    }
}

// ---------------------------------------------------------------- gold_action

#[test]
fn gold_action_table() {
    // Brute-force oracle table: locate=0, operate=1, verify=attr%3.
    for attr in 0..8u64 {
        assert_eq!(gold_action(0, attr), 0);
        assert_eq!(gold_action(1, attr), 1);
        assert_eq!(gold_action(2, attr), attr % 3);
    }
}

// ---------------------------------------------------------------- tasks

#[test]
fn gen_tasks_deterministic_and_shaped() {
    let a = gen_tasks(20261112, 20);
    let b = gen_tasks(20261112, 20);
    assert_eq!(a, b);
    assert_eq!(a.len(), 20);
    for (i, t) in a.iter().enumerate() {
        assert_eq!(t.tid, i as u64);
        assert!(t.attr < 4);
    }
    // First attrs from the frozen Python module.
    assert_eq!(a[0].attr, 3);
    assert_eq!(a[1].attr, 2);
    assert_eq!(a[2].attr, 0);
    // Different seed -> different stream.
    assert_ne!(gen_tasks(7, 20), a);
}

#[test]
fn default_choices_paired_and_known() {
    // Same task -> same choices on every call (paired across arms).
    let t = Task::new(0, 3);
    let c1 = default_choices(&t, 20261112);
    let c2 = default_choices(&t, 20261112);
    assert_eq!(c1, c2);
    // Known values from Python: task 0 attr 3 -> [0, 1, 0]; task 1 attr 2 -> [0, 1, 2].
    assert_eq!(c1, [0, 1, 0]);
    assert_eq!(default_choices(&Task::new(1, 2), 20261112), [0, 1, 2]);
    // Every choice is a valid action.
    for seed in [0u64, 1, 7, 42, 20261112] {
        for t in gen_tasks(seed, 50) {
            for c in default_choices(&t, seed) {
                assert!(c < 3);
            }
        }
    }
}

// ---------------------------------------------------------------- runs

#[test]
fn run_default_records_trajectory() {
    let t = Task::new(0, 3);
    let r = run_default(&t, 20261112, 20261112 + 7);
    assert_eq!(r.rid, 0);
    assert_eq!(r.steps.len(), 3);
    for (i, s) in r.steps.iter().enumerate() {
        assert_eq!(s.step_type, i as u8);
        assert_eq!(s.ok, s.chosen == gold_action(s.step_type, t.attr));
    }
    assert_eq!(r.success, r.steps.iter().all(|s| s.ok));
    // Deterministic across calls.
    assert_eq!(run_default(&t, 20261112, 20261112 + 7).steps, r.steps);
}

// ---------------------------------------------------------------- compiler

fn main_memory() -> Memory {
    let train = gen_tasks(SEED, N_TRAIN);
    let runs: Vec<Run> = train.iter().map(|t| run_default(t, SEED, SEED + 7)).collect();
    compile_memory(runs, SEED + 11)
}

#[test]
fn compile_memory_counts() {
    let mem = main_memory();
    // 12 runs x 3 steps = 36 insights; matches Python results.json.
    assert_eq!(mem.insights.len(), 36);
    assert_eq!(mem.insights.iter().filter(|i| i.corrupted).count(), 4);
    assert_eq!(mem.insights.iter().filter(|i| !i.source_success).count(), 24);
    assert_eq!(mem.compilations.len(), 12);
    assert_eq!(mem.trajectories.len(), 12);
    // Iids are sequential I-0000..I-0035 in insertion order.
    for (i, ins) in mem.insights.iter().enumerate() {
        assert_eq!(ins.iid, format!("I-{:04}", i));
        assert!(ins.step_type < 3);
        assert!(ins.claimed < 3);
    }
    // Corrupted insights only ever come from failed runs, and a corrupted
    // claim is never the action the run actually took.
    for ins in &mem.insights {
        if ins.corrupted {
            assert!(!ins.source_success);
            let run = &mem.trajectories[&ins.source_run];
            let step = run.steps.iter().find(|s| s.step_type == ins.step_type).unwrap();
            assert_ne!(ins.claimed, step.chosen);
        }
    }
}

#[test]
fn memory_applicable_and_subsets() {
    let mem = main_memory();
    // Every insight applies to its own source run's attribute.
    for ins in &mem.insights {
        let run = &mem.trajectories[&ins.source_run];
        assert!(ins.applies_to(run.task.attr));
    }
    let succ = mem.subset(true);
    let fail = mem.subset(false);
    assert_eq!(succ.insights.len(), 12);
    assert_eq!(fail.insights.len(), 24);
    assert!(succ.insights.iter().all(|i| i.source_success));
    assert!(fail.insights.iter().all(|i| !i.source_success));
    // Subsets keep the full evidence links.
    assert_eq!(succ.compilations.len(), 12);
    assert_eq!(fail.trajectories.len(), 12);
}

// ---------------------------------------------------------------- evidence check

fn hand_memory() -> (Memory, Insight, Insight, Insight) {
    let run = Run::new(
        0,
        Task::new(0, 1),
        vec![
            StepRecord::new(0, 0, true, true),
            StepRecord::new(1, 1, true, true),
            StepRecord::new(2, 1, true, true),
        ],
    );
    let mut comp: HashMap<u64, String> = HashMap::new();
    comp.insert(0, "comp".to_string());
    let mut traj: HashMap<u64, Run> = HashMap::new();
    traj.insert(0, run);
    let mem = Memory::new(vec![], comp, traj);
    let valid = Insight::new("I-v".to_string(), 2, 1, Scope::Attr(1), 0, true, false);
    let overgen = Insight::new("I-o".to_string(), 2, 1, Scope::Any, 0, true, false);
    let corrupt = Insight::new("I-c".to_string(), 2, 0, Scope::Attr(1), 0, true, false);
    (mem, valid, overgen, corrupt)
}

#[test]
fn verify_lesson_four_outcomes() {
    let (mem, valid, overgen, corrupt) = hand_memory();
    assert_eq!(verify_lesson(&valid, 1, &mem), (Verdict::Supported, 2));
    assert_eq!(verify_lesson(&overgen, 2, &mem), (Verdict::Unsupported, 2));
    // Same-attribute over-generalization IS supported (evidence matches).
    assert_eq!(verify_lesson(&overgen, 1, &mem), (Verdict::Supported, 2));
    assert_eq!(verify_lesson(&corrupt, 1, &mem), (Verdict::Contradicted, 2));
    // Missing trajectory -> UNKNOWN.
    let ghost = Insight::new("I-g".to_string(), 2, 1, Scope::Attr(1), 99, true, false);
    assert_eq!(verify_lesson(&ghost, 1, &mem), (Verdict::Unknown, 2));
    // Unobserved step -> UNKNOWN.
    let unobs = Run::new(2, Task::new(2, 1), vec![StepRecord::new(2, 1, true, false)]);
    let mut comp: HashMap<u64, String> = HashMap::new();
    comp.insert(2, "comp".to_string());
    let mut traj: HashMap<u64, Run> = HashMap::new();
    traj.insert(2, unobs);
    let mem3 = Memory::new(vec![], comp, traj);
    assert_eq!(
        verify_lesson(&Insight::new("I-u".to_string(), 2, 1, Scope::Attr(1), 2, true, false), 1, &mem3),
        (Verdict::Unknown, 2)
    );
    // Failed-step claim -> CONTRADICTED.
    let run2 = Run::new(
        1,
        Task::new(1, 1),
        vec![
            StepRecord::new(0, 0, true, true),
            StepRecord::new(1, 1, true, true),
            StepRecord::new(2, 2, false, true),
        ],
    );
    let mut comp2: HashMap<u64, String> = HashMap::new();
    comp2.insert(1, "comp".to_string());
    let mut traj2: HashMap<u64, Run> = HashMap::new();
    traj2.insert(1, run2);
    let mem2 = Memory::new(vec![], comp2, traj2);
    assert_eq!(
        verify_lesson(&Insight::new("I-f".to_string(), 2, 2, Scope::Attr(1), 1, false, false), 1, &mem2),
        (Verdict::Contradicted, 2)
    );
}

#[test]
fn descent_triggers_cases() {
    let (_, valid, overgen, _) = hand_memory();
    assert!(!descent_triggers(&[&valid]));
    assert!(descent_triggers(&[&overgen])); // scope "any" -> missing detail
    let failed = Insight::new("I-f".to_string(), 2, 2, Scope::Attr(1), 1, false, false);
    assert!(descent_triggers(&[&failed])); // failed-run origin
    // Conflict: two applicable lessons claiming different actions.
    let other = Insight::new("I-x".to_string(), 2, 0, Scope::Attr(1), 0, true, false);
    assert!(descent_triggers(&[&valid, &other]));
    // Same claim twice, narrow, successful -> no descent.
    let dup = Insight::new("I-d".to_string(), 2, 1, Scope::Attr(1), 0, true, false);
    assert!(!descent_triggers(&[&valid, &dup]));
    // Empty -> no descent.
    assert!(!descent_triggers(&[]));
}

// ---------------------------------------------------------------- solve arms

#[test]
fn solve_arms_on_hand_memory() {
    let (_mem, valid, _overgen, corrupt) = hand_memory();
    // Corrupt lesson listed first: blind-follow takes the wrong action,
    // WikiTrace descends and follows the supported one.
    let mut comp: HashMap<u64, String> = HashMap::new();
    comp.insert(0, "comp".to_string());
    let mut traj: HashMap<u64, Run> = HashMap::new();
    traj.insert(
        0,
        Run::new(
            0,
            Task::new(0, 1),
            vec![
                StepRecord::new(0, 0, true, true),
                StepRecord::new(1, 1, true, true),
                StepRecord::new(2, 1, true, true),
            ],
        ),
    );
    let mem4 = Memory::new(vec![corrupt, valid], comp, traj);
    let (ok, stats) = solve(&Task::new(9, 1), Some(&mem4), Mode::WikiTrace, SEED);
    // Python gives exactly this: the verify step is fixed by the
    // supported lesson, but the default operate choice (step 1) is
    // wrong, so the task still fails — A6f asserts only the descent
    // and the correction, not success.
    assert_eq!((stats.descents, stats.corrections), (1, 1));
    assert_eq!(stats.evidence_reads, 4); // corrupt (2 reads) + valid (2 reads)
    assert_eq!(stats.lessons_followed, 1);
    assert!(!ok);
    // No memory: the default choices for task 9 attr 1 decide it.
    let (ok_none, stats_none) = solve(&Task::new(9, 1), None, Mode::None, SEED);
    assert_eq!(stats_none, SolveStats::default());
    let _ = ok_none;
}

#[test]
fn solve_none_never_follows_lessons() {
    let mem = main_memory();
    for t in gen_tasks(SEED + 5000, 50) {
        let (_, stats) = solve(&t, Some(&mem), Mode::None, SEED);
        assert_eq!(stats, SolveStats::default());
    }
}

// ---------------------------------------------------------------- experiment

#[test]
fn run_experiment_matches_python_exactly() {
    let res = run_experiment(SEED, N_TRAIN, N_TEST);
    assert_eq!(res.none.success, 0.2);
    assert_eq!(res.plain.success, 0.505);
    assert_eq!(res.wikitrace.arm.success, 0.7033333333333334);
    assert_eq!(res.wikitrace.arm.gain, 0.5033333333333333);
    assert_eq!(res.plain.gain, 0.4066666666666667);
    assert_eq!(res.wikitrace.arm.regression, 0.0);
    assert_eq!(res.plain.regression, 0.10166666666666667);
    assert_eq!(res.wikitrace.descent_rate, 1.0);
    assert_eq!(res.wikitrace.correction_rate, 0.2638888888888889);
    assert_eq!(res.wikitrace.evidence_reads_per_task, 23.826666666666668);
    assert_eq!(res.wikitrace_succ_only.success, 0.5833333333333334);
    assert_eq!(res.wikitrace_fail_only.success, 0.3566666666666667);
    assert_eq!(res.wikitrace.lessons_followed_per_task, 2.243333333333333);
    assert_eq!(res.plain_lessons_followed_per_task, 3.0);
    assert_eq!(res.n_insights, 36);
    assert_eq!(res.n_insights_corrupted, 4);
    assert_eq!(res.n_insights_from_failed_runs, 24);
}
