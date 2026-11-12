//! Deep tests for the WikiTrace Rust port: property sweeps over seeds
//! and budgets, exhaustive + random checks against oracles, a soak at a
//! scale the Python test does not reach, and thread-identity.

use wikitrace::*;

// ---------------------------------------------------------------- property sweeps

/// Theorems that must hold for EVERY seed (not just the main one):
///  * WikiTrace never regresses (regression == 0 exactly) — a WikiTrace
///    step takes either the gold action or the default choice, so a
///    baseline success can never turn into a failure.
///  * WikiTrace success >= No-Memory success (same per-step argument).
///  * WikiTrace gain >= 0 and regression_of_solved == 0.
#[test]
fn property_sweep_wikitrace_never_regresses() {
    for seed in 0..200u64 {
        let res = run_experiment(seed, 12, 100);
        assert_eq!(res.wikitrace.arm.regression, 0.0, "seed {}", seed);
        assert_eq!(res.wikitrace.arm.regression_of_solved, 0.0, "seed {}", seed);
        assert!(
            res.wikitrace.arm.success >= res.none.success,
            "seed {}: wt {} < none {}",
            seed,
            res.wikitrace.arm.success,
            res.none.success
        );
        assert!(res.wikitrace.arm.gain >= 0.0, "seed {}", seed);
        assert!(res.wikitrace_succ_only.regression == 0.0, "seed {}", seed);
        assert!(res.wikitrace_fail_only.regression == 0.0, "seed {}", seed);
    }
}

#[test]
fn property_sweep_plain_can_regress_wikitrace_cannot() {
    // Over the sweep, plain lessons regress on at least one seed (the
    // paper's memory-induced degradation), while WikiTrace regresses on
    // none — the mechanism, not the seed, is doing the work.
    let mut plain_regressed = 0;
    for seed in 0..200u64 {
        let res = run_experiment(seed, 12, 100);
        if res.plain.regression > 0.0 {
            plain_regressed += 1;
        }
    }
    assert!(plain_regressed > 0, "plain never regressed over 200 seeds");
}

#[test]
fn property_sweep_vary_train_size_and_budgets() {
    // Vary n_train (0..40) and n_test (1..200): the theorems hold at
    // every scale, including the degenerate n_train=0 (empty memory).
    for n_train in [0usize, 1, 2, 5, 12, 24, 40] {
        for n_test in [1usize, 7, 50, 200] {
            let res = run_experiment(20261112, n_train, n_test);
            assert_eq!(res.wikitrace.arm.regression, 0.0, "n_train={} n_test={}", n_train, n_test);
            assert!(
                res.wikitrace.arm.success >= res.none.success,
                "n_train={} n_test={}",
                n_train,
                n_test
            );
            if n_train == 0 {
                assert_eq!(res.n_insights, 0);
                // Empty memory: all three arms collapse to the default.
                assert_eq!(res.plain.success, res.none.success);
                assert_eq!(res.wikitrace.arm.success, res.none.success);
            }
        }
    }
}

// ---------------------------------------------------------------- soundness oracle

/// Soundness property, checked exhaustively over random memories:
/// every SUPPORTED verdict implies claimed == gold for the task —
/// for lessons *applicable* to the task, which is how `solve` calls
/// `verify_lesson` (Python's `applicable` filter). A narrow lesson is
/// only ever verified against its own scope attribute there; calling
/// `verify_lesson` directly on a non-applicable lesson is outside the
/// contract (its narrow scope is not re-checked inside the function,
/// in Python or here — the scope is set to the source run's attribute
/// at compile time, so an applicable narrow lesson always has
/// run attr == task attr).
#[test]
fn soundness_supported_implies_gold() {
    for seed in 0..100u64 {
        let train = gen_tasks(seed, 12);
        let runs: Vec<Run> = train.iter().map(|t| run_default(t, seed, seed + 7)).collect();
        let mem = compile_memory(runs, seed + 11);
        for ins in &mem.insights {
            for attr in 0..4u64 {
                if !ins.applies_to(attr) {
                    continue;
                }
                let (v, _) = verify_lesson(ins, attr, &mem);
                if v == Verdict::Supported {
                    assert_eq!(
                        ins.claimed,
                        gold_action(ins.step_type, attr),
                        "seed {} insight {} step {} claimed {} attr {}",
                        seed,
                        ins.iid,
                        ins.step_type,
                        ins.claimed,
                        attr
                    );
                }
            }
        }
    }
}

/// Brute-force oracle for gold_action: for every step type and attribute
/// 0..=255, gold matches the defining table (locate 0, operate 1,
/// verify attr%3).
#[test]
fn gold_action_exhaustive_oracle() {
    for st in 0..3u8 {
        for attr in 0..=255u64 {
            let expected = match st {
                0 => 0,
                1 => 1,
                _ => attr % 3,
            };
            assert_eq!(gold_action(st, attr), expected, "st={} attr={}", st, attr);
        }
    }
}

/// Exhaustive verify_lesson outcome table on a hand-built memory:
/// for every (step_type, claimed, scope, task_attr) combination against
/// a fully-observed successful run, the verdict matches the oracle
/// computed from the definition.
#[test]
fn verify_lesson_exhaustive_oracle() {
    use std::collections::HashMap;
    // Source run on attr=1: steps chose [0, 1, 1], all ok, all observed.
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
    let taken = [0u64, 1, 1]; // action actually taken per step type
    for st in 0..3u8 {
        for claimed in 0..3u64 {
            for scope in [Scope::Any, Scope::Attr(1), Scope::Attr(2)] {
                for task_attr in 0..4u64 {
                    let ins = Insight::new(
                        "I-x".to_string(),
                        st,
                        claimed,
                        scope.clone(),
                        0,
                        true,
                        false,
                    );
                    let (v, reads) = verify_lesson(&ins, task_attr, &mem);
                    assert_eq!(reads, 2);
                    let expected = if claimed != taken[st as usize] {
                        Verdict::Contradicted
                    } else if scope == Scope::Any && task_attr != 1 {
                        Verdict::Unsupported
                    } else {
                        Verdict::Supported
                    };
                    assert_eq!(v, expected, "st={} claimed={} scope={:?} attr={}", st, claimed, scope, task_attr);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- soak

#[test]
fn soak_5000_test_tasks() {
    // 5,000 test tasks (8x the Python test) + 50 train runs: the
    // mechanism stays sound at scale; WikiTrace still never regresses.
    let res = run_experiment(20261112, 50, 5000);
    assert_eq!(res.wikitrace.arm.regression, 0.0);
    assert!(res.wikitrace.arm.success > res.plain.success);
    assert!(res.wikitrace.arm.success > res.none.success);
    assert!(res.plain.regression > 0.0);
    assert_eq!(res.n_insights, 150);
    println!(
        "soak 5000: none={:.4} plain={:.4} wt={:.4} descent={:.3} correction={:.3} reads/task={:.2}",
        res.none.success,
        res.plain.success,
        res.wikitrace.arm.success,
        res.wikitrace.descent_rate,
        res.wikitrace.correction_rate,
        res.wikitrace.evidence_reads_per_task
    );
}

#[test]
fn soak_200_seed_sweep_counts() {
    // 200 seeds x full experiment at n_test=600: count how often each
    // ordering holds. WikiTrace beats plain on the overwhelming majority
    // (it is not a one-seed artifact); it never loses to no-memory.
    let mut wt_beats_plain = 0;
    let mut wt_ties_plain = 0;
    let mut plain_beats_none = 0;
    for seed in 0..200u64 {
        let res = run_experiment(seed, 12, 600);
        if res.wikitrace.arm.success > res.plain.success {
            wt_beats_plain += 1;
        } else if res.wikitrace.arm.success == res.plain.success {
            wt_ties_plain += 1;
        }
        if res.plain.success > res.none.success {
            plain_beats_none += 1;
        }
        assert!(res.wikitrace.arm.success >= res.none.success, "seed {}", seed);
    }
    println!(
        "200-seed sweep: WT>plain {}/200, WT==plain {}, plain>none {}/200",
        wt_beats_plain, wt_ties_plain, plain_beats_none
    );
    assert!(wt_beats_plain >= 180, "WT beats plain on only {}/200 seeds", wt_beats_plain);
    // plain>none is seed-dependent at n_train=12 (blind-following a
    // small, partly-corrupted memory often hurts); it is reported, not
    // asserted as a theorem. The main-seed A3 (plain > none) is covered
    // by the verify binary and the exact-numbers unit test.
    assert!(plain_beats_none >= 50, "plain beats none on only {}/200 seeds", plain_beats_none);
}

// ---------------------------------------------------------------- thread identity

#[test]
fn thread_identity_stateless_solve() {
    // The solver is stateless across tasks: the same 200-task batch
    // solved on 4 threads gives byte-identical results to the
    // single-threaded run.
    let train = gen_tasks(SEED, N_TRAIN);
    let runs: Vec<Run> = train.iter().map(|t| run_default(t, SEED, SEED + 7)).collect();
    let mem = std::sync::Arc::new(compile_memory(runs, SEED + 11));
    let tasks = gen_tasks(SEED + 5000, 200);

    let single: Vec<(bool, SolveStats)> = tasks
        .iter()
        .map(|t| solve(t, Some(&mem), Mode::WikiTrace, SEED))
        .collect();

    let chunk = tasks.len() / 4;
    let mut handles = Vec::new();
    for c in 0..4 {
        let mem_c = std::sync::Arc::clone(&mem);
        let chunk_tasks: Vec<Task> = tasks[c * chunk..(c + 1) * chunk].to_vec();
        handles.push(std::thread::spawn(move || {
            chunk_tasks
                .iter()
                .map(|t| solve(t, Some(&mem_c), Mode::WikiTrace, SEED))
                .collect::<Vec<_>>()
        }));
    }
    let mut threaded: Vec<(bool, SolveStats)> = Vec::new();
    for h in handles {
        threaded.extend(h.join().unwrap());
    }
    assert_eq!(single, threaded);
}
