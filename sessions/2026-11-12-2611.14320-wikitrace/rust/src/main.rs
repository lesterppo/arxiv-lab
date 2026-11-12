//! `verify` binary: mirrors the Python CPU test's experiments and
//! assertions (test_wikitrace_cpu.py, A1–A8). Prints the arm summaries,
//! runs A1–A8, writes results.json next to the crate dir, exits 0 iff
//! all assertions pass. Usage: cargo run --release --bin verify

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::exit;
use wikitrace::*;

const SEED: u64 = 20261112;
const N_TRAIN: usize = 12;
const N_TEST: usize = 600;

fn check(name: &str, cond: bool, detail: &str, ok: &mut bool) {
    println!("  [{}] {} {}", if cond { "PASS" } else { "FAIL" }, name, detail);
    *ok &= cond;
}

fn fmt(x: f64) -> String {
    format!("{:.3}", x)
}

/// Format an f64 the way Python's json.dumps does (shortest repr that
/// round-trips). Rust's Display for f64 does the same for these values.
fn json_f64(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e16 {
        // Python prints integral floats as e.g. 0.0 / 1.0
        format!("{:.1}", x)
    } else {
        format!("{}", x)
    }
}

fn main() {
    let mut ok = true;
    let res = run_experiment(SEED, N_TRAIN, N_TEST);

    let wt = res.wikitrace.arm;
    let pl = res.plain;
    let nm = res.none;

    println!("ARM SUCCESS RATES (n={} test tasks, {} train runs):", N_TEST, N_TRAIN);
    println!("  {:12} success={} gain={} regression={}", "none", fmt(nm.success), fmt(nm.gain), fmt(nm.regression));
    println!("  {:12} success={} gain={} regression={} lessons/task={}", "plain", fmt(pl.success), fmt(pl.gain), fmt(pl.regression), fmt(res.plain_lessons_followed_per_task));
    println!(
        "  {:12} success={} gain={} regression={} descent={} correction={} reads/task={} lessons/task={}",
        "wikitrace",
        fmt(wt.success),
        fmt(wt.gain),
        fmt(wt.regression),
        fmt(res.wikitrace.descent_rate),
        fmt(res.wikitrace.correction_rate),
        fmt(res.wikitrace.evidence_reads_per_task),
        fmt(res.wikitrace.lessons_followed_per_task)
    );
    println!(
        "  wikitrace_succ_only success={} (n_insights={}) | wikitrace_fail_only success={} (n_insights={})",
        fmt(res.wikitrace_succ_only.success),
        res.wikitrace_succ_only.n_insights,
        fmt(res.wikitrace_fail_only.success),
        res.wikitrace_fail_only.n_insights
    );
    println!(
        "  n_insights={} corrupted={} from_failed_runs={}",
        res.n_insights, res.n_insights_corrupted, res.n_insights_from_failed_runs
    );

    check(
        "A1 WT > plain by >=0.10",
        wt.success - pl.success >= 0.10,
        &format!("{} vs {}", fmt(wt.success), fmt(pl.success)),
        &mut ok,
    );
    check(
        "A2 WT > none by >=0.10",
        wt.success - nm.success >= 0.10,
        &format!("{} vs {}", fmt(wt.success), fmt(nm.success)),
        &mut ok,
    );
    check(
        "A3 plain > none",
        pl.success > nm.success,
        &format!("{} vs {}", fmt(pl.success), fmt(nm.success)),
        &mut ok,
    );
    check(
        "A4a WT regression == 0",
        wt.regression == 0.0,
        &format!("{:.4}", wt.regression),
        &mut ok,
    );
    check(
        "A4b plain regression > 0.05",
        pl.regression > 0.05,
        &format!("{:.4}", pl.regression),
        &mut ok,
    );
    check(
        "A5 WT gain > plain gain",
        wt.gain > pl.gain,
        &format!("{} vs {}", fmt(wt.gain), fmt(pl.gain)),
        &mut ok,
    );
    check(
        "A7a descent rate > 0.2",
        res.wikitrace.descent_rate > 0.2,
        &format!("{}", fmt(res.wikitrace.descent_rate)),
        &mut ok,
    );
    check(
        "A7b corrections > 0",
        res.wikitrace.correction_rate > 0.0,
        &format!("{}", fmt(res.wikitrace.correction_rate)),
        &mut ok,
    );

    // A6: evidence-check unit cases on a hand-built memory.
    let task = Task::new(0, 1); // attr=1 -> verify gold = 1
    // Source run: locate chose 0 (ok), operate chose 1 (ok), verify chose 1 (ok).
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
    traj.insert(0, run.clone());
    let mem = Memory::new(vec![], comp.clone(), traj.clone());
    let valid = Insight::new("I-v".to_string(), 2, 1, Scope::Attr(1), 0, true, false);
    check(
        "A6a valid narrow -> SUPPORTED",
        verify_lesson(&valid, 1, &mem).0 == Verdict::Supported,
        "",
        &mut ok,
    );
    let overgen = Insight::new("I-o".to_string(), 2, 1, Scope::Any, 0, true, false);
    check(
        "A6b over-generalized -> UNSUPPORTED",
        verify_lesson(&overgen, 2, &mem).0 == Verdict::Unsupported,
        "",
        &mut ok,
    );
    let corrupt = Insight::new("I-c".to_string(), 2, 0, Scope::Attr(1), 0, true, false);
    check(
        "A6c corrupted claim -> CONTRADICTED",
        verify_lesson(&corrupt, 1, &mem).0 == Verdict::Contradicted,
        "",
        &mut ok,
    );
    // Failed-run step: verify chose 2, failed.
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
    let failed_claim = Insight::new("I-f".to_string(), 2, 2, Scope::Attr(1), 1, false, false);
    check(
        "A6d failed-step claim -> CONTRADICTED",
        verify_lesson(&failed_claim, 1, &mem2).0 == Verdict::Contradicted,
        "",
        &mut ok,
    );
    let unobs = Run::new(2, Task::new(2, 1), vec![StepRecord::new(2, 1, true, false)]);
    let mut comp3: HashMap<u64, String> = HashMap::new();
    comp3.insert(2, "comp".to_string());
    let mut traj3: HashMap<u64, Run> = HashMap::new();
    traj3.insert(2, unobs);
    let mem3 = Memory::new(vec![], comp3, traj3);
    check(
        "A6e unobserved step -> UNKNOWN",
        verify_lesson(&Insight::new("I-u".to_string(), 2, 1, Scope::Attr(1), 2, true, false), 1, &mem3).0
            == Verdict::Unknown,
        "",
        &mut ok,
    );
    // Conflict resolves to the supported lesson (corrupt lesson listed
    // first, so blind-follow would take the wrong action).
    let mem4 = Memory::new(vec![corrupt.clone(), valid.clone()], comp, traj);
    let (_ok9, stats) = solve(&Task::new(9, 1), Some(&mem4), Mode::WikiTrace, SEED);
    check(
        "A6f conflict -> supported lesson followed, blind action corrected",
        stats.descents == 1 && stats.corrections == 1,
        &format!("descents={} corrections={}", stats.descents, stats.corrections),
        &mut ok,
    );
    check(
        "A6g failed-run lesson triggers descent",
        descent_triggers(&[&failed_claim]),
        "",
        &mut ok,
    );
    check(
        "A6h single valid narrow lesson does not trigger descent",
        !descent_triggers(&[&valid]),
        "",
        &mut ok,
    );
    let _ = task;

    // A8: determinism.
    let res2 = run_experiment(SEED, N_TRAIN, N_TEST);
    let same = res.none.success == res2.none.success
        && res.plain.success == res2.plain.success
        && res.wikitrace.arm.success == res2.wikitrace.arm.success
        && res.wikitrace.arm.gain == res2.wikitrace.arm.gain
        && res.wikitrace.arm.regression == res2.wikitrace.arm.regression
        && res.wikitrace.descent_rate == res2.wikitrace.descent_rate
        && res.wikitrace.correction_rate == res2.wikitrace.correction_rate
        && res.wikitrace.evidence_reads_per_task == res2.wikitrace.evidence_reads_per_task
        && res.wikitrace.lessons_followed_per_task == res2.wikitrace.lessons_followed_per_task
        && res.plain_lessons_followed_per_task == res2.plain_lessons_followed_per_task
        && res.wikitrace_succ_only.success == res2.wikitrace_succ_only.success
        && res.wikitrace_fail_only.success == res2.wikitrace_fail_only.success
        && res.n_insights == res2.n_insights
        && res.n_insights_corrupted == res2.n_insights_corrupted
        && res.n_insights_from_failed_runs == res2.n_insights_from_failed_runs;
    check("A8 determinism", same, "", &mut ok);

    // results.json (hand-rolled; no serde in this crate). Same key order
    // and layout as Python's json.dumps(res, indent=2, sort_keys=True).
    let out = format!(
        "{{\n  \"n_insights\": {},\n  \"n_insights_corrupted\": {},\n  \"n_insights_from_failed_runs\": {},\n  \"n_test_tasks\": {},\n  \"n_train_runs\": {},\n  \"none\": {{\n    \"gain\": {},\n    \"regression\": {},\n    \"regression_of_solved\": {},\n    \"success\": {}\n  }},\n  \"plain\": {{\n    \"gain\": {},\n    \"lessons_followed_per_task\": {},\n    \"regression\": {},\n    \"regression_of_solved\": {},\n    \"success\": {}\n  }},\n  \"wikitrace\": {{\n    \"correction_rate\": {},\n    \"descent_rate\": {},\n    \"evidence_reads_per_task\": {},\n    \"gain\": {},\n    \"lessons_followed_per_task\": {},\n    \"regression\": {},\n    \"regression_of_solved\": {},\n    \"success\": {}\n  }},\n  \"wikitrace_fail_only\": {{\n    \"gain\": {},\n    \"n_insights\": {},\n    \"regression\": {},\n    \"success\": {}\n  }},\n  \"wikitrace_succ_only\": {{\n    \"gain\": {},\n    \"n_insights\": {},\n    \"regression\": {},\n    \"success\": {}\n  }}\n}}",
        res.n_insights,
        res.n_insights_corrupted,
        res.n_insights_from_failed_runs,
        res.n_test_tasks,
        res.n_train_runs,
        json_f64(res.none.gain),
        json_f64(res.none.regression),
        json_f64(res.none.regression_of_solved),
        json_f64(res.none.success),
        json_f64(res.plain.gain),
        json_f64(res.plain_lessons_followed_per_task),
        json_f64(res.plain.regression),
        json_f64(res.plain.regression_of_solved),
        json_f64(res.plain.success),
        json_f64(res.wikitrace.correction_rate),
        json_f64(res.wikitrace.descent_rate),
        json_f64(res.wikitrace.evidence_reads_per_task),
        json_f64(res.wikitrace.arm.gain),
        json_f64(res.wikitrace.lessons_followed_per_task),
        json_f64(res.wikitrace.arm.regression),
        json_f64(res.wikitrace.arm.regression_of_solved),
        json_f64(res.wikitrace.arm.success),
        json_f64(res.wikitrace_fail_only.gain),
        res.wikitrace_fail_only.n_insights,
        json_f64(res.wikitrace_fail_only.regression),
        json_f64(res.wikitrace_fail_only.success),
        json_f64(res.wikitrace_succ_only.gain),
        res.wikitrace_succ_only.n_insights,
        json_f64(res.wikitrace_succ_only.regression),
        json_f64(res.wikitrace_succ_only.success),
    );
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results.json");
    fs::write(&path, out).expect("write results.json");
    println!("results.json written to {}", path.display());

    if !ok {
        println!("ASSERTIONS FAILED");
        exit(1);
    }
    println!("ALL ASSERTIONS PASSED");
}
