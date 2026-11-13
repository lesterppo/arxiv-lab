//! `verify` binary: mirrors the Python CPU test's experiments and
//! assertions (test_termination_lag_cpu.py, A1–A9). Prints the policy
//! summaries, runs A1–A9, writes results.json next to the crate dir,
//! exits 0 iff all assertions pass. Usage: cargo run --release --bin verify

use std::fs;
use std::process::exit;
use termination_lag::*;

const SEED: u64 = 20261113;
const N_TASKS: usize = 600;

fn check(name: &'static str, cond: bool, detail: &str, passed: &mut Vec<&'static str>, failed: &mut Vec<&'static str>) {
    println!("  [{}] {} {}", if cond { "PASS" } else { "FAIL" }, name, detail);
    if cond {
        passed.push(name);
    } else {
        failed.push(name);
    }
}

fn fmt(x: f64) -> String {
    format!("{:.3}", x)
}

/// Format an f64 the way Python's json.dumps does for these values
/// (shortest repr that round-trips; Rust's Display matches it here).
fn json_f64(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e16 {
        format!("{:.1}", x)
    } else {
        format!("{}", x)
    }
}

fn metrics_json(m: &ExperimentMetrics) -> String {
    format!(
        concat!(
            "{{\"policy\": \"{}\", \"n_tasks\": {}, \"n_solved\": {}, ",
            "\"success_rate\": {}, \"first_edit_solve_share\": {}, ",
            "\"mean_lag\": {}, \"median_lag\": {}, \"median_post_share\": {}, ",
            "\"mean_post_share\": {}, \"final_correct_rate\": {}, ",
            "\"undo_to_incorrect_rate\": {}, \"mean_undos\": {}, ",
            "\"mean_total_steps\": {}, \"post_verify_share\": {}, ",
            "\"post_explore_share\": {}, \"post_reedit_share\": {}}}"
        ),
        m.policy.as_str(),
        m.n_tasks,
        m.n_solved,
        json_f64(m.success_rate),
        json_f64(m.first_edit_solve_share),
        json_f64(m.mean_lag),
        json_f64(m.median_lag),
        json_f64(m.median_post_share),
        json_f64(m.mean_post_share),
        json_f64(m.final_correct_rate),
        json_f64(m.undo_to_incorrect_rate),
        json_f64(m.mean_undos),
        json_f64(m.mean_total_steps),
        json_f64(m.post_verify_share),
        json_f64(m.post_explore_share),
        json_f64(m.post_reedit_share),
    )
}

fn main() {
    let mut passed: Vec<&'static str> = Vec::new();
    let mut failed: Vec<&'static str> = Vec::new();
    let tasks = gen_tasks(N_TASKS, SEED);
    let results = compare_policies(&tasks, SEED, 3.0, P_UNDO, 10, 1.0);
    let cont = &results[0];
    let stop = &results[1];
    let fixed = &results[2];

    println!(
        "main experiment ({} tasks, seed {}):",
        N_TASKS, SEED
    );
    for m in &results {
        println!(
            "  {:13} solved={:3} first_edit={} mean_lag={:.2} med_lag={:.1} med_post={} final_ok={} undo_bad={} steps={:.2} verify={:.2} explore={:.2} reedit={:.2}",
            m.policy.as_str(),
            m.n_solved,
            fmt(m.first_edit_solve_share),
            m.mean_lag,
            m.median_lag,
            fmt(m.median_post_share),
            fmt(m.final_correct_rate),
            fmt(m.undo_to_incorrect_rate),
            m.mean_total_steps,
            m.post_verify_share,
            m.post_explore_share,
            m.post_reedit_share,
        );
    }

    println!("assertions:");
    check(
        "A1 first-edit solve share in [0.30, 0.85]",
        (0.30..=0.85).contains(&cont.first_edit_solve_share),
        &format!("got {}", fmt(cont.first_edit_solve_share)),
        &mut passed, &mut failed,
    );
    check(
        "A2 continue median post-solution share > 0.30",
        cont.median_post_share > 0.30,
        &format!("got {}", fmt(cont.median_post_share)),
        &mut passed, &mut failed,
    );
    check(
        "A3 continue mean lag > 2 steps",
        cont.mean_lag > 2.0,
        &format!("got {:.2}", cont.mean_lag),
        &mut passed, &mut failed,
    );
    check(
        "A4a stop-on-pass median lag == 1",
        stop.median_lag == 1.0 && stop.mean_lag == 1.0,
        &format!("got mean {:.2} median {:.1}", stop.mean_lag, stop.median_lag),
        &mut passed, &mut failed,
    );
    check(
        "A4b stop-on-pass final-correct >= continue final-correct",
        stop.final_correct_rate >= cont.final_correct_rate,
        &format!(
            "{} vs {}",
            fmt(stop.final_correct_rate),
            fmt(cont.final_correct_rate)
        ),
        &mut passed, &mut failed,
    );
    check(
        "A5a continue undo-to-incorrect in (0, 0.15]",
        cont.undo_to_incorrect_rate > 0.0 && cont.undo_to_incorrect_rate <= 0.15,
        &format!("got {}", fmt(cont.undo_to_incorrect_rate)),
        &mut passed, &mut failed,
    );
    check(
        "A5b stop-on-pass undo-to-incorrect == 0",
        stop.undo_to_incorrect_rate == 0.0,
        &format!("got {}", fmt(stop.undo_to_incorrect_rate)),
        &mut passed, &mut failed,
    );
    check(
        "A6 verify+explore dominate post-solution steps (> 0.5)",
        cont.post_verify_share + cont.post_explore_share > 0.5,
        &format!(
            "got {:.3}",
            cont.post_verify_share + cont.post_explore_share
        ),
        &mut passed, &mut failed,
    );
    // A7 unit cases
    check(
        "A7a verifier: correct patch passes, wrong fails",
        verify(true) && !verify(false),
        "",
        &mut passed, &mut failed,
    );
    let mut rng = Rng::new(7);
    let r = run_task(
        Task { idx: 0, difficulty: Difficulty::Easy },
        Policy::Continue,
        &mut rng,
        6.0,
        1.0,
        10,
        1.0,
    );
    check(
        "A7b p_undo=1 forces undos whenever re-edits occur",
        r.solved && r.n_undos >= 1,
        &format!("got {} undos", r.n_undos),
        &mut passed, &mut failed,
    );
    let mut rng2 = Rng::new(7);
    let r2 = run_task(
        Task { idx: 0, difficulty: Difficulty::Easy },
        Policy::StopOnPass,
        &mut rng2,
        3.0,
        P_UNDO,
        10,
        1.0,
    );
    check(
        "A7c stop-on-pass never re-edits after solving",
        r2.solved && r2.n_reedit == 0 && r2.n_verify == 1,
        "",
        &mut passed, &mut failed,
    );
    let mut rng3 = Rng::new(7);
    let r3 = run_task(
        Task { idx: 0, difficulty: Difficulty::Easy },
        Policy::FixedBudget,
        &mut rng3,
        3.0,
        P_UNDO,
        10,
        1.0,
    );
    check(
        "A7d fixed-budget solved run ends at exactly 10 steps",
        !r3.solved || r3.steps == 10,
        &format!("got {}", r3.steps),
        &mut passed, &mut failed,
    );
    // A8 determinism
    let m2 = compare_policies(&tasks, SEED, 3.0, P_UNDO, 10, 1.0);
    let same = (0..3).all(|i| {
        let a = &results[i];
        let b = &m2[i];
        a.n_solved == b.n_solved
            && a.mean_lag == b.mean_lag
            && a.median_lag == b.median_lag
            && a.median_post_share == b.median_post_share
            && a.final_correct_rate == b.final_correct_rate
            && a.undo_to_incorrect_rate == b.undo_to_incorrect_rate
    });
    check("A8 same seed -> identical metrics", same, "", &mut passed, &mut failed);
    // A9 sensitivity sweep over mean_extra
    let mut lags = Vec::new();
    let mut finals = Vec::new();
    for me in [1.0, 3.0, 6.0] {
        let m = run_experiment(Policy::Continue, &tasks, SEED, me, P_UNDO, 10, 1.0);
        lags.push(m.mean_lag);
        finals.push(m.final_correct_rate);
    }
    println!(
        "  sensitivity mean_extra 1/3/6: mean_lag=[{:.2}, {:.2}, {:.2}] final_ok=[{:.3}, {:.3}, {:.3}]",
        lags[0], lags[1], lags[2], finals[0], finals[1], finals[2]
    );
    check(
        "A9a mean lag grows with mean_extra",
        lags[0] < lags[1] && lags[1] < lags[2],
        &format!("got [{:.2}, {:.2}, {:.2}]", lags[0], lags[1], lags[2]),
        &mut passed, &mut failed,
    );
    check(
        "A9b more post-solution steps do not improve final correctness",
        finals[2] <= finals[0] + 1e-9,
        &format!("got [{:.3}, {:.3}, {:.3}]", finals[0], finals[1], finals[2]),
        &mut passed, &mut failed,
    );

    let out = format!(
        concat!(
            "{{\n  \"paper\": \"2611.14960\",\n  \"seed\": {},\n  \"n_tasks\": {},\n",
            "  \"policies\": {{\"continue\": {}, \"stop_on_pass\": {}, \"fixed_budget\": {}}},\n",
            "  \"sensitivity_mean_extra\": {{\"1.0\": {}, \"3.0\": {}, \"6.0\": {}}},\n",
            "  \"assertions_passed\": [{}],\n  \"assertions_failed\": [{}]\n}}\n"
        ),
        SEED,
        N_TASKS,
        metrics_json(cont),
        metrics_json(stop),
        metrics_json(fixed),
        json_f64(lags[0]),
        json_f64(lags[1]),
        json_f64(lags[2]),
        passed.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(", "),
        failed.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(", "),
    );
    let out_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results.json");
    fs::write(&out_path, &out).expect("write results.json");
    println!("\nresults.json written to {}", out_path.display());
    println!("{} passed, {} failed", passed.len(), failed.len());
    // keep the fixed-budget metrics referenced (avoids an unused warning)
    let _ = fixed.mean_total_steps;
    if !failed.is_empty() {
        exit(1);
    }
}
