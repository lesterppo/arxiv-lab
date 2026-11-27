//! `verify` binary: mirrors the Python CPU test's experiments and
//! assertions (tests/test_partial_routing_cpu.py), on Rust-native
//! worlds.
//!
//! Worlds are generated with the crate's SplitMix64 RNG, not Python's
//! `random.Random`, so the *numbers* differ from Python's while the
//! *properties* asserted are the same: partial routing beats
//! lead-alone and task-level routing at lower cost, cost strictly
//! decreases with delta, success peaks at an interior delta,
//! breakdown-only helps, the E4 transfer result is recorded honestly,
//! partial routing reaches most of the oracle, determinism. Exact
//! cross-language equality is the parity sweep's job
//! (tests/parity_cases.txt).
//!
//! Two deliberate, documented differences from the Python test:
//! * Past-run corpus for guidance: 20000 tasks instead of 4000, so
//!   the measured gaps sit closer to their true values (the
//!   delta=0.75 threshold sits only ~0.01 above the measured code
//!   gap at 4000 traces; at 20000 the noise is ~2x smaller). The
//!   delegation sets asserted are the same as Python's.
//! * The PYTHONHASHSEED determinism check has no Rust analogue:
//!   the computation uses fixed arrays indexed by subtask type and
//!   no hash maps anywhere, so there is no hash-order dependence to
//!   vary; repeated runs with the same seed are asserted identical
//!   instead.
//!
//! Usage: `cargo run --release --bin verify`
//! Writes `results.json` next to the crate root on success.

use partial_routing::{
    analytic_expected, delegate, execute, fixed_easy_assignment, guidance_from_gaps,
    learn_guidance, make_tasks, oracle_assignment, past_traces, random_assignment,
    run_policy, task_level_assignment, Subtask, Task, LEAD_COST, SUBTASK_TYPES,
    WORKER_COST,
};
use std::time::Instant;

const PAST_SEED: u64 = 11;
const N_PAST: usize = 20000;
const HELDOUT_SEED: u64 = 101;
const HELDOUT_B_SEED: u64 = 202;
const ORACLE_SEED: u64 = 303;
const N_TASKS: usize = 60;
const N_ORACLE: usize = 12;
const TRIALS: usize = 400;
const MC_SEED: u64 = 5;

fn main() {
    let t0 = Instant::now();
    let mut fails: Vec<String> = Vec::new();
    let check = |name: &str, cond: bool, detail: String, fails: &mut Vec<String>| {
        println!("{}{}{}", if cond { "ok " } else { "FAIL " }, name, if detail.is_empty() { String::new() } else { format!(" [{detail}]") });
        if !cond {
            fails.push(name.to_string());
        }
    };

    // ------------------------------------------------------- guidance (units)
    let traces = past_traces(PAST_SEED, N_PAST);
    let g = learn_guidance(&traces);
    println!(
        "measured gaps: {{{}}} gmax={:.3} n_traces={}",
        SUBTASK_TYPES
            .iter()
            .enumerate()
            .map(|(t, name)| format!("{name}: {:.3}", g.gap[t]))
            .collect::<Vec<_>>()
            .join(", "),
        g.gap_max,
        g.n_traces
    );
    check(
        "guidance gap order is plan > code > verify > test > summarize",
        g.gap[0] > g.gap[1] && g.gap[1] > g.gap[2] && g.gap[2] > g.gap[3] && g.gap[3] > g.gap[4],
        format!("{:?}", g.gap.map(|x| (x * 1000.0).round() / 1000.0)),
        &mut fails,
    );
    check(
        "measured costs are lead 10 / worker 1 per subtask",
        g.lead_cost.iter().all(|&c| c == LEAD_COST) && g.worker_cost.iter().all(|&c| c == WORKER_COST),
        String::new(),
        &mut fails,
    );
    let expected_sets: [(f64, Vec<usize>); 5] = [
        (0.0, vec![]),
        (0.25, vec![4]),
        (0.5, vec![3, 4]),
        (0.75, vec![2, 3, 4]),
        (1.0, vec![0, 1, 2, 3, 4]),
    ];
    for (delta, want) in expected_sets {
        let got = g.delegate_types(delta);
        let name = format!("delta={delta} delegates {want:?}");
        check(&name, got == want, format!("got {got:?}"), &mut fails);
    }

    // hand-built task: 5 summarize subtasks, difficulty 0. Lead
    // success declines .95, .89, .83, .77, .71 (load penalty 0.06);
    // worker is a flat .89.
    let hand = Task {
        subtasks: vec![Subtask { type_idx: 4, difficulty: 0.0 }; 5],
    };
    let lead_assign = vec![false; 5];
    let work_assign = vec![true; 5];
    let (p_lead, c_lead) = analytic_expected(&hand, &lead_assign, false);
    let (p_work, c_work) = analytic_expected(&hand, &work_assign, false);
    check(
        "hand task lead-alone analytic success",
        (p_lead - 0.95 * 0.89 * 0.83 * 0.77 * 0.71).abs() < 1e-9,
        format!("p={p_lead:.6}"),
        &mut fails,
    );
    check(
        "hand task worker-alone analytic success",
        (p_work - 0.89f64.powi(5)).abs() < 1e-9,
        format!("p={p_work:.6}"),
        &mut fails,
    );
    check(
        "hand task costs are 50 / 7.5",
        c_lead == 50.0 && (c_work - 7.5).abs() < 1e-9,
        format!("lead={c_lead} worker={c_work}"),
        &mut fails,
    );
    let oracle_hand = oracle_assignment(&hand, false);
    check(
        "oracle on hand task assigns exactly one subtask to the lead",
        oracle_hand.iter().filter(|&&a| a).count() == 4,
        format!("got {oracle_hand:?}"),
        &mut fails,
    );
    let (p_or, c_or) = analytic_expected(&hand, &oracle_hand, false);
    check(
        "oracle hand-task success is 0.95 * 0.89**4",
        (p_or - 0.95 * 0.89f64.powi(4)).abs() < 1e-9,
        format!("p={p_or:.6}"),
        &mut fails,
    );
    check(
        "oracle hand-task cost is 16.0",
        (c_or - 16.0).abs() < 1e-9,
        format!("c={c_or}"),
        &mut fails,
    );
    check(
        "delta=0 assignment equals lead-alone on hand task",
        delegate(&hand, &g, 0.0) == vec![false; 5],
        String::new(),
        &mut fails,
    );
    check(
        "delta=1 assignment equals worker-alone on hand task",
        delegate(&hand, &g, 1.0) == vec![true; 5],
        String::new(),
        &mut fails,
    );

    // ------------------------------------------------------------- held-out A
    let tasks = make_tasks(HELDOUT_SEED, N_TASKS);
    let lead = run_policy(&tasks, |t, _| vec![false; t.subtasks.len()], MC_SEED, TRIALS, false);
    let mut sweep = Vec::new();
    for d in [0.0, 0.25, 0.5, 0.75, 1.0] {
        sweep.push(run_policy(&tasks, |t, _| delegate(t, &g, d), MC_SEED, TRIALS, false));
    }
    let task_level = run_policy(&tasks, |t, _| task_level_assignment(t), MC_SEED, TRIALS, false);
    let rand = run_policy(&tasks, |t, r| random_assignment(t, r), MC_SEED, TRIALS, false);
    let breakdown = run_policy(&tasks, |t, _| vec![false; t.subtasks.len()], MC_SEED, TRIALS, true);
    let fixed = run_policy(&tasks, |t, _| fixed_easy_assignment(t), MC_SEED, TRIALS, false);
    let part = &sweep[3];
    println!(
        "E1 lead-alone   success={:.3} cost={:.1} analytic=({:.3}, {:.1})",
        lead.success_rate, lead.mean_cost, lead.analytic_success, lead.analytic_cost
    );
    println!(
        "E1 partial d.75 success={:.3} cost={:.1} analytic=({:.3}, {:.1})",
        part.success_rate, part.mean_cost, part.analytic_success, part.analytic_cost
    );
    println!("E1 task-level   success={:.3} cost={:.1}", task_level.success_rate, task_level.mean_cost);
    println!("E1 random       success={:.3} cost={:.1}", rand.success_rate, rand.mean_cost);

    // ---------------------------------------------------------------- E1 core
    check(
        "E1 partial d0.75 beats lead-alone on success (MC)",
        part.success_rate > lead.success_rate + 0.01,
        format!("{:.3} vs {:.3}", part.success_rate, lead.success_rate),
        &mut fails,
    );
    check(
        "E1 partial d0.75 beats lead-alone on success (analytic)",
        part.analytic_success > lead.analytic_success + 0.01,
        format!("{:.3} vs {:.3}", part.analytic_success, lead.analytic_success),
        &mut fails,
    );
    check(
        "E1 partial d0.75 beats task-level routing on success",
        part.success_rate > task_level.success_rate + 0.01,
        format!("{:.3} vs {:.3}", part.success_rate, task_level.success_rate),
        &mut fails,
    );
    check(
        "E1 partial d0.75 costs less than lead-alone",
        part.mean_cost < lead.mean_cost - 5.0,
        format!("{:.1} vs {:.1}", part.mean_cost, lead.mean_cost),
        &mut fails,
    );
    check(
        "E1 MC agrees with analytic (lead)",
        (lead.success_rate - lead.analytic_success).abs() < 0.02,
        format!("mc={:.3} analytic={:.3}", lead.success_rate, lead.analytic_success),
        &mut fails,
    );
    check(
        "E1 MC agrees with analytic (partial d0.75)",
        (part.success_rate - part.analytic_success).abs() < 0.02,
        format!("mc={:.3} analytic={:.3}", part.success_rate, part.analytic_success),
        &mut fails,
    );
    check(
        "E1 random delegation is worse than lead-alone (sanity)",
        rand.success_rate < lead.success_rate,
        format!("random={:.3} lead={:.3}", rand.success_rate, lead.success_rate),
        &mut fails,
    );

    // ---------------------------------------------------------------- E2 sweep
    let costs: Vec<f64> = sweep.iter().map(|r| r.mean_cost).collect();
    let succs: Vec<f64> = sweep.iter().map(|r| r.success_rate).collect();
    println!(
        "E2 sweep: {}",
        [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .zip(sweep.iter())
            .map(|(d, r)| format!("d{d}: {:.3}/{:.1}", r.success_rate, r.mean_cost))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    check(
        "E2 cost strictly decreases with delta",
        (1..5).all(|i| costs[i] < costs[i - 1] - 0.5),
        format!("costs={costs:.1?}"),
        &mut fails,
    );
    let best_i = succs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    check(
        "E2 success peaks at an interior delta (not 0 or 1)",
        best_i > 0 && best_i < 4,
        format!("peak at delta={} success={:.3}", [0.0, 0.25, 0.5, 0.75, 1.0][best_i], succs[best_i]),
        &mut fails,
    );
    check(
        "E2 delta=1 (delegate everything) collapses success",
        succs[4] < succs[0],
        format!("d1.0={:.3} vs lead={:.3}", succs[4], succs[0]),
        &mut fails,
    );

    // ---------------------------------------------------------------- E3 ablation
    println!(
        "E3 breakdown-only success={:.3} cost={:.1} analytic=({:.3}, {:.1})",
        breakdown.success_rate, breakdown.mean_cost, breakdown.analytic_success, breakdown.analytic_cost
    );
    check(
        "E3 breakdown-only costs less than lead-alone",
        breakdown.mean_cost < lead.mean_cost - 1.0,
        format!("{:.1} vs {:.1}", breakdown.mean_cost, lead.mean_cost),
        &mut fails,
    );
    check(
        "E3 breakdown-only does not lose success vs lead-alone",
        breakdown.success_rate > lead.success_rate + 0.01,
        format!("{:.3} vs {:.3}", breakdown.success_rate, lead.success_rate),
        &mut fails,
    );

    // ---------------------------------------------------------------- E4 transfer
    let tasks_b = make_tasks(HELDOUT_B_SEED, N_TASKS);
    let learned_b = run_policy(&tasks_b, |t, _| delegate(t, &g, 0.75), MC_SEED, TRIALS, false);
    let fixed_b = run_policy(&tasks_b, |t, _| fixed_easy_assignment(t), MC_SEED, TRIALS, false);
    println!(
        "E4 set B learned d0.75 success={:.3} cost={:.1} | fixed easy-rule success={:.3} cost={:.1}",
        learned_b.success_rate, learned_b.mean_cost, fixed_b.success_rate, fixed_b.mean_cost
    );
    // Honest negative, as in Python: on set B the fixed easy-type
    // rule beats learned guidance at delta=0.75. Recorded, not
    // asserted as a win.
    check(
        "E4 recorded: fixed easy-rule beats learned d0.75 on set B",
        fixed_b.success_rate > learned_b.success_rate,
        format!("fixed={:.3} learned={:.3}", fixed_b.success_rate, learned_b.success_rate),
        &mut fails,
    );

    // ---------------------------------------------------------------- E6 oracle
    let oracle_tasks = make_tasks(ORACLE_SEED, N_ORACLE);
    let (mut o_succ, mut o_cost, mut p_succ, mut p_cost) = (0.0, 0.0, 0.0, 0.0);
    for t in &oracle_tasks {
        let (op, oc) = analytic_expected(t, &oracle_assignment(t, false), false);
        let (pp, pc) = analytic_expected(t, &delegate(t, &g, 0.75), false);
        o_succ += op;
        o_cost += oc;
        p_succ += pp;
        p_cost += pc;
    }
    o_succ /= N_ORACLE as f64;
    o_cost /= N_ORACLE as f64;
    p_succ /= N_ORACLE as f64;
    p_cost /= N_ORACLE as f64;
    println!(
        "E6 oracle-12 success={:.3} cost={:.1} | partial d0.75 success={:.3} cost={:.1} ({:.0}% of oracle)",
        o_succ, o_cost, p_succ, p_cost, 100.0 * p_succ / o_succ
    );
    check(
        "E6 partial d0.75 reaches >=85% of oracle success",
        p_succ >= 0.85 * o_succ,
        format!("{p_succ:.3} vs oracle {o_succ:.3}"),
        &mut fails,
    );
    check(
        "E6 partial d0.75 costs less than the oracle assignment",
        p_cost < o_cost,
        format!("{p_cost:.1} vs {o_cost:.1}"),
        &mut fails,
    );

    // ---------------------------------------------------------------- E5 determinism
    let lead2 = run_policy(&tasks, |t, _| vec![false; t.subtasks.len()], MC_SEED, TRIALS, false);
    let part2 = run_policy(&tasks, |t, _| delegate(t, &g, 0.75), MC_SEED, TRIALS, false);
    check(
        "E5 repeated run gives identical lead numbers",
        lead2.success_rate.to_bits() == lead.success_rate.to_bits()
            && lead2.mean_cost.to_bits() == lead.mean_cost.to_bits(),
        String::new(),
        &mut fails,
    );
    check(
        "E5 repeated run gives identical partial d0.75 numbers",
        part2.success_rate.to_bits() == part.success_rate.to_bits()
            && part2.mean_cost.to_bits() == part.mean_cost.to_bits(),
        String::new(),
        &mut fails,
    );

    // Silence unused warnings for the fixed-baseline run.
    let _ = fixed;
    let _ = guidance_from_gaps;
    let _ = execute;

    println!("\nverify finished in {:.1}s", t0.elapsed().as_secs_f64());
    let results = format!(
        "{{\n  \"article\": \"arXiv:2611.33402\",\n  \"track\": \"rust\",\n  \"e1_lead_success_mc\": {:.6},\n  \"e1_lead_cost\": {:.6},\n  \"e1_lead_success_analytic\": {:.6},\n  \"e1_partial_d075_success_mc\": {:.6},\n  \"e1_partial_d075_cost\": {:.6},\n  \"e1_partial_d075_success_analytic\": {:.6},\n  \"e1_task_level_success_mc\": {:.6},\n  \"e1_random_success_mc\": {:.6},\n  \"e2_sweep_success_mc\": [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}],\n  \"e2_sweep_cost\": [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}],\n  \"e3_breakdown_success_mc\": {:.6},\n  \"e3_breakdown_cost\": {:.6},\n  \"e4_set_b_learned_success_mc\": {:.6},\n  \"e4_set_b_fixed_success_mc\": {:.6},\n  \"e6_oracle_success\": {:.6},\n  \"e6_oracle_cost\": {:.6},\n  \"e6_partial_success\": {:.6},\n  \"e6_partial_cost\": {:.6},\n  \"e6_partial_fraction_of_oracle\": {:.6},\n  \"e5_determinism_identical\": {},\n  \"n_past_tasks\": {},\n  \"all_checks_passed\": {}\n}}\n",
        lead.success_rate,
        lead.mean_cost,
        lead.analytic_success,
        part.success_rate,
        part.mean_cost,
        part.analytic_success,
        task_level.success_rate,
        rand.success_rate,
        succs[0], succs[1], succs[2], succs[3], succs[4],
        costs[0], costs[1], costs[2], costs[3], costs[4],
        breakdown.success_rate,
        breakdown.mean_cost,
        learned_b.success_rate,
        fixed_b.success_rate,
        o_succ,
        o_cost,
        p_succ,
        p_cost,
        p_succ / o_succ,
        fails.is_empty(),
        N_PAST,
        fails.is_empty()
    );
    let out_path = std::env::current_dir().unwrap().join("results.json");
    // When run via `cargo run`, the cwd is the crate root; write there.
    std::fs::write(&out_path, &results).unwrap();
    println!("wrote {}", out_path.display());
    if fails.is_empty() {
        println!("ALL CHECKS PASSED");
    } else {
        println!("FAILED: {} check(s): {:?}", fails.len(), fails);
        std::process::exit(1);
    }
}
