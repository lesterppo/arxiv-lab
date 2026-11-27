//! Rust tests for the Partial Routing port (arXiv:2611.33402).
//!
//! * unit tests for each mechanism (success probabilities, guidance
//!   / delegation sets, cost accounting, analytic values, oracle on
//!   a hand-built task, determinism);
//! * property sweep over random task sets and deltas (cost
//!   non-increasing in delta, success bounded, MC close to analytic,
//!   delta=0/1 costs exact);
//! * exhaustive check: the oracle's analytic success is >= every
//!   assignment's, and its cost is minimal among max-success
//!   assignments;
//! * soak at a scale beyond the Python test (500 tasks);
//! * thread identity (stateless: same computation on 4 threads,
//!   bit-identical);
//! * Python-vs-Rust parity sweep on cases generated from Python
//!   (`parity_cases.txt`): 0 mismatches required on analytic
//!   quantities and assignments (exact f64 equality).

use partial_routing::*;
use std::time::Instant;

const TRUE_GAPS: [f64; 5] = [0.30, 0.25, 0.19, 0.12, 0.06];

fn hand_task() -> Task {
    Task {
        subtasks: vec![Subtask { type_idx: 4, difficulty: 0.0 }; 5],
    }
}

// ---------------------------------------------------------------------------
// unit tests
// ---------------------------------------------------------------------------

#[test]
fn success_prob_values() {
    let st = Subtask { type_idx: 4, difficulty: 0.0 }; // summarize, d0
    assert!((success_prob(0, &st, 0, false) - 0.95).abs() < 1e-12);
    assert!((success_prob(0, &st, 1, false) - 0.89).abs() < 1e-12);
    assert!((success_prob(0, &st, 2, false) - 0.83).abs() < 1e-12);
    assert!((success_prob(1, &st, 0, false) - 0.89).abs() < 1e-12);
    assert!((success_prob(1, &st, 4, false) - 0.89).abs() < 1e-12);
    // worker takes no load penalty; lead with breakdown halves it
    assert!((success_prob(0, &st, 2, true) - 0.89).abs() < 1e-12);
    // clamp: lead plan d1.0 after many lead subtasks floors at 0.05
    let hard = Subtask { type_idx: 0, difficulty: 1.0 };
    assert!((success_prob(0, &hard, 20, false) - 0.05).abs() < 1e-12);
    // difficulty effect: worker code d1.0 = 0.65 - 0.05
    let code = Subtask { type_idx: 1, difficulty: 1.0 };
    assert!((success_prob(1, &code, 0, false) - 0.60).abs() < 1e-12);
}

#[test]
fn delegation_sets_from_true_gaps() {
    let g = guidance_from_gaps(TRUE_GAPS);
    assert_eq!(g.delegate_types(0.0), Vec::<usize>::new());
    assert_eq!(g.delegate_types(0.25), vec![4]);
    assert_eq!(g.delegate_types(0.5), vec![3, 4]);
    assert_eq!(g.delegate_types(0.75), vec![2, 3, 4]);
    assert_eq!(g.delegate_types(1.0), vec![0, 1, 2, 3, 4]);
    assert!((g.gap_max - 0.30).abs() < 1e-12);
}

#[test]
fn hand_task_analytic_values() {
    let hand = hand_task();
    let (p_lead, c_lead) = analytic_expected(&hand, &[false; 5], false);
    assert!((p_lead - 0.95 * 0.89 * 0.83 * 0.77 * 0.71).abs() < 1e-12);
    assert_eq!(c_lead, 50.0);
    let (p_work, c_work) = analytic_expected(&hand, &[true; 5], false);
    assert!((p_work - 0.89f64.powi(5)).abs() < 1e-12);
    assert!((c_work - 7.5).abs() < 1e-12);
    // breakdown: lead cost 9/subtask, load halved
    let (p_bd, c_bd) = analytic_expected(&hand, &[false; 5], true);
    assert_eq!(c_bd, 45.0);
    assert!(p_bd > p_lead); // halved load penalty raises success
}

#[test]
fn oracle_on_hand_task() {
    let hand = hand_task();
    let oracle = oracle_assignment(&hand, false);
    // exactly one subtask to the lead (4 workers)
    assert_eq!(oracle.iter().filter(|&&a| a).count(), 4);
    let (p, c) = analytic_expected(&hand, &oracle, false);
    assert!((p - 0.95 * 0.89f64.powi(4)).abs() < 1e-12);
    assert!((c - 16.0).abs() < 1e-12);
}

#[test]
fn execute_cost_is_assignment_determined() {
    // Cost does not depend on RNG outcomes: check exact costs for a
    // fixed task under several assignments.
    let hand = hand_task();
    let mut rng = Rng::new(42);
    let (_, c1, _) = execute(&hand, &[false; 5], &mut rng, false);
    let (_, c2, _) = execute(&hand, &[true; 5], &mut rng, false);
    let (_, c3, _) = execute(&hand, &[false, true, false, true, false], &mut rng, false);
    assert_eq!(c1, 50.0);
    assert_eq!(c2, 7.5);
    // 3 lead (30) + 2 worker (2 x 1.5) = 33
    assert_eq!(c3, 33.0);
}

#[test]
fn run_policy_is_deterministic() {
    let tasks = make_tasks(101, 60);
    let g = guidance_from_gaps(TRUE_GAPS);
    let r1 = run_policy(&tasks, |t, _| delegate(t, &g, 0.75), 5, 400, false);
    let r2 = run_policy(&tasks, |t, _| delegate(t, &g, 0.75), 5, 400, false);
    assert_eq!(r1.success_rate.to_bits(), r2.success_rate.to_bits());
    assert_eq!(r1.mean_cost.to_bits(), r2.mean_cost.to_bits());
    assert_eq!(r1.analytic_success.to_bits(), r2.analytic_success.to_bits());
}

#[test]
fn task_level_and_fixed_assignments() {
    // all-hard task -> lead; all-easy task -> worker
    let hard = Task {
        subtasks: vec![Subtask { type_idx: 0, difficulty: 0.0 }; 5],
    };
    let easy = Task {
        subtasks: vec![Subtask { type_idx: 4, difficulty: 0.0 }; 5],
    };
    assert_eq!(task_level_assignment(&hard), vec![false; 5]);
    assert_eq!(task_level_assignment(&easy), vec![true; 5]);
    // fixed easy rule delegates test + summarize only
    let mixed = Task {
        subtasks: vec![
            Subtask { type_idx: 0, difficulty: 0.0 },
            Subtask { type_idx: 3, difficulty: 0.0 },
            Subtask { type_idx: 4, difficulty: 0.0 },
        ],
    };
    assert_eq!(fixed_easy_assignment(&mixed), vec![false, true, true]);
}

// ---------------------------------------------------------------------------
// property sweep
// ---------------------------------------------------------------------------

#[test]
fn property_sweep_cost_and_success() {
    let g = guidance_from_gaps(TRUE_GAPS);
    for seed in 0..20u64 {
        let tasks = make_tasks(seed, 20);
        let mut prev_cost = f64::INFINITY;
        let mut saw_strict_drop = false;
        for d in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let r = run_policy(&tasks, |t, _| delegate(t, &g, d), 5, 200, false);
            assert!(r.success_rate >= 0.0 && r.success_rate <= 1.0);
            assert!(r.mean_cost >= 0.0);
            // cost non-increasing in delta (more delegation never
            // costs more); strictly drops somewhere in the sweep
            assert!(r.mean_cost <= prev_cost + 1e-9, "seed {seed} d{d}");
            if r.mean_cost < prev_cost - 1e-9 {
                saw_strict_drop = true;
            }
            prev_cost = r.mean_cost;
            // MC close to analytic at 200 trials x 20 tasks
            assert!((r.success_rate - r.analytic_success).abs() < 0.05);
        }
        assert!(saw_strict_drop, "seed {seed}: cost never drops");
        // endpoints exact: lead-alone 50, all-worker 7.5
        let r0 = run_policy(&tasks, |t, _| delegate(t, &g, 0.0), 5, 50, false);
        let r1 = run_policy(&tasks, |t, _| delegate(t, &g, 1.0), 5, 50, false);
        assert_eq!(r0.mean_cost, 50.0);
        assert_eq!(r1.mean_cost, 7.5);
    }
}

// ---------------------------------------------------------------------------
// exhaustive + oracle
// ---------------------------------------------------------------------------

#[test]
fn oracle_is_max_over_all_assignments() {
    for seed in 2000..2020u64 {
        let tasks = make_tasks(seed, 6);
        for task in &tasks {
            let oracle = oracle_assignment(task, false);
            let (op, oc) = analytic_expected(task, &oracle, false);
            let n = task.subtasks.len();
            for mask in 0..(1usize << n) {
                let assign: Vec<bool> =
                    (0..n).map(|i| ((mask >> (n - 1 - i)) & 1) == 1).collect();
                let (p, c) = analytic_expected(task, &assign, false);
                assert!(op >= p - 1e-15, "oracle beaten on seed {seed}");
                // tie-break: among max-success assignments the oracle
                // has the lowest cost
                if (p - op).abs() < 1e-15 {
                    assert!(oc <= c + 1e-12, "oracle not cheapest among maxima");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// soak (beyond the Python test's scale)
// ---------------------------------------------------------------------------

#[test]
fn soak_500_tasks() {
    let t0 = Instant::now();
    let traces = past_traces(11, 20000);
    let g = learn_guidance(&traces);
    // delegation sets must match the Python test's sets
    assert_eq!(g.delegate_types(0.75), vec![2, 3, 4]);
    let tasks = make_tasks(101, 500);
    let lead = run_policy(&tasks, |t, _| vec![false; t.subtasks.len()], 5, 400, false);
    let part = run_policy(&tasks, |t, _| delegate(t, &g, 0.75), 5, 400, false);
    let task_level = run_policy(&tasks, |t, _| task_level_assignment(t), 5, 400, false);
    println!(
        "soak 500 tasks: lead {:.3}/{:.1} partial d0.75 {:.3}/{:.1} task-level {:.3} ({:.1}s)",
        lead.success_rate,
        lead.mean_cost,
        part.success_rate,
        part.mean_cost,
        task_level.success_rate,
        t0.elapsed().as_secs_f64()
    );
    assert!(part.success_rate > lead.success_rate + 0.01);
    assert!(part.success_rate > task_level.success_rate + 0.01);
    assert!(part.mean_cost < lead.mean_cost - 5.0);
    // oracle comparison on 100 of the soak tasks (analytic)
    let (mut o_succ, mut p_succ) = (0.0, 0.0);
    for t in tasks.iter().take(100) {
        let (op, _) = analytic_expected(t, &oracle_assignment(t, false), false);
        let (pp, _) = analytic_expected(t, &delegate(t, &g, 0.75), false);
        o_succ += op;
        p_succ += pp;
    }
    assert!(p_succ >= 0.85 * o_succ, "soak oracle fraction {p_succ:.3} vs {o_succ:.3}");
    println!("soak oracle-100: partial reaches {:.1}% of oracle", 100.0 * p_succ / o_succ);
}

// ---------------------------------------------------------------------------
// thread identity
// ---------------------------------------------------------------------------

#[test]
fn thread_identity() {
    let tasks = make_tasks(101, 60);
    let g = guidance_from_gaps(TRUE_GAPS);
    let results: Vec<RunResult> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..4 {
            handles.push(s.spawn(|| {
                run_policy(&tasks, |t, _| delegate(t, &g, 0.75), 5, 400, false)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in &results[1..] {
        assert_eq!(r.success_rate.to_bits(), results[0].success_rate.to_bits());
        assert_eq!(r.mean_cost.to_bits(), results[0].mean_cost.to_bits());
        assert_eq!(r.analytic_success.to_bits(), results[0].analytic_success.to_bits());
    }
}

// ---------------------------------------------------------------------------
// Python-vs-Rust parity sweep (exact)
// ---------------------------------------------------------------------------

fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap()
}

#[test]
fn parity_sweep_exact() {
    let text = include_str!("parity_cases.txt");
    let mut n_guidance = 0usize;
    let mut n_task_cases = 0usize;
    let mut n_analytic = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for (lineno, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let sections: Vec<&str> = line.split(" || ").collect();
        if line.starts_with('G') {
            // G <n> <traces...> || GAPS ... LEADQ ... WORKERQ ...
            n_guidance += 1;
            let head: Vec<&str> = sections[0].split_whitespace().collect();
            assert_eq!(head[0], "G");
            let n: usize = head[1].parse().unwrap();
            assert_eq!(head.len() - 2, n, "line {lineno}: trace count");
            let mut traces = Vec::with_capacity(n);
            for tok in &head[2..] {
                let f: Vec<&str> = tok.split(',').collect();
                traces.push(Trace {
                    model: f[0].parse().unwrap(),
                    type_idx: f[1].parse().unwrap(),
                    difficulty: parse_f64(f[2]),
                    success: f[3] == "1",
                    cost: parse_f64(f[4]),
                });
            }
            let tail: Vec<&str> = sections[1].split_whitespace().collect();
            // GAPS g0..g4 LEADQ l0..l4 WORKERQ w0..w4
            assert_eq!(tail[0], "GAPS");
            let exp_gap: [f64; 5] = std::array::from_fn(|i| parse_f64(tail[1 + i]));
            assert_eq!(tail[6], "LEADQ");
            let exp_leadq: [f64; 5] = std::array::from_fn(|i| parse_f64(tail[7 + i]));
            assert_eq!(tail[12], "WORKERQ");
            let exp_workerq: [f64; 5] = std::array::from_fn(|i| parse_f64(tail[13 + i]));
            let g = learn_guidance(&traces);
            for t in 0..5 {
                if g.gap[t].to_bits() != exp_gap[t].to_bits() {
                    mismatches.push(format!(
                        "line {lineno} gap[{t}]: rust {:?} python {:?}",
                        g.gap[t], exp_gap[t]
                    ));
                }
                if g.lead_q[t].to_bits() != exp_leadq[t].to_bits() {
                    mismatches.push(format!(
                        "line {lineno} lead_q[{t}]: rust {:?} python {:?}",
                        g.lead_q[t], exp_leadq[t]
                    ));
                }
                if g.worker_q[t].to_bits() != exp_workerq[t].to_bits() {
                    mismatches.push(format!(
                        "line {lineno} worker_q[{t}]: rust {:?} python {:?}",
                        g.worker_q[t], exp_workerq[t]
                    ));
                }
            }
            n_analytic += 15;
        } else if line.starts_with('T') {
            // T <n> <tasks...> || GAPS ... || <expectations>
            n_task_cases += 1;
            let head: Vec<&str> = sections[0].split_whitespace().collect();
            assert_eq!(head[0], "T");
            let n: usize = head[1].parse().unwrap();
            let task_strs: Vec<&str> = head[2].split('|').collect();
            assert_eq!(task_strs.len(), n, "line {lineno}: task count");
            let mut tasks = Vec::with_capacity(n);
            for ts in &task_strs {
                let mut subs = Vec::new();
                for ss in ts.split(';') {
                    let f: Vec<&str> = ss.split(',').collect();
                    subs.push(Subtask {
                        type_idx: f[0].parse().unwrap(),
                        difficulty: parse_f64(f[1]),
                    });
                }
                tasks.push(Task { subtasks: subs });
            }
            let gaps_tail: Vec<&str> = sections[1].split_whitespace().collect();
            assert_eq!(gaps_tail[0], "GAPS");
            let gaps: [f64; 5] = std::array::from_fn(|i| parse_f64(gaps_tail[1 + i]));
            let g = guidance_from_gaps(gaps);
            let exps: Vec<&str> = sections[2].split(',').map(|s| s.trim()).collect();
            // 5 policies per task, in order: lead, partial075,
            // tasklevel, fixedeasy, oracle
            assert_eq!(exps.len(), n * 5, "line {lineno}: expectation count");
            for (ti, task) in tasks.iter().enumerate() {
                let assigns: [Vec<bool>; 5] = [
                    vec![false; task.subtasks.len()],
                    delegate(task, &g, 0.75),
                    task_level_assignment(task),
                    fixed_easy_assignment(task),
                    oracle_assignment(task, false),
                ];
                for (pi, assign) in assigns.iter().enumerate() {
                    let parts: Vec<&str> = exps[ti * 5 + pi].split_whitespace().collect();
                    let exp_bits = parts[0];
                    let exp_p = parse_f64(parts[1]);
                    let exp_c = parse_f64(parts[2]);
                    let got_bits: String =
                        assign.iter().map(|&a| if a { '1' } else { '0' }).collect();
                    if got_bits != exp_bits {
                        mismatches.push(format!(
                            "line {lineno} task {ti} policy {pi}: assignment rust {got_bits} python {exp_bits}"
                        ));
                    }
                    let (p, c) = analytic_expected(task, assign, false);
                    if p.to_bits() != exp_p.to_bits() || c.to_bits() != exp_c.to_bits() {
                        mismatches.push(format!(
                            "line {lineno} task {ti} policy {pi}: analytic rust ({p:?}, {c:?}) python ({exp_p:?}, {exp_c:?})"
                        ));
                    }
                    n_analytic += 1;
                }
            }
        } else {
            panic!("line {lineno}: unknown case type");
        }
    }

    println!(
        "parity: {n_guidance} guidance case(s), {n_task_cases} task case(s), {n_analytic} analytic comparisons, {} mismatch(es)",
        mismatches.len()
    );
    for m in mismatches.iter().take(10) {
        println!("  MISMATCH: {m}");
    }
    assert!(n_guidance >= 1 && n_task_cases >= 100, "parity file too small");
    assert!(mismatches.is_empty(), "{} parity mismatch(es)", mismatches.len());
}
