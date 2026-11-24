//! `retrytax` binary: `verify` mirrors the Python CPU test's
//! experiments and assertions (E0–E6) on a Rust-generated population;
//! `metrics-batch` serves the Python-vs-Rust parity sweep over stdin;
//! `soak` checks the inequalities at a scale the Python test does
//! not reach.
//!
//! Usage:
//!   retrytax verify [--json PATH]
//!   retrytax metrics-batch   # stdin: "<cap> <cost> <p1> <p2> ..."
//!   retrytax soak

use retrytax_rs::*;
use std::io::{BufRead, Write};

const SEED: u64 = 20261124;
const N_TASKS: usize = 20000;
const CAPS: [usize; 7] = [1, 2, 3, 4, 5, 6, 8];
const COST_USD: f64 = 0.418;

fn check(cond: bool, msg: &str) {
    if !cond {
        panic!("FAILED: {msg}");
    }
    println!("  ok: {msg}");
}

fn verify() -> (String, String) {
    let mut out = String::new();
    let mut log = |s: &str| {
        println!("{s}");
        out.push_str(s);
        out.push('\n');
    };
    let ps = calibrated_mixture(N_TASKS, SEED);

    // V0 — closed-form inequalities (mirrors Python E0).
    log("V0 closed-form inequalities (heterogeneous mixture)");
    let mut cpcs = Vec::new();
    let mut mcs: Vec<Option<f64>> = Vec::new();
    let mut comps = Vec::new();
    for &k in &CAPS {
        let m = population_metrics(&ps, k, COST_USD);
        comps.push(m.completion);
        cpcs.push(m.cost_per_completion);
        if k >= 2 {
            mcs.push(marginal_cost(&ps, k, COST_USD));
        }
        log(&format!(
            "  cap {k}: completion {:.4}  CPC ${:.4}",
            m.completion, m.cost_per_completion
        ));
    }
    log(&format!(
        "  marginal costs: {:?}",
        mcs.iter().map(|m| format!("${:.4}", m.unwrap())).collect::<Vec<_>>()
    ));
    check(cpcs.iter().all(|&c| c >= cpcs[0] - 1e-12), "CPC(K) >= CPC(1)");
    check(
        cpcs.windows(2).all(|w| w[1] >= w[0] - 1e-12),
        "CPC nondecreasing in cap",
    );
    let mc: Vec<f64> = mcs.iter().map(|m| m.unwrap()).collect();
    check(
        mc.windows(2).all(|w| w[1] >= w[0] - 1e-12),
        "marginal cost nondecreasing in cap",
    );
    check(
        (comps[0] - PAPER_COMPLETION_K1).abs() < 0.005,
        "calibrated C(1) ~= paper 0.597",
    );
    check(
        (comps[3] - PAPER_COMPLETION_K4).abs() < 0.02,
        "calibrated C(4) ~= paper 0.700",
    );

    // V1 — Monte-Carlo simulation (mirrors Python E1).
    log("V1 Monte-Carlo simulation (seeded)");
    let mut rng = Rng::new(SEED + 100);
    let mut sim_cpcs = Vec::new();
    let mut sim_comps = Vec::new();
    for &k in &CAPS {
        let (c, _s, cpc) = simulate_population(&ps, k, COST_USD, &mut rng);
        sim_comps.push(c);
        sim_cpcs.push(cpc);
        log(&format!("  cap {k}: completion {c:.4}  CPC ${cpc:.4}"));
    }
    check(
        sim_cpcs.iter().all(|&c| c >= sim_cpcs[0] - 0.02),
        "simulated CPC(K) >= CPC(1) (within MC noise)",
    );
    check(
        sim_cpcs.windows(2).all(|w| w[1] >= w[0] - 0.02),
        "simulated CPC nondecreasing in cap (within MC noise)",
    );
    check(
        (sim_comps[3] - comps[3]).abs() < 0.01,
        "simulated C(4) agrees with closed form",
    );

    // V2 — replay procedure (mirrors Python E2).
    log("V2 replay of per-task attempt sequences (paper's method)");
    let mut rng = Rng::new(SEED + 1);
    let seqs: Vec<Vec<bool>> = ps
        .iter()
        .map(|&p| (0..8).map(|_| rng.next_f64() < p).collect())
        .collect();
    let mut replay = Vec::new();
    for &k in &[1usize, 2, 3, 4] {
        let (c, s, cpc) = replay_runs(&seqs, k, COST_USD, 40, &mut Rng::new(SEED + 2));
        replay.push((c, s, cpc));
        log(&format!(
            "  cap {k}: completion {c:.4}  spend/task ${s:.4}  CPC ${cpc:.4}"
        ));
    }
    check(replay[3].0 > replay[0].0, "replay: completion rises with cap");
    check(replay[3].2 > replay[0].2, "replay: CPC rises with cap");
    check(
        (replay[0].0 - PAPER_COMPLETION_K1).abs() < 0.01,
        "replay C(1) ~= paper 0.597",
    );
    check(
        (replay[3].0 - PAPER_COMPLETION_K4).abs() < 0.02,
        "replay C(4) ~= paper 0.700",
    );

    // V3 — homogeneous population (mirrors Python E3).
    log("V3 homogeneous population (equality case)");
    let homo = vec![0.4; 5000];
    let homo_cpcs: Vec<f64> = CAPS
        .iter()
        .map(|&k| population_metrics(&homo, k, COST_USD).cost_per_completion)
        .collect();
    log(&format!(
        "  CPCs: [{}]",
        homo_cpcs
            .iter()
            .map(|c| format!("{c:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let spread = homo_cpcs.iter().fold(0.0f64, |a, &b| a.max(b))
        - homo_cpcs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    check(spread < 1e-9, "homogeneous CPC constant in K (== cost/p)");
    check(
        (homo_cpcs[0] - COST_USD / 0.4).abs() < 1e-9,
        "homogeneous CPC == cost / p",
    );

    // V4 — survivor selection (mirrors Python E4).
    log("V4 survivor selection (mean p of unresolved tasks)");
    let mut rng = Rng::new(SEED + 3);
    let mut means = vec![ps.iter().sum::<f64>() / ps.len() as f64];
    for &k in &[1usize, 2, 4] {
        means.push(unresolved_mean_p(&ps, k, &mut rng));
    }
    log(&format!(
        "  mean p after k attempts: [{}]",
        means
            .iter()
            .map(|m| format!("{m:.4}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    check(
        means.windows(2).all(|w| w[1] < w[0]),
        "mean p of unresolved tasks strictly falls with attempts",
    );

    // V5 — determinism (mirrors Python E5).
    log("V5 determinism");
    let a = simulate_population(&ps, 4, COST_USD, &mut Rng::new(SEED + 100));
    let b = simulate_population(&ps, 4, COST_USD, &mut Rng::new(SEED + 100));
    check(a == b, "repeat simulation reproduces identical numbers");
    let ra = replay_runs(&seqs, 4, COST_USD, 40, &mut Rng::new(SEED + 2));
    let rb = replay_runs(&seqs, 4, COST_USD, 40, &mut Rng::new(SEED + 2));
    check(ra == rb, "repeat replay reproduces identical numbers");

    // V6 — edge cases (mirrors Python E6).
    log("V6 edge cases");
    check(completion_prob(0.0, 4) == 0.0, "p=0 never completes");
    check(completion_prob(1.0, 4) == 1.0, "p=1 always completes");
    check(expected_attempts(0.0, 4) == 4.0, "p=0 spends the full cap");
    check(expected_attempts(1.0, 4) == 1.0, "p=1 spends one attempt");
    check(
        marginal_cost(&vec![0.0; 100], 2, COST_USD).is_none(),
        "marginal cost undefined when cap adds no completions",
    );
    check(
        population_metrics(&vec![0.0; 10], 2, COST_USD).cost_per_completion
            == f64::INFINITY,
        "CPC infinite when nothing completes",
    );

    log("ALL RUST CHECKS PASSED");
    let fmt = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let json = format!(
        "{{\n  \"verdict\": \"VERIFIED\",\n  \"seed\": {SEED},\n  \"n_tasks\": {N_TASKS},\n  \"caps\": [1, 2, 3, 4, 5, 6, 8],\n  \"cost_usd_per_attempt\": {COST_USD},\n  \"closed_form_completion\": [{}],\n  \"closed_form_cpc_usd\": [{}],\n  \"marginal_cost_usd\": [{}],\n  \"simulated_completion\": [{}],\n  \"simulated_cpc_usd\": [{}],\n  \"replay_completion_k1_k4\": [{:.6}, {:.6}],\n  \"replay_cpc_usd_k1_k4\": [{:.6}, {:.6}],\n  \"homogeneous_cpc_usd\": [{}],\n  \"survivor_mean_p_after_k_0_1_2_4\": [{}],\n  \"checks\": \"ALL RUST CHECKS PASSED\"\n}}",
        fmt(&comps),
        fmt(&cpcs),
        fmt(&mc),
        fmt(&sim_comps),
        fmt(&sim_cpcs),
        replay[0].0,
        replay[3].0,
        replay[0].2,
        replay[3].2,
        fmt(&homo_cpcs),
        fmt(&means),
    );
    (out, json)
}

fn metrics_batch() {
    // stdin lines: "<cap> <cost> <p1> <p2> ..."
    // stdout lines: "<completion> <spend> <cpc> <marginal_or_nan>"
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.expect("stdin read");
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let cap: usize = it.next().expect("cap").parse().expect("cap int");
        let cost: f64 = it.next().expect("cost").parse().expect("cost f64");
        let ps: Vec<f64> = it.map(|t| t.parse().expect("p f64")).collect();
        let m = population_metrics(&ps, cap, cost);
        let mc = if cap >= 2 {
            marginal_cost(&ps, cap, cost).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        writeln!(
            w,
            "{:.17} {:.17} {:.17} {:.17}",
            m.completion, m.spend_per_task, m.cost_per_completion, mc
        )
        .expect("stdout write");
    }
}

fn soak() {
    // 100k tasks, 12 caps: inequalities must hold at a scale the
    // Python test does not reach; also exercises the closed form on
    // a heavy-tailed population (many near-zero p).
    let mut rng = Rng::new(0xC0FFEE);
    let ps: Vec<f64> = (0..100_000)
        .map(|_| {
            let u = rng.next_f64();
            if u < 0.5 {
                0.02
            } else if u < 0.9 {
                rng.next_f64() * 0.5
            } else {
                0.95
            }
        })
        .collect();
    let mut prev_cpc = 0.0f64;
    let mut prev_mc: Option<f64> = None;
    for k in 1..=12usize {
        let m = population_metrics(&ps, k, 1.0);
        assert!(m.cost_per_completion >= prev_cpc - 1e-12, "soak CPC monotone k={k}");
        if k >= 2 {
            let mc = marginal_cost(&ps, k, 1.0).unwrap();
            if let Some(p) = prev_mc {
                assert!(mc >= p - 1e-12, "soak marginal monotone k={k}");
            }
            prev_mc = Some(mc);
        }
        prev_cpc = m.cost_per_completion;
    }
    println!("SOAK PASSED: 100k tasks, caps 1..=12, CPC and marginal cost monotone");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("verify") => {
            let (_log, json) = verify();
            if args.get(2).map(|s| s.as_str()) == Some("--json") {
                let path = args.get(3).expect("--json PATH");
                std::fs::write(path, &json).expect("write json");
            }
        }
        Some("metrics-batch") => metrics_batch(),
        Some("soak") => soak(),
        _ => {
            eprintln!("usage: retrytax verify [--json PATH] | metrics-batch | soak");
            std::process::exit(2);
        }
    }
}
