//! `lebet` — agent-facing CLI for the LeBet Rust port.
//!
//! Subcommands (all std-only, offline):
//!   verify [--results PATH]   mirror of tests/test_lebet_cpu.py: full
//!                             experiment + the same 8 assertions, incl. a
//!                             rerun determinism check.
//!   parity <kind> ...         machine-readable outputs for the
//!                             Python-vs-Rust parity sweep (see
//!                             tests/parity_gen.py).
//!   soak                      10x-scale experiment (2000+2000 train/test)
//!                             + thread-identity check.

use lebet_rs::*;
use std::collections::BTreeSet;
use std::time::Instant;

/// Format an f64 the way Python's `json.dump` (repr) would: shortest
/// round-trip decimal, with `.0` for integral values.
fn fnum(f: f64) -> String {
    if f == 0.0 {
        // preserve -0.0 vs 0.0 like Python repr
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let s = format!("{}", f);
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{}.0", s)
    }
}

fn summary_json(s: &Summary) -> String {
    format!(
        "{{\"fpr\": {}, \"power\": {}, \"token_savings\": {}, \
         \"precision_completed\": {}, \"median_intervene_step_fail\": {}, \
         \"n_intervened_fail\": {}, \"n_false_positive\": {}}}",
        fnum(s.fpr),
        fnum(s.power),
        fnum(s.token_savings),
        fnum(s.precision_completed),
        match s.median_intervene_step_fail {
            Some(v) => v.to_string(),
            None => "null".into(),
        },
        s.n_intervened_fail,
        s.n_false_positive
    )
}

/// Full experiment result as JSON, same keys/order as the Python
/// `results.json` (floats via `fnum`, so a parsed comparison is exact).
pub fn experiment_json(r: &ExperimentResult) -> String {
    let e = &r.eval;
    let monitor_raw: Vec<String> = r
        .monitor_raw
        .iter()
        .map(|c| {
            let names: Vec<String> = c.iter().map(|n| format!("\"{}\"", n)).collect();
            format!("[{}]", names.join(", "))
        })
        .collect();
    let tl = format!(
        "{{\"fpr\": {}, \"power\": {}, \"token_savings\": {}, \
         \"precision_completed\": {}, \"median_intervene_step_fail\": {}, \
         \"n_intervened_fail\": {}, \"n_false_positive\": {}, \"k_star\": {}}}",
        fnum(e.turn_limit_tuned.fpr),
        fnum(e.turn_limit_tuned.power),
        fnum(e.turn_limit_tuned.token_savings),
        fnum(e.turn_limit_tuned.precision_completed),
        match e.turn_limit_tuned.median_intervene_step_fail {
            Some(v) => v.to_string(),
            None => "null".into(),
        },
        e.turn_limit_tuned.n_intervened_fail,
        e.turn_limit_tuned.n_false_positive,
        match e.k_star {
            Some(v) => v.to_string(),
            None => "null".into(),
        }
    );
    let ef = format!(
        "{{\"fpr\": {}, \"power\": {}, \"token_savings\": {}, \
         \"precision_completed\": {}, \"median_intervene_step_fail\": {}, \
         \"n_intervened_fail\": {}, \"n_false_positive\": {}, \"predicate\": {}, \
         \"train_tpr\": {}, \"train_fpr\": {}}}",
        fnum(e.error_filter_tuned.fpr),
        fnum(e.error_filter_tuned.power),
        fnum(e.error_filter_tuned.token_savings),
        fnum(e.error_filter_tuned.precision_completed),
        match e.error_filter_tuned.median_intervene_step_fail {
            Some(v) => v.to_string(),
            None => "null".into(),
        },
        e.error_filter_tuned.n_intervened_fail,
        e.error_filter_tuned.n_false_positive,
        match e.error_predicate {
            Some(p) => format!("\"{}\"", p),
            None => "null".into(),
        },
        match e.error_train_tpr {
            Some(v) => fnum(v),
            None => "null".into(),
        },
        match e.error_train_fpr {
            Some(v) => fnum(v),
            None => "null".into(),
        }
    );
    format!(
        "{{\n  \"alpha\": {},\n  \"q\": {},\n  \"lam\": {},\n  \
         \"n_success\": {},\n  \"n_fail\": {},\n  \
         \"base_precision_no_monitor\": {},\n  \
         \"betting\": {},\n  \"immediate\": {},\n  \
         \"turn_limit_tuned\": {},\n  \"error_filter_tuned\": {},\n  \
         \"monitor\": \"{}\",\n  \"monitor_raw\": [{}],\n  \
         \"train_tpr\": {},\n  \"train_fpr\": {},\n  \"train_score\": {},\n  \
         \"train_betting_power\": {},\n  \"train_betting_fpr\": {},\n  \
         \"negative_control\": {{\"monitor\": \"{}\", \
         \"train_tpr_minus_fpr\": {}, \"train_tpr\": {}, \"train_fpr\": {}}},\n  \
         \"seed\": {}\n}}",
        fnum(e.alpha),
        fnum(e.q),
        fnum(e.lam),
        e.n_success,
        e.n_fail,
        fnum(e.base_precision_no_monitor),
        summary_json(&e.betting),
        summary_json(&e.immediate),
        tl,
        ef,
        r.monitor,
        monitor_raw.join(", "),
        fnum(r.train_tpr),
        fnum(r.train_fpr),
        fnum(r.train_score),
        fnum(r.train_betting_power),
        fnum(r.train_betting_fpr),
        r.negative_control.monitor,
        fnum(r.negative_control.train_tpr_minus_fpr),
        fnum(r.negative_control.train_tpr),
        fnum(r.negative_control.train_fpr),
        r.seed
    )
}

fn print_experiment(r: &ExperimentResult) {
    let e = &r.eval;
    let b = &e.betting;
    println!("LeBet experiment (seed={}, alpha={})", r.seed, e.alpha);
    println!("  monitor: {}", r.monitor);
    println!(
        "  train TPR={:.3} FPR={:.3} score={:.3}  q={:.4} betting_power={:.3} betting_FPR={:.3}",
        r.train_tpr, r.train_fpr, r.train_score, e.q, r.train_betting_power, r.train_betting_fpr
    );
    println!(
        "  held-out betting: FPR={:.3} power={:.3} token_savings={:.3} precision={:.3} (base {:.3}) median_intervene_step={:?}",
        b.fpr, b.power, b.token_savings, b.precision_completed, e.base_precision_no_monitor,
        b.median_intervene_step_fail
    );
    println!(
        "  immediate: FPR={:.3} power={:.3} savings={:.3}",
        e.immediate.fpr, e.immediate.power, e.immediate.token_savings
    );
    println!(
        "  turn_limit_tuned (K*={:?}): FPR={:.3} power={:.3} savings={:.3}",
        e.k_star, e.turn_limit_tuned.fpr, e.turn_limit_tuned.power, e.turn_limit_tuned.token_savings
    );
    println!(
        "  error_filter_tuned ({:?}): FPR={:.3} power={:.3} savings={:.3}",
        e.error_predicate, e.error_filter_tuned.fpr, e.error_filter_tuned.power,
        e.error_filter_tuned.token_savings
    );
    println!(
        "  negative control (label-permuted): TPR-FPR={:.3} monitor={}",
        r.negative_control.train_tpr_minus_fpr, r.negative_control.monitor
    );
}

/// The same 8 assertions as tests/test_lebet_cpu.py. Panics on failure.
fn assert_experiment(r: &ExperimentResult) {
    let e = &r.eval;
    let b = &e.betting;
    let alpha = e.alpha;
    // 1. FPR control
    assert!(
        b.fpr <= alpha + 0.05,
        "FPR {:.3} exceeds alpha+slack",
        b.fpr
    );
    // 2. Detection power
    assert!(b.power >= 0.60, "power {:.3} < 0.60", b.power);
    assert!(b.power > e.turn_limit_tuned.power, "power <= turn-limit baseline");
    assert!(b.power > e.error_filter_tuned.power, "power <= error-filter baseline");
    // 3. Token savings
    assert!(b.token_savings > 0.10, "savings {:.3} <= 0.10", b.token_savings);
    assert!(
        b.token_savings > e.turn_limit_tuned.token_savings,
        "savings <= turn-limit baseline"
    );
    // 4. Precision improves
    assert!(
        b.precision_completed > e.base_precision_no_monitor,
        "precision did not improve over no-monitor"
    );
    // 5. Betting beats fire-immediately on FPR
    assert!(e.immediate.fpr > b.fpr, "betting did not reduce FPR vs fire-immediately");
    // 6. Interpretability
    let n_conj = r.monitor_raw.len();
    assert!((1..=3).contains(&n_conj), "{} conjunctions", n_conj);
    assert!(
        r.monitor_raw.iter().all(|c| (1..=3).contains(&c.len())),
        "conjunction too large"
    );
    // 7. Negative control
    assert!(
        r.negative_control.train_tpr_minus_fpr < 0.15,
        "negative control shows separation (leakage?)"
    );
    // 8. Determinism: rerun reproduces monitor and headline metrics
    let r2 = run_experiment_default();
    assert_eq!(r2.monitor, r.monitor, "monitor not deterministic");
    assert!((r2.eval.betting.fpr - b.fpr).abs() < 1e-9, "FPR not deterministic");
    assert!((r2.eval.betting.power - b.power).abs() < 1e-9, "power not deterministic");
}

fn cmd_verify(results_path: Option<&str>) -> i32 {
    let t0 = Instant::now();
    let r = run_experiment_default();
    print_experiment(&r);
    assert_experiment(&r);
    println!("ALL LEBET CHECKS PASSED (rust verify, {:.2}s)", t0.elapsed().as_secs_f64());
    if let Some(path) = results_path {
        std::fs::write(path, experiment_json(&r)).expect("write results");
        println!("results written to {}", path);
    }
    0
}

fn cmd_parity(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("usage: lebet parity <rng|trajs|mats|mutate|wealth|experiment> ...");
        return 2;
    }
    match args[0].as_str() {
        "rng" => {
            let seed: u64 = args[1].parse().expect("seed");
            let n: usize = args[2].parse().expect("n");
            let mut u = Vec::new();
            let mut r = Rng::new(seed);
            for _ in 0..n {
                u.push(r.next_u64());
            }
            let mut un = Vec::new();
            let mut r = Rng::new(seed);
            for _ in 0..n {
                un.push(fnum(r.uniform()));
            }
            let mut ri = Vec::new();
            let mut r = Rng::new(seed);
            for _ in 0..n {
                ri.push(r.randint(6));
            }
            let mut g = Vec::new();
            let mut r = Rng::new(seed);
            for _ in 0..n {
                g.push(fnum(r.gauss(0.0, 1.0)));
            }
            println!(
                "{{\"kind\": \"rng\", \"seed\": {}, \"u64\": {:?}, \"uniform\": [{}], \"randint6\": {:?}, \"gauss\": [{}]}}",
                seed, u, un.join(", "), ri, g.join(", ")
            );
        }
        "trajs" => {
            let seed: u64 = args[1].parse().expect("seed");
            let ns: usize = args[2].parse().expect("n_succ");
            let nf: usize = args[3].parse().expect("n_fail");
            let mut rng = Rng::new(seed);
            let data = gen_dataset(&mut rng, ns, nf);
            let mut items = Vec::new();
            for t in &data {
                let steps: Vec<String> = t
                    .steps
                    .iter()
                    .map(|s| {
                        format!(
                            "[{}, {}, {}, {}, {}]",
                            s.action,
                            if s.ok { "true" } else { "false" },
                            s.state,
                            s.obs,
                            s.tokens
                        )
                    })
                    .collect();
                items.push(format!(
                    "{{\"label\": {}, \"steps\": [{}]}}",
                    t.label,
                    steps.join(", ")
                ));
            }
            println!(
                "{{\"kind\": \"trajs\", \"seed\": {}, \"trajs\": [{}]}}",
                seed,
                items.join(", ")
            );
        }
        "mats" => {
            let seed: u64 = args[1].parse().expect("seed");
            let ns: usize = args[2].parse().expect("n_succ");
            let nf: usize = args[3].parse().expect("n_fail");
            let mut rng = Rng::new(seed);
            let data = gen_dataset(&mut rng, ns, nf);
            let mut items = Vec::new();
            for t in &data {
                let mat = prefix_matrix(t);
                let rows: Vec<String> = mat
                    .iter()
                    .map(|row| {
                        let bits: Vec<&str> =
                            row.iter().map(|b| if *b { "1" } else { "0" }).collect();
                        format!("\"{}\"", bits.join(""))
                    })
                    .collect();
                items.push(format!(
                    "{{\"label\": {}, \"mat\": [{}]}}",
                    t.label,
                    rows.join(", ")
                ));
            }
            println!(
                "{{\"kind\": \"mats\", \"seed\": {}, \"mats\": [{}]}}",
                seed,
                items.join(", ")
            );
        }
        "mutate" => {
            let seed: u64 = args[1].parse().expect("seed");
            let n: usize = args[2].parse().expect("n");
            let mut rng = Rng::new(seed);
            let mut cur: Monitor = vec![BTreeSet::from([0usize, 2]), BTreeSet::from([5])];
            let mut out = Vec::new();
            for _ in 0..n {
                cur = mutate(&mut rng, &cur, 3, 3);
                out.push(format!("\"{}\"", monitor_to_text(&cur)));
            }
            println!(
                "{{\"kind\": \"mutate\", \"seed\": {}, \"monitors\": [{}]}}",
                seed,
                out.join(", ")
            );
        }
        "wealth" => {
            let seed: u64 = args[1].parse().expect("seed");
            let ns: usize = args[2].parse().expect("n_succ");
            let nf: usize = args[3].parse().expect("n_fail");
            let mut rng = Rng::new(seed);
            let data = gen_dataset(&mut rng, ns, nf);
            let mats: Vec<(Mat, bool)> =
                data.iter().map(|t| (prefix_matrix(t), t.label)).collect();
            let mon: Monitor = vec![
                BTreeSet::from([4usize, 8]),
                BTreeSet::from([2]),
                BTreeSet::from([3]),
            ];
            let q = baseline_fire_rate(&mon, &mats);
            let mut items = Vec::new();
            for (mat, label) in mats.iter().take(8) {
                let series = betting_wealth_series(&mon, mat, q, 0.75);
                let vals: Vec<String> = series.iter().map(|w| fnum(*w)).collect();
                items.push(format!(
                    "{{\"label\": {}, \"wealth\": [{}], \"intervene\": {}}}",
                    label,
                    vals.join(", "),
                    match betting_intervene_step(&mon, mat, q, 0.15, 0.75) {
                        Some(v) => v.to_string(),
                        None => "null".into(),
                    }
                ));
            }
            println!(
                "{{\"kind\": \"wealth\", \"seed\": {}, \"q\": {}, \"trajs\": [{}]}}",
                seed,
                fnum(q),
                items.join(", ")
            );
        }
        "experiment" => {
            let seed: u64 = args[1].parse().expect("seed");
            let r = run_experiment(seed, 200, 200, 200, 200, 0.15, 6, 200, 0.75);
            println!("{}", experiment_json(&r));
        }
        other => {
            eprintln!("unknown parity kind: {}", other);
            return 2;
        }
    }
    0
}

fn cmd_soak(results_path: Option<&str>) -> i32 {
    let t0 = Instant::now();
    // 10x trajectories vs the Python test (2000+2000 train/test).
    let r = run_experiment(20261116, 2000, 2000, 2000, 2000, 0.15, 6, 200, 0.75);
    let secs = t0.elapsed().as_secs_f64();
    print_experiment(&r);
    println!("SOAK 10x done in {:.2}s", secs);
    // thread identity: same experiment on 4 threads must agree exactly.
    let handles: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(run_experiment_default))
        .collect();
    let first = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect::<Vec<_>>();
    let same = first
        .iter()
        .all(|x| x.monitor == first[0].monitor
            && (x.eval.betting.fpr - first[0].eval.betting.fpr).abs() < 1e-12
            && (x.eval.betting.power - first[0].eval.betting.power).abs() < 1e-12);
    println!("thread_identity_4x: {}", if same { "MATCH" } else { "MISMATCH" });
    assert!(same, "thread identity failed");
    if let Some(path) = results_path {
        std::fs::write(path, experiment_json(&r)).expect("write results");
        println!("results written to {}", path);
    }
    println!("SOAK PASSED");
    0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = if args.is_empty() || args[0] == "verify" {
        let mut results_path: Option<String> = None;
        if let Some(i) = args.iter().position(|a| a == "--results") {
            results_path = args.get(i + 1).cloned();
        }
        cmd_verify(results_path.as_deref())
    } else if args[0] == "parity" {
        cmd_parity(&args[1..])
    } else if args[0] == "soak" {
        let mut results_path: Option<String> = None;
        if let Some(i) = args.iter().position(|a| a == "--results") {
            results_path = args.get(i + 1).cloned();
        }
        cmd_soak(results_path.as_deref())
    } else {
        eprintln!("usage: lebet [verify [--results PATH] | parity <kind> ... | soak [--results PATH]]");
        2
    };
    std::process::exit(code);
}
