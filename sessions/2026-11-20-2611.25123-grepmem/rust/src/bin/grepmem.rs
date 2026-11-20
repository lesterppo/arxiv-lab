//! `grepmem` binary: live-test verification + parity helpers.
//!
//! ```text
//! grepmem verify
//!   # mirrors the Python CPU test's 5 experiments with the same
//!   # assertions, plus exact-equality checks against the Python
//!   # results.json (possible because PyRandom reproduces CPython's
//!   # random.Random streams exactly). Prints measured numbers, then
//!   # ALL CHECKS PASSED and RESULTS_JSON {...}
//! grepmem gen <seed> <n_lines> <n_questions>
//!   # print one generated session in the line-based session format
//! grepmem eval --session <path> --policy <name>
//!          [--max-lines N] [--entity-only] [--attr-only]
//!   # evaluate one policy on a session file, print eval JSON
//! grepmem rng <seed> <n>
//!   # print the first n random() values + raw u32s as JSON
//! ```
//!
//! Build artifacts go to `$CARGO_TARGET_DIR` (keep `target/` out of
//! the session dir): `CARGO_TARGET_DIR=/tmp/grepmem-rs-target cargo
//! build --release --offline`.

use grepmem_rs::{
    evaluate_policy, evaluate_policy_detailed, fmt_f64, generate_session, json_escape,
    policy_grepmem, run_suite, session_from_text, session_to_text, EvalResult, GrepMemOpts,
    Policy, PyRandom, Session, SuiteResult,
};
use std::process::exit;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn check(cond: bool, msg: &str) {
    if !cond {
        eprintln!("CHECK FAILED: {}", msg);
        exit(1);
    }
}

/// Expected Python results.json numbers (main suite, unrounded).
/// (policy, accuracy, mean_context_lines, by_position[early, mid, late],
///  paraphrase_accuracy, paraphrase_n)
const EXPECTED_MAIN: [(&str, f64, f64, [f64; 3], f64, usize); 5] = [
    ("full", 1.0, 500.0, [1.0, 1.0, 1.0], 1.0, 14),
    ("grepmem", 1.0, 14.31, [1.0, 1.0, 1.0], 1.0, 14),
    (
        "summary",
        0.2,
        44.0,
        [0.15151515151515152, 0.15, 0.3333333333333333],
        0.07142857142857142,
        14,
    ),
    (
        "recency",
        0.25,
        60.0,
        [0.12121212121212122, 0.225, 0.4444444444444444],
        0.07142857142857142,
        14,
    ),
    (
        "random_mask",
        0.4,
        97.0,
        [0.30303030303030304, 0.425, 0.48148148148148145],
        0.5,
        14,
    ),
];

fn print_suite(suite: &SuiteResult) {
    for policy in Policy::all() {
        let r = suite.get(policy);
        println!(
            "  {:12} acc={:.3} ctx_lines={:7.1} para_acc={:.3} (n={})",
            r.policy, r.accuracy, r.mean_context_lines, r.paraphrase_accuracy, r.paraphrase_n
        );
    }
}

fn eval_result_json(r: &EvalResult) -> String {
    format!(
        "{{\"policy\": {}, \"accuracy\": {}, \"n\": {}, \"mean_context_lines\": {}, \"by_position\": {{\"early\": {}, \"mid\": {}, \"late\": {}}}, \"paraphrase_accuracy\": {}, \"paraphrase_n\": {}}}",
        json_escape(r.policy),
        fmt_f64(r.accuracy),
        r.n,
        fmt_f64(r.mean_context_lines),
        fmt_f64(r.by_position[0]),
        fmt_f64(r.by_position[1]),
        fmt_f64(r.by_position[2]),
        fmt_f64(r.paraphrase_accuracy),
        r.paraphrase_n,
    )
}

fn verify() {
    let out_main: String;
    let out_scaling: String;
    let out_budget: String;
    let out_mismatch: String;
    let out_recall: String;

    // --- Experiment 1: main comparison, 5 seeds x 20 questions, 500 lines.
    let suite = run_suite(&[0, 1, 2, 3, 4], 500, 20, 40);
    println!("== main suite: 5 seeds x 20 questions, 500-line streams ==");
    print_suite(&suite);
    let r = |p: Policy| suite.get(p);
    check(r(Policy::GrepMem).accuracy >= 0.90, "grepmem acc >= 0.90");
    check(
        r(Policy::Full).accuracy - r(Policy::GrepMem).accuracy <= 0.05,
        "grepmem within 0.05 of full ceiling",
    );
    let best_lossy = r(Policy::Summary)
        .accuracy
        .max(r(Policy::Recency).accuracy)
        .max(r(Policy::RandomMask).accuracy);
    check(best_lossy <= 0.60, "best lossy acc <= 0.60");
    check(
        r(Policy::GrepMem).accuracy - best_lossy >= 0.25,
        "grepmem beats best lossy by >= 0.25",
    );
    check(
        r(Policy::GrepMem).mean_context_lines <= 0.15 * r(Policy::Full).mean_context_lines,
        "grepmem ctx <= 15% of full",
    );
    check(
        r(Policy::GrepMem).by_position[0] >= 0.85,
        "grepmem early-position acc >= 0.85",
    );
    check(
        r(Policy::Recency).by_position[0] <= 0.40,
        "recency early-position acc <= 0.40",
    );
    // Exact-equality against Python results.json (same RNG streams).
    for (name, acc, ctx, by_pos, para, para_n) in EXPECTED_MAIN {
        let res = suite.results.iter().find(|x| x.policy == name).unwrap();
        check(close(res.accuracy, acc), &format!("{} acc == python {}", name, acc));
        check(close(res.mean_context_lines, ctx), &format!("{} ctx == python {}", name, ctx));
        for i in 0..3 {
            check(
                close(res.by_position[i], by_pos[i]),
                &format!("{} by_position[{}] == python {}", name, i, by_pos[i]),
            );
        }
        check(
            close(res.paraphrase_accuracy, para),
            &format!("{} paraphrase_acc == python {}", name, para),
        );
        check(res.paraphrase_n == para_n, &format!("{} paraphrase_n == {}", name, para_n));
        check(res.correct as f64 == acc * res.n as f64, &format!("{} correct count", name));
    }
    {
        let mut parts = Vec::new();
        for policy in Policy::all() {
            parts.push(format!("{}: {}", json_escape(policy.name()), eval_result_json(suite.get(policy))));
        }
        out_main = format!(
            "{{\"n_lines\": 500, \"n_sessions\": 5, \"n_questions\": 100, \"results\": {{{}}}}}",
            parts.join(", ")
        );
    }

    // --- Experiment 2: scaling — stream length 200 -> 2000 lines.
    println!("== scaling: stream length sweep ==");
    let expected_scaling: [(usize, [f64; 5]); 4] = [
        (200, [1.0, 1.0, 0.25, 0.41, 0.2]),
        (500, [1.0, 1.0, 0.2, 0.25, 0.4]),
        (1000, [1.0, 1.0, 0.18, 0.17, 0.46]),
        (2000, [1.0, 1.0, 0.13, 0.22, 0.52]),
    ];
    let mut scaling_parts = Vec::new();
    let mut scaling_acc: Vec<[f64; 5]> = Vec::new();
    for (n_lines, expected) in expected_scaling {
        let s = run_suite(&[0, 1, 2, 3, 4], n_lines, 20, 40);
        let accs: [f64; 5] = [
            s.get(Policy::Full).accuracy,
            s.get(Policy::GrepMem).accuracy,
            s.get(Policy::Summary).accuracy,
            s.get(Policy::Recency).accuracy,
            s.get(Policy::RandomMask).accuracy,
        ];
        println!(
            "  lines={:5} {}",
            n_lines,
            ["full", "grepmem", "summary", "recency", "random_mask"]
                .iter()
                .zip(accs.iter())
                .map(|(n, a)| format!("{}={:.3}", n, a))
                .collect::<Vec<_>>()
                .join("  ")
        );
        for (i, policy) in Policy::all().iter().enumerate() {
            check(
                close(round4(accs[i]), expected[i]),
                &format!("scaling lines={} {} == python {}", n_lines, policy.name(), expected[i]),
            );
        }
        scaling_acc.push(accs);
        scaling_parts.push(format!(
            "\"{}\": {{\"full\": {}, \"grepmem\": {}, \"summary\": {}, \"recency\": {}, \"random_mask\": {}}}",
            n_lines,
            fmt_f64(round4(accs[0])),
            fmt_f64(round4(accs[1])),
            fmt_f64(round4(accs[2])),
            fmt_f64(round4(accs[3])),
            fmt_f64(round4(accs[4])),
        ));
    }
    check(
        scaling_acc[3][1] >= scaling_acc[0][1] - 0.05,
        "grepmem acc at 2000 lines within 0.05 of 200 lines",
    );
    check(
        scaling_acc[3][3] <= scaling_acc[0][3] - 0.15,
        "recency acc drops >= 0.15 from 200 to 2000 lines",
    );
    out_scaling = format!("{{{}}}", scaling_parts.join(", "));

    // --- Experiment 3: grep budget sweep (accuracy vs retrieved lines).
    println!("== grep budget sweep (max_lines) ==");
    let sessions: Vec<Session> = (0..5).map(|s| generate_session(s, 500, 20)).collect();
    let expected_budget: [(usize, f64, f64); 4] = [
        (5, 0.61, 5.0),
        (10, 0.88, 9.6),
        (20, 0.99, 14.0),
        (40, 1.0, 14.3),
    ];
    let mut budget_parts = Vec::new();
    let mut budget_acc = [0.0f64; 4];
    for (i, (k, exp_acc, exp_ctx)) in expected_budget.iter().enumerate() {
        let res = evaluate_policy(Policy::GrepMem, &sessions, *k);
        println!(
            "  max_lines={:3} acc={:.3} ctx={:.1}",
            k, res.accuracy, res.mean_context_lines
        );
        check(
            close(round4(res.accuracy), *exp_acc),
            &format!("budget max_lines={} acc == python {}", k, exp_acc),
        );
        check(
            close(round1(res.mean_context_lines), *exp_ctx),
            &format!("budget max_lines={} ctx == python {}", k, exp_ctx),
        );
        budget_acc[i] = res.accuracy;
        budget_parts.push(format!(
            "\"{}\": {{\"accuracy\": {}, \"mean_context_lines\": {}}}",
            k,
            fmt_f64(round4(res.accuracy)),
            fmt_f64(round1(res.mean_context_lines)),
        ));
    }
    check(budget_acc[3] >= 0.90, "budget max_lines=40 acc >= 0.90");
    check(budget_acc[0] <= budget_acc[3], "budget acc non-decreasing to 40");
    out_budget = format!("{{{}}}", budget_parts.join(", "));

    // --- Experiment 4: vocabulary mismatch (honest limit, reported).
    println!("== vocabulary mismatch: attribute-only grep, paraphrased Qs ==");
    let expected_mismatch: [(&str, f64, f64); 3] = [
        ("entity+attr", 1.0, 1.0),
        ("entity_only", 1.0, 1.0),
        ("attr_only", 0.86, 0.0),
    ];
    let mut mismatch_parts = Vec::new();
    for (label, exp_acc, exp_para) in expected_mismatch {
        let opts = match label {
            "entity_only" => GrepMemOpts {
                max_lines: 40,
                entity_only: true,
                attr_only: false,
            },
            "attr_only" => GrepMemOpts {
                max_lines: 40,
                entity_only: false,
                attr_only: true,
            },
            _ => GrepMemOpts::new(40),
        };
        let res = evaluate_policy_detailed("grepmem", &sessions, |log, q| {
            policy_grepmem(log, q, opts)
        });
        println!(
            "  {:12} acc={:.3} para_acc={:.3}",
            label, res.accuracy, res.paraphrase_accuracy
        );
        check(
            close(round4(res.accuracy), exp_acc),
            &format!("mismatch {} acc == python {}", label, exp_acc),
        );
        check(
            close(round4(res.paraphrase_accuracy), exp_para),
            &format!("mismatch {} para_acc == python {}", label, exp_para),
        );
        mismatch_parts.push(format!(
            "{}: {{\"accuracy\": {}, \"paraphrase_accuracy\": {}}}",
            json_escape(label),
            fmt_f64(round4(res.accuracy)),
            fmt_f64(round4(res.paraphrase_accuracy)),
        ));
    }
    out_mismatch = format!("{{{}}}", mismatch_parts.join(", "));

    // --- Experiment 5: long-stream recall (single fact, 1000-line stream).
    println!("== long-stream recall: 1 buried fact, 1000 lines ==");
    let expected_recall: [(&str, f64); 4] = [
        ("full", 1.0),
        ("grepmem", 1.0),
        ("recency", 0.4),
        ("summary", 0.2),
    ];
    let mut recall_parts = Vec::new();
    let mut recall_acc: [f64; 4] = [0.0; 4];
    for (i, (name, exp)) in expected_recall.iter().enumerate() {
        let policy = match *name {
            "full" => Policy::Full,
            "grepmem" => Policy::GrepMem,
            "recency" => Policy::Recency,
            _ => Policy::Summary,
        };
        let mut accs = Vec::new();
        for seed in 0..5u64 {
            let sess = generate_session(seed, 1000, 1);
            let res = evaluate_policy(policy, std::slice::from_ref(&sess), 40);
            accs.push(res.accuracy);
        }
        let mean = accs.iter().sum::<f64>() / accs.len() as f64;
        recall_acc[i] = mean;
        check(
            close(round3(mean), *exp),
            &format!("recall {} == python {}", name, exp),
        );
        recall_parts.push(format!("{}: {}", json_escape(name), fmt_f64(round3(mean))));
    }
    println!(
        "  {}",
        ["full", "grepmem", "recency", "summary"]
            .iter()
            .zip(recall_acc.iter())
            .map(|(n, a)| format!("{}={:.3}", n, a))
            .collect::<Vec<_>>()
            .join("  ")
    );
    check(recall_acc[1] >= 0.80, "long-stream grepmem recall >= 0.80");
    out_recall = format!("{{{}}}", recall_parts.join(", "));

    println!("ALL CHECKS PASSED");
    println!(
        "RESULTS_JSON {{\"main_suite\": {}, \"scaling\": {}, \"grep_budget\": {}, \"vocabulary_mismatch\": {}, \"long_stream_recall\": {}}}",
        out_main, out_scaling, out_budget, out_mismatch, out_recall
    );
}

fn cmd_gen(seed: u64, n_lines: usize, n_questions: usize) {
    let sess = generate_session(seed, n_lines, n_questions);
    print!("{}", session_to_text(&sess, seed, n_lines, n_questions));
}

fn cmd_rng(seed: u64, n: usize) {
    let mut rng = PyRandom::new(seed);
    let randoms: Vec<String> = (0..n).map(|_| fmt_f64(rng.random())).collect();
    let mut rng2 = PyRandom::new(seed);
    let u32s: Vec<String> = (0..n).map(|_| rng2.gen_u32().to_string()).collect();
    println!(
        "{{\"kind\": \"rng\", \"seed\": {}, \"random\": [{}], \"u32\": [{}]}}",
        seed,
        randoms.join(", "),
        u32s.join(", ")
    );
}

fn cmd_eval(args: &[String]) {
    let mut session_path: Option<String> = None;
    let mut policy_name = "grepmem".to_string();
    let mut max_lines = 40usize;
    let mut entity_only = false;
    let mut attr_only = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--session" => {
                session_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--policy" => {
                policy_name = args[i + 1].clone();
                i += 2;
            }
            "--max-lines" => {
                max_lines = args[i + 1].parse().expect("--max-lines must be a number");
                i += 2;
            }
            "--entity-only" => {
                entity_only = true;
                i += 1;
            }
            "--attr-only" => {
                attr_only = true;
                i += 1;
            }
            other => {
                eprintln!("unknown eval flag: {}", other);
                exit(2);
            }
        }
    }
    let path = session_path.expect("--session is required");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        exit(2);
    });
    let sess = session_from_text(&text).unwrap_or_else(|e| {
        eprintln!("cannot parse {}: {}", path, e);
        exit(2);
    });
    let sessions = vec![sess];
    let res = match policy_name.as_str() {
        "full" => evaluate_policy(Policy::Full, &sessions, max_lines),
        "grepmem" => evaluate_policy_detailed("grepmem", &sessions, |log, q| {
            policy_grepmem(
                log,
                q,
                GrepMemOpts {
                    max_lines,
                    entity_only,
                    attr_only,
                },
            )
        }),
        "summary" => evaluate_policy(Policy::Summary, &sessions, max_lines),
        "recency" => evaluate_policy(Policy::Recency, &sessions, max_lines),
        "random_mask" => evaluate_policy(Policy::RandomMask, &sessions, max_lines),
        other => {
            eprintln!("unknown policy: {}", other);
            exit(2);
        }
    };
    println!("{}", grepmem_rs::eval_to_json(&res));
}

fn usage() -> ! {
    eprintln!("usage: grepmem verify | gen <seed> <n_lines> <n_questions> | rng <seed> <n> | eval --session <path> --policy <name> [--max-lines N] [--entity-only] [--attr-only]");
    exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    match args[0].as_str() {
        "verify" => verify(),
        "gen" => {
            if args.len() != 4 {
                usage();
            }
            cmd_gen(
                args[1].parse().expect("seed must be a number"),
                args[2].parse().expect("n_lines must be a number"),
                args[3].parse().expect("n_questions must be a number"),
            );
        }
        "rng" => {
            if args.len() != 3 {
                usage();
            }
            cmd_rng(
                args[1].parse().expect("seed must be a number"),
                args[2].parse().expect("n must be a number"),
            );
        }
        "eval" => cmd_eval(&args[1..]),
        _ => usage(),
    }
}
