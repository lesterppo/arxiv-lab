//! `vbl verify` mirrors the Python CPU test's headline experiment;
//! `vbl parity` enumerates a fixed case space for Python-vs-Rust
//! comparison (same enumeration in `parity_vbl.py` at the session root).

use std::collections::HashMap;
use vbl_rs::{react_baseline, report_specs, Outcome, Vbl};

/// Scripted env mirroring the Python test: gather ok; analyze injects
/// a data-loss failure on its first `inject_attempts` attempts, then
/// fails while data_ready is false, else ok; regather restores data.
fn run_report_case(
    inject_attempts: usize,
    max_retries: usize,
    max_stack: usize,
) -> (bool, usize, Vec<(String, String)>) {
    let mut attempts: HashMap<String, usize> = HashMap::new();
    let mut env = |action: &str, state: &HashMap<String, bool>| -> Outcome {
        let n = {
            let e = attempts.entry(action.to_string()).or_insert(0);
            *e += 1;
            *e
        };
        match action {
            "gather" => Outcome::Ok(vec![("data_ready".into(), true)]),
            "regather" => Outcome::Ok(vec![("data_ready".into(), true)]),
            "analyze" => {
                if n <= inject_attempts {
                    Outcome::Fail(
                        "MISSING_PRECONDITION".into(),
                        vec![("data_ready".into(), false)],
                    )
                } else if !state.get("data_ready").copied().unwrap_or(false) {
                    Outcome::Fail("MISSING_PRECONDITION".into(), vec![])
                } else {
                    Outcome::Ok(vec![("analysis_done".into(), true)])
                }
            }
            "write" => {
                if !state.get("analysis_done").copied().unwrap_or(false) {
                    Outcome::Fail("MISSING_PRECONDITION".into(), vec![])
                } else {
                    Outcome::Ok(vec![("report_done".into(), true)])
                }
            }
            _ => Outcome::Ok(vec![]),
        }
    };
    let mut vbl = Vbl::new(
        report_specs(),
        "report",
        HashMap::new(),
        max_stack,
        max_retries,
    );
    let (success, steps) = vbl.run(&mut env, 60);
    (success, steps, vbl.op_sequence())
}

fn verify() -> i32 {
    let (success, steps, ops) = run_report_case(1, 2, 8);
    let mut attempts: HashMap<String, usize> = HashMap::new();
    let mut env = |action: &str, state: &HashMap<String, bool>| -> Outcome {
        let n = {
            let e = attempts.entry(action.to_string()).or_insert(0);
            *e += 1;
            *e
        };
        match action {
            "gather" => Outcome::Ok(vec![("data_ready".into(), true)]),
            "regather" => Outcome::Ok(vec![("data_ready".into(), true)]),
            "analyze" => {
                if n == 1 {
                    Outcome::Fail(
                        "MISSING_PRECONDITION".into(),
                        vec![("data_ready".into(), false)],
                    )
                } else if !state.get("data_ready").copied().unwrap_or(false) {
                    Outcome::Fail("MISSING_PRECONDITION".into(), vec![])
                } else {
                    Outcome::Ok(vec![("analysis_done".into(), true)])
                }
            }
            "write" => {
                if !state.get("analysis_done").copied().unwrap_or(false) {
                    Outcome::Fail("MISSING_PRECONDITION".into(), vec![])
                } else {
                    Outcome::Ok(vec![("report_done".into(), true)])
                }
            }
            _ => Outcome::Ok(vec![]),
        }
    };
    let (base_ok, base_steps) =
        react_baseline(&mut env, &["gather", "analyze", "write"], 60);
    println!("vbl_success={} vbl_steps={}", success, steps);
    println!(
        "vbl_ops={}",
        ops.iter()
            .map(|(o, g)| format!("{}:{}", o, g))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("react_success={} react_steps={}", base_ok, base_steps);
    let mut ok = success && steps == 9 && !base_ok && base_steps == 60;
    let expected = [
        ("Push", "gather"),
        ("Pop", "gather"),
        ("Push", "analyze"),
        ("Push", "regather"),
        ("Pop", "regather"),
        ("Pop", "analyze"),
        ("Push", "write"),
        ("Pop", "write"),
        ("Pop", "report"),
    ];
    if ops.len() != expected.len()
        || ops.iter().zip(expected.iter()).any(|(a, b)| {
            a.0 != b.0 || a.1 != b.1
        })
    {
        ok = false;
    }
    if ok {
        println!("VERIFY PASSED");
        0
    } else {
        println!("VERIFY FAILED");
        1
    }
}

fn parity() {
    for inject in 0..=3usize {
        for max_retries in 0..=2usize {
            for max_stack in [2usize, 3, 8] {
                let (success, steps, ops) =
                    run_report_case(inject, max_retries, max_stack);
                println!(
                    "{} {} {} {} {} {}",
                    inject,
                    max_retries,
                    max_stack,
                    success,
                    steps,
                    ops.iter()
                        .map(|(o, g)| format!("{}:{}", o, g))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("verify");
    match cmd {
        "verify" => std::process::exit(verify()),
        "parity" => parity(),
        _ => {
            eprintln!("usage: vbl <verify|parity>");
            std::process::exit(2);
        }
    }
}
