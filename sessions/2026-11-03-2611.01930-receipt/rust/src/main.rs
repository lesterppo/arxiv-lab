//! `receipt verify` -- reproduce the Python test's model-level checks.
//! `receipt parity` -- read "guarded,mode,policy" lines on stdin, print
//! "status,action_done,timed_out,false_success" per line (status -1 = timeout kill).

use receipt::{extract_waiting_prompt, simulate_outcome, Mode};
use std::io::{self, BufRead};

fn fmt(o: (Option<i32>, bool, bool, bool)) -> String {
    format!(
        "{},{},{},{}",
        o.0.map(|v| v.to_string()).unwrap_or_else(|| "-1".to_string()),
        o.1 as u8,
        o.2 as u8,
        o.3 as u8
    )
}

fn verify() -> i32 {
    let mut failures = 0;
    let mut check = |name: &str, cond: bool| {
        if !cond {
            eprintln!("FAIL {}", name);
            failures += 1;
        }
    };
    // Detector cases (mirror the Python unit cases exactly).
    check(
        "prompt-yN",
        extract_waiting_prompt("Confirm: deploy? [y/N] ").as_deref()
            == Some("Confirm: deploy? [y/N]"),
    );
    check(
        "prompt-q",
        extract_waiting_prompt("working...\nProceed? ").as_deref() == Some("Proceed?"),
    );
    check("prompt-none", extract_waiting_prompt("working...\ndone: ok\n").is_none());
    check("prompt-empty", extract_waiting_prompt("").is_none());
    check(
        "prompt-newest",
        extract_waiting_prompt("First? \nSecond? ").as_deref() == Some("Second?"),
    );
    // Outcome model: every guarded x mode x policy cell.
    check(
        "closed-guarded",
        simulate_outcome(true, Mode::Closed, true) == (Some(0), false, false, true),
    );
    check(
        "open-guarded",
        simulate_outcome(true, Mode::Open, true) == (None, false, true, false),
    );
    check(
        "receipt-guarded",
        simulate_outcome(true, Mode::Receipt, true) == (Some(0), true, false, false),
    );
    check(
        "receipt-silent",
        simulate_outcome(true, Mode::Receipt, false) == (Some(0), false, false, false),
    );
    for mode in [Mode::Closed, Mode::Open, Mode::Receipt] {
        check(
            "unguarded",
            simulate_outcome(false, mode, true) == (Some(0), true, false, false),
        );
    }
    // Catalog summary (6 guarded + 4 unguarded, as in the Python test).
    let guarded = 6;
    let unguarded = 4;
    let closed_false = (0..guarded)
        .filter(|_| simulate_outcome(true, Mode::Closed, true).3)
        .count();
    let open_timeouts = (0..guarded)
        .filter(|_| simulate_outcome(true, Mode::Open, true).2)
        .count();
    let receipt_done = (0..guarded)
        .filter(|_| simulate_outcome(true, Mode::Receipt, true).1)
        .count()
        + (0..unguarded)
            .filter(|_| simulate_outcome(false, Mode::Receipt, true).1)
            .count();
    let receipt_false = (0..guarded)
        .filter(|_| simulate_outcome(true, Mode::Receipt, true).3)
        .count();
    println!("catalog: guarded={} unguarded={}", guarded, unguarded);
    println!("closed: false-success {}/{}", closed_false, guarded);
    println!("open-unattended: timeouts {}/{}", open_timeouts, guarded);
    println!(
        "receipt: completed {}/{}, false-success {}",
        receipt_done,
        guarded + unguarded,
        receipt_false
    );
    check("closed-false-all", closed_false == guarded);
    check("open-timeout-all", open_timeouts == guarded);
    check("receipt-all-done", receipt_done == guarded + unguarded);
    check("receipt-no-false", receipt_false == 0);
    if failures == 0 {
        println!("RECEIPT RUST verify PASSED");
        0
    } else {
        eprintln!("RECEIPT RUST verify FAILED: {} failures", failures);
        1
    }
}

fn parity() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 3 {
            println!("ERR");
            continue;
        }
        let guarded = parts[0] == "1";
        let mode = match Mode::parse(parts[1]) {
            Some(m) => m,
            None => {
                println!("ERR");
                continue;
            }
        };
        let policy = parts[2] == "1";
        println!("{}", fmt(simulate_outcome(guarded, mode, policy)));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("verify") => std::process::exit(verify()),
        Some("parity") => parity(),
        _ => {
            eprintln!("usage: receipt <verify|parity>");
            std::process::exit(2);
        }
    }
}
