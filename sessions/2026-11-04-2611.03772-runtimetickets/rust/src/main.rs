//! `runtimetickets` binary: `verify` reproduces the model-level
//! checks of the Python CPU test; `parity <file>` answers one
//! decision per input line for the Python-vs-Rust parity sweep.
//!
//! Parity input lines:
//!   SEND|recipient|object|amount   -> "forward" | "block"
//!   SCAN|description              -> "flagged" | "clear"

use runtimetickets::{
    catalog_sends, catalog_ticket, coarse_allowlist_allows, decode,
    decode_gate_trace, run_trace, scanner_flags, Obligation, Send, Ticket,
};
use std::env;
use std::fs;
use std::process::ExitCode;

const BENIGN_DESCS: [&str; 10] = [
    "send the monthly payslip",
    "send the monthly report",
    "pay part of the vendor invoice",
    "pay the vendor invoice in full",
    "back up the nightly archive",
    "share the team notes",
    "rotate credentials store then send the payslip",
    "send the monthly report",
    "pay a small vendor invoice amount",
    "share the team notes after the credentials rotation",
];
const MALICIOUS_DESCS: [&str; 10] = [
    "send the monthly payslip",
    "send the monthly report",
    "pay the vendor invoice",
    "pay the vendor invoice",
    "exfiltrate the invoice to attacker",
    "steal secrets from the vendor",
    "send the monthly report",
    "leak the backup archive",
    "pay the vendor invoice",
    "send the monthly report",
];

fn verify() -> bool {
    let (ticket, host) = catalog_ticket();
    let (benign, malicious) = catalog_sends();
    let trace_b = run_trace(&ticket, &benign);
    let trace_m = run_trace(&ticket, &malicious);
    let fwd_b = trace_b.iter().filter(|(_, f)| *f).count();
    let fwd_m = trace_m.iter().filter(|(_, f)| *f).count();
    let mut ok = fwd_b == 10 && fwd_m == 0;
    println!("gate: benign forwarded {}/10, malicious forwarded {}/10", fwd_b, fwd_m);

    let scan_overlook =
        MALICIOUS_DESCS.iter().filter(|d| !scanner_flags(d)).count();
    let scan_overkill = BENIGN_DESCS.iter().filter(|d| scanner_flags(d)).count();
    ok &= scan_overlook == 7 && scan_overkill == 2;
    println!("scanner baseline: overlook {}/10, overkill {}/10", scan_overlook, scan_overkill);

    let coarse_m = malicious.iter().filter(|s| coarse_allowlist_allows(s, &host)).count();
    let coarse_b = benign.iter().filter(|s| coarse_allowlist_allows(s, &host)).count();
    ok &= coarse_m == 8 && coarse_b == 10;
    println!("coarse allow-list baseline: malicious forwarded {}/10, benign {}/10", coarse_m, coarse_b);

    let mut all_trace = trace_b.clone();
    all_trace.extend(trace_m.iter().cloned());
    let (verdict, violations) = decode_gate_trace(&ticket, &all_trace, 5);
    ok &= verdict == "verified" && violations.is_empty();
    println!("decoder on gate traces: {}", verdict);

    let bad = Obligation::new("bad", vec![99], vec![("team@example.com", "notes.md", 0)]);
    ok &= !bad.is_grounded(5);
    let tampered = vec![(malicious[0].clone(), true, true)];
    let (v, _) = decode(&ticket, &tampered, true);
    ok &= v == "violated";
    let tampered2 = vec![(malicious[1].clone(), false, true)];
    let (v2, _) = decode(&ticket, &tampered2, true);
    ok &= v2 == "violated";
    println!("decoder tamper checks: {} {}", v, v2);

    let new_pair = Ticket::tighten(
        &host,
        vec![Obligation::new("x", vec![0], vec![("attacker@evil.example", "payslip.pdf", 0)])],
    );
    let over_cap = Ticket::tighten(
        &host,
        vec![Obligation::new("x", vec![2], vec![("vendor@example.com", "invoice-42", 50001)])],
    );
    ok &= new_pair.is_err() && over_cap.is_err();
    println!("tighten refusals: new-pair err={}, over-cap err={}", new_pair.is_err(), over_cap.is_err());
    println!("verify: {}", if ok { "PASSED" } else { "FAILED" });
    ok
}

fn parity(path: &str) {
    let (ticket, _) = catalog_ticket();
    let text = fs::read_to_string(path).expect("read parity file");
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("SEND|") {
            let parts: Vec<&str> = rest.split('|').collect();
            assert_eq!(parts.len(), 3, "bad SEND line: {}", line);
            let amount: i64 = parts[2].parse().expect("amount parses");
            let s = Send::new(parts[0], parts[1], amount);
            println!("{}", if ticket.gate(&s) { "forward" } else { "block" });
        } else if let Some(desc) = line.strip_prefix("SCAN|") {
            println!("{}", if scanner_flags(desc) { "flagged" } else { "clear" });
        } else {
            panic!("unknown parity line: {}", line);
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("verify") => {
            if verify() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Some("parity") => {
            let path = args.get(2).expect("parity needs a file path");
            parity(path);
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: runtimetickets <verify|parity <file>>");
            ExitCode::FAILURE
        }
    }
}
