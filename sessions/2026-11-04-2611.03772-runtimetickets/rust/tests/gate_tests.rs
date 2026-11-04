//! Live extensive tests for the RuntimeTickets Rust port.
//! Deterministic; no network, no model. A small LCG stands in for
//! `rand` (std-only crate policy).

use runtimetickets::{
    catalog_sends, catalog_ticket, decode, decode_gate_trace, run_trace,
    scanner_flags, Obligation, Send, Ticket,
};
use std::collections::BTreeMap;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

#[test]
fn catalog_gate_counts() {
    let (ticket, _) = catalog_ticket();
    let (benign, malicious) = catalog_sends();
    assert_eq!(run_trace(&ticket, &benign).iter().filter(|(_, f)| *f).count(), 10);
    assert_eq!(run_trace(&ticket, &malicious).iter().filter(|(_, f)| *f).count(), 0);
    // Boundary: exactly at cap forwards, cap+1 blocks.
    assert!(ticket.gate(&Send::new("vendor@example.com", "invoice-42", 50000)));
    assert!(!ticket.gate(&Send::new("vendor@example.com", "invoice-42", 50001)));
}

#[test]
fn exhaustive_cells_vs_oracle() {
    let (ticket, _) = catalog_ticket();
    let recipients = ["alice@example.com", "bob@example.com", "vendor@example.com",
        "backup@example.com", "team@example.com", "attacker@evil.example",
        "mallory@example.com"];
    let objects = ["payslip.pdf", "report.pdf", "invoice-42", "backup.tar",
        "notes.md", "credentials.json", "secrets.db"];
    let amounts = [0i64, 1, 100, 1000, 25000, 49999, 50000, 50001, 999999];
    let mut cells = 0;
    for r in recipients {
        for o in objects {
            for a in amounts {
                let s = Send::new(r, o, a);
                assert_eq!(ticket.gate(&s), ticket.oracle_allows(&s), "{:?}", s);
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 7 * 7 * 9); // 441 exhaustive cells
}

#[test]
fn random_cases_vs_oracle() {
    let (ticket, _) = catalog_ticket();
    let recipients = ["alice@example.com", "bob@example.com", "vendor@example.com",
        "backup@example.com", "team@example.com", "attacker@evil.example"];
    let objects = ["payslip.pdf", "report.pdf", "invoice-42", "backup.tar",
        "notes.md", "credentials.json", "secrets.db"];
    let amounts = [0i64, 1, 100, 25000, 50000, 50001, 999999, -5];
    let mut rng = Lcg(0x1104);
    for _ in 0..5000 {
        let s = Send::new(
            recipients[rng.below(recipients.len())],
            objects[rng.below(objects.len())],
            amounts[rng.below(amounts.len())],
        );
        assert_eq!(ticket.gate(&s), ticket.oracle_allows(&s), "{:?}", s);
    }
}

#[test]
fn soak_invariants() {
    let (ticket, _) = catalog_ticket();
    let (benign, malicious) = catalog_sends();
    let mut all = benign.clone();
    all.extend(malicious.iter().cloned());
    let mut fwd_total = 0;
    for _ in 0..10000 {
        let trace = run_trace(&ticket, &all);
        fwd_total += trace.iter().filter(|(_, f)| *f).count();
    }
    assert_eq!(fwd_total, 10000 * 10); // exactly the 10 benign, every round
}

#[test]
fn thread_identity() {
    use std::sync::Arc;
    let (ticket, _) = catalog_ticket();
    let ticket = Arc::new(ticket);
    let (benign, malicious) = catalog_sends();
    let mut all = benign;
    all.extend(malicious);
    let all = Arc::new(all);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let t = Arc::clone(&ticket);
            let a = Arc::clone(&all);
            std::thread::spawn(move || {
                run_trace(&t, &a).iter().map(|(_, f)| *f).collect::<Vec<bool>>()
            })
        })
        .collect();
    let first = handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>();
    for other in &first[1..] {
        assert_eq!(&first[0], other);
    }
    assert_eq!(first[0].iter().filter(|f| **f).count(), 10);
}

#[test]
fn tighten_rules() {
    let (_, host) = catalog_ticket();
    // New pair refused, over-cap refused, narrowing accepted.
    assert!(Ticket::tighten(
        &host,
        vec![Obligation::new("x", vec![0], vec![("attacker@evil.example", "payslip.pdf", 0)])],
    )
    .is_err());
    assert!(Ticket::tighten(
        &host,
        vec![Obligation::new("x", vec![2], vec![("vendor@example.com", "invoice-42", 50001)])],
    )
    .is_err());
    let narrowed = Ticket::tighten(
        &host,
        vec![Obligation::new("x", vec![2], vec![("vendor@example.com", "invoice-42", 1000)])],
    )
    .unwrap();
    assert_eq!(narrowed.allowed[&("vendor@example.com".to_string(), "invoice-42".to_string())], 1000);
    // Fuzz invariant: success implies subset + caps <= host.
    let mut rng = Lcg(0xC0FFEE);
    let pairs: Vec<(String, String)> = host.keys().cloned().collect();
    for _ in 0..500 {
        let mut obs = Vec::new();
        for (i, (r, o)) in pairs.iter().enumerate() {
            let cap = match rng.below(3) {
                0 => 0,
                1 => host[&(r.clone(), o.clone())],
                _ => host[&(r.clone(), o.clone())] + 1,
            };
            obs.push(Obligation::new(&format!("f{}", i), vec![0], vec![(r.as_str(), o.as_str(), cap)]));
        }
        if let Ok(t) = Ticket::tighten(&host, obs) {
            for ((r, o), cap) in &t.allowed {
                assert!(host.contains_key(&(r.clone(), o.clone())));
                assert!(*cap <= host[&(r.clone(), o.clone())]);
            }
        }
    }
    // Empty obligations tighten to an empty ticket that blocks all.
    let empty = Ticket::tighten(&host, vec![]).unwrap();
    assert!(empty.allowed.is_empty());
    assert!(!empty.gate(&Send::new("alice@example.com", "payslip.pdf", 0)));
    let _ = BTreeMap::<(String, String), i64>::new();
}

#[test]
fn decoder_cases() {
    let (ticket, _) = catalog_ticket();
    let (benign, malicious) = catalog_sends();
    let mut trace = run_trace(&ticket, &benign);
    trace.extend(run_trace(&ticket, &malicious));
    let (v, violations) = decode_gate_trace(&ticket, &trace, 5);
    assert_eq!(v, "verified");
    assert!(violations.is_empty());
    // Ungrounded obligation.
    let bad = Obligation::new("bad", vec![99], vec![("team@example.com", "notes.md", 0)]);
    assert!(!bad.is_grounded(5));
    // Tampered traces.
    let (v, violations) = decode(&ticket, &[(malicious[0].clone(), true, true)], true);
    assert_eq!(v, "violated");
    assert!(violations.iter().any(|x| x.starts_with("forwarded_outside_ticket")));
    let (v, violations) = decode(&ticket, &[(malicious[1].clone(), false, true)], true);
    assert_eq!(v, "violated");
    assert!(violations.iter().any(|x| x.starts_with("effect_on_blocked_send")));
}

#[test]
fn scanner_cases() {
    assert!(scanner_flags("exfiltrate the invoice to attacker"));
    assert!(scanner_flags("steal secrets from the vendor"));
    assert!(scanner_flags("leak the backup archive"));
    assert!(scanner_flags("rotate credentials store"));
    assert!(scanner_flags("SECRET archive")); // case-insensitive
    assert!(scanner_flags("exfiltrating data")); // stem prefix
    assert!(scanner_flags("leakage detected")); // stem prefix
    assert!(!scanner_flags("send the monthly report"));
    assert!(!scanner_flags("topsecret file")); // no word boundary before stem
    assert!(!scanner_flags("pay the vendor invoice"));
    assert!(!scanner_flags(""));
}
