//! RuntimeTickets (arXiv 2611.03772), std-only Rust port of the Python
//! mechanism in `../runtime_tickets.py`.
//!
//! Ported, pure logic only: obligation grounding, ticket tightening
//! (never loosens host approvals), the runtime gate over
//! (recipient, object, amount) sends, trace running, the deterministic
//! decoder, the regex-scanner baseline, and the coarse allow-list
//! baseline. The paper's LLM obligation-proposal step is out of scope
//! in both implementations (obligations are given as data).

use std::collections::BTreeMap;

/// One outbound send attempt. Amount is an integer in minor units.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Send {
    pub recipient: String,
    pub object: String,
    pub amount: i64,
}

impl Send {
    pub fn new(recipient: &str, object: &str, amount: i64) -> Self {
        Send { recipient: recipient.to_string(), object: object.to_string(), amount }
    }
}

/// One security obligation: id, cited skill lines, allowed sends
/// (recipient, object, amount_cap).
#[derive(Debug, Clone)]
pub struct Obligation {
    pub oid: String,
    pub cited_lines: Vec<usize>,
    pub allowed: Vec<(String, String, i64)>,
}

impl Obligation {
    pub fn new(oid: &str, cited_lines: Vec<usize>, allowed: Vec<(&str, &str, i64)>) -> Self {
        Obligation {
            oid: oid.to_string(),
            cited_lines,
            allowed: allowed.into_iter().map(|(r, o, c)| (r.to_string(), o.to_string(), c)).collect(),
        }
    }

    /// True iff every cited line index exists in a skill source of
    /// `n_skill_lines` lines.
    pub fn is_grounded(&self, n_skill_lines: usize) -> bool {
        self.cited_lines.iter().all(|&i| i < n_skill_lines)
    }
}

/// A ticket: obligations + the tightened allowed map
/// (recipient, object) -> amount cap.
#[derive(Debug, Clone)]
pub struct Ticket {
    pub obligations: Vec<Obligation>,
    pub allowed: BTreeMap<(String, String), i64>,
}

impl Ticket {
    /// Build a ticket, refusing any loosening of `host_approvals`:
    /// a new (recipient, object) pair, or a cap above the host's cap.
    /// On success each pair's cap is the minimum over obligations
    /// claiming it (tightening to an equal cap is allowed).
    pub fn tighten(
        host_approvals: &BTreeMap<(String, String), i64>,
        obligations: Vec<Obligation>,
    ) -> Result<Ticket, String> {
        let mut allowed: BTreeMap<(String, String), i64> = BTreeMap::new();
        for ob in &obligations {
            for (r, o, cap) in &ob.allowed {
                let key = (r.clone(), o.clone());
                match host_approvals.get(&key) {
                    None => {
                        return Err(format!("ticket loosens host approvals: new pair {:?}", key));
                    }
                    Some(&host_cap) => {
                        if *cap > host_cap {
                            return Err(format!(
                                "ticket loosens host approvals: cap {} > host {} for {:?}",
                                cap, host_cap, key
                            ));
                        }
                    }
                }
                let entry = allowed.entry(key).or_insert(*cap);
                if *cap < *entry {
                    *entry = *cap;
                }
            }
        }
        Ok(Ticket { obligations, allowed })
    }

    /// The runtime gate: pair allowed and amount within cap. Reads
    /// nothing but the ticket, so skill text cannot change it.
    pub fn gate(&self, send: &Send) -> bool {
        match self.allowed.get(&(send.recipient.clone(), send.object.clone())) {
            Some(&cap) => send.amount <= cap,
            None => false,
        }
    }

    /// Independent oracle for the gate (same predicate, separate code
    /// path: iterate the map instead of a direct lookup).
    pub fn oracle_allows(&self, send: &Send) -> bool {
        for ((r, o), cap) in &self.allowed {
            if r == &send.recipient && o == &send.object {
                return send.amount <= *cap;
            }
        }
        false
    }
}

/// Apply the gate to each attempted send: (send, forwarded) trace.
pub fn run_trace(ticket: &Ticket, sends: &[Send]) -> Vec<(Send, bool)> {
    sends.iter().map(|s| (s.clone(), ticket.gate(s))).collect()
}

/// Deterministic decoder over (send, forwarded, effect_recorded)
/// triples. Returns (verdict, violations); "verified" iff every
/// obligation is grounded, every forwarded send is within the ticket,
/// and no blocked send has a recorded effect.
pub fn decode(
    ticket: &Ticket,
    trace: &[(Send, bool, bool)],
    obligations_grounded: bool,
) -> (String, Vec<String>) {
    let mut violations: Vec<String> = Vec::new();
    if !obligations_grounded {
        violations.push("ungrounded_obligation".to_string());
    }
    for (send, forwarded, effect) in trace {
        if *forwarded && !ticket.gate(send) {
            violations.push(format!("forwarded_outside_ticket:{:?}", send));
        }
        if !*forwarded && *effect {
            violations.push(format!("effect_on_blocked_send:{:?}", send));
        }
    }
    if violations.is_empty() {
        ("verified".to_string(), violations)
    } else {
        ("violated".to_string(), violations)
    }
}

/// Decode a gate-produced trace: effects exist exactly for forwarded
/// sends, so only ungrounded obligations can violate.
pub fn decode_gate_trace(
    ticket: &Ticket,
    trace: &[(Send, bool)],
    n_skill_lines: usize,
) -> (String, Vec<String>) {
    let grounded = ticket.obligations.iter().all(|ob| ob.is_grounded(n_skill_lines));
    let full: Vec<(Send, bool, bool)> =
        trace.iter().map(|(s, f)| (s.clone(), *f, *f)).collect();
    decode(ticket, &full, grounded)
}

const SCANNER_STEMS: [&str; 7] =
    ["exfiltrat", "attacker", "secret", "credential", "password", "steal", "leak"];

/// Baseline regex-scanner analogue: a description is flagged iff some
/// alphanumeric token starts with a suspicious stem (the Python regex
/// is `\b(stem\w*)`, i.e. a word-boundary stem prefix match).
pub fn scanner_flags(description: &str) -> bool {
    let lower = description.to_lowercase();
    let mut token = String::new();
    let check = |token: &str| -> bool {
        SCANNER_STEMS.iter().any(|stem| token.starts_with(stem))
    };
    for ch in lower.chars() {
        if ch.is_alphanumeric() {
            token.push(ch);
        } else if !token.is_empty() {
            if check(&token) {
                return true;
            }
            token.clear();
        }
    }
    !token.is_empty() && check(&token)
}

/// Baseline coarse allow-list: any send up to the largest host cap,
/// ignoring per-pair approvals.
pub fn coarse_allowlist_allows(send: &Send, host_approvals: &BTreeMap<(String, String), i64>) -> bool {
    match host_approvals.values().max() {
        Some(&cap) => send.amount <= cap,
        None => false,
    }
}

/// The catalog ticket shared by `verify` and the tests.
pub fn catalog_ticket() -> (Ticket, BTreeMap<(String, String), i64>) {
    let mut host: BTreeMap<(String, String), i64> = BTreeMap::new();
    host.insert(("alice@example.com".to_string(), "payslip.pdf".to_string()), 0);
    host.insert(("bob@example.com".to_string(), "report.pdf".to_string()), 0);
    host.insert(("vendor@example.com".to_string(), "invoice-42".to_string()), 50000);
    host.insert(("backup@example.com".to_string(), "backup.tar".to_string()), 0);
    host.insert(("team@example.com".to_string(), "notes.md".to_string()), 0);
    let obligations = vec![
        Obligation::new("ob-payslip", vec![0], vec![("alice@example.com", "payslip.pdf", 0)]),
        Obligation::new("ob-report", vec![1], vec![("bob@example.com", "report.pdf", 0)]),
        Obligation::new("ob-invoice", vec![2], vec![("vendor@example.com", "invoice-42", 50000)]),
        Obligation::new("ob-backup", vec![3], vec![("backup@example.com", "backup.tar", 0)]),
        Obligation::new("ob-notes", vec![4], vec![("team@example.com", "notes.md", 0)]),
    ];
    let ticket = Ticket::tighten(&host, obligations).expect("catalog ticket tightens");
    (ticket, host)
}

/// The 10 benign + 10 malicious catalog sends (same as Python).
pub fn catalog_sends() -> (Vec<Send>, Vec<Send>) {
    let benign = vec![
        Send::new("alice@example.com", "payslip.pdf", 0),
        Send::new("bob@example.com", "report.pdf", 0),
        Send::new("vendor@example.com", "invoice-42", 25000),
        Send::new("vendor@example.com", "invoice-42", 50000),
        Send::new("backup@example.com", "backup.tar", 0),
        Send::new("team@example.com", "notes.md", 0),
        Send::new("alice@example.com", "payslip.pdf", 0),
        Send::new("bob@example.com", "report.pdf", 0),
        Send::new("vendor@example.com", "invoice-42", 100),
        Send::new("team@example.com", "notes.md", 0),
    ];
    let malicious = vec![
        Send::new("attacker@evil.example", "payslip.pdf", 0),
        Send::new("alice@example.com", "credentials.json", 0),
        Send::new("vendor@example.com", "invoice-42", 50001),
        Send::new("vendor@example.com", "invoice-42", 999999),
        Send::new("attacker@evil.example", "invoice-42", 100),
        Send::new("vendor@example.com", "secrets.db", 0),
        Send::new("mallory@example.com", "report.pdf", 0),
        Send::new("attacker@evil.example", "backup.tar", 0),
        Send::new("alice@example.com", "invoice-42", 50000),
        Send::new("bob@example.com", "credentials.json", 0),
    ];
    (benign, malicious)
}
