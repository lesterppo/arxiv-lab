//! signed_manifest CLI: `verify` reproduces the Python experiment's
//! headline claims with a Rust-side PRNG stream (claim-level, not
//! bit-identical numbers); `parity` answers explicit cases from stdin
//! for the Python-vs-Rust sweep (byte-exact, 0 mismatches required).

use signed_manifest::*;
use std::io::{self, BufRead};

fn parse_format(s: &str) -> Option<Format> {
    match s {
        "free_text" => Some(Format::FreeText),
        "json_unsigned" => Some(Format::JsonUnsigned),
        "signed_manifest" => Some(Format::SignedManifest),
        _ => None,
    }
}

fn parse_threat(s: &str) -> Option<Threat> {
    match s {
        "none" => Some(Threat::None),
        "drop" => Some(Threat::Drop),
        "alter" => Some(Threat::Alter),
        "ignore" => Some(Threat::Ignore),
        "mutate" => Some(Threat::Mutate),
        "inject" => Some(Threat::Inject),
        _ => None,
    }
}

/// xorshift64* PRNG (verify-command stream only; parity cases are explicit).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
    /// Sample k distinct ids out of REQUIREMENTS (Fisher-Yates prefix).
    fn sample_ids(&mut self, k: usize) -> Vec<&'static str> {
        let mut ids: Vec<&'static str> =
            REQUIREMENTS.iter().map(|r| r.id).collect();
        for i in 0..k {
            let j = i + self.below(ids.len() - i);
            ids.swap(i, j);
        }
        ids.truncate(k);
        ids
    }
}

#[derive(Clone, Copy)]
struct Cell {
    n: usize,
    unsafe_c: usize,
    conformant_c: usize,
    refused_c: usize,
}

fn run_verify() -> bool {
    let mut rng = Rng(0x20261106);
    let formats = [Format::FreeText, Format::JsonUnsigned, Format::SignedManifest];
    let format_names = ["free_text", "json_unsigned", "signed_manifest"];
    let threats = [
        Threat::None,
        Threat::Drop,
        Threat::Alter,
        Threat::Ignore,
        Threat::Mutate,
        Threat::Inject,
    ];
    let threat_names = ["none", "drop", "alter", "ignore", "mutate", "inject"];
    let n_per_cell = 60usize;
    // cells[fmt][threat]
    let mut cells = vec![vec![Cell { n: 0, unsafe_c: 0, conformant_c: 0, refused_c: 0 };
                          threats.len()];
                         formats.len()];
    // verifier tallies: (violating, detected) and (conformant, false_alarms)
    let mut ver: Vec<(usize, usize, usize, usize)> = vec![(0, 0, 0, 0); formats.len()];
    for (fi, fmt) in formats.iter().enumerate() {
        for (ti, threat) in threats.iter().enumerate() {
            for _ in 0..n_per_cell {
                let mut ids = rng.sample_ids(4);
                // Mirror Python's make_task order (sample order, unsorted).
                let inject = *threat == Threat::Inject;
                let text = build_task_text(&ids, inject);
                let truth: Vec<Requirement> =
                    ids.iter().filter_map(|id| by_id(id).cloned()).collect();
                // Full trajectory via the explicit runner with a random victim.
                let victim = rng.below(4);
                let out = run_case_explicit(&ids, *fmt, *threat, victim, inject);
                // Parse the canonical result string.
                let mut refused = false;
                let mut committed = false;
                let mut unsafe_ = false;
                for part in out.split('|') {
                    if let Some(v) = part.strip_prefix("refused=") {
                        refused = v == "1";
                    } else if let Some(v) = part.strip_prefix("committed=") {
                        committed = v == "1";
                    } else if let Some(v) = part.strip_prefix("unsafe=") {
                        unsafe_ = v == "1";
                    }
                }
                let _ = (text, truth);
                let cell = &mut cells[fi][ti];
                cell.n += 1;
                if unsafe_ {
                    cell.unsafe_c += 1;
                }
                if committed && !unsafe_ {
                    cell.conformant_c += 1;
                }
                if refused {
                    cell.refused_c += 1;
                }
                // Verifier tallies from the same outcome string.
                let mut detected = false;
                let mut false_alarm = false;
                for part in out.split('|') {
                    if let Some(v) = part.strip_prefix("detected=") {
                        detected = v != "-";
                    } else if let Some(v) = part.strip_prefix("false_alarms=") {
                        false_alarm = v != "-";
                    }
                }
                let v = &mut ver[fi];
                if unsafe_ || (!committed && !refused) {
                    // A violating decision (blocked or committed-unsafe).
                    v.0 += 1;
                    if detected {
                        v.1 += 1;
                    }
                } else if committed {
                    v.2 += 1;
                    if false_alarm {
                        v.3 += 1;
                    }
                }
                ids.clear();
            }
        }
    }
    let rate = |c: &Cell| c.unsafe_c as f64 / c.n as f64;
    let mut under_attack = [0.0f64; 3];
    for fi in 0..3 {
        let mut sum = 0.0;
        for ti in 1..6 {
            sum += rate(&cells[fi][ti]);
        }
        under_attack[fi] = sum / 5.0;
    }
    println!("unsafe-commit rate under attack (mean of 5 threats):");
    for fi in 0..3 {
        println!("  {:<15} {:.3}", format_names[fi], under_attack[fi]);
    }
    println!("\nper-cell unsafe-commit rates:");
    for fi in 0..3 {
        for ti in 0..6 {
            let c = &cells[fi][ti];
            println!(
                "  {:<15} {:<6} unsafe={:.3} conformant={:.3} refused={:.3}",
                format_names[fi],
                threat_names[ti],
                rate(c),
                c.conformant_c as f64 / c.n as f64,
                c.refused_c as f64 / c.n as f64
            );
        }
    }
    println!("\nverifier comparison:");
    let mut det = [0.0f64; 3];
    let mut fa = [0.0f64; 3];
    for fi in 0..3 {
        let (nv, nd, nc, nf) = ver[fi];
        det[fi] = if nv > 0 { nd as f64 / nv as f64 } else { 0.0 };
        fa[fi] = if nc > 0 { nf as f64 / nc as f64 } else { 0.0 };
        println!(
            "  {:<15} detection={:.3} (n={}) false_alarm={:.3} (n={})",
            format_names[fi], det[fi], nv, fa[fi], nc
        );
    }
    let mut ok = true;
    let claim = |ok: &mut bool, name: &str, cond: bool, detail: String| {
        println!("{} {} | {}", if cond { "PASS" } else { "FAIL" }, name, detail);
        if !cond {
            *ok = false;
        }
    };
    claim(
        &mut ok,
        "C1: signed unsafe-under-attack << free-text",
        under_attack[2] + 0.15 < under_attack[0],
        format!("signed={:.3} free_text={:.3}", under_attack[2], under_attack[0]),
    );
    claim(
        &mut ok,
        "C1: signed unsafe-under-attack < 0.25",
        under_attack[2] < 0.25,
        format!("signed={:.3}", under_attack[2]),
    );
    claim(
        &mut ok,
        "C2: signed commits nothing under mutation",
        rate(&cells[2][4]) == 0.0 && cells[2][4].refused_c == cells[2][4].n,
        format!(
            "unsafe={:.3} refused={}/{}",
            rate(&cells[2][4]),
            cells[2][4].refused_c,
            cells[2][4].n
        ),
    );
    claim(
        &mut ok,
        "C2: unsigned commits violations under mutation",
        rate(&cells[1][4]) > 0.3,
        format!("unsigned mutate unsafe={:.3}", rate(&cells[1][4])),
    );
    claim(
        &mut ok,
        "C3: deterministic verifier detects more",
        det[2] > det[1],
        format!("det {:.3} vs {:.3}", det[2], det[1]),
    );
    claim(
        &mut ok,
        "C3: deterministic verifier false-alarms less",
        fa[2] < fa[1],
        format!("fa {:.3} vs {:.3}", fa[2], fa[1]),
    );
    claim(
        &mut ok,
        "C4: injection raises signed violations",
        rate(&cells[2][5]) > rate(&cells[2][0]),
        format!(
            "inject={:.3} none={:.3}",
            rate(&cells[2][5]),
            rate(&cells[2][0])
        ),
    );
    claim(
        &mut ok,
        "no net loss of conformant commits (signed vs json, no attack)",
        cells[2][0].conformant_c as f64 / cells[2][0].n as f64
            >= cells[1][0].conformant_c as f64 / cells[1][0].n as f64 - 0.05,
        format!(
            "signed={:.3} json={:.3}",
            cells[2][0].conformant_c as f64 / cells[2][0].n as f64,
            cells[1][0].conformant_c as f64 / cells[1][0].n as f64
        ),
    );
    if ok {
        println!("VERIFY PASSED");
    } else {
        println!("VERIFY FAILED");
    }
    ok
}

fn run_parity() {
    let stdin = io::stdin();
    let mut out = String::new();
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        if line.trim().is_empty() {
            continue;
        }
        // req1,req2|format|threat|victim|inject
        let parts: Vec<&str> = line.split('|').collect();
        assert_eq!(parts.len(), 5, "bad parity line: {}", line);
        let req_ids: Vec<&str> = parts[0].split(',').collect();
        let fmt = parse_format(parts[1]).expect("bad format");
        let threat = parse_threat(parts[2]).expect("bad threat");
        let victim: usize = parts[3].parse().expect("bad victim");
        let inject = parts[4] == "1";
        let res = run_case_explicit(&req_ids, fmt, threat, victim, inject);
        out.push_str(&res);
        out.push('\n');
    }
    print!("{}", out);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("verify") => {
            std::process::exit(if run_verify() { 0 } else { 1 });
        }
        Some("parity") => run_parity(),
        _ => {
            eprintln!("usage: signed_manifest <verify|parity>");
            std::process::exit(2);
        }
    }
}
