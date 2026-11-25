//! Trace CAPE acquisitions for one parity case (Rust side).
//! Usage: cargo run --release --bin trace -- <line-number> [max-rounds]
//! Prints the same format as /tmp/trace_cape.py for direct diffing.

use cape::*;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let line_no: usize = args[1].parse().expect("line number");
    let max_rounds: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
    let text = fs::read_to_string("tests/parity_cases.txt").expect("parity_cases.txt");
    let lines: Vec<&str> = text.lines().collect();
    let parts: Vec<&str> = lines[line_no - 1].split_whitespace().collect();
    let n_aspects: usize = parts[0].parse().unwrap();
    let cap: usize = parts[1].parse().unwrap();
    let budget: usize = parts[2].parse().unwrap();
    let aspects: Vec<Aspect> = if n_aspects == 4 {
        DEFAULT_ASPECTS.to_vec()
    } else {
        EXTENDED_ASPECTS.to_vec()
    };
    let configs: Vec<Config> = parts[3]
        .split(',')
        .map(|s| s.bytes().map(|b| b - b'0').collect())
        .collect();
    let mut programs: Vec<Config> = vec![vec![]];
    for a in &aspects {
        let mut next = Vec::new();
        for prefix in &programs {
            for v in 0..a.values.len() as u8 {
                let mut c = prefix.clone();
                c.push(v);
                next.push(c);
            }
        }
        programs = next;
    }
    let mut probes = Vec::new();
    for pi in 0..aspects.len() {
        for ri in 0..aspects.len() {
            if ri == pi {
                continue;
            }
            for v in 0..aspects[pi].values.len() as u8 {
                probes.push((pi, v, ri));
            }
        }
    }
    let world = World { aspects, configs, programs, probes };
    let r = compile_policy(&world, cap, budget);
    let (opt, _) = exact_optimum_with(&world, cap, &world.actions(), None);
    println!(
        "line {}: cap={} budget={} ncfg={}",
        line_no,
        cap,
        budget,
        world.configs.len()
    );
    println!(
        "result value=({:?}) queries={} certified={} completed={} calls={}",
        r.value, r.queries, r.certified, r.completed.len(), r.total_calls
    );
    println!("opt={:?}", opt);
    println!("rust recorded acquisitions={} (loop only)", r.acquisitions.len());
    let cfg_str = |c: &Config| -> String {
        c.iter().map(|v| (b'0' + *v) as char).collect()
    };
    for (i, (a, h)) in r.acquisitions.iter().enumerate().take(max_rounds) {
        let astr = match a {
            Act::Prog(enc) => format!("prog:{}", cfg_str(enc)),
            Act::Probe(pi, pv, ri) => format!("probe:{},{},{}", pi, pv, ri),
        };
        println!("  {:3} {} -> {}", i, astr, cfg_str(h));
    }
}
