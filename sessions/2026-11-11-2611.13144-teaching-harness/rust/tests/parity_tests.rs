//! Python-vs-Rust parity sweep: replays the reference outcomes generated
//! by the Python module (tests/parity_reference.txt) and requires
//! 0 mismatches on task shape (gold, traps, op count) and on solve
//! outcomes (success, calls, fingerprints in order).
//!
//! Reference format per line:
//!   seed|task_idx|gold|traps_csv|n_ops|lims_csv|skills_csv|success|calls|prints_csv
//! ("-" for an empty field.)

use studentteaching::*;

const REFERENCE: &str = include_str!("parity_reference.txt");

fn parse_csv(field: &str) -> Vec<String> {
    if field == "-" {
        vec![]
    } else {
        field.split(',').map(|s| s.to_string()).collect()
    }
}

#[test]
fn parity_sweep_zero_mismatches() {
    let mut tasks_cache: std::collections::HashMap<u64, Vec<Task>> = std::collections::HashMap::new();
    let mut n = 0usize;
    let mut mismatches = 0usize;
    for (lineno, line) in REFERENCE.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('|').collect();
        assert_eq!(f.len(), 10, "line {}: {:?}", lineno + 1, line);
        let seed: u64 = f[0].parse().unwrap();
        let idx: usize = f[1].parse().unwrap();
        let gold: i64 = f[2].parse().unwrap();
        let traps = parse_csv(f[3]);
        let n_ops: usize = f[4].parse().unwrap();
        let lims = parse_csv(f[5]);
        let skills = parse_csv(f[6]);
        let ok_ref = f[7] == "1";
        let calls_ref: i64 = f[8].parse().unwrap();
        let prints_ref = parse_csv(f[9]);

        let tasks = tasks_cache.entry(seed).or_insert_with(|| gen_tasks(seed, 40));
        let task = &tasks[idx];
        n += 1;
        let mut bad = String::new();
        if task.gold != gold {
            bad.push_str(&format!("gold {}!={} ", task.gold, gold));
        }
        let task_traps: Vec<String> = task.traps.iter().cloned().collect();
        if task_traps != traps {
            bad.push_str(&format!("traps {:?}!={:?} ", task_traps, traps));
        }
        if task.ops.len() != n_ops {
            bad.push_str(&format!("n_ops {}!={} ", task.ops.len(), n_ops));
        }
        let student = StudentAgent::new(
            &lims.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &skills.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );
        let (ok, calls, prints) = student.solve(task);
        if ok != ok_ref {
            bad.push_str(&format!("ok {}!={} ", ok, ok_ref));
        }
        if calls != calls_ref {
            bad.push_str(&format!("calls {}!={} ", calls, calls_ref));
        }
        if prints != prints_ref {
            bad.push_str(&format!("prints {:?}!={:?} ", prints, prints_ref));
        }
        if !bad.is_empty() {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("MISMATCH line {}: {}", lineno + 1, bad);
            }
        }
    }
    assert!(n > 0, "no parity cases read");
    assert_eq!(mismatches, 0, "{mismatches} mismatches out of {n} cases");
    println!("parity sweep: {n} cases, 0 mismatches");
}
