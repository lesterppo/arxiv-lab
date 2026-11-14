//! Python-vs-Rust parity sweep (arXiv:2611.16466).
//!
//! Replays `tests/parity_reference.txt` (1,600 records generated from the
//! frozen Python module: seeds 1/7/42/20261114 x travel/shopping x 50
//! instances x 4 strategies; per record: n_space, n_admitted, admission,
//! precision, f1, planner success). Integer fields and planner outcomes
//! must match exactly; floats within 1e-12 (Python's `sum()` uses
//! compensated summation, Rust's is naive — see README). 0 mismatches
//! required.

use reshape::{
    domain, gen_instances, plan, revise, space_metrics, strategy_seed, Rng,
};

const REFERENCE: &str = include_str!("parity_reference.txt");

#[test]
fn parity_sweep_zero_mismatches() {
    let mut mismatches = 0usize;
    let mut checked = 0usize;
    let mut cache: std::collections::HashMap<(u64, String), Vec<reshape::Instance>> =
        std::collections::HashMap::new();
    for (lineno, line) in REFERENCE.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 10, "line {}: {:?}", lineno + 1, line);
        let seed: u64 = f[0].parse().unwrap();
        let dom_name = f[1];
        let idx: usize = f[2].parse().unwrap();
        let strategy = f[3];
        let want_n_space: usize = f[4].parse().unwrap();
        let want_n_admitted: usize = f[5].parse().unwrap();
        let want_adm: f64 = f[6].parse().unwrap();
        let want_prec: f64 = f[7].parse().unwrap();
        let want_f1: f64 = f[8].parse().unwrap();
        let want_succ: u8 = f[9].parse().unwrap();

        let key = (seed, dom_name.to_string());
        let insts = cache.entry(key).or_insert_with(|| {
            gen_instances(domain(dom_name), 50, seed)
        });
        let inst = &insts[idx];
        let mut rng = Rng::new(strategy_seed(seed, idx, strategy));
        let space = revise(inst, strategy, &mut rng, reshape::SHOW_K, None);
        let m = space_metrics(inst, &space);
        let succ = if plan(inst, &space, &mut rng, reshape::PLANNER_K) {
            1u8
        } else {
            0u8
        };
        checked += 1;
        let mut bad = Vec::new();
        if m.n_space != want_n_space {
            bad.push(format!("n_space {} != {}", m.n_space, want_n_space));
        }
        if m.n_admitted != want_n_admitted {
            bad.push(format!("n_admitted {} != {}", m.n_admitted, want_n_admitted));
        }
        if (m.admission - want_adm).abs() > 1e-12 {
            bad.push(format!("admission {} != {}", m.admission, want_adm));
        }
        if (m.precision - want_prec).abs() > 1e-12 {
            bad.push(format!("precision {} != {}", m.precision, want_prec));
        }
        if (m.f1 - want_f1).abs() > 1e-12 {
            bad.push(format!("f1 {} != {}", m.f1, want_f1));
        }
        if succ != want_succ {
            bad.push(format!("planner_success {} != {}", succ, want_succ));
        }
        if !bad.is_empty() {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("line {} ({} {} {} {}): {}", lineno + 1, seed, dom_name, idx,
                          strategy, bad.join("; "));
            }
        }
    }
    assert!(checked > 0, "parity reference file is empty");
    assert_eq!(mismatches, 0, "{} mismatches in {} records", mismatches, checked);
    println!("parity: {} records, 0 mismatches", checked);
}
