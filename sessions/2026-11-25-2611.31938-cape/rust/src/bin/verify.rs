//! `verify` binary: mirrors the Python CPU test's experiments and
//! assertions (tests/test_cape_cpu.py), on Rust-native worlds.
//!
//! Worlds are generated with the crate's xorshift RNG, not Python's
//! `random.Random`, so the *numbers* differ from Python's while the
//! *properties* asserted are the same: optimal value rate, R0
//! preservation (Prop. 1), certification rate, completion gain vs the
//! error-feedback reviser proxy. Exact cross-language equality is the
//! parity sweep's job (tests/parity_cases.txt).
//!
//! Usage: `cargo run --release --bin verify`

use cape::{
    compile_policy, error_feedback_policy, exact_optimum, make_world,
    program_only_policy, PAPER_COMPLETION_GAIN_PP,
};
use std::time::Instant;

struct Row {
    seed: u64,
    n_configs: usize,
    cap: usize,
    budget: usize,
    cape_comp: usize,
    cape_calls: usize,
    opt_comp: usize,
    opt_calls: i64,
    optimal_value: bool,
    certified: bool,
    queries: usize,
    table: usize,
    r0: usize,
    r0_kept: usize,
    ef_comp: usize,
    ef_gain_pp: f64,
}

fn run_world(n_configs: usize, cap: usize, budget: usize, seed: u64) -> Row {
    let w = make_world(n_configs, 4, seed);
    let t0 = Instant::now();
    let r = compile_policy(&w, cap, budget);
    let dt = t0.elapsed().as_secs_f64();
    let (opt, _) = exact_optimum(&w, cap);
    let opt = opt.expect("exact optimum");
    let (ef_c, _) = error_feedback_policy(&w, cap);
    let (po, _) = program_only_policy(&w, cap);
    let table = w.actions().len() * w.configs.len();
    let row = Row {
        seed,
        n_configs,
        cap,
        budget,
        cape_comp: r.completed.len(),
        cape_calls: r.total_calls,
        opt_comp: opt.0,
        opt_calls: opt.1,
        optimal_value: r.value == opt,
        certified: r.certified,
        queries: r.queries,
        table,
        r0: r.r0.len(),
        r0_kept: r.r0.intersection(&r.completed).count(),
        ef_comp: ef_c.len(),
        ef_gain_pp: (r.completed.len() as f64 - ef_c.len() as f64) / n_configs as f64 * 100.0,
    };
    println!(
        "  world n={} cap={} seed={}: cape=({},{}) exact=({},{}) cert={} queries={}/{} ef={} gain={:.1}pp {:.1}s",
        n_configs, cap, seed, row.cape_comp, -(row.cape_calls as i64), row.opt_comp,
        row.opt_calls, row.certified, row.queries, table, row.ef_comp, row.ef_gain_pp, dt
    );
    // Consistency (E4 of the Python test).
    assert_eq!(r.value, (r.completed.len(), -(r.total_calls as i64)), "value mismatch");
    assert!(r.r0.is_subset(&r.completed), "R0 not preserved");
    assert!(r.queries <= budget, "budget exceeded");
    if r.certified {
        assert_eq!(r.value, r.upper.unwrap(), "certified but value != upper");
    }
    let _ = po;
    row
}

fn main() {
    println!("E1 main suite: 12 configs, cap 3, budget 1000, seeds 0..7");
    let mut e1 = Vec::new();
    for s in 0..8u64 {
        e1.push(run_world(12, 3, 1000, s));
    }
    let opt1 = e1.iter().filter(|r| r.optimal_value).count();
    let cert1 = e1.iter().filter(|r| r.certified).count();
    let saving1 = 1.0 - e1.iter().map(|r| r.queries).sum::<usize>() as f64 / e1.iter().map(|r| r.table).sum::<usize>() as f64;
    let ef1 = e1.iter().map(|r| r.ef_gain_pp).sum::<f64>() / e1.len() as f64;
    println!(
        "E1: optimal value {}/8, certified {}/8, mean query saving {:.1}% (paper {:.1}%), mean EF gain {:.2}pp (paper {:.2}-{:.2}pp)",
        opt1, cert1, saving1 * 100.0, cape::PAPER_TIME_SAVING * 100.0, ef1,
        PAPER_COMPLETION_GAIN_PP.0, PAPER_COMPLETION_GAIN_PP.1
    );
    assert_eq!(opt1, 8, "E1: CAPE must reach the exact optimum value");
    assert!(e1.iter().all(|r| r.r0_kept == r.r0), "E1: R0 must be kept");
    assert!(cert1 >= 6, "E1: certification rate below 6/8");
    assert!(ef1 > PAPER_COMPLETION_GAIN_PP.0, "E1: EF gain below the paper's range");

    println!("E2 small worlds: 8 configs, cap 3, budget 400, seeds 0..3");
    let mut e2 = Vec::new();
    for s in 0..4u64 {
        e2.push(run_world(8, 3, 400, s));
    }
    let opt2 = e2.iter().filter(|r| r.optimal_value).count();
    let cert2 = e2.iter().filter(|r| r.certified).count();
    println!("E2: optimal value {}/4, certified {}/4", opt2, cert2);
    assert_eq!(opt2, 4, "E2: must reach optimum");
    assert!(cert2 >= 2, "E2: certification too low");

    println!("E3 scale spot check: 24 configs, cap 3, budget 2000, seed 0");
    let e3 = run_world(24, 3, 2000, 0);
    assert!(e3.optimal_value, "E3: must reach the optimum value");
    println!("E3: optimal value reached, certified={}, queries={}/{}", e3.certified, e3.queries, e3.table);

    println!(
        "RESULTS_JSON {{\"e1_optimal_value\": \"{}/8\", \"e1_certified\": \"{}/8\", \
         \"e1_mean_query_saving\": {:.4}, \"e1_mean_ef_gain_pp\": {:.2}, \
         \"e2_optimal_value\": \"{}/4\", \"e2_certified\": \"{}/4\", \
         \"e3_optimal_value\": {}, \"e3_certified\": {}}}",
        opt1, cert1, saving1, ef1, opt2, cert2, e3.optimal_value, e3.certified
    );
    println!("ALL CHECKS PASSED");
}
