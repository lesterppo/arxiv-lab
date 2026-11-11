//! `verify` binary: mirrors the Python CPU test's experiments and
//! assertions (test_sat_cpu.py). Prints the arm summaries, runs A1–A7,
//! writes results.json next to the binary's crate dir, exits 0 iff all
//! assertions pass. Usage: cargo run --release --bin verify

use studentteaching::*;
use std::fs;
use std::path::PathBuf;
use std::process::exit;

const SEED: u64 = 20261111;
const N_DEV: usize = 120;
const N_TEST: usize = 200;

fn check(name: &str, cond: bool, detail: &str, ok: &mut bool) {
    println!("  [{}] {} {}", if cond { "PASS" } else { "FAIL" }, name, detail);
    *ok &= cond;
}

fn fmt_rate(r: f64) -> String {
    format!("{:.3}", r)
}

fn main() {
    let mut ok = true;
    let dev_tasks = gen_tasks(SEED, N_DEV);
    let test_tasks = gen_tasks(SEED + 5000, N_TEST);
    let teacher = Teacher::default();

    let student_a = StudentAgent::new(&["L1", "L3", "L5"], &[]);
    let student_b = StudentAgent::new(&["L2", "L4"], &[]);

    let (acc_a0, _) = run_batch(&student_a, &test_tasks);
    let (acc_b0, _) = run_batch(&student_b, &test_tasks);

    let (lib_a, rep_a) = teacher.teach(&student_a, &dev_tasks);
    let sat_a = StudentAgent::new(
        &["L1", "L3", "L5"],
        &lib_a.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    let (acc_a_sat, _) = run_batch(&sat_a, &test_tasks);

    let lib_pf = profile_free_teach();
    let pf_a = StudentAgent::new(
        &["L1", "L3", "L5"],
        &lib_pf.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    let (acc_a_pf, _) = run_batch(&pf_a, &test_tasks);

    let (lib_po, _top_po, _rates_po) = profile_only_teach(&student_a, &dev_tasks, SEED);
    let po_a = StudentAgent::new(
        &["L1", "L3", "L5"],
        &lib_po.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    let (acc_a_po, _) = run_batch(&po_a, &test_tasks);

    let wrong: Vec<&str> = LIMITATIONS
        .iter()
        .copied()
        .filter(|l| !student_a.limitations.contains(*l))
        .collect();
    let (lib_mis, _rep_mis) = misleading_teach(&student_a, &dev_tasks, &wrong, SEED);

    let (lib_b, _rep_b) = teacher.teach(&student_b, &dev_tasks);
    let sat_b = StudentAgent::new(
        &["L2", "L4"],
        &lib_b.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    let (acc_b_sat, _) = run_batch(&sat_b, &test_tasks);
    let a_with_b = StudentAgent::new(
        &["L1", "L3", "L5"],
        &lib_b.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    let b_with_a = StudentAgent::new(
        &["L2", "L4"],
        &lib_a.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    let (acc_a_with_b, _) = run_batch(&a_with_b, &test_tasks);
    let (acc_b_with_a, _) = run_batch(&b_with_a, &test_tasks);

    // A3 probe accuracy
    let mut probe_ok = 0usize;
    let mut probe_tot = 0usize;
    for (lim, pr) in &rep_a.probes {
        probe_tot += 1;
        if pr.confirmed == student_a.limitations.contains(lim) {
            probe_ok += 1;
        }
    }
    let probe_acc = if probe_tot > 0 { probe_ok as f64 / probe_tot as f64 } else { 0.0 };

    // A7 determinism
    let dev2 = gen_tasks(SEED, N_DEV);
    let test2 = gen_tasks(SEED + 5000, N_TEST);
    let (lib_a2, _) = Teacher::default().teach(&StudentAgent::new(&["L1", "L3", "L5"], &[]), &dev2);
    let (acc_a_sat2, _) = run_batch(
        &StudentAgent::new(&["L1", "L3", "L5"], &lib_a2.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
        &test2,
    );

    println!("ARM SUCCESS RATES (n={} held-out tasks):", N_TEST);
    for (k, v) in [
        ("untaught_A", acc_a0),
        ("untaught_B", acc_b0),
        ("sat_A", acc_a_sat),
        ("sat_B", acc_b_sat),
        ("profile_free_A", acc_a_pf),
        ("profile_only_A", acc_a_po),
        ("A_with_B_library", acc_a_with_b),
        ("B_with_A_library", acc_b_with_a),
    ] {
        println!("  {:18} {}", k, fmt_rate(v));
    }
    println!("SAT-A library: {:?} | SAT-B library: {:?}", lib_a, lib_b);
    println!(
        "profile rates A: {:?}",
        rep_a.profile_rates.iter().map(|(k, v)| (k.clone(), fmt_rate(*v))).collect::<Vec<_>>()
    );

    check(
        "A1 sat>profile_free>untaught (A)",
        acc_a_sat > acc_a_pf && acc_a_pf > acc_a0,
        &format!("sat={} pf={} untaught={}", fmt_rate(acc_a_sat), fmt_rate(acc_a_pf), fmt_rate(acc_a0)),
        &mut ok,
    );
    check("A2 misleading admits 0", lib_mis.is_empty(), &format!("admitted={:?}", lib_mis), &mut ok);
    check(
        "A3 probe accuracy>=0.80",
        probe_acc >= 0.80,
        &format!("{}/{}", probe_ok, probe_tot),
        &mut ok,
    );
    check(
        "A4 exchange: A own>B lib",
        acc_a_sat > acc_a_with_b,
        &format!("own={} other={}", fmt_rate(acc_a_sat), fmt_rate(acc_a_with_b)),
        &mut ok,
    );
    check(
        "A4 exchange: B own>A lib",
        acc_b_sat > acc_b_with_a,
        &format!("own={} other={}", fmt_rate(acc_b_sat), fmt_rate(acc_b_with_a)),
        &mut ok,
    );
    check(
        "A5 profile-only < full SAT",
        acc_a_po < acc_a_sat,
        &format!("po={} sat={}", fmt_rate(acc_a_po), fmt_rate(acc_a_sat)),
        &mut ok,
    );
    let admitted_ok = rep_a
        .admissions
        .values()
        .filter(|r| r.admitted)
        .all(|r| r.delta >= 0.15)
        && rep_a.admissions.values().filter(|r| r.admitted).count() == lib_a.len();
    check("A6 admitted deltas>=0.15", admitted_ok, "", &mut ok);
    check(
        "A7 deterministic rerun",
        lib_a2 == lib_a && acc_a_sat2 == acc_a_sat,
        &format!("lib {:?} vs {:?}; acc {} vs {}", lib_a, lib_a2, fmt_rate(acc_a_sat), fmt_rate(acc_a_sat2)),
        &mut ok,
    );

    // results.json (hand-rolled; no serde in this crate)
    let mut arms = String::new();
    for (k, v) in [
        ("untaught_A", acc_a0),
        ("untaught_B", acc_b0),
        ("sat_A", acc_a_sat),
        ("sat_B", acc_b_sat),
        ("profile_free_A", acc_a_pf),
        ("profile_only_A", acc_a_po),
        ("A_with_B_library", acc_a_with_b),
        ("B_with_A_library", acc_b_with_a),
    ] {
        arms.push_str(&format!("\"{}\": {}, ", k, v));
    }
    let out = format!(
        "{{\n  \"seed\": {},\n  \"n_dev\": {},\n  \"n_test\": {},\n  \"arms\": {{{}}},\n  \"sat_A_library\": {:?},\n  \"sat_B_library\": {:?},\n  \"misleading_A_admitted\": {:?},\n  \"probe_accuracy_A\": {},\n  \"assertions_passed\": {}\n}}\n",
        SEED,
        N_DEV,
        N_TEST,
        arms.trim_end_matches(", "),
        lib_a,
        lib_b,
        lib_mis,
        probe_acc,
        ok
    );
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results.json");
    fs::write(&path, out).expect("write results.json");
    println!("results.json written to {}", path.display());

    if !ok {
        println!("ASSERTIONS FAILED");
        exit(1);
    }
    println!("ALL ASSERTIONS PASSED");
}
