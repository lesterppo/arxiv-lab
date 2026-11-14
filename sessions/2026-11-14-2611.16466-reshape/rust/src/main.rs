//! `verify` binary: mirrors the Python test's experiments and assertions
//! (A1-A13) against the Rust port, and writes `results.json` in the
//! current directory. Run from the `rust/` directory:
//!   cargo run --release --bin verify

use reshape::{
    domain, gen_instances, gen_instances_with_bonus, pearson, plan, revise, revise_oracle,
    run_experiment, space_metrics, strategy_seed, ExperimentOptions, Rng, STRATEGIES,
};
use std::collections::BTreeMap;
use std::fs;

const SEED: u64 = 20261114;
const N_INSTANCES: usize = 300;

fn main() {
    let mut passed: Vec<String> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    let mut check = |name: &str, cond: bool, detail: String| {
        if cond {
            passed.push(name.to_string());
        } else {
            failed.push(name.to_string());
        }
        println!("  [{}] {} {}", if cond { "PASS" } else { "FAIL" }, name, detail);
    };

    let mut results: BTreeMap<&str, BTreeMap<&'static str, reshape::Agg>> = BTreeMap::new();
    let mut instances_map: BTreeMap<&str, Vec<reshape::Instance>> = BTreeMap::new();
    for dom_name in ["travel", "shopping"] {
        let dom = domain(dom_name);
        let insts = gen_instances(dom, N_INSTANCES, SEED);
        let res = run_experiment(&insts, SEED, &ExperimentOptions::default());
        instances_map.insert(dom_name, insts);
        results.insert(dom_name, res);
    }
    println!(
        "main experiment ({} instances/domain, seed {}, planner K={}, show K={}):",
        N_INSTANCES,
        SEED,
        reshape::PLANNER_K,
        reshape::SHOW_K
    );
    for dom_name in ["travel", "shopping"] {
        for s in STRATEGIES {
            let m = &results[dom_name][s];
            println!(
                "  {:9} {:7} adm={:.3} prec={:.3} f1={:.3} succ={:.3} |S|={:.1}",
                dom_name, s, m.admission, m.precision, m.f1, m.planner_success,
                m.mean_space_size
            );
        }
    }

    println!("assertions:");
    let tr = &results["travel"];
    let sh = &results["shopping"];
    let pooled = |s: &str, f: fn(&reshape::Agg) -> f64| -> f64 {
        (f(&results["travel"][s]) + f(&results["shopping"][s])) / 2.0
    };
    check(
        "A1 travel F1_SHOW - F1_ASK >= 0.05",
        tr["show"].f1 - tr["ask"].f1 >= 0.05,
        format!("got {:.3} - {:.3} = {:.3}", tr["show"].f1, tr["ask"].f1,
                tr["show"].f1 - tr["ask"].f1),
    );
    check(
        "A2 travel admission_SHOW > admission_ASK",
        tr["show"].admission > tr["ask"].admission,
        format!("got {:.3} vs {:.3}", tr["show"].admission, tr["ask"].admission),
    );
    check(
        "A3 travel planner SHOW > ASK, lift >= 1.15",
        tr["show"].planner_success >= 1.15 * tr["ask"].planner_success,
        format!(
            "got {:.3} vs {:.3} (lift {:.2}x)",
            tr["show"].planner_success,
            tr["ask"].planner_success,
            tr["show"].planner_success / tr["ask"].planner_success
        ),
    );
    check(
        "A4 shopping F1_SHOW - F1_ASK >= 0.03",
        sh["show"].f1 - sh["ask"].f1 >= 0.03,
        format!("got {:.3} - {:.3} = {:.3}", sh["show"].f1, sh["ask"].f1,
                sh["show"].f1 - sh["ask"].f1),
    );
    check(
        "A5 pooled ORACLE admission > 0.52",
        pooled("oracle", |m| m.admission) > 0.52,
        format!("got {:.3}", pooled("oracle", |m| m.admission)),
    );
    let adm = |s: &str| pooled(s, |m| m.admission);
    let f1 = |s: &str| pooled(s, |m| m.f1);
    let succ = |s: &str| pooled(s, |m| m.planner_success);
    check(
        "A6a pooled admission: none < ask < show < oracle",
        adm("none") < adm("ask") && adm("ask") < adm("show") && adm("show") < adm("oracle"),
        format!("got [{:.3}, {:.3}, {:.3}, {:.3}]", adm("none"), adm("ask"),
                adm("show"), adm("oracle")),
    );
    check(
        "A6b pooled F1: show > ask and oracle >= show",
        f1("show") > f1("ask") && f1("oracle") >= f1("show"),
        format!("got [{:.3}, {:.3}, {:.3}, {:.3}]", f1("none"), f1("ask"),
                f1("show"), f1("oracle")),
    );
    check(
        "A7a pooled planner SHOW > ASK, lift >= 1.15",
        succ("show") >= 1.15 * succ("ask"),
        format!("got {:.3} vs {:.3} (lift {:.2}x)", succ("show"), succ("ask"),
                succ("show") / succ("ask")),
    );
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for dom_name in ["travel", "shopping"] {
        for s in STRATEGIES {
            xs.push(results[dom_name][s].f1);
            ys.push(results[dom_name][s].planner_success);
        }
    }
    let r = pearson(&xs, &ys);
    check(
        "A7b F1 predicts planner success: Pearson r > 0.8",
        r > 0.8,
        format!("got r={:.3}", r),
    );
    check(
        "A8a pooled planner ASK < NONE (asking dilutes the space)",
        succ("ask") < succ("none"),
        format!("got ask {:.3} vs none {:.3}", succ("ask"), succ("none")),
    );
    check(
        "A8b pooled planner SHOW > NONE",
        succ("show") > succ("none"),
        format!("got show {:.3} vs none {:.3}", succ("show"), succ("none")),
    );
    // A9 sensitivity
    let mut show_by_k: BTreeMap<usize, f64> = BTreeMap::new();
    for k in [4usize, 8, 16] {
        let mut v = 0.0;
        for dom_name in ["travel", "shopping"] {
            let opts = ExperimentOptions { show_k: k, ..Default::default() };
            v += run_experiment(&instances_map[dom_name], SEED, &opts)["show"].f1;
        }
        show_by_k.insert(k, v / 2.0);
    }
    let mut ask_by_q: BTreeMap<usize, f64> = BTreeMap::new();
    for q in [1usize, 3] {
        let mut v = 0.0;
        for dom_name in ["travel", "shopping"] {
            let opts = ExperimentOptions { ask_q: Some(q), ..Default::default() };
            v += run_experiment(&instances_map[dom_name], SEED, &opts)["ask"].f1;
        }
        ask_by_q.insert(q, v / 2.0);
    }
    println!(
        "  sensitivity show K 4/8/16: F1=[{:.3}, {:.3}, {:.3}]; ask Q 1/3: F1=[{:.3}, {:.3}]",
        show_by_k[&4], show_by_k[&8], show_by_k[&16], ask_by_q[&1], ask_by_q[&3]
    );
    check(
        "A9a SHOW F1 non-decreasing in k_show",
        show_by_k[&4] <= show_by_k[&8] && show_by_k[&8] <= show_by_k[&16],
        format!("got [{:.3}, {:.3}, {:.3}]", show_by_k[&4], show_by_k[&8], show_by_k[&16]),
    );
    check(
        "A9b SHOW gain 4->16 exceeds ASK gain Q 1->3",
        show_by_k[&16] - show_by_k[&4] > ask_by_q[&3] - ask_by_q[&1],
        format!("got SHOW +{:.3} vs ASK +{:.3}",
                show_by_k[&16] - show_by_k[&4], ask_by_q[&3] - ask_by_q[&1]),
    );
    // A10 invariants + determinism
    let ok_range = results.values().all(|dm| {
        dm.values().all(|m| {
            (0.0..=1.0).contains(&m.admission)
                && (0.0..=1.0).contains(&m.precision)
                && (0.0..=1.0).contains(&m.f1)
                && (0.0..=1.0).contains(&m.planner_success)
        })
    });
    check("A10a all metrics in [0, 1]", ok_range, String::new());
    let mut subset_ok = true;
    for dom_name in ["travel", "shopping"] {
        for inst in instances_map[dom_name].iter().take(50) {
            for s in ["ask", "show", "oracle"] {
                let mut rng = Rng::new(strategy_seed(SEED, inst.idx, s));
                let space = revise(inst, s, &mut rng, reshape::SHOW_K, None);
                if !inst
                    .spec0
                    .iter()
                    .zip(space.iter())
                    .all(|(a, b)| !*a || *b)
                {
                    subset_ok = false;
                }
            }
        }
    }
    check("A10b S0 is a subset of every revised space", subset_ok, String::new());
    let mut succ_k: BTreeMap<usize, f64> = BTreeMap::new();
    for k in [1usize, 2, 4] {
        let mut v = 0.0;
        for dom_name in ["travel", "shopping"] {
            let opts = ExperimentOptions { planner_k: k, ..Default::default() };
            v += run_experiment(&instances_map[dom_name], SEED, &opts)["show"]
                .planner_success;
        }
        succ_k.insert(k, v / 2.0);
    }
    println!(
        "  sensitivity planner K 1/2/4 (SHOW): succ=[{:.3}, {:.3}, {:.3}]",
        succ_k[&1], succ_k[&2], succ_k[&4]
    );
    check(
        "A10c SHOW planner success monotone in K",
        succ_k[&1] <= succ_k[&2] && succ_k[&2] <= succ_k[&4],
        format!("got [{:.3}, {:.3}, {:.3}]", succ_k[&1], succ_k[&2], succ_k[&4]),
    );
    let mut m2: BTreeMap<&str, BTreeMap<&'static str, reshape::Agg>> = BTreeMap::new();
    for dom_name in ["travel", "shopping"] {
        m2.insert(
            dom_name,
            run_experiment(&instances_map[dom_name], SEED, &ExperimentOptions::default()),
        );
    }
    let same = ["travel", "shopping"].iter().all(|d| {
        STRATEGIES.iter().all(|s| {
            let a = &results[d][s];
            let b = &m2[d][s];
            a.admission.to_bits() == b.admission.to_bits()
                && a.precision.to_bits() == b.precision.to_bits()
                && a.f1.to_bits() == b.f1.to_bits()
                && a.planner_success.to_bits() == b.planner_success.to_bits()
        })
    });
    check("A10d same seed -> identical metrics on re-run", same, String::new());
    // A11 unit cases
    let inst0 = &instances_map["travel"][0];
    let oracle_space = revise_oracle(inst0);
    let tradeoff_ok = inst0
        .plans
        .iter()
        .enumerate()
        .all(|(i, p)| !inst0.tradeoff_holds(p) || oracle_space[i]);
    check(
        "A11a oracle admits every trade-off plan",
        tradeoff_ok,
        format!(
            "{} trade-off plans",
            inst0.plans.iter().filter(|p| inst0.tradeoff_holds(p)).count()
        ),
    );
    let m0 = space_metrics(inst0, &inst0.spec0);
    let inter = inst0
        .spec0
        .iter()
        .zip(inst0.acceptable.iter())
        .filter(|(s, a)| **s && **a)
        .count();
    let exp_adm = inter as f64 / inst0.acceptable.iter().filter(|&&b| b).count() as f64;
    let exp_prec = inter as f64 / inst0.spec0.iter().filter(|&&b| b).count() as f64;
    let exp_f1 = 2.0 * exp_prec * exp_adm / (exp_prec + exp_adm);
    check(
        "A11b space_metrics matches hand computation",
        (m0.admission - exp_adm).abs() < 1e-12
            && (m0.precision - exp_prec).abs() < 1e-12
            && (m0.f1 - exp_f1).abs() < 1e-12,
        format!("adm={:.3} prec={:.3} f1={:.3}", m0.admission, m0.precision, m0.f1),
    );
    let mut rng_a = Rng::new(strategy_seed(SEED, 0, "show"));
    let space_full = revise(inst0, "show", &mut rng_a, usize::MAX, None);
    let n_full = space_full.iter().filter(|&&b| b).count();
    let full_ok = plan(inst0, &space_full, &mut Rng::new(5), n_full)
        == space_full
            .iter()
            .zip(inst0.acceptable.iter())
            .any(|(s, a)| *s && *a);
    check(
        "A11c planner with K >= |S| succeeds iff S holds an acceptable plan",
        full_ok,
        String::new(),
    );
    let mut r1 = Rng::new(12345);
    let mut r2 = Rng::new(12345);
    let same_rng: bool = (0..5).all(|_| r1.next_u64() == r2.next_u64());
    check("A11d RNG streams reproduce exactly", same_rng, String::new());
    // A12 paired rescue
    let (mut rescue, mut loss, mut n_tot) = (0usize, 0usize, 0usize);
    for dom_name in ["travel", "shopping"] {
        for inst in &instances_map[dom_name] {
            let mut outs = [false; 2];
            for (j, s) in ["none", "show"].iter().enumerate() {
                let mut rng = Rng::new(strategy_seed(SEED, inst.idx, s));
                let space = revise(inst, s, &mut rng, reshape::SHOW_K, None);
                outs[j] = plan(inst, &space, &mut rng, reshape::PLANNER_K);
            }
            n_tot += 1;
            if outs[1] && !outs[0] {
                rescue += 1;
            } else if outs[0] && !outs[1] {
                loss += 1;
            }
        }
    }
    println!(
        "  paired rescue: SHOW succeeds/NONE fails on {}/{} instances; reverse on {}/{}",
        rescue, n_tot, loss, n_tot
    );
    check(
        "A12 SHOW rescues more instances than it loses",
        rescue > loss,
        format!("rescue {} vs loss {}", rescue, loss),
    );
    // A13 robustness: F1_SHOW > F1_ASK at BONUS_FRAC 0.15 and 0.35 too.
    let mut robust = true;
    let mut robust_detail = String::new();
    let mut robust_rows: Vec<(f64, &'static str, f64, f64)> = Vec::new();
    for bf in [0.15f64, 0.35] {
        for dom_name in ["travel", "shopping"] {
            let insts = gen_instances_with_bonus(domain(dom_name), N_INSTANCES, SEED, bf);
            let res = run_experiment(&insts, SEED, &ExperimentOptions::default());
            robust_detail.push_str(&format!(
                "bf={} {}: show {:.3} vs ask {:.3}; ",
                bf, dom_name, res["show"].f1, res["ask"].f1
            ));
            robust_rows.push((bf, dom_name, res["show"].f1, res["ask"].f1));
            if !(res["show"].f1 > res["ask"].f1) {
                robust = false;
            }
        }
    }
    check(
        "A13 F1_SHOW > F1_ASK at BONUS_FRAC 0.15 and 0.35, both domains",
        robust,
        robust_detail,
    );

    // ---- write results.json (same shape as the Python results.json) ----
    let mut dom_json = String::new();
    for (di, dom_name) in ["travel", "shopping"].iter().enumerate() {
        if di > 0 {
            dom_json.push_str(",\n");
        }
        dom_json.push_str(&format!("    \"{}\": {{\n", dom_name));
        for (si, s) in STRATEGIES.iter().enumerate() {
            let m = &results[dom_name][s];
            dom_json.push_str(&format!(
                "      \"{}\": {{\"strategy\": \"{}\", \"n_instances\": {}, \
                 \"admission\": {}, \"precision\": {}, \"f1\": {}, \
                 \"planner_success\": {}, \"mean_space_size\": {}}}{}",
                s, s, m.n_instances, m.admission, m.precision, m.f1,
                m.planner_success, m.mean_space_size,
                if si + 1 < STRATEGIES.len() { "," } else { "" }
            ));
            dom_json.push('\n');
        }
        dom_json.push_str("    }");
    }
    let pooled_json = STRATEGIES
        .iter()
        .map(|s| {
            format!(
                "    \"{}\": {{\"admission\": {}, \"precision\": {}, \
                 \"f1\": {}, \"planner_success\": {}, \
                 \"mean_space_size\": {}}}",
                s,
                pooled(s, |m| m.admission),
                pooled(s, |m| m.precision),
                pooled(s, |m| m.f1),
                pooled(s, |m| m.planner_success),
                pooled(s, |m| m.mean_space_size)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let passed_json = passed
        .iter()
        .map(|s| format!("\"{}\"", s))
        .collect::<Vec<_>>()
        .join(", ");
    let failed_json = failed
        .iter()
        .map(|s| format!("\"{}\"", s))
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        "{{\n  \"paper\": \"2611.16466\",\n  \"seed\": {},\n  \
         \"n_instances_per_domain\": {},\n  \"planner_k\": {},\n  \
         \"show_k\": {},\n  \"bonus_frac\": 0.25,\n  \"n_constrained\": {},\n  \
         \"n_acceptable\": {},\n  \"implementation\": \"rust\",\n  \
         \"domains\": {{\n{}\n  }},\n  \"pooled\": {{\n{}\n  }},\n  \
         \"pearson_f1_vs_planner_success\": {},\n  \
         \"lifts\": {{\"show_over_ask_f1_travel\": {}, \
         \"show_over_ask_f1_shopping\": {}, \
         \"show_over_ask_planner_lift_pooled\": {}, \
         \"oracle_over_none_planner_lift_pooled\": {}, \
         \"show_over_none_planner_lift_pooled\": {}}},\n  \
         \"sensitivity\": {{\"show_f1_by_k_show\": {{\"4\": {}, \"8\": {}, \
         \"16\": {}}}, \"ask_f1_by_n_questions\": {{\"1\": {}, \"3\": {}}}, \
         \"show_planner_success_by_k\": {{\"1\": {}, \"2\": {}, \"4\": {}}}}},\n  \
         \"paired_rescue\": {{\"show_rescues\": {}, \"show_loses\": {}, \
         \"n_instances\": {}}},\n  \
         \"robustness_bonus_frac\": {{{}}},\n  \
         \"assertions_passed\": [{}],\n  \"assertions_failed\": [{}]\n}}\n",
        SEED,
        N_INSTANCES,
        reshape::PLANNER_K,
        reshape::SHOW_K,
        reshape::N_CONSTRAINED,
        reshape::N_ACCEPTABLE,
        dom_json,
        pooled_json,
        r,
        tr["show"].f1 - tr["ask"].f1,
        sh["show"].f1 - sh["ask"].f1,
        succ("show") / succ("ask"),
        succ("oracle") / succ("none"),
        succ("show") / succ("none"),
        show_by_k[&4],
        show_by_k[&8],
        show_by_k[&16],
        ask_by_q[&1],
        ask_by_q[&3],
        succ_k[&1],
        succ_k[&2],
        succ_k[&4],
        rescue,
        loss,
        n_tot,
        robust_rows
            .iter()
            .map(|(bf, dom, s, a)| format!(
                "\"{}:{}\": {{\"show_f1\": {}, \"ask_f1\": {}}}",
                bf, dom, s, a
            ))
            .collect::<Vec<_>>()
            .join(", "),
        passed_json,
        failed_json
    );
    fs::write("results.json", json).expect("write results.json");
    println!("\n{} passed, {} failed", passed.len(), failed.len());
    if !failed.is_empty() {
        println!("FAILED: {:?}", failed);
        std::process::exit(1);
    }
}
