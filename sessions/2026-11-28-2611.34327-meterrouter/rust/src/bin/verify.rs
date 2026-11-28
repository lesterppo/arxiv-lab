//! `verify` binary: mirrors the Python CPU test's experiments and
//! assertions (tests/test_meterrouter_cpu.py), on Rust-native worlds.
//!
//! Worlds are generated with the crate's SplitMix64 RNG, not Python's
//! `random.Random`, so the *numbers* differ from Python's while the
//! *properties* asserted are the same: the router never overruns its
//! budget; at B = 5e-2 it is within 5 points of the strongest MAS at
//! <= 60% of its cost and beats the best single agent by >= 3 points;
//! cost and accuracy rise with the budget with diminishing returns;
//! the random-tier ablation loses >= 3 points at every budget; the
//! budget matters more on hard queries than on easy ones; repeated
//! runs are identical. Exact cross-language equality is the parity
//! sweep's job (tests/parity_cases.txt).
//!
//! Usage: `cargo run --release --bin verify`
//! Writes `results.json` next to the crate root on success.

use meter_router::{
    call_cost, greedy_router, make_queries, remaining_budget, run_experiment,
    template_graph, utility, Agent, BUDGETS, N_MAX,
};
use std::time::Instant;

const N: usize = 500;
const SEED: u64 = 20261128;

fn main() {
    let t0 = Instant::now();
    let mut fails: Vec<String> = Vec::new();
    let check = |name: &str, cond: bool, detail: String, fails: &mut Vec<String>| {
        println!("{}{}{}", if cond { "ok " } else { "FAIL " }, name, if detail.is_empty() { String::new() } else { format!(" [{detail}]") });
        if !cond {
            fails.push(name.to_string());
        }
    };

    // ------------------------------------------------------------- E1 units
    let c = call_cost(0, 0, 0);
    check("E1 cost low solver", (c - (0.06 * 2000.0 + 0.19 * 1200.0) / 1e6).abs() < 1e-12,
        format!("cost={c:.6}"), &mut fails);
    let c = call_cost(2, 2, 3);
    check("E1 cost high inspector 2 parents",
        (c - (0.30 * 3000.0 + 1.20 * 700.0) / 1e6).abs() < 1e-12,
        format!("cost={c:.6}"), &mut fails);
    check("E1 remaining budget",
        (remaining_budget(5e-3, &[1e-3, 2e-3]) - 2e-3).abs() < 1e-15,
        String::new(), &mut fails);
    check("E1 utility correct",
        (utility(true, 1e-3, 5e-3, 0.5) - (1.0 - 0.5 * 0.2)).abs() < 1e-12,
        String::new(), &mut fails);
    check("E1 utility wrong",
        (utility(false, 1e-3, 5e-3, 0.5) - (-0.5 * 0.2)).abs() < 1e-12,
        String::new(), &mut fails);
    check("E1 utility gamma 1",
        (utility(true, 1e-3, 5e-3, 1.0) - 0.8).abs() < 1e-12,
        String::new(), &mut fails);

    // ------------------------------------------------------------- E2 templates
    check("E2 single", template_graph("single", 1) == vec![Vec::<usize>::new()], String::new(), &mut fails);
    check("E2 chain 3", template_graph("chain", 3) == vec![vec![], vec![0], vec![1]], String::new(), &mut fails);
    check("E2 fan-in 3", template_graph("fan-in", 3) == vec![vec![], vec![], vec![0, 1]], String::new(), &mut fails);
    check("E2 dense 4 edges",
        template_graph("dense", 4).iter().map(|p| p.len()).sum::<usize>() == 6,
        String::new(), &mut fails);
    check("E2 diamond 4", template_graph("diamond", 4) == vec![vec![], vec![0], vec![0], vec![1, 2]], String::new(), &mut fails);
    check("E2 chain 5 edges",
        template_graph("chain", 5).iter().map(|p| p.len()).sum::<usize>() == 4,
        String::new(), &mut fails);
    check("E2 dense 5 edges",
        template_graph("dense", 5).iter().map(|p| p.len()).sum::<usize>() == 10,
        String::new(), &mut fails);

    // ------------------------------------------------------- headline run
    let results = run_experiment(N, SEED);
    let idx = |m: &str| match m {
        "meterrouter" => 0, "random_tier" => 1, "strongest" => 2,
        "single_high" => 3, "single_low" => 4, _ => 5,
    };
    let router: Vec<&meter_router::RunResult> =
        results.iter().map(|r| &r[idx("meterrouter")]).collect();
    let strongest: Vec<&meter_router::RunResult> =
        results.iter().map(|r| &r[idx("strongest")]).collect();
    let single_high: Vec<&meter_router::RunResult> =
        results.iter().map(|r| &r[idx("single_high")]).collect();
    let random_tier: Vec<&meter_router::RunResult> =
        results.iter().map(|r| &r[idx("random_tier")]).collect();
    for (i, b) in BUDGETS.iter().enumerate() {
        println!("B={b}: router acc={:.3} cost={:.2}e-3 agents={:.2} overrun={:.3} stops={:?}",
            router[i].accuracy(), router[i].mean_cost() * 1e3, router[i].mean_agents(),
            router[i].overrun_rate(), router[i].stops);
    }

    // ---------------------------------------------------------------- E3
    for (i, b) in BUDGETS.iter().enumerate() {
        check(&format!("E3 overrun B={b}"), router[i].overrun_rate() == 0.0,
            format!("overrun={:.3}", router[i].overrun_rate()), &mut fails);
    }

    // ---------------------------------------------------------------- E4
    let main = 3; // B = 5e-2
    check("E4 acc within 5pts of strongest",
        router[main].accuracy() >= strongest[main].accuracy() - 0.05,
        format!("router={:.3} strongest={:.3}", router[main].accuracy(), strongest[main].accuracy()),
        &mut fails);
    check("E4 cost <= 60% of strongest",
        router[main].mean_cost() <= 0.6 * strongest[main].mean_cost(),
        format!("router={:.2}e-3 strongest={:.2}e-3",
            router[main].mean_cost() * 1e3, strongest[main].mean_cost() * 1e3),
        &mut fails);
    check("E4 beats best single agent by >= 3pts",
        router[main].accuracy() >= single_high[main].accuracy() + 0.03,
        format!("router={:.3} single_high={:.3}", router[main].accuracy(), single_high[main].accuracy()),
        &mut fails);

    // ---------------------------------------------------------------- E5
    let costs: Vec<f64> = router.iter().map(|r| r.mean_cost()).collect();
    let accs: Vec<f64> = router.iter().map(|r| r.accuracy()).collect();
    check("E5 cost rises with budget",
        (0..3).all(|i| costs[i] <= costs[i + 1]) && costs[3] > costs[0] && costs[2] > costs[0],
        format!("costs={costs:.2?}"), &mut fails);
    check("E5 accuracy rises with budget",
        (0..3).all(|i| accs[i] <= accs[i + 1]),
        format!("accs={accs:.3?}"), &mut fails);
    check("E5 diminishing returns", accs[3] - accs[2] < accs[2] - accs[0],
        format!("gain 1e-3->5e-3={:.3} 5e-3->5e-2={:.3}", accs[2] - accs[0], accs[3] - accs[2]),
        &mut fails);

    // ---------------------------------------------------------------- E6
    for (i, b) in BUDGETS.iter().enumerate() {
        check(&format!("E6 ablation loss B={b}"),
            random_tier[i].accuracy() <= router[i].accuracy() - 0.03,
            format!("router={:.3} random={:.3}", router[i].accuracy(), random_tier[i].accuracy()),
            &mut fails);
        check(&format!("E6 ablation cost similar B={b}"),
            random_tier[i].mean_cost() <= router[i].mean_cost() * 1.2 + 1e-9,
            format!("random={:.2}e-3 router={:.2}e-3",
                random_tier[i].mean_cost() * 1e3, router[i].mean_cost() * 1e3),
            &mut fails);
    }

    // ---------------------------------------------------------------- E7
    let queries = make_queries(N, SEED);
    let mut gains = std::collections::HashMap::new();
    for diff in ["easy", "hard"] {
        let idxs: Vec<usize> = queries.iter().enumerate()
            .filter(|(_, q)| q.difficulty == diff).map(|(i, _)| i).collect();
        let mut acc_by_b = [0.0; 2];
        for (j, b) in [1e-3, 50e-3].iter().enumerate() {
            let mut corr = 0usize;
            for qi in &idxs {
                let q = &queries[*qi];
                let (team, _) = greedy_router(q.difficulty, *b, false, &[]);
                if !team.is_empty() {
                    // Paired simulation draws, mirroring run_method.
                    let mut rng = meter_router::SplitMix64::new(
                        SEED.wrapping_mul(1_000_003).wrapping_add(*qi as u64));
                    let mut spare: Option<f64> = None;
                    if meter_router::simulate_team(&team, q, &mut rng, &mut spare) {
                        corr += 1;
                    }
                }
            }
            acc_by_b[j] = corr as f64 / idxs.len() as f64;
        }
        gains.insert(diff, acc_by_b[1] - acc_by_b[0]);
    }
    check("E7 hard gain > easy gain", gains["hard"] > gains["easy"],
        format!("hard gain={:.3} easy gain={:.3}", gains["hard"], gains["easy"]), &mut fails);

    // ---------------------------------------------------------------- E8
    let results2 = run_experiment(N, SEED);
    let same = (0..4).all(|i| {
        (results[i][idx("meterrouter")].accuracy() - results2[i][idx("meterrouter")].accuracy()).abs() < 1e-12
            && (results[i][idx("meterrouter")].mean_cost() - results2[i][idx("meterrouter")].mean_cost()).abs() < 1e-15
    });
    check("E8 determinism", same, String::new(), &mut fails);

    // ---------------------------------------------------------------- output
    let _ = N_MAX;
    let _ = Agent::new(0, 0, &[]);
    let out = format!(
        "{{\n  \"article\": \"arXiv:2611.34327\",\n  \"track\": \"rust\",\n  \"n_queries\": {},\n  \"seed\": {},\n  \"budgets\": [{{\n    \"budget\": 0.001,\n    \"router_accuracy\": {:.6},\n    \"router_mean_cost\": {:.6},\n    \"router_mean_agents\": {:.6},\n    \"router_overrun_rate\": {:.6},\n    \"random_tier_accuracy\": {:.6},\n    \"strongest_accuracy\": {:.6},\n    \"strongest_mean_cost\": {:.6},\n    \"single_high_accuracy\": {:.6}\n  }}, {{\n    \"budget\": 0.004,\n    \"router_accuracy\": {:.6},\n    \"router_mean_cost\": {:.6},\n    \"router_mean_agents\": {:.6},\n    \"router_overrun_rate\": {:.6},\n    \"random_tier_accuracy\": {:.6},\n    \"strongest_accuracy\": {:.6},\n    \"strongest_mean_cost\": {:.6},\n    \"single_high_accuracy\": {:.6}\n  }}, {{\n    \"budget\": 0.005,\n    \"router_accuracy\": {:.6},\n    \"router_mean_cost\": {:.6},\n    \"router_mean_agents\": {:.6},\n    \"router_overrun_rate\": {:.6},\n    \"random_tier_accuracy\": {:.6},\n    \"strongest_accuracy\": {:.6},\n    \"strongest_mean_cost\": {:.6},\n    \"single_high_accuracy\": {:.6}\n  }}, {{\n    \"budget\": 0.05,\n    \"router_accuracy\": {:.6},\n    \"router_mean_cost\": {:.6},\n    \"router_mean_agents\": {:.6},\n    \"router_overrun_rate\": {:.6},\n    \"random_tier_accuracy\": {:.6},\n    \"strongest_accuracy\": {:.6},\n    \"strongest_mean_cost\": {:.6},\n    \"single_high_accuracy\": {:.6}\n  }}],\n  \"difficulty_gain_easy\": {:.6},\n  \"difficulty_gain_hard\": {:.6},\n  \"all_checks_passed\": {}\n}}\n",
        N, SEED,
        router[0].accuracy(), router[0].mean_cost(), router[0].mean_agents(), router[0].overrun_rate(),
        random_tier[0].accuracy(), strongest[0].accuracy(), strongest[0].mean_cost(), single_high[0].accuracy(),
        router[1].accuracy(), router[1].mean_cost(), router[1].mean_agents(), router[1].overrun_rate(),
        random_tier[1].accuracy(), strongest[1].accuracy(), strongest[1].mean_cost(), single_high[1].accuracy(),
        router[2].accuracy(), router[2].mean_cost(), router[2].mean_agents(), router[2].overrun_rate(),
        random_tier[2].accuracy(), strongest[2].accuracy(), strongest[2].mean_cost(), single_high[2].accuracy(),
        router[3].accuracy(), router[3].mean_cost(), router[3].mean_agents(), router[3].overrun_rate(),
        random_tier[3].accuracy(), strongest[3].accuracy(), strongest[3].mean_cost(), single_high[3].accuracy(),
        gains["easy"], gains["hard"], fails.is_empty(),
    );
    let out_path = std::env::current_dir().unwrap().join("results.json");
    std::fs::write(&out_path, &out).unwrap();
    println!("wrote {}", out_path.display());
    println!("\nverify finished in {:.1}s", t0.elapsed().as_secs_f64());
    if fails.is_empty() {
        println!("ALL CHECKS PASSED");
    } else {
        println!("FAILED: {} check(s): {:?}", fails.len(), fails);
        std::process::exit(1);
    }
}
