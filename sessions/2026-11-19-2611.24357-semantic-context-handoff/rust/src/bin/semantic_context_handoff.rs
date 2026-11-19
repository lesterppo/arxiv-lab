//! Binary for the semantic-context-handoff Rust port.
//!
//!     semantic-context-handoff verify
//!         Live-test verification mirroring the Python CPU test's
//!         experiments and assertions (RNG vectors, task-generation
//!         invariants, policy sanity on 200 generated tasks, the main
//!         5-seed x 2000-task experiment with exact headline-number
//!         checks against the Python results.json, the no-delay sanity
//!         and the 27-cell sensitivity sweep). Prints ALL CHECKS PASSED
//!         plus a JSON summary on success; exits non-zero on any
//!         failed check.
//!     semantic-context-handoff parity <kind> <args...>
//!         Machine-readable JSON for the Python-vs-Rust parity sweep
//!         (tests/parity_gen.py is the Python oracle). Kinds:
//!         rng <seed> <n> | task <seed> <params...> |
//!         experiment <seed> <n_tasks> <params...>.
//!         Params (13, in order): steps facts_per_step delayed_frac
//!         max_delay min_fact_tokens max_fact_tokens state_retention
//!         compression state_overhead recall lookup_overhead
//!         fp_per_lookup pass_step_fraction.
//!
//! No network, no model calls, std only.

use semantic_context_handoff_rs::*;
use std::process::exit;

const SEEDS: [u64; 5] = [20261119, 7, 123, 2026, 99];
const N_TASKS: usize = 2000;

fn check(cond: bool, msg: &str) {
    if !cond {
        eprintln!("FAILED: {}", msg);
        exit(1);
    }
    println!("  ok: {}", msg);
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn test_rng() {
    let mut r = SplitMix64::new(0);
    check(
        r.next_u64() == 0xE220A8397B1DCDAF,
        "SplitMix64 seed 0 first output matches the reference vector",
    );
    check(
        r.next_u64() == 0x6E789E6AA1B965F4,
        "SplitMix64 seed 0 second output matches the reference vector",
    );
    check(
        r.next_u64() == 0x6C45D188009454F,
        "SplitMix64 seed 0 third output matches the reference vector",
    );
    let mut r = SplitMix64::new(42);
    check(
        r.next_u64() == 0xBDD732262FEB6E95,
        "SplitMix64 seed 42 first output matches Python",
    );
    let mut a = SplitMix64::new(42);
    let mut b = SplitMix64::new(42);
    let xs: Vec<u64> = (0..100).map(|_| a.next_u64()).collect();
    let ys: Vec<u64> = (0..100).map(|_| b.next_u64()).collect();
    check(xs == ys, "same seed -> identical u64 stream");
    let mut c = SplitMix64::new(43);
    check(
        (0..10).map(|_| c.next_u64()).collect::<Vec<_>>() != xs[..10].to_vec(),
        "different seed -> different stream",
    );
    let mut r = SplitMix64::new(1);
    let us: Vec<f64> = (0..10000).map(|_| r.uniform()).collect();
    check(
        us.iter().all(|&u| (0.0..1.0).contains(&u)),
        "uniform() in [0, 1)",
    );
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    check(
        (mean - 0.5).abs() < 0.02,
        &format!("uniform() mean ~ 0.5 (got {:.4})", mean),
    );
    let mut r = SplitMix64::new(1);
    check(
        r.mantissa() == 5103132997656651,
        "first uniform mantissa (seed 1) matches Python",
    );
}

fn test_task_generation() {
    let params = TaskParams::default();
    let t = generate_task(&mut SplitMix64::new(123), &params);
    check(
        t.facts.len() == params.steps * params.facts_per_step,
        "facts = steps x facts_per_step",
    );
    for f in &t.facts {
        assert!(
            f.needed_step - f.produced_step <= params.max_delay,
            "delay in [0, max_delay]"
        );
        assert!(
            (params.min_fact_tokens..=params.max_fact_tokens).contains(&f.tokens),
            "fact tokens in range"
        );
    }
    println!("  ok: delay in [0, max_delay] and fact tokens in range");
    for (s, keys) in t.needs.iter().enumerate() {
        for &k in keys {
            assert_eq!(t.facts[k].needed_step, s, "need lands on needed_step");
        }
    }
    println!("  ok: every need lands on its fact's needed_step");
    check(t.needs.len() == params.steps, "one need list per step");
    let d1 = task_digest(&generate_task(&mut SplitMix64::new(999), &params));
    let d2 = task_digest(&generate_task(&mut SplitMix64::new(999), &params));
    check(d1 == d2, "same seed -> identical task digest");
    let d3 = task_digest(&generate_task(&mut SplitMix64::new(1000), &params));
    check(d1 != d3, "different seed -> different task digest");
    check(
        task_digest(&generate_task(&mut SplitMix64::new(123), &params))
            == 16768509106337255366,
        "task digest (seed 123) matches Python exactly",
    );
    let t123 = generate_task(&mut SplitMix64::new(123), &params);
    check(
        t123.facts[0].tokens == 24
            && t123.facts[0].needed_step == 0
            && t123.facts[0].retained
            && t123.facts[1].tokens == 28
            && t123.facts[2].tokens == 24,
        "first facts of the seed-123 task match Python exactly",
    );
    check(
        t123.needs == vec![
            vec![0, 1, 2],
            vec![3, 5],
            vec![6, 7],
            vec![9, 10, 11],
            vec![4, 12, 13, 14],
            vec![8, 15, 16, 17],
            vec![18, 19, 20],
            vec![21, 23],
            vec![25, 26],
            vec![27, 28, 29],
            vec![22, 24, 30, 32],
            vec![33, 34, 35],
        ],
        "need lists of the seed-123 task match Python exactly",
    );
    let delayed = t
        .facts
        .iter()
        .filter(|f| f.needed_step > f.produced_step)
        .count();
    check(delayed > 0, "calibrated tasks contain delayed-relevance facts");
    println!("  delayed facts in one task: {} / {}", delayed, t.facts.len());
}

fn test_policies_on_generated() {
    let params = TaskParams::default();
    let tasks: Vec<Task> = (0..200)
        .map(|i| generate_task(&mut SplitMix64::new(20261119 + i as u64), &params))
        .collect();
    let res = evaluate_all(&tasks, &params, 20261119);
    for p in Policy::all() {
        let a = res.get(p);
        println!(
            "  {:8} task_pass={:.4} step={:.4} tokens/task={:.1}",
            p.name(),
            a.task_pass_rate,
            a.step_success_rate,
            a.mean_tokens_per_task
        );
    }
    check(
        res.full.task_pass_rate == 1.0,
        "FULL always passes (quality ceiling)",
    );
    check(
        res.summary.task_pass_rate < res.full.task_pass_rate,
        "SUMMARY loses tasks vs FULL",
    );
    check(
        res.sch.task_pass_rate > res.summary.task_pass_rate,
        "SCH beats SUMMARY",
    );
    check(
        res.sch.mean_tokens_per_task < res.full.mean_tokens_per_task,
        "SCH costs less than FULL",
    );
    let hit = res.sch.recovery_hit_rate.unwrap();
    check(
        (hit - params.recall).abs() < 0.05,
        &format!("recovery hit rate tracks recall ({:.2})", params.recall),
    );
    // Ablation: recall=0 -> SCH pass == SUMMARY pass exactly.
    let p0 = TaskParams {
        recall: 0.0,
        ..params
    };
    let r0 = evaluate_all(&tasks, &p0, 20261119);
    check(
        r0.sch.passes == r0.summary.passes,
        "recall=0: SCH collapses exactly to SUMMARY",
    );
    // Oracle: recall=0.9 SCH within 6pp of ORACLE.
    let gap_pp = 100.0 * (res.oracle.task_pass_rate - res.sch.task_pass_rate).abs();
    check(
        gap_pp < 6.0,
        &format!("recall=0.9: SCH within 6pp of ORACLE (got {:.1}pp)", gap_pp),
    );
    // Monotonicity in recall (step level).
    let mut passes = Vec::new();
    for rc in [0.5, 0.7, 0.9, 1.0] {
        let pr = TaskParams {
            recall: rc,
            ..params
        };
        let rr = evaluate_all(&tasks, &pr, 20261119);
        passes.push(rr.sch.step_success_rate);
    }
    check(
        passes.windows(2).all(|w| w[0] <= w[1] + 1e-9),
        &format!(
            "SCH step success rises monotonically with recall: {:?}",
            passes.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>()
        ),
    );
}

fn verify() {
    println!("== rng determinism ==");
    test_rng();
    println!("== task generation invariants ==");
    test_task_generation();
    println!("== policy sanity on generated tasks ==");
    test_policies_on_generated();

    println!(
        "== main experiment: {} seeds x {} tasks ==",
        SEEDS.len(),
        N_TASKS
    );
    let params = TaskParams::default();
    let t0 = std::time::Instant::now();
    let exp = run_experiment(&params, SEEDS[0], N_TASKS, SEEDS.len());
    println!("  ran in {:.1}s", t0.elapsed().as_secs_f64());
    let pol = &exp.policies;
    for (name, p) in [
        ("full", &pol.full),
        ("summary", &pol.summary),
        ("sch", &pol.sch),
        ("oracle", &pol.oracle),
    ] {
        println!(
            "  {:8} task_pass={:.4} strict={:.4} step={:.4} tokens/task={:.1} tokens/step={:.1}",
            name,
            p.task_pass_rate,
            p.strict_task_pass_rate,
            p.step_success_rate,
            p.mean_tokens_per_task,
            p.mean_tokens_per_step
        );
    }
    println!(
        "  per-seed SCH task-pass %: {:?}",
        exp.per_seed_task_pass_pct
            .sch
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "  per-seed SUMMARY task-pass %: {:?}",
        exp.per_seed_task_pass_pct
            .summary
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );
    let gain_pp = 100.0 * (pol.sch.task_pass_rate - pol.summary.task_pass_rate);
    let gap_pp = 100.0 * (pol.full.task_pass_rate - pol.sch.task_pass_rate);
    let step_gain_pp = 100.0 * (pol.sch.step_success_rate - pol.summary.step_success_rate);
    let step_gap_pp = 100.0 * (pol.full.step_success_rate - pol.sch.step_success_rate);
    let saving = 100.0 * (1.0 - pol.sch.mean_tokens_per_task / pol.full.mean_tokens_per_task);
    let sch_over_summary =
        100.0 * (pol.sch.mean_tokens_per_task / pol.summary.mean_tokens_per_task - 1.0);
    println!(
        "  SCH-SUMMARY gain: {:.2}pp task / {:.2}pp step | SCH-FULL gap: {:.2}pp task / {:.2}pp step | SCH token saving vs FULL: {:.1}% | SCH over SUMMARY: {:.1}%",
        gain_pp, step_gain_pp, gap_pp, step_gap_pp, saving, sch_over_summary
    );

    println!("== assertions: paper's core claims, directionally ==");
    check(
        step_gain_pp >= 3.0,
        &format!(
            "evidence recovery restores >=3pp step success over SUMMARY (got {:.2}pp)",
            step_gain_pp
        ),
    );
    check(
        step_gap_pp <= 5.0,
        &format!(
            "SCH step success within 5pp of the FULL ceiling (gap {:.2}pp)",
            step_gap_pp
        ),
    );
    check(
        gain_pp >= 3.0,
        &format!(
            "evidence recovery restores >=3pp task pass over SUMMARY (got {:.2}pp)",
            gain_pp
        ),
    );
    check(
        gap_pp <= 5.0,
        &format!("SCH within 5pp of the FULL ceiling (gap {:.2}pp)", gap_pp),
    );
    check(
        saving >= 35.0,
        &format!("SCH saves >=35% tokens vs FULL (got {:.1}%)", saving),
    );
    check(
        sch_over_summary <= 15.0,
        &format!(
            "recovery costs <=15% over SUMMARY tokens (got {:.1}%)",
            sch_over_summary
        ),
    );

    println!("== exact headline check vs Python results.json ==");
    // Full-precision values from sessions/2026-11-19-2611.24357-.../results.json.
    let expected: [(Policy, f64, f64, f64, f64, f64, f64, Option<f64>); 4] = [
        (
            Policy::Full,
            1.0,
            1.0,
            1.0,
            4677.5122,
            389.79268333333334,
            0.0,
            None,
        ),
        (
            Policy::Summary,
            0.9288000000000001,
            0.33540000000000003,
            0.9148833333333334,
            1585.4896999999999,
            132.12414166666667,
            0.0,
            None,
        ),
        (
            Policy::Sch,
            0.9999,
            0.8991,
            0.99115,
            1614.2591,
            134.52159166666667,
            1.067,
            Some(0.9002690161651146),
        ),
        (
            Policy::Oracle,
            1.0,
            1.0,
            1.0,
            1612.2505,
            134.35420833333333,
            1.067,
            Some(1.0),
        ),
    ];
    for (p, tp, st, ss, mt, ms, rec, hit) in expected {
        let a = pol.get(p);
        check(
            close(a.task_pass_rate, tp, 1e-9)
                && close(a.strict_task_pass_rate, st, 1e-9)
                && close(a.step_success_rate, ss, 1e-9)
                && close(a.mean_tokens_per_task, mt, 1e-9)
                && close(a.mean_tokens_per_step, ms, 1e-9)
                && close(a.recoveries_per_task, rec, 1e-9)
                && a.recovery_hit_rate.map(|h| close(h, hit.unwrap_or(f64::NAN), 1e-9))
                    == hit.map(|_| true),
            &format!(
                "{} headline numbers identical to Python (task_pass {:.4}, step {:.4}, tokens/task {:.4})",
                p.name(),
                a.task_pass_rate,
                a.step_success_rate,
                a.mean_tokens_per_task
            ),
        );
    }
    let exp_sch_pct = [100.0, 99.95, 100.0, 100.0, 100.0];
    let exp_sum_pct = [93.65, 93.2, 92.2, 92.55, 92.8];
    check(
        exp.per_seed_task_pass_pct
            .sch
            .iter()
            .zip(exp_sch_pct.iter())
            .all(|(a, b)| close(*a, *b, 1e-9))
            && exp
                .per_seed_task_pass_pct
                .summary
                .iter()
                .zip(exp_sum_pct.iter())
                .all(|(a, b)| close(*a, *b, 1e-9)),
        "per-seed task-pass percentages identical to Python",
    );

    // No-delay sanity: nothing ever needed later -> all pass.
    let nd = &exp.no_delay_sanity;
    check(
        nd.full == 1.0 && nd.summary == 1.0 && nd.sch == 1.0 && nd.oracle == 1.0,
        &format!(
            "no delayed relevance -> every policy passes everything: full={} summary={} sch={} oracle={}",
            nd.full, nd.summary, nd.sch, nd.oracle
        ),
    );
    // Sensitivity sweep: the effect is not a one-point artifact.
    let sweep = &exp.sensitivity_sweep;
    let wins = sweep.iter().filter(|c| c.sch_gain_pp > 0.0).count();
    let close_cells = sweep
        .iter()
        .filter(|c| c.sch_within_full_pp <= 10.0)
        .count();
    println!(
        "  sweep: SCH>SUMMARY in {}/{} cells; SCH within 10pp of FULL in {}/{} cells",
        wins,
        sweep.len(),
        close_cells,
        sweep.len()
    );
    check(
        wins as f64 >= 0.9 * sweep.len() as f64,
        "SCH beats SUMMARY in >=90% of sweep cells",
    );
    check(
        close_cells as f64 >= 0.8 * sweep.len() as f64,
        "SCH within 10pp of FULL in >=80% of sweep cells",
    );
    check(
        wins == 27,
        &format!("sweep wins exactly 27/27 as in Python (got {})", wins),
    );
    check(
        close_cells == 25,
        &format!(
            "sweep close cells exactly 25/27 as in Python (got {})",
            close_cells
        ),
    );
    let first = &sweep[0];
    check(
        close(first.sch_gain_pp, 18.75, 1e-9)
            && close(first.sch_token_saving_vs_full_pct, 73.52695387297507, 1e-9)
            && close(first.sch_within_full_pp, 2.75, 1e-9),
        "first sweep cell identical to Python (gain 18.75pp, saving 73.53%, within-full 2.75pp)",
    );

    println!("ALL CHECKS PASSED");
    println!(
        "RESULTS_JSON {{\"n_tasks\": {}, \"full_task_pass\": {:.4}, \"summary_task_pass\": {:.4}, \"sch_task_pass\": {:.4}, \"oracle_task_pass\": {:.4}, \"summary_step_success\": {:.4}, \"sch_step_success\": {:.4}, \"full_mean_tokens_per_task\": {:.4}, \"summary_mean_tokens_per_task\": {:.4}, \"sch_mean_tokens_per_task\": {:.4}, \"oracle_mean_tokens_per_task\": {:.4}, \"sch_recoveries_per_task\": {:.4}, \"sch_recovery_hit_rate\": {:.4}, \"sch_token_saving_vs_full_pct\": {:.4}, \"sch_over_summary_tokens_pct\": {:.4}, \"sweep_wins\": {}, \"sweep_close\": {}, \"sweep_cells\": {}}}",
        pol.sch.tasks,
        pol.full.task_pass_rate,
        pol.summary.task_pass_rate,
        pol.sch.task_pass_rate,
        pol.oracle.task_pass_rate,
        pol.summary.step_success_rate,
        pol.sch.step_success_rate,
        pol.full.mean_tokens_per_task,
        pol.summary.mean_tokens_per_task,
        pol.sch.mean_tokens_per_task,
        pol.oracle.mean_tokens_per_task,
        pol.sch.recoveries_per_task,
        pol.sch.recovery_hit_rate.unwrap(),
        saving,
        sch_over_summary,
        wins,
        close_cells,
        sweep.len()
    );
}

fn parse_params(args: &[String], start: usize) -> TaskParams {
    let f = |i: usize| args[start + i].parse::<f64>().unwrap();
    let u = |i: usize| args[start + i].parse::<u64>().unwrap();
    TaskParams {
        steps: args[start].parse::<usize>().unwrap(),
        facts_per_step: args[start + 1].parse::<usize>().unwrap(),
        delayed_frac: f(2),
        max_delay: args[start + 3].parse::<usize>().unwrap(),
        min_fact_tokens: u(4),
        max_fact_tokens: u(5),
        state_retention: f(6),
        compression: f(7),
        state_overhead: u(8),
        recall: f(9),
        lookup_overhead: u(10),
        fp_per_lookup: f(11),
        pass_step_fraction: f(12),
    }
}

fn parity(args: &[String]) {
    // args[0] is the kind.
    match args[0].as_str() {
        "rng" => {
            let seed: u64 = args[1].parse().unwrap();
            let n: usize = args[2].parse().unwrap();
            let mut r = SplitMix64::new(seed);
            let u64s: Vec<u64> = (0..n).map(|_| r.next_u64()).collect();
            let mut r = SplitMix64::new(seed);
            let mantissas: Vec<u64> = (0..n).map(|_| r.mantissa()).collect();
            let mut r = SplitMix64::new(seed);
            let bern: Vec<bool> = (0..n).map(|_| r.bernoulli(0.5)).collect();
            println!(
                "{{\"kind\": \"rng\", \"seed\": {}, \"u64\": {:?}, \"mantissa\": {:?}, \"bernoulli_half\": {:?}}}",
                seed, u64s, mantissas, bern
            );
        }
        "task" => {
            let seed: u64 = args[1].parse().unwrap();
            let params = parse_params(args, 2);
            let mut rng = SplitMix64::new(seed);
            let task = generate_task(&mut rng, &params);
            let facts: Vec<String> = task
                .facts
                .iter()
                .map(|f| {
                    format!(
                        "[{}, {}, {}, {}, {}]",
                        f.key, f.produced_step, f.needed_step, f.tokens, f.retained
                    )
                })
                .collect();
            let needs: Vec<String> = task
                .needs
                .iter()
                .map(|ns| format!("{:?}", ns))
                .collect();
            let mut pol_parts = Vec::new();
            for p in Policy::all() {
                let r = evaluate_policy(&task, p, &params, seed, 0);
                pol_parts.push(format!(
                    "\"{}\": {{\"steps_ok\": {}, \"task_pass\": {}, \"strict_task_pass\": {}, \"total_tokens\": {}, \"recoveries\": {}, \"recovery_hits\": {}}}",
                    p.name(),
                    r.steps_ok,
                    r.task_pass,
                    r.strict_task_pass,
                    r.total_tokens,
                    r.recoveries,
                    r.recovery_hits
                ));
            }
            println!(
                "{{\"kind\": \"task\", \"seed\": {}, \"facts\": [{}], \"needs\": [{}], \"digest\": {}, \"policies\": {{{}}}}}",
                seed,
                facts.join(", "),
                needs.join(", "),
                task_digest(&task),
                pol_parts.join(", ")
            );
        }
        "experiment" => {
            let seed: u64 = args[1].parse().unwrap();
            let n_tasks: usize = args[2].parse().unwrap();
            let params = parse_params(args, 3);
            let mut rng = SplitMix64::new(seed);
            let tasks: Vec<Task> = (0..n_tasks)
                .map(|_| generate_task(&mut rng, &params))
                .collect();
            let ev = evaluate_all(&tasks, &params, seed);
            let mut pol_parts = Vec::new();
            for p in Policy::all() {
                let a = ev.get(p);
                pol_parts.push(format!(
                    "\"{}\": {{\"tasks\": {}, \"passes\": {}, \"strict_passes\": {}, \"steps_ok\": {}, \"steps\": {}, \"tokens\": {}, \"recoveries\": {}, \"recovery_hits\": {}}}",
                    p.name(),
                    a.tasks,
                    a.passes,
                    a.strict_passes,
                    a.steps_ok,
                    a.steps,
                    a.tokens,
                    a.recoveries,
                    a.recovery_hits
                ));
            }
            println!(
                "{{\"kind\": \"experiment\", \"seed\": {}, \"n_tasks\": {}, \"policies\": {{{}}}}}",
                seed,
                n_tasks,
                pol_parts.join(", ")
            );
        }
        other => {
            eprintln!("unknown parity kind: {}", other);
            exit(2);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: semantic-context-handoff <verify|parity ...>");
        exit(2);
    }
    match args[0].as_str() {
        "verify" => verify(),
        "parity" => parity(&args[1..]),
        other => {
            eprintln!("unknown subcommand: {}", other);
            exit(2);
        }
    }
}
