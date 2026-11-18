//! Binary for the anytime-stopping Rust port.
//!
//!     anytime-stopping verify
//!         Live-test verification mirroring the Python CPU test's
//!         experiments and assertions (hand-built policy checks, RNG
//!         vector, main experiment over 5 seeds x 2000 runs, and the
//!         sensitivity sweep). Prints ALL CHECKS PASSED plus a JSON
//!         summary on success; exits non-zero on any failed check.
//!     anytime-stopping parity <kind> <args...>
//!         Machine-readable JSON for the Python-vs-Rust parity sweep
//!         (tests/parity_gen.py is the Python oracle). Kinds:
//!         rng <seed> <n> | trajectory <seed> <params...> |
//!         experiment <seed> <n_runs> <params...>.
//!         Params (8, in order): budget p_solvable p_fix
//!         p_change_after_self_pass p_change_after_self_fail p_break
//!         p_self_false_pass p_self_false_fail.
//!
//! No network, no model calls, std only.

use anytime_stopping_rs::*;
use std::process::exit;

const SEEDS: [u64; 5] = [20261118, 7, 123, 2026, 99];
const N_RUNS: usize = 2000;

fn check(cond: bool, msg: &str) {
    if !cond {
        eprintln!("FAILED: {}", msg);
        exit(1);
    }
    println!("  ok: {}", msg);
}

fn step(step: usize, patch_id: usize, changed: bool, hidden: bool, self_p: bool) -> Step {
    Step {
        step,
        patch_id,
        changed,
        hidden_pass: hidden,
        self_pass: self_p,
    }
}

fn hand_trajectory() {
    let t1 = vec![
        step(1, 0, true, true, true),
        step(2, 0, false, true, true),
        step(3, 0, false, true, true),
    ];
    check(stop_double_pass(&t1) == 2, "double-pass fires at 2nd same-patch pass");
    check(stop_first_self_pass(&t1) == 1, "first-self-pass fires at step 1");
    check(stop_oracle(&t1) == 1, "oracle fires at first hidden pass");
    let t2 = vec![
        step(1, 0, true, false, true),
        step(2, 1, true, false, true),
        step(3, 1, false, false, true),
    ];
    check(stop_double_pass(&t2) == 3, "double-pass waits for same patch twice");
    check(
        stop_double_pass_cumulative(&t2) == 3,
        "cumulative variant agrees on patch-change reset",
    );
    let t3: Vec<Step> = (1..=5).map(|i| step(i, i - 1, true, false, false)).collect();
    for (name, f) in [
        ("baseline", stop_run_to_budget as fn(&[Step]) -> usize),
        ("first_self_pass", stop_first_self_pass),
        ("double_pass", stop_double_pass),
        ("oracle", stop_oracle),
    ] {
        check(f(&t3) == 5, &format!("{} runs to budget when nothing passes", name));
    }
    let t4 = vec![
        step(1, 0, true, false, false),
        step(2, 1, true, false, false),
        step(3, 2, true, true, false),
        step(4, 2, false, true, false),
        step(5, 3, true, true, true),
        step(6, 3, false, true, true),
    ];
    check(stop_oracle(&t4) == 3, "oracle stops at first hidden pass (step 3)");
    check(stop_first_self_pass(&t4) == 5, "first self-pass at step 5");
    check(stop_double_pass(&t4) == 6, "double-pass at step 6");
}

fn test_rng() {
    let mut r = SplitMix64::new(0);
    check(
        r.next_u64() == 0xE220A8397B1DCDAF,
        "SplitMix64 seed 0 first output matches the reference vector",
    );
    let mut a = SplitMix64::new(20261118);
    let mut b = SplitMix64::new(20261118);
    let sa: Vec<u64> = (0..100).map(|_| a.next_u64()).collect();
    let sb: Vec<u64> = (0..100).map(|_| b.next_u64()).collect();
    check(sa == sb, "same seed -> identical stream");
    check(
        (0..50u64).all(|s| {
            let u = SplitMix64::new(s).uniform();
            (0.0..1.0).contains(&u)
        }),
        "uniform() in [0, 1)",
    );
    let params = TrajectoryParams::default();
    let d1: Vec<[usize; 6]> = (0..20)
        .map(|_| generate_trajectory(&mut SplitMix64::new(7), &params))
        .map(|t| trajectory_digest(&t))
        .collect();
    let d2: Vec<[usize; 6]> = (0..20)
        .map(|_| generate_trajectory(&mut SplitMix64::new(7), &params))
        .map(|t| trajectory_digest(&t))
        .collect();
    check(d1 == d2, "same seed -> identical trajectories");
}

fn test_policies_on_generated() {
    let params = TrajectoryParams::default();
    let mut rng = SplitMix64::new(20261118);
    let mut mismatch = 0;
    for _ in 0..2000 {
        let t = generate_trajectory(&mut rng, &params);
        if stop_double_pass(&t) != stop_double_pass_cumulative(&t) {
            mismatch += 1;
        }
    }
    check(mismatch == 0, "cumulative == consecutive double-pass on 2000 trajectories");
    let mut rng = SplitMix64::new(99);
    for _ in 0..500 {
        let t = generate_trajectory(&mut rng, &params);
        for f in [
            stop_run_to_budget as fn(&[Step]) -> usize,
            stop_first_self_pass,
            stop_double_pass,
            stop_oracle,
        ] {
            let s = f(&t);
            assert!((1..=t.len()).contains(&s));
        }
    }
    println!("  ok: all policy stop steps in range on 500 trajectories");
    // Sticky self-tests: an unchanged patch keeps its hidden and
    // self-test states, and its patch id.
    let mut rng = SplitMix64::new(123);
    for _ in 0..500 {
        let t = generate_trajectory(&mut rng, &params);
        for w in t.windows(2) {
            if !w[1].changed {
                assert_eq!(w[0].patch_id, w[1].patch_id);
                assert_eq!(w[0].hidden_pass, w[1].hidden_pass);
                assert_eq!(w[0].self_pass, w[1].self_pass);
            }
        }
    }
    println!("  ok: unchanged patch keeps patch id and both test states (500 runs)");
}

fn verify() {
    println!("== hand-built policy checks ==");
    hand_trajectory();
    println!("== RNG checks ==");
    test_rng();
    println!("== generated-trajectory policy checks ==");
    test_policies_on_generated();

    println!(
        "== main experiment: {} seeds x {} runs = {} runs ==",
        SEEDS.len(),
        N_RUNS,
        SEEDS.len() * N_RUNS
    );
    let params = TrajectoryParams::default();
    let (res, per_seed) = run_experiment(&params, N_RUNS, &SEEDS);
    for (name, p) in [
        ("baseline", &res.baseline),
        ("first_self_pass", &res.first_self_pass),
        ("double_pass", &res.double_pass),
        ("oracle", &res.oracle),
    ] {
        println!(
            "  {:16} saving={:6.2}% pass={:.4} fire={:.3} mean_steps={:.2}",
            name, p.saving_pct, p.final_hidden_pass_rate, p.fire_rate, p.mean_steps
        );
    }
    let pf = &res.post_fire;
    println!(
        "  post-fire: n_fired={} changed_after={} ({:.1}%) repairs={} breaks={}",
        pf.n_fired,
        pf.changed_after_fire,
        100.0 * pf.changed_after_fire as f64 / pf.n_fired as f64,
        pf.repairs_after_fire,
        pf.breaks_after_fire
    );
    println!(
        "  hindsight: conditional={:.2}% (n={}) all-runs={:.2}%",
        res.hindsight_conditional_pct,
        res.n_true_first_pass,
        res.hindsight_all_runs_pct
    );
    println!(
        "  quality delta double-pass vs baseline: {:+.2}pp (wins={} losses={} ties={}) 95% CI ({:+.2}, {:+.2})pp",
        res.quality_delta_pp,
        res.quality_wins,
        res.quality_losses,
        res.quality_ties,
        res.quality_ci_95_pp.0,
        res.quality_ci_95_pp.1
    );
    let seed_savings: Vec<f64> = per_seed.iter().map(|p| p.1).collect();
    println!(
        "  per-seed double-pass saving: {}",
        seed_savings
            .iter()
            .map(|s| format!("{:.2}%", s))
            .collect::<Vec<_>>()
            .join(", ")
    );

    println!("== assertions: paper's core claims, directionally ==");
    check(
        (0.28..=0.50).contains(&res.baseline.final_hidden_pass_rate),
        &format!(
            "baseline final pass rate in open-model SWE-bench band (~0.30-0.50; 100k-run check gives 0.297): {:.3}",
            res.baseline.final_hidden_pass_rate
        ),
    );
    check(
        (15.0..=35.0).contains(&res.double_pass.saving_pct),
        &format!(
            "double-pass rule saves substantially, near paper's 23%: {:.2}%",
            res.double_pass.saving_pct
        ),
    );
    check(
        (5.0..=57.0).contains(&res.oracle.saving_pct),
        &format!(
            "oracle (hindsight) saving in paper's <1%-57% range: {:.2}%",
            res.oracle.saving_pct
        ),
    );
    check(
        (1.0..=57.0).contains(&res.hindsight_all_runs_pct),
        &format!(
            "hindsight saving at passing self-tests (all runs) in paper's range: {:.2}%",
            res.hindsight_all_runs_pct
        ),
    );
    check(
        (1.0..=60.0).contains(&res.hindsight_conditional_pct),
        &format!(
            "hindsight saving, conditional reading, near paper's range: {:.2}%",
            res.hindsight_conditional_pct
        ),
    );
    check(
        (0.30..=0.90).contains(&res.double_pass.fire_rate),
        &format!(
            "double-pass rule fires on a substantial share of runs: {:.3}",
            res.double_pass.fire_rate
        ),
    );
    check(
        pf.repairs_after_fire > 0 && pf.breaks_after_fire > 0,
        &format!(
            "post-fire changes both repair and break the patch (repairs={} breaks={})",
            pf.repairs_after_fire, pf.breaks_after_fire
        ),
    );
    check(
        pf.changed_after_fire as f64 / pf.n_fired as f64 > 0.5,
        &format!(
            "the patch usually changes after the rule fires: {:.1}%",
            100.0 * pf.changed_after_fire as f64 / pf.n_fired as f64
        ),
    );
    check(
        res.quality_losses > 0,
        &format!(
            "quality-loss cases exist: the rule stops on {} patches that would have repaired",
            res.quality_losses
        ),
    );
    check(
        res.quality_delta_pp.abs() <= 4.0,
        &format!(
            "quality delta is small (rule neither clearly wins nor clearly loses): {:+.2}pp",
            res.quality_delta_pp
        ),
    );
    check(
        res.first_self_pass.saving_pct > res.double_pass.saving_pct,
        &format!(
            "first-self-pass rule saves more than double-pass (earlier stop, worse quality: pass {:.4} vs {:.4})",
            res.first_self_pass.final_hidden_pass_rate,
            res.double_pass.final_hidden_pass_rate
        ),
    );
    let spread = seed_savings
        .iter()
        .fold(f64::NEG_INFINITY, |a, b| a.max(*b))
        - seed_savings
            .iter()
            .fold(f64::INFINITY, |a, b| a.min(*b));
    check(
        spread < 5.0,
        &format!("per-seed double-pass saving stable (spread {:.2}pp)", spread),
    );

    println!("== sensitivity sweep: does the quality delta change sign? ==");
    let cells = sensitivity_sweep(&params, N_RUNS, &[20261118, 7]);
    let mut min_delta = f64::INFINITY;
    let mut max_delta = f64::NEG_INFINITY;
    for c in &cells {
        println!(
            "  p_break={:.2} p_change_after_self_pass={:.2} p_self_false_pass={:.2}: saving={:.1}% delta={:+.2}pp (base pass {:.3})",
            c.p_break, c.p_change_after_self_pass, c.p_self_false_pass,
            c.double_pass_saving_pct, c.quality_delta_pp, c.baseline_pass_rate
        );
        min_delta = min_delta.min(c.quality_delta_pp);
        max_delta = max_delta.max(c.quality_delta_pp);
    }
    check(
        min_delta < 0.0 && max_delta > 0.0,
        &format!(
            "quality delta changes sign across regimes (min {:+.2}pp, max {:+.2}pp) -- quality preservation is regime-dependent, non-inferiority NOT established",
            min_delta, max_delta
        ),
    );
    check(
        cells
            .iter()
            .all(|c| (10.0..=65.0).contains(&c.double_pass_saving_pct)),
        "double-pass saving stays substantial in every sweep cell",
    );

    println!("ALL CHECKS PASSED");
    println!(
        "RESULTS_JSON {{\"n_runs\": {}, \"double_pass_saving_pct\": {:.4}, \"oracle_saving_pct\": {:.4}, \"baseline_pass_rate\": {:.4}, \"double_pass_pass_rate\": {:.4}, \"hindsight_conditional_pct\": {:.4}, \"hindsight_all_runs_pct\": {:.4}, \"quality_delta_pp\": {:.4}, \"quality_wins\": {}, \"quality_losses\": {}, \"quality_ties\": {}, \"repairs_after_fire\": {}, \"breaks_after_fire\": {}, \"sweep_min_delta_pp\": {:.4}, \"sweep_max_delta_pp\": {:.4}}}",
        res.n_runs,
        res.double_pass.saving_pct,
        res.oracle.saving_pct,
        res.baseline.final_hidden_pass_rate,
        res.double_pass.final_hidden_pass_rate,
        res.hindsight_conditional_pct,
        res.hindsight_all_runs_pct,
        res.quality_delta_pp,
        res.quality_wins,
        res.quality_losses,
        res.quality_ties,
        pf.repairs_after_fire,
        pf.breaks_after_fire,
        min_delta,
        max_delta
    );
}

fn parse_params(args: &[String], start: usize) -> TrajectoryParams {
    let f = |i: usize| args[start + i].parse::<f64>().unwrap();
    TrajectoryParams {
        budget: args[start].parse::<usize>().unwrap(),
        p_solvable: f(1),
        p_fix: f(2),
        p_change_after_self_pass: f(3),
        p_change_after_self_fail: f(4),
        p_break: f(5),
        p_self_false_pass: f(6),
        p_self_false_fail: f(7),
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
        "trajectory" => {
            let seed: u64 = args[1].parse().unwrap();
            let params = parse_params(args, 2);
            let mut rng = SplitMix64::new(seed);
            let traj = generate_trajectory(&mut rng, &params);
            let steps: Vec<String> = traj
                .iter()
                .map(|s| {
                    format!(
                        "[{}, {}, {}, {}, {}]",
                        s.step, s.patch_id, s.changed, s.hidden_pass, s.self_pass
                    )
                })
                .collect();
            println!(
                "{{\"kind\": \"trajectory\", \"seed\": {}, \"steps\": [{}], \"digest\": {:?}, \"stops\": {{\"baseline\": {}, \"first_self_pass\": {}, \"double_pass\": {}, \"double_pass_cumulative\": {}, \"oracle\": {}}}}}",
                seed,
                steps.join(", "),
                trajectory_digest(&traj),
                stop_run_to_budget(&traj),
                stop_first_self_pass(&traj),
                stop_double_pass(&traj),
                stop_double_pass_cumulative(&traj),
                stop_oracle(&traj)
            );
        }
        "experiment" => {
            let seed: u64 = args[1].parse().unwrap();
            let n_runs: usize = args[2].parse().unwrap();
            let params = parse_params(args, 3);
            let mut rng = SplitMix64::new(seed);
            let trajs: Vec<Vec<Step>> =
                (0..n_runs).map(|_| generate_trajectory(&mut rng, &params)).collect();
            let ev = evaluate_all(&trajs, &params);
            println!(
                "{{\"kind\": \"experiment\", \"seed\": {}, \"n_runs\": {}, \"baseline\": {{\"sum_stop_steps\": {}, \"n_final_pass\": {}, \"n_fired\": {}}}, \"first_self_pass\": {{\"sum_stop_steps\": {}, \"n_final_pass\": {}, \"n_fired\": {}}}, \"double_pass\": {{\"sum_stop_steps\": {}, \"n_final_pass\": {}, \"n_fired\": {}}}, \"double_pass_cumulative\": {{\"sum_stop_steps\": {}, \"n_final_pass\": {}, \"n_fired\": {}}}, \"oracle\": {{\"sum_stop_steps\": {}, \"n_final_pass\": {}, \"n_fired\": {}}}, \"post_fire\": {{\"n_fired\": {}, \"changed_after_fire\": {}, \"repairs_after_fire\": {}, \"breaks_after_fire\": {}}}, \"n_true_first_pass\": {}, \"quality_wins\": {}, \"quality_losses\": {}, \"quality_ties\": {}}}",
                seed,
                n_runs,
                ev.baseline.sum_stop_steps, ev.baseline.n_final_pass, ev.baseline.n_fired,
                ev.first_self_pass.sum_stop_steps, ev.first_self_pass.n_final_pass, ev.first_self_pass.n_fired,
                ev.double_pass.sum_stop_steps, ev.double_pass.n_final_pass, ev.double_pass.n_fired,
                ev.double_pass_cumulative.sum_stop_steps, ev.double_pass_cumulative.n_final_pass, ev.double_pass_cumulative.n_fired,
                ev.oracle.sum_stop_steps, ev.oracle.n_final_pass, ev.oracle.n_fired,
                ev.post_fire.n_fired, ev.post_fire.changed_after_fire,
                ev.post_fire.repairs_after_fire, ev.post_fire.breaks_after_fire,
                ev.n_true_first_pass, ev.quality_wins, ev.quality_losses, ev.quality_ties
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
        eprintln!("usage: anytime-stopping <verify|parity ...>");
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
