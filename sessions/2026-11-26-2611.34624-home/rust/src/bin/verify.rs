//! `verify` binary: mirrors the Python CPU test's experiments and
//! assertions (tests/test_home_cpu.py), on Rust-native worlds.
//!
//! Worlds are generated with the crate's xorshift RNG, not Python's
//! `random.Random`, so the *numbers* differ from Python's while the
//! *properties* asserted are the same: loop accuracy gain + token
//! reduction, oracle reach, localized-vs-random gain per formal eval,
//! smoke-gate trap rejection, gating neutrality, determinism. Exact
//! cross-language equality is the parity sweep's job
//! (tests/parity_cases.txt).
//!
//! Usage: `cargo run --release --bin verify`
//! Writes `results.json` next to the crate root on success.

use home::{
    build_memory, dominant_cause, evaluate, failure_breakdown, home_loop,
    localized_candidates, make_world, oracle_best, Harness, N_FORMAL,
};
use std::time::Instant;

const START: Harness = Harness {
    top_k: 3,
    recency: 0,
    revision: home::Revision::All,
    fuzzy: false,
    answer: home::AnswerRule::Majority,
};

fn main() {
    let t0 = Instant::now();
    let mut failures: Vec<String> = Vec::new();
    let mut check = |name: &str, cond: bool, detail: String| {
        println!("{} {} [{}]", if cond { "ok" } else { "FAIL" }, name, detail);
        if !cond {
            failures.push(name.to_string());
        }
    };

    // ------------------------------------------------------------ E6 units
    let cases = make_world(0, 48);
    let mem_all = build_memory(&cases, home::Revision::All);
    let mem_latest = build_memory(&cases, home::Revision::Latest);
    let n_hist = cases.iter().filter(|c| c.historical).count();
    let n_syn = cases.iter().filter(|c| c.query_synonym).count();
    check("world mix", n_hist > 0 && n_hist < 48 && n_syn > 0 && n_syn < 48,
          format!("historical={}/48 synonym={}/48", n_hist, n_syn));
    let dropped: usize = (0..48).map(|i| mem_all[i].len() - mem_latest[i].len()).sum();
    check("latest drops superseded versions", dropped > 0,
          format!("dropped={}", dropped));
    let start_res = evaluate(&START, &cases, &mem_all, 0);
    let bd = failure_breakdown(&start_res);
    check("start harness is weak", start_res.accuracy < 0.5,
          format!("acc={:.3} breakdown={:?}", start_res.accuracy, bd));
    check("start failures are STALE-dominant",
          dominant_cause(&start_res) == Some(home::Cause::Stale),
          format!("cause={:?}", dominant_cause(&start_res)));
    let cands = localized_candidates(&START, &home::Cause::Stale);
    check("localized STALE first fix is recency",
          !cands.is_empty() && cands[0].recency == 100
              && cands[0].revision == home::Revision::All,
          format!("first={:?}", cands.first()));
    check("localized STALE last fix is the risky latest-only",
          !cands.is_empty()
              && cands.last().unwrap().revision == home::Revision::Latest,
          format!("last={:?}", cands.last()));

    // ------------------------------------------------------------ E4 oracle
    let (oracle_h, oracle_r) = oracle_best(&cases);
    check("oracle accuracy is 1.0", oracle_r.accuracy == 1.0,
          format!("acc={}", oracle_r.accuracy));
    check("oracle harness is all/fuzzy/top1 with recency > 0",
          oracle_h.revision == home::Revision::All && oracle_h.recency > 0
              && oracle_h.answer == home::AnswerRule::Top1 && oracle_h.fuzzy,
          format!("{:?}", oracle_h));
    check("oracle tokens beat start tokens",
          oracle_r.mean_tokens < start_res.mean_tokens,
          format!("oracle={} start={}", oracle_r.mean_tokens, start_res.mean_tokens));

    // ------------------------------------------------------------ E1 loop
    let lp = home_loop(&cases, &START, 0, N_FORMAL, true, true);
    check("E1 loop beats start accuracy",
          lp.best_result.accuracy > lp.start_result.accuracy,
          format!("{:.3} -> {:.3}", lp.start_result.accuracy, lp.best_result.accuracy));
    check("E1 loop uses fewer tokens",
          lp.best_result.mean_tokens < lp.start_result.mean_tokens,
          format!("{} -> {}", lp.start_result.mean_tokens, lp.best_result.mean_tokens));
    check("E1 loop reaches oracle token cost",
          lp.best_result.mean_tokens == oracle_r.mean_tokens,
          format!("loop={} oracle={}", lp.best_result.mean_tokens, oracle_r.mean_tokens));
    check("E1 loop reaches oracle accuracy",
          lp.best_result.accuracy == oracle_r.accuracy,
          format!("loop={} oracle={}", lp.best_result.accuracy, oracle_r.accuracy));
    check("E1 loop within the 6-formal-eval budget",
          lp.formal_evals <= 6, format!("formal={}", lp.formal_evals));
    check("E1 memory built once per revision policy",
          lp.constructions <= 2, format!("constructions={}", lp.constructions));
    println!("    E1 detail: best={:?} formal={} smoke={} rejects={} tried={}",
             lp.best, lp.formal_evals, lp.smoke_evals, lp.smoke_rejects,
             lp.candidates_tried);

    // ------------------------------------------------------------ E2 localized vs random
    let seeds: Vec<u64> = (0..12).collect();
    let mut loc_gains = Vec::new();
    let mut rnd_gains = Vec::new();
    for &s in &seeds {
        let cs = make_world(100 + s, 48);
        let l = home_loop(&cs, &START, s, N_FORMAL, true, true);
        let r = home_loop(&cs, &START, s, N_FORMAL, true, false);
        loc_gains.push((l.best_result.accuracy - l.start_result.accuracy)
                       / l.formal_evals.max(1) as f64);
        rnd_gains.push((r.best_result.accuracy - r.start_result.accuracy)
                       / r.formal_evals.max(1) as f64);
    }
    let loc_mean = loc_gains.iter().sum::<f64>() / loc_gains.len() as f64;
    let rnd_mean = rnd_gains.iter().sum::<f64>() / rnd_gains.len() as f64;
    println!("    E2 detail: localized gain/formal-eval={:.4} random={:.4}",
             loc_mean, rnd_mean);
    check("E2 localized beats random per formal eval",
          loc_mean > rnd_mean + 0.05,
          format!("localized={:.4} random={:.4}", loc_mean, rnd_mean));

    // ------------------------------------------------------------ E3 gating
    let fixed = Harness { recency: 100, ..START };
    let fixed_res = evaluate(&fixed, &cases,
                             &build_memory(&cases, home::Revision::All), 1);
    let trap = Harness { recency: 100, revision: home::Revision::Latest, ..START };
    let sidx = home::smoke_indices(&cases, home::SMOKE_N);
    let smoke_cases: Vec<home::Case> = sidx.iter().map(|&i| cases[i].clone()).collect();
    let sm_fixed = build_memory(&cases, home::Revision::All);
    let sm_trap = build_memory(&cases, home::Revision::Latest);
    let sm_fixed_s: Vec<Vec<home::Fact>> = sidx.iter().map(|&i| sm_fixed[i].clone()).collect();
    let sm_trap_s: Vec<Vec<home::Fact>> = sidx.iter().map(|&i| sm_trap[i].clone()).collect();
    let s_fixed = evaluate(&fixed, &smoke_cases, &sm_fixed_s, 0);
    let s_trap = evaluate(&trap, &smoke_cases, &sm_trap_s, 0);
    let trap_res = evaluate(&trap, &cases, &sm_trap, 1);
    println!("    E3a detail: trap smoke acc={:.3} vs incumbent={:.3}; \
              formal trap={:.3} vs incumbent={:.3}",
             s_trap.accuracy, s_fixed.accuracy, trap_res.accuracy, fixed_res.accuracy);
    check("E3a smoke gate rejects the trap candidate",
          s_trap.accuracy < s_fixed.accuracy - 1e-9,
          format!("trap smoke={} incumbent smoke={}", s_trap.accuracy, s_fixed.accuracy));
    check("E3a trap is genuinely worse formally too",
          trap_res.accuracy < fixed_res.accuracy - 1e-9,
          format!("trap={} incumbent={}", trap_res.accuracy, fixed_res.accuracy));
    let mut gated_formals = Vec::new();
    let mut ungated_formals = Vec::new();
    let mut gated_accs = Vec::new();
    for &s in &seeds {
        let cs = make_world(100 + s, 48);
        let g = home_loop(&cs, &START, s, N_FORMAL, true, true);
        let u = home_loop(&cs, &START, s, N_FORMAL, false, true);
        gated_formals.push(g.formal_evals);
        ungated_formals.push(u.formal_evals);
        gated_accs.push(g.best_result.accuracy);
    }
    let g_mean = gated_formals.iter().sum::<usize>() as f64 / gated_formals.len() as f64;
    let u_mean = ungated_formals.iter().sum::<usize>() as f64 / ungated_formals.len() as f64;
    let g_acc = gated_accs.iter().sum::<f64>() / gated_accs.len() as f64;
    println!("    E3b detail: gated formal-evals mean={:.2} ungated={:.2} \
              gated acc mean={:.3}", g_mean, u_mean, g_acc);
    check("E3b gating never costs formal evaluations", g_mean <= u_mean,
          format!("gated={:.2} ungated={:.2}", g_mean, u_mean));
    check("E3b gating keeps the accuracy",
          gated_accs.iter().all(|&a| a >= 0.95),
          format!("min gated acc={:.3}",
                  gated_accs.iter().fold(f64::INFINITY, |a, &b| a.min(b))));

    // ------------------------------------------------------------ E5 determinism
    let lp2 = home_loop(&cases, &START, 0, N_FORMAL, true, true);
    check("E5 deterministic loop",
          lp.history.iter().map(|(h, a, _)| (*h, *a)).collect::<Vec<_>>()
              == lp2.history.iter().map(|(h, a, _)| (*h, *a)).collect::<Vec<_>>(),
          format!("histories equal: {} steps", lp.history.len()));

    // ------------------------------------------------------------ results.json
    let elapsed = t0.elapsed().as_secs_f64();
    let json = format!(
        "{{\n  \"e1_start_accuracy\": {:.4},\n  \"e1_best_accuracy\": {:.4},\n  \"e1_start_tokens\": {:.1},\n  \"e1_best_tokens\": {:.1},\n  \"e1_oracle_accuracy\": {:.4},\n  \"e1_oracle_tokens\": {:.1},\n  \"e1_formal_evals\": {},\n  \"e1_smoke_evals\": {},\n  \"e1_smoke_rejects\": {},\n  \"e1_constructions\": {},\n  \"e2_localized_gain_per_formal\": {:.4},\n  \"e2_random_gain_per_formal\": {:.4},\n  \"e3a_trap_smoke_accuracy\": {:.4},\n  \"e3a_incumbent_smoke_accuracy\": {:.4},\n  \"e3a_trap_formal_accuracy\": {:.4},\n  \"e3a_incumbent_formal_accuracy\": {:.4},\n  \"e3b_gated_formal_mean\": {:.2},\n  \"e3b_ungated_formal_mean\": {:.2},\n  \"e3b_gated_accuracy_mean\": {:.4},\n  \"elapsed_s\": {:.2}\n}}\n",
        lp.start_result.accuracy, lp.best_result.accuracy,
        lp.start_result.mean_tokens, lp.best_result.mean_tokens,
        oracle_r.accuracy, oracle_r.mean_tokens,
        lp.formal_evals, lp.smoke_evals, lp.smoke_rejects, lp.constructions,
        loc_mean, rnd_mean,
        s_trap.accuracy, s_fixed.accuracy, trap_res.accuracy, fixed_res.accuracy,
        g_mean, u_mean, g_acc, elapsed);
    std::fs::write("results.json", &json).expect("write results.json");
    println!("wrote results.json ({:.2}s)", elapsed);
    if failures.is_empty() {
        println!("ALL CHECKS PASSED");
    } else {
        println!("FAILED {} checks: {:?}", failures.len(), failures);
        std::process::exit(1);
    }
}
