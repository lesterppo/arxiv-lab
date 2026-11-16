//! Deep tests for the LeBet Rust port: property sweeps over random
//! seeds/inputs/budgets, exhaustive + random predicate-sequence checks
//! against independent brute-force oracles, a 10x soak, and
//! thread-identity (stateless determinism).
//!
//! Run: `cargo test --release --offline --test deep` (release is
//! recommended: the 10x soak is slow in debug). The lib unit tests run
//! under plain `cargo test --offline`.

use lebet_rs::*;
use std::collections::BTreeSet;

// ---------------------------------------------------------------- oracles
// Independent brute-force re-implementations of the predicates, written
// differently from the lib (counting/sorting instead of incremental
// scans) so agreement is a real check, not a tautology.

fn oracle_loop(prefix: &[Step]) -> bool {
    let mut states: Vec<u64> = prefix.iter().map(|s| s.state).collect();
    states.sort_unstable();
    states.windows(3).any(|w| w[0] == w[1] && w[1] == w[2])
}

fn oracle_err_rate(prefix: &[Step]) -> bool {
    let n = prefix.len().min(10);
    if n < 5 {
        return false;
    }
    let w = &prefix[prefix.len() - n..];
    let mut errs = 0;
    for s in w {
        if !s.ok {
            errs += 1;
        }
    }
    (errs as f64) / (n as f64) > 0.3
}

fn oracle_same_action_5(prefix: &[Step]) -> bool {
    let n = prefix.len().min(8);
    if n < 5 {
        return false;
    }
    let w = &prefix[prefix.len() - n..];
    let a = prefix[prefix.len() - 1].action;
    let mut count = 0;
    for s in w {
        if s.action == a {
            count += 1;
        }
    }
    count >= 5
}

fn oracle_tok_high(prefix: &[Step]) -> bool {
    let mut sum = 0i64;
    for s in prefix {
        sum += s.tokens;
    }
    sum > 6000
}

fn oracle_repeat_tool_fail(prefix: &[Step]) -> bool {
    if prefix.len() < 2 || prefix[prefix.len() - 1].ok {
        return false;
    }
    let a = prefix[prefix.len() - 1].action;
    let mut count = 0;
    for s in prefix {
        if s.action == a && !s.ok {
            count += 1;
        }
    }
    count >= 2
}

fn oracle_row(prefix: &[Step]) -> Row {
    [
        prefix.len() >= 3
            && prefix[prefix.len() - 1].action == prefix[prefix.len() - 2].action
            && prefix[prefix.len() - 2].action == prefix[prefix.len() - 3].action,
        prefix.len() >= 2 && !prefix[prefix.len() - 1].ok && !prefix[prefix.len() - 2].ok,
        oracle_err_rate(prefix),
        prefix.len() >= 3
            && prefix[prefix.len() - 1].state == prefix[prefix.len() - 2].state
            && prefix[prefix.len() - 2].state == prefix[prefix.len() - 3].state,
        oracle_loop(prefix),
        prefix.len() > 20,
        oracle_repeat_tool_fail(prefix),
        prefix.len() >= 2 && prefix[prefix.len() - 1].obs == prefix[prefix.len() - 2].obs,
        oracle_tok_high(prefix),
        oracle_same_action_5(prefix),
    ]
}

fn random_step(rng: &mut Rng) -> Step {
    Step {
        action: rng.randint(6),
        ok: rng.uniform() < 0.8,
        state: rng.randint(4), // tiny alphabet: loops/stalls occur often
        obs: rng.randint(4),
        tokens: 250 + rng.randint(150) as i64,
    }
}

#[test]
fn predicates_match_oracle_random() {
    let mut rng = Rng::new(20261116);
    let mut checked = 0usize;
    for _ in 0..2000 {
        let n = rng.randint(41) as usize;
        let prefix: Vec<Step> = (0..n).map(|_| random_step(&mut rng)).collect();
        let got: Row = {
            let mut row = [false; 10];
            for (i, (_, f)) in PREDICATES.iter().enumerate() {
                row[i] = f(&prefix);
            }
            row
        };
        assert_eq!(got, oracle_row(&prefix), "prefix len {}", n);
        checked += 1;
    }
    assert_eq!(checked, 2000);
}

#[test]
fn predicates_match_oracle_exhaustive() {
    // Exhaustive short sequences over a 3-state alphabet: every
    // length-0..=6 sequence (3^0+...+3^6 = 1093). Catches any
    // incremental-vs-counting slip in p_loop / p_stalled.
    let mut count = 0usize;
    for len in 0..=6usize {
        let total = 3usize.pow(len as u32);
        for code in 0..total {
            let mut prefix = Vec::with_capacity(len);
            let mut c = code;
            for _ in 0..len {
                let s = (c % 3) as u64;
                c /= 3;
                prefix.push(Step {
                    action: s % 2,
                    ok: s != 1,
                    state: s,
                    obs: s,
                    tokens: 300,
                });
            }
            let got: Row = {
                let mut row = [false; 10];
                for (i, (_, f)) in PREDICATES.iter().enumerate() {
                    row[i] = f(&prefix);
                }
                row
            };
            assert_eq!(got, oracle_row(&prefix), "code {} len {}", code, len);
            count += 1;
        }
    }
    assert_eq!(count, 1093);
}

#[test]
fn monitor_fires_matches_oracle() {
    // Random monitors x random matrices: firing must agree with a
    // direct subset check written independently.
    let mut rng = Rng::new(555);
    for _ in 0..500 {
        let mon = random_monitor(&mut rng, 3, 3);
        let n = rng.randint(30) as usize;
        let prefix: Vec<Step> = (0..n).map(|_| random_step(&mut rng)).collect();
        let mat = {
            let t = Traj { steps: prefix, label: true };
            prefix_matrix(&t)
        };
        for (ti, row) in mat.iter().enumerate() {
            let oracle = mon.iter().any(|conj| {
                !conj.is_empty() && conj.iter().all(|&i| row[i])
            });
            assert_eq!(monitor_fires_row(&mon, row), oracle, "t={}", ti);
        }
        let first = mat
            .iter()
            .position(|row| {
                mon.iter()
                    .any(|conj| !conj.is_empty() && conj.iter().all(|&i| row[i]))
            })
            .map(|t| t + 1);
        assert_eq!(monitor_fires_traj(&mon, &mat), first);
    }
}

// ---------------------------------------------------------------- property sweeps
#[test]
fn property_sweep_seeds() {
    // 50 seeds: determinism, monitor size bounds, q in [0,1], wealth /
    // intervene consistency, tuning validity. Small training sets keep
    // this fast; the full-scale check is `verify` + the soak.
    for seed in 0..50u64 {
        let mut rng = Rng::new(seed);
        let train = gen_dataset(&mut rng, 30, 30);
        let test = gen_dataset(&mut rng, 30, 30);
        let train_mats: Vec<(Mat, bool)> = train
            .iter()
            .map(|t| (prefix_matrix(t), t.label))
            .collect();
        // determinism: same seed reproduces the dataset exactly
        let mut rng2 = Rng::new(seed);
        let train2 = gen_dataset(&mut rng2, 30, 30);
        assert_eq!(train.len(), train2.len());
        for (a, b) in train.iter().zip(train2.iter()) {
            assert_eq!(a.steps, b.steps);
            assert_eq!(a.label, b.label);
        }
        // learner: bounded, interpretable monitor; score finite
        let (mon, score) = learn_monitor(&mut rng, &train_mats, 2, 50, 0.15, 0.75);
        assert!(!mon.is_empty() && mon.len() <= 3, "seed {}", seed);
        assert!(mon.iter().all(|c| !c.is_empty() && c.len() <= 3));
        assert!(score.is_finite());
        // q is a rate in [0, 1]
        let q = baseline_fire_rate(&mon, &train_mats);
        assert!((0.0..=1.0).contains(&q), "seed {} q={}", seed, q);
        // wealth series: length matches, first step is the closed form,
        // intervene step is the first crossing of 1/alpha
        for (mat, _) in train_mats.iter().take(5) {
            let series = betting_wealth_series(&mon, mat, q, 0.75);
            assert_eq!(series.len(), mat.len());
            if let Some(row) = mat.first() {
                let x = if monitor_fires_row(&mon, row) { 1.0 } else { 0.0 };
                let expect = 1.0 + 0.75 * (x - q) / (1.0 - q);
                assert!((series[0] - expect).abs() < 1e-12);
            }
            let step = betting_intervene_step(&mon, mat, q, 0.15, 0.75);
            let expect = series
                .iter()
                .position(|w| *w >= 1.0 / 0.15)
                .map(|t| t + 1);
            assert_eq!(step, expect, "seed {}", seed);
            // wealth is positive (betting fraction keeps 1 + lam*(x-q)/(1-q) > 0
            // for lam=0.75, q in [0,1): min factor 1 - 0.75*q/(1-q) can go
            // negative only for q > 4/7, which never happens here; assert
            // positivity only when q <= 0.5)
            if q <= 0.5 {
                assert!(series.iter().all(|w| *w > 0.0));
            }
        }
        // tuning validity on train
        if let Some(k) = tune_turn_limit(&train_mats, 0.15) {
            let succ: Vec<usize> = train_mats
                .iter()
                .filter(|(_, l)| *l)
                .map(|(m, _)| m.len())
                .collect();
            let fpr = succ.iter().filter(|&&l| l > k).count() as f64 / succ.len() as f64;
            assert!(fpr <= 0.15, "seed {} K*={} fpr={}", seed, k, fpr);
        }
        if let Some((_, _, fpr)) = tune_error_filter(&train_mats, 0.15) {
            assert!(fpr <= 0.15, "seed {} err-filter fpr={}", seed, fpr);
        }
        // evaluation is well-formed on held-out data
        let ev = evaluate(&mon, q, 0.15, &train, &test, 0.75);
        assert!((0.0..=1.0).contains(&ev.betting.fpr));
        assert!((0.0..=1.0).contains(&ev.betting.power));
        assert!(ev.betting.token_savings > -1.0 && ev.betting.token_savings < 1.0);
        let _ = ev;
    }
}

#[test]
fn monitor_score_formula() {
    // monitor_score must equal tpr - lam*fpr - complexity*size, exactly.
    let mut rng = Rng::new(11);
    let data = gen_dataset(&mut rng, 20, 20);
    let mats: Vec<(Mat, bool)> = data
        .iter()
        .map(|t| (prefix_matrix(t), t.label))
        .collect();
    let monitors: Vec<Monitor> = vec![
        vec![BTreeSet::from([5])], // "long"
        vec![BTreeSet::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])], // all 10
        vec![BTreeSet::from([2]), BTreeSet::from([3])],
        random_monitor(&mut rng, 3, 3),
    ];
    for mon in &monitors {
        let (tpr, fpr) = monitor_tpr_fpr(mon, &mats);
        let expect = tpr - 2.0 * fpr - 0.005 * monitor_complexity(mon) as f64;
        assert!((monitor_score(mon, &mats, 2.0, 0.005) - expect).abs() < 1e-12);
    }
    // A monitor that fires a lot has a worse (more FPR-penalised) score
    // than one that fires rarely on this data: "long" fires on most
    // trajectories, the 10-way conjunction on (nearly) none.
    let (tpr_long, fpr_long) = monitor_tpr_fpr(&monitors[0], &mats);
    let (tpr_all, fpr_all) = monitor_tpr_fpr(&monitors[1], &mats);
    assert!(fpr_long > fpr_all);
    assert!(tpr_long > tpr_all);
}

// ---------------------------------------------------------------- soak + threads
#[test]
fn soak_10x() {
    // 10x the Python test's trajectories (2000+2000 train/test). Same
    // seed, same SLS budget, same assertions as `verify`.
    let r = run_experiment(20261116, 2000, 2000, 2000, 2000, 0.15, 6, 200, 0.75);
    let e = &r.eval;
    let b = &e.betting;
    assert!(b.fpr <= e.alpha + 0.05, "soak FPR {:.3}", b.fpr);
    assert!(b.power >= 0.60, "soak power {:.3}", b.power);
    assert!(b.power > e.turn_limit_tuned.power, "soak power <= turn-limit");
    assert!(b.power > e.error_filter_tuned.power, "soak power <= error-filter");
    assert!(b.token_savings > 0.10, "soak savings {:.3}", b.token_savings);
    assert!(
        b.token_savings > e.turn_limit_tuned.token_savings,
        "soak savings <= turn-limit"
    );
    assert!(
        b.precision_completed > e.base_precision_no_monitor,
        "soak precision did not improve"
    );
    assert!(e.immediate.fpr > b.fpr, "soak betting did not reduce FPR");
    assert!((1..=3).contains(&r.monitor_raw.len()));
    assert!(r.monitor_raw.iter().all(|c| (1..=3).contains(&c.len())));
    assert!(r.negative_control.train_tpr_minus_fpr < 0.15, "soak NC separation");
    println!(
        "soak 10x: monitor={} FPR={:.3} power={:.3} savings={:.3} precision={:.3}",
        r.monitor, b.fpr, b.power, b.token_savings, b.precision_completed
    );
}

#[test]
fn thread_identity() {
    // The experiment is stateless/deterministic: 8 threads running the
    // same default experiment must agree exactly.
    let handles: Vec<_> = (0..8)
        .map(|_| std::thread::spawn(run_experiment_default))
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for r in &results[1..] {
        assert_eq!(r.monitor, results[0].monitor);
        assert!((r.eval.betting.fpr - results[0].eval.betting.fpr).abs() < 1e-12);
        assert!((r.eval.betting.power - results[0].eval.betting.power).abs() < 1e-12);
        assert!((r.eval.q - results[0].eval.q).abs() < 1e-12);
        assert_eq!(
            r.negative_control.monitor,
            results[0].negative_control.monitor
        );
    }
}
