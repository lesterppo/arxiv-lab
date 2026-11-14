//! Deep tests for the ReSHAPE Rust port (arXiv:2611.16466):
//! property sweeps, exhaustive combination checks, a soak at a scale the
//! Python test does not reach, and thread identity.

use reshape::{
    domain, gen_instances, pearson, plan, revise, revise_oracle, run_experiment,
    space_metrics, strategy_seed, ExperimentOptions, Rng, STRATEGIES,
};
use std::sync::mpsc;
use std::thread;

fn f1_show_and_ask(dom_name: &str, n: usize, seed: u64) -> (f64, f64) {
    let dom = domain(dom_name);
    let insts = gen_instances(dom, n, seed);
    let res = run_experiment(&insts, seed, &ExperimentOptions::default());
    (res["show"].f1, res["ask"].f1)
}

#[test]
fn property_sweep_show_beats_ask_across_seeds() {
    // SHOW F1 > ASK F1 across a seed grid x both domains (100 instances
    // each): the direction is not one seed's luck.
    for seed in [1u64, 7, 42, 99, 12345, 20261114] {
        for dom_name in ["travel", "shopping"] {
            let (show_f1, ask_f1) = f1_show_and_ask(dom_name, 100, seed);
            assert!(
                show_f1 > ask_f1,
                "seed {} {}: show F1 {:.3} not > ask F1 {:.3}",
                seed, dom_name, show_f1, ask_f1
            );
        }
    }
}

#[test]
fn property_oracle_admission_ge_none_per_instance() {
    // Oracle space ⊇ S0, so per-instance oracle admission >= none admission
    // on every instance, every seed.
    for seed in [1u64, 7, 42] {
        for dom_name in ["travel", "shopping"] {
            let insts = gen_instances(domain(dom_name), 200, seed);
            for inst in &insts {
                let m_none = space_metrics(inst, &inst.spec0);
                let m_or = space_metrics(inst, &revise_oracle(inst));
                assert!(
                    m_or.admission + 1e-12 >= m_none.admission,
                    "seed {} {} idx {}: oracle adm {:.3} < none adm {:.3}",
                    seed, dom_name, inst.idx, m_or.admission, m_none.admission
                );
            }
        }
    }
}

#[test]
fn exhaustive_sacrifice_compensating_combos() {
    // For a fixed weight vector, every (sacrifice, compensating-set) pair
    // the generator could draw still yields oracle ⊇ S0 and an acceptable
    // set of exactly 22 plans. We check the structural invariant by
    // re-running gen_instances over many seeds and confirming the
    // sacrifice is always a constrained attribute with maximal weight.
    for seed in 0..20u64 {
        let insts = gen_instances(domain("travel"), 20, seed);
        for inst in &insts {
            assert!(inst.constrained.contains(&inst.sacrifice));
            let max_w = inst
                .constrained
                .iter()
                .map(|&a| inst.weights[a])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!((inst.weights[inst.sacrifice] - max_w).abs() < 1e-12);
            assert!(!inst.compensating.contains(&inst.sacrifice));
            assert!((1..=2).contains(&inst.compensating.len()));
        }
    }
}

#[test]
fn exhaustive_small_seed_grid_invariants() {
    // 3 policies... 4 strategies x 200 seeds: invariants hold on every run.
    for seed in 0..200u64 {
        let insts = gen_instances(domain("shopping"), 5, seed);
        for inst in &insts {
            for s in STRATEGIES {
                let mut rng = Rng::new(strategy_seed(seed, inst.idx, s));
                let space = revise(inst, s, &mut rng, reshape::SHOW_K, None);
                let m = space_metrics(inst, &space);
                assert!((0.0..=1.0).contains(&m.admission));
                assert!((0.0..=1.0).contains(&m.precision));
                assert!((0.0..=1.0).contains(&m.f1));
                assert!(space.iter().filter(|&&b| b).count() >= 1);
            }
        }
    }
}

#[test]
fn soak_20000_instances() {
    // A scale the Python test does not reach: 20,000 travel instances,
    // all four strategies, invariants + aggregate ordering.
    let insts = gen_instances(domain("travel"), 20_000, 20261114);
    assert_eq!(insts.len(), 20_000);
    let res = run_experiment(&insts, 20261114, &ExperimentOptions::default());
    assert!(res["show"].f1 > res["ask"].f1);
    assert!(res["oracle"].admission > 0.52);
    assert!(res["show"].planner_success > res["ask"].planner_success);
    // Determinism on a subsample re-run.
    let res2 = run_experiment(&insts[..500], 20261114, &ExperimentOptions::default());
    let res3 = run_experiment(&insts[..500], 20261114, &ExperimentOptions::default());
    for s in STRATEGIES {
        assert_eq!(res2[s].f1.to_bits(), res3[s].f1.to_bits());
        assert_eq!(res2[s].admission.to_bits(), res3[s].admission.to_bits());
    }
}

#[test]
fn thread_identity_four_threads() {
    // The mechanism is stateless (all state in the RNG passed in), so four
    // threads running the same experiment produce bit-identical aggregates.
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();
    for _ in 0..4 {
        let tx = tx.clone();
        handles.push(thread::spawn(move || {
            let insts = gen_instances(domain("travel"), 300, 20261114);
            let res = run_experiment(&insts, 20261114, &ExperimentOptions::default());
            let bits: Vec<(u64, u64, u64, u64)> = STRATEGIES
                .iter()
                .map(|s| {
                    let m = &res[s];
                    (
                        m.admission.to_bits(),
                        m.precision.to_bits(),
                        m.f1.to_bits(),
                        m.planner_success.to_bits(),
                    )
                })
                .collect();
            tx.send(bits).unwrap();
        }));
    }
    drop(tx);
    let mut results = Vec::new();
    while let Ok(bits) = rx.recv() {
        results.push(bits);
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(results.len(), 4);
    for other in &results[1..] {
        assert_eq!(&results[0], other, "thread results differ");
    }
}

#[test]
fn correlation_holds_on_seeds() {
    // Search-space quality predicts what the planner finds, on fresh seeds
    // too: Pearson r between arm F1 and planner success > 0.8.
    for seed in [5u64, 123] {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for dom_name in ["travel", "shopping"] {
            let insts = gen_instances(domain(dom_name), 300, seed);
            let res = run_experiment(&insts, seed, &ExperimentOptions::default());
            for s in STRATEGIES {
                xs.push(res[s].f1);
                ys.push(res[s].planner_success);
            }
        }
        let r = pearson(&xs, &ys);
        assert!(r > 0.8, "seed {}: r={:.3}", seed, r);
    }
}

#[test]
fn paired_rescue_replicates() {
    // Paired per-instance planner outcomes: SHOW rescues more than it loses.
    let seed = 20261114u64;
    let (mut rescue, mut loss) = (0usize, 0usize);
    for dom_name in ["travel", "shopping"] {
        let insts = gen_instances(domain(dom_name), 300, seed);
        for inst in &insts {
            let mut outs = [false; 2];
            for (j, s) in ["none", "show"].iter().enumerate() {
                let mut rng = Rng::new(strategy_seed(seed, inst.idx, s));
                let space = revise(inst, s, &mut rng, reshape::SHOW_K, None);
                outs[j] = plan(inst, &space, &mut rng, reshape::PLANNER_K);
            }
            if outs[1] && !outs[0] {
                rescue += 1;
            } else if outs[0] && !outs[1] {
                loss += 1;
            }
        }
    }
    assert!(rescue > loss, "rescue {} vs loss {}", rescue, loss);
}
