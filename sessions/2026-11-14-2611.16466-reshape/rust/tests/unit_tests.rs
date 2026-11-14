//! Unit tests for the ReSHAPE Rust port (arXiv:2611.16466).
//! Each mechanism gets a direct test; RNG vectors are pinned to the
//! Python SplitMix64 streams.

use reshape::{
    domain, flexibility, gen_instances, plan, revise, revise_oracle, space_metrics,
    strategy_seed, universe, Rng, BONUS_FRAC, N_ACCEPTABLE, PLANNER_K, SHOW_K,
    STRATEGIES,
};

#[test]
fn rng_vectors_match_python() {
    // Pinned from the frozen Python module (RNG(seed).next_u64() x5).
    let cases: [(u64, [u64; 5]); 3] = [
        (
            0,
            [
                16294208416658607535,
                7960286522194355700,
                487617019471545679,
                17909611376780542444,
                1961750202426094747,
            ],
        ),
        (
            1,
            [
                10451216379200822465,
                13757245211066428519,
                17911839290282890590,
                8196980753821780235,
                8195237237126968761,
            ],
        ),
        (
            20261114,
            [
                8826110057919892546,
                15535437939828757043,
                5096786482040423729,
                9341220242798358144,
                6070102674540371777,
            ],
        ),
    ];
    for (seed, expected) in cases {
        let mut rng = Rng::new(seed);
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(rng.next_u64(), *want, "seed {} draw {}", seed, i);
        }
    }
}

#[test]
fn uniform_range_and_first_values() {
    let mut rng = Rng::new(20261114);
    let first = [0.4784643849696454, 0.8421777782438034, 0.27629734882614887];
    for want in first {
        let u = rng.uniform();
        assert!((u - want).abs() < 1e-15, "uniform {} vs {}", u, want);
    }
    let mut rng = Rng::new(7);
    for _ in 0..1000 {
        let u = rng.uniform();
        assert!(u >= 0.0 && u < 1.0);
    }
}

#[test]
fn randint_bounds() {
    let mut rng = Rng::new(3);
    for _ in 0..1000 {
        let v = rng.randint(0, 3);
        assert!(v < 3);
        let w = rng.randint(2, 5);
        assert!((2..5).contains(&w));
    }
}

#[test]
fn shuffle_and_sample_are_deterministic() {
    let xs: Vec<usize> = (0..10).collect();
    let mut a = Rng::new(9);
    let mut b = Rng::new(9);
    assert_eq!(a.shuffle(&xs), b.shuffle(&xs));
    assert_eq!(
        a.sample_without_replacement(&xs, 4),
        b.sample_without_replacement(&xs, 4)
    );
    let s = Rng::new(9).sample_without_replacement(&xs, 4);
    assert_eq!(s.len(), 4);
    let mut sorted = s.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 4, "sampled plans are distinct");
    assert!(Rng::new(9).sample_without_replacement(&xs, 100).len() == 10);
}

#[test]
fn domains_have_108_plans() {
    for name in ["travel", "shopping"] {
        let dom = domain(name);
        let plans = universe(dom);
        assert_eq!(plans.len(), 108);
        assert_eq!(plans[0], [0, 0, 0, 0, 0]);
        let last: [usize; 5] = [
            dom.levels[0] - 1,
            dom.levels[1] - 1,
            dom.levels[2] - 1,
            dom.levels[3] - 1,
            dom.levels[4] - 1,
        ];
        assert_eq!(plans[107], last);
    }
}

#[test]
fn instance_structure() {
    let dom = domain("travel");
    let insts = gen_instances(dom, 50, 20261114);
    assert_eq!(insts.len(), 50);
    for inst in &insts {
        assert_eq!(inst.acceptable.iter().filter(|&&b| b).count(), N_ACCEPTABLE);
        assert!(!inst.spec0.iter().all(|&b| !b), "S0 is non-empty");
        assert_eq!(inst.constrained.len(), 3);
        assert!(inst.constrained.contains(&inst.sacrifice));
        let wsum: f64 = inst.weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-12, "weights sum to 1, got {}", wsum);
        assert!((inst.bonus - BONUS_FRAC * (0..5).map(|a| inst.weights[a] * inst.domain.values[a][0] as f64).sum::<f64>()).abs() < 1e-15);
    }
}

#[test]
fn flexibility_bounds_and_ordering() {
    // Flexibility falls with weight, and stays in [0.05, 0.65].
    assert!((flexibility(0.0) - 0.65).abs() < 1e-12);
    assert!((flexibility(1.0) - 0.05).abs() < 1e-12);
    assert!(flexibility(0.1) > flexibility(0.3));
    for w in [0.0, 0.05, 0.2, 0.5, 1.0] {
        let f = flexibility(w);
        assert!((0.05..=0.65).contains(&f));
    }
}

#[test]
fn oracle_admits_every_tradeoff_plan() {
    for name in ["travel", "shopping"] {
        let insts = gen_instances(domain(name), 100, 42);
        for inst in &insts {
            let space = revise_oracle(inst);
            for (i, p) in inst.plans.iter().enumerate() {
                assert!(!inst.tradeoff_holds(p) || space[i]);
            }
            // Oracle space is a superset of S0.
            assert!(inst
                .spec0
                .iter()
                .zip(space.iter())
                .all(|(a, b)| !*a || *b));
        }
    }
}

#[test]
fn space_metrics_hand_check() {
    let insts = gen_instances(domain("travel"), 1, 20261114);
    let inst = &insts[0];
    let m = space_metrics(inst, &inst.spec0);
    let inter = inst
        .spec0
        .iter()
        .zip(inst.acceptable.iter())
        .filter(|(s, a)| **s && **a)
        .count();
    let n_spec0 = inst.spec0.iter().filter(|&&b| b).count();
    assert!((m.admission - inter as f64 / N_ACCEPTABLE as f64).abs() < 1e-12);
    assert!((m.precision - inter as f64 / n_spec0 as f64).abs() < 1e-12);
    assert!((0.0..=1.0).contains(&m.f1));
}

#[test]
fn planner_full_budget_property() {
    let insts = gen_instances(domain("shopping"), 50, 7);
    for inst in &insts {
        let mut rng = Rng::new(strategy_seed(7, inst.idx, "show"));
        let space = revise(inst, "show", &mut rng, SHOW_K, None);
        let n = space.iter().filter(|&&b| b).count();
        let has_acceptable = space
            .iter()
            .zip(inst.acceptable.iter())
            .any(|(s, a)| *s && *a);
        assert_eq!(plan(inst, &space, &mut Rng::new(5), n), has_acceptable);
        assert!(!plan(inst, &vec![false; 108], &mut Rng::new(5), PLANNER_K));
    }
}

#[test]
fn revised_spaces_contain_spec0() {
    for name in ["travel", "shopping"] {
        let insts = gen_instances(domain(name), 50, 1);
        for inst in &insts {
            for s in STRATEGIES {
                let mut rng = Rng::new(strategy_seed(1, inst.idx, s));
                let space = revise(inst, s, &mut rng, SHOW_K, None);
                assert!(inst
                    .spec0
                    .iter()
                    .zip(space.iter())
                    .all(|(a, b)| !*a || *b));
            }
        }
    }
}

#[test]
fn ask_never_relaxes_more_than_one_dimension() {
    // ASK drops at most one constrained dimension: |S_ask| is either |S0|
    // or the size of S0 with exactly one constraint removed.
    let insts = gen_instances(domain("travel"), 200, 11);
    for inst in &insts {
        let mut rng = Rng::new(strategy_seed(11, inst.idx, "ask"));
        let space = revise(inst, "ask", &mut rng, SHOW_K, None);
        let n0 = inst.spec0.iter().filter(|&&b| b).count();
        let n = space.iter().filter(|&&b| b).count();
        assert!(n >= n0, "ASK only ever adds plans");
        // Removing one constraint at most triples the space (3-level attr).
        assert!(n <= 3 * n0 + 1, "n={} n0={}", n, n0);
    }
}
