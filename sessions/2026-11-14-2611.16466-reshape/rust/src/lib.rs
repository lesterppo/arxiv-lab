//! ReSHAPE: revising a planning agent's search space (arXiv:2611.16466).
//!
//! Line-for-line port of the Python mechanism in `../reshape.py`, no
//! external crates. Same SplitMix64 RNG, same draw order, same per
//! (instance, strategy) seeding, so RNG streams and experiment results
//! reproduce the Python run exactly (float aggregates may differ by 1 ulp
//! where Python's `sum()` uses compensated summation; see README).

use std::collections::BTreeMap;
use std::rc::Rc;

// ---------------------------------------------------------------- RNG
pub const MASK64: u64 = u64::MAX;
const PHI: u64 = 0x9E3779B97F4A7C15;

/// SplitMix64. `next_u64` returns the next 64-bit draw.
#[derive(Clone, Debug)]
pub struct Rng {
    pub state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed & MASK64 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(PHI);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// 53-bit precision float in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }

    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Inclusive lo, exclusive hi.
    pub fn randint(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % ((hi - lo) as u64)) as usize
    }

    /// Fisher-Yates, returns a shuffled copy.
    pub fn shuffle<T: Clone>(&mut self, xs: &[T]) -> Vec<T> {
        let mut xs = xs.to_vec();
        for i in (1..xs.len()).rev() {
            let j = self.randint(0, i + 1);
            xs.swap(i, j);
        }
        xs
    }

    /// Partial Fisher-Yates: the first k entries of a shuffled copy.
    pub fn sample_without_replacement<T: Clone>(&mut self, xs: &[T], k: usize) -> Vec<T> {
        let mut xs = xs.to_vec();
        let k = k.min(xs.len());
        for i in 0..k {
            let j = i + self.randint(0, xs.len() - i);
            xs.swap(i, j);
        }
        xs.truncate(k);
        xs
    }
}

// ---------------------------------------------------------------- domains
/// Levels are ordered best-first; `values` give the per-level utility
/// value. A stated constraint is "value >= 1" (exclude the worst level).
/// Value rows are padded with 0 to length 3; only indices below
/// `levels[d]` are ever read.
#[derive(Clone, Copy, Debug)]
pub struct Domain {
    pub name: &'static str,
    pub attrs: [&'static str; 5],
    pub levels: [usize; 5],
    pub values: [[i32; 3]; 5],
}

pub const TRAVEL: Domain = Domain {
    name: "travel",
    attrs: ["price", "duration", "stops", "pace", "lodging"],
    levels: [3, 3, 2, 2, 3],
    values: [[2, 1, 0], [2, 1, 0], [1, 0, 0], [1, 0, 0], [2, 1, 0]],
};

pub const SHOPPING: Domain = Domain {
    name: "shopping",
    attrs: ["price", "brand", "delivery", "quality", "warranty"],
    levels: [3, 3, 2, 3, 2],
    values: [[2, 1, 0], [2, 1, 0], [1, 0, 0], [2, 1, 0], [1, 0, 0]],
};

pub fn domain(name: &str) -> &'static Domain {
    match name {
        "travel" => &TRAVEL,
        "shopping" => &SHOPPING,
        _ => panic!("unknown domain {name}"),
    }
}

pub const N_ATTRS: usize = 5;
pub const N_CONSTRAINED: usize = 3;
pub const N_ACCEPTABLE: usize = 22;
pub const BONUS_FRAC: f64 = 0.25;
pub const PLANNER_K: usize = 2;
pub const SHOW_K: usize = 8;

/// All plans as level-index tuples, lexicographic order.
pub fn universe(dom: &Domain) -> Vec<[usize; 5]> {
    let mut plans = Vec::new();
    for a in 0..dom.levels[0] {
        for b in 0..dom.levels[1] {
            for c in 0..dom.levels[2] {
                for d in 0..dom.levels[3] {
                    for e in 0..dom.levels[4] {
                        plans.push([a, b, c, d, e]);
                    }
                }
            }
        }
    }
    plans
}

fn value(dom: &Domain, attr: usize, level: usize) -> i32 {
    dom.values[attr][level]
}

fn max_value(dom: &Domain, attr: usize) -> i32 {
    dom.values[attr][0]
}

// ---------------------------------------------------------------- instances
pub type PlanSet = Vec<bool>; // indexed by plan index; true = member

#[derive(Clone, Debug)]
pub struct Instance {
    pub idx: usize,
    pub domain: &'static Domain,
    pub plans: Rc<Vec<[usize; 5]>>,
    pub weights: [f64; 5],
    pub constrained: Vec<usize>,
    pub sacrifice: usize,
    pub compensating: Vec<usize>,
    pub bonus: f64,
    pub acceptable: PlanSet,
    pub spec0: PlanSet,
}

impl Instance {
    /// The hidden trade-off: worst level on the sacrifice dimension,
    /// best level on every compensating attribute.
    pub fn tradeoff_holds(&self, plan: &[usize; 5]) -> bool {
        if plan[self.sacrifice] != self.domain.levels[self.sacrifice] - 1 {
            return false;
        }
        self.compensating.iter().all(|&a| plan[a] == 0)
    }

    pub fn utility(&self, plan: &[usize; 5]) -> f64 {
        let mut u = 0.0;
        for d in 0..N_ATTRS {
            u += self.weights[d] * value(self.domain, d, plan[d]) as f64;
        }
        if self.tradeoff_holds(plan) {
            u += self.bonus;
        }
        u
    }

    pub fn in_spec0(&self, plan: &[usize; 5]) -> bool {
        self.constrained
            .iter()
            .all(|&d| value(self.domain, d, plan[d]) >= 1)
    }
}

pub fn gen_instances(dom: &'static Domain, n: usize, seed: u64) -> Vec<Instance> {
    gen_instances_with_bonus(dom, n, seed, BONUS_FRAC)
}

pub fn gen_instances_with_bonus(
    dom: &'static Domain,
    n: usize,
    seed: u64,
    bonus_frac: f64,
) -> Vec<Instance> {
    let mut rng = Rng::new(seed);
    let plans = Rc::new(universe(dom));
    let mut instances = Vec::with_capacity(n);
    for idx in 0..n {
        let mut raw = [0.0f64; N_ATTRS];
        for d in 0..N_ATTRS {
            raw[d] = 0.5 + rng.uniform();
        }
        let total: f64 = raw.iter().sum();
        let mut weights = [0.0f64; N_ATTRS];
        for d in 0..N_ATTRS {
            weights[d] = raw[d] / total;
        }
        let order: Vec<usize> = (0..N_ATTRS).collect();
        let mut constrained = rng.shuffle(&order);
        constrained.truncate(N_CONSTRAINED);
        constrained.sort_unstable();
        // The trade-off sacrifices the constrained attribute the user
        // cares most about (highest weight); ties -> lowest index.
        let mut sacrifice = constrained[0];
        for &a in &constrained[1..] {
            if weights[a] > weights[sacrifice]
                || (weights[a] == weights[sacrifice] && a < sacrifice)
            {
                sacrifice = a;
            }
        }
        let others: Vec<usize> = (0..N_ATTRS).filter(|&a| a != sacrifice).collect();
        let n_comp = 1 + rng.randint(0, 2);
        let mut compensating = rng.shuffle(&others);
        compensating.truncate(n_comp);
        compensating.sort_unstable();
        let u_max: f64 = (0..N_ATTRS)
            .map(|a| weights[a] * max_value(dom, a) as f64)
            .sum();
        let bonus = bonus_frac * u_max;
        let mut inst = Instance {
            idx,
            domain: dom,
            plans: Rc::clone(&plans),
            weights,
            constrained,
            sacrifice,
            compensating,
            bonus,
            acceptable: vec![false; plans.len()],
            spec0: vec![false; plans.len()],
        };
        let mut utils: Vec<(f64, usize)> = plans
            .iter()
            .enumerate()
            .map(|(i, p)| (inst.utility(p), i))
            .collect();
        // Rank by utility desc, plan index asc: exact, tie-safe.
        utils.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap()
                .then_with(|| a.1.cmp(&b.1))
        });
        let mut acceptable = vec![false; plans.len()];
        for &(_, i) in utils.iter().take(N_ACCEPTABLE) {
            acceptable[i] = true;
        }
        let mut spec0 = vec![false; plans.len()];
        for (i, p) in plans.iter().enumerate() {
            if inst.in_spec0(p) {
                spec0[i] = true;
            }
        }
        inst.acceptable = acceptable;
        inst.spec0 = spec0;
        instances.push(inst);
    }
    instances
}

// ---------------------------------------------------------------- revision strategies
/// Stated flexibility for an attribute, from its marginal weight.
pub fn flexibility(weight: f64) -> f64 {
    (0.65 - 1.5 * weight).clamp(0.05, 0.65)
}

pub fn revise_ask(inst: &Instance, rng: &mut Rng, n_questions: Option<usize>) -> PlanSet {
    let asked: &[usize] = match n_questions {
        Some(q) => &inst.constrained[..q.min(inst.constrained.len())],
        None => &inst.constrained,
    };
    let mut flexible: Vec<usize> = Vec::new();
    for &a in asked {
        if rng.bernoulli(flexibility(inst.weights[a])) {
            flexible.push(a);
        }
    }
    if flexible.is_empty() {
        return inst.spec0.clone();
    }
    // Relax the dimension the user called most flexible; ties -> first asked.
    let mut relax = flexible[0];
    for &a in &flexible[1..] {
        let fa = flexibility(inst.weights[a]);
        let fr = flexibility(inst.weights[relax]);
        if fa > fr
            || (fa == fr
                && inst.constrained.iter().position(|&x| x == a).unwrap()
                    < inst.constrained.iter().position(|&x| x == relax).unwrap())
        {
            relax = a;
        }
    }
    let keep: Vec<usize> = inst
        .constrained
        .iter()
        .copied()
        .filter(|&a| a != relax)
        .collect();
    inst.plans
        .iter()
        .map(|p| {
            keep.iter()
                .all(|&a| value(inst.domain, a, p[a]) >= 1)
        })
        .collect()
}

fn candidate_conditions(inst: &Instance, d: usize) -> Vec<Vec<usize>> {
    let others: Vec<usize> = (0..N_ATTRS).filter(|&a| a != d).collect();
    let mut conds: Vec<Vec<usize>> = vec![Vec::new()];
    for &a in &others {
        conds.push(vec![a]);
    }
    for (i, &a) in others.iter().enumerate() {
        for &b in &others[i + 1..] {
            conds.push(vec![a, b]);
        }
    }
    let _ = inst;
    conds
}

pub fn revise_show(inst: &Instance, rng: &mut Rng, k_show: usize) -> PlanSet {
    // Neighbourhood: plans violating exactly one S0 constraint.
    let mut neighbourhood: Vec<usize> = Vec::new();
    for (i, p) in inst.plans.iter().enumerate() {
        let bad = inst
            .constrained
            .iter()
            .filter(|&&a| value(inst.domain, a, p[a]) < 1)
            .count();
        if bad == 1 {
            neighbourhood.push(i);
        }
    }
    let shown = rng.sample_without_replacement(&neighbourhood, k_show);
    let spec0 = &inst.spec0;

    let score = |space: &PlanSet| -> (f64, f64) {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut f_n = 0usize;
        for &i in &shown {
            if inst.acceptable[i] && space[i] {
                tp += 1;
            } else if !inst.acceptable[i] && space[i] {
                fp += 1;
            } else if inst.acceptable[i] && !space[i] {
                f_n += 1;
            }
        }
        if tp == 0 {
            return (0.0, 0.0);
        }
        let prec = tp as f64 / (tp + fp) as f64;
        let rec = tp as f64 / (tp + f_n) as f64;
        (2.0 * prec * rec / (prec + rec), prec)
    };

    let mut best_space = spec0.clone();
    let mut best_key: (f64, f64, i64) = (0.0, 0.0, 0);
    for &d in &inst.constrained {
        for cond in candidate_conditions(inst, d) {
            let region: PlanSet = inst
                .plans
                .iter()
                .map(|p| {
                    value(inst.domain, d, p[d]) < 1
                        && cond.iter().all(|&a| p[a] == 0)
                })
                .collect();
            let space: PlanSet = spec0
                .iter()
                .zip(region.iter())
                .map(|(a, b)| *a || *b)
                .collect();
            let (f1, prec) = score(&space);
            let key = (f1, prec, -(space.iter().filter(|&&b| b).count() as i64));
            let better = key.0 > best_key.0
                || (key.0 == best_key.0 && key.1 > best_key.1)
                || (key.0 == best_key.0 && key.1 == best_key.1 && key.2 > best_key.2);
            if better {
                best_key = key;
                best_space = space;
            }
        }
    }
    best_space
}

pub fn revise_oracle(inst: &Instance) -> PlanSet {
    inst.spec0
        .iter()
        .zip(inst.plans.iter())
        .map(|(s, p)| *s || inst.tradeoff_holds(p))
        .collect()
}

pub const STRATEGIES: [&str; 4] = ["none", "ask", "show", "oracle"];

pub fn revise(inst: &Instance, strategy: &str, rng: &mut Rng, k_show: usize,
              n_questions: Option<usize>) -> PlanSet {
    match strategy {
        "none" => inst.spec0.clone(),
        "ask" => revise_ask(inst, rng, n_questions),
        "show" => revise_show(inst, rng, k_show),
        "oracle" => revise_oracle(inst),
        _ => panic!("unknown strategy {strategy}"),
    }
}

// ---------------------------------------------------------------- planner + metrics
pub fn plan(inst: &Instance, space: &PlanSet, rng: &mut Rng, k: usize) -> bool {
    let members: Vec<usize> = space
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    if members.is_empty() {
        return false;
    }
    let drawn = rng.sample_without_replacement(&members, k);
    drawn.iter().any(|&i| inst.acceptable[i])
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpaceMetrics {
    pub admission: f64,
    pub precision: f64,
    pub f1: f64,
    pub n_space: usize,
    pub n_acceptable: usize,
    pub n_admitted: usize,
}

pub fn space_metrics(inst: &Instance, space: &PlanSet) -> SpaceMetrics {
    let inter = space
        .iter()
        .zip(inst.acceptable.iter())
        .filter(|(s, a)| **s && **a)
        .count();
    let n_space = space.iter().filter(|&&b| b).count();
    let n_acceptable = inst.acceptable.iter().filter(|&&b| b).count();
    let admission = if n_acceptable > 0 {
        inter as f64 / n_acceptable as f64
    } else {
        0.0
    };
    let precision = if n_space > 0 {
        inter as f64 / n_space as f64
    } else {
        0.0
    };
    let f1 = if precision + admission > 0.0 {
        2.0 * precision * admission / (precision + admission)
    } else {
        0.0
    };
    SpaceMetrics {
        admission,
        precision,
        f1,
        n_space,
        n_acceptable,
        n_admitted: inter,
    }
}

pub fn strategy_seed(seed: u64, idx: usize, strategy: &str) -> u64 {
    let salt: u64 = match strategy {
        "none" => 0x11,
        "ask" => 0x22,
        "show" => 0x33,
        "oracle" => 0x44,
        _ => panic!("unknown strategy {strategy}"),
    };
    seed ^ ((idx as u64).wrapping_mul(PHI)) ^ salt
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Agg {
    pub n_instances: usize,
    pub admission: f64,
    pub precision: f64,
    pub f1: f64,
    pub planner_success: f64,
    pub mean_space_size: f64,
}

pub struct ExperimentOptions {
    pub planner_k: usize,
    pub show_k: usize,
    pub ask_q: Option<usize>,
}

impl Default for ExperimentOptions {
    fn default() -> Self {
        ExperimentOptions {
            planner_k: PLANNER_K,
            show_k: SHOW_K,
            ask_q: None,
        }
    }
}

pub fn run_experiment(
    instances: &[Instance],
    seed: u64,
    opts: &ExperimentOptions,
) -> BTreeMap<&'static str, Agg> {
    let mut out = BTreeMap::new();
    for &strategy in STRATEGIES.iter() {
        let mut admits = Vec::new();
        let mut precs = Vec::new();
        let mut f1s = Vec::new();
        let mut succs = Vec::new();
        let mut sizes = Vec::new();
        for inst in instances {
            let mut rng = Rng::new(strategy_seed(seed, inst.idx, strategy));
            let space = revise(inst, strategy, &mut rng, opts.show_k, opts.ask_q);
            let m = space_metrics(inst, &space);
            admits.push(m.admission);
            precs.push(m.precision);
            f1s.push(m.f1);
            sizes.push(m.n_space as f64);
            succs.push(if plan(inst, &space, &mut rng, opts.planner_k) {
                1.0
            } else {
                0.0
            });
        }
        let n = instances.len() as f64;
        out.insert(
            strategy,
            Agg {
                n_instances: instances.len(),
                admission: admits.iter().sum::<f64>() / n,
                precision: precs.iter().sum::<f64>() / n,
                f1: f1s.iter().sum::<f64>() / n,
                planner_success: succs.iter().sum::<f64>() / n,
                mean_space_size: sizes.iter().sum::<f64>() / n,
            },
        );
    }
    out
}

pub fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let cov: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let vx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let vy: f64 = ys.iter().map(|y| (y - my) * (y - my)).sum();
    if vx == 0.0 || vy == 0.0 {
        0.0
    } else {
        cov / (vx * vy).sqrt()
    }
}
