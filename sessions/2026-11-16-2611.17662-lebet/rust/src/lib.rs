//! LeBet: learning interpretable rules for reliable agent intervention
//! (arXiv:2611.17662, "Learn then Bet: Learning Interpretable Rules for
//! Reliable Agent Intervention").
//!
//! Faithful Rust port of `arxiv_lab/agent/lebet.py` (stdlib-only Python).
//! The Python module is the primary verdict carrier; this port exists to
//! cross-check it. Every RNG draw, predicate, learner move and betting
//! update follows the Python code in the same order, so the same seed
//! reproduces the same streams, trajectories, monitor and metrics
//! bit-for-bit (see `tests/parity_gen.py` and the `parity` subcommand).
//!
//! Std only. No network, no model API.

use std::collections::{BTreeSet, HashMap};

// ---------------------------------------------------------------- RNG
const MASK64: u64 = u64::MAX;

/// One SplitMix64 step; returns (value, new_state).
pub fn splitmix64(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z, state)
}

/// Deterministic RNG (SplitMix64); uniform() in [0, 1), randint, choice.
/// Mirrors `lebet.Rng` exactly, including the `% n` randint (not a
/// rejection sampler) and the Box-Muller gauss.
#[derive(Clone, Debug)]
pub struct Rng {
    pub state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed & MASK64 }
    }

    pub fn next_u64(&mut self) -> u64 {
        let (v, s) = splitmix64(self.state);
        self.state = s;
        v
    }

    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform int in [0, n).
    pub fn randint(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn choice<T: Clone>(&mut self, seq: &[T]) -> T {
        seq[self.randint(seq.len() as u64) as usize].clone()
    }

    /// Box-Muller normal.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------- trajectories
// A step: (action, ok, state_hash, obs_fp, tokens)
//   action: tool id 0..5; ok: bool; state_hash: u64; obs_fp: u64; tokens: i64
pub const N_ACTIONS: u64 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub action: u64,
    pub ok: bool,
    pub state: u64,
    pub obs: u64,
    pub tokens: i64,
}

#[derive(Clone, Debug)]
pub struct Traj {
    pub steps: Vec<Step>,
    /// true = SUCCESS (would eventually succeed if left alone),
    /// false = FAIL (will fail).
    pub label: bool,
}

/// Simulate one agent run. RNG consumption order mirrors
/// `lebet.gen_trajectory` exactly: subtype draw (fail only), then
/// gauss for n_steps, then state / prev_action / prev_obs, then per
/// step: repeat draw (only when a step exists), ok draw, stall draw,
/// conditional repeated-observation draw, tokens draw.
pub fn gen_trajectory(rng: &mut Rng, label: bool) -> Traj {
    let (n_steps, p_repeat, p_err, p_stall) = if label {
        // success: mostly clean, but with a long tail and occasional noise
        (std::cmp::max(5, rng.gauss(16.0, 6.0) as i64), 0.12, 0.08, 0.06)
    } else if rng.uniform() < 0.6 {
        // fail, subtype A (error-prone): long, repetitive, error-prone, stalls
        (std::cmp::max(8, rng.gauss(28.0, 5.0) as i64), 0.38, 0.30, 0.30)
    } else {
        // fail, subtype B (silent looper): fails by looping/stalling with
        // almost no errors -- no single error predicate can catch it.
        (std::cmp::max(8, rng.gauss(30.0, 5.0) as i64), 0.55, 0.06, 0.45)
    };
    let mut steps: Vec<Step> = Vec::with_capacity(n_steps as usize);
    let mut state = rng.randint(1 << 30);
    let mut prev_action = rng.randint(N_ACTIONS);
    let mut prev_obs = rng.randint(1 << 30);
    for _ in 0..n_steps {
        let action = if !steps.is_empty() && rng.uniform() < p_repeat {
            prev_action
        } else {
            rng.randint(N_ACTIONS)
        };
        let ok = rng.uniform() >= p_err;
        if rng.uniform() < p_stall {
            // state unchanged: a stall step
        } else {
            state = (state.wrapping_mul(2654435761).wrapping_add(97)) & ((1 << 30) - 1);
        }
        let obs = if !ok
            && !steps.is_empty()
            && !steps.last().unwrap().ok
            && rng.uniform() < 0.5
        {
            prev_obs // repeated failed observation
        } else {
            rng.randint(1 << 30)
        };
        let tokens = 250 + rng.randint(150) as i64;
        steps.push(Step {
            action,
            ok,
            state,
            obs,
            tokens,
        });
        prev_action = action;
        prev_obs = obs;
    }
    Traj { steps, label }
}

pub fn gen_dataset(rng: &mut Rng, n_success: usize, n_fail: usize) -> Vec<Traj> {
    let mut data: Vec<Traj> = (0..n_success)
        .map(|_| gen_trajectory(rng, true))
        .collect();
    data.extend((0..n_fail).map(|_| gen_trajectory(rng, false)));
    // deterministic shuffle (Fisher-Yates with our RNG)
    let len = data.len();
    for i in (1..len).rev() {
        let j = rng.randint((i + 1) as u64) as usize;
        data.swap(i, j);
    }
    data
}

// ---------------------------------------------------------------- predicates
// Each predicate maps a prefix (slice of steps) to bool. Index order is
// the canonical predicate order used by monitors and the Python module.
pub fn p_repeat3(prefix: &[Step]) -> bool {
    prefix.len() >= 3
        && prefix[prefix.len() - 1].action == prefix[prefix.len() - 2].action
        && prefix[prefix.len() - 2].action == prefix[prefix.len() - 3].action
}

pub fn p_err2(prefix: &[Step]) -> bool {
    prefix.len() >= 2 && !prefix[prefix.len() - 1].ok && !prefix[prefix.len() - 2].ok
}

pub fn p_err_rate(prefix: &[Step]) -> bool {
    let w = &prefix[prefix.len().saturating_sub(10)..];
    w.len() >= 5 && (w.iter().filter(|s| !s.ok).count() as f64) / (w.len() as f64) > 0.3
}

pub fn p_stalled(prefix: &[Step]) -> bool {
    prefix.len() >= 3
        && prefix[prefix.len() - 1].state == prefix[prefix.len() - 2].state
        && prefix[prefix.len() - 2].state == prefix[prefix.len() - 3].state
}

pub fn p_loop(prefix: &[Step]) -> bool {
    let mut seen: HashMap<u64, usize> = HashMap::new();
    for s in prefix {
        let c = seen.entry(s.state).or_insert(0);
        *c += 1;
        if *c >= 3 {
            return true;
        }
    }
    false
}

pub fn p_long(prefix: &[Step]) -> bool {
    prefix.len() > 20
}

pub fn p_repeat_tool_fail(prefix: &[Step]) -> bool {
    if prefix.len() < 2 || prefix[prefix.len() - 1].ok {
        return false;
    }
    let a = prefix[prefix.len() - 1].action;
    prefix.iter().filter(|s| s.action == a && !s.ok).count() >= 2
}

pub fn p_no_new_obs(prefix: &[Step]) -> bool {
    prefix.len() >= 2 && prefix[prefix.len() - 1].obs == prefix[prefix.len() - 2].obs
}

pub fn p_tok_high(prefix: &[Step]) -> bool {
    prefix.iter().map(|s| s.tokens).sum::<i64>() > 6000
}

pub fn p_same_action_5(prefix: &[Step]) -> bool {
    let w = &prefix[prefix.len().saturating_sub(8)..];
    if w.len() < 5 {
        return false;
    }
    let a = prefix[prefix.len() - 1].action;
    w.iter().filter(|s| s.action == a).count() >= 5
}

/// (name, predicate) in canonical index order.
pub const PREDICATES: [(&str, fn(&[Step]) -> bool); 10] = [
    ("repeat3", p_repeat3),
    ("err2", p_err2),
    ("err_rate", p_err_rate),
    ("stalled", p_stalled),
    ("loop", p_loop),
    ("long", p_long),
    ("repeat_tool_fail", p_repeat_tool_fail),
    ("no_new_obs", p_no_new_obs),
    ("tok_high", p_tok_high),
    ("same_action_5", p_same_action_5),
];
pub const N_PRED: usize = PREDICATES.len();

pub type Row = [bool; 10];
pub type Mat = Vec<Row>;

/// Precompute the [n_steps x N_PRED] boolean matrix for a trajectory.
pub fn prefix_matrix(traj: &Traj) -> Mat {
    let steps = &traj.steps;
    let mut mat: Mat = Vec::with_capacity(steps.len());
    for t in 1..=steps.len() {
        let prefix = &steps[..t];
        let mut row = [false; 10];
        for (i, (_, f)) in PREDICATES.iter().enumerate() {
            row[i] = f(prefix);
        }
        mat.push(row);
    }
    mat
}

// ---------------------------------------------------------------- monitor (OR of ANDs)
// A monitor is a Vec of conjunctions; a conjunction is a BTreeSet of
// predicate indices (BTreeSet iteration is sorted, like Python
// sorted(frozenset)). Fires on a prefix-row iff any conjunction's
// literals are all true. Empty conjunctions never fire (as in Python).
pub type Monitor = Vec<BTreeSet<usize>>;

pub fn monitor_fires_row(monitor: &Monitor, row: &Row) -> bool {
    monitor
        .iter()
        .any(|conj| !conj.is_empty() && conj.iter().all(|&i| row[i]))
}

/// First firing step (1-based) or None if it never fires.
pub fn monitor_fires_traj(monitor: &Monitor, mat: &Mat) -> Option<usize> {
    mat.iter()
        .position(|row| monitor_fires_row(monitor, row))
        .map(|t| t + 1)
}

/// mats: list of (matrix, label). Returns (tpr, fpr) at trajectory level.
pub fn monitor_tpr_fpr(monitor: &Monitor, mats: &[(Mat, bool)]) -> (f64, f64) {
    let (mut tp, mut fp, mut n_pos, mut n_neg) = (0u64, 0u64, 0u64, 0u64);
    for (mat, label) in mats {
        let fired = monitor_fires_traj(monitor, mat).is_some();
        if *label {
            n_neg += 1;
            fp += fired as u64;
        } else {
            n_pos += 1;
            tp += fired as u64;
        }
    }
    (
        if n_pos > 0 { tp as f64 / n_pos as f64 } else { 0.0 },
        if n_neg > 0 { fp as f64 / n_neg as f64 } else { 0.0 },
    )
}

pub fn monitor_complexity(monitor: &Monitor) -> usize {
    monitor.iter().map(|c| c.len()).sum()
}

/// Trajectory-level score (kept for parity with Python; the learner
/// uses `monitor_betting_score` instead).
pub fn monitor_score(monitor: &Monitor, mats: &[(Mat, bool)], lam: f64, complexity: f64) -> f64 {
    let (tpr, fpr) = monitor_tpr_fpr(monitor, mats);
    tpr - lam * fpr - complexity * monitor_complexity(monitor) as f64
}

pub fn random_conjunction(rng: &mut Rng, max_lits: usize) -> BTreeSet<usize> {
    let k = 1 + rng.randint(max_lits as u64);
    let mut lits = BTreeSet::new();
    while lits.len() < k as usize {
        lits.insert(rng.randint(N_PRED as u64) as usize);
    }
    lits
}

pub fn random_monitor(rng: &mut Rng, max_conj: usize, max_lits: usize) -> Monitor {
    let k = 1 + rng.randint(max_conj as u64);
    (0..k).map(|_| random_conjunction(rng, max_lits)).collect()
}

/// One stochastic-local-search edit, exploiting the Boolean structure.
/// The elif chain mirrors Python `_mutate` exactly, including the
/// fall-through to the fresh-conjunction branch when a move's guard
/// fails (e.g. move 2 at max_conj, move 3 with one conjunction).
pub fn mutate(rng: &mut Rng, monitor: &Monitor, max_conj: usize, max_lits: usize) -> Monitor {
    let mut m: Monitor = monitor.to_vec();
    let mv = rng.randint(6);
    if mv == 0 && !m.is_empty() {
        // add a literal to a random conjunction
        let ci = rng.randint(m.len() as u64) as usize;
        if m[ci].len() < max_lits {
            m[ci].insert(rng.randint(N_PRED as u64) as usize);
        }
    } else if mv == 1 && !m.is_empty() {
        // remove a literal (or the whole conjunction if it had one)
        let ci = rng.randint(m.len() as u64) as usize;
        if m[ci].len() > 1 {
            let sorted: Vec<usize> = m[ci].iter().copied().collect();
            let pick = sorted[rng.randint(sorted.len() as u64) as usize];
            m[ci].remove(&pick);
        } else {
            m.remove(ci);
        }
    } else if mv == 2 && m.len() < max_conj {
        // add a conjunction
        m.push(random_conjunction(rng, max_lits));
    } else if mv == 3 && m.len() > 1 {
        // remove a conjunction
        m.remove(rng.randint(m.len() as u64) as usize);
    } else if mv == 4 && !m.is_empty() {
        // replace a literal
        let ci = rng.randint(m.len() as u64) as usize;
        if !m[ci].is_empty() {
            let sorted: Vec<usize> = m[ci].iter().copied().collect();
            let pick = sorted[rng.randint(sorted.len() as u64) as usize];
            m[ci].remove(&pick);
            m[ci].insert(rng.randint(N_PRED as u64) as usize);
        }
    } else {
        // fresh random conjunction in place of one.
        // NOTE (Python evaluation-order trap): the Python line is
        //   monitor[rng.randint(len(monitor))] = set(_random_conjunction(rng, max_lits))
        // and CPython evaluates the RHS (the fresh conjunction) BEFORE the
        // subscript index, so the conjunction's draws come first. Drawing
        // ci first here would silently diverge the RNG stream.
        if !m.is_empty() {
            let fresh = random_conjunction(rng, max_lits);
            let ci = rng.randint(m.len() as u64) as usize;
            m[ci] = fresh;
        }
    }
    let out: Monitor = m.into_iter().filter(|c| !c.is_empty()).collect();
    if out.is_empty() {
        vec![BTreeSet::from([rng.randint(N_PRED as u64) as usize])]
    } else {
        out
    }
}

/// Stochastic local search for the OR-of-AND monitor. The objective is
/// the betting score on the training trajectories.
pub fn learn_monitor(
    rng: &mut Rng,
    mats: &[(Mat, bool)],
    restarts: usize,
    iters: usize,
    alpha: f64,
    bet_lam: f64,
) -> (Monitor, f64) {
    let mut best: Option<Monitor> = None;
    let mut best_score = -1e18f64;
    for _ in 0..restarts {
        let mut cur = random_monitor(rng, 3, 3);
        let mut cur_score = monitor_betting_score(&cur, mats, alpha, bet_lam);
        for _ in 0..iters {
            let cand = mutate(rng, &cur, 3, 3);
            let s = monitor_betting_score(&cand, mats, alpha, bet_lam);
            if s > cur_score || (s == cur_score && rng.uniform() < 0.1) {
                cur = cand;
                cur_score = s;
            }
        }
        if cur_score > best_score {
            best = Some(cur);
            best_score = cur_score;
        }
    }
    (best.expect("at least one restart"), best_score)
}

// ---------------------------------------------------------------- betting statistic
/// Per-step fire rate on SUCCESS trajectories (the H0 rate q).
pub fn baseline_fire_rate(monitor: &Monitor, mats: &[(Mat, bool)]) -> f64 {
    let (mut fires, mut steps) = (0u64, 0u64);
    for (mat, label) in mats {
        if !label {
            continue;
        }
        for row in mat {
            steps += 1;
            if monitor_fires_row(monitor, row) {
                fires += 1;
            }
        }
    }
    if steps > 0 {
        fires as f64 / steps as f64
    } else {
        0.0
    }
}

/// Wealth W_t under the fixed betting rule; returns the series.
pub fn betting_wealth_series(monitor: &Monitor, mat: &Mat, q: f64, lam: f64) -> Vec<f64> {
    let mut w = 1.0;
    let mut series = Vec::with_capacity(mat.len());
    for row in mat {
        let x = if monitor_fires_row(monitor, row) { 1.0 } else { 0.0 };
        w *= 1.0 + lam * (x - q) / (1.0 - q);
        series.push(w);
    }
    series
}

/// First step with wealth >= 1/alpha, or None.
pub fn betting_intervene_step(
    monitor: &Monitor,
    mat: &Mat,
    q: f64,
    alpha: f64,
    lam: f64,
) -> Option<usize> {
    let threshold = 1.0 / alpha;
    let mut w = 1.0;
    for (t, row) in mat.iter().enumerate() {
        let x = if monitor_fires_row(monitor, row) { 1.0 } else { 0.0 };
        w *= 1.0 + lam * (x - q) / (1.0 - q);
        if w >= threshold {
            return Some(t + 1);
        }
    }
    None
}

/// Betting power/FPR of a monitor on a set of (matrix, label) pairs,
/// with q calibrated on the successes of the same set.
pub fn monitor_betting_eval(
    monitor: &Monitor,
    mats: &[(Mat, bool)],
    alpha: f64,
    lam: f64,
) -> (f64, f64, f64) {
    let q = baseline_fire_rate(monitor, mats);
    let (mut fp, mut tp, mut n_pos, mut n_neg) = (0u64, 0u64, 0u64, 0u64);
    for (mat, label) in mats {
        let step = betting_intervene_step(monitor, mat, q, alpha, lam);
        if *label {
            n_neg += 1;
            fp += step.is_some() as u64;
        } else {
            n_pos += 1;
            tp += step.is_some() as u64;
        }
    }
    (
        if n_pos > 0 { tp as f64 / n_pos as f64 } else { 0.0 },
        if n_neg > 0 { fp as f64 / n_neg as f64 } else { 0.0 },
        q,
    )
}

/// SLS objective: betting power, penalised by betting FPR and rule size.
pub fn monitor_betting_score(
    monitor: &Monitor,
    mats: &[(Mat, bool)],
    alpha: f64,
    lam: f64,
) -> f64 {
    monitor_betting_score_w(monitor, mats, alpha, lam, 2.0, 0.005)
}

pub fn monitor_betting_score_w(
    monitor: &Monitor,
    mats: &[(Mat, bool)],
    alpha: f64,
    lam: f64,
    fpr_weight: f64,
    complexity: f64,
) -> f64 {
    let (power, fpr, _) = monitor_betting_eval(monitor, mats, alpha, lam);
    power - fpr_weight * fpr - complexity * monitor_complexity(monitor) as f64
}

// ---------------------------------------------------------------- evaluation
pub fn traj_tokens(traj: &Traj) -> i64 {
    traj.steps.iter().map(|s| s.tokens).sum()
}

/// Smallest step cap K whose training FPR (success runs longer than K)
/// is <= alpha; the most aggressive valid turn limit. None = never stop.
pub fn tune_turn_limit(train_mats: &[(Mat, bool)], alpha: f64) -> Option<usize> {
    let mut succ_lens: Vec<usize> = train_mats
        .iter()
        .filter(|(_, lab)| *lab)
        .map(|(m, _)| m.len())
        .collect();
    if succ_lens.is_empty() {
        return None;
    }
    succ_lens.sort_unstable();
    let n = succ_lens.len();
    for k in 5..=60usize {
        let fpr = succ_lens.iter().filter(|&&l| l > k).count() as f64 / n as f64;
        if fpr <= alpha {
            return Some(k);
        }
    }
    None
}

/// Best single-predicate immediate-stop rule on train, subject to
/// training FPR <= alpha. Candidates: err2, err_rate, repeat_tool_fail.
pub fn tune_error_filter(train_mats: &[(Mat, bool)], alpha: f64) -> Option<(usize, f64, f64)> {
    let mut best: Option<(usize, f64, f64)> = None;
    for idx in [1usize, 2, 6] {
        let (mut tp, mut fp, mut n_pos, mut n_neg) = (0u64, 0u64, 0u64, 0u64);
        for (mat, label) in train_mats {
            let fired = mat.iter().any(|row| row[idx]);
            if *label {
                n_neg += 1;
                fp += fired as u64;
            } else {
                n_pos += 1;
                tp += fired as u64;
            }
        }
        let tpr = if n_pos > 0 { tp as f64 / n_pos as f64 } else { 0.0 };
        let fpr = if n_neg > 0 { fp as f64 / n_neg as f64 } else { 0.0 };
        if fpr <= alpha && (best.is_none() || tpr > best.unwrap().1) {
            best = Some((idx, tpr, fpr));
        }
    }
    best
}

#[derive(Clone, Debug)]
pub struct Summary {
    pub fpr: f64,
    pub power: f64,
    pub token_savings: f64,
    pub precision_completed: f64,
    pub median_intervene_step_fail: Option<usize>,
    pub n_intervened_fail: u64,
    pub n_false_positive: u64,
}

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub alpha: f64,
    pub q: f64,
    pub lam: f64,
    pub n_success: usize,
    pub n_fail: usize,
    pub base_precision_no_monitor: f64,
    pub betting: Summary,
    pub immediate: Summary,
    pub turn_limit_tuned: Summary,
    pub k_star: Option<usize>,
    pub error_filter_tuned: Summary,
    pub error_predicate: Option<&'static str>,
    pub error_train_tpr: Option<f64>,
    pub error_train_fpr: Option<f64>,
}

/// Held-out evaluation of the betting monitor. Baselines are TUNED ON
/// TRAIN to the same validity budget (FPR <= alpha).
pub fn evaluate(
    monitor: &Monitor,
    q: f64,
    alpha: f64,
    train_data: &[Traj],
    test_data: &[Traj],
    lam: f64,
) -> EvalResult {
    let train_mats: Vec<(Mat, bool)> = train_data
        .iter()
        .map(|t| (prefix_matrix(t), t.label))
        .collect();
    let test_mats: Vec<(Mat, bool)> = test_data
        .iter()
        .map(|t| (prefix_matrix(t), t.label))
        .collect();
    let k_star = tune_turn_limit(&train_mats, alpha);
    let err_star = tune_error_filter(&train_mats, alpha);
    let n_succ = test_data.iter().filter(|t| t.label).count();
    let n_fail = test_data.len() - n_succ;
    let total_tokens: i64 = test_data.iter().map(traj_tokens).sum();

    let summarise = |intervene_fn: &dyn Fn(&Mat) -> Option<usize>| -> Summary {
        let (mut fp, mut tp) = (0u64, 0u64);
        let mut spent: i64 = 0;
        let (mut completed_ok, mut completed) = (0u64, 0u64);
        let mut intervene_steps_fail: Vec<usize> = Vec::new();
        for (traj, (mat, label)) in test_data.iter().zip(test_mats.iter()) {
            match intervene_fn(mat) {
                None => {
                    spent += traj_tokens(traj);
                    completed += 1;
                    completed_ok += *label as u64;
                }
                Some(step) => {
                    spent += traj.steps[..step].iter().map(|s| s.tokens).sum::<i64>();
                    if *label {
                        fp += 1;
                    } else {
                        tp += 1;
                        intervene_steps_fail.push(step);
                    }
                }
            }
        }
        intervene_steps_fail.sort_unstable();
        Summary {
            fpr: if n_succ > 0 { fp as f64 / n_succ as f64 } else { 0.0 },
            power: if n_fail > 0 { tp as f64 / n_fail as f64 } else { 0.0 },
            token_savings: if total_tokens > 0 {
                1.0 - spent as f64 / total_tokens as f64
            } else {
                0.0
            },
            precision_completed: if completed > 0 {
                completed_ok as f64 / completed as f64
            } else {
                0.0
            },
            median_intervene_step_fail: if intervene_steps_fail.is_empty() {
                None
            } else {
                Some(intervene_steps_fail[intervene_steps_fail.len() / 2])
            },
            n_intervened_fail: tp,
            n_false_positive: fp,
        }
    };

    let betting = summarise(&|mat| betting_intervene_step(monitor, mat, q, alpha, lam));
    let immediate = summarise(&|mat| monitor_fires_traj(monitor, mat));
    let turn_limit_tuned = match k_star {
        None => summarise(&|_| None),
        Some(k) => summarise(&|mat| if mat.len() > k { Some(k) } else { None }),
    };
    let (error_filter_tuned, error_predicate, error_train_tpr, error_train_fpr) =
        match err_star {
            None => (summarise(&|_| None), None, None, None),
            Some((idx, tpr, fpr)) => (
                summarise(&|mat| {
                    mat.iter()
                        .position(|row| row[idx])
                        .map(|t| t + 1)
                }),
                Some(PREDICATES[idx].0),
                Some(tpr),
                Some(fpr),
            ),
        };
    EvalResult {
        alpha,
        q,
        lam,
        n_success: n_succ,
        n_fail,
        base_precision_no_monitor: if !test_data.is_empty() {
            n_succ as f64 / test_data.len() as f64
        } else {
            0.0
        },
        betting,
        immediate,
        turn_limit_tuned,
        k_star,
        error_filter_tuned,
        error_predicate,
        error_train_tpr,
        error_train_fpr,
    }
}

pub fn monitor_to_text(monitor: &Monitor) -> String {
    let parts: Vec<String> = monitor
        .iter()
        .map(|conj| {
            let mut names: Vec<&str> = conj.iter().map(|&i| PREDICATES[i].0).collect();
            names.sort_unstable(); // Python: sorted() by predicate NAME
            format!("({})", names.join(" AND "))
        })
        .collect();
    parts.join(" OR ")
}

/// Monitor as nested predicate-name lists, conjunctions in order,
/// literals sorted by predicate index (Python `monitor_raw`).
pub fn monitor_raw(monitor: &Monitor) -> Vec<Vec<&'static str>> {
    monitor
        .iter()
        .map(|conj| conj.iter().map(|&i| PREDICATES[i].0).collect())
        .collect()
}

// ---------------------------------------------------------------- experiment
#[derive(Clone, Debug)]
pub struct NegativeControl {
    pub monitor: String,
    pub train_tpr_minus_fpr: f64,
    pub train_tpr: f64,
    pub train_fpr: f64,
}

#[derive(Clone, Debug)]
pub struct ExperimentResult {
    pub eval: EvalResult,
    pub monitor: String,
    pub monitor_raw: Vec<Vec<&'static str>>,
    pub train_tpr: f64,
    pub train_fpr: f64,
    pub train_score: f64,
    pub train_betting_power: f64,
    pub train_betting_fpr: f64,
    pub negative_control: NegativeControl,
    pub seed: u64,
}

/// Full LeBet experiment: generate, learn, calibrate, evaluate.
/// Mirrors `lebet.run_experiment` (seed 20261116, train/test 200+200,
/// alpha 0.15, restarts 6, iters 200, bet_lam 0.75).
///
/// Note: Python's `run_experiment` takes an unused `lam=2.0` argument
/// (never read inside the function); it is dropped here rather than
/// carried as dead API.
pub fn run_experiment(
    seed: u64,
    n_train_succ: usize,
    n_train_fail: usize,
    n_test_succ: usize,
    n_test_fail: usize,
    alpha: f64,
    restarts: usize,
    iters: usize,
    bet_lam: f64,
) -> ExperimentResult {
    let mut rng = Rng::new(seed);
    let train = gen_dataset(&mut rng, n_train_succ, n_train_fail);
    let test = gen_dataset(&mut rng, n_test_succ, n_test_fail);
    let train_mats: Vec<(Mat, bool)> = train
        .iter()
        .map(|t| (prefix_matrix(t), t.label))
        .collect();
    let (monitor, train_score) =
        learn_monitor(&mut rng, &train_mats, restarts, iters, alpha, bet_lam);
    let (train_tpr, train_fpr) = monitor_tpr_fpr(&monitor, &train_mats);
    let (train_power, train_bet_fpr, _) =
        monitor_betting_eval(&monitor, &train_mats, alpha, bet_lam);
    let q = baseline_fire_rate(&monitor, &train_mats);
    let eval = evaluate(&monitor, q, alpha, &train, &test, bet_lam);
    // negative control: same learner on label-permuted training data
    let mut rng2 = Rng::new(seed.wrapping_add(999));
    let mut labels: Vec<bool> = train_mats.iter().map(|(_, lab)| *lab).collect();
    for i in (1..labels.len()).rev() {
        let j = rng2.randint((i + 1) as u64) as usize;
        labels.swap(i, j);
    }
    let shuffled: Vec<(Mat, bool)> = train_mats
        .iter()
        .zip(labels.iter())
        .map(|((m, _), lab)| (m.clone(), *lab))
        .collect();
    let (monitor_nc, _) = learn_monitor(&mut rng2, &shuffled, 3, 100, alpha, bet_lam);
    let (nc_tpr, nc_fpr) = monitor_tpr_fpr(&monitor_nc, &shuffled);
    ExperimentResult {
        monitor: monitor_to_text(&monitor),
        monitor_raw: monitor_raw(&monitor),
        train_tpr,
        train_fpr,
        train_score,
        train_betting_power: train_power,
        train_betting_fpr: train_bet_fpr,
        negative_control: NegativeControl {
            monitor: monitor_to_text(&monitor_nc),
            train_tpr_minus_fpr: nc_tpr - nc_fpr,
            train_tpr: nc_tpr,
            train_fpr: nc_fpr,
        },
        eval,
        seed,
    }
}

/// Defaults matching `lebet.run_experiment()` with no arguments.
pub fn run_experiment_default() -> ExperimentResult {
    run_experiment(20261116, 200, 200, 200, 200, 0.15, 6, 200, 0.75)
}

// ---------------------------------------------------------------- tests
#[cfg(test)]
mod tests {
    use super::*;

    fn step(action: u64, ok: bool, state: u64, obs: u64, tokens: i64) -> Step {
        Step {
            action,
            ok,
            state,
            obs,
            tokens,
        }
    }

    #[test]
    fn rng_stream_first_values() {
        // Values from the Python reference (tests/parity_gen.py `rng`,
        // seed 20261116). If these drift, the whole parity chain breaks.
        let mut rng = Rng::new(20261116);
        assert_eq!(rng.next_u64(), 16677456083611059126);
        assert_eq!(rng.next_u64(), 17228767783036868879);
        assert_eq!(rng.next_u64(), 13718819615347423340);
        let mut rng = Rng::new(20261116);
        assert_eq!(rng.randint(6), 0);
        assert_eq!(rng.randint(1 << 30), 161175823);
        let mut rng = Rng::new(0);
        assert_eq!(rng.next_u64(), 16294208416658607535);
        assert_eq!(rng.next_u64(), 7960286522194355700);
    }

    #[test]
    fn rng_uniform_bounds_and_gauss_finite() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let u = rng.uniform();
            assert!(u >= 0.0 && u < 1.0);
            let g = rng.gauss(0.0, 1.0);
            assert!(g.is_finite());
        }
    }

    #[test]
    fn predicates_hand_cases() {
        // repeat3: three identical trailing actions
        let p = vec![step(2, true, 1, 1, 300), step(2, true, 2, 2, 300), step(2, true, 3, 3, 300)];
        assert!(p_repeat3(&p));
        assert!(!p_repeat3(&p[..2]));
        // err2: two trailing errors
        let p = vec![step(0, true, 1, 1, 300), step(1, false, 2, 2, 300), step(1, false, 3, 3, 300)];
        assert!(p_err2(&p));
        assert!(!p_err2(&p[..2]));
        // err_rate: 2/5 errors in last 10 -> 0.4 > 0.3
        let p: Vec<Step> = (0..5)
            .map(|i| step(i, i >= 3, i, i, 300))
            .collect();
        assert!(p_err_rate(&p));
        // stalled: same state 3 in a row
        let p = vec![step(0, true, 7, 1, 300), step(1, true, 7, 2, 300), step(2, true, 7, 3, 300)];
        assert!(p_stalled(&p));
        assert!(!p_stalled(&p[..2]));
        // loop: one state seen 3 times anywhere
        let p = vec![
            step(0, true, 5, 1, 300),
            step(1, true, 6, 2, 300),
            step(2, true, 5, 3, 300),
            step(3, true, 5, 4, 300),
        ];
        assert!(p_loop(&p));
        assert!(!p_loop(&p[..3]));
        // long
        let p: Vec<Step> = (0..21).map(|i| step(i % 6, true, i, i, 300)).collect();
        assert!(p_long(&p));
        assert!(!p_long(&p[..20]));
        // repeat_tool_fail: last step failed, same tool failed >= 2 times
        let p = vec![
            step(3, false, 1, 1, 300),
            step(1, true, 2, 2, 300),
            step(3, false, 3, 3, 300),
        ];
        assert!(p_repeat_tool_fail(&p));
        assert!(!p_repeat_tool_fail(&p[..2]));
        // no_new_obs
        let p = vec![step(0, true, 1, 9, 300), step(1, true, 2, 9, 300)];
        assert!(p_no_new_obs(&p));
        // tok_high: 21 * 300 = 6300 > 6000
        let p: Vec<Step> = (0..21).map(|i| step(i % 6, true, i, i, 300)).collect();
        assert!(p_tok_high(&p));
        assert!(!p_tok_high(&p[..20]));
        // same_action_5: 5 of last 8 with the last action (action 4)
        let p = vec![
            step(4, true, 1, 1, 300),
            step(0, true, 2, 2, 300),
            step(4, true, 3, 3, 300),
            step(1, true, 4, 4, 300),
            step(4, true, 5, 5, 300),
            step(2, true, 6, 6, 300),
            step(4, true, 7, 7, 300),
            step(4, true, 8, 8, 300),
        ];
        assert!(p_same_action_5(&p));
        assert!(!p_same_action_5(&p[..4]));
    }

    #[test]
    fn monitor_firing_and_text() {
        let mon: Monitor = vec![
            BTreeSet::from([4, 8]), // loop AND tok_high
            BTreeSet::from([2]),    // err_rate
        ];
        assert_eq!(monitor_to_text(&mon), "(loop AND tok_high) OR (err_rate)");
        assert_eq!(
            monitor_raw(&mon),
            vec![vec!["loop", "tok_high"], vec!["err_rate"]]
        );
        let mut row = [false; 10];
        assert!(!monitor_fires_row(&mon, &row));
        row[4] = true;
        assert!(!monitor_fires_row(&mon, &row));
        row[8] = true;
        assert!(monitor_fires_row(&mon, &row));
        row[8] = false;
        row[2] = true;
        assert!(monitor_fires_row(&mon, &row));
        // empty conjunction never fires
        let empty: Monitor = vec![BTreeSet::new()];
        assert!(!monitor_fires_row(&empty, &[true; 10]));
        // first firing step is 1-based
        let mat = vec![[false; 10], [false, false, true, false, false, false, false, false, false, false]];
        assert_eq!(monitor_fires_traj(&mon, &mat), Some(2));
        assert_eq!(monitor_fires_traj(&mon, &vec![[false; 10]]), None);
    }

    #[test]
    fn betting_wealth_series_hand_case() {
        // q = 0.1, lam = 0.5: fire multiplies by 1 + 0.5*(1-0.1)/0.9 = 1.5,
        // no-fire multiplies by 1 + 0.5*(0-0.1)/0.9 = 0.9444...
        let mon: Monitor = vec![BTreeSet::from([0])];
        let mat = vec![
            [true, false, false, false, false, false, false, false, false, false],
            [false; 10],
            [true, false, false, false, false, false, false, false, false, false],
        ];
        let series = betting_wealth_series(&mon, &mat, 0.1, 0.5);
        assert_eq!(series.len(), 3);
        assert!((series[0] - 1.5).abs() < 1e-12);
        assert!((series[1] - 1.5 * (1.0 - 0.5 * 0.1 / 0.9)).abs() < 1e-12);
        assert!((series[2] - series[1] * 1.5).abs() < 1e-12);
        // intervene at 1/alpha = 1/0.15 = 6.67: never here
        assert_eq!(betting_intervene_step(&mon, &mat, 0.1, 0.15, 0.5), None);
        // with a tiny alpha the threshold is low: fires at step 1 already
        // crosses 1/0.9 = 1.11? 1.5 >= 1.11 yes
        assert_eq!(betting_intervene_step(&mon, &mat, 0.1, 0.9, 0.5), Some(1));
    }

    #[test]
    fn tuning_functions_hand_cases() {
        // turn limit: success lengths [10, 20, 30]; alpha 0.15:
        // k=25 -> FPR 1/3 > 0.15; k=30 -> FPR 0 -> K*=30
        let mats: Vec<(Mat, bool)> = vec![
            (vec![[false; 10]; 10], true),
            (vec![[false; 10]; 20], true),
            (vec![[false; 10]; 30], true),
            (vec![[false; 10]; 40], false),
        ];
        assert_eq!(tune_turn_limit(&mats, 0.15), Some(30));
        assert_eq!(tune_turn_limit(&mats, 0.0), Some(30));
        assert_eq!(tune_turn_limit(&[], 0.15), None);
        // error filter: err2 fires on the fail traj only -> picked
        let mut fail_mat = vec![[false; 10]; 3];
        fail_mat[1][1] = true;
        fail_mat[2][1] = true;
        let mats: Vec<(Mat, bool)> = vec![
            (vec![[false; 10]; 5], true),
            (vec![[false; 10]; 5], true),
            (fail_mat, false),
        ];
        let best = tune_error_filter(&mats, 0.15).unwrap();
        assert_eq!(best.0, 1); // err2
        assert!((best.1 - 1.0).abs() < 1e-12);
        assert!((best.2 - 0.0).abs() < 1e-12);
    }

    #[test]
    fn mutate_keeps_monitor_valid() {
        let mut rng = Rng::new(7);
        let start: Monitor = vec![BTreeSet::from([0, 2]), BTreeSet::from([5])];
        let mut cur = start;
        for _ in 0..5000 {
            cur = mutate(&mut rng, &cur, 3, 3);
            assert!(!cur.is_empty() && cur.len() <= 3);
            for c in &cur {
                assert!(!c.is_empty() && c.len() <= 3);
                assert!(c.iter().all(|&i| i < N_PRED));
            }
        }
    }

    #[test]
    fn gen_dataset_label_balance_and_shuffle() {
        let mut rng = Rng::new(123);
        let data = gen_dataset(&mut rng, 10, 10);
        assert_eq!(data.len(), 20);
        assert_eq!(data.iter().filter(|t| t.label).count(), 10);
        // determinism: same seed -> identical trajectories
        let mut rng2 = Rng::new(123);
        let data2 = gen_dataset(&mut rng2, 10, 10);
        for (a, b) in data.iter().zip(data2.iter()) {
            assert_eq!(a.label, b.label);
            assert_eq!(a.steps, b.steps);
        }
    }
}
