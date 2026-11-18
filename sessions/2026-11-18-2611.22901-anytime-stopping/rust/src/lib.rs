//! Anytime evaluation of when coding agents should stop (arXiv:2611.22901).
//!
//! Rust port of `arxiv_lab.agent.anytime_stopping` (Python, stdlib only).
//! Std only, no external crates, no network, no model calls.
//!
//! The Python module stays the primary verdict carrier; this port
//! cross-checks it. A failed Rust leg never overturns the Python verdict.
//!
//! Parity note: the PRNG is SplitMix64 with the exact same arithmetic
//! as Python (`(next_u64() >> 11) * 2**-53` for `uniform()`), and the
//! trajectory generator draws in the exact same order and branches, so
//! a trajectory generated from the same seed is bit-identical in both
//! languages. The parity sweep (`tests/parity_sweep.py`) checks this
//! on random cases with 0 mismatches allowed.

/// SplitMix64 PRNG (Stafford). Identical arithmetic to the Python port.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Float in [0, 1) with 53 bits of randomness.
    /// Same formula as Python: `(next_u64() >> 11) * 2**-53`.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }

    /// The 53-bit mantissa integer behind `uniform()`, for parity checks.
    pub fn mantissa(&mut self) -> u64 {
        self.next_u64() >> 11
    }

    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }
}

/// Parameters of the calibrated synthetic trajectory model.
/// Field meanings and calibration sources: see the Python module
/// (`arxiv_lab.agent.anytime_stopping.TrajectoryParams`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryParams {
    pub budget: usize,
    pub p_solvable: f64,
    pub p_fix: f64,
    pub p_change_after_self_pass: f64,
    pub p_change_after_self_fail: f64,
    pub p_break: f64,
    pub p_self_false_pass: f64,
    pub p_self_false_fail: f64,
}

impl Default for TrajectoryParams {
    /// The calibrated regime (same numbers as Python `CALIBRATED`).
    fn default() -> Self {
        TrajectoryParams {
            budget: 30,
            p_solvable: 0.65,
            p_fix: 0.03,
            p_change_after_self_pass: 0.15,
            p_change_after_self_fail: 0.90,
            p_break: 0.08,
            p_self_false_pass: 0.01,
            p_self_false_fail: 0.15,
        }
    }
}

impl TrajectoryParams {
    pub fn with_overrides(
        &self,
        p_break: Option<f64>,
        p_change_after_self_pass: Option<f64>,
        p_self_false_pass: Option<f64>,
        p_fix: Option<f64>,
    ) -> Self {
        let mut p = *self;
        if let Some(v) = p_break {
            p.p_break = v;
        }
        if let Some(v) = p_change_after_self_pass {
            p.p_change_after_self_pass = v;
        }
        if let Some(v) = p_self_false_pass {
            p.p_self_false_pass = v;
        }
        if let Some(v) = p_fix {
            p.p_fix = v;
        }
        p
    }
}

/// One step of a recorded run: the patch emitted and its test states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub step: usize,
    pub patch_id: usize,
    pub changed: bool,
    pub hidden_pass: bool,
    pub self_pass: bool,
}

/// Generate one run (list of Step) under `params`.
///
/// Draw order per step (fixed -- mirrors Python exactly): solvable
/// flag once, then per step: change?, hidden update (only on a change,
/// and only the draw the branch needs), self-test outcome (only on a
/// change; unchanged patches keep their sticky result).
pub fn generate_trajectory(rng: &mut SplitMix64, params: &TrajectoryParams) -> Vec<Step> {
    let solvable = rng.bernoulli(params.p_solvable);
    let mut steps = Vec::with_capacity(params.budget);
    let mut hidden = false;
    let mut patch_id: usize = 0;
    let mut prev_self = false;
    for t in 1..=params.budget {
        let changed = if t == 1 {
            true
        } else {
            let p_change = if prev_self {
                params.p_change_after_self_pass
            } else {
                params.p_change_after_self_fail
            };
            rng.bernoulli(p_change)
        };
        let self_pass: bool;
        if changed {
            patch_id += 1;
            if hidden {
                if rng.bernoulli(params.p_break) {
                    hidden = false;
                }
            } else if solvable {
                if rng.bernoulli(params.p_fix) {
                    hidden = true;
                }
            }
            self_pass = if hidden {
                !rng.bernoulli(params.p_self_false_fail)
            } else {
                rng.bernoulli(params.p_self_false_pass)
            };
        } else {
            // Same patch, same tests: the sticky self-test result.
            self_pass = prev_self;
        }
        steps.push(Step {
            step: t,
            patch_id,
            changed,
            hidden_pass: hidden,
            self_pass,
        });
        prev_self = self_pass;
    }
    steps
}

/// (a) Baseline: run to the step budget.
pub fn stop_run_to_budget(traj: &[Step]) -> usize {
    traj.len()
}

/// (b) Stop at the first self-test pass.
pub fn stop_first_self_pass(traj: &[Step]) -> usize {
    for s in traj {
        if s.self_pass {
            return s.step;
        }
    }
    traj.len()
}

/// (c) The paper's rule: stop after the SAME patch passes self-tests
/// twice in a row. The counter resets on patch change or self-fail.
pub fn stop_double_pass(traj: &[Step]) -> usize {
    let mut count = 0usize;
    let mut current_patch: Option<usize> = None;
    for s in traj {
        if Some(s.patch_id) != current_patch {
            current_patch = Some(s.patch_id);
            count = 0;
        }
        if s.self_pass {
            count += 1;
            if count >= 2 {
                return s.step;
            }
        } else {
            count = 0;
        }
    }
    traj.len()
}

/// Sensitivity variant of (c): the same patch passes twice
/// cumulatively (only a patch change resets the count). Under sticky
/// self-tests this coincides with the consecutive rule exactly.
pub fn stop_double_pass_cumulative(traj: &[Step]) -> usize {
    let mut count = 0usize;
    let mut current_patch: Option<usize> = None;
    for s in traj {
        if Some(s.patch_id) != current_patch {
            current_patch = Some(s.patch_id);
            count = 0;
        }
        if s.self_pass {
            count += 1;
            if count >= 2 {
                return s.step;
            }
        }
    }
    traj.len()
}

/// (d) Hindsight oracle: stop at the first step where hidden tests
/// pass; if the run never passes, run to budget.
pub fn stop_oracle(traj: &[Step]) -> usize {
    for s in traj {
        if s.hidden_pass {
            return s.step;
        }
    }
    traj.len()
}

/// Integer summary of a trajectory for parity checks: (steps,
/// hidden-pass count, self-pass count, final patch id, change count,
/// first hidden-pass step or 0).
pub fn trajectory_digest(traj: &[Step]) -> [usize; 6] {
    let hidden_passes = traj.iter().filter(|s| s.hidden_pass).count();
    let self_passes = traj.iter().filter(|s| s.self_pass).count();
    let changes = traj.iter().filter(|s| s.changed).count();
    let first_hidden = traj
        .iter()
        .find(|s| s.hidden_pass)
        .map(|s| s.step)
        .unwrap_or(0);
    [
        traj.len(),
        hidden_passes,
        self_passes,
        traj.last().map(|s| s.patch_id).unwrap_or(0),
        changes,
        first_hidden,
    ]
}

/// Per-policy aggregate over one set of trajectories (paired).
#[derive(Clone, Debug)]
pub struct PolicyAggregate {
    pub mean_steps: f64,
    pub saving_pct: f64,
    pub final_hidden_pass_rate: f64,
    pub fire_rate: f64,
    /// Integer sums, for exact parity checks.
    pub sum_stop_steps: u64,
    pub n_final_pass: u64,
    pub n_fired: u64,
}

/// Post-fire analysis for the double-pass rule.
#[derive(Clone, Debug, Default)]
pub struct PostFire {
    pub n_fired: u64,
    pub changed_after_fire: u64,
    pub repairs_after_fire: u64,
    pub breaks_after_fire: u64,
}

/// Paired quality delta of policy A vs policy B (delta, wins, losses,
/// ties, 95% CI on the paired mean difference).
pub fn paired_quality_delta(
    final_a: &[bool],
    final_b: &[bool],
) -> (f64, u64, u64, u64, (f64, f64)) {
    let n = final_a.len() as f64;
    let mut wins = 0u64;
    let mut losses = 0u64;
    let mut sum = 0i64;
    let mut sum_sq = 0i64;
    for (a, b) in final_a.iter().zip(final_b.iter()) {
        let d: i64 = (*a as i64) - (*b as i64);
        if d > 0 {
            wins += 1;
        } else if d < 0 {
            losses += 1;
        }
        sum += d;
        sum_sq += d * d;
    }
    let ties = final_a.len() as u64 - wins - losses;
    let delta = sum as f64 / n;
    let mean_sq = sum_sq as f64 / n;
    let var = (mean_sq - delta * delta).max(0.0);
    let se = (var / n).sqrt();
    (delta, wins, losses, ties, (delta - 1.96 * se, delta + 1.96 * se))
}

/// Full paired evaluation of every policy on the same trajectories.
#[derive(Clone, Debug)]
pub struct Evaluation {
    pub n_runs: usize,
    pub budget: usize,
    pub baseline: PolicyAggregate,
    pub first_self_pass: PolicyAggregate,
    pub double_pass: PolicyAggregate,
    pub double_pass_cumulative: PolicyAggregate,
    pub oracle: PolicyAggregate,
    pub post_fire: PostFire,
    pub hindsight_conditional_pct: f64,
    pub hindsight_all_runs_pct: f64,
    pub n_true_first_pass: u64,
    pub quality_delta_pp: f64,
    pub quality_wins: u64,
    pub quality_losses: u64,
    pub quality_ties: u64,
    pub quality_ci_95_pp: (f64, f64),
}

fn aggregate_policy(
    trajs: &[Vec<Step>],
    stop_fn: fn(&[Step]) -> usize,
    budget: usize,
) -> (PolicyAggregate, Vec<usize>, Vec<bool>) {
    let n = trajs.len();
    let mut sum_steps = 0u64;
    let mut n_pass = 0u64;
    let mut n_fired = 0u64;
    let mut stops = Vec::with_capacity(n);
    let mut finals = Vec::with_capacity(n);
    for t in trajs {
        let stop = stop_fn(t);
        let final_pass = t[stop - 1].hidden_pass;
        stops.push(stop);
        finals.push(final_pass);
        sum_steps += stop as u64;
        if final_pass {
            n_pass += 1;
        }
        if stop < budget {
            n_fired += 1;
        }
    }
    let mean = sum_steps as f64 / n as f64;
    (
        PolicyAggregate {
            mean_steps: mean,
            saving_pct: 100.0 * (budget as f64 - mean) / budget as f64,
            final_hidden_pass_rate: n_pass as f64 / n as f64,
            fire_rate: n_fired as f64 / n as f64,
            sum_stop_steps: sum_steps,
            n_final_pass: n_pass,
            n_fired,
        },
        stops,
        finals,
    )
}

pub fn evaluate_all(trajs: &[Vec<Step>], params: &TrajectoryParams) -> Evaluation {
    let budget = params.budget;
    let (baseline, _, base_finals) = aggregate_policy(trajs, stop_run_to_budget, budget);
    let (first_self_pass, _, _) =
        aggregate_policy(trajs, stop_first_self_pass, budget);
    let (double_pass, dp_stops, dp_finals) =
        aggregate_policy(trajs, stop_double_pass, budget);
    let (double_pass_cumulative, _, _) =
        aggregate_policy(trajs, stop_double_pass_cumulative, budget);
    let (oracle, _, _) = aggregate_policy(trajs, stop_oracle, budget);

    let mut post_fire = PostFire::default();
    for (traj, stop) in trajs.iter().zip(dp_stops.iter()) {
        if *stop < budget {
            post_fire.n_fired += 1;
            let fire_hidden = traj[stop - 1].hidden_pass;
            let final_hidden = traj[traj.len() - 1].hidden_pass;
            if traj[*stop..].iter().any(|s| s.changed) {
                post_fire.changed_after_fire += 1;
            }
            if !fire_hidden && final_hidden {
                post_fire.repairs_after_fire += 1;
            }
            if fire_hidden && !final_hidden {
                post_fire.breaks_after_fire += 1;
            }
        }
    }

    let mut hindsight = Vec::new();
    let mut hindsight_all = Vec::new();
    for traj in trajs {
        let mut found = false;
        for s in traj {
            if s.self_pass {
                if s.hidden_pass {
                    hindsight.push(100.0 * (budget - s.step) as f64 / budget as f64);
                    hindsight_all.push(100.0 * (budget - s.step) as f64 / budget as f64);
                } else {
                    hindsight_all.push(0.0);
                }
                found = true;
                break;
            }
        }
        if !found {
            hindsight_all.push(0.0);
        }
    }
    let hindsight_conditional_pct = if hindsight.is_empty() {
        0.0
    } else {
        hindsight.iter().sum::<f64>() / hindsight.len() as f64
    };
    let hindsight_all_runs_pct =
        hindsight_all.iter().sum::<f64>() / hindsight_all.len() as f64;

    let (delta, wins, losses, ties, ci) =
        paired_quality_delta(&dp_finals, &base_finals);
    Evaluation {
        n_runs: trajs.len(),
        budget,
        baseline,
        first_self_pass,
        double_pass,
        double_pass_cumulative,
        oracle,
        post_fire,
        hindsight_conditional_pct,
        hindsight_all_runs_pct,
        n_true_first_pass: hindsight.len() as u64,
        quality_delta_pp: 100.0 * delta,
        quality_wins: wins,
        quality_losses: losses,
        quality_ties: ties,
        quality_ci_95_pp: (100.0 * ci.0, 100.0 * ci.1),
    }
}

/// One sensitivity-sweep cell.
#[derive(Clone, Debug)]
pub struct SweepCell {
    pub p_break: f64,
    pub p_change_after_self_pass: f64,
    pub p_self_false_pass: f64,
    pub p_fix: f64,
    pub double_pass_saving_pct: f64,
    pub quality_delta_pp: f64,
    pub baseline_pass_rate: f64,
    pub double_pass_pass_rate: f64,
}

/// Sensitivity sweep: 3x3 grid over (p_break, p_change_after_self_pass)
/// at the calibrated params, plus the self-test fidelity axis
/// (p_self_false_pass 0.03 / 0.05) at the calibrated break/edit rates.
/// Mirrors Python `sensitivity_sweep` exactly (same seeds, same order).
pub fn sensitivity_sweep(
    params: &TrajectoryParams,
    n_runs: usize,
    seeds: &[u64],
) -> Vec<SweepCell> {
    let mut cells = Vec::new();
    let run_cell = |p: TrajectoryParams, cells: &mut Vec<SweepCell>| {
        let mut trajs = Vec::new();
        for seed in seeds {
            let mut rng = SplitMix64::new(*seed);
            for _ in 0..n_runs {
                trajs.push(generate_trajectory(&mut rng, &p));
            }
        }
        let agg = evaluate_all(&trajs, &p);
        cells.push(SweepCell {
            p_break: p.p_break,
            p_change_after_self_pass: p.p_change_after_self_pass,
            p_self_false_pass: p.p_self_false_pass,
            p_fix: p.p_fix,
            double_pass_saving_pct: agg.double_pass.saving_pct,
            quality_delta_pp: agg.quality_delta_pp,
            baseline_pass_rate: agg.baseline.final_hidden_pass_rate,
            double_pass_pass_rate: agg.double_pass.final_hidden_pass_rate,
        });
    };
    for p_break in [0.05, 0.12, 0.30] {
        for p_change in [0.10, 0.25, 0.45] {
            run_cell(
                params.with_overrides(Some(p_break), Some(p_change), None, None),
                &mut cells,
            );
        }
    }
    for fpp in [0.03, 0.05] {
        run_cell(
            params.with_overrides(None, None, Some(fpp), None),
            &mut cells,
        );
    }
    cells
}

/// Main experiment: `seeds.len()` seeds x `n_runs` runs, pooled.
/// Mirrors Python `run_experiment` exactly.
pub fn run_experiment(
    params: &TrajectoryParams,
    n_runs: usize,
    seeds: &[u64],
) -> (Evaluation, Vec<(u64, f64, f64, f64, f64)>) {
    let mut trajs = Vec::new();
    let mut per_seed = Vec::new();
    for seed in seeds {
        let mut rng = SplitMix64::new(*seed);
        let mut seed_trajs = Vec::new();
        for _ in 0..n_runs {
            seed_trajs.push(generate_trajectory(&mut rng, params));
        }
        let agg = evaluate_all(&seed_trajs, params);
        per_seed.push((
            *seed,
            agg.double_pass.saving_pct,
            agg.oracle.saving_pct,
            agg.baseline.final_hidden_pass_rate,
            agg.double_pass.final_hidden_pass_rate,
        ));
        trajs.extend(seed_trajs);
    }
    (evaluate_all(&trajs, params), per_seed)
}
