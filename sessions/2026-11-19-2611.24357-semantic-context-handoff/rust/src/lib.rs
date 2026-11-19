//! Semantic Context Handoff (arXiv:2611.24357, "Semantic Context Handoff
//! with On-Demand Evidence for Long-Horizon Agents").
//!
//! Rust port of `arxiv_lab.agent.semantic_context_handoff` (Python,
//! stdlib only). Std only, no external crates, no network, no model calls.
//!
//! The Python module stays the primary verdict carrier; this port
//! cross-checks it. A failed Rust leg never overturns the Python verdict.
//!
//! Parity note: the PRNG is SplitMix64 with the exact same arithmetic
//! as Python (`(next_u64() >> 11) * 2**-53` for `uniform()`), the task
//! generator draws in the exact same order and branches, the recovery
//! stream is seeded identically (`seed * 1_000_003 + task_index * 131 + 7`,
//! wrapping 64-bit), compressed tokens truncate identically
//! (`int(tokens * compression)`), and the task-pass rule is identical
//! (pass iff steps_ok >= ceil(0.8 * steps)). A task generated from the
//! same seed is therefore bit-identical in both languages, and every
//! policy aggregate is an exact integer sum in both. The parity sweep
//! (`tests/parity_sweep.py`) checks this on random cases with 0
//! mismatches allowed.

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

    /// Uniform integer in [lo, hi] inclusive. Mirrors Python
    /// `lo + int(uniform() * (hi - lo + 1))`, including the degenerate
    /// hi < lo case (range 0, returns lo): the range is computed in i64
    /// so it cannot underflow.
    pub fn randint(&mut self, lo: u64, hi: u64) -> u64 {
        let span = hi as i64 - lo as i64 + 1;
        lo + (self.uniform() * span as f64) as u64
    }
}

/// Parameters of the calibrated synthetic long-horizon task model.
/// Field meanings and calibration sources: see the Python module
/// (`arxiv_lab.agent.semantic_context_handoff.TaskParams`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaskParams {
    pub steps: usize,
    pub facts_per_step: usize,
    pub delayed_frac: f64,
    pub max_delay: usize,
    pub min_fact_tokens: u64,
    pub max_fact_tokens: u64,
    pub state_retention: f64,
    pub compression: f64,
    pub state_overhead: u64,
    pub recall: f64,
    pub lookup_overhead: u64,
    pub fp_per_lookup: f64,
    pub pass_step_fraction: f64,
}

impl Default for TaskParams {
    /// The calibrated regime (same numbers as Python `CALIBRATED`).
    fn default() -> Self {
        TaskParams {
            steps: 12,
            facts_per_step: 3,
            delayed_frac: 0.25,
            max_delay: 4,
            min_fact_tokens: 10,
            max_fact_tokens: 30,
            state_retention: 0.85,
            compression: 0.3,
            state_overhead: 40,
            recall: 0.9,
            lookup_overhead: 5,
            fp_per_lookup: 0.2,
            pass_step_fraction: 0.8,
        }
    }
}

/// One evidence item produced at a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fact {
    pub key: usize,
    pub produced_step: usize,
    pub needed_step: usize,
    pub tokens: u64,
    pub retained: bool,
}

/// A generated long-horizon task: facts in production order + the list
/// of fact keys needed at each step. `key == index into facts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub facts: Vec<Fact>,
    pub needs: Vec<Vec<usize>>,
}

impl Task {
    pub fn steps(&self) -> usize {
        self.needs.len()
    }
}

/// Generate one task. Draw order is fixed -- see the Python module
/// docstring: per step, per fact in order: size draw, delay decision,
/// delay draw iff delayed; then retention draws in production order.
pub fn generate_task(rng: &mut SplitMix64, params: &TaskParams) -> Task {
    let mut facts = Vec::with_capacity(params.steps * params.facts_per_step);
    let mut needs: Vec<Vec<usize>> = vec![Vec::new(); params.steps];
    let mut key = 0usize;
    for t in 0..params.steps {
        for _ in 0..params.facts_per_step {
            let tokens = rng.randint(params.min_fact_tokens, params.max_fact_tokens);
            let delay = if rng.bernoulli(params.delayed_frac) {
                rng.randint(1, params.max_delay as u64) as usize
            } else {
                0
            };
            let needed_step = t + delay;
            facts.push(Fact {
                key,
                produced_step: t,
                needed_step,
                tokens,
                retained: false,
            });
            if needed_step < params.steps {
                needs[needed_step].push(key);
            }
            key += 1;
        }
    }
    // Retention draws: production order, after all facts are generated.
    for f in facts.iter_mut() {
        f.retained = rng.bernoulli(params.state_retention);
    }
    Task { facts, needs }
}

/// Recovery stream for one task, seeded identically to Python:
/// `(seed * 1_000_003 + task_index * 131 + 7) mod 2**64`.
pub fn recovery_rng(seed: u64, task_index: u64) -> SplitMix64 {
    SplitMix64::new(
        seed.wrapping_mul(1_000_003)
            .wrapping_add(task_index.wrapping_mul(131))
            .wrapping_add(7),
    )
}

/// The four context policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Full,
    Summary,
    Sch,
    Oracle,
}

impl Policy {
    pub fn name(&self) -> &'static str {
        match self {
            Policy::Full => "full",
            Policy::Summary => "summary",
            Policy::Sch => "sch",
            Policy::Oracle => "oracle",
        }
    }

    pub fn from_str(s: &str) -> Option<Policy> {
        match s {
            "full" => Some(Policy::Full),
            "summary" => Some(Policy::Summary),
            "sch" => Some(Policy::Sch),
            "oracle" => Some(Policy::Oracle),
            _ => None,
        }
    }

    pub fn all() -> [Policy; 4] {
        [Policy::Full, Policy::Summary, Policy::Sch, Policy::Oracle]
    }
}

/// Per-policy result on one task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyResult {
    pub policy: Policy,
    pub steps_ok: usize,
    pub steps: usize,
    pub task_pass: bool,
    pub strict_task_pass: bool,
    pub total_tokens: u64,
    pub recoveries: u64,
    pub recovery_hits: u64,
}

/// Evaluate one policy on one task. Mirrors Python `evaluate_policy`
/// exactly, including draw consumption: the recovery stream is drawn
/// from for every missing delayed fact in need order -- a recall draw,
/// then a false-positive draw, then a size draw iff a distractor is
/// pulled -- in both the hit and the miss branch, so `recall = 1.0`
/// (oracle) and `recall = 0.0` consume the same draws as `recall = 0.9`.
pub fn evaluate_policy(
    task: &Task,
    policy: Policy,
    params: &TaskParams,
    seed: u64,
    task_index: u64,
) -> PolicyResult {
    let mut steps_ok = 0usize;
    let mut total_tokens: u64 = 0;
    let mut recoveries: u64 = 0;
    let mut recovery_hits: u64 = 0;
    let mut rng = match policy {
        Policy::Sch | Policy::Oracle => Some(recovery_rng(seed, task_index)),
        _ => None,
    };

    for t in 0..task.steps() {
        if policy == Policy::Full {
            // Context = every fact produced so far, at full size.
            let produced_tokens: u64 = task
                .facts
                .iter()
                .filter(|f| f.produced_step <= t)
                .map(|f| f.tokens)
                .sum();
            total_tokens += produced_tokens;
            steps_ok += 1;
            continue;
        }
        // State tokens: retained facts so far at compressed size
        // (integer truncation, as in Python), plus the fixed
        // interpretation overhead.
        let state_tokens: u64 = params.state_overhead
            + task
                .facts
                .iter()
                .filter(|f| f.produced_step <= t && f.retained)
                .map(|f| (f.tokens as f64 * params.compression) as u64)
                .sum::<u64>();
        let mut step_tokens = state_tokens;
        let mut step_ok = true;
        // A fact needed at its own production step is the current
        // observation: always in context. Only delayed facts can be
        // lost to the state.
        let delayed_needed: Vec<&Fact> = task.needs[t]
            .iter()
            .map(|&k| &task.facts[k])
            .filter(|f| f.produced_step < t)
            .collect();
        match policy {
            Policy::Summary => {
                for f in delayed_needed {
                    if !f.retained {
                        step_ok = false;
                        break;
                    }
                }
            }
            Policy::Sch | Policy::Oracle => {
                let recall = if policy == Policy::Oracle {
                    1.0
                } else {
                    params.recall
                };
                let fp = if policy == Policy::Oracle {
                    0.0
                } else {
                    params.fp_per_lookup
                };
                let rng = rng.as_mut().unwrap();
                for f in delayed_needed {
                    if f.retained {
                        continue;
                    }
                    recoveries += 1;
                    if rng.bernoulli(recall) {
                        recovery_hits += 1;
                        step_tokens += f.tokens + params.lookup_overhead;
                        if rng.bernoulli(fp) {
                            step_tokens +=
                                rng.randint(params.min_fact_tokens, params.max_fact_tokens);
                        }
                    } else {
                        step_ok = false;
                        // Still pay the lookup overhead on a miss.
                        step_tokens += params.lookup_overhead;
                        if rng.bernoulli(fp) {
                            step_tokens +=
                                rng.randint(params.min_fact_tokens, params.max_fact_tokens);
                        }
                    }
                }
            }
            Policy::Full => unreachable!(),
        }
        total_tokens += step_tokens;
        if step_ok {
            steps_ok += 1;
        }
    }

    let threshold = (params.pass_step_fraction * task.steps() as f64).ceil() as usize;
    PolicyResult {
        policy,
        steps_ok,
        steps: task.steps(),
        task_pass: steps_ok >= threshold,
        strict_task_pass: steps_ok == task.steps(),
        total_tokens,
        recoveries,
        recovery_hits,
    }
}

/// One value per policy.
#[derive(Clone, Debug)]
pub struct PolicySet<T> {
    pub full: T,
    pub summary: T,
    pub sch: T,
    pub oracle: T,
}

impl<T> PolicySet<T> {
    pub fn get(&self, policy: Policy) -> &T {
        match policy {
            Policy::Full => &self.full,
            Policy::Summary => &self.summary,
            Policy::Sch => &self.sch,
            Policy::Oracle => &self.oracle,
        }
    }
}

/// Per-policy aggregate over a set of tasks (exact integer sums plus
/// the float rates Python reports).
#[derive(Clone, Debug)]
pub struct PolicyAggregate {
    pub tasks: usize,
    pub passes: u64,
    pub strict_passes: u64,
    pub steps_ok: u64,
    pub steps: u64,
    pub tokens: u64,
    pub recoveries: u64,
    pub recovery_hits: u64,
    pub task_pass_rate: f64,
    pub strict_task_pass_rate: f64,
    pub step_success_rate: f64,
    pub mean_tokens_per_task: f64,
    pub mean_tokens_per_step: f64,
    pub recoveries_per_task: f64,
    pub recovery_hit_rate: Option<f64>,
}

/// Evaluate every policy on the same tasks. Mirrors Python
/// `evaluate_all` (same per-task seed/task_index for the recovery
/// stream).
pub fn evaluate_all(
    tasks: &[Task],
    params: &TaskParams,
    seed: u64,
) -> PolicySet<PolicyAggregate> {
    let mut sums: PolicySet<(u64, u64, u64, u64, u64, u64, u64)> = PolicySet {
        full: (0, 0, 0, 0, 0, 0, 0),
        summary: (0, 0, 0, 0, 0, 0, 0),
        sch: (0, 0, 0, 0, 0, 0, 0),
        oracle: (0, 0, 0, 0, 0, 0, 0),
    };
    for (i, task) in tasks.iter().enumerate() {
        for p in Policy::all() {
            let r = evaluate_policy(task, p, params, seed, i as u64);
            let s = match p {
                Policy::Full => &mut sums.full,
                Policy::Summary => &mut sums.summary,
                Policy::Sch => &mut sums.sch,
                Policy::Oracle => &mut sums.oracle,
            };
            s.0 += r.task_pass as u64;
            s.1 += r.strict_task_pass as u64;
            s.2 += r.steps_ok as u64;
            s.3 += r.steps as u64;
            s.4 += r.total_tokens;
            s.5 += r.recoveries;
            s.6 += r.recovery_hits;
        }
    }
    let n = tasks.len();
    let mk = |s: (u64, u64, u64, u64, u64, u64, u64)| PolicyAggregate {
        tasks: n,
        passes: s.0,
        strict_passes: s.1,
        steps_ok: s.2,
        steps: s.3,
        tokens: s.4,
        recoveries: s.5,
        recovery_hits: s.6,
        task_pass_rate: s.0 as f64 / n as f64,
        strict_task_pass_rate: s.1 as f64 / n as f64,
        step_success_rate: s.2 as f64 / s.3 as f64,
        mean_tokens_per_task: s.4 as f64 / n as f64,
        mean_tokens_per_step: s.4 as f64 / s.3 as f64,
        recoveries_per_task: s.5 as f64 / n as f64,
        recovery_hit_rate: if s.5 > 0 {
            Some(s.6 as f64 / s.5 as f64)
        } else {
            None
        },
    };
    PolicySet {
        full: mk(sums.full),
        summary: mk(sums.summary),
        sch: mk(sums.sch),
        oracle: mk(sums.oracle),
    }
}

/// One sensitivity-sweep cell.
#[derive(Clone, Debug)]
pub struct SweepCell {
    pub delayed_frac: f64,
    pub state_retention: f64,
    pub recall: f64,
    pub summary_task_pass: f64,
    pub sch_task_pass: f64,
    pub sch_gain_pp: f64,
    pub sch_token_saving_vs_full_pct: f64,
    pub sch_within_full_pp: f64,
}

/// Sensitivity sweep: 3x3x3 grid over (delayed_frac, state_retention,
/// recall), same seeds and order as Python `sensitivity_sweep`.
pub fn sensitivity_sweep(params: &TaskParams, seed: u64, n_tasks: usize) -> Vec<SweepCell> {
    let mut cells = Vec::new();
    for df in [0.15, 0.35, 0.55] {
        for sr in [0.55, 0.75, 0.9] {
            for rc in [0.7, 0.9, 1.0] {
                let p = TaskParams {
                    delayed_frac: df,
                    state_retention: sr,
                    recall: rc,
                    ..*params
                };
                let tasks: Vec<Task> = (0..n_tasks)
                    .map(|i| generate_task(&mut SplitMix64::new(seed + i as u64), &p))
                    .collect();
                let res = evaluate_all(&tasks, &p, seed);
                cells.push(SweepCell {
                    delayed_frac: df,
                    state_retention: sr,
                    recall: rc,
                    summary_task_pass: res.summary.task_pass_rate,
                    sch_task_pass: res.sch.task_pass_rate,
                    sch_gain_pp: 100.0
                        * (res.sch.task_pass_rate - res.summary.task_pass_rate),
                    sch_token_saving_vs_full_pct: 100.0
                        * (1.0 - res.sch.mean_tokens_per_task / res.full.mean_tokens_per_task),
                    sch_within_full_pp: 100.0
                        * (res.full.task_pass_rate - res.sch.task_pass_rate),
                });
            }
        }
    }
    cells
}

/// Per-policy summary of the pooled multi-seed experiment (float rates
/// averaged across seeds, as in Python `run_experiment`).
#[derive(Clone, Debug)]
pub struct PolicySummary {
    pub task_pass_rate: f64,
    pub strict_task_pass_rate: f64,
    pub step_success_rate: f64,
    pub mean_tokens_per_task: f64,
    pub mean_tokens_per_step: f64,
    pub recoveries_per_task: f64,
    pub tasks: usize,
    pub recovery_hit_rate: Option<f64>,
}

/// Main experiment: `n_seeds` seeds x `n_tasks` tasks, pooled.
/// Mirrors Python `run_experiment` exactly (same per-seed base seeds
/// `seed + s * 100_003`, same no-delay sanity, same sweep).
pub struct Experiment {
    pub params: TaskParams,
    pub seeds: usize,
    pub tasks_per_seed: usize,
    pub policies: PolicySet<PolicySummary>,
    pub per_seed_task_pass_pct: PolicySet<Vec<f64>>,
    pub per_seed: Vec<PolicySet<PolicyAggregate>>,
    pub sensitivity_sweep: Vec<SweepCell>,
    pub no_delay_sanity: PolicySet<f64>,
}

pub fn run_experiment(
    params: &TaskParams,
    seed: u64,
    n_tasks: usize,
    n_seeds: usize,
) -> Experiment {
    let mut per_seed = Vec::new();
    for s in 0..n_seeds {
        let base = seed + s as u64 * 100_003;
        let tasks: Vec<Task> = (0..n_tasks)
            .map(|i| generate_task(&mut SplitMix64::new(base + i as u64), params))
            .collect();
        per_seed.push(evaluate_all(&tasks, params, base));
    }
    let avg = |f: fn(&PolicyAggregate) -> f64, p: Policy| -> f64 {
        per_seed
            .iter()
            .map(|ps| f(ps.get(p)))
            .sum::<f64>()
            / n_seeds as f64
    };
    let mk = |p: Policy| PolicySummary {
        task_pass_rate: avg(|a| a.task_pass_rate, p),
        strict_task_pass_rate: avg(|a| a.strict_task_pass_rate, p),
        step_success_rate: avg(|a| a.step_success_rate, p),
        mean_tokens_per_task: avg(|a| a.mean_tokens_per_task, p),
        mean_tokens_per_step: avg(|a| a.mean_tokens_per_step, p),
        recoveries_per_task: avg(|a| a.recoveries_per_task, p),
        tasks: n_tasks * n_seeds,
        recovery_hit_rate: {
            let hits: Vec<f64> = per_seed
                .iter()
                .filter_map(|ps| ps.get(p).recovery_hit_rate)
                .collect();
            if hits.is_empty() {
                None
            } else {
                Some(hits.iter().sum::<f64>() / hits.len() as f64)
            }
        },
    };
    let pct = |p: Policy| -> Vec<f64> {
        per_seed
            .iter()
            .map(|ps| 100.0 * ps.get(p).task_pass_rate)
            .collect()
    };
    // No-delay sanity: nothing ever needed later -> all pass.
    let nd_params = TaskParams {
        delayed_frac: 0.0,
        ..*params
    };
    let nd_tasks: Vec<Task> = (0..400)
        .map(|i| generate_task(&mut SplitMix64::new(seed + 999_983 + i as u64), &nd_params))
        .collect();
    let no_delay = evaluate_all(&nd_tasks, &nd_params, seed + 999_983);
    Experiment {
        params: *params,
        seeds: n_seeds,
        tasks_per_seed: n_tasks,
        policies: PolicySet {
            full: mk(Policy::Full),
            summary: mk(Policy::Summary),
            sch: mk(Policy::Sch),
            oracle: mk(Policy::Oracle),
        },
        per_seed_task_pass_pct: PolicySet {
            full: pct(Policy::Full),
            summary: pct(Policy::Summary),
            sch: pct(Policy::Sch),
            oracle: pct(Policy::Oracle),
        },
        per_seed,
        sensitivity_sweep: sensitivity_sweep(params, seed, 400),
        no_delay_sanity: PolicySet {
            full: no_delay.full.task_pass_rate,
            summary: no_delay.summary.task_pass_rate,
            sch: no_delay.sch.task_pass_rate,
            oracle: no_delay.oracle.task_pass_rate,
        },
    }
}

/// Stable short digest of a generated task (parity checks). Same
/// recurrence as Python `task_digest` (wrapping 64-bit).
pub fn task_digest(task: &Task) -> u64 {
    let mut h: u64 = 0;
    for f in &task.facts {
        h = h
            .wrapping_mul(31)
            .wrapping_add(f.key as u64);
        h = h
            .wrapping_mul(31)
            .wrapping_add(f.tokens);
        h = h
            .wrapping_mul(31)
            .wrapping_add(f.needed_step as u64);
        h = h
            .wrapping_mul(31)
            .wrapping_add(if f.retained { 1 } else { 0 });
    }
    h
}
