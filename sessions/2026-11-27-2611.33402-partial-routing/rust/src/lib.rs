//! Partial Routing: Trace-Tuned Delegation for Resource-Efficient
//! Agentic Workflows.
//!
//! arXiv:2611.33402 "Partial Routing: Trace-Tuned Delegation for
//! Resource-Efficient Agentic Workflows".
//!
//! Rust port of `src/arxiv_lab/agent/partial_routing.py` (std only, no
//! external crates, no network). The Python version stays the primary
//! verdict carrier; this port mirrors its mechanism and is verified
//! against it (see `src/bin/verify.rs` and `tests/`).
//!
//! Core claim implemented here: an agentic workflow can be split into
//! subtasks and routed between two models — an expensive high-quality
//! LEAD and a cheap lower-quality WORKER — at subtask granularity.
//! From both models' past runs, recurring subtasks are measured per
//! subtask type (quality and cost); those measurements produce
//! delegation guidance, and the lead decides during execution when
//! and what to delegate, WITHOUT training either model. A parameter
//! delta controls how much is delegated.
//!
//! Bounded CPU reproduction (no model, no network; deterministic
//! given a seed). World, constants, and arithmetic order mirror the
//! Python module exactly so analytic quantities agree bit-for-bit
//! on identical inputs (see the parity sweep in `tests/`).
//!
//! Deliberate divergences from Python (see also `rust/README.md`):
//! * RNG: Python uses `random.Random` (Mersenne Twister); Rust uses
//!   an in-repo SplitMix64. Task sets and Monte-Carlo streams
//!   therefore differ in sampled values while keeping the same
//!   shapes and distributions. Exact cross-language equality is
//!   the parity sweep's job (`tests/parity_cases.txt`, generated
//!   from Python): it compares deterministic/analytic quantities
//!   exactly and never compares raw Monte-Carlo streams.
//! * No hash maps in the computation: guidance is stored in fixed
//!   arrays indexed by subtask type, so there is no hash-order
//!   dependence anywhere (no PYTHONHASHSEED analogue needed).

// ------------------------------------------------------------------- world
/// Subtask types, in the Python module's `SUBTASK_TYPES` order.
/// 0 = plan, 1 = code, 2 = verify, 3 = test, 4 = summarize.
pub const SUBTASK_TYPES: [&str; 5] = ["plan", "code", "verify", "test", "summarize"];
/// Types counted as "hard" for task-level routing.
pub const HARD_TYPES: [usize; 2] = [0, 1]; // plan, code
/// Base success probabilities per subtask type (before difficulty and
/// lead-load adjustments). Gaps lead-worker: plan .30, code .25,
/// verify .19, test .12, summarize .06.
pub const LEAD_Q: [f64; 5] = [0.88, 0.90, 0.92, 0.93, 0.95];
pub const WORKER_Q: [f64; 5] = [0.58, 0.65, 0.73, 0.81, 0.89];
pub const LEAD_COST: f64 = 10.0;
pub const WORKER_COST: f64 = 1.0;
/// Lead coordination cost per delegated subtask.
pub const DELEGATION_OVERHEAD: f64 = 0.5;
/// Lead success lost per prior lead-handled subtask in the task.
pub const LOAD_PENALTY: f64 = 0.06;
/// Success lost per unit subtask difficulty.
pub const DIFF_EFFECT: f64 = 0.05;
pub const N_SUBTASKS: usize = 5;
/// Lead cost per subtask with a subtask breakdown (ablation).
pub const BREAKDOWN_COST: f64 = 9.0;
/// Breakdown halves the load penalty (stated modelling assumption).
pub const BREAKDOWN_LOAD_FACTOR: f64 = 0.5;
/// Route a whole task to the worker iff hard-subtask fraction <= this.
pub const TASK_LEVEL_HARD_FRAC: f64 = 0.4;
pub const DIFFICULTIES: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
/// Per-task P(a subtask is a hard type).
pub const HARD_PROFILES: [f64; 4] = [0.2, 0.4, 0.6, 0.8];

/// Deterministic RNG: SplitMix64. `next_f64` returns a uniform
/// double in [0, 1) from the top 53 bits, the standard construction.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        const DEN: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * DEN
    }

    /// Uniform index into `n` (n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    pub fn choice_f64(&mut self, vals: &[f64]) -> f64 {
        vals[self.below(vals.len())]
    }

    pub fn choice_usize(&mut self, vals: &[usize]) -> usize {
        vals[self.below(vals.len())]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subtask {
    /// Index into SUBTASK_TYPES.
    pub type_idx: usize,
    /// Difficulty in [0, 1].
    pub difficulty: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    /// Subtasks in execution order.
    pub subtasks: Vec<Subtask>,
}

#[derive(Clone, Debug)]
pub struct Trace {
    /// 0 = lead, 1 = worker.
    pub model: usize,
    pub type_idx: usize,
    pub difficulty: f64,
    pub success: bool,
    pub cost: f64,
}

/// Delegation guidance measured from past-run traces.
#[derive(Clone, Debug)]
pub struct Guidance {
    pub lead_q: [f64; 5],
    pub worker_q: [f64; 5],
    pub lead_cost: [f64; 5],
    pub worker_cost: [f64; 5],
    /// lead_q - worker_q per type.
    pub gap: [f64; 5],
    pub gap_max: f64,
    pub n_traces: usize,
}

impl Guidance {
    /// Types delegated at this delta, in SUBTASK_TYPES order.
    pub fn delegate_types(&self, delta: f64) -> Vec<usize> {
        let thr = delta * self.gap_max;
        (0..5).filter(|&t| self.gap[t] <= thr).collect()
    }
}

pub fn make_tasks(seed: u64, n_tasks: usize) -> Vec<Task> {
    let mut rng = Rng::new(seed);
    let mut tasks = Vec::with_capacity(n_tasks);
    for _ in 0..n_tasks {
        let profile = rng.choice_f64(&HARD_PROFILES);
        let mut subs = Vec::with_capacity(N_SUBTASKS);
        for _ in 0..N_SUBTASKS {
            let hard = rng.next_f64() < profile;
            let type_idx = if hard {
                rng.choice_usize(&HARD_TYPES)
            } else {
                // easy types are verify/test/summarize (indices 2,3,4)
                rng.choice_usize(&[2, 3, 4])
            };
            subs.push(Subtask {
                type_idx,
                difficulty: rng.choice_f64(&DIFFICULTIES),
            });
        }
        tasks.push(Task { subtasks: subs });
    }
    tasks
}

// -------------------------------------------------------------- past runs
/// True success probability of one subtask attempt. `lead_done` is
/// the number of subtasks the lead has already handled in this task.
pub fn success_prob(model: usize, subtask: &Subtask, lead_done: usize, breakdown: bool) -> f64 {
    let q = if model == 0 {
        let mut load = LOAD_PENALTY * (lead_done as f64);
        if breakdown {
            load *= BREAKDOWN_LOAD_FACTOR;
        }
        LEAD_Q[subtask.type_idx] - DIFF_EFFECT * subtask.difficulty - load
    } else {
        WORKER_Q[subtask.type_idx] - DIFF_EFFECT * subtask.difficulty
    };
    q.clamp(0.05, 0.99)
}

pub fn past_traces(seed: u64, n_tasks: usize) -> Vec<Trace> {
    let mut rng = Rng::new(seed);
    let tasks = make_tasks(seed, n_tasks);
    let mut traces = Vec::with_capacity(n_tasks * N_SUBTASKS * 2);
    for task in &tasks {
        for model in [0usize, 1usize] {
            for st in &task.subtasks {
                let q = success_prob(model, st, 0, false);
                let ok = rng.next_f64() < q;
                let cost = if model == 0 { LEAD_COST } else { WORKER_COST };
                traces.push(Trace {
                    model,
                    type_idx: st.type_idx,
                    difficulty: st.difficulty,
                    success: ok,
                    cost,
                });
            }
        }
    }
    traces
}

/// Measure per-type quality and cost from past-run traces.
/// Summation follows trace order, matching the Python module.
pub fn learn_guidance(traces: &[Trace]) -> Guidance {
    let mut lead_q = [0.0; 5];
    let mut worker_q = [0.0; 5];
    let mut lead_cost = [0.0; 5];
    let mut worker_cost = [0.0; 5];
    for t in 0..5 {
        for model in [0usize, 1usize] {
            let mut n = 0usize;
            let mut succ = 0usize;
            let mut cost = 0.0f64;
            for tr in traces {
                if tr.model == model && tr.type_idx == t {
                    n += 1;
                    if tr.success {
                        succ += 1;
                    }
                    cost += tr.cost;
                }
            }
            let q = (succ as f64) / (n as f64);
            let c = cost / (n as f64);
            if model == 0 {
                lead_q[t] = q;
                lead_cost[t] = c;
            } else {
                worker_q[t] = q;
                worker_cost[t] = c;
            }
        }
    }
    let mut gap = [0.0; 5];
    for t in 0..5 {
        gap[t] = lead_q[t] - worker_q[t];
    }
    let mut gap_max = gap[0];
    for t in 1..5 {
        if gap[t] > gap_max {
            gap_max = gap[t];
        }
    }
    Guidance {
        lead_q,
        worker_q,
        lead_cost,
        worker_cost,
        gap,
        gap_max,
        n_traces: traces.len(),
    }
}

/// Build guidance directly from per-type measured gaps (used by the
/// unit tests and the parity sweep to fix guidance exactly).
pub fn guidance_from_gaps(gap: [f64; 5]) -> Guidance {
    let mut gap_max = gap[0];
    for t in 1..5 {
        if gap[t] > gap_max {
            gap_max = gap[t];
        }
    }
    Guidance {
        lead_q: [0.0; 5],
        worker_q: [0.0; 5],
        lead_cost: [LEAD_COST; 5],
        worker_cost: [WORKER_COST; 5],
        gap,
        gap_max,
        n_traces: 0,
    }
}

// --------------------------------------------------------------- policies
/// Partial-routing assignment: false = lead, true = worker, per
/// subtask, from the trace-tuned guidance at this delta.
pub fn delegate(task: &Task, guidance: &Guidance, delta: f64) -> Vec<bool> {
    let allowed = guidance.delegate_types(delta);
    task.subtasks
        .iter()
        .map(|st| allowed.contains(&st.type_idx))
        .collect()
}

/// No-guidance heuristic: delegate summarize + test always.
pub fn fixed_easy_assignment(task: &Task) -> Vec<bool> {
    task.subtasks
        .iter()
        .map(|st| st.type_idx == 3 || st.type_idx == 4)
        .collect()
}

/// Task-level routing baseline: the WHOLE task goes to one model —
/// worker if its hard-subtask fraction is low, else lead.
pub fn task_level_assignment(task: &Task) -> Vec<bool> {
    let hard = task
        .subtasks
        .iter()
        .filter(|st| HARD_TYPES.contains(&st.type_idx))
        .count();
    let whole = (hard as f64) / (task.subtasks.len() as f64) <= TASK_LEVEL_HARD_FRAC;
    vec![whole; task.subtasks.len()]
}

/// Random delegation baseline: each subtask to the worker w.p. 1/2.
pub fn random_assignment(task: &Task, rng: &mut Rng) -> Vec<bool> {
    task.subtasks.iter().map(|_| rng.next_f64() < 0.5).collect()
}

/// Brute-force best assignment by analytic expected success
/// (tie-break: lowest cost). Iterates assignments in the same order
/// as Python's `itertools.product((0, 1), repeat=n)` (last subtask
/// varies fastest) and keeps the first maximum, so ties resolve
/// identically. Uses TRUE qualities — an upper bound.
pub fn oracle_assignment(task: &Task, breakdown: bool) -> Vec<bool> {
    let n = task.subtasks.len();
    let mut best_p = f64::NEG_INFINITY;
    let mut best_c = f64::INFINITY;
    let mut best: Vec<bool> = vec![false; n];
    for mask in 0..(1usize << n) {
        // product order: subtask 0 is the most significant bit.
        let assign: Vec<bool> = (0..n).map(|i| ((mask >> (n - 1 - i)) & 1) == 1).collect();
        let (p, c) = analytic_expected(task, &assign, breakdown);
        if p > best_p || (p == best_p && c < best_c) {
            best_p = p;
            best_c = c;
            best = assign;
        }
    }
    best
}

// -------------------------------------------------------------- execution
/// One stochastic run of a task under an assignment.
/// Returns (task_success, cost, subtask_score). Task success is
/// strict: every subtask must succeed.
pub fn execute(task: &Task, assignment: &[bool], rng: &mut Rng, breakdown: bool) -> (bool, f64, f64) {
    let mut lead_done = 0usize;
    let mut cost = 0.0f64;
    let mut ok_n = 0usize;
    for (st, &a) in task.subtasks.iter().zip(assignment.iter()) {
        let q;
        if !a {
            q = success_prob(0, st, lead_done, breakdown);
            cost += if breakdown { BREAKDOWN_COST } else { LEAD_COST };
            lead_done += 1;
        } else {
            q = success_prob(1, st, lead_done, breakdown);
            cost += WORKER_COST + DELEGATION_OVERHEAD;
        }
        if rng.next_f64() < q {
            ok_n += 1;
        }
    }
    let n = task.subtasks.len();
    (ok_n == n, cost, (ok_n as f64) / (n as f64))
}

/// Exact (success probability, cost) of an assignment under the
/// world's true qualities — no sampling noise. Multiplication and
/// accumulation follow subtask order, matching Python bit-for-bit
/// on identical inputs.
pub fn analytic_expected(task: &Task, assignment: &[bool], breakdown: bool) -> (f64, f64) {
    let mut lead_done = 0usize;
    let mut p = 1.0f64;
    let mut cost = 0.0f64;
    for (st, &a) in task.subtasks.iter().zip(assignment.iter()) {
        let q;
        if !a {
            q = success_prob(0, st, lead_done, breakdown);
            cost += if breakdown { BREAKDOWN_COST } else { LEAD_COST };
            lead_done += 1;
        } else {
            q = success_prob(1, st, lead_done, breakdown);
            cost += WORKER_COST + DELEGATION_OVERHEAD;
        }
        p *= q;
    }
    (p, cost)
}

#[derive(Clone, Copy, Debug)]
pub struct RunResult {
    pub success_rate: f64,
    pub mean_cost: f64,
    pub mean_score: f64,
    pub n: usize,
    pub trials: usize,
    /// Analytic (exact-expectation) counterparts, same policies/tasks.
    pub analytic_success: f64,
    pub analytic_cost: f64,
}

/// Monte Carlo evaluation of one routing policy over tasks:
/// `trials` seeded repetitions per task. The assignment closure
/// receives a separate RNG (mirroring Python, where the random
/// baseline draws from its own `Random(77)` stream while trials
/// draw from the policy seed). One RNG drives all trials, so
/// policies compared with the same seed share common random numbers.
pub fn run_policy<F>(
    tasks: &[Task],
    mut assignment_fn: F,
    seed: u64,
    trials: usize,
    breakdown: bool,
) -> RunResult
where
    F: FnMut(&Task, &mut Rng) -> Vec<bool>,
{
    let mut rng = Rng::new(seed);
    let mut assign_rng = Rng::new(77);
    let mut succ = 0u64;
    let mut cost = 0.0f64;
    let mut score = 0.0f64;
    let mut n_runs = 0u64;
    let mut a_succ = 0.0f64;
    let mut a_cost = 0.0f64;
    for task in tasks {
        let assign = assignment_fn(task, &mut assign_rng);
        let (ap, ac) = analytic_expected(task, &assign, breakdown);
        a_succ += ap;
        a_cost += ac;
        for _ in 0..trials {
            let (ok, c, s) = execute(task, &assign, &mut rng, breakdown);
            if ok {
                succ += 1;
            }
            cost += c;
            score += s;
            n_runs += 1;
        }
    }
    RunResult {
        success_rate: (succ as f64) / (n_runs as f64),
        mean_cost: cost / (n_runs as f64),
        mean_score: score / (n_runs as f64),
        n: tasks.len(),
        trials,
        analytic_success: a_succ / (tasks.len() as f64),
        analytic_cost: a_cost / (tasks.len() as f64),
    }
}

/// Lead-alone assignment (all false) for a task of `n` subtasks.
pub fn lead_assignment(n: usize) -> Vec<bool> {
    vec![false; n]
}
