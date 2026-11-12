//! WikiTrace: evidence-linked memory for LLM agents (arXiv:2611.14320) —
//! Rust port of the Python mechanism in `wikitrace.py` (same session
//! folder). The Python version stays the primary verdict carrier; this
//! port never replaces it.
//!
//! Line-for-line port of the pure logic: SplitMix64 RNG, task generation,
//! the default agent's paired choices, run trajectories, the compiler
//! (compilations + insights with source links), the evidence check
//! (`verify_lesson`), the descent triggers, the three solve arms
//! (no-memory / plain lessons / WikiTrace), and the main experiment with
//! its gain / regression / descent / correction / evidence-read metrics
//! and the successful-runs-only / failed-runs-only memory subsets.
//! No external crates, std only, no network.
//!
//! Deliberate divergences from Python (all documented here; anything else
//! is a bug):
//!  * `randbelow` compares in `u128`: for `n == 1` the rejection limit is
//!    2^64, which wraps to 0 as `u64` and would loop forever (a `choice()`
//!    on a 1-element slice calls `randbelow(1)`). Python computes the
//!    same limit in bigint. (Same trap as the 2026-11-11 port.)
//!  * `f64` conversion of a `u64` uses the same round-to-nearest-even rule
//!    in both languages, so `SplitMix64::random` streams match exactly.
//!  * Python `Memory` stores compilations/trajectories as dicts keyed by
//!    run id; Rust uses `HashMap<u64, ...>` — iteration order over those
//!    maps is never observable (lookups are by key only), so results are
//!    order-independent. Insights keep insertion order in a `Vec`, as in
//!    Python.
//!  * Python `Scope` is `"any"` or `("attr", a)`; Rust models it as an
//!    enum (`Scope::Any` / `Scope::Attr(u64)`) — same semantics.
//!  * Python `solve` takes `mode` as a string (`"none"`/`"plain"`/
//!    `"wikitrace"`); Rust takes a `Mode` enum — same three arms.

use std::collections::HashMap;

// ---------------------------------------------------------------- RNG

const MASK64: u64 = u64::MAX;

#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed & MASK64 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn randbelow(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        // Compare in u128: for n == 1 the limit is 2^64, which wraps to 0
        // as u64 and would loop forever (choice() on a 1-element slice
        // calls randbelow(1)). Python computes the same limit in bigint.
        let total: u128 = 1u128 << 64;
        let limit = total - (total % n as u128);
        loop {
            let v = self.next_u64();
            if (v as u128) < limit {
                return v % n;
            }
        }
    }

    pub fn random(&mut self) -> f64 {
        self.next_u64() as f64 / 18446744073709551616.0
    }

    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        &seq[self.randbelow(seq.len() as u64) as usize]
    }
}

// ---------------------------------------------------------------- parameters

pub const STEP_TYPES: [u8; 3] = [0, 1, 2]; // 0 locate, 1 operate, 2 verify
pub const STEP_NAMES: [&str; 3] = ["locate", "operate", "verify"];
pub const N_ACTIONS: u64 = 3;
pub const N_ATTRS: u64 = 4; // object kinds 0..3
pub const P_DEFAULT: f64 = 0.6; // default agent picks the gold action w.p.
pub const CORRUPT_P: f64 = 0.20; // failed-run insight misattributes w.p.
pub const OBSERVED_P: f64 = 0.80; // trajectory step observation recorded w.p.
pub const SCOPE_ANY_P: f64 = 0.50; // insight scope recorded as "any" w.p.

pub const SEED: u64 = 20261112;
pub const N_TRAIN: usize = 12;
pub const N_TEST: usize = 600;

pub fn gold_action(step_type: u8, attr: u64) -> u64 {
    match step_type {
        0 => 0,
        1 => 1,
        _ => attr % N_ACTIONS, // verify step: gold depends on the object kind
    }
}

// ---------------------------------------------------------------- tasks

#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub tid: u64,
    pub attr: u64,
}

impl Task {
    pub fn new(tid: u64, attr: u64) -> Self {
        Task { tid, attr }
    }
}

pub fn gen_tasks(seed: u64, n: usize) -> Vec<Task> {
    let mut rng = SplitMix64::new(seed);
    (0..n as u64)
        .map(|i| Task::new(i, rng.randbelow(N_ATTRS)))
        .collect()
}

pub fn default_choices(task: &Task, base_seed: u64) -> [u64; 3] {
    let mut rng = SplitMix64::new(
        base_seed
            .wrapping_mul(1_000_003)
            .wrapping_add(task.tid.wrapping_mul(7919)),
    );
    let mut choices = [0u64; 3];
    for (i, &st) in STEP_TYPES.iter().enumerate() {
        let gold = gold_action(st, task.attr);
        if rng.random() < P_DEFAULT {
            choices[i] = gold;
        } else {
            let wrong: Vec<u64> = (0..N_ACTIONS).filter(|&a| a != gold).collect();
            choices[i] = *rng.choice(&wrong);
        }
    }
    choices
}

// ---------------------------------------------------------------- runs

#[derive(Clone, Debug, PartialEq)]
pub struct StepRecord {
    pub step_type: u8,
    pub chosen: u64,
    pub ok: bool,
    pub observed: bool,
}

impl StepRecord {
    pub fn new(step_type: u8, chosen: u64, ok: bool, observed: bool) -> Self {
        StepRecord { step_type, chosen, ok, observed }
    }
}

#[derive(Clone, Debug)]
pub struct Run {
    pub rid: u64,
    pub task: Task,
    pub steps: Vec<StepRecord>,
    pub success: bool,
}

impl Run {
    pub fn new(rid: u64, task: Task, steps: Vec<StepRecord>) -> Self {
        let success = steps.iter().all(|s| s.ok);
        Run { rid, task, steps, success }
    }
}

pub fn run_default(task: &Task, base_seed: u64, obs_seed: u64) -> Run {
    let mut rng = SplitMix64::new(
        obs_seed
            .wrapping_mul(1_000_003)
            .wrapping_add(task.tid.wrapping_mul(104729)),
    );
    let choices = default_choices(task, base_seed);
    let mut steps = Vec::with_capacity(3);
    for (i, &st) in STEP_TYPES.iter().enumerate() {
        let chosen = choices[i];
        let ok = chosen == gold_action(st, task.attr);
        let observed = rng.random() < OBSERVED_P;
        steps.push(StepRecord::new(st, chosen, ok, observed));
    }
    Run::new(task.tid, task.clone(), steps)
}

// ---------------------------------------------------------------- compiler

#[derive(Clone, Debug, PartialEq)]
pub enum Scope {
    Any,
    Attr(u64),
}

#[derive(Clone, Debug)]
pub struct Insight {
    pub iid: String,
    pub step_type: u8,
    pub claimed: u64,
    pub scope: Scope,
    pub source_run: u64,
    pub source_success: bool,
    /// Ground truth only; never shown to the agent.
    pub corrupted: bool,
}

impl Insight {
    pub fn new(
        iid: String,
        step_type: u8,
        claimed: u64,
        scope: Scope,
        source_run: u64,
        source_success: bool,
        corrupted: bool,
    ) -> Self {
        Insight { iid, step_type, claimed, scope, source_run, source_success, corrupted }
    }

    pub fn applies_to(&self, attr: u64) -> bool {
        match self.scope {
            Scope::Any => true,
            Scope::Attr(a) => a == attr,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Memory {
    pub insights: Vec<Insight>,
    pub compilations: HashMap<u64, String>, // rid -> str
    pub trajectories: HashMap<u64, Run>,    // rid -> Run
}

impl Memory {
    pub fn new(
        insights: Vec<Insight>,
        compilations: HashMap<u64, String>,
        trajectories: HashMap<u64, Run>,
    ) -> Self {
        Memory { insights, compilations, trajectories }
    }

    pub fn applicable(&self, step_type: u8, attr: u64) -> Vec<&Insight> {
        self.insights
            .iter()
            .filter(|ins| ins.step_type == step_type && ins.applies_to(attr))
            .collect()
    }

    /// Memory keeping only insights from runs with the given outcome
    /// (paper section 5.3: memory built from successful / failed runs).
    pub fn subset(&self, source_success: bool) -> Memory {
        let kept: Vec<Insight> = self
            .insights
            .iter()
            .filter(|ins| ins.source_success == source_success)
            .cloned()
            .collect();
        Memory::new(kept, self.compilations.clone(), self.trajectories.clone())
    }
}

pub fn compile_memory(runs: Vec<Run>, seed: u64) -> Memory {
    let mut rng = SplitMix64::new(seed);
    let mut insights: Vec<Insight> = Vec::new();
    let mut compilations: HashMap<u64, String> = HashMap::new();
    let mut trajectories: HashMap<u64, Run> = HashMap::new();
    for run in runs {
        trajectories.insert(run.rid, run.clone());
        if run.success {
            let decisions: Vec<String> = run
                .steps
                .iter()
                .map(|s| format!("step {}: action {} worked", STEP_NAMES[s.step_type as usize], s.chosen))
                .collect();
            compilations.insert(
                run.rid,
                format!(
                    "Run {} succeeded on attr={}. Decisions associated with success: {}.",
                    run.rid,
                    run.task.attr,
                    decisions.join(", ")
                ),
            );
        } else {
            let failed = run.steps.iter().find(|s| !s.ok).expect("failed run has a failing step");
            compilations.insert(
                run.rid,
                format!(
                    "Run {} FAILED on attr={}. First failing step: {} with action {} (observation: step failed). Plausible cause: wrong action for this object kind.",
                    run.rid,
                    run.task.attr,
                    STEP_NAMES[failed.step_type as usize],
                    failed.chosen
                ),
            );
        }
        for s in &run.steps {
            let mut claimed = s.chosen;
            let mut corrupted = false;
            if !run.success && rng.random() < CORRUPT_P {
                // Misattribution: the compiler blames an action the run
                // never took.
                let others: Vec<u64> = (0..N_ACTIONS).filter(|&a| a != s.chosen).collect();
                claimed = *rng.choice(&others);
                corrupted = true;
            }
            let scope = if rng.random() < SCOPE_ANY_P {
                Scope::Any
            } else {
                Scope::Attr(run.task.attr)
            };
            insights.push(Insight::new(
                format!("I-{:04}", insights.len()),
                s.step_type,
                claimed,
                scope,
                run.rid,
                run.success,
                corrupted,
            ));
        }
    }
    Memory::new(insights, compilations, trajectories)
}

// ---------------------------------------------------------------- evidence check

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Supported,
    Contradicted,
    Unsupported,
    Unknown,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Supported => "supported",
            Verdict::Contradicted => "contradicted",
            Verdict::Unsupported => "unsupported",
            Verdict::Unknown => "unknown",
        }
    }
}

pub fn descent_triggers(lessons: &[&Insight]) -> bool {
    if lessons.is_empty() {
        return false;
    }
    if lessons.iter().any(|ins| !ins.source_success) {
        return true;
    }
    let distinct: std::collections::HashSet<u64> =
        lessons.iter().map(|ins| ins.claimed).collect();
    if distinct.len() > 1 {
        return true;
    }
    if lessons.iter().any(|ins| ins.scope == Scope::Any) {
        return true;
    }
    false
}

pub fn verify_lesson(lesson: &Insight, task_attr: u64, memory: &Memory) -> (Verdict, u64) {
    let reads = 2; // one compilation read + one raw-trajectory read
    let _comp = memory.compilations.get(&lesson.source_run);
    let run = match memory.trajectories.get(&lesson.source_run) {
        Some(r) => r,
        None => return (Verdict::Unknown, reads),
    };
    let step = run.steps.iter().find(|s| s.step_type == lesson.step_type);
    let step = match step {
        Some(s) => s,
        None => return (Verdict::Unknown, reads),
    };
    if !step.observed {
        return (Verdict::Unknown, reads);
    }
    if lesson.claimed != step.chosen {
        return (Verdict::Contradicted, reads); // insight misattributes the run
    }
    if !step.ok {
        return (Verdict::Contradicted, reads); // claimed action failed in its own run
    }
    if lesson.scope == Scope::Any && run.task.attr != task_attr {
        return (Verdict::Unsupported, reads); // over-generalized beyond its evidence
    }
    (Verdict::Supported, reads)
}

// ---------------------------------------------------------------- agents

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    None,
    Plain,
    WikiTrace,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SolveStats {
    pub descents: u64,
    pub corrections: u64,
    pub evidence_reads: u64,
    pub lessons_followed: u64,
}

pub fn solve(task: &Task, memory: Option<&Memory>, mode: Mode, base_seed: u64) -> (bool, SolveStats) {
    let choices = default_choices(task, base_seed);
    let mut stats = SolveStats::default();
    let mut success = true;
    for (i, &st) in STEP_TYPES.iter().enumerate() {
        let mut action = choices[i];
        if let Some(mem) = memory {
            let lessons = mem.applicable(st, task.attr);
            if !lessons.is_empty() {
                let blind = lessons[0].claimed;
                match mode {
                    Mode::Plain => {
                        action = blind;
                        stats.lessons_followed += 1;
                    }
                    Mode::WikiTrace => {
                        if descent_triggers(&lessons) {
                            stats.descents += 1;
                            // Descend through the links in order and stop
                            // at the first lesson the evidence supports.
                            let mut supported: Option<&Insight> = None;
                            for ins in &lessons {
                                let (verdict, reads) = verify_lesson(ins, task.attr, mem);
                                stats.evidence_reads += reads;
                                if verdict == Verdict::Supported {
                                    supported = Some(ins);
                                    break;
                                }
                            }
                            if let Some(ins) = supported {
                                action = ins.claimed;
                                stats.lessons_followed += 1;
                            }
                            // else: no verified lesson -> fall back to the
                            // default policy
                        } else {
                            action = blind;
                            stats.lessons_followed += 1;
                        }
                        if action != blind {
                            stats.corrections += 1;
                        }
                    }
                    Mode::None => {}
                }
            }
        }
        if action != gold_action(st, task.attr) {
            success = false;
        }
    }
    (success, stats)
}

// ---------------------------------------------------------------- experiment

#[derive(Clone, Copy, Debug, Default)]
pub struct ArmMetrics {
    pub success: f64,
    pub gain: f64,
    pub regression: f64,
    pub regression_of_solved: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WikiTraceMetrics {
    pub arm: ArmMetrics,
    pub descent_rate: f64,
    pub correction_rate: f64,
    pub evidence_reads_per_task: f64,
    pub lessons_followed_per_task: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SubsetMetrics {
    pub success: f64,
    pub gain: f64,
    pub regression: f64,
    pub n_insights: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ExperimentResults {
    pub none: ArmMetrics,
    pub plain: ArmMetrics,
    pub plain_lessons_followed_per_task: f64,
    pub wikitrace: WikiTraceMetrics,
    pub wikitrace_succ_only: SubsetMetrics,
    pub wikitrace_fail_only: SubsetMetrics,
    pub n_train_runs: usize,
    pub n_test_tasks: usize,
    pub n_insights: usize,
    pub n_insights_corrupted: usize,
    pub n_insights_from_failed_runs: usize,
}

fn arm_metrics(results: &[(bool, SolveStats)], base: &[bool]) -> ArmMetrics {
    let n = results.len() as f64;
    let success = results.iter().filter(|(ok, _)| *ok).count() as f64 / n;
    let gain = results
        .iter()
        .zip(base.iter())
        .filter(|((ok, _), b)| !**b && *ok)
        .count() as f64
        / n;
    let regr = results
        .iter()
        .zip(base.iter())
        .filter(|((ok, _), b)| **b && !*ok)
        .count() as f64
        / n;
    let solved = base.iter().filter(|b| **b).count().max(1) as f64;
    let regr_solved = results
        .iter()
        .zip(base.iter())
        .filter(|((ok, _), b)| **b && !*ok)
        .count() as f64
        / solved;
    ArmMetrics { success, gain, regression: regr, regression_of_solved: regr_solved }
}

pub fn run_experiment(seed: u64, n_train: usize, n_test: usize) -> ExperimentResults {
    let train_tasks = gen_tasks(seed, n_train);
    let test_tasks = gen_tasks(seed + 5000, n_test);
    let runs: Vec<Run> = train_tasks.iter().map(|t| run_default(t, seed, seed + 7)).collect();
    let memory = compile_memory(runs, seed + 11);

    let mut arms: [Vec<(bool, SolveStats)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (idx, mode) in [Mode::None, Mode::Plain, Mode::WikiTrace].iter().enumerate() {
        arms[idx] = test_tasks
            .iter()
            .map(|t| {
                let mem = if *mode == Mode::None { None } else { Some(&memory) };
                solve(t, mem, *mode, seed)
            })
            .collect();
    }
    let base: Vec<bool> = arms[0].iter().map(|(ok, _)| *ok).collect();

    let none_m = arm_metrics(&arms[0], &base);
    let plain_m = arm_metrics(&arms[1], &base);
    let wt_m = arm_metrics(&arms[2], &base);

    let n_steps = (test_tasks.len() * STEP_TYPES.len()) as f64;
    let n_test_f = test_tasks.len() as f64;
    let wt_stats = &arms[2];
    let descent_rate = wt_stats.iter().map(|(_, s)| s.descents).sum::<u64>() as f64 / n_steps;
    let correction_rate = wt_stats.iter().map(|(_, s)| s.corrections).sum::<u64>() as f64 / n_steps;
    let evidence_reads_per_task =
        wt_stats.iter().map(|(_, s)| s.evidence_reads).sum::<u64>() as f64 / n_test_f;
    let wt_lessons = wt_stats.iter().map(|(_, s)| s.lessons_followed).sum::<u64>() as f64 / n_test_f;
    let plain_lessons =
        arms[1].iter().map(|(_, s)| s.lessons_followed).sum::<u64>() as f64 / n_test_f;

    let mut succ_only = SubsetMetrics::default();
    let mut fail_only = SubsetMetrics::default();
    for (subset, out) in [(true, &mut succ_only), (false, &mut fail_only)] {
        let mem = memory.subset(subset);
        let res: Vec<(bool, SolveStats)> = test_tasks
            .iter()
            .map(|t| solve(t, Some(&mem), Mode::WikiTrace, seed))
            .collect();
        let m = arm_metrics(&res, &base);
        *out = SubsetMetrics {
            success: m.success,
            gain: m.gain,
            regression: m.regression,
            n_insights: mem.insights.len(),
        };
    }

    ExperimentResults {
        none: none_m,
        plain: plain_m,
        plain_lessons_followed_per_task: plain_lessons,
        wikitrace: WikiTraceMetrics {
            arm: wt_m,
            descent_rate,
            correction_rate,
            evidence_reads_per_task,
            lessons_followed_per_task: wt_lessons,
        },
        wikitrace_succ_only: succ_only,
        wikitrace_fail_only: fail_only,
        n_train_runs: n_train,
        n_test_tasks: n_test,
        n_insights: memory.insights.len(),
        n_insights_corrupted: memory.insights.iter().filter(|i| i.corrupted).count(),
        n_insights_from_failed_runs: memory.insights.iter().filter(|i| !i.source_success).count(),
    }
}
