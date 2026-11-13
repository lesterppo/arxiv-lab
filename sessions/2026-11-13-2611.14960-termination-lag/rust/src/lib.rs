//! Termination lag in coding agents (arXiv:2611.14960) — Rust port of the
//! Python mechanism in `../termination_lag.py`.
//!
//! Line-for-line port of the pure logic, no external crates. The Python
//! version stays the primary verdict carrier; this port never replaces it.
//!
//! Deliberate divergences from Python: none in the mechanism. `f64`
//! conversion of a `u64` uses the same round-to-nearest-even rule in both
//! languages, so `SplitMix64::uniform` streams match exactly, and the
//! Poisson sampler calls the platform `exp` in both languages, so its
//! streams match too (verified by the parity sweep).

pub const MASK64: u64 = u64::MAX;

pub const P_FIRST_EASY: f64 = 0.82;
pub const P_FIRST_MEDIUM: f64 = 0.60;
pub const P_FIRST_HARD: f64 = 0.38;
pub const P_REFIX: f64 = 0.55;
pub const P_UNDO: f64 = 0.15;
pub const MAX_EDITS: u32 = 6;
pub const MAX_STEPS: u32 = 20;

// ---------------------------------------------------------------- RNG
/// SplitMix64. `next_u64` returns the next 64-bit draw.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed & MASK64 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
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
    pub fn randint(&mut self, lo: u64, hi: u64) -> u64 {
        lo + (self.next_u64() % (hi - lo))
    }

    /// Knuth's algorithm; fine for the small lambdas used here.
    pub fn poisson(&mut self, lam: f64) -> u64 {
        let limit = (-lam).exp();
        let mut k: u64 = 0;
        let mut prod = 1.0f64;
        loop {
            prod *= self.uniform();
            if prod <= limit {
                return k;
            }
            k += 1;
        }
    }
}

// ---------------------------------------------------------------- tasks
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

impl Difficulty {
    pub fn p_first(self) -> f64 {
        match self {
            Difficulty::Easy => P_FIRST_EASY,
            Difficulty::Medium => P_FIRST_MEDIUM,
            Difficulty::Hard => P_FIRST_HARD,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Difficulty::Easy => "easy",
            Difficulty::Medium => "medium",
            Difficulty::Hard => "hard",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Task {
    pub idx: u64,
    pub difficulty: Difficulty,
}

pub fn gen_tasks(n: usize, seed: u64) -> Vec<Task> {
    let mut rng = Rng::new(seed);
    let diffs = [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard];
    (0..n as u64)
        .map(|i| Task {
            idx: i,
            difficulty: diffs[rng.randint(0, 3) as usize],
        })
        .collect()
}

// ---------------------------------------------------------------- verifier
/// The test suite: deterministic, passes iff the current patch is correct.
pub fn verify(patch_correct: bool) -> bool {
    patch_correct
}

// ---------------------------------------------------------------- agent run
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Continue,
    StopOnPass,
    FixedBudget,
}

impl Policy {
    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Continue => "continue",
            Policy::StopOnPass => "stop_on_pass",
            Policy::FixedBudget => "fixed_budget",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RunResult {
    pub task_idx: u64,
    pub difficulty: Difficulty,
    pub policy: Policy,
    pub steps: u32,
    pub first_correct_step: Option<u32>,
    pub first_edit_solved: bool,
    pub final_correct: bool,
    pub lag: u32,
    pub post_share: f64,
    pub undo_to_incorrect: bool,
    pub n_verify: u32,
    pub n_explore: u32,
    pub n_reedit: u32,
    pub n_undos: u32,
    pub solved: bool,
}

pub fn run_task(
    task: Task,
    policy: Policy,
    rng: &mut Rng,
    mean_extra: f64,
    p_undo: f64,
    fixed_budget: u32,
    mean_pre_explore: f64,
) -> RunResult {
    let mut res = RunResult {
        task_idx: task.idx,
        difficulty: task.difficulty,
        policy,
        steps: 0,
        first_correct_step: None,
        first_edit_solved: false,
        final_correct: false,
        lag: 0,
        post_share: 0.0,
        undo_to_incorrect: false,
        n_verify: 0,
        n_explore: 0,
        n_reedit: 0,
        n_undos: 0,
        solved: false,
    };
    let mut patch_correct = false;
    let mut step: u32 = 0;
    let mut edit_no: u32 = 0;

    // Local helpers as closures are awkward with &mut rng borrows, so the
    // edit/verify/explore steps are inlined below in the same order as the
    // Python closures (do_edit / do_verify / do_explore / do_reedit).
    macro_rules! do_edit {
        () => {{
            step += 1;
            edit_no += 1;
            if patch_correct {
                if rng.bernoulli(p_undo) {
                    patch_correct = false;
                    res.n_undos += 1;
                }
            } else if edit_no == 1 {
                if rng.bernoulli(task.difficulty.p_first()) {
                    patch_correct = true;
                    res.first_correct_step = Some(step);
                    res.first_edit_solved = true;
                }
            } else if rng.bernoulli(P_REFIX) {
                patch_correct = true;
                if res.first_correct_step.is_none() {
                    res.first_correct_step = Some(step);
                }
            }
            patch_correct
        }};
    }
    macro_rules! do_verify {
        () => {{
            step += 1;
            res.n_verify += 1;
            verify(patch_correct)
        }};
    }
    macro_rules! do_explore {
        () => {{
            step += 1;
            res.n_explore += 1;
        }};
    }

    // --- pre-solution phase: explore the repo, then edit until the first
    // correct patch or give up.
    let pre = 1 + rng.poisson(mean_pre_explore);
    for _ in 0..pre {
        if step >= MAX_STEPS {
            break;
        }
        step += 1; // pre-solution exploration (not counted in post shares)
    }
    let mut edits: u32 = 0;
    while edits < MAX_EDITS && step < MAX_STEPS {
        do_edit!();
        edits += 1;
        if patch_correct {
            break;
        }
    }
    if !patch_correct {
        res.steps = step;
        res.final_correct = false;
        res.solved = false;
        return res;
    }
    res.solved = true;
    let snap_v = res.n_verify;
    let snap_e = res.n_explore;
    let mut post_edits: u32 = 0;

    // --- post-solution phase, by policy.
    match policy {
        Policy::StopOnPass => {
            // One confirmation test, then stop.
            assert!(do_verify!());
        }
        Policy::FixedBudget => {
            while step < fixed_budget && step < MAX_STEPS {
                let r = rng.uniform();
                if r < 0.55 {
                    do_verify!();
                } else if r < 0.80 {
                    do_explore!();
                } else {
                    post_edits += 1;
                    do_edit!();
                }
            }
        }
        Policy::Continue => {
            let extra = 1 + rng.poisson(mean_extra);
            for _ in 0..extra {
                if step >= MAX_STEPS {
                    break;
                }
                let r = rng.uniform();
                if r < 0.55 {
                    do_verify!();
                } else if r < 0.80 {
                    do_explore!();
                } else {
                    post_edits += 1;
                    do_edit!();
                }
            }
        }
    }

    res.steps = step;
    res.final_correct = patch_correct;
    res.lag = step - res.first_correct_step.unwrap();
    res.post_share = if step > 0 {
        res.lag as f64 / step as f64
    } else {
        0.0
    };
    res.undo_to_incorrect = !patch_correct;
    res.n_verify -= snap_v;
    res.n_explore -= snap_e;
    res.n_reedit = post_edits;
    res
}

// ---------------------------------------------------------------- experiment
pub fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

#[derive(Clone, Debug)]
pub struct ExperimentMetrics {
    pub policy: Policy,
    pub n_tasks: usize,
    pub n_solved: usize,
    pub success_rate: f64,
    pub first_edit_solve_share: f64,
    pub mean_lag: f64,
    pub median_lag: f64,
    pub median_post_share: f64,
    pub mean_post_share: f64,
    pub final_correct_rate: f64,
    pub undo_to_incorrect_rate: f64,
    pub mean_undos: f64,
    pub mean_total_steps: f64,
    pub post_verify_share: f64,
    pub post_explore_share: f64,
    pub post_reedit_share: f64,
}

pub fn run_experiment(
    policy: Policy,
    tasks: &[Task],
    seed: u64,
    mean_extra: f64,
    p_undo: f64,
    fixed_budget: u32,
    mean_pre_explore: f64,
) -> ExperimentMetrics {
    let mut runs: Vec<RunResult> = Vec::with_capacity(tasks.len());
    for t in tasks {
        let mut rng = Rng::new(seed ^ (t.idx.wrapping_mul(0x9E3779B97F4A7C15)));
        runs.push(run_task(
            *t,
            policy,
            &mut rng,
            mean_extra,
            p_undo,
            fixed_budget,
            mean_pre_explore,
        ));
    }
    let solved: Vec<&RunResult> = runs.iter().filter(|r| r.solved).collect();
    let n = runs.len();
    let n_solved = solved.len();
    let first_edit_solved = solved.iter().filter(|r| r.first_edit_solved).count();
    let total_lag: u64 = solved.iter().map(|r| r.lag as u64).sum();
    ExperimentMetrics {
        policy,
        n_tasks: n,
        n_solved,
        success_rate: n_solved as f64 / n as f64,
        first_edit_solve_share: if n_solved > 0 {
            first_edit_solved as f64 / n_solved as f64
        } else {
            0.0
        },
        mean_lag: if n_solved > 0 {
            total_lag as f64 / n_solved as f64
        } else {
            0.0
        },
        median_lag: median(solved.iter().map(|r| r.lag as f64).collect()),
        median_post_share: median(solved.iter().map(|r| r.post_share).collect()),
        mean_post_share: if n_solved > 0 {
            solved.iter().map(|r| r.post_share).sum::<f64>() / n_solved as f64
        } else {
            0.0
        },
        final_correct_rate: if n_solved > 0 {
            solved.iter().filter(|r| r.final_correct).count() as f64 / n_solved as f64
        } else {
            0.0
        },
        undo_to_incorrect_rate: if n_solved > 0 {
            solved.iter().filter(|r| r.undo_to_incorrect).count() as f64 / n_solved as f64
        } else {
            0.0
        },
        mean_undos: if n_solved > 0 {
            solved.iter().map(|r| r.n_undos as u64).sum::<u64>() as f64 / n_solved as f64
        } else {
            0.0
        },
        mean_total_steps: runs.iter().map(|r| r.steps as u64).sum::<u64>() as f64 / n as f64,
        post_verify_share: if total_lag > 0 {
            solved.iter().map(|r| r.n_verify as u64).sum::<u64>() as f64 / total_lag as f64
        } else {
            0.0
        },
        post_explore_share: if total_lag > 0 {
            solved.iter().map(|r| r.n_explore as u64).sum::<u64>() as f64 / total_lag as f64
        } else {
            0.0
        },
        post_reedit_share: if total_lag > 0 {
            solved.iter().map(|r| r.n_reedit as u64).sum::<u64>() as f64 / total_lag as f64
        } else {
            0.0
        },
    }
}

pub fn compare_policies(
    tasks: &[Task],
    seed: u64,
    mean_extra: f64,
    p_undo: f64,
    fixed_budget: u32,
    mean_pre_explore: f64,
) -> [ExperimentMetrics; 3] {
    [
        run_experiment(Policy::Continue, tasks, seed, mean_extra, p_undo, fixed_budget, mean_pre_explore),
        run_experiment(Policy::StopOnPass, tasks, seed, mean_extra, p_undo, fixed_budget, mean_pre_explore),
        run_experiment(Policy::FixedBudget, tasks, seed, mean_extra, p_undo, fixed_budget, mean_pre_explore),
    ]
}

/// Default parameters, matching the Python test's defaults.
pub fn default_experiment(policy: Policy, tasks: &[Task], seed: u64) -> ExperimentMetrics {
    run_experiment(policy, tasks, seed, 3.0, P_UNDO, 10, 1.0)
}
