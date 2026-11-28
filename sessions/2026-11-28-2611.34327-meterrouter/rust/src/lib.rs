//! MeterRouter: Budget-Aware Dynamic Routing for Multi-Agent Systems.
//!
//! Rust port of `src/arxiv_lab/agent/meter_router.py` (arXiv:2611.34327),
//! std-only (no external crates, no network). The Python version stays
//! the primary verdict carrier; this port mirrors its mechanism and is
//! verified against it.
//!
//! Prices (Table 6, $/1M tokens), cost (Eq. 1), remaining budget
//! (Eq. 2), utility (Eq. 5), N_max = 5, budgets {1, 4, 5, 50} x 10^-3,
//! Appendix E templates, and the greedy budget-aware router are all
//! ported. The latent-threshold simulation uses a SplitMix64 RNG;
//! Python uses `random.Random`, so sampled accuracies differ while
//! the asserted properties are the same.
//!
//! Deliberate divergences from Python (all documented here):
//! * `erf`: Rust std has no erf, so the Abramowitz-Stegun 7.1.26
//!   approximation is used (|error| <= 1.5e-7). The analytic expected
//!   success therefore differs from Python's `math.erf` version by
//!   ~1e-7; parity comparisons use a 1e-6 tolerance for expected
//!   success, and exact equality for teams and costs.
//! * Skills are rounded to 3 decimals before the analytic integral,
//!   matching Python's cache key (`round(s, 3)`).
//! * RNG: SplitMix64 + Box-Muller for Gaussian draws; Python uses
//!   Mersenne Twister. Query sets and Monte-Carlo streams differ,
//!   so the verify binary asserts properties, not Python's numbers.

/// (name, $/1M input tokens, $/1M output tokens) -- Table 6, exact.
pub const POOL: [(&str, f64, f64); 3] = [
    ("gpt-oss-20b", 0.06, 0.19),
    ("gpt-oss-120b", 0.15, 0.595),
    ("MiniMax-M3", 0.30, 1.20),
];

pub const ROLES: [&str; 5] = ["solver", "analyst", "programmer", "inspector", "tester"];

/// Output tokens per role (simulation choice, at the paper's scale).
pub const ROLE_OUT_TOKENS: [u32; 5] = [1200, 1000, 1500, 700, 800];

/// Role adjustment to base solve probability (simulation choice).
pub const ROLE_ADJ: [f64; 5] = [0.04, 0.02, 0.00, -0.12, -0.06];

/// Base solve probability by tier at medium difficulty (simulation).
pub const TIER_SKILL: [f64; 3] = [0.38, 0.53, 0.65];

pub const HINT_BONUS: f64 = 0.03;
pub const NOISE_SD: f64 = 0.08;
pub const BASE_IN_TOKENS: u32 = 2000;
pub const IN_TOKENS_PER_PARENT: u32 = 500;
pub const N_MAX: usize = 5;
pub const BUDGETS: [f64; 4] = [1e-3, 4e-3, 5e-3, 50e-3];
pub const GAMMA: f64 = 0.5;
pub const MIN_GAIN: f64 = 0.01;

pub fn role_idx(role: &str) -> usize {
    ROLES.iter().position(|r| *r == role).expect("unknown role")
}

pub fn difficulty_shift(difficulty: &str) -> f64 {
    match difficulty {
        "easy" => 0.13,
        "medium" => 0.0,
        "hard" => -0.13,
        _ => panic!("unknown difficulty"),
    }
}

/// One agent in a built MAS: role index, tier index, parent indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Agent {
    pub role: usize,
    pub tier: usize,
    pub parents: [usize; 4],
    pub n_parents: usize,
}

impl Agent {
    pub fn new(role: usize, tier: usize, parents: &[usize]) -> Agent {
        let mut p = [0usize; 4];
        for (i, v) in parents.iter().enumerate() {
            p[i] = *v;
        }
        Agent { role, tier, parents: p, n_parents: parents.len() }
    }

    pub fn parent_vec(&self) -> Vec<usize> {
        self.parents[..self.n_parents].to_vec()
    }
}

/// A synthetic query: difficulty band and latent threshold.
#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub difficulty: &'static str,
    pub threshold: f64,
}

/// Eq. (1): C_i = c_in * T_in + c_out * T_out, in USD.
pub fn call_cost(tier: usize, n_parents: usize, role: usize) -> f64 {
    let (_, c_in, c_out) = POOL[tier];
    let t_in = (BASE_IN_TOKENS + IN_TOKENS_PER_PARENT * n_parents as u32) as f64;
    let t_out = ROLE_OUT_TOKENS[role] as f64;
    (c_in * t_in + c_out * t_out) / 1e6
}

/// Eq. (2): b_i = B - sum_{j<i} C_j.
pub fn remaining_budget(budget: f64, costs: &[f64]) -> f64 {
    budget - costs.iter().sum::<f64>()
}

/// Eq. (5): U = S - gamma * C / B.
pub fn utility(correct: bool, cost: f64, budget: f64, gamma: f64) -> f64 {
    (if correct { 1.0 } else { 0.0 }) - gamma * cost / budget
}

/// Parents per node for the Appendix E templates (n = 1..5).
pub fn template_graph(name: &str, n: usize) -> Vec<Vec<usize>> {
    if n == 1 {
        return vec![Vec::new()];
    }
    match name {
        "single" => vec![Vec::new(); n],
        "chain" => {
            let mut out = vec![Vec::new()];
            for i in 1..n {
                out.push(vec![i - 1]);
            }
            out
        }
        "fan-in" => {
            let mut out = vec![Vec::new(); n - 1];
            out.push((0..n - 1).collect());
            out
        }
        "dense" => (0..n).map(|i| (0..i).collect()).collect(),
        "diamond" if n == 4 => {
            vec![vec![], vec![0], vec![0], vec![1, 2]]
        }
        "pruned" => {
            // Deterministic pseudo-random pruning, seeded by n.
            let mut rng = SplitMix64::new(1000 + n as u64);
            let mut out = vec![Vec::new()];
            for i in 1..n {
                let k = 1 + (rng.next_u64() as usize % i);
                let mut pool: Vec<usize> = (0..i).collect();
                let mut chosen = Vec::new();
                for _ in 0..k {
                    let j = rng.next_u64() as usize % pool.len();
                    chosen.push(pool.remove(j));
                }
                chosen.sort_unstable();
                out.push(chosen);
            }
            out
        }
        _ => panic!("unknown template {name} for n={n}"),
    }
}

/// Abramowitz-Stegun 7.1.26 approximation of erf (|error| <= 1.5e-7).
pub fn erf_approx(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
        - 0.284496736)
        * t
        + 0.254829592)
        * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

pub fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn effective_skill(role: usize, tier: usize, difficulty: &str,
                   n_distinct_predecessors: usize) -> f64 {
    let skill = TIER_SKILL[tier] + ROLE_ADJ[role]
        + difficulty_shift(difficulty)
        + HINT_BONUS * n_distinct_predecessors as f64;
    skill.clamp(0.01, 0.95)
}

fn team_skills(team: &[Agent], difficulty: &str) -> Vec<f64> {
    let mut skills = Vec::new();
    let mut seen_roles = [false; 5];
    let mut n_seen = 0usize;
    for agent in team {
        skills.push(effective_skill(agent.role, agent.tier, difficulty, n_seen));
        if !seen_roles[agent.role] {
            seen_roles[agent.role] = true;
            n_seen += 1;
        }
    }
    skills
}

/// Analytic P(team solves) under the latent-threshold model.
/// Skills are rounded to 3 decimals first, matching Python's cache key.
pub fn expected_team_success(team: &[Agent], difficulty: &str) -> f64 {
    if team.is_empty() {
        return 0.0;
    }
    let skills: Vec<f64> = team_skills(team, difficulty)
        .iter()
        .map(|s| (s * 1000.0).round() / 1000.0)
        .collect();
    let steps = 100usize;
    let mut total = 0.0;
    for k in 0..steps {
        let theta = (k as f64 + 0.5) / steps as f64;
        let mut p_fail = 1.0;
        for s in &skills {
            p_fail *= phi((theta - *s) / NOISE_SD);
        }
        total += 1.0 - p_fail;
    }
    total / steps as f64
}

pub fn team_cost(team: &[Agent]) -> f64 {
    team.iter()
        .map(|a| call_cost(a.tier, a.n_parents, a.role))
        .sum()
}

/// SplitMix64 RNG (std-only, deterministic).
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        // 53-bit uniform in [0, 1).
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller (one spare cached).
    pub fn next_gauss(&mut self, spare: &mut Option<f64>) -> f64 {
        if let Some(v) = spare.take() {
            return v;
        }
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        *spare = Some(r * t.sin());
        r * t.cos()
    }
}

/// One seeded live run of a team on a query (latent-threshold).
pub fn simulate_team(team: &[Agent], query: &Query, rng: &mut SplitMix64,
                     spare: &mut Option<f64>) -> bool {
    let skills = team_skills(team, query.difficulty);
    for s in skills {
        if s + rng.next_gauss(spare) * NOISE_SD > query.threshold {
            return true;
        }
    }
    false
}

/// Greedy budget-aware router (the mechanism). `tier_draws` supplies
/// the exogenous tier for each step in random-tier mode (one draw per
/// step, before any role/parents choice); pass an empty slice for the
/// budget-aware mode. Returns (team, stop reason).
pub fn greedy_router(difficulty: &str, budget: f64, random_tier: bool,
                     tier_draws: &[usize]) -> (Vec<Agent>, &'static str) {
    let mut team: Vec<Agent> = Vec::new();
    let mut spent = 0.0;
    let mut used_roles = [false; 5];
    let mut draw_i = 0usize;
    while team.len() < N_MAX {
        let remaining = budget - spent;
        let base = expected_team_success(&team, difficulty);
        let step_tier = if random_tier {
            let t = tier_draws.get(draw_i).copied().unwrap_or(0);
            draw_i += 1;
            Some(t)
        } else {
            None
        };
        // best: (score, gain, cost, agent)
        let mut best: Option<(f64, f64, f64, Agent)> = None;
        for role in 0..ROLES.len() {
            if used_roles[role] {
                continue;
            }
            let mut parent_options: Vec<Vec<usize>> = vec![Vec::new()];
            if !team.is_empty() {
                parent_options.push(vec![team.len() - 1]);
                parent_options.push((0..team.len()).collect());
            }
            for parents in &parent_options {
                let tiers: Vec<usize> = match step_tier {
                    Some(t) => vec![t],
                    None => (0..POOL.len()).collect(),
                };
                for tier in tiers {
                    let cost = call_cost(tier, parents.len(), role);
                    if cost > remaining {
                        continue;
                    }
                    let mut cand = team.clone();
                    cand.push(Agent::new(role, tier, parents));
                    let gain = expected_team_success(&cand, difficulty) - base;
                    if gain < MIN_GAIN {
                        continue;
                    }
                    // Eq. (5) after the first agent; the first agent
                    // is mandatory (best affordable gain).
                    let score = if team.is_empty() {
                        gain
                    } else {
                        let d_utility = gain - GAMMA * cost / budget;
                        if d_utility <= 0.0 {
                            continue;
                        }
                        d_utility
                    };
                    if best.as_ref().map_or(true, |b| score > b.0) {
                        best = Some((score, gain, cost,
                                     Agent::new(role, tier, parents)));
                    }
                }
            }
        }
        match best {
            None => {
                let cheapest = (0..POOL.len())
                    .flat_map(|t| (0..ROLES.len()).map(move |r| call_cost(t, 0, r)))
                    .fold(f64::INFINITY, f64::min);
                let cheapest = match step_tier {
                    Some(t) => (0..ROLES.len())
                        .map(|r| call_cost(t, 0, r))
                        .fold(f64::INFINITY, f64::min),
                    None => cheapest,
                };
                if remaining < cheapest {
                    return (team, "budget");
                }
                return (team, "learned");
            }
            Some((_, _, cost, agent)) => {
                spent += cost;
                used_roles[agent.role] = true;
                team.push(agent);
            }
        }
    }
    (team, "cap")
}

/// Strongest MAS baseline: all 5 roles, high tier, dense edges.
pub fn strongest_mas() -> Vec<Agent> {
    (0..ROLES.len())
        .map(|i| Agent::new(i, 2, &(0..i).collect::<Vec<usize>>()))
        .collect()
}

pub fn single_agent(tier: usize) -> Vec<Agent> {
    vec![Agent::new(0, tier, &[])]
}

/// Fixed 3-agent mid-tier chain, ignoring the budget (baseline).
pub fn fixed_router() -> Vec<Agent> {
    let parents = template_graph("chain", 3);
    [0usize, 1, 3]
        .iter()
        .enumerate()
        .map(|(i, role)| Agent::new(*role, 1, &parents[i]))
        .collect()
}

/// Synthetic query set: difficulty mix easy/medium/hard 25/50/25,
/// latent threshold uniform within each band.
pub fn make_queries(n: usize, seed: u64) -> Vec<Query> {
    let mut rng = SplitMix64::new(seed);
    let mut queries = Vec::new();
    for _ in 0..n {
        let r = rng.next_f64();
        let difficulty = if r < 0.25 {
            "easy"
        } else if r < 0.75 {
            "medium"
        } else {
            "hard"
        };
        queries.push(Query { difficulty, threshold: rng.next_f64() });
    }
    queries
}

#[derive(Clone, Debug, Default)]
pub struct RunResult {
    pub method: &'static str,
    pub budget: f64,
    pub n: usize,
    pub correct: usize,
    pub total_cost: f64,
    pub overruns: usize,
    pub total_agents: usize,
    pub stops: Vec<(&'static str, usize)>,
}

impl RunResult {
    pub fn accuracy(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.correct as f64 / self.n as f64 }
    }

    pub fn mean_cost(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.total_cost / self.n as f64 }
    }

    pub fn overrun_rate(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.overruns as f64 / self.n as f64 }
    }

    pub fn mean_agents(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.total_agents as f64 / self.n as f64 }
    }

    fn note_stop(&mut self, stop: &'static str) {
        if let Some(e) = self.stops.iter_mut().find(|(s, _)| *s == stop) {
            e.1 += 1;
        } else {
            self.stops.push((stop, 1));
        }
    }
}

fn run_method(method: &'static str, queries: &[Query], budget: f64,
              seed: u64) -> RunResult {
    let mut result = RunResult { method, budget, ..Default::default() };
    for (qi, query) in queries.iter().enumerate() {
        let mut rng = SplitMix64::new(seed.wrapping_mul(1_000_003).wrapping_add(qi as u64));
        let mut tier_rng = SplitMix64::new(seed.wrapping_mul(7919).wrapping_add(qi as u64));
        let mut spare: Option<f64> = None;
        let (team, stop) = match method {
            "meterrouter" => greedy_router(query.difficulty, budget, false, &[]),
            "random_tier" => {
                let draws: Vec<usize> = (0..N_MAX)
                    .map(|_| tier_rng.next_u64() as usize % POOL.len())
                    .collect();
                greedy_router(query.difficulty, budget, true, &draws)
            }
            "strongest" => (strongest_mas(), "fixed"),
            "single_high" => (single_agent(2), "fixed"),
            "single_low" => (single_agent(0), "fixed"),
            "fixed" => (fixed_router(), "fixed"),
            _ => panic!("unknown method"),
        };
        let cost = team_cost(&team);
        result.n += 1;
        result.total_cost += cost;
        result.total_agents += team.len();
        result.note_stop(stop);
        if cost > budget + 1e-12 {
            result.overruns += 1;
        }
        if !team.is_empty() && simulate_team(&team, query, &mut rng, &mut spare) {
            result.correct += 1;
        }
    }
    result
}

/// Run the router and all baselines on the same query set.
/// Returns {budget: {method: RunResult}} in BUDGETS order.
pub fn run_experiment(n_queries: usize, seed: u64) -> Vec<Vec<RunResult>> {
    let queries = make_queries(n_queries, seed);
    let methods = ["meterrouter", "random_tier", "strongest",
                   "single_high", "single_low", "fixed"];
    BUDGETS
        .iter()
        .map(|b| {
            methods
                .iter()
                .map(|m| run_method(m, &queries, *b, seed))
                .collect()
        })
        .collect()
}
