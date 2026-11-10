//! Executable finite-state control for LLM agents on procedure
//! (arXiv:2611.10884) — Rust port of `fsm_control.py`.
//!
//! Deliberate-divergence policy: NONE intended. This is a line-for-line
//! mirror of the Python module, including the exact SplitMix64 stream and
//! the exact order in which the RNG is consumed, so that identical seeds
//! produce identical batches in both languages (see `tests/parity_tests.rs`).
//! The one place Python's semantics need care: `Account` balances in
//! `make_accounts` use Python's floor-modulo on a negated u64
//! (`-u % 500 - 1`), reproduced here as `((500 - u % 500) % 500) - 1`.
//! Balance is cosmetic metadata — no decision logic reads it.

pub const START: &str = "START";
pub const IDENTITY_VERIFIED: &str = "IDENTITY_VERIFIED";
pub const BALANCE_CHECKED: &str = "BALANCE_CHECKED";
pub const ELIGIBILITY_CHECKED: &str = "ELIGIBILITY_CHECKED";
pub const FEE_WAIVED: &str = "FEE_WAIVED";
pub const CLOSED: &str = "CLOSED";
pub const ESCALATED: &str = "ESCALATED";

pub const TOOLS: [&str; 6] = [
    "verify_identity",
    "check_balance",
    "check_eligibility",
    "waive_fee",
    "close_account",
    "escalate",
];

/// (state, tool) -> next state. Anything not in the table is invalid.
pub fn transition(state: &str, tool: &str) -> Option<&'static str> {
    match (state, tool) {
        (START, "verify_identity") => Some(IDENTITY_VERIFIED),
        (IDENTITY_VERIFIED, "check_balance") => Some(BALANCE_CHECKED),
        (BALANCE_CHECKED, "check_eligibility") => Some(ELIGIBILITY_CHECKED),
        (ELIGIBILITY_CHECKED, "waive_fee") => Some(FEE_WAIVED),
        (ELIGIBILITY_CHECKED, "close_account") => Some(CLOSED),
        (FEE_WAIVED, "close_account") => Some(CLOSED),
        (ELIGIBILITY_CHECKED, "escalate") => Some(ESCALATED),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Account {
    pub id: usize,
    pub balance: i64,
    pub eligible: bool,
    pub needs_escalation: bool,
}

impl Account {
    pub fn new(id: usize, balance: i64, eligible: bool, needs_escalation: bool) -> Self {
        Account { id, balance, eligible, needs_escalation }
    }

    pub fn correct_terminal(&self) -> &'static str {
        if self.needs_escalation { "escalate" } else { "close" }
    }
}

pub fn correct_next_tool(state: &str, account: &Account) -> Option<&'static str> {
    match state {
        START => Some("verify_identity"),
        IDENTITY_VERIFIED => Some("check_balance"),
        BALANCE_CHECKED => Some("check_eligibility"),
        ELIGIBILITY_CHECKED => {
            if account.needs_escalation {
                Some("escalate")
            } else if account.eligible {
                Some("waive_fee")
            } else {
                Some("close_account")
            }
        }
        FEE_WAIVED => Some("close_account"),
        _ => None, // terminal
    }
}

pub fn branch_is_valid(state: &str, tool: &str, account: &Account) -> bool {
    match tool {
        "waive_fee" => account.eligible,
        "escalate" => account.needs_escalation,
        "close_account" if state == ELIGIBILITY_CHECKED => {
            !account.needs_escalation && !account.eligible
        }
        _ => true,
    }
}

pub fn is_terminal(state: &str) -> bool {
    state == CLOSED || state == ESCALATED
}

/// Shared runtime state: accepted trace, rejections, violation count.
#[derive(Clone, Debug, Default)]
pub struct RuntimeState {
    pub state: String,
    pub trace: Vec<(String, String, String)>,
    pub rejected: Vec<(String, String, String)>,
    pub violations: u64,
    pub terminal_decision: Option<String>,
}

impl RuntimeState {
    pub fn new() -> Self {
        RuntimeState { state: START.to_string(), ..Default::default() }
    }
}

pub trait Runtime {
    fn state(&self) -> &RuntimeState;
    fn state_mut(&mut self) -> &mut RuntimeState;
    fn account(&self) -> &Account;
    /// Attempt a tool call. Returns (accepted, message).
    fn call(&mut self, tool: &str) -> (bool, String);
    fn end_turn(&self) -> (bool, Option<String>) {
        let st = self.state();
        if is_terminal(&st.state) {
            (true, st.terminal_decision.clone())
        } else {
            (false, None)
        }
    }
}

/// Executable FSM: state lives in the runtime; enforcement at tool calls,
/// transitions, and end of turn.
pub struct FsmRuntime {
    pub account: Account,
    pub st: RuntimeState,
}

impl FsmRuntime {
    pub fn new(account: Account) -> Self {
        FsmRuntime { account, st: RuntimeState::new() }
    }
}

impl Runtime for FsmRuntime {
    fn state(&self) -> &RuntimeState {
        &self.st
    }
    fn state_mut(&mut self) -> &mut RuntimeState {
        &mut self.st
    }
    fn account(&self) -> &Account {
        &self.account
    }
    fn call(&mut self, tool: &str) -> (bool, String) {
        let cur = self.st.state.clone();
        match transition(&cur, tool) {
            None => {
                self.st.rejected.push((
                    tool.to_string(),
                    cur.clone(),
                    "no transition".to_string(),
                ));
                (false, format!("invalid call: {} not allowed in {}", tool, cur))
            }
            Some(nxt) => {
                if !branch_is_valid(&cur, tool, &self.account) {
                    self.st.rejected.push((
                        tool.to_string(),
                        cur.clone(),
                        "branch condition".to_string(),
                    ));
                    return (
                        false,
                        format!("invalid call: {} branch condition fails in {}", tool, cur),
                    );
                }
                self.st.trace.push((tool.to_string(), cur.clone(), nxt.to_string()));
                self.st.state = nxt.to_string();
                if is_terminal(nxt) {
                    self.st.terminal_decision = Some(if tool == "escalate" {
                        "escalate".to_string()
                    } else {
                        "close".to_string()
                    });
                }
                (true, format!("ok: {} --{}--> {}", cur, tool, nxt))
            }
        }
    }
}

/// No enforcement: every call executes; forbidden calls count as violations.
pub struct PlainRuntime {
    pub account: Account,
    pub st: RuntimeState,
}

impl PlainRuntime {
    pub fn new(account: Account) -> Self {
        PlainRuntime { account, st: RuntimeState::new() }
    }
}

impl Runtime for PlainRuntime {
    fn state(&self) -> &RuntimeState {
        &self.st
    }
    fn state_mut(&mut self) -> &mut RuntimeState {
        &mut self.st
    }
    fn account(&self) -> &Account {
        &self.account
    }
    fn call(&mut self, tool: &str) -> (bool, String) {
        let cur = self.st.state.clone();
        let nxt = transition(&cur, tool);
        let valid = match nxt {
            Some(_) => branch_is_valid(&cur, tool, &self.account),
            None => false,
        };
        if !valid {
            self.st.violations += 1;
        }
        match nxt {
            Some(n) => {
                self.st.trace.push((tool.to_string(), cur.clone(), n.to_string()));
                self.st.state = n.to_string();
                if is_terminal(n) {
                    self.st.terminal_decision = Some(if tool == "escalate" {
                        "escalate".to_string()
                    } else {
                        "close".to_string()
                    });
                }
            }
            None => {
                self.st.trace.push((tool.to_string(), cur.clone(), cur.clone()));
            }
        }
        (true, "executed (no enforcement)".to_string())
    }
}

// ------------------------------------------------------------------- RNG

/// Deterministic 64-bit RNG, identical stream to the Python SplitMix64.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn random(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    pub fn choice<'a>(&mut self, seq: &'a [&'a str]) -> &'a str {
        seq[(self.next_u64() % seq.len() as u64) as usize]
    }
}

// ------------------------------------------------------------- drift agent

pub const DEVIATIONS: [&str; 3] = ["skip_to_terminal", "wrong_branch", "skip_one_step"];

pub fn deviation_tool(state: &str, account: &Account, kind: &str, rng: &mut SplitMix64) -> String {
    match kind {
        "skip_to_terminal" => rng.choice(&["close_account", "escalate"]).to_string(),
        "wrong_branch" => {
            if state == ELIGIBILITY_CHECKED {
                if account.needs_escalation {
                    "close_account".to_string()
                } else if account.eligible {
                    "close_account".to_string()
                } else {
                    rng.choice(&["waive_fee", "escalate"]).to_string()
                }
            } else {
                rng.choice(&["waive_fee", "close_account", "escalate"]).to_string()
            }
        }
        _ => {
            // skip_one_step: call a tool from a later state (invalid order).
            let correct = correct_next_tool(state, account).unwrap_or("close_account");
            match correct {
                "verify_identity" => "check_balance".to_string(),
                "check_balance" => "check_eligibility".to_string(),
                "check_eligibility" => "waive_fee".to_string(),
                _ => "close_account".to_string(),
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub account: usize,
    pub complete: bool,
    pub decision: Option<String>,
    pub correct_decision: bool,
    pub by_procedure: bool,
    pub correct_and_by_procedure: bool,
    pub violations: u64,
    pub invalid_attempts: u64,
    pub retries_used: u64,
    pub steps: usize,
}

#[derive(Clone, Debug)]
pub struct DriftAgent {
    pub rng: SplitMix64,
    pub p0: f64,
    pub drift_rate: f64,
    pub drift_scale: f64,
    pub max_retries: u32,
    pub steps_done: u64,
    pub context_tokens: u64,
}

impl DriftAgent {
    pub fn new(rng: SplitMix64, p0: f64, drift_rate: f64, drift_scale: f64, max_retries: u32) -> Self {
        DriftAgent {
            rng,
            p0,
            drift_rate,
            drift_scale,
            max_retries,
            steps_done: 0,
            context_tokens: 0,
        }
    }

    pub fn p_deviate(&self) -> f64 {
        let frac = (self.steps_done as f64 / self.drift_scale).min(1.0);
        (self.p0 + self.drift_rate * frac).min(0.95)
    }

    /// Re-issue the SOP. It only ADDS context — it does not remove drift.
    pub fn restate(&mut self, tokens: u64) {
        self.context_tokens += tokens;
        self.steps_done += tokens / 10;
    }

    pub fn act<R: Runtime>(&mut self, runtime: &mut R) -> Outcome {
        let account = runtime.account().clone();
        let mut invalid_attempts: u64 = 0;
        let mut retries_used: u64 = 0;
        let mut guard: u32 = 0;
        loop {
            guard += 1;
            if guard > 200 {
                break; // runaway guard: account abandoned, not completed
            }
            let correct = match correct_next_tool(&runtime.state().state, &account) {
                Some(t) => t.to_string(),
                None => break, // terminal
            };
            self.steps_done += 1;
            let mut tool = correct.clone();
            if self.rng.random() < self.p_deviate() {
                let kind = self.rng.choice(&DEVIATIONS).to_string();
                tool = deviation_tool(&runtime.state().state, &account, &kind, &mut self.rng);
            }
            let (accepted, _) = runtime.call(&tool);
            if !accepted {
                invalid_attempts += 1;
                let mut ok = false;
                for _ in 0..self.max_retries {
                    retries_used += 1;
                    self.steps_done += 1;
                    let retry_tool = match correct_next_tool(&runtime.state().state, &account) {
                        Some(t) => t.to_string(),
                        None => {
                            ok = true;
                            break;
                        }
                    };
                    let mut rt = retry_tool;
                    if self.rng.random() < self.p_deviate() {
                        let kind = self.rng.choice(&DEVIATIONS).to_string();
                        rt = deviation_tool(&runtime.state().state, &account, &kind, &mut self.rng);
                    }
                    let (acc, _) = runtime.call(&rt);
                    if acc {
                        ok = true;
                        break;
                    }
                    invalid_attempts += 1;
                }
                if !ok {
                    break; // retry budget exhausted: account abandoned
                }
                continue;
            }
        }
        let (complete, decision) = runtime.end_turn();
        let st = runtime.state();
        let by_procedure = st.violations == 0 && complete;
        let correct_decision = decision.as_deref() == Some(account.correct_terminal());
        Outcome {
            account: account.id,
            complete,
            decision,
            correct_decision,
            by_procedure,
            correct_and_by_procedure: by_procedure && correct_decision,
            violations: st.violations,
            invalid_attempts,
            retries_used,
            steps: st.trace.len(),
        }
    }
}

pub fn make_accounts(n: usize, seed: u64) -> Vec<Account> {
    let mut rng = SplitMix64::new(seed);
    let mut accounts = Vec::with_capacity(n);
    for i in 0..n {
        let r = rng.random();
        if r < 0.20 {
            // Python: balance = -rng.next_u64() % 500 - 1 (floor modulo).
            let u = rng.next_u64();
            let m = u % 500;
            let balance = (((500 - m) % 500) as i64) - 1;
            accounts.push(Account::new(i, balance, false, true));
        } else if r < 0.45 {
            let balance = (rng.next_u64() % 5000 + 100) as i64;
            accounts.push(Account::new(i, balance, true, false));
        } else {
            let balance = (rng.next_u64() % 5000 + 100) as i64;
            accounts.push(Account::new(i, balance, false, false));
        }
    }
    accounts
}

#[derive(Clone, Debug)]
pub struct Summary {
    pub arm: String,
    pub n: usize,
    pub correct_and_by_procedure: usize,
    pub rate: f64,
    pub correct_decision_only: f64,
    pub complete_rate: f64,
    pub quartile_rates: [f64; 4],
    pub total_violations: u64,
    pub total_invalid_attempts: u64,
    pub total_retries: u64,
    pub final_p_deviate: f64,
}

pub fn run_batch(
    accounts: &[Account],
    arm: &str,
    seed: u64,
    p0: f64,
    drift_rate: f64,
    drift_scale: f64,
    max_retries: u32,
    restate_every: usize,
) -> (Vec<Outcome>, Summary) {
    let rng = SplitMix64::new(seed);
    let mut agent = DriftAgent::new(rng, p0, drift_rate, drift_scale, max_retries);
    let mut outcomes = Vec::with_capacity(accounts.len());
    for (idx, account) in accounts.iter().enumerate() {
        if arm == "restate" && idx > 0 && idx % restate_every == 0 {
            agent.restate(400);
        }
        if arm == "fsm" {
            let mut rt = FsmRuntime::new(account.clone());
            outcomes.push(agent.act(&mut rt));
        } else {
            let mut rt = PlainRuntime::new(account.clone());
            outcomes.push(agent.act(&mut rt));
        }
    }
    let n = outcomes.len();
    let good = outcomes.iter().filter(|o| o.correct_and_by_procedure).count();
    let q = n / 4;
    let mut quartiles = [0.0f64; 4];
    for k in 0..4 {
        let chunk = &outcomes[k * q..(k + 1) * q];
        quartiles[k] = chunk.iter().filter(|o| o.correct_and_by_procedure).count() as f64
            / chunk.len() as f64;
    }
    let summary = Summary {
        arm: arm.to_string(),
        n,
        correct_and_by_procedure: good,
        rate: good as f64 / n as f64,
        correct_decision_only: outcomes.iter().filter(|o| o.correct_decision).count() as f64
            / n as f64,
        complete_rate: outcomes.iter().filter(|o| o.complete).count() as f64 / n as f64,
        quartile_rates: quartiles,
        total_violations: outcomes.iter().map(|o| o.violations).sum(),
        total_invalid_attempts: outcomes.iter().map(|o| o.invalid_attempts).sum(),
        total_retries: outcomes.iter().map(|o| o.retries_used).sum(),
        final_p_deviate: agent.p_deviate(),
    };
    (outcomes, summary)
}
