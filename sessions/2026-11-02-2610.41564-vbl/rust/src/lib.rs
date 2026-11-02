//! Rust port of `arxiv_lab.agent.vbl` (arXiv:2610.41564).
//!
//! VBL keeps a task decomposition as a bounded goal stack and, on goal
//! activation, execution failure, or material state change, verifies two
//! horizons -- whether the active goal contributes to the terminal goal
//! and whether it is feasible in the current state -- then applies an
//! explicit operation: Push, Pop, retry, or BacktrackTo the parent.
//! Std only; no model, no network. The Python version is the primary
//! verdict carrier; this port mirrors its control logic exactly for
//! the operation + goal sequence (see README for documented divergences).

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct GoalSpec {
    pub name: String,
    pub parent: Option<String>,
    pub subgoals: Vec<String>,
    pub preconditions: Vec<String>,
    /// (state key, provider goal name) pairs.
    pub provider_for: Vec<(String, String)>,
    pub action: Option<String>,
}

impl GoalSpec {
    pub fn new(name: &str) -> Self {
        GoalSpec {
            name: name.to_string(),
            parent: None,
            subgoals: vec![],
            preconditions: vec![],
            provider_for: vec![],
            action: None,
        }
    }
    pub fn provider(&self, key: &str) -> Option<String> {
        self.provider_for
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }
}

#[derive(Clone, Debug)]
struct GoalState {
    name: String,
    retries: usize,
    done_subgoals: HashSet<String>,
}

#[derive(Clone, Debug)]
pub enum Outcome {
    Ok(Vec<(String, bool)>),
    Fail(String, Vec<(String, bool)>),
}

pub struct Vbl {
    pub specs: HashMap<String, GoalSpec>,
    pub terminal: String,
    pub state: HashMap<String, bool>,
    pub max_stack: usize,
    pub max_retries: usize,
    pub stack: Vec<GoalState>,
    pub trace: Vec<(String, String, String)>,
    pub steps: usize,
}

impl Vbl {
    pub fn new(
        specs: HashMap<String, GoalSpec>,
        terminal: &str,
        state: HashMap<String, bool>,
        max_stack: usize,
        max_retries: usize,
    ) -> Self {
        Vbl {
            specs,
            terminal: terminal.to_string(),
            state,
            max_stack,
            max_retries,
            stack: vec![GoalState {
                name: terminal.to_string(),
                retries: 0,
                done_subgoals: HashSet::new(),
            }],
            trace: vec![],
            steps: 0,
        }
    }

    /// Test support only: stack a goal directly, bypassing decomposition
    /// (used to simulate a stray activation). Not a controller operation.
    pub fn stack_push_for_test(&mut self, name: &str) {
        self.stack.push(GoalState {
            name: name.to_string(),
            retries: 0,
            done_subgoals: HashSet::new(),
        });
    }

    fn spec(&self, name: &str) -> GoalSpec {
        self.specs.get(name).expect("unknown goal").clone()
    }

    fn contributes(&self, name: &str) -> bool {
        if name == self.terminal {
            return true;
        }
        // Provider goals contribute transitively.
        for spec in self.specs.values() {
            if spec.provider_for.iter().any(|(_, v)| v == name) {
                return true;
            }
        }
        // Terminal closure.
        let mut seen: HashSet<String> = HashSet::new();
        let mut frontier = vec![self.terminal.clone()];
        while let Some(cur) = frontier.pop() {
            if !seen.insert(cur.clone()) {
                continue;
            }
            if let Some(spec) = self.specs.get(&cur) {
                frontier.extend(spec.subgoals.iter().cloned());
            }
        }
        seen.contains(name)
    }

    pub fn verify(&self, name: &str) -> (bool, bool, Vec<String>) {
        let spec = self.spec(name);
        let missing: Vec<String> = spec
            .preconditions
            .iter()
            .filter(|k| !self.state.get(*k).copied().unwrap_or(false))
            .cloned()
            .collect();
        (self.contributes(name), missing.is_empty(), missing)
    }

    fn push(&mut self, name: &str, reason: &str) -> bool {
        if self.stack.len() >= self.max_stack {
            return false;
        }
        self.stack.push(GoalState {
            name: name.to_string(),
            retries: 0,
            done_subgoals: HashSet::new(),
        });
        self.trace
            .push(("Push".to_string(), name.to_string(), reason.to_string()));
        true
    }

    fn pop(&mut self, reason: &str) -> Option<GoalState> {
        let top = self.stack.pop()?;
        self.trace.push((
            "Pop".to_string(),
            top.name.clone(),
            reason.to_string(),
        ));
        Some(top)
    }

    fn backtrack_to_parent(&mut self, reason: &str) -> bool {
        let parent = match self.stack.last() {
            Some(top) => match self.spec(&top.name).parent {
                Some(p) => p,
                None => return false,
            },
            None => return false,
        };
        while !self.stack.is_empty()
            && self.stack.last().unwrap().name != parent
        {
            let detail = format!("BacktrackTo:{}:{}", parent, reason);
            self.pop(&detail);
        }
        if !self.stack.is_empty() {
            self.trace.push((
                "BacktrackTo".to_string(),
                parent,
                reason.to_string(),
            ));
            return true;
        }
        false
    }

    /// Mirrors Python `diagnose`; returns a short outcome label.
    fn diagnose(
        &mut self,
        name: &str,
        failure_type: &str,
        missing: &[String],
    ) -> &'static str {
        let spec = self.spec(name);
        if failure_type == "MISSING_PRECONDITION" && !missing.is_empty() {
            let key = &missing[0];
            if let Some(provider) = spec.provider(key) {
                if self.specs.contains_key(&provider) {
                    if self.push(&provider, &format!("missing:{}", key)) {
                        return "pushed-provider";
                    }
                    return "backtracked-stack-full";
                }
            }
            if spec.parent.is_some()
                && self.backtrack_to_parent(&format!("missing:{}", key))
            {
                return "backtracked";
            }
        } else if failure_type == "DECOMPOSITION_FAILED" {
            if spec.parent.is_some()
                && self.backtrack_to_parent("decomposition")
            {
                return "backtracked";
            }
        }
        // Generic failure: retry up to the limit, then give up on it.
        let idx = self.stack.len() - 1;
        self.stack[idx].retries += 1;
        if self.stack[idx].retries <= self.max_retries {
            self.trace.push((
                "retry".to_string(),
                name.to_string(),
                failure_type.to_string(),
            ));
            return "retried";
        }
        self.pop(&format!("retry-limit:{}", failure_type));
        "popped-retry-limit"
    }

    pub fn run(
        &mut self,
        env: &mut dyn FnMut(&str, &HashMap<String, bool>) -> Outcome,
        max_steps: usize,
    ) -> (bool, usize) {
        let mut success = false;
        while !self.stack.is_empty() && self.steps < max_steps {
            self.steps += 1;
            let top_name = self.stack.last().unwrap().name.clone();
            let spec = self.spec(&top_name);
            let (contributes, feasible, missing) = self.verify(&top_name);
            if !contributes {
                self.pop("non-contributing");
                continue;
            }
            if !feasible {
                let key = &missing[0];
                if let Some(provider) = spec.provider(key) {
                    if self.specs.contains_key(&provider) {
                        if !self.push(&provider, &format!("verify-missing:{}", key)) {
                            self.backtrack_to_parent("verify-stack-full");
                        }
                        continue;
                    }
                }
                if spec.parent.is_some() {
                    self.backtrack_to_parent(&format!("verify-missing:{}", key));
                    continue;
                }
                self.pop("verify-missing-no-provider");
                continue;
            }
            let pending: Vec<String> = spec
                .subgoals
                .iter()
                .filter(|s| {
                    !self
                        .stack
                        .last()
                        .unwrap()
                        .done_subgoals
                        .contains(*s)
                })
                .cloned()
                .collect();
            if let Some(next) = pending.first() {
                self.push(next, "decompose");
                continue;
            }
            if spec.action.is_none() {
                if top_name == self.terminal {
                    success = true;
                }
                self.pop("completed");
                if let Some(parent) = self.stack.last_mut() {
                    parent.done_subgoals.insert(top_name.clone());
                }
                continue;
            }
            let action = spec.action.clone().unwrap();
            match env(&action, &self.state) {
                Outcome::Ok(effects) => {
                    for (k, v) in effects {
                        self.state.insert(k, v);
                    }
                    self.pop("completed");
                    if let Some(parent) = self.stack.last_mut() {
                        parent.done_subgoals.insert(top_name.clone());
                    }
                }
                Outcome::Fail(failure_type, effects) => {
                    for (k, v) in effects {
                        self.state.insert(k, v);
                    }
                    let (_, _, missing_now) = self.verify(&top_name);
                    let result =
                        self.diagnose(&top_name, &failure_type, &missing_now);
                    if result == "backtracked-stack-full" {
                        self.backtrack_to_parent("stack-full");
                    }
                }
            }
        }
        (success, self.steps)
    }

    pub fn op_sequence(&self) -> Vec<(String, String)> {
        self.trace
            .iter()
            .map(|(op, goal, _)| (op.clone(), goal.clone()))
            .collect()
    }
}

/// Free-form retry baseline, mirroring Python `react_baseline`.
pub fn react_baseline(
    env: &mut dyn FnMut(&str, &HashMap<String, bool>) -> Outcome,
    actions: &[&str],
    max_steps: usize,
) -> (bool, usize) {
    let mut steps = 0usize;
    let mut state: HashMap<String, bool> = HashMap::new();
    for action in actions {
        loop {
            if steps >= max_steps {
                return (false, steps);
            }
            steps += 1;
            match env(action, &state) {
                Outcome::Ok(effects) => {
                    for (k, v) in effects {
                        state.insert(k, v);
                    }
                    break;
                }
                Outcome::Fail(_, effects) => {
                    for (k, v) in effects {
                        state.insert(k, v);
                    }
                }
            }
        }
        if steps >= max_steps {
            return (false, steps);
        }
    }
    (true, steps)
}

/// The scripted report task used by both the Python test and `verify`.
pub fn report_specs() -> HashMap<String, GoalSpec> {
    let mut m: HashMap<String, GoalSpec> = HashMap::new();
    let mut report = GoalSpec::new("report");
    report.subgoals = vec!["gather".into(), "analyze".into(), "write".into()];
    m.insert("report".into(), report);
    let mut gather = GoalSpec::new("gather");
    gather.parent = Some("report".into());
    gather.action = Some("gather".into());
    m.insert("gather".into(), gather);
    let mut analyze = GoalSpec::new("analyze");
    analyze.parent = Some("report".into());
    analyze.preconditions = vec!["data_ready".into()];
    analyze.provider_for = vec![("data_ready".into(), "regather".into())];
    analyze.action = Some("analyze".into());
    m.insert("analyze".into(), analyze);
    let mut regather = GoalSpec::new("regather");
    regather.parent = Some("analyze".into());
    regather.action = Some("regather".into());
    m.insert("regather".into(), regather);
    let mut write = GoalSpec::new("write");
    write.parent = Some("report".into());
    write.preconditions = vec!["analysis_done".into()];
    write.action = Some("write".into());
    m.insert("write".into(), write);
    let mut stray = GoalSpec::new("stray");
    stray.parent = Some("report".into());
    stray.action = Some("stray".into());
    m.insert("stray".into(), stray);
    m
}
