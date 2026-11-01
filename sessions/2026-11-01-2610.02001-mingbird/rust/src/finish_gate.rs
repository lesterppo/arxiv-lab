//! Finish gate that re-reads the task (Mingbird, arXiv:2610.02001).
//!
//! Rust port of `arxiv_lab/harness/finish_gate.py`. Python criteria are
//! `check_fn(env) -> bool` closures; the Rust version is generic over
//! the environment type and stores the same closures as boxed functions,
//! so the gate still checks the *current* environment, never the
//! model's claim.

/// Gate completion claims against re-checked task criteria.
pub struct FinishGate<Env> {
    pub task_text: String,
    pub criteria: Vec<(String, Box<dyn Fn(&Env) -> bool>)>,
}

impl<Env> FinishGate<Env> {
    pub fn new(
        task_text: impl Into<String>,
        criteria: Vec<(String, Box<dyn Fn(&Env) -> bool>)>,
    ) -> Self {
        Self {
            task_text: task_text.into(),
            criteria,
        }
    }

    /// Re-read the task; check each criterion against the CURRENT
    /// environment. Returns `(accepted, failed_criteria)`.
    pub fn review(&self, env: &Env) -> (bool, Vec<String>) {
        let failed: Vec<String> = self
            .criteria
            .iter()
            .filter(|(_, check)| !check(env))
            .map(|(name, _)| name.clone())
            .collect();
        (failed.is_empty(), failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn rejects_when_any_criterion_fails() {
        let gate: FinishGate<HashMap<String, String>> = FinishGate::new(
            "write x",
            vec![
                (
                    "a".to_string(),
                    Box::new(|e: &HashMap<String, String>| {
                        e.get("a").map(|v| v.as_str()) == Some("1")
                    }),
                ),
                (
                    "b".to_string(),
                    Box::new(|e: &HashMap<String, String>| e.contains_key("b")),
                ),
            ],
        );
        let mut env = HashMap::new();
        env.insert("a".to_string(), "1".to_string());
        assert_eq!(gate.review(&env), (false, vec!["b".to_string()]));
        env.insert("b".to_string(), "2".to_string());
        assert_eq!(gate.review(&env), (true, Vec::<String>::new()));
    }
}
