//! Unit tests for each VBL mechanism, mirroring the Python CPU test.

use std::collections::HashMap;
use vbl_rs::{react_baseline, report_specs, GoalSpec, Outcome, Vbl};

fn state(pairs: &[(&str, bool)]) -> HashMap<String, bool> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect()
}

#[test]
fn verify_contribution_and_feasibility() {
    let vbl = Vbl::new(report_specs(), "report", state(&[]), 8, 2);
    assert!(vbl.verify("report").0);
    assert!(vbl.verify("gather").0);
    assert!(vbl.verify("regather").0); // provider goal contributes
    assert!(!vbl.verify("stray").0); // stray: neither closure nor provider
    let (c, f, missing) = vbl.verify("analyze");
    assert!(c && !f && missing == vec!["data_ready".to_string()]);
    let vbl2 = Vbl::new(
        report_specs(),
        "report",
        state(&[("data_ready", true)]),
        8,
        2,
    );
    assert!(vbl2.verify("analyze").1);
}

#[test]
fn non_contributing_goal_is_popped() {
    // A goal outside the terminal closure and outside every provider
    // list must be popped, never executed.
    let mut specs = report_specs();
    let mut alien = GoalSpec::new("alien");
    alien.parent = Some("report".into());
    alien.action = Some("alien".into());
    specs.insert("alien".into(), alien);
    let mut vbl = Vbl::new(specs, "report", state(&[]), 8, 2);
    // Manually stack the alien goal on top (as a stray activation).
    vbl.stack_push_for_test("alien");
    let mut env = |action: &str, _state: &HashMap<String, bool>| -> Outcome {
        assert_ne!(action, "alien", "alien goal was executed, not popped");
        Outcome::Ok(vec![])
    };
    // The stack top is alien; run a single step by bounding max_steps
    // through a fresh controller is awkward, so run fully: report has
    // subgoals, but the alien pop must be the first trace entry.
    vbl.run(&mut env, 1);
    assert_eq!(vbl.trace[0].0, "Pop");
    assert_eq!(vbl.trace[0].1, "alien");
}

#[test]
fn retry_limit_then_pop() {
    let mut specs: HashMap<String, GoalSpec> = HashMap::new();
    let mut task = GoalSpec::new("task");
    task.action = Some("flaky".into());
    specs.insert("task".into(), task);
    let mut vbl = Vbl::new(specs, "task", state(&[]), 8, 2);
    let mut env = |_a: &str, _s: &HashMap<String, bool>| -> Outcome {
        Outcome::Fail("GENERIC".into(), vec![])
    };
    let (success, steps) = vbl.run(&mut env, 60);
    assert!(!success && steps == 3);
    let ops: Vec<&str> = vbl.trace.iter().map(|t| t.0.as_str()).collect();
    assert_eq!(ops, vec!["retry", "retry", "Pop"]);
}

#[test]
fn decomposition_failure_backtracks() {
    let mut specs: HashMap<String, GoalSpec> = HashMap::new();
    let mut report = GoalSpec::new("report");
    report.subgoals = vec!["analyze".into()];
    specs.insert("report".into(), report);
    let mut analyze = GoalSpec::new("analyze");
    analyze.parent = Some("report".into());
    analyze.action = Some("analyze".into());
    specs.insert("analyze".into(), analyze);
    let mut vbl = Vbl::new(specs, "report", state(&[]), 8, 2);
    let mut env = |_a: &str, _s: &HashMap<String, bool>| -> Outcome {
        Outcome::Fail("DECOMPOSITION_FAILED".into(), vec![])
    };
    vbl.run(&mut env, 6);
    assert!(vbl.trace.iter().any(|t| t.0 == "BacktrackTo"));
}

#[test]
fn react_baseline_burns_budget() {
    let mut attempts: HashMap<String, usize> = HashMap::new();
    let mut env = |action: &str, st: &HashMap<String, bool>| -> Outcome {
        let n = {
            let e = attempts.entry(action.to_string()).or_insert(0);
            *e += 1;
            *e
        };
        match action {
            "gather" => Outcome::Ok(vec![("data_ready".into(), true)]),
            "regather" => Outcome::Ok(vec![("data_ready".into(), true)]),
            "analyze" => {
                if n == 1 {
                    Outcome::Fail(
                        "MISSING_PRECONDITION".into(),
                        vec![("data_ready".into(), false)],
                    )
                } else if !st.get("data_ready").copied().unwrap_or(false) {
                    Outcome::Fail("MISSING_PRECONDITION".into(), vec![])
                } else {
                    Outcome::Ok(vec![("analysis_done".into(), true)])
                }
            }
            "write" => Outcome::Ok(vec![("report_done".into(), true)]),
            _ => Outcome::Ok(vec![]),
        }
    };
    let (ok, steps) =
        react_baseline(&mut env, &["gather", "analyze", "write"], 60);
    assert!(!ok && steps == 60);
}
