//! Deeper tests for the SAT mechanism: property sweeps, exhaustive
//! limitation subsets against the probe oracle, a 2,000-task soak, and
//! thread-identity (the student is stateless across tasks).

use studentteaching::*;
use std::sync::Arc;

#[test]
fn probe_oracle_over_all_limitation_subsets() {
    // For every subset S of limitations, a student with limitations S and
    // no skills fails probe L iff L in S; with skills S it passes all.
    let subsets: Vec<Vec<&str>> = (0..32u32)
        .map(|mask| {
            LIMITATIONS
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, l)| *l)
                .collect()
        })
        .collect();
    for s in &subsets {
        let unskilled = StudentAgent::new(s, &[]);
        let skilled = StudentAgent::new(s, s);
        for lim in LIMITATIONS {
            let probe = gen_probe_task(lim);
            let (ok_u, _, _) = unskilled.solve(&probe);
            assert_eq!(ok_u, !s.contains(&lim), "subset {:?} probe {}", s, lim);
            let (ok_s, _, _) = skilled.solve(&probe);
            assert!(ok_s, "subset {:?} skilled probe {}", s, lim);
        }
    }
}

#[test]
fn soak_2000_tasks() {
    let tasks = gen_tasks(424242, 2000);
    // A student with no limitations and no skills follows the correct
    // procedure everywhere (budget 14 >= the 13-call maximum) and must
    // solve everything; an all-skills student cannot, because 5 skills
    // cut its budget to 9 and long tasks need up to 13 calls.
    let clean = StudentAgent::new(&[], &[]);
    let (acc, _) = run_batch(&clean, &tasks);
    // A clean student fails only the ~0.2% of tasks that need 15 calls
    // (flaky retries on two keys + stale re-read) against BASE_BUDGET=14;
    // every failure must be a budget failure, never a wrong answer.
    assert!(acc >= 0.99, "clean soak acc {acc}");
    for t in &tasks {
        let (ok, _, prints) = clean.solve(t);
        if !ok {
            assert!(
                prints.iter().any(|p| p == "budget_exhausted"),
                "clean student failed without budget exhaustion: {:?}",
                prints
            );
        }
    }
    let untaught_a = StudentAgent::new(&["L1", "L3", "L5"], &[]);
    let (acc_a, _) = run_batch(&untaught_a, &tasks);
    assert!((0.25..0.55).contains(&acc_a), "untaught A soak acc {acc_a}");
    let sat = StudentAgent::new(&["L1", "L3", "L5"], &["L1", "L3", "L5"]);
    let (acc_sat, _) = run_batch(&sat, &tasks);
    assert!(acc_sat > acc_a + 0.3, "sat {acc_sat} vs untaught {acc_a}");
}

#[test]
fn property_sweep_more_skills_never_hurt_correctness() {
    // Adding a skill for a limitation the student HAS can only remove a
    // deviation; on tasks whose call count fits the reduced budget, success
    // is monotone in the skill set. Check per-task: skilled outcome is
    // never worse than unskilled on tasks the skilled student completes
    // within budget.
    for seed in [11, 222, 3333] {
        let tasks = gen_tasks(seed, 100);
        let base = StudentAgent::new(&["L1", "L3", "L5"], &[]);
        for skill in ["L1", "L3", "L5"] {
            let skilled = StudentAgent::new(&["L1", "L3", "L5"], &[skill]);
            for t in &tasks {
                let (ok_b, _, _) = base.solve(t);
                let (ok_s, calls_s, prints_s) = skilled.solve(t);
                if ok_b && !prints_s.iter().any(|p| p == "budget_exhausted") {
                    assert!(ok_s, "seed {seed} skill {skill}: base passed but skilled failed (calls {calls_s})");
                }
            }
        }
    }
}

#[test]
fn property_sweep_budget_monotone() {
    // Holding limitations fixed, fewer skills (larger budget) never turns a
    // budget failure into a pass... (inverse check) — instead: a student
    // with MORE skills installed fails at least as many tasks on budget
    // as the same student with fewer skills, when both fix the same
    // limitations. Compare PF (5 skills) vs SAT (3 skills) budget failures.
    let tasks = gen_tasks(20261111 + 5000, 200);
    let mut pf_budget_fails = 0;
    let mut sat_budget_fails = 0;
    for t in &tasks {
        let pf = StudentAgent::new(&["L1", "L3", "L5"], &["L1", "L2", "L3", "L4", "L5"]);
        let sat = StudentAgent::new(&["L1", "L3", "L5"], &["L1", "L3", "L5"]);
        if pf.solve(t).2.iter().any(|p| p == "budget_exhausted") {
            pf_budget_fails += 1;
        }
        if sat.solve(t).2.iter().any(|p| p == "budget_exhausted") {
            sat_budget_fails += 1;
        }
    }
    assert!(pf_budget_fails > sat_budget_fails, "pf {pf_budget_fails} sat {sat_budget_fails}");
}

#[test]
fn thread_identity_stateless_student() {
    let tasks = Arc::new(gen_tasks(20261111 + 5000, 200));
    let student = Arc::new(StudentAgent::new(&["L1", "L3", "L5"], &["L1", "L3", "L5"]));
    let single: Vec<(bool, i64, Vec<String>)> =
        tasks.iter().map(|t| student.solve(t)).collect();
    let mut handles = vec![];
    for chunk in 0..4 {
        let tasks = Arc::clone(&tasks);
        let student = Arc::clone(&student);
        handles.push(std::thread::spawn(move || {
            let part: Vec<(bool, i64, Vec<String>)> = tasks
                .iter()
                .skip(chunk * 50)
                .take(50)
                .map(|t| student.solve(t))
                .collect();
            (chunk, part)
        }));
    }
    let mut threaded: Vec<(bool, i64, Vec<String>)> = vec![Default::default(); 200];
    for h in handles {
        let (chunk, part) = h.join().unwrap();
        for (i, v) in part.into_iter().enumerate() {
            threaded[chunk * 50 + i] = v;
        }
    }
    assert_eq!(single, threaded);
}
