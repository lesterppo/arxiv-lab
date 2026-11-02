//! Deeper Rust tests: property sweeps, exhaustive retry sequences
//! against an independent oracle, a soak beyond the Python test's
//! scale, and thread-identity for the stateless controller logic.

use std::collections::HashMap;
use vbl_rs::{report_specs, GoalSpec, Outcome, Vbl};

fn splitmix64(mut x: u64) -> impl FnMut() -> u64 {
    move || {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn run_report(
    inject: usize,
    max_retries: usize,
    max_stack: usize,
    init_data: bool,
) -> (bool, usize, bool) {
    let mut attempts: HashMap<String, usize> = HashMap::new();
    let mut env = |action: &str, state: &HashMap<String, bool>| -> Outcome {
        let n = {
            let e = attempts.entry(action.to_string()).or_insert(0);
            *e += 1;
            *e
        };
        match action {
            "gather" => Outcome::Ok(vec![("data_ready".into(), true)]),
            "regather" => Outcome::Ok(vec![("data_ready".into(), true)]),
            "analyze" => {
                if n <= inject {
                    Outcome::Fail(
                        "MISSING_PRECONDITION".into(),
                        vec![("data_ready".into(), false)],
                    )
                } else if !state.get("data_ready").copied().unwrap_or(false) {
                    Outcome::Fail("MISSING_PRECONDITION".into(), vec![])
                } else {
                    Outcome::Ok(vec![("analysis_done".into(), true)])
                }
            }
            "write" => {
                if !state.get("analysis_done").copied().unwrap_or(false) {
                    Outcome::Fail("MISSING_PRECONDITION".into(), vec![])
                } else {
                    Outcome::Ok(vec![("report_done".into(), true)])
                }
            }
            _ => Outcome::Ok(vec![]),
        }
    };
    let mut init: HashMap<String, bool> = HashMap::new();
    if init_data {
        init.insert("data_ready".into(), true);
    }
    let mut vbl = Vbl::new(report_specs(), "report", init, max_stack, max_retries);
    let (success, steps) = vbl.run(&mut env, 60);
    let done = vbl.state.get("report_done").copied().unwrap_or(false);
    (success, steps, done)
}

#[test]
fn random_report_cases_hold_invariants() {
    let mut rng = splitmix64(0xC0FFEE);
    for _ in 0..2000 {
        let inject = (rng() % 5) as usize;
        let max_retries = (rng() % 4) as usize;
        let max_stack = 1 + (rng() % 10) as usize;
        let init_data = rng() % 2 == 0;
        let (success, steps, done) =
            run_report(inject, max_retries, max_stack, init_data);
        assert!(steps <= 60, "steps exceeded budget");
        assert_eq!(success, done, "success must imply report_done");
        if inject == 0 && max_stack >= 3 {
            assert!(success, "no-failure case must succeed");
        }
    }
}

#[test]
fn exhaustive_retry_sequences_match_oracle() {
    // Single leaf goal, no parent: generic and decomposition failures
    // both fall through to retry-then-pop. Oracle, written
    // independently of the controller: count attempts; completion on
    // the first ok at or before retry budget + 1 attempts, else a
    // retry-limit pop. NOTE: a lone leaf terminal never sets the
    // controller's `success` flag (Python shares this quirk: only a
    // composite terminal completing sets it), so the oracle compares
    // step counts and the final trace entry, not `success`.
    fn oracle(seq: &[u8], max_retries: usize) -> (bool, usize) {
        let mut attempts = 0usize;
        for &outcome in seq {
            attempts += 1;
            if outcome == 0 {
                return (true, attempts);
            }
            if attempts > max_retries {
                return (false, attempts);
            }
        }
        // Sequence exhausted without ok: controller would keep
        // retrying on further (ok) calls; the test env returns ok
        // after the sequence, so the next attempt succeeds.
        (true, attempts + 1)
    }
    for max_retries in 0..=2usize {
        // All sequences up to length 4 over {ok, generic, decomp}.
        let mut seqs: Vec<Vec<u8>> = vec![vec![]];
        for len in 1..=4usize {
            let mut cur: Vec<Vec<u8>> = vec![vec![]];
            for _ in 0..len {
                let mut next = vec![];
                for s in &cur {
                    for v in 0..=2u8 {
                        let mut t = s.clone();
                        t.push(v);
                        next.push(t);
                    }
                }
                cur = next;
            }
            seqs.extend(cur);
        }
        for seq in seqs {
            let mut specs: HashMap<String, GoalSpec> = HashMap::new();
            let mut task = GoalSpec::new("task");
            task.action = Some("flaky".into());
            specs.insert("task".into(), task);
            let mut idx = 0usize;
            let seq_c = seq.clone();
            let mut env = |_a: &str, _s: &HashMap<String, bool>| -> Outcome {
                let v = if idx < seq_c.len() {
                    let x = seq_c[idx];
                    idx += 1;
                    x
                } else {
                    0
                };
                match v {
                    0 => Outcome::Ok(vec![]),
                    1 => Outcome::Fail("GENERIC".into(), vec![]),
                    _ => Outcome::Fail("DECOMPOSITION_FAILED".into(), vec![]),
                }
            };
            let mut vbl = Vbl::new(specs, "task", HashMap::new(), 8, max_retries);
            // The controller's step count counts loop iterations, not
            // env calls: for a lone leaf goal they coincide (one env
            // call per step until pop/completion), except that a
            // popped goal ends the run without an extra step.
            let (success, steps) = vbl.run(&mut env, 60);
            let (o_completed, o_steps) = oracle(&seq, max_retries);
            assert!(!success, "lone leaf terminal never sets success");
            let last = vbl.trace.last().expect("trace nonempty");
            let completed = last.2 == "completed";
            assert_eq!(
                (completed, steps),
                (o_completed, o_steps),
                "seq={:?} max_retries={} trace={:?}",
                seq,
                max_retries,
                vbl.trace
            );
        }
    }
}

#[test]
fn soak_twenty_thousand_cases() {
    let mut rng = splitmix64(12345);
    let mut successes = 0usize;
    for _ in 0..20000 {
        let inject = (rng() % 5) as usize;
        let (success, _, _) = run_report(inject, 2, 8, false);
        successes += success as usize;
    }
    // Every inject level 0..=4 succeeds within 60 steps: each injected
    // failure is repaired by exactly one regather cycle, so all 20000
    // cases must succeed (a failure here would mean a recovery loop
    // that never converges).
    assert_eq!(successes, 20000, "soak successes out of range");
}

#[test]
fn thread_identity() {
    use std::thread;
    let cases: Vec<(usize, usize, usize, bool)> = (0..200)
        .map(|i| (i % 5, i % 4, 1 + (i % 10), i % 2 == 0))
        .collect();
    let seq: Vec<_> = cases
        .iter()
        .map(|&(a, b, c, d)| run_report(a, b, c, d))
        .collect();
    let mut handles = vec![];
    for _ in 0..8 {
        let cases_c = cases.clone();
        handles.push(thread::spawn(move || {
            cases_c
                .iter()
                .map(|&(a, b, c, d)| run_report(a, b, c, d))
                .collect::<Vec<_>>()
        }));
    }
    for h in handles {
        assert_eq!(h.join().unwrap(), seq);
    }
}
