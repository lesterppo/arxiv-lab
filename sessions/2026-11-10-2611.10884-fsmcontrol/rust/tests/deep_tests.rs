//! Deeper tests adapted to the mechanism:
//!  - property sweeps over random seeds / drift rates: the FSM arm never
//!    accepts a procedure violation, and always beats plain by a wide gap;
//!  - exhaustive + random sequences against an oracle (the transition
//!    table): every tool from every state is accepted iff the oracle says so;
//!  - a soak at a scale the Python test does not reach (2000 accounts);
//!  - thread-identity: the code is stateless (all state in explicit
//!    structs), so the same batch run on parallel threads is identical.

use fsmcontrol::*;
use std::collections::HashSet;

fn oracle_accept(state: &str, tool: &str, account: &Account) -> bool {
    transition(state, tool).is_some() && branch_is_valid(state, tool, account)
}

#[test]
fn fsm_matches_oracle_on_every_state_tool_pair() {
    let accounts = [
        Account::new(0, 100, true, false),
        Account::new(1, -50, false, true),
        Account::new(2, 300, false, false),
    ];
    let states = [
        START, IDENTITY_VERIFIED, BALANCE_CHECKED, ELIGIBILITY_CHECKED,
        FEE_WAIVED, CLOSED, ESCALATED,
    ];
    for account in &accounts {
        for state in states {
            for tool in TOOLS {
                // Drive an FSM to `state` first (via the legal walk), then
                // probe `tool` — except for terminal states, which we reach
                // and then probe directly (every call must be rejected).
                let mut rt = FsmRuntime::new(account.clone());
                // Walk to the target state using correct_next_tool.
                let mut guard = 0;
                while rt.state().state != state && guard < 10 {
                    guard += 1;
                    match correct_next_tool(&rt.state().state, account) {
                        Some(t) => {
                            assert!(rt.call(t).0);
                        }
                        None => break,
                    }
                }
                if rt.state().state != state {
                    // Unreachable target for this account (e.g. FEE_WAIVED
                    // for a non-eligible account): skip, it is covered by
                    // the other accounts.
                    continue;
                }
                let before = rt.state().state.clone();
                let (accepted, _) = rt.call(tool);
                assert_eq!(
                    accepted,
                    oracle_accept(&before, tool, account),
                    "state={} tool={} eligible={} esc={}",
                    before, tool, account.eligible, account.needs_escalation
                );
            }
        }
    }
}

#[test]
fn random_tool_sequences_never_violate_in_fsm() {
    let mut rng = SplitMix64::new(424242);
    for _ in 0..2000 {
        // Coherent accounts, as in make_accounts: escalating => not eligible.
        let needs_esc = rng.random() < 0.3;
        let eligible = !needs_esc && rng.random() < 0.5;
        let account = Account::new(
            0,
            rng.next_u64() as i64 % 1000,
            eligible,
            needs_esc,
        );
        let mut rt = FsmRuntime::new(account.clone());
        for _ in 0..12 {
            let tool = rng.choice(&TOOLS);
            rt.call(tool);
        }
        assert_eq!(rt.state().violations, 0);
        // Every accepted trace step must be a tabled, branch-valid move.
        for (tool, from, to) in &rt.state().trace {
            assert_eq!(transition(from, tool), Some(to.as_str()));
            assert!(branch_is_valid(from, tool, &account));
        }
        // End-of-turn completeness matches terminal state.
        let (complete, decision) = rt.end_turn();
        assert_eq!(complete, is_terminal(&rt.state().state));
        if complete {
            assert_eq!(decision.as_deref(), Some(account.correct_terminal()));
        }
    }
}

#[test]
fn sweep_fsm_never_violates_and_always_wins() {
    for seed in [1u64, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233] {
        let accounts = make_accounts(200, seed);
        for drift in [0.1f64, 0.35, 0.6, 0.9] {
            let (_, p) = run_batch(&accounts, "plain", seed, 0.05, drift, 1000.0, 3, 20);
            let (_, f) = run_batch(&accounts, "fsm", seed, 0.05, drift, 1000.0, 3, 20);
            assert_eq!(f.total_violations, 0, "seed {} drift {}", seed, drift);
            assert!(
                f.correct_and_by_procedure >= p.correct_and_by_procedure + 40,
                "seed {} drift {}: fsm {} plain {}",
                seed, drift, f.correct_and_by_procedure, p.correct_and_by_procedure
            );
        }
    }
}

#[test]
fn sweep_restate_never_fixes_drift() {
    for seed in [1u64, 7, 42, 12345] {
        let accounts = make_accounts(200, seed);
        let (_, p) = run_batch(&accounts, "plain", seed, 0.05, 0.35, 1000.0, 3, 20);
        let (_, r) = run_batch(&accounts, "restate", seed, 0.05, 0.35, 1000.0, 3, 20);
        assert!(
            r.correct_and_by_procedure <= p.correct_and_by_procedure + 10,
            "seed {}: restate {} plain {}",
            seed, r.correct_and_by_procedure, p.correct_and_by_procedure
        );
    }
}

#[test]
fn soak_2000_accounts_fsm_holds_up() {
    // 10x the Python test's scale; drift_scale grows with the run too
    // (10000 steps ~ 2000 accounts), so the drift curve keeps the same
    // shape as the 200-account / drift_scale=1000 setup instead of
    // saturating in the first quartile.
    let accounts = make_accounts(2000, 20261110);
    let (_, f) = run_batch(&accounts, "fsm", 20261110, 0.05, 0.35, 10000.0, 3, 20);
    let (_, p) = run_batch(&accounts, "plain", 20261110, 0.05, 0.35, 10000.0, 3, 20);
    assert_eq!(f.total_violations, 0);
    assert!(f.rate >= 0.90, "fsm soak rate {}", f.rate);
    assert!(f.rate - p.rate >= 0.4, "fsm {} plain {}", f.rate, p.rate);
    // FSM holds its rate to the end even at 10x scale.
    assert!((f.quartile_rates[3] - f.quartile_rates[0]).abs() <= 0.10);
    assert!(p.quartile_rates[0] - p.quartile_rates[3] >= 0.20);
}

#[test]
fn thread_identity_same_batch_same_result() {
    // The mechanism is stateless: all state lives in explicit structs
    // (SplitMix64, DriftAgent, runtimes), so parallel runs agree exactly.
    let accounts = make_accounts(200, 20261110);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let acc = accounts.clone();
            std::thread::spawn(move || {
                let (_, s) = run_batch(&acc, "fsm", 20261110, 0.05, 0.35, 1000.0, 3, 20);
                (s.correct_and_by_procedure, s.total_violations, s.total_invalid_attempts, s.total_retries)
            })
        })
        .collect();
    let results: HashSet<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.len(), 1, "threads disagreed: {:?}", results);
}

#[test]
fn drift_disabled_every_arm_perfect() {
    let accounts = make_accounts(200, 20261110);
    for arm in ["plain", "restate", "fsm"] {
        let (_, s) = run_batch(&accounts, arm, 20261110, 0.0, 0.0, 1000.0, 3, 20);
        assert_eq!(s.correct_and_by_procedure, 200, "arm {}", arm);
    }
}
