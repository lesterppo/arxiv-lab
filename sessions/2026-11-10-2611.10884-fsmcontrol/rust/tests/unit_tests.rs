//! Unit tests for each mechanism: transition table, branch conditions,
//! rejection semantics, end-of-turn, SplitMix64 stream, determinism.

use fsmcontrol::*;

fn eligible_account() -> Account {
    Account::new(0, 100, true, false)
}

fn escalating_account() -> Account {
    Account::new(1, -50, false, true)
}

fn plain_account() -> Account {
    Account::new(2, 300, false, false)
}

#[test]
fn transitions_follow_the_table() {
    let cases = [
        (START, "verify_identity", Some(IDENTITY_VERIFIED)),
        (IDENTITY_VERIFIED, "check_balance", Some(BALANCE_CHECKED)),
        (BALANCE_CHECKED, "check_eligibility", Some(ELIGIBILITY_CHECKED)),
        (ELIGIBILITY_CHECKED, "waive_fee", Some(FEE_WAIVED)),
        (ELIGIBILITY_CHECKED, "close_account", Some(CLOSED)),
        (FEE_WAIVED, "close_account", Some(CLOSED)),
        (ELIGIBILITY_CHECKED, "escalate", Some(ESCALATED)),
        (START, "check_balance", None),
        (START, "close_account", None),
        (IDENTITY_VERIFIED, "waive_fee", None),
        (FEE_WAIVED, "escalate", None),
        (CLOSED, "verify_identity", None),
        (ESCALATED, "close_account", None),
    ];
    for (state, tool, want) in cases {
        assert_eq!(transition(state, tool), want, "{} + {}", state, tool);
    }
}

#[test]
fn branch_conditions() {
    let el = eligible_account();
    let esc = escalating_account();
    let pl = plain_account();
    assert!(branch_is_valid(ELIGIBILITY_CHECKED, "waive_fee", &el));
    assert!(!branch_is_valid(ELIGIBILITY_CHECKED, "waive_fee", &pl));
    assert!(branch_is_valid(ELIGIBILITY_CHECKED, "escalate", &esc));
    assert!(!branch_is_valid(ELIGIBILITY_CHECKED, "escalate", &pl));
    // close from ELIGIBILITY_CHECKED only when neither eligible nor escalating
    assert!(!branch_is_valid(ELIGIBILITY_CHECKED, "close_account", &el));
    assert!(!branch_is_valid(ELIGIBILITY_CHECKED, "close_account", &esc));
    assert!(branch_is_valid(ELIGIBILITY_CHECKED, "close_account", &pl));
    assert!(branch_is_valid(FEE_WAIVED, "close_account", &el));
}

#[test]
fn correct_next_tool_walk() {
    let el = eligible_account();
    assert_eq!(correct_next_tool(START, &el), Some("verify_identity"));
    assert_eq!(correct_next_tool(IDENTITY_VERIFIED, &el), Some("check_balance"));
    assert_eq!(correct_next_tool(BALANCE_CHECKED, &el), Some("check_eligibility"));
    assert_eq!(correct_next_tool(ELIGIBILITY_CHECKED, &el), Some("waive_fee"));
    assert_eq!(correct_next_tool(FEE_WAIVED, &el), Some("close_account"));
    assert_eq!(correct_next_tool(CLOSED, &el), None);
    let esc = escalating_account();
    assert_eq!(correct_next_tool(ELIGIBILITY_CHECKED, &esc), Some("escalate"));
    let pl = plain_account();
    assert_eq!(correct_next_tool(ELIGIBILITY_CHECKED, &pl), Some("close_account"));
}

#[test]
fn fsm_rejects_and_keeps_state() {
    let mut rt = FsmRuntime::new(eligible_account());
    let (ok, _) = rt.call("check_balance");
    assert!(!ok);
    assert_eq!(rt.state().state, START);
    assert_eq!(rt.state().rejected.len(), 1);
    assert_eq!(rt.state().trace.len(), 0);
    let (ok, _) = rt.call("verify_identity");
    assert!(ok);
    assert_eq!(rt.state().state, IDENTITY_VERIFIED);
    // A rejected call mid-walk leaves the state unchanged.
    let (ok, _) = rt.call("waive_fee");
    assert!(!ok);
    assert_eq!(rt.state().state, IDENTITY_VERIFIED);
}

#[test]
fn fsm_end_of_turn_requires_terminal() {
    let mut rt = FsmRuntime::new(plain_account());
    assert_eq!(rt.end_turn(), (false, None));
    for t in ["verify_identity", "check_balance", "check_eligibility", "close_account"] {
        assert!(rt.call(t).0);
    }
    assert_eq!(rt.end_turn(), (true, Some("close".to_string())));
}

#[test]
fn plain_runtime_counts_violations_but_executes() {
    let mut rt = PlainRuntime::new(eligible_account());
    // Skip straight to close: executes, but counts a violation.
    let (ok, _) = rt.call("close_account");
    assert!(ok);
    assert_eq!(rt.state().violations, 1);
    // No transition from START for close_account: state unchanged.
    assert_eq!(rt.state().state, START);
    // A valid walk afterwards still reaches the terminal.
    for t in ["verify_identity", "check_balance", "check_eligibility"] {
        rt.call(t);
    }
    // Eligible account must waive; close here is a second violation.
    let (ok, _) = rt.call("close_account");
    assert!(ok);
    assert_eq!(rt.state().violations, 2);
    assert_eq!(rt.end_turn(), (true, Some("close".to_string())));
}

#[test]
fn splitmix64_known_stream() {
    let mut rng = SplitMix64::new(0);
    assert_eq!(rng.next_u64(), 0xE220A8397B1DCDAF);
    assert_eq!(rng.next_u64(), 0x6E789E6AA1B965F4);
    assert_eq!(rng.next_u64(), 0x06C45D188009454F);
    // random() stays in [0, 1).
    let mut rng = SplitMix64::new(12345);
    for _ in 0..1000 {
        let v = rng.random();
        assert!(v >= 0.0 && v < 1.0);
    }
}

#[test]
fn make_accounts_is_deterministic_and_mixed() {
    let a = make_accounts(200, 20261110);
    let b = make_accounts(200, 20261110);
    assert_eq!(a, b);
    let esc = a.iter().filter(|x| x.needs_escalation).count();
    let el = a.iter().filter(|x| x.eligible).count();
    assert!(esc > 20 && esc < 60, "escalations: {}", esc);
    assert!(el > 30 && el < 70, "eligible: {}", el);
    // A different seed gives a different batch.
    let c = make_accounts(200, 7);
    assert_ne!(a, c);
}
