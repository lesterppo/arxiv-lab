//! `verify` binary: mirrors the Python CPU test's experiments and
//! assertions (test_fsm_cpu.py). Prints the arm summaries, runs A1–A7,
//! writes results.json next to the binary's crate dir, exits 0 iff all
//! assertions pass. Usage: cargo run --release --bin verify

use fsmcontrol::*;
use std::fs;
use std::path::PathBuf;
use std::process::exit;

const SEED: u64 = 20261110;
const N: usize = 200;

fn check(name: &str, cond: bool, detail: &str, ok: &mut bool) {
    println!("  [{}] {} {}", if cond { "PASS" } else { "FAIL" }, name, detail);
    *ok &= cond;
}

fn main() {
    let mut ok = true;
    let accounts = make_accounts(N, SEED);

    println!("== main experiment (p0=0.05, drift_rate=0.35) ==");
    let (_, plain) = run_batch(&accounts, "plain", SEED, 0.05, 0.35, 1000.0, 3, 20);
    let (_, restate) = run_batch(&accounts, "restate", SEED, 0.05, 0.35, 1000.0, 3, 20);
    let (_, fsm) = run_batch(&accounts, "fsm", SEED, 0.05, 0.35, 1000.0, 3, 20);
    for s in [&plain, &restate, &fsm] {
        println!(
            "  {:8} correct-by-procedure {}/{} ({:.3}) quartiles {:?} violations {} invalid_attempts {} retries {}",
            s.arm,
            s.correct_and_by_procedure,
            s.n,
            s.rate,
            s.quartile_rates,
            s.total_violations,
            s.total_invalid_attempts,
            s.total_retries
        );
    }

    check("A1 FSM >= 170/200", fsm.correct_and_by_procedure >= 170,
          &format!("({})", fsm.correct_and_by_procedure), &mut ok);
    check("A1 FSM >= plain + 80",
          fsm.correct_and_by_procedure >= plain.correct_and_by_procedure + 80,
          &format!("(gap {})", fsm.correct_and_by_procedure - plain.correct_and_by_procedure), &mut ok);
    check("A2 FSM holds to the end (last q within 10pp of first)",
          (fsm.quartile_rates[3] - fsm.quartile_rates[0]).abs() <= 0.10,
          &format!("({:.2} -> {:.2})", fsm.quartile_rates[0], fsm.quartile_rates[3]), &mut ok);
    check("A2 plain falls >= 20pp first -> last quartile",
          plain.quartile_rates[0] - plain.quartile_rates[3] >= 0.20,
          &format!("({:.2} -> {:.2})", plain.quartile_rates[0], plain.quartile_rates[3]), &mut ok);
    check("A3 restate <= plain + 10 (restating does not fix drift)",
          restate.correct_and_by_procedure <= plain.correct_and_by_procedure + 10,
          &format!("(restate {}, plain {})", restate.correct_and_by_procedure, plain.correct_and_by_procedure), &mut ok);
    check("A3 restate < FSM - 60",
          fsm.correct_and_by_procedure > restate.correct_and_by_procedure + 60,
          &format!("(gap {})", fsm.correct_and_by_procedure - restate.correct_and_by_procedure), &mut ok);
    check("A4 FSM zero violations in accepted trace", fsm.total_violations == 0, "", &mut ok);
    check("A4 FSM rejected invalid calls (attempts > 0)", fsm.total_invalid_attempts > 0,
          &format!("({} rejected)", fsm.total_invalid_attempts), &mut ok);
    check("A4 plain arm actually violated procedure", plain.total_violations > 0,
          &format!("({} violations)", plain.total_violations), &mut ok);

    println!("== A5 no false blocking (drift off) ==");
    for arm in ["plain", "restate", "fsm"] {
        let (_, s) = run_batch(&accounts, arm, SEED, 0.0, 0.0, 1000.0, 3, 20);
        check(&format!("A5 {} 200/200 with drift off", arm),
              s.correct_and_by_procedure == 200,
              &format!("({})", s.correct_and_by_procedure), &mut ok);
    }

    println!("== A6 sensitivity over drift_rate ==");
    for mult in [0.5f64, 1.0, 1.5] {
        let (_, p) = run_batch(&accounts, "plain", SEED, 0.05, 0.35 * mult, 1000.0, 3, 20);
        let (_, f) = run_batch(&accounts, "fsm", SEED, 0.05, 0.35 * mult, 1000.0, 3, 20);
        let gap = f.correct_and_by_procedure as i64 - p.correct_and_by_procedure as i64;
        check(&format!("A6 drift x{}: FSM beats plain by >= 60", mult),
              gap >= 60,
              &format!("(plain {}, fsm {}, gap {})", p.correct_and_by_procedure, f.correct_and_by_procedure, gap), &mut ok);
    }

    println!("== A7 unit checks ==");
    let acc = Account::new(0, 100, true, false);
    let mut rt = FsmRuntime::new(acc);
    let (a, _) = rt.call("check_balance");
    check("A7 call before verify_identity rejected", !a && rt.state().state == START, "", &mut ok);
    let (a, _) = rt.call("verify_identity");
    check("A7 verify_identity accepted", a && rt.state().state == IDENTITY_VERIFIED, "", &mut ok);
    let (a, _) = rt.call("waive_fee");
    check("A7 waive_fee before eligibility rejected", !a && rt.state().state == IDENTITY_VERIFIED, "", &mut ok);
    rt.call("check_balance");
    rt.call("check_eligibility");
    let (a, _) = rt.call("close_account");
    check("A7 close while eligible rejected (must waive first)", !a && rt.state().state == ELIGIBILITY_CHECKED, "", &mut ok);
    let (a, _) = rt.call("waive_fee");
    check("A7 waive when eligible accepted", a && rt.state().state == FEE_WAIVED, "", &mut ok);
    let (a, _) = rt.call("escalate");
    check("A7 escalate from FEE_WAIVED rejected", !a && rt.state().state == FEE_WAIVED, "", &mut ok);
    let (complete, decision) = rt.end_turn();
    check("A7 end_turn non-terminal is incomplete", !complete && decision.is_none(), "", &mut ok);
    rt.call("close_account");
    let (complete, decision) = rt.end_turn();
    check("A7 end_turn terminal after close", complete && decision.as_deref() == Some("close"), "", &mut ok);

    let acc2 = Account::new(1, -50, false, true);
    let mut rt2 = FsmRuntime::new(acc2);
    for t in ["verify_identity", "check_balance", "check_eligibility"] {
        rt2.call(t);
    }
    let (a, _) = rt2.call("close_account");
    check("A7 close when escalation needed rejected", !a && rt2.state().state == ELIGIBILITY_CHECKED, "", &mut ok);
    let (a, _) = rt2.call("escalate");
    check("A7 escalate when needed accepted", a && rt2.state().state == ESCALATED, "", &mut ok);

    let mut rng = SplitMix64::new(0);
    let first3 = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
    check("A7 SplitMix64 seed-0 stream",
          first3 == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
          &format!("({:x?})", first3), &mut ok);

    // Write results.json next to the crate (session rust/ dir).
    let out = format!(
        "{{\n  \"paper\": \"2611.10884\",\n  \"lang\": \"rust\",\n  \"n_accounts\": {},\n  \"seed\": {},\n  \"plain_correct\": {},\n  \"restate_correct\": {},\n  \"fsm_correct\": {},\n  \"plain_quartiles\": {:?},\n  \"restate_quartiles\": {:?},\n  \"fsm_quartiles\": {:?},\n  \"fsm_violations\": {},\n  \"fsm_invalid_attempts\": {},\n  \"assertions_passed\": {}\n}}\n",
        N, SEED,
        plain.correct_and_by_procedure,
        restate.correct_and_by_procedure,
        fsm.correct_and_by_procedure,
        plain.quartile_rates, restate.quartile_rates, fsm.quartile_rates,
        fsm.total_violations, fsm.total_invalid_attempts, ok
    );
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results.json");
    fs::write(&path, out).expect("write results.json");
    println!("\n{}", if ok { "ALL ASSERTIONS PASSED" } else { "ASSERTIONS FAILED" });
    println!("results.json written to {}", path.display());
    exit(if ok { 0 } else { 1 });
}
