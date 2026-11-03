//! Deep tests for the RECEIPT port: exhaustive cells vs a hand-written
//! oracle table, a property sweep over random inputs, a soak, and
//! thread-identity (the model is stateless).

use receipt::{extract_waiting_prompt, simulate_outcome, Mode};

fn oracle(guarded: bool, mode: Mode, policy: bool) -> (Option<i32>, bool, bool, bool) {
    if !guarded {
        return (Some(0), true, false, false);
    }
    match mode {
        Mode::Closed => (Some(0), false, false, true),
        Mode::Open => (None, false, true, false),
        Mode::Receipt => {
            if policy {
                (Some(0), true, false, false)
            } else {
                (Some(0), false, false, false)
            }
        }
    }
}

// Deterministic xorshift RNG (no crates, no OS entropy needed).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn bit(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

#[test]
fn exhaustive_cells_match_oracle() {
    let modes = [Mode::Closed, Mode::Open, Mode::Receipt];
    for guarded in [false, true] {
        for mode in modes {
            for policy in [false, true] {
                assert_eq!(
                    simulate_outcome(guarded, mode, policy),
                    oracle(guarded, mode, policy),
                    "guarded={} mode={:?} policy={}",
                    guarded,
                    mode,
                    policy
                );
            }
        }
    }
}

#[test]
fn property_random_inputs_match_oracle() {
    let modes = [Mode::Closed, Mode::Open, Mode::Receipt];
    let mut rng = Rng(0x2026_1103_1930);
    for _ in 0..5000 {
        let guarded = rng.bit();
        let mode = modes[(rng.next() % 3) as usize];
        let policy = rng.bit();
        assert_eq!(simulate_outcome(guarded, mode, policy), oracle(guarded, mode, policy));
    }
}

#[test]
fn soak_100k_cases_all_valid() {
    // Every outcome must satisfy the paper's invariants: a timed-out
    // call never claims an action, and a false success is status 0
    // with no action (closed + guarded only).
    let modes = [Mode::Closed, Mode::Open, Mode::Receipt];
    let mut rng = Rng(0x1103);
    for _ in 0..100_000 {
        let guarded = rng.bit();
        let mode = modes[(rng.next() % 3) as usize];
        let policy = rng.bit();
        let (status, done, timed_out, false_success) = simulate_outcome(guarded, mode, policy);
        if timed_out {
            assert!(!done);
            assert!(status.is_none());
        }
        if false_success {
            assert_eq!(status, Some(0));
            assert!(!done);
            assert!(guarded && mode == Mode::Closed);
        }
        if done {
            assert_eq!(status, Some(0));
            assert!(!timed_out);
        }
    }
}

#[test]
fn prompt_detector_cases() {
    assert_eq!(
        extract_waiting_prompt("Confirm: x? [y/N] ").as_deref(),
        Some("Confirm: x? [y/N]")
    );
    assert_eq!(
        extract_waiting_prompt("Are you sure? [YES/NO] ").as_deref(),
        Some("Are you sure? [YES/NO]")
    );
    assert_eq!(extract_waiting_prompt("status: ok\n").as_deref(), None);
    assert_eq!(extract_waiting_prompt("a question? in the middle\nmore text").as_deref(), None);
    assert_eq!(
        extract_waiting_prompt("first?\nsecond?").as_deref(),
        Some("second?")
    );
}

#[test]
fn thread_identity_stateless() {
    use std::sync::Arc;
    use std::thread;
    let modes = Arc::new([Mode::Closed, Mode::Open, Mode::Receipt]);
    let mut handles = vec![];
    for t in 0..8 {
        let modes = Arc::clone(&modes);
        handles.push(thread::spawn(move || {
            let mut rng = Rng(0xabc + t);
            let mut n = 0;
            for _ in 0..2500 {
                let m = modes[(rng.next() % 3) as usize];
                let o = simulate_outcome(rng.bit(), m, rng.bit());
                assert!(o.0.is_none() || o.0 == Some(0));
                n += 1;
            }
            n
        }));
    }
    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total, 8 * 2500);
}
