//! Property sweeps: random inputs / budgets / seeds. All properties are
//! derived from the mechanism's guarantees, not from one battery run.

use caplite::*;

/// Tiny SplitMix-based RNG for generating test inputs (independent of
/// the mechanism's own streams).
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        let (v, s) = splitmix64(self.0);
        self.0 = s;
        v
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

const ALPHABET: &[u8] = b"abcxyzABCXYZ019 @$,./-_:%+";

fn random_request(rng: &mut TestRng, max_len: usize) -> String {
    let len = rng.below(max_len as u64 + 1) as usize;
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        s.push(*rng.pick(ALPHABET) as char);
    }
    s
}

#[test]
fn parse_invariants_over_random_inputs() {
    let mut rng = TestRng(0xC0FFEE);
    for _ in 0..2000 {
        let req = random_request(&mut rng, 60);
        let st = parse_stated(&req);
        // Deterministic: parsing twice gives the same sets.
        assert_eq!(st, parse_stated(&req));
        // Every stated email is lowercase and contains '@'.
        for e in &st.emails {
            assert_eq!(e, &e.to_ascii_lowercase());
            assert!(e.contains('@'), "stated email without @: {}", e);
            assert!(e.len() >= 6, "stated email too short: {}", e); // a@b.cc
        }
        // Every stated amount is a finite non-negative number.
        for a in &st.amounts {
            assert!(a.0.is_finite() && a.0 >= 0.0, "bad amount: {}", a.0);
        }
        // Every stated path starts with '/'.
        for p in &st.paths {
            assert!(p.starts_with('/'), "stated path without /: {}", p);
            assert!(!p.ends_with('.'), "stated path keeps a trailing dot: {}", p);
        }
    }
}

#[test]
fn parse_never_panics_on_hostile_bytes() {
    // Bytes outside the ASCII classes must act as separators, never panic.
    let mut rng = TestRng(42);
    for _ in 0..500 {
        let mut bytes = vec![0u8; rng.below(40) as usize + 1];
        for b in bytes.iter_mut() {
            *b = rng.below(256) as u8;
        }
        // Keep it valid UTF-8 (parsers take &str); control the alphabet.
        let s: String = bytes
            .iter()
            .map(|b| match b {
                0..=31 => ' ',
                127..=159 => 'x',
                _ => (*b as char),
            })
            .collect();
        let st = parse_stated(&s);
        assert_eq!(st, parse_stated(&s));
    }
}

#[test]
fn battery_sweep_capLite_never_admits_an_attack() {
    for seed in [1u64, 2, 3, 7, 42, 999] {
        for (n_attack, n_benign) in [(10usize, 5usize), (50, 25), (300, 150)] {
            for lookup_frac in [0.0, 0.1, 0.5, 1.0] {
                let b = run_battery(seed, n_attack, n_benign, lookup_frac);
                assert_eq!(
                    b.caplite.attack_success,
                    0,
                    "seed {} {:?}",
                    seed,
                    (n_attack, n_benign, lookup_frac)
                );
                assert_eq!(b.caplite.bad_write, 0);
                // Undefended really is attacked whenever there is an attack.
                assert_eq!(b.undefended.attack_success, b.undefended.attack_n);
            }
        }
    }
}

#[test]
fn completion_monotone_in_lookup_frac() {
    // Same seed stream: raising lookup_frac only converts benign
    // episodes into lookup tasks, which CapLite blocks by design, so
    // CapLite benign completion can only fall.
    for seed in [1u64, 7, 42] {
        let mut prev = u64::MAX;
        for lookup_frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] {
            let b = run_battery(seed, 100, 200, lookup_frac);
            assert!(
                b.caplite.completed <= prev,
                "seed {} frac {}: {} > {}",
                seed,
                lookup_frac,
                b.caplite.completed,
                prev
            );
            prev = b.caplite.completed;
        }
        // At lookup_frac 0 there are no lookup tasks; at 1.0 every benign
        // episode is a lookup task, so CapLite completes only the
        // non-planner-error... actually none: lookup writes are never
        // committed, so completion is 0.
        let b = run_battery(seed, 100, 200, 1.0);
        assert_eq!(b.caplite.completed, 0);
    }
}

#[test]
fn token_accounting_is_exact() {
    // The CaMeL-style arm differs from CapLite only by the +180-token
    // program-writing call per episode; undefended has identical token
    // counts to CapLite. Exact, not approximate.
    for seed in [1u64, 7, 123] {
        let b = run_battery(seed, 120, 80, 0.1);
        let n = (b.caplite.attack_n + b.caplite.benign_n) as u64;
        assert_eq!(b.undefended.tokens, b.caplite.tokens);
        assert_eq!(b.camel.tokens, b.caplite.tokens + 180 * n);
        assert!(b.caplite.mean_tokens < b.camel.mean_tokens);
    }
}

#[test]
fn lookup_tasks_complete_only_without_enforcement() {
    // Benign lookup episodes: CapLite completes 0 by design; both
    // baselines complete all but planner-error ones.
    let b = run_battery(7, 0, 150, 1.0);
    assert_eq!(b.caplite.completed, 0);
    assert!(
        b.camel.completed > 130,
        "camel lookup completion {}",
        b.camel.completed
    );
    assert_eq!(b.camel.completed, b.undefended.completed);
}
