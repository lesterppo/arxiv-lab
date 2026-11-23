//! Deep tests: property sweeps, exhaustive sequences against the
//! Python oracle's semantics (re-checked in parity_sweep.py),
//! a soak at a scale the Python test does not reach, and
//! thread-identity for the stateless pure logic.

use splitbench_rs::{
    grade_outputs, parse_json, random_json, values_equal, Json, SplitMix64,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn digest_of<I: IntoIterator<Item = bool>>(items: I) -> u64 {
    let mut h = DefaultHasher::new();
    for b in items {
        b.hash(&mut h);
    }
    h.finish()
}

#[test]
fn property_reflexivity_symmetry() {
    let mut rng = SplitMix64::new(4242);
    for _ in 0..2000 {
        let a = random_json(&mut rng, 0);
        let b = random_json(&mut rng, 0);
        assert!(values_equal(&a, &a), "reflexivity: {}", a.to_canonical());
        assert_eq!(
            values_equal(&a, &b),
            values_equal(&b, &a),
            "symmetry: {} vs {}",
            a.to_canonical(),
            b.to_canonical()
        );
        // Canonical round-trip preserves equality.
        let rt = parse_json(&a.to_canonical()).unwrap();
        assert!(values_equal(&a, &rt));
    }
}

#[test]
fn property_transitivity_spot() {
    // Equal values form equivalence classes on the generated stream:
    // if a == b and b == c (same canonical form), then a == c.
    let mut rng = SplitMix64::new(77);
    let mut seen: Vec<Json> = Vec::new();
    for _ in 0..500 {
        let v = random_json(&mut rng, 0);
        for w in &seen {
            if values_equal(&v, w) {
                assert_eq!(v.to_canonical(), w.to_canonical(),
                           "equal values must share a canonical form");
            }
        }
        seen.push(v);
    }
}

#[test]
fn exhaustive_sequences_oracle() {
    // All sequences of length <= 3 over a 4-symbol alphabet, compared
    // against the direct element-wise oracle implemented inline.
    let atoms = [Json::Num(0.0), Json::Num(1.0), Json::Bool(true), Json::Null];
    let mut seqs: Vec<Vec<Json>> = vec![vec![]];
    for len in 1..=3 {
        let mut next = Vec::new();
        for s in &seqs {
            if s.len() == len - 1 {
                for a in &atoms {
                    let mut t = s.clone();
                    t.push(a.clone());
                    next.push(t);
                }
            }
        }
        seqs.extend(next);
    }
    let mut checked = 0;
    for a in &seqs {
        for b in &seqs {
            let got = values_equal(&Json::Arr(a.clone()), &Json::Arr(b.clone()));
            let want = a.len() == b.len()
                && a.iter().zip(b.iter()).all(|(x, y)| values_equal(x, y));
            assert_eq!(got, want, "sequence {:?} vs {:?}", a, b);
            checked += 1;
        }
    }
    assert!(checked > 1000, "checked {} sequence pairs", checked);
}

#[test]
fn soak_scale() {
    // 100k comparisons -- a scale the Python test does not reach.
    let mut rng = SplitMix64::new(20261123);
    let mut eq = 0usize;
    let t0 = std::time::Instant::now();
    for _ in 0..100_000 {
        let a = random_json(&mut rng, 0);
        let b = random_json(&mut rng, 0);
        if values_equal(&a, &b) {
            eq += 1;
        }
    }
    let dt = t0.elapsed();
    println!("soak: 100k comparisons, {} equal, {:?}", eq, dt);
    assert!(dt.as_secs() < 30, "soak too slow: {:?}", dt);
    // Grading soak: 20k verdicts, all-identical inputs must pass.
    let mut passes = 0;
    for _ in 0..20_000 {
        let k = 1 + rng.below(4);
        let expected: Vec<Json> =
            (0..k).map(|_| random_json(&mut rng, 0)).collect();
        let (pass, _) = grade_outputs(&expected.clone(), &expected);
        if pass {
            passes += 1;
        }
    }
    assert_eq!(passes, 20_000);
}

#[test]
fn thread_identity() {
    // The pure logic is stateless: 8 threads with the same seed must
    // produce identical digests.
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                let mut rng = SplitMix64::new(5);
                let mut bools = Vec::with_capacity(5000);
                for _ in 0..5000 {
                    let a = random_json(&mut rng, 0);
                    let b = random_json(&mut rng, 0);
                    bools.push(values_equal(&a, &b));
                }
                digest_of(bools)
            })
        })
        .collect();
    let digests: Vec<u64> =
        handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(digests.iter().all(|d| *d == digests[0]),
            "thread digests differ: {:?}", digests);
}
