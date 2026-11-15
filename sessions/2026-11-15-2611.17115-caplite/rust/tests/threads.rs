//! Thread identity: the mechanism is stateless (no globals, no shared
//! mutable state), so the same spec / battery must give bit-identical
//! outputs on every thread, and a battery split across threads must
//! equal the single-threaded result.

use caplite::*;
use std::collections::BTreeMap;

#[test]
fn same_spec_identical_across_threads() {
    let spec = build_episode(7 * 1000003 + 5, true, false);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(move || {
                let mut out = BTreeMap::new();
                for arm in Arm::all() {
                    out.insert(arm.name(), run_episode(&spec, arm));
                }
                out
            })
        })
        .collect();
    let first = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect::<Vec<_>>();
    for other in &first[1..] {
        assert_eq!(
            &first[0], other,
            "same spec gave different outcomes across threads"
        );
    }
}

#[test]
fn battery_split_across_threads_equals_single_thread() {
    // Four disjoint seed ranges, each a run_battery on its own thread.
    let handles: Vec<_> = (0..4u64)
        .map(|k| std::thread::spawn(move || run_battery(100 + k, 75, 40, 0.1)))
        .collect();
    let threaded: Vec<Battery> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for (k, b) in threaded.iter().enumerate() {
        let single = run_battery(100 + k as u64, 75, 40, 0.1);
        assert_eq!(b.caplite, single.caplite, "caplite mismatch on shard {}", k);
        assert_eq!(b.camel, single.camel, "camel mismatch on shard {}", k);
        assert_eq!(
            b.undefended, single.undefended,
            "undefended mismatch on shard {}",
            k
        );
    }
}

#[test]
fn soak_slice_identical_across_threads() {
    let handles: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(|| run_soak(2000, 7)))
        .collect();
    let results: Vec<Battery> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for other in &results[1..] {
        assert_eq!(results[0].caplite, other.caplite);
        assert_eq!(results[0].camel, other.camel);
        assert_eq!(results[0].undefended, other.undefended);
    }
}
