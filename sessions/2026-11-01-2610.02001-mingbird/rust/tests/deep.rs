//! Deeper tests for the Mingbird harness trio.
//!
//! These go beyond the hand-picked cases in `verify` (and in the Python
//! `test_harness_cpu.py`):
//! - prefill: a deterministic property sweep over random schemas,
//!   budgets, system texts and unicode, asserting the budget invariant
//!   and caller non-mutation on every case;
//! - loop detection: exhaustive short sequences plus random sequences,
//!   each checked step-by-step against an independently written oracle
//!   for the streak/cycle definitions, plus a 200k-action soak;
//! - finish gate: random criterion sets checked against the exact
//!   expected failure list, in order;
//! - threads: the stateless parts give identical results in parallel.
//!
//! The RNG is a fixed-seed splitmix64, so every run is reproducible and
//! no external crate (and no network) is needed.

use arxiv_harness_rs::jsonutil::dumps;
use arxiv_harness_rs::{FinishGate, LoopDetector, PrefillBudget};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

// ── Deterministic RNG ─────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn usize(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
    fn text(&mut self, len: usize) -> String {
        let alphabet: Vec<char> = "abcXYZ 012.,;:éü😀中\n\t\"\\".chars().collect();
        (0..len)
            .map(|_| alphabet[self.usize(alphabet.len())])
            .collect()
    }
}

fn random_schema(rng: &mut Rng) -> Value {
    let mut map = Map::new();
    map.insert("name".to_string(), json!(format!("tool{}", rng.usize(5))));
    if rng.usize(4) != 0 {
        let dlen = rng.usize(600);
        map.insert("description".to_string(), json!(rng.text(dlen)));
    }
    let mut params = Map::new();
    for i in 0..rng.usize(4) {
        let mut field = Map::new();
        field.insert("type".to_string(), json!("string"));
        if rng.usize(3) != 0 {
            let dlen = rng.usize(200);
            let desc = rng.text(dlen);
            field.insert("description".to_string(), json!(desc));
        }
        params.insert(format!("p{i}"), Value::Object(field));
    }
    if rng.usize(3) == 0 {
        params.insert("examples".to_string(), json!([{"in": 1, "out": 2}]));
    }
    map.insert("parameters".to_string(), Value::Object(params));
    if rng.usize(2) == 0 {
        let examples: Vec<Value> = (0..rng.usize(6))
            .map(|i| json!({"in": {"path": format!("/f{i}")}, "out": "x"}))
            .collect();
        map.insert("examples".to_string(), Value::Array(examples));
    }
    Value::Object(map)
}

// ── Prefill ─────────────────────────────────────────────────────────────────

#[test]
fn prefill_property_random_schemas() {
    let mut rng = Rng::new(0xC0FF_EE);
    let budgets = [1usize, 2, 5, 10, 50, 100, 600, 1000, 4096, 10000];
    for _ in 0..2000 {
        let slen = rng.usize(200);
        let system = rng.text(slen);
        let schemas: Vec<Value> = (0..rng.usize(4)).map(|_| random_schema(&mut rng)).collect();
        let budget = if rng.usize(2) == 0 {
            budgets[rng.usize(budgets.len())]
        } else {
            rng.usize(10001)
        };
        let before: Vec<String> = schemas.iter().map(dumps).collect();
        let (prefill, used, compressed) =
            PrefillBudget::new(budget).build_prefill(&system, &schemas);
        // Caller schemas are never mutated.
        let after: Vec<String> = schemas.iter().map(dumps).collect();
        assert_eq!(before, after, "build_prefill mutated a caller schema");
        // Budget invariant (budget 0 is the documented degenerate case:
        // both Python and this port emit a 1-byte newline blob there).
        if budget >= 1 {
            assert!(used <= budget, "used {used} > budget {budget}");
            assert!(
                prefill.as_bytes().len() <= budget,
                "prefill text exceeds budget {budget}"
            );
        }
        // Compression flag and full-text form follow the definition.
        let mut parts = vec![system.clone()];
        parts.extend(schemas.iter().map(dumps));
        let full = parts.join("\n");
        if full.as_bytes().len() <= budget {
            assert!(!compressed, "fits but reported compressed");
            assert_eq!(used, full.as_bytes().len());
            assert_eq!(prefill, full);
        } else {
            assert!(compressed, "over budget but reported uncompressed");
        }
    }
}

#[test]
fn prefill_tiny_budgets_never_panic() {
    let big = vec![json!({
        "name": "read",
        "description": "x".repeat(500),
        "parameters": {"path": {"type": "string"}},
        "examples": [{"in": 1}]
    })];
    for budget in [0usize, 1, 2, 3, 10] {
        let (_text, used, _compressed) =
            PrefillBudget::new(budget).build_prefill("system text", &big);
        if budget >= 1 {
            assert!(used <= budget);
        }
    }
    // Empty input is the trivial fixed point.
    let (text, used, compressed) = PrefillBudget::new(600).build_prefill("", &[]);
    assert_eq!((text.as_str(), used, compressed), ("", 0, false));
}

// ── Loop detection ──────────────────────────────────────────────────────────

/// Independently written oracle for the streak/cycle definitions.
fn oracle(history: &[String], repeat_k: usize, max_cycle: usize) -> Option<String> {
    if history.len() >= repeat_k
        && history[history.len() - repeat_k..]
            .iter()
            .all(|s| s == &history[history.len() - 1])
    {
        return Some(format!(
            "loop: signature {} repeated {}x",
            history[history.len() - 1],
            repeat_k
        ));
    }
    for cyc in 2..=max_cycle {
        if history.len() >= 2 * cyc
            && history[history.len() - 2 * cyc..history.len() - cyc]
                == history[history.len() - cyc..]
        {
            return Some(format!("loop: {cyc}-step cycle repeated"));
        }
    }
    None
}

fn check_sequence(seq: &[(&str, Value)], repeat_k: usize, max_cycle: usize) {
    let mut det = LoopDetector::new(repeat_k, max_cycle);
    for (tool, args) in seq {
        let hit = det.observe(tool, args);
        let expected = oracle(&det.history, repeat_k, max_cycle);
        assert_eq!(hit, expected, "sequence {seq:?}");
        if hit.is_some() {
            break; // usage breaks on the first hit, as the harness does
        }
    }
}

#[test]
fn loop_exhaustive_short_sequences() {
    let a = ("a", json!({"x": 1}));
    let b = ("b", json!({"x": 2}));
    let mut count = 0;
    for len in 1..=7 {
        for mask in 0..(1usize << len) {
            let seq: Vec<(&str, Value)> = (0..len)
                .map(|i| {
                    if (mask >> i) & 1 == 0 {
                        (a.0, a.1.clone())
                    } else {
                        (b.0, b.1.clone())
                    }
                })
                .collect();
            check_sequence(&seq, 3, 4);
            count += 1;
        }
    }
    assert_eq!(count, 254); // 2 + 4 + ... + 128
}

#[test]
fn loop_random_sequences_match_oracle() {
    let mut rng = Rng::new(0x1000);
    let actions: Vec<(&str, Value)> = vec![
        ("read", json!({"path": "/a"})),
        ("read", json!({"path": "/b"})),
        ("write", json!({"path": "/a", "content": "x"})),
        ("read", json!({"path": "/a"})),
    ];
    for _ in 0..5000 {
        let len = rng.usize(21);
        let seq: Vec<(&str, Value)> = (0..len)
            .map(|_| {
                let (t, a) = &actions[rng.usize(actions.len())];
                (*t, a.clone())
            })
            .collect();
        check_sequence(&seq, 3, 4);
    }
}

#[test]
fn loop_soak_and_reset() {
    // 200k distinct actions: no false loop, history just grows.
    let mut det = LoopDetector::default();
    for i in 0..200_000 {
        let hit = det.observe("read", &json!({"n": i}));
        assert_eq!(hit, None, "false loop at distinct action {i}");
    }
    assert_eq!(det.history.len(), 200_000);

    // A pure looper breaks at exactly repeat_k, and reset restores it.
    let mut det = LoopDetector::new(3, 4);
    assert_eq!(det.observe("read", &json!({"path": "/a"})), None);
    assert_eq!(det.observe("read", &json!({"path": "/a"})), None);
    assert!(det.observe("read", &json!({"path": "/a"})).is_some());
    det.reset();
    assert!(det.history.is_empty());
    assert_eq!(det.observe("read", &json!({"path": "/a"})), None);
    assert_eq!(det.observe("read", &json!({"path": "/a"})), None);
    assert!(det.observe("read", &json!({"path": "/a"})).is_some());
}

// ── Finish gate ─────────────────────────────────────────────────────────────

#[test]
fn finish_gate_random_criterion_sets() {
    let mut rng = Rng::new(0x6A_76);
    for _ in 0..1000 {
        let n = rng.usize(6);
        let mut env: HashMap<String, String> = HashMap::new();
        for k in 0..5 {
            if rng.usize(2) == 0 {
                env.insert(
                    format!("k{k}"),
                    if rng.usize(2) == 0 { "yes" } else { "no" }.to_string(),
                );
            }
        }
        // Specs first, so the expected failure list is computed from the
        // spec, not from the gate under test.
        let mut specs: Vec<(String, String, bool)> = Vec::new();
        for j in 0..n {
            let key = format!("k{}", rng.usize(5));
            let want = rng.usize(2) == 0;
            specs.push((format!("c{j}"), key, want));
        }
        let expected_failed: Vec<String> = specs
            .iter()
            .filter(|(_, key, want)| (env.get(key).map(|v| v.as_str()) == Some("yes")) != *want)
            .map(|(name, _, _)| name.clone())
            .collect();
        let criteria: Vec<(String, Box<dyn Fn(&HashMap<String, String>) -> bool>)> = specs
            .iter()
            .map(|(name, key, want)| {
                let key = key.clone();
                let want = *want;
                (
                    name.clone(),
                    Box::new(move |e: &HashMap<String, String>| {
                        (e.get(&key).map(|v| v.as_str()) == Some("yes")) == want
                    }) as Box<dyn Fn(&HashMap<String, String>) -> bool>,
                )
            })
            .collect();
        let gate: FinishGate<HashMap<String, String>> = FinishGate::new("task", criteria);
        let (accepted, failed) = gate.review(&env);
        assert_eq!(accepted, expected_failed.is_empty());
        assert_eq!(failed, expected_failed, "failure list/order mismatch");
    }
}

// ── Combined invariants + threads ───────────────────────────────────────────

#[test]
fn combined_invariants_across_budgets_and_repeat_k() {
    let big = vec![json!({
        "name": "read",
        "description": "x".repeat(500),
        "parameters": {"path": {"type": "string"}},
        "examples": [{"in": 1}]
    })];
    for budget in [100usize, 600, 1000, 4096, 8000] {
        let (_t, used, _c) = PrefillBudget::new(budget).build_prefill("system", &big);
        assert!(used <= budget);
    }
    for repeat_k in [2usize, 3] {
        let mut det = LoopDetector::new(repeat_k, 4);
        for i in 1..=repeat_k {
            let hit = det.observe("read", &json!({"path": "/a"}));
            if i < repeat_k {
                assert_eq!(hit, None);
            } else {
                assert!(hit.is_some(), "repeat_k={repeat_k} did not fire");
            }
        }
    }
    // Interaction worth pinning: identical actions are also a 2-step
    // cycle, so with repeat_k=5 the cycle fires first, at step 4 —
    // exactly as the Python detector does (streak is checked first,
    // but at step 4 the streak has not yet reached 5).
    {
        let mut det = LoopDetector::new(5, 4);
        for _ in 0..3 {
            assert_eq!(det.observe("read", &json!({"path": "/a"})), None);
        }
        assert_eq!(
            det.observe("read", &json!({"path": "/a"})).as_deref(),
            Some("loop: 2-step cycle repeated")
        );
    }
    // Premature finish is rejected with exactly the missing criterion.
    let mut env: HashMap<String, String> = HashMap::new();
    env.insert("/out/log.txt".to_string(), "done\n".to_string());
    let gate: FinishGate<HashMap<String, String>> = FinishGate::new(
        "gate task",
        vec![
            (
                "log_artifact".to_string(),
                Box::new(|e: &HashMap<String, String>| {
                    e.get("/out/log.txt").map(|v| v.as_str()) == Some("done\n")
                }),
            ),
            (
                "status_artifact".to_string(),
                Box::new(|e: &HashMap<String, String>| {
                    e.get("/out/status.txt").map(|v| v.as_str()) == Some("verified\n")
                }),
            ),
        ],
    );
    assert_eq!(
        gate.review(&env),
        (false, vec!["status_artifact".to_string()])
    );
}

#[test]
fn stateless_parts_are_identical_across_threads() {
    let budget = PrefillBudget::new(600);
    let schemas = vec![json!({
        "name": "read",
        "description": "Read a file. ".repeat(40),
        "parameters": {"path": {"type": "string"}},
        "examples": [{"in": {"path": "/a"}, "out": "x"}]
    })];
    let expected = {
        let (t, used, compressed) = budget.build_prefill("system", &schemas);
        (used, compressed, t.len())
    };
    let sig_expected = LoopDetector::signature("read", &json!({"path": "/a"}));
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..8 {
            handles.push(s.spawn(|| {
                let mut out = Vec::new();
                for _ in 0..500 {
                    let (t, used, compressed) = budget.build_prefill("system", &schemas);
                    out.push((used, compressed, t.len()));
                    out.push((
                        LoopDetector::signature("read", &json!({"path": "/a"})).len(),
                        false,
                        0,
                    ));
                }
                out
            }));
        }
        for h in handles {
            let out = h.join().unwrap();
            assert_eq!(out.len(), 1000);
            assert!(out
                .iter()
                .step_by(2)
                .all(|&(u, c, l)| (u, c, l) == expected));
        }
    });
    assert_eq!(
        sig_expected,
        LoopDetector::signature("read", &json!({"path": "/a"}))
    );
}
