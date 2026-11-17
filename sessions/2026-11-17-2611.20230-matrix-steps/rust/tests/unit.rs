//! Unit tests for the matrix-steps mechanism (one per mechanism).
//!
//! Run: cargo test --release --offline
//! (debug is fine too, but the capacity-adjacent tests are slower there.)

use matrix_steps_rs::*;

fn episode(
    id: &str, scope: &str, steps: Vec<&str>, outcome: &str, verification: &str,
) -> Episode {
    Episode::new(
        id,
        scope,
        "tyre",
        "change tyre",
        steps.into_iter().map(|s| s.to_string()).collect(),
        outcome,
        verification,
    )
}

#[test]
fn mt19937_test_vector() {
    let mut mt = Mt19937::from_int_seed(5489);
    let v: Vec<u32> = (0..5).map(|_| mt.uint32()).collect();
    assert_eq!(v, vec![3499211612, 581869302, 3890346734, 3586334585, 545404204]);
}

#[test]
fn fnv1a_is_stable_and_spreads() {
    assert_eq!(fnv1a_64("proc_0001"), fnv1a_64("proc_0001"));
    assert_ne!(fnv1a_64("proc_0001"), fnv1a_64("proc_0002"));
    assert_eq!(fnv1a_64(""), 0xCBF29CE484222325);
}

#[test]
fn rng_streams_are_deterministic() {
    let mut a = Rng::new(12345);
    let mut b = Rng::new(12345);
    for _ in 0..100 {
        assert_eq!(a.uint32(), b.uint32());
    }
    let mut c = Rng::new(54321);
    assert_ne!(a.uint32(), c.uint32());
}

#[test]
fn sign_vector_is_deterministic_and_balanced() {
    let s1 = procedure_sign_vector("proc_0001", 1024);
    let s2 = procedure_sign_vector("proc_0001", 1024);
    assert_eq!(s1, s2);
    assert_ne!(s1, procedure_sign_vector("proc_0002", 1024));
    assert!(s1.iter().all(|x| *x == 1.0 || *x == -1.0));
    let pos = s1.iter().filter(|x| **x > 0.0).count();
    assert!((400..624).contains(&pos), "signs should be roughly balanced, got {}", pos);
}

#[test]
fn embed_is_deterministic_and_normalized() {
    let mut e = HashEmbedder::new();
    let v1 = e.embed("change the tyre on the bicycle");
    let v2 = e.embed("change the tyre on the bicycle");
    assert_eq!(v1, v2);
    let n: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((n - 1.0).abs() < 1e-12);
    // Stopwords only -> zero vector.
    assert!(e.embed("the a an on in at").iter().all(|x| *x == 0.0));
    // Disjoint texts are near-orthogonal.
    let a = e.embed("w0001 w0002 w0003 w0004 w0005 w0006");
    let b = e.embed("w1001 w1002 w1003 w1004 w1005 w1006");
    assert!(cosine(&a, &b).abs() < 0.2);
}

#[test]
fn delta_write_then_read_recovers_value() {
    let mut m = DeltaMatrix::new(64, None, "sign", 777);
    let mut rng = Rng::new(5);
    let k: Vec<f32> = {
        let mut v: Vec<f32> = (0..64).map(|_| rng.uniform() as f32).collect();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    };
    let v: Vec<f32> = (0..64).map(|_| rng.uniform() as f32).collect();
    m.write(&k, &v, 1.0, 1.0);
    let got = m.read(&k);
    for (a, b) in got.iter().zip(v.iter()) {
        assert!((*a - *b).abs() < 1e-5, "read-back must recover the value");
    }
}

#[test]
fn widening_sign_vs_linear() {
    let m = DeltaMatrix::new(16, Some(64), "sign", 777);
    let k: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) / 16.0).collect();
    let w = m.widen(&k);
    assert_eq!(w.len(), 64);
    // Sign widening: every component is +/- 1/sqrt(64).
    assert!(w.iter().all(|x| (*x - 0.125).abs() < 1e-6 || (*x + 0.125).abs() < 1e-6));
    let m2 = DeltaMatrix::new(16, Some(64), "linear", 777);
    let w2 = m2.widen(&k);
    let n: f32 = w2.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n - 1.0).abs() < 1e-5);
}

#[test]
fn writer_limit_is_enforced() {
    let m = DeltaMatrix::new(1024, None, "sign", 777);
    assert!(m.check_limit(200).is_ok());
    assert!(m.check_limit(201).is_err());
    let m4 = DeltaMatrix::new(1024, Some(4096), "sign", 777);
    assert!(m4.check_limit(500).is_ok());
    assert!(m4.check_limit(501).is_err());
}

#[test]
fn write_gate_rules() {
    let ok = episode("a", "alice", vec!["x"], "success", "external_check");
    assert_eq!(recorded_outcome(&ok).as_deref(), Some("success"));
    let claimed = episode("b", "alice", vec!["x"], "success", "model_claim");
    assert_eq!(recorded_outcome(&claimed), None);
    let mut failed = episode("c", "alice", vec!["x"], "success", "tool_result");
    failed.tool_failed = true;
    failed.agent_claimed_done = true;
    assert_eq!(recorded_outcome(&failed).as_deref(), Some("failure"));
    let mut over = episode("d", "alice", vec!["x"], "success", "tool_result");
    over.person_verdict = Some("failure".to_string());
    assert_eq!(recorded_outcome(&over).as_deref(), Some("failure"));
}

#[test]
fn consolidation_minimums_and_confidence() {
    let seq = vec!["deflate", "lift tyre", "fit tube"];
    let mk = |i: usize, outcome: &str| {
        episode(&format!("c{}", i), "alice", seq.clone(), outcome, "external_check")
    };
    let eps: Vec<Episode> = (0..4).map(|i| mk(i, "success")).collect();
    assert!(consolidate_episodes(&eps, 5, 3, 0.6).is_none());
    let mut eps: Vec<Episode> = (0..2).map(|i| mk(i, "success")).collect();
    eps.extend((2..5).map(|i| mk(i, "failure")));
    assert!(consolidate_episodes(&eps, 5, 3, 0.6).is_none());
    let mut eps: Vec<Episode> = (0..4).map(|i| mk(i, "success")).collect();
    eps.extend((4..6).map(|i| mk(i, "failure")));
    let rec = consolidate_episodes(&eps, 5, 3, 0.6).unwrap();
    assert!((rec.confidence - confidence(4, 6)).abs() < 1e-12);
    assert_eq!(rec.confidence, 5.0 / 8.0);
}

#[test]
fn z_test_one_sided_behaves() {
    // Identical arms: p ~ 0.5.
    let p = z_test_one_sided_p(20, 40, 20, 40);
    assert!((p - 0.5).abs() < 1e-6);
    // Candidate clearly better: tiny p.
    assert!(z_test_one_sided_p(32, 40, 12, 40) < 0.001);
    // Candidate worse: p near 1.
    assert!(z_test_one_sided_p(12, 40, 32, 40) > 0.999);
}

#[test]
fn demotion_restores_predecessor() {
    let mut mem = EpisodicMemory::new("alice", EMBED_DIM, 0.25);
    let prev = ProcedureRecord::new(
        "prev", "safe place", "place part slowly", "when placing",
        vec![Step::new("lower slowly")],
    );
    let mut bad = ProcedureRecord::new(
        "bad", "fast place", "place part fast", "when placing",
        vec![Step::new("drop quickly")],
    );
    bad.predecessor = Some("prev".to_string());
    mem.add_record(prev);
    mem.add_record(bad);
    mem.set_default(Some("bad"), Some("prev"));
    mem.rebuild_matrix(1, true);
    assert_eq!(mem.demote_on_guard_trip("bad").as_deref(), Some("prev"));
    assert_eq!(mem.default_id.as_deref(), Some("prev"));
    assert_eq!(mem.records.get("bad").unwrap().status, "demoted");
    // A non-default cannot be demoted.
    assert_eq!(mem.demote_on_guard_trip("prev"), None);
}

#[test]
fn scope_isolation_in_memory() {
    let mut a = EpisodicMemory::new("alice", EMBED_DIM, 0.25);
    let mut b = EpisodicMemory::new("bob", EMBED_DIM, 0.25);
    let e = episode("x", "bob", vec!["s"], "success", "external_check");
    b.record_episode(e);
    assert!(a.verified_episodes(None).is_empty());
    assert_eq!(b.verified_episodes(None).len(), 1);
}

#[test]
fn synthetic_procedures_have_shared_steps() {
    let procs = generate_synthetic_procedures(50, 20261117, 3, 8, 0.15);
    assert_eq!(procs.len(), 50);
    assert!(procs.iter().all(|p| (3..=8).contains(&p.steps.len())));
    // Step texts repeat across procedures (the shared 15%).
    let mut counts = std::collections::HashMap::new();
    for p in &procs {
        for s in &p.steps {
            *counts.entry(s.text().to_string()).or_insert(0) += 1;
        }
    }
    assert!(counts.values().any(|c| *c > 1), "some step text must be shared");
}
