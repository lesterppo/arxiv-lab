//! Unit tests for the SAT mechanism — one test per mechanism.

use studentteaching::*;

#[test]
fn splitmix64_matches_python_vectors() {
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next_u64(), 16294208416658607535);
    assert_eq!(r.next_u64(), 7960286522194355700);
    assert_eq!(r.next_u64(), 487617019471545679);
    let mut r = SplitMix64::new(20261111);
    assert_eq!(r.next_u64(), 18157476869487094667);
    assert_eq!(r.next_u64(), 5560150052093433963);
    assert_eq!(r.next_u64(), 16154423555003978995);
}

#[test]
fn splitmix64_randbelow_matches_python() {
    assert_eq!(SplitMix64::new(42).randbelow(100), 13);
    assert_eq!(SplitMix64::new(42).randbelow(3), 1);
}

#[test]
fn splitmix64_streams_are_deterministic() {
    let mut a = SplitMix64::new(123);
    let mut b = SplitMix64::new(123);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    let mut c = SplitMix64::new(124);
    assert_ne!(a.next_u64(), c.next_u64());
}

#[test]
fn gen_tasks_deterministic() {
    let a = gen_tasks(20261111, 20);
    let b = gen_tasks(20261111, 20);
    for (ta, tb) in a.iter().zip(b.iter()) {
        assert_eq!(ta.ops, tb.ops);
        assert_eq!(ta.gold, tb.gold);
        assert_eq!(ta.traps, tb.traps);
    }
    let c = gen_tasks(7, 20);
    assert_ne!(a[0].gold, c[0].gold);
}

#[test]
fn gen_tasks_shape_invariants() {
    for t in gen_tasks(99, 100) {
        // ops = n_ops (4..7) + answer + up to 2 extra for the stale trap
        assert!((5..=10).contains(&t.ops.len()));
        assert!(matches!(t.ops.last(), Some(Op::Answer { .. })));
        for trap in &t.traps {
            assert!(["missing", "flaky", "stale", "search_sub", "big_mult"].contains(&trap.as_str()));
        }
    }
}

#[test]
fn probe_tasks_isolate_each_limitation() {
    // A student with ONLY lim L fails probe L; a student with no
    // limitations passes every probe; a student with skill L passes probe L.
    for lim in LIMITATIONS {
        let probe = gen_probe_task(lim);
        let with_lim = StudentAgent::new(&[lim], &[]);
        let (ok, _, prints) = with_lim.solve(&probe);
        assert!(!ok, "probe {lim} should fail a student with {lim}");
        assert!(prints.iter().any(|p| p == fingerprint(lim)));
        let clean = StudentAgent::new(&[], &[]);
        let (ok, _, _) = clean.solve(&probe);
        assert!(ok, "probe {lim} should pass a clean student");
        let skilled = StudentAgent::new(&[lim], &[lim]);
        let (ok, _, _) = skilled.solve(&probe);
        assert!(ok, "probe {lim} should pass a student with the {lim} skill");
    }
}

#[test]
fn all_skills_student_solves_everything_within_budget() {
    // An all-limitations student with all skills removes every deviation,
    // so it can only fail on budget (5 skills -> budget 9; long tasks need
    // up to 15 calls). Every failure must be budget_exhausted, never a
    // wrong answer.
    let s = StudentAgent::new(&["L1", "L2", "L3", "L4", "L5"], &["L1", "L2", "L3", "L4", "L5"]);
    for seed in [1, 7, 42, 20261111] {
        let tasks = gen_tasks(seed, 50);
        let (acc, _) = run_batch(&s, &tasks);
        // Budget 9 loses every task needing >=10 calls (~16-28% depending
        // on the seed's task-length mix); the bound is deliberately loose,
        // the budget-only-failures check below is the strong assertion.
        assert!(acc >= 0.65, "seed {seed} acc {acc}");
        for t in &tasks {
            let (ok, _, prints) = s.solve(t);
            if !ok {
                assert!(prints.iter().any(|p| p == "budget_exhausted"), "seed {seed}: {:?}", prints);
            }
        }
    }
}

#[test]
fn budget_arithmetic() {
    assert_eq!(StudentAgent::new(&[], &[]).budget(), 14);
    assert_eq!(StudentAgent::new(&["L1"], &["L1"]).budget(), 13);
    assert_eq!(StudentAgent::new(&[], &["L1", "L2", "L3", "L4", "L5"]).budget(), 9);
}

#[test]
fn admission_gate_admits_real_and_rejects_spurious() {
    let dev = gen_tasks(20261111, 120);
    let teacher = Teacher::default();
    let student_a = StudentAgent::new(&["L1", "L3", "L5"], &[]);
    // L3 is a real limitation of student A: the skill must be admitted.
    let matched: Vec<_> = dev.iter().filter(|t| t.traps.contains("stale")).take(12).cloned().collect();
    let rep = teacher.admit(&student_a, "L3", &matched);
    assert!(rep.admitted && rep.delta >= 0.15, "delta {}", rep.delta);
    // L2 is NOT a limitation of student A: the skill must be rejected.
    let matched2: Vec<_> = dev.iter().filter(|t| t.traps.contains("search_sub")).take(12).cloned().collect();
    let rep2 = teacher.admit(&student_a, "L2", &matched2);
    assert!(!rep2.admitted, "delta {}", rep2.delta);
}

#[test]
fn skill_texts_present() {
    for lim in LIMITATIONS {
        assert!(!skill_text(lim).is_empty());
    }
}

#[test]
fn teacher_profile_finds_true_limitations() {
    let dev = gen_tasks(20261111, 120);
    let teacher = Teacher::default();
    let student_a = StudentAgent::new(&["L1", "L3", "L5"], &[]);
    let (rates, hyps) = teacher.profile(&student_a, &dev);
    for lim in ["L1", "L3", "L5"] {
        assert!(hyps.contains(&lim.to_string()), "hypotheses {:?} rates {:?}", hyps, rates);
        assert!(rates[lim] >= 0.05);
    }
}
