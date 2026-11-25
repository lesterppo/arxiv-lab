//! Rust tests for the CAPE port (arXiv:2611.31938).
//!
//! * unit tests for each mechanism (world effects, table, execution,
//!   initial policy, exact oracle, bounds);
//! * property sweep over random worlds (invariants from Prop. 1 and
//!   the compiler contract);
//! * exhaustive check on all 2-aspect worlds (CAPE value == exact
//!   optimum at full budget);
//! * soak at a scale beyond the Python test;
//! * thread identity (stateless: same world on 4 threads, identical);
//! * Python-vs-Rust parity sweep on cases generated from Python
//!   (`parity_cases.txt`): 0 mismatches required.

use cape::*;
use std::collections::BTreeSet;
use std::time::Instant;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn tiny_world() -> World {
    // 2 aspects x 2 values = 4 configs, all present.
    let aspects = vec![
        Aspect { name: "delim", values: &[",", ";"] },
        Aspect { name: "date", values: &["DMY", "MDY"] },
    ];
    let configs: Vec<Config> = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
    let programs = configs.clone();
    let mut probes = Vec::new();
    for pi in 0..2 {
        for ri in 0..2 {
            if ri == pi {
                continue;
            }
            for v in 0..2u8 {
                probes.push((pi, v, ri));
            }
        }
    }
    World { aspects, configs, programs, probes }
}

fn check_invariants(world: &World, r: &CapeResult, cap: usize, budget: usize) {
    assert_eq!(r.value, (r.completed.len(), -(r.total_calls as i64)), "value mismatch");
    assert!(r.r0.is_subset(&r.completed), "R0 not preserved");
    assert!(r.queries <= budget, "budget exceeded");
    if r.certified {
        assert_eq!(r.value, r.upper.unwrap(), "certified but value != upper");
    }
    let (done, calls) = evaluate_policy(world, &r.tree, cap);
    assert_eq!(done, r.completed, "evaluated completions != reported");
    assert_eq!(calls, r.total_calls, "evaluated calls != reported");
}

// ---------------------------------------------------------------------------
// unit tests
// ---------------------------------------------------------------------------

#[test]
fn world_program_commits_iff_full_match() {
    let w = tiny_world();
    let h: Config = vec![0, 1];
    let (resp, commit) = w.effect(&Act::Prog(vec![0, 1]), &h);
    assert!(commit && resp == Resp::Ok);
    // First mismatching aspect is named (delim is aspect 0).
    let (resp, commit) = w.effect(&Act::Prog(vec![1, 0]), &h);
    assert!(!commit && resp == Resp::Err(0));
    // Match on delim, mismatch on date -> err:date (aspect 1).
    let (resp, commit) = w.effect(&Act::Prog(vec![0, 0]), &h);
    assert!(!commit && resp == Resp::Err(1));
}

#[test]
fn world_probe_reveals_or_fails() {
    let w = tiny_world();
    let h: Config = vec![1, 0];
    // Probe primary=delim value ';' (1), reveal=date.
    let (resp, commit) = w.effect(&Act::Probe(0, 1, 1), &h);
    assert!(!commit && resp == Resp::Reveal(1, 0));
    // Primary mismatch -> error naming the primary aspect.
    let (resp, commit) = w.effect(&Act::Probe(0, 0, 1), &h);
    assert!(!commit && resp == Resp::Err(0));
}

#[test]
fn effect_table_acquire_closed_missing_complete() {
    let w = tiny_world();
    let mut t = EffectTable::new(&w);
    let a = Act::Prog(vec![0, 0]);
    let v: BTreeSet<Config> = w.configs.iter().cloned().collect();
    assert!(!t.closed(&a, &v));
    assert_eq!(t.missing(&a, &v).len(), 4);
    for h in &w.configs {
        t.acquire(&a, h);
    }
    assert!(t.closed(&a, &v));
    assert!(t.missing(&a, &v).is_empty());
    assert!(!t.complete());
    for act in w.actions() {
        for h in &w.configs {
            t.acquire(&act, h);
        }
    }
    assert!(t.complete());
}

#[test]
fn execute_policy_cap_and_missing_child() {
    let w = tiny_world();
    // cap 0: never acts.
    let (ok, calls) = execute_policy(&w, &initial_policy(&w, 2), &vec![0, 0], 0);
    assert!(!ok && calls == 0);
    // A tree that acts once then stops on any response.
    let tree = Node::Act(Act::Prog(vec![0, 0]), vec![]);
    let (ok, calls) = execute_policy(&w, &tree, &vec![0, 0], 3);
    assert!(ok && calls == 1);
    let (ok, calls) = execute_policy(&w, &tree, &vec![1, 1], 3);
    assert!(!ok && calls == 1); // error, no child -> stop
}

#[test]
fn initial_policy_r0_matches_evaluation() {
    let w = make_world(12, 4, 3);
    let pi0 = initial_policy(&w, 3);
    let (done, _) = evaluate_policy(&w, &pi0, 3);
    // pi0 tries the first `cap` programs in catalog order; its
    // completions are exactly the configs matching one of them.
    let expect: BTreeSet<Config> = w
        .configs
        .iter()
        .filter(|h| w.programs.iter().take(3).any(|p| p == *h))
        .cloned()
        .collect();
    assert_eq!(done, expect);
}

#[test]
fn exact_optimum_trivial_worlds() {
    // One config: one call completes it.
    let mut w = tiny_world();
    w.configs = vec![vec![0, 0]];
    let (val, tree) = exact_optimum(&w, 1);
    assert_eq!(val, Some((1, -1)));
    assert!(tree.is_some());
    // cap 0: nothing can complete, value (0, 0) via stop.
    let (val, _) = exact_optimum(&w, 0);
    assert_eq!(val, Some((0, 0)));
}

#[test]
fn error_feedback_is_deterministic_and_bounded() {
    let w = make_world(12, 4, 5);
    let (c1, t1) = error_feedback_policy(&w, 3);
    let (c2, t2) = error_feedback_policy(&w, 3);
    assert_eq!((c1.clone(), t1), (c2, t2));
    assert!(t1 <= 3 * w.configs.len());
    assert!(!c1.is_empty() || w.configs.is_empty());
}

// ---------------------------------------------------------------------------
// property sweep
// ---------------------------------------------------------------------------

#[test]
fn property_sweep_random_worlds() {
    let mut rng = Rng::new(20261125);
    for i in 0..200 {
        let ncfg = 4 + rng.below(9); // 4..12
        let cap = 1 + rng.below(3); // 1..3
        let budget = 50 + rng.below(551); // 50..600
        let seed = rng.next_u64();
        let w = make_world(ncfg, 4, seed);
        let r = compile_policy(&w, cap, budget);
        check_invariants(&w, &r, cap, budget);
        // CAPE completes at least the initial policy's set (R0).
        assert!(r.completed.len() >= r.r0.len());
        // Upper bound must dominate the witness value whenever both exist.
        if let Some(uv) = r.upper {
            assert!(uv >= r.value, "case {}: upper {:?} < value {:?}", i, uv, r.value);
        }
    }
}

// ---------------------------------------------------------------------------
// exhaustive: all 2-aspect worlds at full budget
// ---------------------------------------------------------------------------

#[test]
fn exhaustive_two_aspect_worlds() {
    let base = tiny_world();
    let all: Vec<Config> = base.configs.clone();
    // Every nonempty subset of the 4 configs.
    for mask in 1..16usize {
        let mut configs = Vec::new();
        for (i, c) in all.iter().enumerate() {
            if mask & (1 << i) != 0 {
                configs.push(c.clone());
            }
        }
        for cap in [1usize, 2] {
            let w = World {
                aspects: base.aspects.clone(),
                configs: configs.clone(),
                programs: base.programs.clone(),
                probes: base.probes.clone(),
            };
            let table_size = w.actions().len() * w.configs.len();
            let r = compile_policy(&w, cap, table_size);
            let (opt, _) = exact_optimum(&w, cap);
            assert_eq!(
                r.value,
                opt.unwrap(),
                "mask {} cap {}: cape {:?} != exact {:?}",
                mask,
                cap,
                r.value,
                opt
            );
            check_invariants(&w, &r, cap, table_size);
        }
    }
}

// ---------------------------------------------------------------------------
// soak + thread identity
// ---------------------------------------------------------------------------

#[test]
fn soak_many_small_worlds() {
    let t0 = Instant::now();
    let mut rng = Rng::new(777);
    let mut n_opt = 0;
    let n = 500;
    for _ in 0..n {
        let ncfg = 6 + rng.below(5); // 6..10
        let seed = rng.next_u64();
        let w = make_world(ncfg, 4, seed);
        let r = compile_policy(&w, 2, 200);
        check_invariants(&w, &r, 2, 200);
        let (opt, _) = exact_optimum(&w, 2);
        if r.value == opt.unwrap() {
            n_opt += 1;
        }
    }
    println!("soak: {} worlds in {:.1}s, optimal value in {}/{}", n, t0.elapsed().as_secs_f64(), n_opt, n);
    // Calibrated 2026-11-25: Python measures 449/500 = 0.898 on these
    // exact worlds (same xorshift stream, ../soak_py.py replicates the
    // Rust soak world-for-world), so 0.9 was stricter than CAPE itself
    // at this scale — the 51 misses all exhaust the 200-query budget
    // uncertified. The pre-fix index-order Rust measured 430/500 =
    // 0.860, so 0.89 still catches that regression.
    assert!(n_opt as f64 / n as f64 >= 0.89, "soak optimal rate too low");
}

#[test]
fn thread_identity_same_world() {
    let w = std::sync::Arc::new(make_world(12, 4, 42));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let w = w.clone();
            std::thread::spawn(move || {
                let r = compile_policy(&w, 3, 1000);
                (r.value, r.queries, r.certified, r.completed, r.total_calls)
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for r in &results[1..] {
        assert_eq!(r, &results[0], "same world must compile identically on every thread");
    }
}

// ---------------------------------------------------------------------------
// parity sweep vs Python (generated cases)
// ---------------------------------------------------------------------------

fn parse_world(n_aspects: usize, cfgs: &str) -> World {
    let aspects: Vec<Aspect> = if n_aspects == 4 {
        DEFAULT_ASPECTS.to_vec()
    } else {
        EXTENDED_ASPECTS.to_vec()
    };
    let configs: Vec<Config> = cfgs
        .split(',')
        .map(|s| s.bytes().map(|b| b - b'0').collect())
        .collect();
    let programs = {
        let mut out: Vec<Config> = vec![vec![]];
        for a in &aspects {
            let mut next = Vec::new();
            for prefix in &out {
                for v in 0..a.values.len() as u8 {
                    let mut c = prefix.clone();
                    c.push(v);
                    next.push(c);
                }
            }
            out = next;
        }
        out
    };
    let mut probes = Vec::new();
    for pi in 0..aspects.len() {
        for ri in 0..aspects.len() {
            if ri == pi {
                continue;
            }
            for v in 0..aspects[pi].values.len() as u8 {
                probes.push((pi, v, ri));
            }
        }
    }
    World { aspects, configs, programs, probes }
}

#[test]
fn parity_sweep_python_cases() {
    let data = include_str!("parity_cases.txt");
    let mut n = 0;
    let mut mismatches = 0;
    for (lineno, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(' ').collect();
        assert_eq!(f.len(), 12, "line {}: bad field count", lineno + 1);
        let n_aspects: usize = f[0].parse().unwrap();
        let cap: usize = f[1].parse().unwrap();
        let budget: usize = f[2].parse().unwrap();
        let w = parse_world(n_aspects, f[3]);
        let exp_comp: usize = f[4].parse().unwrap();
        let exp_negcalls: i64 = f[5].parse().unwrap();
        let exp_queries: usize = f[6].parse().unwrap();
        let exp_certified: bool = f[7] == "1";
        let exp_completed: usize = f[8].parse().unwrap();
        let exp_calls: usize = f[9].parse().unwrap();
        let exp_opt_comp: usize = f[10].parse().unwrap();
        let exp_opt_negcalls: i64 = f[11].parse().unwrap();
        let r = compile_policy(&w, cap, budget);
        let (opt, _) = exact_optimum(&w, cap);
        let ok = r.value == (exp_comp, exp_negcalls)
            && r.queries == exp_queries
            && r.certified == exp_certified
            && r.completed.len() == exp_completed
            && r.total_calls == exp_calls
            && opt.unwrap() == (exp_opt_comp, exp_opt_negcalls);
        if !ok {
            mismatches += 1;
            println!(
                "MISMATCH line {}: rust value={:?} queries={} cert={} completed={} calls={} opt={:?} | python value=({},{}) queries={} cert={} completed={} calls={} opt=({},{})",
                lineno + 1, r.value, r.queries, r.certified, r.completed.len(), r.total_calls, opt,
                exp_comp, exp_negcalls, exp_queries, exp_certified, exp_completed, exp_calls,
                exp_opt_comp, exp_opt_negcalls
            );
        }
        n += 1;
    }
    println!("parity sweep: {} cases, {} mismatches", n, mismatches);
    assert!(n >= 400, "parity file too small: {} cases", n);
    assert_eq!(mismatches, 0, "parity mismatches vs Python");
}
