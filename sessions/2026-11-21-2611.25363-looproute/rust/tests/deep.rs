//! Deep tests for the LoopRoute Rust port: property sweeps over random
//! task specs and thresholds, exhaustive + random sequences checked
//! against the BFS oracle, a soak at a scale the Python test does not
//! reach, and thread identity (parallel == serial).

use looproute_rs::{
    oracle_assistant, plan_to_goal, run_episode, run_experiment, Aggregate,
    FastPolicy, KitchenWorld, LoopRouter, PyRandom,
};
use std::time::Instant;

/// Check an oracle plan from a given state: it must be legal, cook the
/// meal, and take exactly `len(plan)` steps.
fn check_plan_from(env: &KitchenWorld, label: &str) {
    let plan = plan_to_goal(env);
    assert!(!plan.is_empty(), "{label}: oracle plan must exist");
    assert_eq!(plan.last().unwrap(), "cook", "{label}: plan ends with cook");
    let mut probe = env.clone_task();
    probe.set_state(&env.state());
    for act in &plan {
        let (_, _, _, ok) = probe.step(act);
        assert!(ok, "{label}: illegal plan step {act}");
    }
    assert_eq!(
        probe.steps,
        plan.len(),
        "{label}: executed steps must equal plan length"
    );
    assert!(probe.recipe_collected(), "{label}: plan must collect all");
}

// ---------------------------------------------------------------------------
// Property sweeps: thresholds K=2,3,4 over 120 random seeded tasks
// ---------------------------------------------------------------------------

#[test]
fn property_routed_never_worse_than_fast() {
    let policy = FastPolicy;
    for threshold in [2usize, 3, 4] {
        let mut fast_wins = 0usize;
        let mut routed_wins = 0usize;
        let mut total_calls = 0usize;
        let mut max_calls = 0usize;
        for seed in 7000..7120u64 {
            let locked = seed % 2 == 0;
            let mut env = KitchenWorld::new(seed, 3, locked);
            let f = run_episode(&mut env, &policy, None, None, 60);
            let mut router = LoopRouter::new(threshold);
            let mut env2 = KitchenWorld::new(seed, 3, locked);
            let r = run_episode(
                &mut env2,
                &policy,
                Some(&mut router),
                Some(oracle_assistant),
                60,
            );
            // Per-seed monotonicity: routing can only rescue, never
            // break, a task the fast policy already solves.
            assert!(
                !f.success || r.success,
                "K={threshold} seed {seed}: fast solved but routed failed"
            );
            // An assist plan runs to completion, so routed must solve.
            assert!(r.success, "K={threshold} seed {seed}: routed must solve");
            assert!(r.llm_calls <= 3, "K={threshold} seed {seed}: too many calls");
            fast_wins += usize::from(f.success);
            routed_wins += usize::from(r.success);
            total_calls += r.llm_calls;
            max_calls = max_calls.max(r.llm_calls);
        }
        println!(
            "K={threshold}: fast {fast_wins}/120 routed {routed_wins}/120 \
             mean_calls {:.2} max_calls {max_calls}",
            total_calls as f64 / 120.0
        );
        assert_eq!(routed_wins, 120);
        assert_eq!(fast_wins, 60, "even seeds locked, odd seeds unlocked");
    }
}

#[test]
fn property_no_false_triggers_on_unlocked_tasks() {
    let policy = FastPolicy;
    let mut assists = 0usize;
    for seed in 3000..3100u64 {
        let mut router = LoopRouter::new(2);
        let mut env = KitchenWorld::new(seed, 3, false);
        let res = run_episode(
            &mut env,
            &policy,
            Some(&mut router),
            Some(oracle_assistant),
            60,
        );
        assert!(res.success, "seed {seed} unlocked must solve");
        assists += res.assists;
    }
    assert_eq!(assists, 0, "100 unlocked tasks must trigger 0 assists");
}

#[test]
fn property_varying_recipe_sizes() {
    // n = 1..5 ingredients, locked and unlocked, K=2.
    let policy = FastPolicy;
    for n in 1..=5usize {
        for seed in 9000..9010u64 {
            for locked in [true, false] {
                let mut router = LoopRouter::new(2);
                let mut env = KitchenWorld::new(seed, n, locked);
                let res = run_episode(
                    &mut env,
                    &policy,
                    Some(&mut router),
                    Some(oracle_assistant),
                    60,
                );
                assert!(
                    res.success,
                    "n={n} seed {seed} locked={locked}: routed must solve"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Oracle checks: exhaustive short sequences + random walks
// ---------------------------------------------------------------------------

#[test]
fn exhaustive_sequences_checked_against_oracle() {
    // All action sequences from the start state of one locked and one
    // unlocked task (the reachable state graph is finite: 6 rooms x
    // collected subsets x chest x key, so the DFS terminates); every
    // reached state must have a legal, complete oracle plan that
    // executes in len(plan) steps.
    // legal, complete oracle plan that executes in len(plan) steps.
    for seed in [42u64, 43] {
        let locked = seed % 2 == 0;
        let base = KitchenWorld::new(seed, 3, locked);
        let mut states = 0usize;
        // DFS over the whole reachable state graph, tracking visited
        // states (finite: rooms x collected subsets x chest x key).
        let mut seen = std::collections::HashSet::new();
        seen.insert(base.state());
        let mut stack: Vec<KitchenWorld> = vec![base];
        while let Some(env) = stack.pop() {
            check_plan_from(&env, &format!("seed {seed}"));
            states += 1;
            for act in env.legal_actions() {
                let mut next = env.clone_task();
                next.set_state(&env.state());
                let before = next.state();
                let (_, _, done, ok) = next.step(&act);
                if !ok {
                    // A legal action can still fail (e.g. "open
                    // chest" without the key): state must be
                    // unchanged, and there is no new state to visit.
                    assert_eq!(next.state(), before);
                    continue;
                }
                if done {
                    continue;
                }
                if seen.insert(next.state()) {
                    stack.push(next);
                }
            }
        }
        println!("seed {seed}: {states} states checked against the oracle");
        assert!(states > 80, "seed {seed}: too few states reached");
    }
}

#[test]
fn random_walks_checked_against_oracle() {
    // 200 random walks (random legal actions) on random seeded tasks;
    // every reached state must have a legal, complete oracle plan.
    let mut rng = PyRandom::new(20261121);
    let mut checked = 0usize;
    for i in 0..200 {
        let seed = 10000 + rng.randbelow(1000) as u64;
        let locked = rng.randbelow(2) == 0;
        let n = 1 + rng.randbelow(5);
        let mut env = KitchenWorld::new(seed, n, locked);
        let walk_len = rng.randbelow(16);
        for _ in 0..walk_len {
            let acts = env.legal_actions();
            let act = acts[rng.randbelow(acts.len())].clone();
            let before = env.state();
            let (_, _, done, ok) = env.step(&act);
            if !ok {
                // Legal but failing (e.g. "open chest" without the key):
                // a no-op, walk continues from the same state.
                assert_eq!(env.state(), before);
                continue;
            }
            if done {
                break;
            }
        }
        if env.steps > 0 && !env.recipe_collected() {
            check_plan_from(&env, &format!("walk {i} seed {seed}"));
            checked += 1;
        }
    }
    println!("random walks: {checked} non-terminal states checked");
    assert!(checked >= 150, "too few non-terminal walk states");
}

// ---------------------------------------------------------------------------
// Soak: 600 seeded tasks (Python test covers 60)
// ---------------------------------------------------------------------------

#[test]
fn soak_600_seeds() {
    let t0 = Instant::now();
    let seeds: Vec<u64> = (5000..5600).collect();
    let (fast, routed) = run_experiment(&seeds, 2);
    let (_, routed3) = run_experiment(&seeds, 3);
    let (_, routed4) = run_experiment(&seeds, 4);
    let elapsed_ms = t0.elapsed().as_millis();
    println!(
        "soak 600 seeds: fast success={:.3} steps={:.1} | K=2 routed \
         success={:.3} steps={:.1} assists={:.2} calls={:.2} | K=3 \
         success={:.3} | K=4 success={:.3} | {:.0} ms",
        fast.success_rate,
        fast.mean_steps,
        routed.success_rate,
        routed.mean_steps,
        routed.mean_assists,
        routed.mean_llm_calls,
        routed3.success_rate,
        routed4.success_rate,
        elapsed_ms
    );
    assert_eq!(fast.success_rate, 0.5, "even seeds locked, odd unlocked");
    assert_eq!(routed.success_rate, 1.0);
    assert_eq!(routed3.success_rate, 1.0);
    assert_eq!(routed4.success_rate, 1.0);
    assert!(routed.mean_llm_calls <= 1.0);
    assert!(elapsed_ms < 60_000, "soak must finish in under a minute");

    // 200-step cap check: a locked task with a generous cap still
    // terminates within the cap and succeeds.
    let policy = FastPolicy;
    let mut router = LoopRouter::new(2);
    let mut env = KitchenWorld::new(4242, 3, true);
    let res = run_episode(
        &mut env,
        &policy,
        Some(&mut router),
        Some(oracle_assistant),
        200,
    );
    assert!(res.success);
    assert!(res.steps <= 200);
    assert!(res.steps < 60, "oracle rescue finishes well under 60 steps");
}

// ---------------------------------------------------------------------------
// Thread identity: parallel == serial (router/env are per-episode
// state; nothing is shared between episodes)
// ---------------------------------------------------------------------------

#[test]
fn thread_identity_parallel_equals_serial() {
    let seeds: Vec<u64> = (1000..1060).collect();
    let serial = run_experiment(&seeds, 2);
    // Four shards, one thread each.
    let results: Vec<(Aggregate, Aggregate)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for shard in seeds.chunks(15) {
            let shard = shard.to_vec();
            handles.push(s.spawn(move || run_experiment(&shard, 2)));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("shard thread"))
            .collect()
    });
    // Re-aggregate the shards the same way run_experiment does.
    let combine = |idx: usize| -> Aggregate {
        let n: usize = results.iter().map(|r| r.0.n).sum();
        let mean = |f: fn(&Aggregate) -> f64| {
            results
                .iter()
                .map(|r| f(if idx == 0 { &r.0 } else { &r.1 }) * r.0.n as f64)
                .sum::<f64>()
                / n as f64
        };
        Aggregate {
            n,
            success_rate: mean(|a| a.success_rate),
            mean_steps: mean(|a| a.mean_steps),
            mean_assists: mean(|a| a.mean_assists),
            mean_llm_calls: mean(|a| a.mean_llm_calls),
        }
    };
    assert_eq!(combine(0), serial.0, "fast-only parallel == serial");
    assert_eq!(combine(1), serial.1, "routed parallel == serial");
}
