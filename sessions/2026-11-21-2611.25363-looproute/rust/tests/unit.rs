//! Unit tests for the LoopRoute Rust port: RNG vs CPython, seeded task
//! generation, env transitions, the fast policy's flaw, the router, the
//! BFS planner, and spec round-trips.

use looproute_rs::{
    bfs_path, oracle_assistant, parse_spec, plan_to_goal, spec_of_env,
    FastPolicy, KitchenWorld, LoopRouter, PyRandom,
};

// ---------------------------------------------------------------------------
// CPython-compatible RNG (values from CPython 3.12 `random.Random`)
// ---------------------------------------------------------------------------

#[test]
fn pyrandom_matches_cpython() {
    let expected: [[f64; 3]; 6] = [
        [
            0.8444218515250481,
            0.7579544029403025,
            0.420571580830845,
        ],
        [
            0.13436424411240122,
            0.8474337369372327,
            0.763774618976614,
        ],
        [
            0.9560342718892494,
            0.9478274870593494,
            0.05655136772680869,
        ],
        [
            0.23796462709189137,
            0.5442292252959519,
            0.36995516654807925,
        ],
        [
            0.23604808973743452,
            0.1031660342307158,
            0.396058242610681,
        ],
        [
            0.6229016948897019,
            0.7417869892607294,
            0.7951935655656966,
        ],
    ];
    for (seed, want) in expected.iter().enumerate() {
        let mut rng = PyRandom::new(seed as u64);
        for (i, w) in want.iter().enumerate() {
            assert_eq!(rng.random(), *w, "seed {seed} draw {i}");
        }
    }
}

#[test]
fn pyrandom_sample_matches_cpython() {
    // CPython: Random(1000).sample(INGREDIENTS, 3) == the indices of
    // ['basil', 'onion', 'chili'].
    let mut rng = PyRandom::new(1000);
    assert_eq!(rng.sample_indices(10, 3), vec![6, 1, 9]);
}

// ---------------------------------------------------------------------------
// Seeded task generation (CPython-verified draw order)
// ---------------------------------------------------------------------------

#[test]
fn seeded_generation_matches_python() {
    // Python: KitchenWorld(1000, locked=True) has recipe
    // ['basil', 'onion', 'chili'], rooms shuffled to
    // ['hallway', 'bedroom', 'garden', 'pantry', 'cellar'], key in
    // 'cellar' (choice over [garden, pantry, cellar]).
    let env = KitchenWorld::new(1000, 3, true);
    assert_eq!(
        env.recipe
            .iter()
            .map(|&i| looproute_rs::INGREDIENTS[i])
            .collect::<Vec<_>>(),
        vec!["basil", "onion", "chili"]
    );
    assert_eq!(env.placement, vec![5, 4, 0]); // hallway, bedroom, kitchen
    assert!(env.in_chest[2]);
    assert!(!env.in_chest[0] && !env.in_chest[1]);
    assert_eq!(env.key_room, 3); // cellar
    assert!(env.locked);
}

#[test]
fn bfs_path_values() {
    assert_eq!(bfs_path(0, 0), Some(vec![0]));
    // kitchen -> hallway -> bedroom beats kitchen -> pantry -> cellar
    // -> bedroom (adjacency order decides the tie).
    assert_eq!(bfs_path(0, 4), Some(vec![0, 5, 4]));
    assert_eq!(bfs_path(1, 2), Some(vec![1, 0, 2]));
    assert_eq!(bfs_path(3, 5), Some(vec![3, 4, 5]));
}

// ---------------------------------------------------------------------------
// Env transitions
// ---------------------------------------------------------------------------

fn env_1000_locked() -> KitchenWorld {
    KitchenWorld::new(1000, 3, true)
}

#[test]
fn env_reset_and_go() {
    let mut env = env_1000_locked();
    assert_eq!(env.room, 0);
    // kitchen -> pantry is adjacent; pantry -> garden is not.
    let (obs, _, done, ok) = env.step("go pantry");
    assert!(ok && !done);
    assert_eq!(obs, "You go to the pantry.");
    assert_eq!(env.room, 1);
    let (_, _, _, ok) = env.step("go garden");
    assert!(!ok, "pantry -> garden is not an edge");
    assert_eq!(env.room, 1);
    // pantry -> cellar is an edge.
    let (_, _, _, ok) = env.step("go cellar");
    assert!(ok);
    assert_eq!(env.room, 3);
    let (_, _, _, ok) = env.step("dance");
    assert!(!ok);
}

#[test]
fn env_take_key_and_open_chest() {
    let mut env = env_1000_locked();
    // Key is in the cellar: kitchen -> pantry -> cellar.
    env.step("go pantry");
    env.step("go cellar");
    assert!(env.legal_actions().contains(&"take brass key".to_string()));
    let (obs, _, _, ok) = env.step("take brass key");
    assert!(ok);
    assert_eq!(obs, "You take the brass key.");
    assert!(env.key_taken);
    // Back to the kitchen; the chest ingredient is 'chili'.
    env.step("go pantry");
    env.step("go kitchen");
    let (obs, _, _, ok) = env.step("open chest");
    assert!(ok);
    assert_eq!(obs, "You unlock and open the chest.");
    assert!(env.chest_open);
    let (_, _, _, ok) = env.step("open chest");
    assert!(!ok, "already open");
}

#[test]
fn env_locked_chest_without_key() {
    let mut env = env_1000_locked();
    let (obs, _, _, ok) = env.step("open chest");
    assert!(!ok);
    assert_eq!(obs, "The chest is locked. You need the brass key.");
    let (obs, _, _, ok) = env.step("take chili");
    assert!(!ok);
    assert_eq!(obs, "The chest is locked.");
    // The chest ingredient is not a legal action while locked.
    assert!(!env.legal_actions().contains(&"take chili".to_string()));
}

#[test]
fn env_cook_requires_all_ingredients() {
    let mut env = KitchenWorld::new(1001, 3, false);
    let (_, _, _, ok) = env.step("cook");
    assert!(!ok, "cannot cook with nothing collected");
    // Solve it with the oracle plan and check the final observation.
    let plan = plan_to_goal(&env);
    env.reset();
    let mut done = false;
    let mut last_obs = String::new();
    for act in &plan {
        let (obs, reward, d, ok) = env.step(act);
        assert!(ok, "oracle plan step {act} must be legal");
        assert_eq!(reward, if d { 1 } else { 0 });
        last_obs = obs;
        done = d;
    }
    assert!(done);
    assert_eq!(last_obs, "You cook the meal. Task complete!");
    assert_eq!(env.steps, plan.len());
}

#[test]
fn env_describe_mentions_task() {
    let env = env_1000_locked();
    let d = env.describe();
    assert!(d.contains("You are in the kitchen."));
    assert!(d.contains("Recipe (in order): basil, onion, chili."));
    assert!(d.contains("A locked chest is here."));
}

// ---------------------------------------------------------------------------
// Fast policy flaw + router
// ---------------------------------------------------------------------------

#[test]
fn fast_policy_loops_on_locked_task() {
    let mut env = env_1000_locked();
    let policy = FastPolicy;
    let res = looproute_rs::run_episode(&mut env, &policy, None, None, 60);
    assert!(!res.success, "fast-only must fail the locked task");
    assert_eq!(res.steps, 60);
    // The flaw: it keeps proposing "open chest" in the kitchen.
    let opens = res
        .trajectory
        .iter()
        .filter(|t| t.action == "open chest")
        .count();
    assert!(opens >= 10, "expected repeated open-chest, got {opens}");
}

#[test]
fn fast_policy_solves_unlocked_task() {
    let mut env = KitchenWorld::new(1001, 3, false);
    let policy = FastPolicy;
    let res = looproute_rs::run_episode(&mut env, &policy, None, None, 60);
    assert!(res.success);
    assert!(res.steps < 60);
    assert_eq!(res.llm_calls, 0);
}

#[test]
fn router_units() {
    let state = KitchenWorld::new(7, 3, true).state();
    let mut r = LoopRouter::new(2);
    assert!(!r.should_assist(&state, "open chest"));
    r.note(&state, "open chest");
    assert!(r.should_assist(&state, "open chest"));
    let plan = r.assist(&state, "open chest", vec!["go pantry".to_string()]);
    assert_eq!(plan, vec!["go pantry".to_string()]);
    assert_eq!(r.assists, 1);
    assert!(!r.should_assist(&state, "open chest"));
    // A different action is tracked independently.
    assert!(!r.should_assist(&state, "take basil"));
    r.note(&state, "take basil");
    assert!(r.should_assist(&state, "take basil"));
    // Threshold 1 triggers on the first proposal.
    let r1 = LoopRouter::new(1);
    assert!(r1.should_assist(&state, "open chest"));
    // Reset clears counts and assists.
    r.reset();
    assert_eq!(r.assists, 0);
    assert!(!r.should_assist(&state, "open chest"));
}

// ---------------------------------------------------------------------------
// Oracle planner
// ---------------------------------------------------------------------------

#[test]
fn oracle_plan_is_legal_and_minimal_length() {
    for seed in 0..20u64 {
        let mut env = KitchenWorld::new(seed, 3, seed % 2 == 0);
        let plan = plan_to_goal(&env);
        assert!(!plan.is_empty(), "seed {seed}");
        assert_eq!(
            plan.last().unwrap(),
            "cook",
            "plan must end with cook (seed {seed})"
        );
        env.reset();
        for act in &plan {
            let (_, _, _, ok) = env.step(act);
            assert!(ok, "seed {seed} illegal plan step {act}");
        }
        assert_eq!(env.steps, plan.len(), "seed {seed}");
        assert!(env.recipe_collected());
    }
}

#[test]
fn oracle_assistant_matches_plan_to_goal() {
    let env = KitchenWorld::new(123, 3, true);
    assert_eq!(oracle_assistant(&env), plan_to_goal(&env));
}

// ---------------------------------------------------------------------------
// Spec round-trip
// ---------------------------------------------------------------------------

#[test]
fn spec_round_trip_preserves_task_and_episode() {
    for seed in [0u64, 1, 42, 1000, 1001, 2026] {
        for locked in [true, false] {
            let env = KitchenWorld::new(seed, 3, locked);
            let text = spec_of_env(&env);
            let spec = parse_spec(&text).expect("parse spec");
            let env2 = KitchenWorld::from_spec(&spec).expect("build from spec");
            assert_eq!(env2.recipe, env.recipe, "seed {seed} locked {locked}");
            assert_eq!(env2.placement, env.placement);
            assert_eq!(env2.in_chest, env.in_chest);
            assert_eq!(env2.key_room, env.key_room);
            assert_eq!(env2.locked, env.locked);
            // The same episode from the spec-built env is identical.
            let policy = FastPolicy;
            let mut a = KitchenWorld::new(seed, 3, locked);
            let mut b = KitchenWorld::from_spec(&spec).unwrap();
            let mut ra = LoopRouter::new(2);
            let mut rb = LoopRouter::new(2);
            let ea = looproute_rs::run_episode(
                &mut a,
                &policy,
                Some(&mut ra),
                Some(oracle_assistant),
                60,
            );
            let eb = looproute_rs::run_episode(
                &mut b,
                &policy,
                Some(&mut rb),
                Some(oracle_assistant),
                60,
            );
            assert_eq!(ea.success, eb.success);
            assert_eq!(ea.steps, eb.steps);
            assert_eq!(ea.assists, eb.assists);
            assert_eq!(ea.llm_calls, eb.llm_calls);
            assert_eq!(ea.trajectory.len(), eb.trajectory.len());
            for (ta, tb) in ea.trajectory.iter().zip(eb.trajectory.iter()) {
                assert_eq!(ta.state, tb.state);
                assert_eq!(ta.action, tb.action);
                assert_eq!(ta.source, tb.source);
                assert_eq!(ta.obs, tb.obs);
            }
        }
    }
}

#[test]
fn spec_parse_rejects_bad_input() {
    assert!(parse_spec("locked=1\nrecipe tomato\n").is_err()); // no key
    assert!(parse_spec("nonsense line\n").is_err());
    let spec = parse_spec(
        "# looproute-spec-v1\nlocked=0\nrecipe tomato onion\nplace tomato pantry\nplace onion garden\nkey bedroom\n",
    )
    .unwrap();
    assert_eq!(spec.recipe, vec!["tomato", "onion"]);
    assert!(spec.chest.is_none());
    let env = KitchenWorld::from_spec(&spec).unwrap();
    assert_eq!(env.key_room, 4);
}
