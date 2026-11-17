//! matrix-steps binary: verify | parity <kind> ... | soak
//!
//!   verify [--results PATH]  mirror of tests/test_matrix_steps_cpu.py:
//!                            full experiment + assertions
//!   parity <kind> ...        machine-readable outputs for the
//!                            Python-vs-Rust parity sweep (see
//!                            tests/parity_gen.py)
//!   soak [--results PATH]    10x-scale experiment + thread identity
//!
//! Std only, no network, no model calls.

use matrix_steps_rs::*;
use std::io::Write;

fn check(name: &str, cond: bool, detail: &str) {
    let status = if cond { "PASS" } else { "FAIL" };
    println!("  [{}] {} {}", status, name, detail);
    if !cond {
        panic!("FAILED: {} {}", name, detail);
    }
}

fn jstr(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn jvec_f64(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{}", x)).collect();
    format!("[{}]", parts.join(","))
}

fn jvec_i32(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{}", *x as i64)).collect();
    format!("[{}]", parts.join(","))
}

// ---------------------------------------------------------------- verify
fn cmd_verify(results_path: Option<&str>) -> i32 {
    let t0 = std::time::Instant::now();
    println!("matrix-steps verify (arXiv:2611.20230, rust)");

    // MT19937 known test vector (init_genrand(5489)).
    let mut mt = Mt19937::from_int_seed(5489);
    let vec: Vec<u32> = (0..5).map(|_| mt.uint32()).collect();
    check(
        "MT19937 test vector",
        vec == vec![3499211612, 581869302, 3890346734, 3586334585, 545404204],
        "",
    );
    let s1 = procedure_sign_vector("proc_0001", 64);
    let s2 = procedure_sign_vector("proc_0001", 64);
    let s3 = procedure_sign_vector("proc_0002", 64);
    check(
        "sign vector deterministic per procedure id",
        s1 == s2 && s1 != s3 && s1.iter().all(|x| *x == 1.0 || *x == -1.0),
        "",
    );

    // -- exp 1: write gate + scope isolation
    println!("== 1. write gate + scope isolation ==");
    let mut mem_a = EpisodicMemory::new("alice", EMBED_DIM, 0.25);
    let mut mem_b = EpisodicMemory::new("bob", EMBED_DIM, 0.25);
    let mut e1 = Episode::new(
        "e1", "alice", "tyre", "change tyre",
        vec!["deflate".into(), "lift tyre".into(), "fit tube".into()],
        "success", "external_check",
    );
    let mut e2 = Episode::new(
        "e2", "alice", "tyre", "change tyre",
        vec!["deflate".into(), "lift tyre".into(), "fit tube".into()],
        "success", "model_claim",
    );
    let e3 = Episode::new(
        "e3", "bob", "tyre", "change tyre",
        vec!["deflate".into(), "lift tyre".into(), "fit tube".into()],
        "success", "external_check",
    );
    let mut e4 = Episode::new(
        "e4", "alice", "tyre", "change tyre",
        vec!["deflate".into(), "lift tyre".into()],
        "success", "tool_result",
    );
    e4.agent_claimed_done = true;
    e4.tool_failed = true;
    let mut e5 = Episode::new(
        "e5", "alice", "tyre", "change tyre",
        vec!["deflate".into(), "lift tyre".into(), "fit tube".into()],
        "success", "tool_result",
    );
    e5.person_verdict = Some("failure".to_string());
    mem_a.record_episode(e1.clone());
    mem_a.record_episode(e2.clone());
    mem_b.record_episode(e3.clone());
    mem_a.record_episode(e4.clone());
    mem_a.record_episode(e5.clone());
    let verified_a = mem_a.verified_episodes(Some("tyre"));
    let mut ids: Vec<String> = verified_a.iter().map(|e| e.id.clone()).collect();
    ids.sort();
    println!("  verified episodes in alice scope: {:?}", ids);
    check("model-only-vouched success never recalled", !ids.contains(&"e2".to_string()), "");
    check("no episode from another scope returned", !ids.contains(&"e3".to_string()), "");
    check(
        "verified list holds the success and both failures",
        ids == vec!["e1".to_string(), "e4".to_string(), "e5".to_string()],
        "",
    );
    check(
        "'done' over failed tool call recorded as failure",
        recorded_outcome(&e4).as_deref() == Some("failure"),
        "",
    );
    check(
        "person verdict overrides recorded outcome",
        recorded_outcome(&e5).as_deref() == Some("failure"),
        "",
    );
    let succ: Vec<String> = verified_a
        .iter()
        .filter(|e| recorded_outcome(e).as_deref() == Some("success"))
        .map(|e| e.id.clone())
        .collect();
    check("only the externally verified success counts as a success", succ == vec!["e1".to_string()], "");
    let mut eps = vec![e1.clone(); 5];
    eps.extend(vec![e2.clone(); 5]);
    let rec = consolidate_episodes(&eps, MIN_EPISODES, MIN_SUCCESSES, MIN_COVERAGE);
    check(
        "unverified episodes do not count toward consolidation",
        rec.as_ref().map_or(false, |r| {
            r.n_episodes == 5 && r.episode_ids == vec!["e1".to_string(); 5]
        }),
        "",
    );
    println!("  1. write gate + scope isolation: OK");

    // -- exp 2: hidden criterion
    println!("== 2. self-judged vs verified (hidden arrival-speed criterion) ==");
    let r = simulate_hidden_criterion(7, 200, 40, 6);
    for (name, d) in [
        ("no_memory", &r.no_memory),
        ("self_judged_stops", &r.self_judged_stops),
        ("verified", &r.verified),
    ] {
        println!(
            "  {:<17} reported {:.3}  solved {:.3}  attempts/task {:.2}  damaging {}",
            name, d.reported_done, d.solved, d.attempts_per_task, d.damaging
        );
    }
    println!("  ceiling (solvable share): {:.3}", r.ceiling);
    let sj = &r.self_judged_stops;
    let ver = &r.verified;
    let nm = &r.no_memory;
    check(
        "self-judged reports far more than it solves",
        sj.reported_done >= 0.90
            && sj.solved <= 0.45
            && sj.reported_done - sj.solved >= 0.40,
        &format!("(reported {:.3} vs solved {:.3})", sj.reported_done, sj.solved),
    );
    check(
        "verified solves near the ceiling",
        ver.solved >= 0.70 && ver.solved >= r.ceiling - 0.05,
        &format!("(verified {:.3}, ceiling {:.3})", ver.solved, r.ceiling),
    );
    check(
        "verified solves much more than self-judged-stops",
        ver.solved - sj.solved >= 0.25,
        "",
    );
    check(
        "verified takes fewer attempts than no-memory",
        ver.attempts_per_task < nm.attempts_per_task,
        "",
    );
    check("verified damages fewer parts than no-memory", ver.damaging < nm.damaging, "");
    println!("  2. hidden criterion: OK");

    // -- exp 3: capacity
    println!("== 3. matrix capacity (d=1024 square, hash embedder) ==");
    let t_cap = std::time::Instant::now();
    let r200 = capacity_experiment(200, 20261117, 10, 0.25, EMBED_DIM);
    println!(
        "  200 procs: found(matrix) {:.3}  found(exact) {:.3}  next {:.3}  median lookup {:.2} ms  ({} assoc, {:.0}s)",
        r200.found_matrix, r200.found_exact, r200.next_step,
        r200.median_lookup_ms, r200.n_associations, t_cap.elapsed().as_secs_f64()
    );
    let t_cap = std::time::Instant::now();
    let r500 = capacity_experiment(500, 20261117, 10, 0.25, EMBED_DIM);
    println!(
        "  500 procs: found(matrix) {:.3}  found(exact) {:.3}  next {:.3}  median lookup {:.2} ms  ({} assoc, {:.0}s)",
        r500.found_matrix, r500.found_exact, r500.next_step,
        r500.median_lookup_ms, r500.n_associations, t_cap.elapsed().as_secs_f64()
    );
    check("found >= 0.95 at 200 procedures", r200.found_matrix >= 0.95,
          &format!("({:.3})", r200.found_matrix));
    check("next-step >= 0.90 at 200 procedures", r200.next_step >= 0.90,
          &format!("({:.3})", r200.next_step));
    check(
        "exact scan finds everything at both sizes",
        r200.found_exact == 1.0 && r500.found_exact == 1.0,
        "",
    );
    println!("  3. capacity: OK (500-procedure numbers reported honestly)");

    // -- exp 4: consolidation
    println!("== 4. deterministic consolidation ==");
    let seq = vec!["deflate".to_string(), "lift tyre".to_string(), "fit tube".to_string()];
    let other = vec![
        "deflate".to_string(),
        "lift tyre".to_string(),
        "fit tube".to_string(),
        "extra step".to_string(),
    ];
    let mk = |i: usize, steps: Vec<String>, outcome: &str, failure: Option<&str>| -> Episode {
        let mut e = Episode::new(
            &format!("c{}", i), "alice", "tyre", "change tyre", steps, outcome,
            "external_check",
        );
        e.args = vec![("torque".to_string(), "40".to_string())];
        e.condition = "dry".to_string();
        e.failure_mode = failure.map(|s| s.to_string());
        e
    };
    let eps: Vec<Episode> = (0..4).map(|i| mk(i, seq.clone(), "success", None)).collect();
    check("no record below 5 episodes", consolidate_episodes(&eps, 5, 3, 0.6).is_none(), "");
    let mut eps: Vec<Episode> = (0..2).map(|i| mk(i, seq.clone(), "success", None)).collect();
    eps.extend((2..5).map(|i| mk(i, seq.clone(), "failure", Some("slipped"))));
    check("no record below 3 successes", consolidate_episodes(&eps, 5, 3, 0.6).is_none(), "");
    let mut eps: Vec<Episode> = (0..2).map(|i| mk(i, seq.clone(), "success", None)).collect();
    eps.extend((2..4).map(|i| mk(i, other.clone(), "success", None)));
    eps.extend((4..6).map(|i| mk(i, seq.clone(), "failure", Some("slipped"))));
    check(
        "no record when top sequence < 60% of successes",
        consolidate_episodes(&eps, 5, 3, 0.6).is_none(),
        "",
    );
    let mut eps: Vec<Episode> = (0..4).map(|i| mk(i, seq.clone(), "success", None)).collect();
    eps.extend((4..6).map(|i| mk(i, seq.clone(), "failure", Some("slipped"))));
    let rec = consolidate_episodes(&eps, 5, 3, 0.6);
    println!(
        "  record: confidence {:.4} n_episodes {} n_successes {} top_share {:.2} fixed {:?} failures {:?}",
        rec.as_ref().map(|r| r.confidence).unwrap_or(0.0),
        rec.as_ref().map(|r| r.n_episodes).unwrap_or(0),
        rec.as_ref().map(|r| r.n_successes).unwrap_or(0),
        rec.as_ref().map(|r| r.top_sequence_share).unwrap_or(0.0),
        rec.as_ref().map(|r| r.fixed_args.clone()).unwrap_or_default(),
        rec.as_ref().map(|r| r.failure_modes.clone()).unwrap_or_default(),
    );
    check("record produced once minimums are met", rec.is_some(), "");
    let rec = rec.unwrap();
    check(
        "confidence == (s+1)/(n+2)",
        (rec.confidence - confidence(4, 6)).abs() < 1e-12,
        &format!("({:.4})", rec.confidence),
    );
    check(
        "fixed args listed, failures counted",
        rec.fixed_args == vec![("torque".to_string(), "40".to_string())]
            && rec.failure_modes == vec![("slipped".to_string(), 2)],
        "",
    );
    println!("  4. consolidation: OK");

    // -- exp 5: revisit
    println!("== 5. revisit first-attempt ==");
    let r5 = simulate_revisit_stream(11, 1000, 4, 4);
    println!(
        "  situations {} tasks {} solved {} revisits {} first-attempt on revisits {:.3}",
        r5.n_situations, r5.n_tasks, r5.solved_situations, r5.revisits,
        r5.revisit_first_attempt
    );
    check(
        "exact records solve every revisit at the first attempt",
        r5.revisit_first_attempt == 1.0,
        "",
    );
    println!("  5. revisit: OK");

    // -- exp 6: erasure
    println!("== 6. erasure (10% of situations) ==");
    let mut mem = EpisodicMemory::new("alice", EMBED_DIM, 0.25);
    let procs = generate_synthetic_procedures(200, 20261117, 3, 8, 0.15);
    let mut rng = Rng::new(4242);
    let mut idx: Vec<usize> = (0..200).collect();
    idx = rng.shuffle(&idx);
    let erased_idx: Vec<usize> = idx[..20].to_vec();
    let erased_set: std::collections::HashSet<usize> =
        erased_idx.iter().cloned().collect();
    for (i, p) in procs.iter().enumerate() {
        let mut p = p.clone();
        let ep = Episode::new(
            &format!("ep_{:04}", i), "alice", "proc", &p.goal,
            p.steps.iter().map(|s| s.text().to_string()).collect(),
            "success", "external_check",
        );
        mem.record_episode(ep);
        p.episode_ids = vec![format!("ep_{:04}", i)];
        mem.add_record(p);
    }
    mem.rebuild_matrix(10, true);
    let before: Vec<bool> = procs
        .iter()
        .map(|p| {
            let (rec, _) = mem.find(&p.goal, "matrix", 0.0);
            rec.as_deref() == Some(p.id.as_str())
        })
        .collect();
    check("all 200 procedures found before erasure", before.iter().all(|b| *b), "");
    let erased_ids: Vec<String> =
        erased_idx.iter().map(|i| format!("ep_{:04}", i)).collect();
    let (erased_out, _) = mem.erase_episodes(&erased_ids, "erasure request (test)");
    println!("  erased {} episodes; audit entries: {}", erased_out.len(), mem.audit.len());
    let mut kept_ok = 0usize;
    let mut erased_gone = 0usize;
    let mut erased_cos: Vec<f64> = Vec::new();
    for (i, p) in procs.iter().enumerate() {
        let (rec, _) = mem.find(&p.goal, "matrix", 0.0);
        if erased_set.contains(&i) {
            if rec.is_none() || rec.as_deref() != Some(p.id.as_str()) {
                erased_gone += 1;
            }
            let q = mem.embedder.embed(&p.goal);
            let v_hat = to_f64(&mem.matrix.read(&to_f32(&q)));
            let v_erased = mem.embedder.embed(&p.answer_text());
            erased_cos.push(cosine(&v_hat, &v_erased));
        } else if rec.as_deref() == Some(p.id.as_str()) {
            kept_ok += 1;
        }
    }
    let max_erased_cos = erased_cos.iter().cloned().fold(0.0f64, f64::max);
    println!(
        "  kept procedures still found: {}/180; erased not returned: {}/20; max cosine of erased key to erased value: {:.3}",
        kept_ok, erased_gone, max_erased_cos
    );
    check(
        "exactly the erased situations lose their record",
        erased_gone == 20 && kept_ok == 180,
        "",
    );
    check(
        "matrix holds no trace of erased episodes",
        max_erased_cos < 0.50,
        &format!("(max cos {:.3})", max_erased_cos),
    );
    check(
        "audit names record + cause, never a person",
        mem.audit.len() == 20
            && mem.audit.iter().all(|a| !a.record.is_empty() && !a.cause.is_empty()),
        "",
    );
    let r_all = simulate_revisit_stream(11, 200, 4, 4);
    check(
        "revisit stream still solves every kept revisit at first attempt",
        r_all.revisit_first_attempt == 1.0,
        "",
    );
    println!("  6. erasure: OK");

    // -- exp 7: demotion
    println!("== 7. demotion on a protective stop ==");
    let r7 = simulate_demotion(13, 200, 4, 3, 0.97);
    println!(
        "  solved without rule {:.3} (trips {}) -> with rule {:.3} (trips {}, demotions {}, first trip at task {:?})",
        r7.solved_without_rule, r7.trips_without_rule, r7.solved_with_rule,
        r7.trips_with_rule, r7.demotions, r7.first_trip_task
    );
    check(
        "default demoted at once on its first protective stop",
        r7.demotions == 1 && r7.trips_with_rule == 1,
        "",
    );
    check(
        "solved share recovers with the rule",
        r7.solved_with_rule >= 0.90
            && r7.solved_with_rule - r7.solved_without_rule >= 0.15,
        "",
    );
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
    mem.rebuild_matrix(10, true);
    let restored = mem.demote_on_guard_trip("bad");
    check(
        "predecessor restored as default",
        restored.as_deref() == Some("prev")
            && mem.default_id.as_deref() == Some("prev")
            && mem.records.get("bad").map(|r| r.status.as_str()) == Some("demoted"),
        "",
    );
    println!("  7. demotion: OK");

    // -- exp 8: promotion honesty
    println!("== 8. promotion-test honesty (200 coin-decided windows) ==");
    let (prom_ident, naive_ident) = run_promotion_windows(101, 0.5, 0.5, 200, 40);
    let (prom_better, naive_better) = run_promotion_windows(101, 0.3, 0.6, 200, 40);
    println!(
        "  identical variant: promoted {}/200 (naive rule keeps {}/200)",
        prom_ident, naive_ident
    );
    println!(
        "  better variant (30% -> 60%): promoted {}/200 (naive rule keeps {}/200)",
        prom_better, naive_better
    );
    check("identical variant is never promoted", prom_ident == 0, "");
    check(
        "naive keep-any-higher-score rule false-promotes often",
        naive_ident >= 70,
        &format!("({}/200)", naive_ident),
    );
    check(
        "a genuinely better variant is promoted in most windows",
        prom_better >= 80,
        &format!("({}/200)", prom_better),
    );
    println!("  8. promotion honesty: OK");

    println!(
        "ALL MATRIX-STEPS CHECKS PASSED (rust verify, {:.1}s)",
        t0.elapsed().as_secs_f64()
    );
    if let Some(path) = results_path {
        let json = format!(
            "{{\n  \"exp2\": {{\"no_memory_solved\": {}, \"self_judged_reported\": {}, \"self_judged_solved\": {}, \"verified_solved\": {}, \"ceiling\": {}}},\n  \"exp3\": {{\"found_200\": {}, \"next_200\": {}, \"found_500\": {}, \"next_500\": {}, \"median_lookup_ms_200\": {}}},\n  \"exp5\": {{\"revisit_first_attempt\": {}}},\n  \"exp7\": {{\"solved_without_rule\": {}, \"solved_with_rule\": {}}},\n  \"exp8\": {{\"identical_promoted\": {}, \"identical_naive\": {}, \"better_promoted\": {}}}\n}}\n",
            r.no_memory.solved, sj_reported(&r), sj_solved(&r), r.verified.solved, r.ceiling,
            r200.found_matrix, r200.next_step, r500.found_matrix, r500.next_step,
            r200.median_lookup_ms, r5.revisit_first_attempt,
            r7.solved_without_rule, r7.solved_with_rule,
            prom_ident, naive_ident, prom_better
        );
        std::fs::write(path, json).unwrap();
        println!("results written to {}", path);
    }
    0
}

fn sj_reported(r: &HiddenCriterionResult) -> f64 {
    r.self_judged_stops.reported_done
}
fn sj_solved(r: &HiddenCriterionResult) -> f64 {
    r.self_judged_stops.solved
}

// ---------------------------------------------------------------- parity
fn cmd_parity(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("usage: matrix-steps parity <rng|embed|sign|matrix|consolidate|promotion|hidden|revisit|demotion> ...");
        return 2;
    }
    match args[0].as_str() {
        "rng" => {
            let seed: u64 = args[1].parse().expect("seed");
            let n: usize = args[2].parse().expect("n");
            let mut r = Rng::new(seed);
            let u32s: Vec<u32> = (0..n).map(|_| r.uint32()).collect();
            let mut r = Rng::new(seed);
            let uniforms: Vec<f64> = (0..n).map(|_| r.uniform()).collect();
            let mut r = Rng::new(seed);
            let randint6: Vec<i64> = (0..n).map(|_| r.randint(0, 5)).collect();
            let mut r = Rng::new(seed);
            let bern: Vec<bool> = (0..n).map(|_| r.bernoulli(0.5)).collect();
            println!(
                "{{\"kind\":\"rng\",\"seed\":{},\"u32\":{:?},\"uniform\":{},\"randint6\":{:?},\"bernoulli\":{:?}}}",
                seed,
                u32s,
                jvec_f64(&uniforms),
                randint6,
                bern
            );
        }
        "embed" => {
            let text = &args[1];
            let mut emb = HashEmbedder::new();
            let v = emb.embed(text);
            println!(
                "{{\"kind\":\"embed\",\"text\":{},\"vec\":{},\"norm\":{}}}",
                jstr(text),
                jvec_f64(&v),
                v.iter().map(|x| x * x).sum::<f64>().sqrt()
            );
        }
        "sign" => {
            let id = &args[1];
            let width: usize = args[2].parse().expect("width");
            let s = procedure_sign_vector(id, width);
            println!(
                "{{\"kind\":\"sign\",\"id\":{},\"width\":{},\"signs\":{}}}",
                jstr(id),
                width,
                jvec_i32(&s)
            );
        }
        "matrix" => {
            // matrix <seed> <n_procs> <passes> <beta>
            let seed: u64 = args[1].parse().expect("seed");
            let n_procs: usize = args[2].parse().expect("n_procs");
            let passes: usize = args[3].parse().expect("passes");
            let beta: f32 = args[4].parse().expect("beta");
            let mut mem = EpisodicMemory::new("parity", EMBED_DIM, beta);
            let procs = generate_synthetic_procedures(n_procs, seed, 3, 8, 0.15);
            for p in &procs {
                mem.add_record(p.clone());
            }
            let pairs = mem.write_pairs();
            mem.matrix.rebuild(&pairs, passes, beta);
            let mut found: Vec<String> = Vec::new();
            let mut next: Vec<String> = Vec::new();
            for p in &procs {
                let (rec, _) = mem.find(&p.goal, "matrix", 0.0);
                found.push(rec.unwrap_or_default());
                for i in 0..p.steps.len().saturating_sub(1) {
                    let (best, _) = mem.next_step_via_matrix(&p.id, i);
                    next.push(best.unwrap_or_default());
                }
            }
            let checksum: f64 = mem.matrix.m.iter().map(|x| *x as f64).sum();
            let found_s: Vec<String> = found.iter().map(|s| jstr(s)).collect();
            let next_s: Vec<String> = next.iter().map(|s| jstr(s)).collect();
            println!(
                "{{\"kind\":\"matrix\",\"seed\":{},\"n_procs\":{},\"passes\":{},\"beta\":{},\"found\":[{}],\"next\":[{}],\"matrix_checksum\":{},\"matrix_bytes\":{}}}",
                seed, n_procs, passes, beta,
                found_s.join(","), next_s.join(","), checksum,
                mem.matrix.m.len() * 4
            );
        }
        "consolidate" => {
            // consolidate <seed> <n_episodes>
            let seed: u64 = args[1].parse().expect("seed");
            let n: usize = args[2].parse().expect("n");
            let mut rng = Rng::new(seed);
            let seq = vec!["deflate".to_string(), "lift tyre".to_string(), "fit tube".to_string()];
            let other = vec![
                "deflate".to_string(),
                "lift tyre".to_string(),
                "fit tube".to_string(),
                "extra step".to_string(),
            ];
            let mut eps: Vec<Episode> = Vec::new();
            for i in 0..n {
                let success = rng.bernoulli(0.7);
                let steps = if rng.bernoulli(0.8) { seq.clone() } else { other.clone() };
                let mut e = Episode::new(
                    &format!("e{}", i), "alice", "tyre", "change tyre", steps,
                    if success { "success" } else { "failure" },
                    if rng.bernoulli(0.9) { "external_check" } else { "model_claim" },
                );
                e.args = vec![("torque".to_string(), "40".to_string())];
                e.condition = "dry".to_string();
                if !success {
                    e.failure_mode = Some("slipped".to_string());
                }
                eps.push(e);
            }
            let rec = consolidate_episodes(&eps, MIN_EPISODES, MIN_SUCCESSES, MIN_COVERAGE);
            match rec {
                None => println!(
                    "{{\"kind\":\"consolidate\",\"seed\":{},\"n\":{},\"record\":null}}",
                    seed, n
                ),
                Some(r) => println!(
                    "{{\"kind\":\"consolidate\",\"seed\":{},\"n\":{},\"record\":{{\"confidence\":{},\"n_episodes\":{},\"n_successes\":{},\"top_share\":{},\"steps\":[{}],\"episode_ids\":[{}]}}}}",
                    seed,
                    n,
                    r.confidence,
                    r.n_episodes,
                    r.n_successes,
                    r.top_sequence_share,
                    r.steps.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(","),
                    r.episode_ids.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(","),
                ),
            }
        }
        "promotion" => {
            // promotion <seed> <p_default> <p_candidate> <n_windows>
            let seed: u64 = args[1].parse().expect("seed");
            let p_default: f64 = args[2].parse().expect("p_default");
            let p_candidate: f64 = args[3].parse().expect("p_candidate");
            let n_windows: usize = args[4].parse().expect("n_windows");
            let (promoted, naive) =
                run_promotion_windows(seed, p_default, p_candidate, n_windows, 40);
            println!(
                "{{\"kind\":\"promotion\",\"seed\":{},\"p_default\":{},\"p_candidate\":{},\"n_windows\":{},\"promoted\":{},\"naive_kept\":{}}}",
                seed, p_default, p_candidate, n_windows, promoted, naive
            );
        }
        "hidden" => {
            let seed: u64 = args[1].parse().expect("seed");
            let r = simulate_hidden_criterion(seed, 200, 40, 6);
            let counts = |d: &PolicyOutcome| -> (usize, usize, usize) {
                (
                    (d.reported_done * 200.0).round() as usize,
                    (d.solved * 200.0).round() as usize,
                    d.damaging,
                )
            };
            let (nm_r, nm_s, nm_d) = counts(&r.no_memory);
            let (sj_r, sj_s, sj_d) = counts(&r.self_judged_stops);
            let (v_r, v_s, v_d) = counts(&r.verified);
            println!(
                "{{\"kind\":\"hidden\",\"seed\":{},\"no_memory\":{{\"reported\":{},\"solved\":{},\"attempts\":{},\"damaging\":{}}},\"self_judged\":{{\"reported\":{},\"solved\":{},\"attempts\":{},\"damaging\":{}}},\"verified\":{{\"reported\":{},\"solved\":{},\"attempts\":{},\"damaging\":{}}},\"ceiling\":{}}}",
                seed,
                nm_r, nm_s, (r.no_memory.attempts_per_task * 200.0).round() as usize, nm_d,
                sj_r, sj_s, (r.self_judged_stops.attempts_per_task * 200.0).round() as usize, sj_d,
                v_r, v_s, (r.verified.attempts_per_task * 200.0).round() as usize, v_d,
                r.ceiling
            );
        }
        "revisit" => {
            let seed: u64 = args[1].parse().expect("seed");
            let n: usize = args[2].parse().expect("n");
            let repeats: usize = args[3].parse().expect("repeats");
            let r = simulate_revisit_stream(seed, n, repeats, 4);
            println!(
                "{{\"kind\":\"revisit\",\"seed\":{},\"n\":{},\"repeats\":{},\"solved_situations\":{},\"revisits\":{},\"revisit_first\":{},\"first_encounter_attempts\":{}}}",
                seed, n, repeats, r.solved_situations, r.revisits,
                (r.revisit_first_attempt * r.revisits as f64).round() as usize,
                r.first_encounter_attempts
            );
        }
        "demotion" => {
            let seed: u64 = args[1].parse().expect("seed");
            let r = simulate_demotion(seed, 200, 4, 3, 0.97);
            println!(
                "{{\"kind\":\"demotion\",\"seed\":{},\"solved_without\":{},\"solved_with\":{},\"trips_without\":{},\"trips_with\":{},\"demotions\":{},\"first_trip\":{}}}",
                seed,
                (r.solved_without_rule * 200.0).round() as usize,
                (r.solved_with_rule * 200.0).round() as usize,
                r.trips_without_rule, r.trips_with_rule, r.demotions,
                r.first_trip_task.map(|t| t.to_string()).unwrap_or_else(|| "null".to_string())
            );
        }
        other => {
            eprintln!("unknown parity kind: {}", other);
            return 2;
        }
    }
    0
}

// ---------------------------------------------------------------- soak
fn cmd_soak(results_path: Option<&str>) -> i32 {
    let t0 = std::time::Instant::now();
    println!("matrix-steps soak (10x revisit + thread identity)");

    // 10x the Python revisit scale: 10 000 situations, 4 repeats.
    let r = simulate_revisit_stream(11, 10000, 4, 4);
    println!(
        "  revisit 10k: situations {} tasks {} solved {} revisits {} first-attempt {:.3}",
        r.n_situations, r.n_tasks, r.solved_situations, r.revisits,
        r.revisit_first_attempt
    );
    check(
        "10k-situation revisit still solves every revisit at first attempt",
        r.revisit_first_attempt == 1.0,
        "",
    );

    // Thread identity: the code is stateless (embedder + sign vectors +
    // matrix reads); four threads computing the same outputs must agree.
    let handles: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                let mut emb = HashEmbedder::new();
                let v = emb.embed("change the tyre on the bicycle w0142 w2088");
                let s = procedure_sign_vector("proc_0042", 1024);
                let mut m = DeltaMatrix::new(1024, None, "sign", 777);
                let k = to_f32(&emb.embed("w0001 w0002 w0003 w0004 w0005 w0006"));
                let val = to_f32(&emb.embed("w0001 w0002 w0007 w0008"));
                m.write(&k, &val, 1.0, 1.0);
                let read = m.read(&k);
                let checksum: f64 = read.iter().map(|x| *x as f64).sum();
                (v, s, checksum)
            })
        })
        .collect();
    let mut outs = Vec::new();
    for h in handles {
        outs.push(h.join().unwrap());
    }
    let identical = outs.windows(2).all(|w| {
        w[0].0 == w[1].0 && w[0].1 == w[1].1 && (w[0].2 - w[1].2).abs() < 1e-9
    });
    check("thread identity: 4 threads agree bit-for-bit", identical, "");
    println!("  thread identity: 4 threads agree");

    // Property sweep: random small matrices recover what was written.
    let mut rng = Rng::new(20261117);
    let mut worst = 0.0f64;
    for _ in 0..50 {
        let dim = rng.randint(16, 64) as usize;
        let mut m = DeltaMatrix::new(dim, None, "sign", 777);
        let mut pairs: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for _ in 0..rng.randint(1, 8) {
            let k: Vec<f32> = (0..dim).map(|_| rng.uniform() as f32 * 2.0 - 1.0).collect();
            let v: Vec<f32> = (0..dim).map(|_| rng.uniform() as f32 * 2.0 - 1.0).collect();
            let kn = {
                let n: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
                k.iter().map(|x| x / n).collect::<Vec<f32>>()
            };
            pairs.push((kn, v));
        }
        for _ in 0..20 {
            for (k, v) in &pairs {
                m.write(k, v, 1.0, 0.25);
            }
        }
        for (k, v) in &pairs {
            let got = m.read(k);
            let err: f64 = got
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            worst = worst.max(err);
        }
    }
    println!("  property sweep: worst read-back error over 50 random matrices: {:.4}", worst);
    check("delta-rule read-back error stays bounded", worst < 1.0, &format!("({:.4})", worst));

    println!("SOAK PASSED in {:.1}s", t0.elapsed().as_secs_f64());
    if let Some(path) = results_path {
        let json = format!(
            "{{\"revisit_first_attempt_10k\": {}, \"thread_identity\": true, \"worst_readback_error\": {}}}\n",
            r.revisit_first_attempt, worst
        );
        std::fs::write(path, json).unwrap();
        println!("results written to {}", path);
    }
    0
}

// ---------------------------------------------------------------- main
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = if args.is_empty() || args[0] == "verify" {
        let path = args
            .windows(2)
            .find(|w| w[0] == "--results")
            .map(|w| w[1].as_str());
        cmd_verify(path)
    } else if args[0] == "parity" {
        cmd_parity(&args[1..])
    } else if args[0] == "soak" {
        let path = args
            .windows(2)
            .find(|w| w[0] == "--results")
            .map(|w| w[1].as_str());
        cmd_soak(path)
    } else {
        eprintln!("usage: matrix-steps [verify [--results PATH] | parity <kind> ... | soak [--results PATH]]");
        2
    };
    let _ = std::io::stdout().flush();
    std::process::exit(code);
}
