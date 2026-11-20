//! Deep tests for the GrepMem Rust port: property sweeps, grep checked
//! against a naive word-scan oracle, exhaustive oracle sequences, a
//! soak at a scale Python does not reach, and thread identity.

use grepmem_rs::{
    answer_question, apply_policy, evaluate_policy, generate_session, keywords,
    session_to_text, AppendOnlyLog, Policy, Question,
};
use std::time::Instant;

/// Naive word-scan oracle for `\bTOKEN\b` (case-insensitive): a line
/// matches iff one of its `[a-z0-9]+` words (lowercased) equals the
/// token. Independent of the `regex` crate.
fn naive_word_match(line: &str, token: &str) -> bool {
    keywords(line).iter().any(|w| w == token)
}

#[test]
fn grep_matches_naive_word_scan() {
    let tokens = [
        "project", "atlas", "beacon", "dr", "chen", "okafor", "lab", "7", "server", "node", "3",
        "experiment", "12", "27", "dataset", "vega", "model", "orion", "budget", "location",
        "port", "temperature", "status", "lead", "funding", "whereabouts", "endpoint", "heat",
        "state", "supervisor", "8080", "443", "37", "active", "paused", "coffee", "vpn",
        "numpy", "disk", "backup", "xyzzy", "the",
    ];
    for seed in 0..10u64 {
        let sess = generate_session(seed, 500, 20);
        for token in tokens {
            let pat = format!(r"\b{}\b", regex::escape(token));
            let hits = sess.log.grep(&pat, None);
            let expected: Vec<String> = sess
                .log
                .lines
                .iter()
                .filter(|ln| naive_word_match(ln, token))
                .cloned()
                .collect();
            assert_eq!(hits, expected, "seed {} token {}", seed, token);
            // Capped grep keeps the most recent matches.
            for cap in [1usize, 3, 7] {
                let capped = sess.log.grep(&pat, Some(cap));
                let want: Vec<String> = if expected.len() > cap {
                    expected[expected.len() - cap..].to_vec()
                } else {
                    expected.clone()
                };
                assert_eq!(capped, want, "seed {} token {} cap {}", seed, token, cap);
            }
        }
    }
}

#[test]
fn policy_properties_across_streams() {
    // Full is a perfect ceiling; GrepMem stays near it and above every
    // lossy baseline; accuracy is monotone in the grep budget.
    for (seed, n_lines) in [(0u64, 200usize), (1, 500), (2, 1000), (3, 200), (4, 2000)] {
        let sess = generate_session(seed, n_lines, 20);
        let sessions = vec![sess];
        let full = evaluate_policy(Policy::Full, &sessions, 40);
        assert_eq!(full.accuracy, 1.0, "seed {} full must be perfect", seed);
        let mut prev = 0.0f64;
        for k in [5usize, 10, 20, 40] {
            let g = evaluate_policy(Policy::GrepMem, &sessions, k);
            assert!(g.accuracy >= prev, "seed {} budget {} not monotone", seed, k);
            prev = g.accuracy;
        }
        let g40 = evaluate_policy(Policy::GrepMem, &sessions, 40);
        assert!(g40.accuracy >= 0.90, "seed {} grepmem {}", seed, g40.accuracy);
        for p in [Policy::Summary, Policy::Recency, Policy::RandomMask] {
            let lossy = evaluate_policy(p, &sessions, 40);
            assert!(
                g40.accuracy >= lossy.accuracy,
                "seed {} grepmem {} < {:?} {}",
                seed,
                g40.accuracy,
                p,
                lossy.accuracy
            );
        }
        // GrepMem context stays a small fraction of full.
        assert!(
            g40.mean_context_lines <= 0.15 * full.mean_context_lines,
            "seed {} ctx fraction",
            seed
        );
    }
}

#[test]
fn oracle_latest_wins_exhaustive() {
    // Exhaustive over (entity, attr, number of updates): the oracle
    // must always return the latest value, and never a value that only
    // appears as another fact's value.
    for (entity, _) in grepmem_rs::ENTITIES {
        for attr_def in grepmem_rs::ATTRIBUTES {
            for k in 1..=5usize {
                let mut log = AppendOnlyLog::new();
                let mut expected = "";
                for i in 0..k {
                    let v = attr_def.values[i % attr_def.values.len()];
                    expected = v;
                    log.append(format!(
                        "[t{:04}] OBS tool:web_search | {} {} {} {}",
                        i, entity, attr_def.name, attr_def.verb, v
                    ));
                }
                // Trap line: the entity as another fact's value. It is
                // itself a genuine (Project Atlas, lead) fact, so skip
                // it for that combination; the trap is covered by the
                // unit test oracle_entity_as_value_trap.
                if !(entity == "Project Atlas" && attr_def.name == "lead") {
                    log.append(
                        "[t0009] OBS tool:db_query | Project Atlas lead is led by Dr. Chen"
                            .to_string(),
                    );
                }
                let q = Question {
                    text: format!("What is the {} of {}?", attr_def.name, entity),
                    entity: entity.to_string(),
                    attr: attr_def.name.to_string(),
                    answer: expected.to_string(),
                    step: k - 1,
                    position: 0.5,
                    paraphrased: false,
                };
                assert_eq!(
                    answer_question(&q, &log.lines).as_deref(),
                    Some(expected),
                    "entity={} attr={} k={}",
                    entity,
                    attr_def.name,
                    k
                );
            }
        }
    }
}

#[test]
fn soak_20k_lines_x_200_questions() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let t0 = Instant::now();
    let sess = generate_session(7, 20_000, 200);
    let gen_ms = t0.elapsed().as_millis();
    // Determinism: regenerating the same seed reproduces the session.
    let sess2 = generate_session(7, 20_000, 200);
    assert_eq!(
        session_to_text(&sess, 7, 20_000, 200),
        session_to_text(&sess2, 7, 20_000, 200),
        "soak session must be deterministic"
    );
    let sessions = vec![sess];
    let mut parts: Vec<String> = Vec::new();
    let mut grepmem_acc = 0.0f64;
    for policy in Policy::all() {
        let t1 = Instant::now();
        let res = evaluate_policy(policy, &sessions, 40);
        let ms = t1.elapsed().as_millis();
        println!(
            "soak {:12} acc={:.4} ctx={:.1} para={:.4} ({} ms)",
            policy.name(),
            res.accuracy,
            res.mean_context_lines,
            res.paraphrase_accuracy,
            ms
        );
        if policy == Policy::GrepMem {
            grepmem_acc = res.accuracy;
        }
        parts.push(format!(
            "{{\"policy\": \"{}\", \"accuracy\": {:.4}, \"mean_context_lines\": {:.1}, \"paraphrase_accuracy\": {:.4}, \"eval_ms\": {}}}",
            policy.name(),
            res.accuracy,
            res.mean_context_lines,
            res.paraphrase_accuracy,
            ms
        ));
    }
    assert!(grepmem_acc >= 0.90, "soak grepmem acc {}", grepmem_acc);
    let total_ms = t0.elapsed().as_millis();
    let json = format!(
        "{{\"seed\": 7, \"n_lines\": 20000, \"n_questions\": 200, \"gen_ms\": {}, \"total_ms\": {}, \"results\": [{}]}}",
        gen_ms,
        total_ms,
        parts.join(", ")
    );
    std::fs::write(format!("{}/soak_results.json", manifest), &json)
        .expect("write soak_results.json");
    println!("soak total {} ms", total_ms);
}

#[test]
fn thread_identity_parallel_equals_serial() {
    // Parallel evaluation across threads gives identical results to
    // serial evaluation (the policies are pure functions of the log).
    let sess = generate_session(11, 2000, 100);
    let log = &sess.log;
    let questions = &sess.questions;
    // Serial.
    let serial: Vec<(Option<String>, usize)> = questions
        .iter()
        .map(|q| {
            let ctx = apply_policy(Policy::GrepMem, log, q, 40);
            (answer_question(q, &ctx), ctx.len())
        })
        .collect();
    // Parallel: 4 threads, split by question index.
    let n_threads = 4;
    let chunk = questions.len().div_ceil(n_threads);
    let parallel: Vec<(Option<String>, usize)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let qs = &questions[t * chunk..((t + 1) * chunk).min(questions.len())];
            handles.push(s.spawn(move || {
                qs.iter()
                    .map(|q| {
                        let ctx = apply_policy(Policy::GrepMem, log, q, 40);
                        (answer_question(q, &ctx), ctx.len())
                    })
                    .collect::<Vec<_>>()
            }));
        }
        let mut out = Vec::new();
        for h in handles {
            out.extend(h.join().unwrap());
        }
        out
    });
    assert_eq!(serial, parallel, "parallel != serial results");
}
