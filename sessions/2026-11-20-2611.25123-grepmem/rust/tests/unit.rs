//! Unit tests for the GrepMem Rust port.
//!
//! The RNG tests pin CPython `random.Random` values (computed with
//! CPython 3.12); the policy/oracle tests pin values computed with the
//! Python module on hand-built logs.

use grepmem_rs::{
    answer_question, generate_session, keywords, policy_full, policy_grepmem,
    policy_random_mask, policy_recency, policy_summary, session_from_text, session_to_text,
    AppendOnlyLog, GrepMemOpts, PyRandom, Question,
};

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-15
}

#[test]
fn rng_random_matches_python() {
    let expected: [[f64; 3]; 6] = [
        [0.8444218515250481, 0.7579544029403025, 0.420571580830845],
        [0.13436424411240122, 0.8474337369372327, 0.763774618976614],
        [0.9560342718892494, 0.9478274870593494, 0.05655136772680869],
        [0.23796462709189137, 0.5442292252959519, 0.36995516654807925],
        [0.23604808973743452, 0.1031660342307158, 0.396058242610681],
        [0.6229016948897019, 0.7417869892607294, 0.7951935655656966],
    ];
    for (seed, vals) in expected.iter().enumerate() {
        let mut rng = PyRandom::new(seed as u64);
        for (i, &v) in vals.iter().enumerate() {
            assert!(approx(rng.random(), v), "seed {} draw {}", seed, i);
        }
    }
}

#[test]
fn rng_getrandbits_matches_python() {
    // Fresh Random(0) each time.
    assert_eq!(PyRandom::new(0).getrandbits(32), 3_626_764_237);
    assert_eq!(PyRandom::new(0).getrandbits(6), 54);
    assert_eq!(PyRandom::new(0).getrandbits(33), 3_626_764_237);
    assert_eq!(PyRandom::new(0).getrandbits(40), 424_533_559_245);
    assert_eq!(PyRandom::new(0).getrandbits(64), 7_106_521_602_475_165_645);
    // Sequential draws on one stream (as in the Python probe).
    let mut rng = PyRandom::new(0);
    assert_eq!(rng.getrandbits(6), 54);
    assert_eq!(rng.getrandbits(3), 3);
    assert_eq!(rng.randbelow(6), 3);
}

#[test]
fn rng_choice_sample_shuffle_match_python() {
    assert_eq!(*PyRandom::new(0).choice(&[0, 1, 2, 3, 4]), 3);
    assert_eq!(*PyRandom::new(0).choice(&(0..10).collect::<Vec<_>>()), 6);
    assert_eq!(
        PyRandom::new(0).sample_indices(20, 5),
        vec![12, 13, 1, 8, 15]
    );
    assert_eq!(PyRandom::new(0).sample_indices(24, 3), vec![12, 13, 1]);
    assert_eq!(
        PyRandom::new(0).sample_indices(500, 41)[..10].to_vec(),
        vec![432, 197, 388, 455, 215, 20, 132, 494, 261, 248]
    );
    let mut xs: Vec<i32> = (0..10).collect();
    PyRandom::new(0).shuffle(&mut xs);
    assert_eq!(xs, vec![7, 8, 1, 5, 3, 4, 2, 0, 9, 6]);
}

fn handbuilt_log() -> AppendOnlyLog {
    let mut log = AppendOnlyLog::new();
    for step in 0..30usize {
        if step % 2 == 0 {
            log.append(format!(
                "[t{:04}] OBS tool:web_search | Project Atlas budget is $1.2M",
                step
            ));
        } else {
            log.append(format!(
                "[t{:04}] CHAT | coffee machine needs descaling",
                step
            ));
        }
    }
    log
}

fn budget_question() -> Question {
    Question {
        text: "What is the budget of Project Atlas?".to_string(),
        entity: "Project Atlas".to_string(),
        attr: "budget".to_string(),
        answer: "$1.2M".to_string(),
        step: 0,
        position: 0.0,
        paraphrased: false,
    }
}

#[test]
fn log_grep_case_insensitive_and_cap() {
    let mut log = AppendOnlyLog::new();
    for s in ["Alpha one", "BETA two", "alpha three", "gamma ALPHA", "delta"] {
        log.append(s.to_string());
    }
    assert_eq!(log.len(), 5);
    assert_eq!(
        log.grep("alpha", None),
        vec!["Alpha one", "alpha three", "gamma ALPHA"]
    );
    assert_eq!(log.grep("alpha", Some(2)), vec!["alpha three", "gamma ALPHA"]);
    assert!(log.grep("zzz", None).is_empty());
    assert_eq!(log.grep("BETA", None), vec!["BETA two"]);
}

#[test]
fn policy_grepmem_matches_python() {
    let log = handbuilt_log();
    let q = budget_question();
    let hits = policy_grepmem(&log, &q, GrepMemOpts::new(40));
    assert_eq!(hits.len(), 15);
    assert!(hits.iter().all(|l| l.contains("budget is $1.2M")));
    assert_eq!(hits[0], "[t0000] OBS tool:web_search | Project Atlas budget is $1.2M");
    assert_eq!(hits[14], "[t0028] OBS tool:web_search | Project Atlas budget is $1.2M");
    // Cap keeps the most recent matches.
    let capped = policy_grepmem(&log, &q, GrepMemOpts::new(3));
    assert_eq!(
        capped,
        vec![
            "[t0024] OBS tool:web_search | Project Atlas budget is $1.2M",
            "[t0026] OBS tool:web_search | Project Atlas budget is $1.2M",
            "[t0028] OBS tool:web_search | Project Atlas budget is $1.2M",
        ]
    );
    // entity_only and attr_only both hit all 15 here (same lines carry
    // both the entity and the attribute).
    assert_eq!(
        policy_grepmem(
            &log,
            &q,
            GrepMemOpts {
                max_lines: 40,
                entity_only: true,
                attr_only: false,
            }
        )
        .len(),
        15
    );
    assert_eq!(
        policy_grepmem(
            &log,
            &q,
            GrepMemOpts {
                max_lines: 40,
                entity_only: false,
                attr_only: true,
            }
        )
        .len(),
        15
    );
    // Full policy is the whole log.
    assert_eq!(policy_full(&log, &q).len(), 30);
}

#[test]
fn policy_summary_and_random_mask_match_python() {
    let log = handbuilt_log();
    let q = budget_question();
    let summary = policy_summary(&log, &q, 25, 0.15, 0);
    assert_eq!(
        summary,
        vec![
            "[t0000] OBS tool:web_search | Project Atlas budget is $1.2M",
            "[t0014] OBS tool:web_search | Project Atlas budget is $1.2M",
            "[t0025] CHAT | coffee machine needs descaling",
            "[t0026] OBS tool:web_search | Project Atlas budget is $1.2M",
            "[t0027] CHAT | coffee machine needs descaling",
            "[t0028] OBS tool:web_search | Project Atlas budget is $1.2M",
            "[t0029] CHAT | coffee machine needs descaling",
        ]
    );
    let masked = policy_random_mask(&log, &q, 0.20, 0);
    assert_eq!(masked, vec!["[t0025] CHAT | coffee machine needs descaling"]);
    // Deterministic: same seed, same output.
    assert_eq!(policy_random_mask(&log, &q, 0.20, 0), masked);
    assert_eq!(policy_summary(&log, &q, 25, 0.15, 0), summary);
}

#[test]
fn policy_recency_window() {
    let log = handbuilt_log();
    let q = budget_question();
    let last5 = policy_recency(&log, &q, 5);
    assert_eq!(last5.len(), 5);
    assert_eq!(last5[0], "[t0025] CHAT | coffee machine needs descaling");
    assert_eq!(last5[4], "[t0029] CHAT | coffee machine needs descaling");
    assert_eq!(policy_recency(&log, &q, 100).len(), 30);
}

#[test]
fn oracle_entity_as_value_trap() {
    let mut log = AppendOnlyLog::new();
    log.append("[t0001] OBS tool:db_query | Project Atlas lead is led by Dr. Chen".to_string());
    log.append("[t0002] OBS tool:web_search | Dr. Chen budget is $2.4M".to_string());
    log.append("[t0003] OBS tool:web_search | Dr. Chen budget is $3.1M".to_string());
    let ctx = log.lines.clone();
    // The "lead is led by Dr. Chen" line must NOT answer a question
    // about Dr. Chen's budget; the latest budget fact wins.
    let q = Question {
        text: "What is the budget of Dr. Chen?".to_string(),
        entity: "Dr. Chen".to_string(),
        attr: "budget".to_string(),
        answer: "$3.1M".to_string(),
        step: 3,
        position: 1.0,
        paraphrased: false,
    };
    assert_eq!(answer_question(&q, &ctx).as_deref(), Some("$3.1M"));
    // And the lead line does answer the lead question.
    let q2 = Question {
        text: "What is the lead of Project Atlas?".to_string(),
        entity: "Project Atlas".to_string(),
        attr: "lead".to_string(),
        answer: "Dr. Chen".to_string(),
        step: 1,
        position: 0.5,
        paraphrased: false,
    };
    assert_eq!(answer_question(&q2, &ctx).as_deref(), Some("Dr. Chen"));
    // Missing fact -> None.
    let q3 = Question {
        text: "What is the port of Lab 7?".to_string(),
        entity: "Lab 7".to_string(),
        attr: "port".to_string(),
        answer: "port 8080".to_string(),
        step: 0,
        position: 0.0,
        paraphrased: false,
    };
    assert_eq!(answer_question(&q3, &ctx), None);
}

#[test]
fn keywords_stopwords() {
    assert_eq!(keywords("What is the budget of Project Atlas?"), vec!["budget", "project", "atlas"]);
    assert_eq!(keywords("How much funding does Dr. Chen have?"), vec!["funding", "dr", "chen"]);
    assert_eq!(keywords("Server Node 3"), vec!["server", "node", "3"]);
    assert!(keywords("what is the").is_empty());
}

#[test]
fn generator_determinism() {
    let a = generate_session(7, 500, 20);
    let b = generate_session(7, 500, 20);
    assert_eq!(session_to_text(&a, 7, 500, 20), session_to_text(&b, 7, 500, 20));
    let c = generate_session(8, 500, 20);
    assert_ne!(session_to_text(&a, 7, 500, 20), session_to_text(&c, 8, 500, 20));
    assert_eq!(a.log.len(), 500);
    assert_eq!(a.questions.len(), 20);
}

#[test]
fn session_file_roundtrip() {
    let sess = generate_session(42, 200, 20);
    let text = session_to_text(&sess, 42, 200, 20);
    let back = session_from_text(&text).expect("parse");
    assert_eq!(back.log.lines, sess.log.lines);
    assert_eq!(back.questions.len(), sess.questions.len());
    for (a, b) in back.questions.iter().zip(sess.questions.iter()) {
        assert_eq!(a.text, b.text);
        assert_eq!(a.entity, b.entity);
        assert_eq!(a.attr, b.attr);
        assert_eq!(a.answer, b.answer);
        assert_eq!(a.step, b.step);
        assert_eq!(a.paraphrased, b.paraphrased);
        assert!((a.position - b.position).abs() < 1e-15);
    }
    assert!(session_from_text("Q\ttoo\tfew\n").is_err());
    assert!(session_from_text("X nonsense\n").is_err());
}
