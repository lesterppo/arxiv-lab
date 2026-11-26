//! Rust tests for the HoMe port (arXiv:2611.34624).
//!
//! * unit tests for each mechanism (world, memory, retrieval,
//!   answering, classification, failure attribution, localized
//!   edits, smoke stratification);
//! * property sweep over random worlds and harnesses;
//! * exhaustive check: the oracle is the max over all 256 harnesses
//!   and the loop never beats it;
//! * soak at a scale beyond the Python test (500 worlds);
//! * thread identity (stateless: same world on 4 threads, identical);
//! * Python-vs-Rust parity sweep on cases generated from Python
//!   (`parity_cases.txt`): 0 mismatches required.

use home::*;
use std::time::Instant;

const START: Harness = Harness {
    top_k: 3,
    recency: 0,
    revision: Revision::All,
    fuzzy: false,
    answer: AnswerRule::Majority,
};

fn small_world() -> Vec<Case> {
    make_world(7, 24)
}

// ---------------------------------------------------------------------------
// unit tests
// ---------------------------------------------------------------------------

#[test]
fn score_exact_beats_distractor() {
    // One case, hand-built: correct record + other-entity distractor.
    let case = Case {
        entity: 0, attribute: 0, query_attribute: 0, query_synonym: false,
        historical: false, target_session: 3, gold: 5,
        history: vec![
            Fact { entity: 0, attribute: 0, value: 5, session: 2 },
            Fact { entity: 1, attribute: 0, value: 7, session: 3 },
        ],
    };
    let cases = vec![case];
    let mem = build_memory(&cases, Revision::All);
    let r = evaluate(&Harness { top_k: 1, recency: 0, ..START }, &cases, &mem, 0);
    assert_eq!(r.accuracy, 1.0);
    assert_eq!(r.traces[0].answer, Some(5));
}

#[test]
fn score_synonym_needs_fuzzy() {
    let case = Case {
        entity: 0, attribute: 0, query_attribute: 0, query_synonym: true,
        historical: false, target_session: 3, gold: 5,
        history: vec![Fact { entity: 0, attribute: 0, value: 5, session: 2 }],
    };
    let cases = vec![case];
    let mem = build_memory(&cases, Revision::All);
    let plain = evaluate(&START, &cases, &mem, 0);
    assert_eq!(plain.traces[0].cause, Cause::Missing);
    let fuzzy = evaluate(&Harness { fuzzy: true, ..START }, &cases, &mem, 0);
    assert_eq!(fuzzy.traces[0].answer, Some(5));
}

#[test]
fn score_stale_wins_without_recency() {
    // Two versions of the same key; no recency, top1 -> oldest first.
    let case = Case {
        entity: 0, attribute: 0, query_attribute: 0, query_synonym: false,
        historical: false, target_session: 3, gold: 9,
        history: vec![
            Fact { entity: 0, attribute: 0, value: 4, session: 0 },
            Fact { entity: 0, attribute: 0, value: 9, session: 3 },
        ],
    };
    let cases = vec![case];
    let mem = build_memory(&cases, Revision::All);
    let stale = evaluate(&Harness { top_k: 1, recency: 0, ..START }, &cases, &mem, 0);
    assert_eq!(stale.traces[0].answer, Some(4));
    assert_eq!(stale.traces[0].cause, Cause::Stale);
    let fresh = evaluate(&Harness { top_k: 1, recency: 100, ..START }, &cases, &mem, 0);
    assert_eq!(fresh.traces[0].answer, Some(9));
}

#[test]
fn answer_majority_votes_stale() {
    // top3 + majority over {stale, stale, current} -> stale wins.
    let case = Case {
        entity: 0, attribute: 0, query_attribute: 0, query_synonym: false,
        historical: false, target_session: 3, gold: 9,
        history: vec![
            Fact { entity: 0, attribute: 0, value: 4, session: 0 },
            Fact { entity: 0, attribute: 0, value: 4, session: 1 },
            Fact { entity: 0, attribute: 0, value: 9, session: 3 },
        ],
    };
    let cases = vec![case];
    let mem = build_memory(&cases, Revision::All);
    let r = evaluate(&START, &cases, &mem, 0);
    assert_eq!(r.traces[0].answer, Some(4));
    assert_eq!(r.traces[0].cause, Cause::Stale);
}

#[test]
fn latest_only_breaks_historical() {
    let case = Case {
        entity: 0, attribute: 0, query_attribute: 0, query_synonym: false,
        historical: true, target_session: 1, gold: 4,
        history: vec![
            Fact { entity: 0, attribute: 0, value: 4, session: 1 },
            Fact { entity: 0, attribute: 0, value: 9, session: 3 },
        ],
    };
    let cases = vec![case];
    let mem_all = build_memory(&cases, Revision::All);
    let ok = evaluate(&Harness { top_k: 1, recency: 100, ..START }, &cases, &mem_all, 0);
    assert_eq!(ok.traces[0].answer, Some(4));
    let mem_latest = build_memory(&cases, Revision::Latest);
    let broken = evaluate(&Harness { top_k: 1, recency: 100, ..START }, &cases, &mem_latest, 0);
    assert_eq!(broken.traces[0].cause, Cause::Missing);
}

#[test]
fn localized_candidate_order() {
    let cands = localized_candidates(&START, &Cause::Stale);
    assert!(!cands.is_empty());
    assert_eq!(cands[0].recency, 100);
    assert_eq!(cands.last().unwrap().revision, Revision::Latest);
    let cands = localized_candidates(&START, &Cause::Missing);
    assert!(cands[0].fuzzy);
}

#[test]
fn dominant_cause_tie_breaks_missing() {
    // Equal STALE and MISSING counts -> Missing wins (alphabetical).
    let cases = vec![
        Case {
            entity: 0, attribute: 0, query_attribute: 0, query_synonym: true,
            historical: false, target_session: 3, gold: 5,
            history: vec![Fact { entity: 0, attribute: 0, value: 5, session: 2 }],
        },
        Case {
            entity: 0, attribute: 0, query_attribute: 0, query_synonym: false,
            historical: false, target_session: 3, gold: 9,
            history: vec![
                Fact { entity: 0, attribute: 0, value: 4, session: 0 },
                Fact { entity: 0, attribute: 0, value: 9, session: 3 },
            ],
        },
    ];
    let mem = build_memory(&cases, Revision::All);
    let r = evaluate(&START, &cases, &mem, 0);
    assert_eq!(dominant_cause(&r), Some(Cause::Missing));
}

#[test]
fn smoke_indices_cover_all_buckets() {
    let cases = make_world(0, 48);
    let idx = smoke_indices(&cases, SMOKE_N);
    assert_eq!(idx.len(), 8);
    assert!(idx.windows(2).all(|w| w[0] < w[1]), "indices sorted");
    let has_hist = idx.iter().any(|&i| cases[i].historical);
    let has_syn = idx.iter().any(|&i| cases[i].query_synonym);
    assert!(has_hist && has_syn, "smoke set covers historical + synonym");
}

// ---------------------------------------------------------------------------
// property sweep
// ---------------------------------------------------------------------------

#[test]
fn property_sweep() {
    let mut rng = Rng::new(999);
    for w in 0..60 {
        let cases = make_world(1000 + w, 32);
        // Every harness: accuracy in [0,1], tokens follow the formula.
        for h in all_harnesses().iter().step_by(17) {
            let r = evaluate_fresh(h, &cases);
            assert!((0.0..=1.0).contains(&r.accuracy));
            assert_eq!(r.mean_tokens, h.tokens());
            assert_eq!(r.n, 32);
            // Deterministic: same harness twice, identical traces.
            let r2 = evaluate_fresh(h, &cases);
            assert_eq!(r.accuracy, r2.accuracy);
            assert_eq!(r.mean_tokens, r2.mean_tokens);
            for (a, b) in r.traces.iter().zip(r2.traces.iter()) {
                assert_eq!(a.answer, b.answer);
                assert_eq!(a.cause, b.cause);
            }
        }
        // Oracle >= every harness; loop <= oracle; loop >= start.
        let (_, ores) = oracle_best(&cases);
        for h in all_harnesses().iter().step_by(31) {
            let r = evaluate_fresh(h, &cases);
            assert!(ores.accuracy >= r.accuracy - 1e-12);
            if (ores.accuracy - r.accuracy).abs() < 1e-12 {
                assert!(ores.mean_tokens <= r.mean_tokens + 1e-9);
            }
        }
        let lp = home_loop(&cases, &START, w as u64, N_FORMAL, true, true);
        assert!(lp.best_result.accuracy <= ores.accuracy + 1e-12);
        assert!(lp.best_result.accuracy >= lp.start_result.accuracy - 1e-12);
        assert!(lp.formal_evals <= N_FORMAL);
        let _ = rng.below(2);
    }
}

// ---------------------------------------------------------------------------
// exhaustive check
// ---------------------------------------------------------------------------

#[test]
fn exhaustive_oracle_is_max() {
    let cases = small_world();
    let (oh, ores) = oracle_best(&cases);
    // Independent brute force over the same space, recomputed here.
    let mut best_acc = -1.0f64;
    let mut best_tok = f64::INFINITY;
    for h in all_harnesses() {
        let r = evaluate_fresh(&h, &cases);
        if r.accuracy > best_acc
            || (r.accuracy == best_acc && r.mean_tokens < best_tok)
        {
            best_acc = r.accuracy;
            best_tok = r.mean_tokens;
        }
    }
    assert_eq!(ores.accuracy, best_acc);
    assert_eq!(ores.mean_tokens, best_tok);
    assert_eq!(oh, {
        // first harness attaining (best_acc, best_tok) in enum order
        all_harnesses().into_iter().find(|h| {
            let r = evaluate_fresh(h, &cases);
            r.accuracy == best_acc && r.mean_tokens == best_tok
        }).unwrap()
    });
}

// ---------------------------------------------------------------------------
// soak
// ---------------------------------------------------------------------------

#[test]
fn soak_loop_reaches_oracle() {
    let t0 = Instant::now();
    let n = 500;
    let mut reached = 0usize;
    let mut total_formal = 0usize;
    for s in 0..n {
        let cases = make_world(5000 + s as u64, 48);
        let lp = home_loop(&cases, &START, s as u64, N_FORMAL, true, true);
        let (_, ores) = oracle_best(&cases);
        if (lp.best_result.accuracy - ores.accuracy).abs() < 1e-12 {
            reached += 1;
        }
        total_formal += lp.formal_evals;
    }
    let rate = reached as f64 / n as f64;
    println!("soak: loop reaches oracle accuracy in {}/{} worlds ({:.3}), \
              mean formal evals {:.2}, {:.1}s",
             reached, n, rate, total_formal as f64 / n as f64,
             t0.elapsed().as_secs_f64());
    // Threshold calibrated against the Python cross-check in
    // ../soak_py.py (same worlds, same loop): Python reaches the
    // oracle at the same rate; the threshold sits 0.02 below it.
    assert!(rate >= 0.97, "soak rate {:.3} below 0.97", rate);
}

// ---------------------------------------------------------------------------
// thread identity
// ---------------------------------------------------------------------------

#[test]
fn thread_identity() {
    let cases = make_world(42, 48);
    let handles: Vec<_> = (0..4).map(|_| {
        let cases = cases.clone();
        std::thread::spawn(move || {
            let r = evaluate_fresh(&START, &cases);
            let lp = home_loop(&cases, &START, 3, N_FORMAL, true, true);
            let (oh, ores) = oracle_best(&cases);
            (r.accuracy, lp.best_result.accuracy, lp.best, oh,
             ores.accuracy, lp.formal_evals)
        })
    }).collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for r in &results[1..] {
        assert_eq!(*r, results[0], "thread results differ");
    }
}

// ---------------------------------------------------------------------------
// soak world dump (dev aid: prints the seed-5000 world in parity
// format; ../soak_py.py --check compares its Python replica of the
// Rust world generator against this dump field by field)
// ---------------------------------------------------------------------------

#[test]
fn dump_soak_world() {
    let cases = make_world(5000, 48);
    let mut parts: Vec<String> = vec!["5000".to_string(), "48".to_string()];
    for c in &cases {
        let facts: Vec<String> = c.history.iter().map(|f| {
            format!("{},{},{},{}", f.entity, f.attribute, f.value, f.session)
        }).collect();
        parts.push(format!(
            "{},{},{},{},{},{},{}:{}",
            c.entity, c.attribute, c.query_attribute,
            if c.query_synonym { 1 } else { 0 },
            if c.historical { 1 } else { 0 },
            c.target_session, c.gold,
            facts.join(";")));
    }
    println!("DUMP {}", parts.join(" "));
}

// ---------------------------------------------------------------------------
// parity sweep
// ---------------------------------------------------------------------------

fn parse_usize(s: &str) -> usize {
    s.parse().unwrap()
}

#[test]
fn parity_sweep() {
    let text = include_str!("parity_cases.txt");
    let mut worlds = 0usize;
    let mut mismatches = 0usize;
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(' ').collect();
        let seed: u64 = fields[0].parse().unwrap();
        let n: usize = parse_usize(fields[1]);
        assert_eq!(fields.len(), 2 + n + 2, "line {} field count", ln);
        let mut cases: Vec<Case> = Vec::with_capacity(n);
        for f in &fields[2..2 + n] {
            let (head, facts_s) = f.split_once(':').unwrap();
            let hp: Vec<&str> = head.split(',').collect();
            let mut history = Vec::new();
            if !facts_s.is_empty() {
                for fs in facts_s.split(';') {
                    let fp: Vec<&str> = fs.split(',').collect();
                    history.push(Fact {
                        entity: parse_usize(fp[0]),
                        attribute: parse_usize(fp[1]),
                        value: parse_usize(fp[2]),
                        session: parse_usize(fp[3]),
                    });
                }
            }
            cases.push(Case {
                entity: parse_usize(hp[0]),
                attribute: parse_usize(hp[1]),
                query_attribute: parse_usize(hp[2]),
                query_synonym: hp[3] == "1",
                historical: hp[4] == "1",
                target_session: parse_usize(hp[5]),
                gold: parse_usize(hp[6]),
                history,
            });
        }
        let loop_f: Vec<&str> = fields[2 + n].split(',').collect();
        let oracle_f: Vec<&str> = fields[3 + n].split(',').collect();
        let want_loop = Harness {
            top_k: parse_usize(loop_f[0]),
            recency: parse_usize(loop_f[1]) as u8,
            revision: if loop_f[2] == "1" { Revision::Latest } else { Revision::All },
            fuzzy: loop_f[3] == "1",
            answer: if loop_f[4] == "1" { AnswerRule::Majority } else { AnswerRule::Top1 },
        };
        let want_oracle = Harness {
            top_k: parse_usize(oracle_f[0]),
            recency: parse_usize(oracle_f[1]) as u8,
            revision: if oracle_f[2] == "1" { Revision::Latest } else { Revision::All },
            fuzzy: oracle_f[3] == "1",
            answer: if oracle_f[4] == "1" { AnswerRule::Majority } else { AnswerRule::Top1 },
        };
        let lp = home_loop(&cases, &START, seed, N_FORMAL, true, true);
        let (oh, ores) = oracle_best(&cases);
        let lp_correct = (lp.best_result.accuracy * n as f64).round() as usize;
        let or_correct = (ores.accuracy * n as f64).round() as usize;
        let mut bad: Vec<String> = Vec::new();
        if lp.best != want_loop {
            bad.push(format!("loop best {:?} != {:?}", lp.best, want_loop));
        }
        if lp_correct != parse_usize(loop_f[5]) {
            bad.push(format!("loop correct {} != {}", lp_correct, loop_f[5]));
        }
        if lp.best_result.mean_tokens as usize != parse_usize(loop_f[7]) {
            bad.push(format!("loop tokens {} != {}", lp.best_result.mean_tokens, loop_f[7]));
        }
        if lp.formal_evals != parse_usize(loop_f[8]) {
            bad.push(format!("loop formal {} != {}", lp.formal_evals, loop_f[8]));
        }
        if lp.smoke_evals != parse_usize(loop_f[9]) {
            bad.push(format!("loop smoke {} != {}", lp.smoke_evals, loop_f[9]));
        }
        if lp.smoke_rejects != parse_usize(loop_f[10]) {
            bad.push(format!("loop rejects {} != {}", lp.smoke_rejects, loop_f[10]));
        }
        if lp.candidates_tried != parse_usize(loop_f[11]) {
            bad.push(format!("loop tried {} != {}", lp.candidates_tried, loop_f[11]));
        }
        if oh != want_oracle {
            bad.push(format!("oracle best {:?} != {:?}", oh, want_oracle));
        }
        if or_correct != parse_usize(oracle_f[5]) {
            bad.push(format!("oracle correct {} != {}", or_correct, oracle_f[5]));
        }
        if ores.mean_tokens as usize != parse_usize(oracle_f[7]) {
            bad.push(format!("oracle tokens {} != {}", ores.mean_tokens, oracle_f[7]));
        }
        if !bad.is_empty() {
            mismatches += 1;
            if mismatches <= 3 {
                println!("line {} seed {} mismatches: {:?}", ln, seed, bad);
            }
        }
        worlds += 1;
    }
    println!("parity sweep: {} worlds, {} mismatches", worlds, mismatches);
    assert_eq!(worlds, 300, "expected 300 parity worlds");
    assert_eq!(mismatches, 0, "parity mismatches found");
}
