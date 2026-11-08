//! Extensive tests for the Rust CORE port: unit tests per mechanism,
//! property sweeps, a brute-force BM25 oracle, a soak at a scale the
//! Python test does not reach, and thread-identity for the stateless
//! scorers.

use std::sync::Arc;
use std::time::Instant;

use core_rust::corpus::{build_corpus, build_questions, SEED};
use core_rust::experiments::{
    recall_both, run_e1, run_e2, run_e3, sorted_institutions, K, SNIPPET_WIDTH,
};
use core_rust::mechanism::{
    content_tokens, count_tokens, jaccard, tokenize, Bm25, CmAction, ContextManager,
    SnippetExtractor,
};
use core_rust::rng::XorShift64;

// ------------------------------------------------------------------ helpers

fn make_extractor(sentences: &[&str]) -> SnippetExtractor {
    SnippetExtractor::new(&sentences.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}

/// Independent brute-force BM25 reference: recomputes df / idf / tf
/// straight from the passage strings, in a separate code path from
/// `Bm25`, and returns the full ranking.
fn brute_bm25(passages: &[String], query: &str) -> Vec<(usize, f64)> {
    use std::collections::{HashMap, HashSet};
    let docs: Vec<Vec<String>> = passages.iter().map(|p| content_tokens(p)).collect();
    let n = docs.len() as f64;
    let avgdl = docs.iter().map(|d| d.len()).sum::<usize>() as f64 / docs.len().max(1) as f64;
    let qt = content_tokens(query);
    let mut out = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        if doc.is_empty() {
            out.push((i, 0.0));
            continue;
        }
        let mut s = 0.0;
        // Sum in first-occurrence query order (deduped), the same order
        // `Bm25::score` uses. The previous version iterated a HashSet,
        // whose order is randomized per process (RandomState), so two
        // documents with scores equal to the last ulp could flip order
        // between runs and the exact ranking assertion failed flakily
        // (4/6 runs). The oracle still recomputes df/idf/tf from the
        // passage strings in an independent code path; only the float
        // summation order is aligned, so scores are bit-identical.
        let mut seen_terms: Vec<&String> = Vec::new();
        for t in &qt {
            if !seen_terms.contains(&t) {
                seen_terms.push(t);
            }
        }
        for t in seen_terms {
            let df = docs.iter().filter(|d| d.contains(t)).count() as f64;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            let f = doc.iter().filter(|x| *x == t).count() as f64;
            if f == 0.0 {
                continue;
            }
            let dl = doc.len() as f64;
            s += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * dl / avgdl));
        }
        out.push((i, s));
    }
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

const WORD_POOL: &[&str] = &[
    "quantum",
    "photon",
    "qubit",
    "coral",
    "plankton",
    "magnetar",
    "exoplanet",
    "parser",
    "lexicon",
    "dialogue",
    "gripper",
    "quadruped",
    "swarm",
    "enzyme",
    "protein",
    "carbon",
    "cloud",
    "glacier",
    "memory",
    "neural",
    "reward",
    "ceramic",
    "battery",
    "polymer",
    "prime",
    "sieve",
    "residue",
    "researcher",
    "institute",
    "university",
    "college",
    "award",
    "medal",
    "prize",
    "fellowship",
    "discovery",
    "laboratory",
    "experiment",
    "theory",
    "survey",
    "circuit",
    "feedback",
    "dynamics",
    "genomics",
    "ecology",
    "optics",
    "lattice",
];

fn random_sentence(rng: &mut XorShift64, pool: &[&str]) -> String {
    let n = 4 + rng.next_below(7);
    let words: Vec<&str> = (0..n).map(|_| pool[rng.next_below(pool.len())]).collect();
    let mut s = words.join(" ");
    s.push('.');
    // Capitalize the first letter, as real sentences do.
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
        None => s,
    }
}

fn random_passages(rng: &mut XorShift64, n: usize) -> (Vec<String>, Vec<String>) {
    let mut passages = Vec::with_capacity(n);
    let mut all_sentences = Vec::new();
    for _ in 0..n {
        let n_sent = 4 + rng.next_below(5);
        let sents: Vec<String> = (0..n_sent)
            .map(|_| random_sentence(rng, WORD_POOL))
            .collect();
        all_sentences.extend(sents.iter().cloned());
        passages.push(sents.join(" "));
    }
    (passages, all_sentences)
}

// ------------------------------------------------------------------ unit

#[test]
fn tokenize_and_stopwords() {
    assert_eq!(
        tokenize("Hello, World! 42 is the answer."),
        vec!["hello", "world", "42", "is", "the", "answer"]
    );
    assert_eq!(tokenize(""), Vec::<String>::new());
    assert_eq!(tokenize("..."), Vec::<String>::new());
    // Stopwords removed for retrieval tokens.
    assert_eq!(
        content_tokens("Which institution is the researcher affiliated with?"),
        vec!["institution", "researcher"]
    );
    assert_eq!(count_tokens("one two  three\nfour"), 4);
    assert_eq!(count_tokens(""), 0);
}

#[test]
fn bm25_ranks_exact_match_first() {
    let passages = vec![
        "the cat sat on the mat".to_string(),
        "quantum computing qubits coherence".to_string(),
        "the dog barked loudly".to_string(),
    ];
    let bm25 = Bm25::new(passages, 1.5, 0.75);
    let top = bm25.search("quantum qubits", 3);
    assert_eq!(top[0].0, 1);
    assert!(top[0].1 > 0.0);
    // Empty query scores all zero; ties broken by lower index.
    let top_empty = bm25.search("zzzzqqqq", 3);
    assert_eq!(
        top_empty.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert!(top_empty.iter().all(|(_, s)| *s == 0.0));
    // k larger than the corpus truncates.
    assert_eq!(bm25.search("quantum", 10).len(), 3);
}

#[test]
fn bm25_score_matches_oracle() {
    let mut rng = XorShift64::new(99);
    let (passages, _) = random_passages(&mut rng, 40);
    let bm25 = Bm25::new(passages.clone(), 1.5, 0.75);
    for qi in 0..10 {
        let query = format!(
            "{} {} {}",
            WORD_POOL[rng.next_below(WORD_POOL.len())],
            WORD_POOL[rng.next_below(WORD_POOL.len())],
            WORD_POOL[rng.next_below(WORD_POOL.len())]
        );
        let got = bm25.search(&query, 40);
        let want = brute_bm25(&passages, &query);
        assert_eq!(got.len(), want.len(), "query {}", qi);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.0, w.0, "query {} ranking", qi);
            assert!(
                (g.1 - w.1).abs() <= 1e-9 * w.1.abs().max(1.0),
                "query {} score: rust {} oracle {}",
                qi,
                g.1,
                w.1
            );
        }
    }
}

#[test]
fn snippet_window_picks_evidence_sentence() {
    let sentences = vec![
        "The committee met on Tuesday to discuss budgets.".to_string(),
        "Dr Alvarez won the Rosenfeld Prize in 2021 for quantum computing.".to_string(),
        "The laboratory ordered new glassware last week.".to_string(),
        "Rain is expected over the weekend.".to_string(),
    ];
    let ext = make_extractor(&sentences.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let (idx, text) = ext.best_window(
        &sentences,
        "Which researcher won the Rosenfeld Prize in 2021?",
        2,
    );
    // The evidence sentence (index 1) must be inside the best window.
    assert!(idx.contains(&1), "best window {:?} misses sentence 1", idx);
    assert!(text.contains("Rosenfeld Prize"));
    // First-window ablation returns the first two sentences.
    let (fidx, _) = ext.first_window(&sentences, 2);
    assert_eq!(fidx, vec![0, 1]);
    // Width-1 window on a single evidence sentence.
    let (idx1, _) = ext.best_window(&sentences, "Rosenfeld Prize 2021", 1);
    assert_eq!(idx1, vec![1]);
}

#[test]
fn jaccard_values() {
    let a = content_tokens("alpha beta gamma");
    let b = content_tokens("beta gamma delta");
    assert!((jaccard(&a, &b) - 0.5).abs() < 1e-12);
    assert!((jaccard(&a, &a) - 1.0).abs() < 1e-12);
    let empty: Vec<String> = vec![];
    assert!((jaccard(&empty, &empty) - 1.0).abs() < 1e-12);
    assert!((jaccard(&a, &empty) - 0.0).abs() < 1e-12);
}

#[test]
fn context_manager_decisions() {
    let mut cm = ContextManager::new(0.8);
    // Unseen passage -> fresh.
    let (a, t) = cm.turn2_decision("p1", "fresh snippet here", "answer sentence here");
    assert_eq!(a, CmAction::Fresh);
    assert_eq!(t, "fresh snippet here");
    assert!(!cm.was_seen("p1"));

    cm.register("p1", "alpha beta gamma delta epsilon zeta");
    assert!(cm.was_seen("p1"));
    // Near-duplicate fresh snippet -> suppress.
    let (a, t) = cm.turn2_decision(
        "p1",
        "alpha beta gamma delta epsilon zeta eta",
        "some other sentence",
    );
    assert_eq!(a, CmAction::Suppress);
    assert_eq!(t, "");
    assert!(cm.is_duplicate("alpha beta gamma delta epsilon zeta"));
    assert!(!cm.is_duplicate("completely different words here now"));
    // New snippet, answer already in the returned snippet -> suppress.
    let (a, _) = cm.turn2_decision(
        "p1",
        "totally new words never seen before",
        "alpha beta gamma delta epsilon zeta",
    );
    assert_eq!(a, CmAction::Suppress);
    // New snippet, answer missing -> expand with the answer sentence.
    let (a, t) = cm.turn2_decision(
        "p1",
        "totally new words never seen before",
        "the missing answer sentence",
    );
    assert_eq!(a, CmAction::Expand);
    assert_eq!(t, "the missing answer sentence");
}

#[test]
fn corpus_shape() {
    let entities = build_corpus(120, SEED);
    assert_eq!(entities.len(), 120);
    for e in &entities {
        assert_eq!(e.sentences.len(), 7);
        assert_eq!(e.gold_hop2, e.sentences[0]);
        assert_eq!(e.gold_hop1, e.sentences[1]);
        assert!(e.passage.starts_with(&e.sentences[0]));
    }
    let (two, one) = build_questions(&entities, 60, 30, SEED);
    assert!(!two.is_empty() && two.len() <= 60);
    assert_eq!(one.len(), 30);
    // Two-hop questions are unique per entity.
    let mut ids: Vec<&str> = two.iter().map(|q| q.entity_id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), two.len());
}

// ------------------------------------------------------------------ property

#[test]
fn property_topk_monotone_in_k() {
    let mut rng = XorShift64::new(1234);
    let (passages, _) = random_passages(&mut rng, 60);
    let bm25 = Bm25::new(passages, 1.5, 0.75);
    for _ in 0..50 {
        let query = format!(
            "{} {}",
            WORD_POOL[rng.next_below(WORD_POOL.len())],
            WORD_POOL[rng.next_below(WORD_POOL.len())]
        );
        let k = 1 + rng.next_below(10);
        let small = bm25.search(&query, k);
        let big = bm25.search(&query, k + 3);
        assert_eq!(small.len(), k.min(60));
        assert_eq!(&big[..small.len()], small.as_slice());
    }
}

#[test]
fn property_best_window_is_argmax_window() {
    // The best 2-sentence window must be the argmax over all windows when
    // each window is scored independently (window text scored as one
    // bag of tokens). This checks the selection logic, not the scorer.
    let mut rng = XorShift64::new(77);
    let (passages, all_sentences) = random_passages(&mut rng, 30);
    let ext = SnippetExtractor::new(&all_sentences);
    for _ in 0..100 {
        let passage = &passages[rng.next_below(passages.len())];
        // Split the passage back into sentences on ". " boundaries.
        let sents: Vec<String> = passage
            .split(". ")
            .map(|s| s.trim_end_matches('.').to_string() + ".")
            .collect();
        let query = format!(
            "{} {}",
            WORD_POOL[rng.next_below(WORD_POOL.len())],
            WORD_POOL[rng.next_below(WORD_POOL.len())]
        );
        let (idx, _) = ext.best_window(&sents, &query, 2);
        let got_score = ext.sentence_cosine(
            &idx.iter()
                .map(|&i| sents[i].as_str())
                .collect::<Vec<_>>()
                .join(" "),
            &query,
        );
        // Independent argmax over all width-2 windows.
        let n_starts = sents.len().saturating_sub(1).max(1);
        let mut best = f64::NEG_INFINITY;
        for start in 0..n_starts {
            let end = (start + 2).min(sents.len());
            let text = (start..end)
                .map(|i| sents[i].as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let sc = ext.sentence_cosine(&text, &query);
            if sc > best {
                best = sc;
            }
        }
        assert!(
            (got_score - best).abs() <= 1e-12,
            "best window {:?} scores {} but independent argmax is {}",
            idx,
            got_score,
            best
        );
        // And the best window must contain at least one sentence with
        // above-zero similarity whenever any sentence has any.
        let any_positive = sents.iter().any(|s| ext.sentence_cosine(s, &query) > 0.0);
        if any_positive {
            assert!(
                idx.iter()
                    .any(|&i| ext.sentence_cosine(&sents[i], &query) > 0.0),
                "best window {:?} contains no similar sentence",
                idx
            );
        }
    }
}

#[test]
fn property_suppression_is_idempotent() {
    let mut rng = XorShift64::new(55);
    for _ in 0..100 {
        let mut cm = ContextManager::new(0.8);
        let snippet = random_sentence(&mut rng, WORD_POOL);
        cm.register("p", &snippet);
        let first = cm.turn2_decision("p", &snippet, "other sentence");
        let second = cm.turn2_decision("p", &snippet, "other sentence");
        assert_eq!(first, second);
        assert_eq!(first.0, CmAction::Suppress);
        // Registering the same snippet again changes nothing.
        cm.register("p", &snippet);
        let third = cm.turn2_decision("p", &snippet, "other sentence");
        assert_eq!(third, first);
    }
}

#[test]
fn property_token_count_is_additive() {
    let mut rng = XorShift64::new(31);
    for _ in 0..200 {
        let a = random_sentence(&mut rng, WORD_POOL);
        let b = random_sentence(&mut rng, WORD_POOL);
        assert_eq!(
            count_tokens(&format!("{} {}", a, b)),
            count_tokens(&a) + count_tokens(&b)
        );
        // Empty parts add nothing.
        assert_eq!(count_tokens(&format!("{} ", a)), count_tokens(&a));
    }
}

#[test]
fn property_bm25_scores_are_finite_and_ordered() {
    let mut rng = XorShift64::new(2026);
    let (passages, _) = random_passages(&mut rng, 100);
    let bm25 = Bm25::new(passages, 1.5, 0.75);
    for _ in 0..50 {
        let query = (0..3)
            .map(|_| WORD_POOL[rng.next_below(WORD_POOL.len())])
            .collect::<Vec<_>>()
            .join(" ");
        let top = bm25.search(&query, 10);
        assert!(!top.is_empty());
        assert!(top.iter().all(|(_, s)| s.is_finite() && *s >= 0.0));
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }
}

// ------------------------------------------------------------------ soak

#[test]
fn soak_5000_passages_200_questions() {
    let mut rng = XorShift64::new(4242);
    let (passages, all_sentences) = random_passages(&mut rng, 5000);
    let start = Instant::now();
    let bm25 = Bm25::new(passages.clone(), 1.5, 0.75);
    let ext = SnippetExtractor::new(&all_sentences);
    let build_ms = start.elapsed().as_millis();
    let mut total_hits = 0usize;
    for qi in 0..200 {
        let query = (0..4)
            .map(|_| WORD_POOL[rng.next_below(WORD_POOL.len())])
            .collect::<Vec<_>>()
            .join(" ");
        let top = bm25.search(&query, K);
        assert_eq!(top.len(), K);
        total_hits += top.len();
        // Snippet extraction on the top passage, as the verify binary does.
        let sents: Vec<String> = passages[top[0].0]
            .split(". ")
            .map(|s| s.trim_end_matches('.').to_string() + ".")
            .collect();
        let (idx, _) = ext.best_window(&sents, &query, SNIPPET_WIDTH);
        assert!(!idx.is_empty());
        let _ = qi;
    }
    let total_ms = start.elapsed().as_millis();
    println!(
        "soak: 5000 passages / 200 questions: index build {} ms, total {} ms, {} top-k hits",
        build_ms, total_ms, total_hits
    );
    assert_eq!(total_hits, 200 * K);
}

// ------------------------------------------------------------------ threads

#[test]
fn thread_identity_stateless_scorers() {
    let mut rng = XorShift64::new(808);
    let (passages, all_sentences) = random_passages(&mut rng, 200);
    let bm25 = Arc::new(Bm25::new(passages.clone(), 1.5, 0.75));
    let ext = Arc::new(SnippetExtractor::new(&all_sentences));
    let queries: Vec<String> = (0..20)
        .map(|_| {
            (0..3)
                .map(|_| WORD_POOL[rng.next_below(WORD_POOL.len())])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let queries = Arc::new(queries);
    let mut handles = Vec::new();
    for _ in 0..8 {
        let bm25 = Arc::clone(&bm25);
        let ext = Arc::clone(&ext);
        let queries = Arc::clone(&queries);
        let passages = passages.clone();
        handles.push(std::thread::spawn(move || {
            let mut results = Vec::new();
            for q in queries.iter() {
                let top = bm25.search(q, K);
                let sents: Vec<String> = passages[top[0].0]
                    .split(". ")
                    .map(|s| s.trim_end_matches('.').to_string() + ".")
                    .collect();
                let (idx, text) = ext.best_window(&sents, q, SNIPPET_WIDTH);
                results.push((top, idx, text));
            }
            results
        }));
    }
    let mut all: Vec<Vec<(Vec<(usize, f64)>, Vec<usize>, String)>> = Vec::new();
    for h in handles {
        all.push(h.join().expect("thread panicked"));
    }
    for other in all.iter().skip(1) {
        assert_eq!(&all[0], other, "thread results differ");
    }
}

// ------------------------------------------------------- end-to-end sanity

#[test]
fn end_to_end_experiments_run_and_assertions_hold() {
    let entities = build_corpus(120, SEED);
    let (two_hop, _) = build_questions(&entities, 60, 30, SEED);
    let institutions = sorted_institutions(&entities);
    let all_sentences: Vec<String> = entities
        .iter()
        .flat_map(|e| e.sentences.iter().cloned())
        .collect();
    let bm25 = Bm25::new(
        entities.iter().map(|e| e.passage.clone()).collect(),
        1.5,
        0.75,
    );
    let ext = SnippetExtractor::new(&all_sentences);
    let e1 = run_e1(&bm25, &ext, &entities, &two_hop, &institutions);
    let e2 = run_e2(&bm25, &ext, &entities, &two_hop);
    let e3 = run_e3(&bm25, &ext, &entities, &two_hop);
    // Same assertions as the verify binary.
    assert!(
        e1.full.recall_both - e1.core.recall_both <= 0.05,
        "A1 failed: {:?}",
        e1
    );
    assert!(
        e1.core.mean_tokens / e1.full.mean_tokens <= 0.5,
        "A2 failed: {:?}",
        e1
    );
    assert!(
        e1.core.recall_both - e1.naive.recall_both >= 0.10,
        "A3 failed: {:?}",
        e1
    );
    assert!(
        e2.core_cm_cum_tokens < e2.core_no_cm_cum_tokens,
        "A4 failed: {:?}",
        e2
    );
    // Evidence: the ablation must show token-sim >> random on this corpus.
    assert!(e3.token_sim > e3.random_window);
    // Spot-check the reader is sane: recall is computed on real contexts.
    assert!(recall_both(
        &entities[two_hop[0].entity_idx].passage,
        &entities[two_hop[0].entity_idx]
    ));
}
