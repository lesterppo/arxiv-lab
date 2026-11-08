//! Experiments E1/E1b/E2/E3 mirroring `test_core_cpu.py` in the parent
//! session directory, plus JSON serialization of the results.
//!
//! The deterministic extractive reader, the experiment loop orders, and
//! the assertion thresholds are the same as Python. The corpus itself
//! differs (xorshift RNG instead of Mersenne Twister), so the measured
//! numbers are the Rust corpus's own — compare them against the Python
//! numbers as a sanity check, not as an exact match.

use std::collections::BTreeSet;

use crate::corpus::{Entity, Question};
use crate::mechanism::{count_tokens, tokenize, Bm25, CmAction, ContextManager, SnippetExtractor};
use crate::rng::XorShift64;

pub const K: usize = 5;
pub const SNIPPET_WIDTH: usize = 2;
pub const JACCARD_THRESHOLD: f64 = 0.8;

// ---------------------------------------------------------------- reader

fn find_all(hay: &[String], needle: &[String]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..=(hay.len() - needle.len()) {
        if hay[i..i + needle.len()] == needle[..] {
            out.push(i);
        }
    }
    out
}

/// Deterministic extractive reader: pick the institution string whose
/// closest mention is nearest (in tokens) to the researcher's full name
/// (fallback: last name). Score 1/(1+min_distance); absent -> None.
/// Strict `>` keeps the first institution on ties.
pub fn reader_answer(
    context: &str,
    full_name: &str,
    last_name: &str,
    institutions: &[String],
) -> Option<String> {
    let ctx = tokenize(context);
    let mut name_pos = find_all(&ctx, &tokenize(full_name));
    if name_pos.is_empty() {
        name_pos = find_all(&ctx, &tokenize(last_name));
    }
    if name_pos.is_empty() {
        return None;
    }
    let mut best: Option<String> = None;
    let mut best_score = -1.0f64;
    for inst in institutions {
        let ipos = find_all(&ctx, &tokenize(inst));
        if ipos.is_empty() {
            continue;
        }
        let d = ipos
            .iter()
            .flat_map(|a| name_pos.iter().map(move |b| (*a as i64 - *b as i64).abs()))
            .min()
            .unwrap() as f64;
        let score = 1.0 / (1.0 + d);
        if score > best_score {
            best_score = score;
            best = Some(inst.clone());
        }
    }
    best
}

/// Both gold evidence sentences present in the returned context?
pub fn recall_both(context: &str, entity: &Entity) -> bool {
    context.contains(&entity.gold_hop1) && context.contains(&entity.gold_hop2)
}

// ---------------------------------------------------------------- results

#[derive(Debug, Clone)]
pub struct MethodAgg {
    pub recall_both: f64,
    pub accuracy: f64,
    pub mean_tokens: f64,
}

#[derive(Debug, Clone)]
pub struct E1Result {
    pub full: MethodAgg,
    pub core: MethodAgg,
    pub naive: MethodAgg,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct ControlAgg {
    pub accuracy: f64,
    pub mean_tokens: f64,
}

#[derive(Debug, Clone)]
pub struct E1bResult {
    pub full: ControlAgg,
    pub core: ControlAgg,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct E2Result {
    pub n: usize,
    pub full_cum_tokens: f64,
    pub core_no_cm_cum_tokens: f64,
    pub core_cm_cum_tokens: f64,
    pub cm_token_ratio_vs_full: f64,
    pub suppression_rate: f64,
    pub n_seen_slots: usize,
    pub n_suppressed: usize,
    pub n_expanded: usize,
    pub turn2_evidence_coverage_core_no_cm: f64,
    pub turn2_evidence_coverage_core_cm: f64,
}

#[derive(Debug, Clone)]
pub struct E3Result {
    pub token_sim: f64,
    pub first_sentence: f64,
    pub random_window: f64,
    pub n: usize,
}

/// Sorted unique institution list, as in `test_core_cpu.py`.
pub fn sorted_institutions(entities: &[Entity]) -> Vec<String> {
    entities
        .iter()
        .map(|e| e.institution.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

// ---------------------------------------------------------------- E1

pub fn run_e1(
    bm25: &Bm25,
    extractor: &SnippetExtractor,
    entities: &[Entity],
    two_hop: &[Question],
    institutions: &[String],
) -> E1Result {
    let mut full_rec = 0usize;
    let mut full_corr = 0usize;
    let mut full_tok = 0usize;
    let mut core_rec = 0usize;
    let mut core_corr = 0usize;
    let mut core_tok = 0usize;
    let mut naive_rec = 0usize;
    let mut naive_corr = 0usize;
    let mut naive_tok = 0usize;
    for q in two_hop {
        let e = &entities[q.entity_idx];
        let ranked = bm25.search(&q.text, K);
        let passages: Vec<&Entity> = ranked.iter().map(|(i, _)| &entities[*i]).collect();
        let full_ctx = passages
            .iter()
            .map(|p| p.passage.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let mut core_snips = Vec::new();
        let mut naive_snips = Vec::new();
        for p in &passages {
            let (_, s) = extractor.best_window(&p.sentences, &q.text, SNIPPET_WIDTH);
            core_snips.push(s);
            // NAIVE-SNIPPET per spec: the FIRST SENTENCE only (width 1).
            let (_, s2) = extractor.first_window(&p.sentences, 1);
            naive_snips.push(s2);
        }
        let core_ctx = core_snips.join(" ");
        let naive_ctx = naive_snips.join(" ");
        if recall_both(&full_ctx, e) {
            full_rec += 1;
        }
        if reader_answer(&full_ctx, &e.name, &e.last, institutions).as_deref()
            == Some(e.institution.as_str())
        {
            full_corr += 1;
        }
        full_tok += count_tokens(&full_ctx);
        if recall_both(&core_ctx, e) {
            core_rec += 1;
        }
        if reader_answer(&core_ctx, &e.name, &e.last, institutions).as_deref()
            == Some(e.institution.as_str())
        {
            core_corr += 1;
        }
        core_tok += count_tokens(&core_ctx);
        if recall_both(&naive_ctx, e) {
            naive_rec += 1;
        }
        if reader_answer(&naive_ctx, &e.name, &e.last, institutions).as_deref()
            == Some(e.institution.as_str())
        {
            naive_corr += 1;
        }
        naive_tok += count_tokens(&naive_ctx);
    }
    let n = two_hop.len() as f64;
    E1Result {
        full: MethodAgg {
            recall_both: full_rec as f64 / n,
            accuracy: full_corr as f64 / n,
            mean_tokens: full_tok as f64 / n,
        },
        core: MethodAgg {
            recall_both: core_rec as f64 / n,
            accuracy: core_corr as f64 / n,
            mean_tokens: core_tok as f64 / n,
        },
        naive: MethodAgg {
            recall_both: naive_rec as f64 / n,
            accuracy: naive_corr as f64 / n,
            mean_tokens: naive_tok as f64 / n,
        },
        n: two_hop.len(),
    }
}

// ---------------------------------------------------------------- E1b

pub fn run_e1b(
    bm25: &Bm25,
    extractor: &SnippetExtractor,
    entities: &[Entity],
    single_hop: &[Question],
    institutions: &[String],
) -> E1bResult {
    let mut out = E1bResult {
        full: ControlAgg {
            accuracy: 0.0,
            mean_tokens: 0.0,
        },
        core: ControlAgg {
            accuracy: 0.0,
            mean_tokens: 0.0,
        },
        n: single_hop.len(),
    };
    for name in ["FULL", "CORE"] {
        let mut correct = 0usize;
        let mut toks = 0usize;
        for q in single_hop {
            let e = &entities[q.entity_idx];
            let ranked = bm25.search(&q.text, K);
            let ctx = if name == "FULL" {
                ranked
                    .iter()
                    .map(|(i, _)| entities[*i].passage.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                let mut snips = Vec::new();
                for (i, _) in &ranked {
                    let (_, s) =
                        extractor.best_window(&entities[*i].sentences, &q.text, SNIPPET_WIDTH);
                    snips.push(s);
                }
                snips.join(" ")
            };
            if reader_answer(&ctx, &e.name, &e.last, institutions).as_deref()
                == Some(e.institution.as_str())
            {
                correct += 1;
            }
            toks += count_tokens(&ctx);
        }
        let agg = ControlAgg {
            accuracy: correct as f64 / single_hop.len() as f64,
            mean_tokens: toks as f64 / single_hop.len() as f64,
        };
        if name == "FULL" {
            out.full = agg;
        } else {
            out.core = agg;
        }
    }
    out
}

// ---------------------------------------------------------------- E2

pub fn run_e2(
    bm25: &Bm25,
    extractor: &SnippetExtractor,
    entities: &[Entity],
    two_hop: &[Question],
) -> E2Result {
    let trajs = &two_hop[..30.min(two_hop.len())];
    let mut full_cum = 0usize;
    let mut core_cum = 0usize;
    let mut cm_cum = 0usize;
    let mut n_seen_slots = 0usize;
    let mut n_suppressed = 0usize;
    let mut n_expanded = 0usize;
    let mut cm_evidence = 0usize;
    let mut core_evidence = 0usize;
    for q in trajs {
        let e = &entities[q.entity_idx];
        let t1 = format!(
            "Which researcher won the {} in {} in the field of {}?",
            e.award, e.year, e.field
        );
        let t2 = format!(
            "Which institution is {}, the {} researcher, affiliated with?",
            e.name, e.field
        );
        let r1 = bm25.search(&t1, K);
        let r2 = bm25.search(&t2, K);

        full_cum += r1
            .iter()
            .map(|(i, _)| count_tokens(&entities[*i].passage))
            .sum::<usize>();
        full_cum += r2
            .iter()
            .map(|(i, _)| count_tokens(&entities[*i].passage))
            .sum::<usize>();

        let mut cm = ContextManager::new(JACCARD_THRESHOLD);
        let mut t1_snips: Vec<String> = Vec::new();
        let mut t1_tokens = 0usize;
        for (i, _) in &r1 {
            let (_, s) = extractor.best_window(&entities[*i].sentences, &t1, SNIPPET_WIDTH);
            t1_snips.push(s.clone());
            cm.register(&entities[*i].id, &s);
            t1_tokens += count_tokens(&s);
        }
        core_cum += t1_tokens;
        cm_cum += t1_tokens;

        let mut t2_snips: Vec<String> = Vec::new();
        let mut t2_adds: Vec<String> = Vec::new();
        for (i, _) in &r2 {
            let p = &entities[*i];
            let (_, fresh) = extractor.best_window(&p.sentences, &t2, SNIPPET_WIDTH);
            t2_snips.push(fresh.clone());
            let (action, text) = cm.turn2_decision(&p.id, &fresh, &p.gold_hop2);
            if cm.was_seen(&p.id) {
                n_seen_slots += 1;
                match action {
                    CmAction::Suppress => n_suppressed += 1,
                    CmAction::Expand => n_expanded += 1,
                    CmAction::Fresh => {}
                }
            }
            if matches!(action, CmAction::Fresh | CmAction::Expand) {
                t2_adds.push(text);
            }
        }
        core_cum += t2_snips.iter().map(|s| count_tokens(s)).sum::<usize>();
        cm_cum += t2_adds.iter().map(|s| count_tokens(s)).sum::<usize>();

        let mut cm_parts = t1_snips.clone();
        cm_parts.extend(t2_adds.iter().cloned());
        let cm_ctx = cm_parts.join(" ");
        let mut core_parts = t1_snips.clone();
        core_parts.extend(t2_snips.iter().cloned());
        let core_ctx = core_parts.join(" ");
        if cm_ctx.contains(&e.gold_hop2) {
            cm_evidence += 1;
        }
        if core_ctx.contains(&e.gold_hop2) {
            core_evidence += 1;
        }
    }
    let n = trajs.len() as f64;
    E2Result {
        n: trajs.len(),
        full_cum_tokens: full_cum as f64 / n,
        core_no_cm_cum_tokens: core_cum as f64 / n,
        core_cm_cum_tokens: cm_cum as f64 / n,
        cm_token_ratio_vs_full: (cm_cum as f64 / n) / (full_cum as f64 / n),
        suppression_rate: if n_seen_slots > 0 {
            n_suppressed as f64 / n_seen_slots as f64
        } else {
            0.0
        },
        n_seen_slots,
        n_suppressed,
        n_expanded,
        turn2_evidence_coverage_core_no_cm: core_evidence as f64 / n,
        turn2_evidence_coverage_core_cm: cm_evidence as f64 / n,
    }
}

// ---------------------------------------------------------------- E3

pub fn run_e3(
    bm25: &Bm25,
    extractor: &SnippetExtractor,
    entities: &[Entity],
    two_hop: &[Question],
) -> E3Result {
    // Same seed (7) and draw order as Python's random.Random(7): one draw
    // per passage, only in the random_window branch, questions in order.
    // The stream differs (xorshift vs Mersenne Twister) — same distribution.
    let mut rng = XorShift64::new(7);
    let mut token_sim = 0usize;
    let mut first_sentence = 0usize;
    let mut random_window = 0usize;
    let n = two_hop.len();
    for q in two_hop {
        let e = &entities[q.entity_idx];
        let ranked = bm25.search(&q.text, K);
        for name in ["token_sim", "first_sentence", "random_window"] {
            let mut parts: Vec<String> = Vec::new();
            for (i, _) in &ranked {
                let sents = &entities[*i].sentences;
                let s = match name {
                    "token_sim" => extractor.best_window(sents, &q.text, SNIPPET_WIDTH).1,
                    "first_sentence" => extractor.first_window(sents, 1).1,
                    _ => {
                        let start = rng.next_below(sents.len() - SNIPPET_WIDTH + 1);
                        crate::mechanism::window_text(
                            sents,
                            &(start..start + SNIPPET_WIDTH).collect::<Vec<_>>(),
                        )
                    }
                };
                parts.push(s);
            }
            if recall_both(&parts.join(" "), e) {
                match name {
                    "token_sim" => token_sim += 1,
                    "first_sentence" => first_sentence += 1,
                    _ => random_window += 1,
                }
            }
        }
    }
    E3Result {
        token_sim: token_sim as f64 / n as f64,
        first_sentence: first_sentence as f64 / n as f64,
        random_window: random_window as f64 / n as f64,
        n,
    }
}

// ---------------------------------------------------------------- verdict

/// Same verdict logic as `test_core_cpu.py`, evaluated on the Rust
/// corpus's numbers. This is the Rust leg's own check — the paper verdict
/// stays with the Python run (SUPPORTED).
pub fn verdict(e1: &E1Result, e2: &E2Result) -> &'static str {
    let rec_gap = e1.full.recall_both - e1.core.recall_both;
    let acc_gap = e1.full.accuracy - e1.core.accuracy;
    let tok_ratio = e1.core.mean_tokens / e1.full.mean_tokens;
    if (rec_gap.abs() <= 0.01
        && acc_gap.abs() <= 0.01
        && tok_ratio <= 0.5
        && e2.core_cm_cum_tokens < e2.core_no_cm_cum_tokens)
        || (rec_gap <= 0.05 && acc_gap <= 0.05 && tok_ratio <= 0.5)
    {
        "SUPPORTED"
    } else if rec_gap > 0.05 || acc_gap > 0.05 || tok_ratio > 0.5 {
        "NEGATIVE"
    } else {
        "PARTIAL"
    }
}

/// Pre-registered assertions A1–A4 (same thresholds as Python).
pub fn assertions(e1: &E1Result, e2: &E2Result) -> Vec<(&'static str, bool)> {
    vec![
        (
            "A1 CORE recall within 5pp of FULL",
            e1.full.recall_both - e1.core.recall_both <= 0.05,
        ),
        (
            "A2 CORE mean tokens <= 50% of FULL",
            e1.core.mean_tokens / e1.full.mean_tokens <= 0.5,
        ),
        (
            "A3 CORE recall beats NAIVE by >= 10pp",
            e1.core.recall_both - e1.naive.recall_both >= 0.10,
        ),
        (
            "A4 CORE+CM cum tokens < CORE no-CM (E2)",
            e2.core_cm_cum_tokens < e2.core_no_cm_cum_tokens,
        ),
    ]
}

// ---------------------------------------------------------------- JSON out

fn fnum(x: f64) -> String {
    // Shortest round-trip representation, matching Python json's repr
    // closely enough for a human-readable results file.
    if x == x.trunc() && x.abs() < 1e15 {
        format!("{:.1}", x)
    } else {
        format!("{}", x)
    }
}

fn agg_json(a: &MethodAgg) -> String {
    format!(
        "{{\"recall_both\": {}, \"accuracy\": {}, \"mean_tokens\": {}}}",
        fnum(a.recall_both),
        fnum(a.accuracy),
        fnum(a.mean_tokens)
    )
}

/// Serialize the full experiment results, mirroring the Python
/// `results.json` structure.
pub fn results_json(e1: &E1Result, e1b: &E1bResult, e2: &E2Result, e3: &E3Result) -> String {
    let v = verdict(e1, e2);
    let checks = assertions(e1, e2);
    let checks_json = checks
        .iter()
        .map(|(label, ok)| format!("    \"{}\": {}", label, ok))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        r#"{{
  "paper": "arXiv 2611.07859 CORE: Coarse-to-Fine Search with Context Management for Deep Research",
  "implementation": "Rust port (stdlib only) of the Python CPU proxy in the parent session directory",
  "config": {{
    "k": {},
    "snippet_width_sentences": {},
    "jaccard_threshold": {},
    "n_entities": 120,
    "retriever": "BM25 (CPU proxy for dense retriever)",
    "snippet_scorer": "TF-IDF cosine (CPU proxy for late-interaction reranker token sims)"
  }},
  "E1_two_hop": {{
    "FULL": {},
    "CORE": {},
    "NAIVE": {},
    "n": {}
  }},
  "E1b_single_hop_control": {{
    "FULL": {{"accuracy": {}, "mean_tokens": {}}},
    "CORE": {{"accuracy": {}, "mean_tokens": {}}},
    "n": {}
  }},
  "E2_multi_turn": {{
    "n": {},
    "FULL_cum_tokens": {},
    "CORE_noCM_cum_tokens": {},
    "CORE_CM_cum_tokens": {},
    "CM_token_ratio_vs_FULL": {},
    "suppression_rate": {},
    "n_seen_slots": {},
    "n_suppressed": {},
    "n_expanded": {},
    "turn2_evidence_coverage_CORE_noCM": {},
    "turn2_evidence_coverage_CORE_CM": {}
  }},
  "E3_snippet_ablation": {{
    "token_sim": {},
    "first_sentence": {},
    "random_window": {},
    "n": {}
  }},
  "assertions": {{
{}
  }},
  "verdict": "{}"
}}"#,
        K,
        SNIPPET_WIDTH,
        JACCARD_THRESHOLD,
        agg_json(&e1.full),
        agg_json(&e1.core),
        agg_json(&e1.naive),
        e1.n,
        fnum(e1b.full.accuracy),
        fnum(e1b.full.mean_tokens),
        fnum(e1b.core.accuracy),
        fnum(e1b.core.mean_tokens),
        e1b.n,
        e2.n,
        fnum(e2.full_cum_tokens),
        fnum(e2.core_no_cm_cum_tokens),
        fnum(e2.core_cm_cum_tokens),
        fnum(e2.cm_token_ratio_vs_full),
        fnum(e2.suppression_rate),
        e2.n_seen_slots,
        e2.n_suppressed,
        e2.n_expanded,
        fnum(e2.turn2_evidence_coverage_core_no_cm),
        fnum(e2.turn2_evidence_coverage_core_cm),
        fnum(e3.token_sim),
        fnum(e3.first_sentence),
        fnum(e3.random_window),
        e3.n,
        checks_json,
        v
    )
}
