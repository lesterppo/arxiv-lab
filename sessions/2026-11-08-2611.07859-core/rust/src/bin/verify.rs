//! `verify` binary: mirrors `test_core_cpu.py` — runs E1, E1b, E2, E3 on
//! the Rust corpus, prints the report, checks assertions A1–A4, and
//! writes `results.json` (path from argv[1], default `results.json` in
//! the current directory; run from the `rust/` directory).
//!
//! Exit code 0 if all assertions pass, 1 otherwise.

use std::fs;
use std::process::ExitCode;

use core_rust::corpus::{build_corpus, build_questions, SEED};
use core_rust::experiments::{
    assertions, results_json, run_e1, run_e1b, run_e2, run_e3, sorted_institutions, verdict,
    JACCARD_THRESHOLD, K, SNIPPET_WIDTH,
};
use core_rust::mechanism::{Bm25, SnippetExtractor};

fn main() -> ExitCode {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "results.json".to_string());

    let entities = build_corpus(120, SEED);
    let (two_hop, single_hop) = build_questions(&entities, 60, 30, SEED);
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
    let extractor = SnippetExtractor::new(&all_sentences);

    let e1 = run_e1(&bm25, &extractor, &entities, &two_hop, &institutions);
    let e1b = run_e1b(&bm25, &extractor, &entities, &single_hop, &institutions);
    let e2 = run_e2(&bm25, &extractor, &entities, &two_hop);
    let e3 = run_e3(&bm25, &extractor, &entities, &two_hop);

    let rec_gap = e1.full.recall_both - e1.core.recall_both;
    let acc_gap = e1.full.accuracy - e1.core.accuracy;
    let tok_ratio = e1.core.mean_tokens / e1.full.mean_tokens;

    println!("{}", "=".repeat(70));
    println!("CORE (arXiv 2611.07859) — Rust port replication results");
    println!("{}", "=".repeat(70));
    println!("E1 two-hop (n={}, k={}):", e1.n, K);
    for (name, d) in [("FULL", &e1.full), ("CORE", &e1.core), ("NAIVE", &e1.naive)] {
        println!(
            "  {:6} recall_both={:.3} accuracy={:.3} mean_tokens={:.1}",
            name, d.recall_both, d.accuracy, d.mean_tokens
        );
    }
    println!(
        "  CORE recall gap vs FULL: {:+.3} | accuracy gap: {:+.3} | token ratio: {:.3}",
        rec_gap, acc_gap, tok_ratio
    );
    println!(
        "E1b single-hop control (n={}): FULL acc={:.3} ({:.1} tok) | CORE acc={:.3} ({:.1} tok)",
        e1b.n, e1b.full.accuracy, e1b.full.mean_tokens, e1b.core.accuracy, e1b.core.mean_tokens
    );
    println!("E2 multi-turn (n={} trajectories):", e2.n);
    println!(
        "  FULL cum tokens={:.1} | CORE no-CM={:.1} | CORE+CM={:.1} (ratio vs FULL {:.3})",
        e2.full_cum_tokens,
        e2.core_no_cm_cum_tokens,
        e2.core_cm_cum_tokens,
        e2.cm_token_ratio_vs_full
    );
    println!(
        "  suppression rate={:.3} ({}/{} seen slots; expanded {})",
        e2.suppression_rate, e2.n_suppressed, e2.n_seen_slots, e2.n_expanded
    );
    println!(
        "  turn-2 evidence coverage: CORE no-CM {:.3} | CORE+CM {:.3}",
        e2.turn2_evidence_coverage_core_no_cm, e2.turn2_evidence_coverage_core_cm
    );
    println!(
        "E3 snippet ablation (evidence recall, n={}): token_sim={:.3} | first_sentence={:.3} | random_window={:.3}",
        e3.n, e3.token_sim, e3.first_sentence, e3.random_window
    );
    println!("{}", "-".repeat(70));
    let checks = assertions(&e1, &e2);
    let mut all_pass = true;
    for (label, ok) in &checks {
        println!("  [{}] {}", if *ok { "PASS" } else { "FAIL" }, label);
        all_pass = all_pass && *ok;
    }
    println!("VERDICT (Rust corpus): {}", verdict(&e1, &e2));
    println!("{}", "=".repeat(70));

    let json = results_json(&e1, &e1b, &e2, &e3);
    match fs::write(&out_path, json) {
        Ok(()) => println!("wrote {}", out_path),
        Err(e) => {
            eprintln!("failed to write {}: {}", out_path, e);
            return ExitCode::from(1);
        }
    }
    let _ = (SNIPPET_WIDTH, JACCARD_THRESHOLD);
    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
