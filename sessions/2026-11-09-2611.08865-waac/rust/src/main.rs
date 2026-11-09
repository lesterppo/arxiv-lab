//! `waac-verify`: re-run the Python CPU test's experiment in Rust.
//!
//! Same 60 seeded workflows, same 20 dev workflows for the history
//! library, same 7 methods, same assertions A1-A7, same `results.json`
//! shape. Run from the `rust/` directory:
//!
//!   cargo run --release --bin waac-verify
//!   # or: ./target/release/waac-verify [output path]
//!
//! The raw output is Python-shaped (byte-identical to the Python test's
//! `results.json` when compared — see README.md); it defaults to
//! `verify_results.json` so it never clobbers the curated `results.json`
//! record, which also carries the deep-test and parity numbers.
//!
//! Exit 0 iff all assertions pass.

use std::collections::{BTreeMap, HashMap};
use std::process::ExitCode;
use waac_rs::corpus::{build_corpus, build_dev_corpus, domain_words, Workflow};
use waac_rs::executor::{
    aggregate, build_history, run_workflow, Aggregate, RunResult, METHODS,
};
use waac_rs::json::{dumps_pretty, obj, Json};
use waac_rs::mt19937::PyRandom;
use waac_rs::policy::{fuse, Need, Policy};
use waac_rs::waac::{
    canonical_key, effective_caps, make_fragment, project_field, Field,
    HistoryLibrary,
};

// ------------------------------------------------------------ property checks
struct FusionCheck {
    ok: bool,
    n_pairs: usize,
    n_random_triples: usize,
    summary_pointer_join: String,
}

fn check_fusion_properties() -> FusionCheck {
    // A4: fusion never relaxes an online requirement. Mirrors the Python
    // check, including its Random(99) triple stream.
    let mut rng = PyRandom::new(99);
    let mut n_pairs = 0usize;
    let mut n_triples = 0usize;
    let mut ok = true;
    for a in Policy::ALL {
        for b in Policy::ALL {
            let f = fuse(a, b);
            n_pairs += 1;
            if !(a.caps() & !f.caps() == 0 && b.caps() & !f.caps() == 0) {
                ok = false;
            }
            if fuse(a, b) != fuse(b, a) {
                ok = false;
            }
            if fuse(a, Policy::Drop) != a || fuse(Policy::Drop, a) != a {
                ok = false;
            }
        }
    }
    if fuse(Policy::Summary, Policy::Pointer) != Policy::Exact {
        ok = false;
    }
    for _ in 0..200 {
        let a = *rng.choice(&Policy::ALL);
        let b = *rng.choice(&Policy::ALL);
        let c = *rng.choice(&Policy::ALL);
        n_triples += 1;
        if fuse(fuse(a, b), c) != fuse(a, fuse(b, c)) {
            ok = false;
        }
        if !(a.caps() & !fuse(a, b).caps() == 0) {
            ok = false;
        }
    }
    FusionCheck {
        ok,
        n_pairs,
        n_random_triples: n_triples,
        summary_pointer_join: fuse(Policy::Summary, Policy::Pointer).name().to_string(),
    }
}

struct CoarseCheck {
    ok: bool,
    kind: Option<String>,
    score: Option<f64>,
    different_op_sequence_matches: bool,
}

fn check_coarse_match(lib: &HistoryLibrary) -> CoarseCheck {
    // A7: same structure, different wiring -> coarse match; different op
    // sequence -> no match.
    let entry = lib.entries.iter().find(|e| !e.fragment.edges.is_empty());
    let Some(entry) = entry else {
        return CoarseCheck {
            ok: false,
            kind: None,
            score: None,
            different_op_sequence_matches: false,
        };
    };
    let mut frag = entry.fragment.clone();
    frag.edges = frag.edges[1..].to_vec(); // rewire: drop one edge
    if canonical_key(&frag) == entry.key {
        return CoarseCheck {
            ok: false,
            kind: None,
            score: None,
            different_op_sequence_matches: false,
        };
    }
    let m = lib.match_fragment(&frag, 0.5);
    let coarse_ok = matches!(&m, Some((_, score, kind)) if *kind == "coarse" && *score >= 0.5);
    let mut frag2 = entry.fragment.clone();
    frag2.ops = frag2.ops.iter().rev().cloned().collect();
    let none_ok = lib.match_fragment(&frag2, 0.5).is_none();
    CoarseCheck {
        ok: coarse_ok && none_ok,
        kind: m.as_ref().map(|(_, _, k)| k.to_string()),
        score: m.as_ref().map(|(_, s, _)| *s),
        different_op_sequence_matches: !none_ok,
    }
}

struct AdmissionCheck {
    ok: bool,
    unverified_rejected: bool,
    verified_admitted: bool,
    failed_run_rejected: Option<bool>,
    library_size_before: usize,
    library_size_after: usize,
    sabotaged_workflow_failed: bool,
}

fn check_history_admission(dev_workflows: &[Workflow], lib: &mut HistoryLibrary) -> AdmissionCheck {
    // A5: only verified-successful trajectories become precedents.
    let before = lib.len();
    // (i) an explicitly unverified fragment is rejected.
    let wf = &dev_workflows[0];
    let dag = &wf.dag;
    let frag = make_fragment(dag, 0, wf.n_ops.min(4), &domain_words(wf));
    let rejected_unverified = !lib.add_fragment(
        frag.clone(),
        &BTreeMap::new(),
        false,
        "unverified-probe",
    );
    // (ii) a failed execution never becomes a precedent: find a dev
    // workflow that fails under a zero-budget compressor and confirm
    // verified=false blocks admission of its fragments.
    let mut failed_wf: Option<&Workflow> = None;
    for cand in dev_workflows {
        let zeros = vec![0usize; cand.n_ops + 1];
        let r = run_workflow(cand, "text_importance", None, Some(&zeros));
        if !r.success {
            failed_wf = Some(cand);
            break;
        }
    }
    let mut sabotage_rejected: Option<bool> = None;
    if let Some(failed_wf) = failed_wf {
        let fdag = &failed_wf.dag;
        let ffrag = make_fragment(fdag, 0, failed_wf.n_ops.min(4), &domain_words(failed_wf));
        sabotage_rejected = Some(!lib.add_fragment(
            ffrag,
            &BTreeMap::new(),
            false,
            "failed-run",
        ));
    }
    // (iii) a verified fragment IS admitted.
    let mut pols: BTreeMap<(usize, String), Policy> = BTreeMap::new();
    pols.insert((0, "result_url".to_string()), Policy::Exact);
    let admitted_verified =
        lib.add_fragment(frag, &pols, true, "verified-probe");
    AdmissionCheck {
        ok: rejected_unverified
            && admitted_verified
            && sabotage_rejected != Some(false)
            && lib.len() >= before,
        unverified_rejected: rejected_unverified,
        verified_admitted: admitted_verified,
        failed_run_rejected: sabotage_rejected,
        library_size_before: before,
        library_size_after: lib.len(),
        sabotaged_workflow_failed: failed_wf.is_some(),
    }
}

struct PointerCheck {
    ok: bool,
    valid_pointer_renders: String,
    invalid_pointer_falls_back_to_exact: bool,
    invalid_pointer_caps: Vec<String>,
}

fn check_pointer_fallback() -> PointerCheck {
    // A6: a pointer without a valid reference falls back to EXACT.
    let words: Vec<String> = (0..60).map(|i| format!("word{i}")).collect();
    let field = Field::new(
        "report_body",
        "ReportBody",
        words.join(" "),
        true,
        Some("art_missing".to_string()),
    );
    let mut runtime_ok: HashMap<String, String> = HashMap::new();
    runtime_ok.insert("art_missing".to_string(), field.value.clone());
    let (text_ok, fb_ok) = project_field(&field, Policy::Pointer, &runtime_ok);
    let runtime_empty: HashMap<String, String> = HashMap::new();
    let (text_bad, fb_bad) = project_field(&field, Policy::Pointer, &runtime_empty);
    let caps_bad = effective_caps(&field, Policy::Pointer, &runtime_empty);
    let mut caps_names: Vec<String> = Vec::new();
    if caps_bad & 0b001 != 0 {
        caps_names.push("literal".to_string());
    }
    if caps_bad & 0b010 != 0 {
        caps_names.push("semantic".to_string());
    }
    if caps_bad & 0b100 != 0 {
        caps_names.push("recoverable".to_string());
    }
    caps_names.sort();
    PointerCheck {
        ok: text_ok.as_deref() == Some("<ptr:art_missing>")
            && !fb_ok
            && text_bad.as_deref() == Some(field.value.as_str())
            && fb_bad
            && caps_bad == Policy::Exact.caps(),
        valid_pointer_renders: text_ok.unwrap_or_default(),
        invalid_pointer_falls_back_to_exact: fb_bad,
        invalid_pointer_caps: caps_names,
    }
}

// ------------------------------------------------------------------- main
fn agg_json(a: &Aggregate) -> Json {
    obj(&[
        ("n", Json::Int(a.n as i64)),
        ("success_rate", Json::Float(a.success_rate)),
        ("steps_completed_frac", Json::Float(a.steps_completed_frac)),
        ("critical_retention", Json::Float(a.critical_retention)),
        ("min_step_retention", Json::Float(a.min_step_retention)),
        ("mean_tokens_per_step", Json::Float(a.mean_tokens_per_step)),
        ("n_fallbacks", Json::Int(a.n_fallbacks as i64)),
    ])
}

fn main() -> ExitCode {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "verify_results.json".to_string());
    let workflows = build_corpus(60, 20261109);
    let dev = build_dev_corpus(20, 777);
    let (mut lib, n_dev_ok, n_dev_bad) = build_history(&dev);

    // WAAC-full first: its per-step tokens are the matched budget for the
    // selective baselines.
    let mut per_method: BTreeMap<&str, Vec<RunResult>> = BTreeMap::new();
    let mut waac_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
    for method in ["waac_full", "waac_online", "no_compression"] {
        let res: Vec<RunResult> = workflows
            .iter()
            .map(|wf| run_workflow(wf, method, Some(&lib), None))
            .collect();
        if method == "waac_full" {
            for (wf, r) in workflows.iter().zip(res.iter()) {
                waac_tokens.insert(wf.wid, r.tok_per_step.clone());
            }
        }
        per_method.insert(method, res);
    }
    for method in ["keep_last_n", "keep_last_5", "text_importance", "summary_all"] {
        let res: Vec<RunResult> = workflows
            .iter()
            .map(|wf| {
                let toks = waac_tokens.get(&wf.wid).unwrap();
                run_workflow(wf, method, Some(&lib), Some(toks))
            })
            .collect();
        per_method.insert(method, res);
    }

    let mut agg: BTreeMap<&str, Aggregate> = BTreeMap::new();
    for m in METHODS {
        agg.insert(m, aggregate(&per_method[m]));
    }
    let n_wf = workflows.len();
    let no_comp = &agg["no_compression"];
    let waac = &agg["waac_full"];
    let token_reduction =
        1.0 - waac.mean_tokens_per_step / no_comp.mean_tokens_per_step;

    // History match stats over WAAC-full projections.
    let (mut m_exact, mut m_coarse, mut m_none) = (0usize, 0usize, 0usize);
    for r in &per_method["waac_full"] {
        m_exact += r.matches_exact;
        m_coarse += r.matches_coarse;
        m_none += r.matches_none;
    }
    let n_lookups = m_exact + m_coarse + m_none;
    let match_rate = if n_lookups > 0 {
        (m_exact + m_coarse) as f64 / n_lookups as f64
    } else {
        0.0
    };

    // ---- property checks
    let fusion = check_fusion_properties();
    let coarse = check_coarse_match(&lib);
    let admission = check_history_admission(&dev, &mut lib);
    let pointer = check_pointer_fallback();

    // ---- assertions
    let waac_success_n = per_method["waac_full"].iter().filter(|r| r.success).count();
    let nocomp_success_n = per_method["no_compression"].iter().filter(|r| r.success).count();
    let a1 = per_method["waac_full"]
        .iter()
        .all(|r| r.ret_min_step >= 1.0 - 1e-9)
        && per_method["waac_online"]
            .iter()
            .all(|r| r.ret_min_step >= 1.0 - 1e-9);
    let a2 = (waac_success_n as i64 - nocomp_success_n as i64).abs() <= 1
        && waac.mean_tokens_per_step < no_comp.mean_tokens_per_step;
    let mut a3_bits: BTreeMap<&str, bool> = BTreeMap::new();
    for m in ["keep_last_n", "keep_last_5", "text_importance", "summary_all"] {
        let beats = waac.success_rate > agg[m].success_rate;
        let fewest = waac.success_rate >= no_comp.success_rate - 1e-9
            && waac.mean_tokens_per_step < agg[m].mean_tokens_per_step;
        a3_bits.insert(m, beats || fewest);
    }
    let a3 = a3_bits.values().all(|b| *b);
    let a4 = fusion.ok;
    let a5 = admission.ok;
    let a6 = pointer.ok;
    let a7 = coarse.ok;
    let checks: Vec<(&str, bool)> = vec![
        ("A1 WAAC retains 100% of gold-required fields at every step", a1),
        ("A2 WAAC success == no-compression (within 1 workflow) with strictly fewer tokens", a2),
        ("A3 WAAC beats each baseline at matched budget (or fewest tokens at no-compression success)", a3),
        ("A4 fusion never relaxes an online requirement (SUMMARY+POINTER -> EXACT)", a4),
        ("A5 history admits only verified-successful trajectories", a5),
        ("A6 invalid pointer falls back to EXACT", a6),
        ("A7 coarse historical match works (same structure, different wiring -> coarse; different ops -> none)", a7),
    ];
    let all_pass = checks.iter().all(|(_, ok)| *ok);

    // Verdict capped at SUPPORTED (same honest limits as the Python test:
    // synthetic corpus, TF-IDF proxy, stub SUMMARY, no LLM in the loop).
    let verdict = if all_pass {
        "SUPPORTED"
    } else if !a1 || !a2 {
        "NEGATIVE"
    } else {
        "PARTIAL"
    };

    // per_baseline_a3 in the Python insertion order (keep_last_n,
    // keep_last_5, text_importance, summary_all), not sorted order.
    let a3_order = ["keep_last_n", "keep_last_5", "text_importance", "summary_all"];
    let a3_json: Vec<(String, Json)> = a3_order
        .iter()
        .map(|m| (m.to_string(), Json::Bool(a3_bits[m])))
        .collect();
    let mut methods_obj: Vec<(String, Json)> = Vec::new();
    for m in METHODS {
        methods_obj.push((m.to_string(), agg_json(&agg[m])));
    }
    let results = obj(&[
        ("paper", Json::Str("arXiv 2611.08865 WAAC: Rethinking Agent Trajectory Compression: A Workflow-Aware Perspective".to_string())),
        ("config", obj(&[
            ("n_workflows", Json::Int(n_wf as i64)),
            ("n_dev_workflows", Json::Int(dev.len() as i64)),
            ("ops_per_workflow", Json::Str("7-13 (observed; generator aims 6-14)".to_string())),
            ("summary_stub_tokens", Json::Int(25)),
            ("history_match_threshold", Json::Float(0.5)),
            ("history_window_ops", Json::Int(4)),
            ("matched_budget", Json::Str("per-(workflow, step) WAAC-full tokens".to_string())),
            ("semantic_similarity", Json::Str("TF-IDF cosine over op/field/domain tokens (CPU proxy for embeddings)".to_string())),
            ("summary", Json::Str("stub: first 25 tokens of long text; short scalars retained whole".to_string())),
        ])),
        ("history", obj(&[
            ("library_entries", Json::Int(lib.len() as i64)),
            ("dev_workflows_verified", Json::Int(n_dev_ok as i64)),
            ("dev_workflows_failed", Json::Int(n_dev_bad as i64)),
            ("lookups", obj(&[
                ("exact", Json::Int(m_exact as i64)),
                ("coarse", Json::Int(m_coarse as i64)),
                ("none", Json::Int(m_none as i64)),
            ])),
            ("match_rate", Json::Float(match_rate)),
        ])),
        ("methods", Json::Obj(methods_obj)),
        ("token_reduction_vs_no_compression", Json::Float(token_reduction)),
        ("per_baseline_a3", Json::Obj(a3_json)),
        ("fusion_property_check", obj(&[
            ("ok", Json::Bool(fusion.ok)),
            ("n_pairs", Json::Int(fusion.n_pairs as i64)),
            ("n_random_triples", Json::Int(fusion.n_random_triples as i64)),
            ("summary_pointer_join", Json::Str(fusion.summary_pointer_join)),
        ])),
        ("history_coarse_match_check", obj(&[
            ("ok", Json::Bool(coarse.ok)),
            ("coarse_match_kind", coarse.kind.map(Json::Str).unwrap_or(Json::Null)),
            ("coarse_match_score", coarse.score.map(Json::Float).unwrap_or(Json::Null)),
            ("different_op_sequence_matches", Json::Bool(coarse.different_op_sequence_matches)),
        ])),
        ("history_admission_check", obj(&[
            ("ok", Json::Bool(admission.ok)),
            ("unverified_rejected", Json::Bool(admission.unverified_rejected)),
            ("verified_admitted", Json::Bool(admission.verified_admitted)),
            ("failed_run_rejected", admission.failed_run_rejected.map(Json::Bool).unwrap_or(Json::Null)),
            ("library_size_before", Json::Int(admission.library_size_before as i64)),
            ("library_size_after", Json::Int(admission.library_size_after as i64)),
            ("sabotaged_workflow_failed", Json::Bool(admission.sabotaged_workflow_failed)),
        ])),
        ("pointer_fallback_check", obj(&[
            ("ok", Json::Bool(pointer.ok)),
            ("valid_pointer_renders", Json::Str(pointer.valid_pointer_renders)),
            ("invalid_pointer_falls_back_to_exact", Json::Bool(pointer.invalid_pointer_falls_back_to_exact)),
            ("invalid_pointer_caps", Json::Arr(pointer.invalid_pointer_caps.into_iter().map(Json::Str).collect())),
        ])),
        ("assertions", Json::Obj(checks.iter().map(|(l, ok)| (l.to_string(), Json::Bool(*ok))).collect())),
        ("verdict", Json::Str(verdict.to_string())),
    ]);

    println!("{}", "=".repeat(72));
    println!("WAAC (arXiv 2611.08865) -- Rust replication results");
    println!("{}", "=".repeat(72));
    println!(
        "workflows: {n_wf} (7-13 ops observed) | dev workflows for history: {} ({n_dev_ok} verified, {n_dev_bad} failed)",
        dev.len()
    );
    println!(
        "history library: {} fragments | match rate {:.3} (exact {m_exact}, coarse {m_coarse}, none {m_none})",
        lib.len(),
        match_rate
    );
    println!(
        "{:16} {:>8} {:>6} {:>9} {:>9}",
        "method", "success", "steps", "crit-ret", "tok/step"
    );
    for m in METHODS {
        let d = &agg[m];
        println!(
            "{m:16} {:8.3} {:6.3} {:9.3} {:9.1}",
            d.success_rate, d.steps_completed_frac, d.critical_retention, d.mean_tokens_per_step
        );
    }
    println!(
        "WAAC token reduction vs no-compression: {:.1}%",
        token_reduction * 100.0
    );
    println!(
        "WAAC-full vs WAAC-online-only tokens/step: {:.1} vs {:.1} (history adds retention, same success)",
        waac.mean_tokens_per_step, agg["waac_online"].mean_tokens_per_step
    );
    println!("{}", "-".repeat(72));
    for (label, ok) in &checks {
        println!("  [{}] {label}", if *ok { "PASS" } else { "FAIL" });
    }
    println!("VERDICT: {verdict}");
    println!("{}", "=".repeat(72));

    // Python `json.dump` writes no trailing newline.
    std::fs::write(&out_path, dumps_pretty(&results))
        .unwrap_or_else(|e| panic!("cannot write {out_path}: {e}"));
    println!("wrote {out_path}");
    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

// Keep `Need` in scope for the doc reference in checks (unused import guard).
#[allow(dead_code)]
fn _need_ref(_: Need) {}
