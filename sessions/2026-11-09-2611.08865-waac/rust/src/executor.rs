//! Deterministic recoverability executor + baselines (port of
//! `test_waac_cpu.py`).
//!
//! The executor sees ONLY the projected visible context (+ the runtime
//! state for pointer resolution), never the full trajectory. A step
//! succeeds iff every field it needs is recoverable from the visible
//! context at the required fidelity.

use crate::corpus::{domain_words, Workflow};
use crate::policy::Policy;
use crate::py_sum;
use crate::waac::{
    fuse_policies, make_fragment, obligations_for_step, online_requirements,
    project_context, FieldKey, HistoryLibrary, TraceDAG,
};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Method names, mirroring the Python test.
pub const METHODS: [&str; 7] = [
    "no_compression",
    "waac_full",
    "waac_online",
    "keep_last_n",
    "keep_last_5",
    "text_importance",
    "summary_all",
];

/// Python `re.findall(r"[a-z0-9]+", text.lower())`.
pub fn tokens(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_ascii_alphanumeric() {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// [(op_idx, field_name, &Field)] produced by ops < step, in order.
pub fn visible_fields<'a>(
    dag: &'a TraceDAG,
    step: usize,
) -> Vec<(usize, String, &'a crate::waac::Field)> {
    let mut out = Vec::new();
    for i in 0..step {
        for field in &dag.nodes[i].outputs {
            out.push((i, field.name.clone(), field));
        }
    }
    out
}

/// Policy assignment for the visible fields at `step`.
///
/// Returns (policies, match_info) with match_info (kind, score) or None
/// for the WAAC-full history lookup. `budget` is the per-step token cap
/// for the budget-matched baselines (WAAC-full's own tokens at this
/// step).
pub fn policies_for_step(
    method: &str,
    wf: &Workflow,
    step: usize,
    history: Option<&HistoryLibrary>,
    budget: Option<usize>,
) -> (BTreeMap<FieldKey, Policy>, Option<(String, f64)>) {
    let dag = &wf.dag;
    let visible = visible_fields(dag, step);
    match method {
        "no_compression" => (
            visible.iter().map(|(i, f, _)| ((*i, f.clone()), Policy::Exact)).collect(),
            None,
        ),
        "summary_all" => (
            visible.iter().map(|(i, f, _)| ((*i, f.clone()), Policy::Summary)).collect(),
            None,
        ),
        "waac_full" | "waac_online" => {
            let obligations = obligations_for_step(dag, &wf.final_needs, step);
            let online = online_requirements(dag, &obligations);
            let mut proposal: BTreeMap<FieldKey, Policy> = BTreeMap::new();
            let mut match_info = None;
            if method == "waac_full" && history.is_some() && step > 0 {
                let start = step.saturating_sub(4);
                let frag = make_fragment(dag, start, step, &domain_words(wf));
                let lib = history.unwrap();
                if let Some((idx, score, kind)) = lib.match_fragment(&frag, 0.5) {
                    match_info = Some((kind.to_string(), score));
                    let entry = &lib.entries[idx];
                    for ((rel, fname), pol) in &entry.policies {
                        let oi = start + rel;
                        if oi < step && dag.nodes[oi].output(fname).is_some() {
                            proposal.insert((oi, fname.clone()), *pol);
                        }
                    }
                }
            }
            let fused = fuse_policies(&online, Some(&proposal));
            let pols = visible
                .iter()
                .map(|(i, f, _)| {
                    ((*i, f.clone()), fused.get(&(*i, f.clone())).copied().unwrap_or(Policy::Drop))
                })
                .collect();
            (pols, match_info)
        }
        "keep_last_5" => {
            let keep_from = step.saturating_sub(5);
            (
                visible
                    .iter()
                    .map(|(i, f, _)| {
                        ((*i, f.clone()), if *i >= keep_from { Policy::Exact } else { Policy::Drop })
                    })
                    .collect(),
                None,
            )
        }
        "keep_last_n" | "text_importance" => {
            let cap = budget.unwrap_or(0);
            let mut ranked: Vec<(usize, String, &crate::waac::Field)> =
                visible.iter().map(|(i, f, fld)| (*i, f.clone(), *fld)).collect();
            if method == "keep_last_n" {
                // Most recent first (op recency, then field order in the op).
                ranked.sort_by(|a, b| {
                    b.0.cmp(&a.0).then_with(|| {
                        let ia = dag.nodes[a.0]
                            .outputs
                            .iter()
                            .position(|f| f.name == a.1)
                            .unwrap();
                        let ib = dag.nodes[b.0]
                            .outputs
                            .iter()
                            .position(|f| f.name == b.1)
                            .unwrap();
                        ia.cmp(&ib)
                    })
                });
            } else {
                let mut obl_text = wf.task_text.clone();
                for j in step..dag.len() {
                    let names: Vec<&str> = dag.nodes[j]
                        .inputs
                        .iter()
                        .map(|(n, _)| n.as_str())
                        .collect();
                    obl_text.push(' ');
                    obl_text.push_str(&names.join(" "));
                }
                let fnames: Vec<&str> = wf
                    .final_needs
                    .iter()
                    .map(|(_, f, _)| f.as_str())
                    .collect();
                obl_text.push(' ');
                obl_text.push_str(&fnames.join(" "));
                let obl_toks: HashSet<String> = tokens(&obl_text).into_iter().collect();
                let mut scored: Vec<(i64, i64, usize, (usize, String, &crate::waac::Field))> =
                    Vec::new();
                for (pos, k) in ranked.iter().enumerate() {
                    let overlap = tokens(&k.2.value)
                        .into_iter()
                        .collect::<HashSet<String>>()
                        .intersection(&obl_toks)
                        .count() as i64;
                    scored.push((-overlap, -(k.0 as i64), pos, (k.0, k.1.clone(), k.2)));
                }
                scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
                ranked = scored.into_iter().map(|(_, _, _, k)| k).collect();
            }
            let mut pols: BTreeMap<FieldKey, Policy> = BTreeMap::new();
            let mut spent = 0usize;
            for k in &ranked {
                let cost = k.2.token_count();
                if spent + cost <= cap {
                    pols.insert((k.0, k.1.clone()), Policy::Exact);
                    spent += cost;
                } else {
                    pols.insert((k.0, k.1.clone()), Policy::Drop);
                }
            }
            (pols, None)
        }
        other => panic!("unknown method {other}"),
    }
}

/// Outcome of one workflow run under one method.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub success: bool,
    pub steps_done: usize,
    pub n_ops: usize,
    pub tok_per_step: Vec<usize>,
    pub retention: f64,
    pub ret_min_step: f64,
    pub matches_exact: usize,
    pub matches_coarse: usize,
    pub matches_none: usize,
    pub n_fallbacks: usize,
}

/// Deterministic executor: sees only the projected visible context.
pub fn run_workflow(
    wf: &Workflow,
    method: &str,
    history: Option<&HistoryLibrary>,
    waac_tokens: Option<&[usize]>,
) -> RunResult {
    let dag = &wf.dag;
    let n_ops = wf.n_ops;
    let mut runtime: HashMap<String, String> = HashMap::new();
    let mut success = true;
    let mut steps_done = 0usize;
    let mut tok_per_step: Vec<usize> = Vec::new();
    let mut ret_hits = 0usize;
    let mut ret_total = 0usize;
    let mut ret_min_step = 1.0f64;
    let (mut m_exact, mut m_coarse, mut m_none) = (0usize, 0usize, 0usize);
    let mut n_fallbacks = 0usize;
    for t in 0..=n_ops {
        let budget = waac_tokens.map(|w| w[t]);
        let (policies, match_info) =
            policies_for_step(method, wf, t, history, budget);
        let (rendered, caps, fb) = project_context(dag, &policies, &runtime, t);
        n_fallbacks += fb;
        tok_per_step.push(
            rendered
                .values()
                .map(|v| v.split_whitespace().count())
                .sum(),
        );
        match &match_info {
            Some((kind, _)) => match kind.as_str() {
                "exact" => m_exact += 1,
                "coarse" => m_coarse += 1,
                _ => m_none += 1,
            },
            None => {
                if method == "waac_full" && t > 0 {
                    m_none += 1;
                }
            }
        }
        // Critical-field retention vs gold at this step.
        let mut step_hits = 0usize;
        let mut step_total = 0usize;
        for (fkey, gcaps) in &wf.gold[t] {
            step_total += 1;
            let have = caps.get(fkey).copied().unwrap_or(0);
            if gcaps & !have == 0 {
                step_hits += 1;
            }
        }
        ret_hits += step_hits;
        ret_total += step_total;
        if step_total > 0 {
            ret_min_step = ret_min_step.min(step_hits as f64 / step_total as f64);
        }
        if t == n_ops {
            // Final answer: recoverable iff all gold fields at t == n_ops
            // (exactly the final-answer needs) are satisfied.
            if step_total > 0 && step_hits < step_total {
                success = false;
            }
            break;
        }
        // Execute op t against the projected context only.
        let mut ok = true;
        for spec in dag.nodes[t].inputs.iter().map(|(_, s)| s) {
            let Some(src) = &spec.source else { continue };
            let have = caps.get(src).copied().unwrap_or(0);
            if spec.need.caps() & !have != 0 {
                ok = false;
                break;
            }
        }
        if !ok {
            success = false;
            break;
        }
        steps_done += 1;
        for field in &dag.nodes[t].outputs {
            if let Some(aid) = &field.artifact_id {
                runtime.insert(aid.clone(), field.value.clone());
            }
        }
    }
    RunResult {
        success,
        steps_done,
        n_ops,
        tok_per_step,
        retention: if ret_total > 0 {
            ret_hits as f64 / ret_total as f64
        } else {
            1.0
        },
        ret_min_step,
        matches_exact: m_exact,
        matches_coarse: m_coarse,
        matches_none: m_none,
        n_fallbacks,
    }
}

/// Offline stage (Sec. 3.2.2): run dev workflows with WAAC-online only;
/// admit fragments only from trajectories that verify (succeed).
pub fn build_history(dev_workflows: &[Workflow]) -> (HistoryLibrary, usize, usize) {
    let mut lib = HistoryLibrary::new();
    let mut n_ok = 0usize;
    let mut n_bad = 0usize;
    for wf in dev_workflows {
        let res = run_workflow(wf, "waac_online", None, None);
        if !res.success {
            n_bad += 1;
            continue;
        }
        n_ok += 1;
        let dag = &wf.dag;
        for t in 1..=wf.n_ops {
            let start = t.saturating_sub(4);
            let frag = make_fragment(dag, start, t, &domain_words(wf));
            let obligations = obligations_for_step(dag, &wf.final_needs, t);
            let online = online_requirements(dag, &obligations);
            let pols: BTreeMap<(usize, String), Policy> = online
                .iter()
                .filter(|((i, _), _)| *i >= start && *i < t)
                .map(|((i, fname), pol)| ((*i - start, fname.clone()), *pol))
                .collect();
            lib.add_fragment(frag, &pols, true, &format!("dev-{}", wf.wid));
        }
    }
    (lib, n_ok, n_bad)
}

/// Per-method aggregate, mirroring the Python test's `aggregate`.
#[derive(Clone, Debug)]
pub struct Aggregate {
    pub n: usize,
    pub success_rate: f64,
    pub steps_completed_frac: f64,
    pub critical_retention: f64,
    pub min_step_retention: f64,
    pub mean_tokens_per_step: f64,
    pub n_fallbacks: usize,
}

pub fn aggregate(results: &[RunResult]) -> Aggregate {
    let n = results.len();
    Aggregate {
        n,
        // Python `sum()` over floats is Neumaier-compensated; use py_sum
        // so the aggregate ratios match bit-for-bit.
        success_rate: py_sum(results.iter().map(|r| r.success as usize as f64)) / n as f64,
        steps_completed_frac: py_sum(
            results.iter().map(|r| r.steps_done as f64 / r.n_ops as f64),
        ) / n as f64,
        critical_retention: py_sum(results.iter().map(|r| r.retention)) / n as f64,
        min_step_retention: results
            .iter()
            .map(|r| r.ret_min_step)
            .fold(f64::INFINITY, f64::min),
        mean_tokens_per_step: py_sum(
            results.iter().map(|r| r.tok_per_step.iter().sum::<usize>() as f64),
        ) / results.iter().map(|r| r.tok_per_step.len()).sum::<usize>() as f64,
        n_fallbacks: results.iter().map(|r| r.n_fallbacks).sum(),
    }
}
