//! Deep tests for the WAAC Rust port: property sweeps, exhaustive
//! sequences checked against an independent oracle, a soak beyond the
//! Python test's scale, and thread-identity.
//!
//! The test-side RNG is splitmix64 (fixed seed, no external crate).
//! Oracles here are written independently from the spec (capability-set
//! unions, direct data-flow accumulation), not by calling the code
//! under test.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use waac_rs::corpus::{build_corpus, build_dev_corpus, domain_words};
use waac_rs::executor::{build_history, run_workflow};
use waac_rs::mt19937::PyRandom;
use waac_rs::policy::{fuse, fuse_many, Need, Policy};
use waac_rs::waac::{
    canonical_key, effective_caps, fuse_policies, make_fragment,
    obligations_for_step, online_requirements, project_field, Field,
    FieldKey, HistoryLibrary, InputSpec, OpNode, TraceDAG,
};

// ------------------------------------------------------------- test RNG
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }

    fn boolean(&mut self) -> bool {
        self.next() & 1 == 0
    }
}

// ------------------------------------------------------------- oracle
/// Independent oracle: join raw capability bits, then pick the
/// least-capability policy whose caps cover the union.
fn policy_for_caps(caps: u8) -> Policy {
    // Least-capability policy whose caps COVER the union.
    Policy::ALL
        .iter()
        .copied()
        .filter(|p| caps & !p.caps() == 0)
        .min_by_key(|p| p.caps().count_ones())
        .expect("EXACT covers everything")
}

fn oracle_fuse(a: Policy, b: Policy) -> Policy {
    policy_for_caps(a.caps() | b.caps())
}

// ------------------------------------------------------------- fusion
#[test]
fn fusion_property_sweep_5000_triples() {
    let mut rng = SplitMix64::new(20261109);
    for _ in 0..5000 {
        let a = *rng.pick(&Policy::ALL);
        let b = *rng.pick(&Policy::ALL);
        let c = *rng.pick(&Policy::ALL);
        // Associativity / commutativity.
        assert_eq!(fuse(fuse(a, b), c), fuse(a, fuse(b, c)));
        assert_eq!(fuse(a, b), fuse(b, a));
        // Never relaxes either input.
        assert_eq!(a.caps() & !fuse(a, b).caps(), 0);
        assert_eq!(b.caps() & !fuse(a, b).caps(), 0);
        // Matches the independent capability-union oracle.
        assert_eq!(fuse(a, b), oracle_fuse(a, b));
        assert_eq!(fuse_many([a, b, c]), oracle_fuse(oracle_fuse(a, b), c));
    }
}

#[test]
fn fusion_exhaustive_sequences() {
    // Every policy sequence of length 0..=4 (1 + 4 + 16 + 64 + 256 = 341).
    fn check(seq: &[Policy]) {
        let got = fuse_many(seq.iter().copied());
        let caps = seq.iter().fold(0u8, |acc, p| acc | p.caps());
        assert_eq!(got, policy_for_caps(caps), "seq {seq:?}");
    }
    check(&[]);
    let mut seq = vec![Policy::Drop];
    fn rec(seq: &mut Vec<Policy>, depth: usize) {
        check(seq);
        if depth == 4 {
            return;
        }
        for p in Policy::ALL {
            seq.push(p);
            rec(seq, depth + 1);
            seq.pop();
        }
    }
    rec(&mut seq, 1);
}

// ------------------------------------------------------------- random DAGs
const OP_TYPES: [&str; 7] = [
    "search", "fetch", "extract", "compute", "verify", "write", "audit",
];
const FIELD_NAMES: [&str; 8] = [
    "page_text", "title", "key_number", "report_body", "url", "status",
    "value", "notes",
];
const FIELD_TYPES: [&str; 6] =
    ["PageText", "Title", "Number", "ReportBody", "URL", "Status"];
const WORDS: [&str; 12] = [
    "alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "lambda",
    "kappa", "mu", "nu", "xi",
];

fn random_dag(rng: &mut SplitMix64) -> (TraceDAG, Vec<(usize, String, Need)>) {
    let n_ops = rng.below(7) + 2; // 2..=8
    let mut nodes: Vec<OpNode> = Vec::new();
    let mut produced: Vec<(usize, String)> = Vec::new();
    for i in 0..n_ops {
        let mut inputs: Vec<(String, InputSpec)> = Vec::new();
        let n_in = rng.below(3);
        for k in 0..n_in {
            let src = if produced.is_empty() || rng.boolean() {
                None
            } else {
                Some(produced[rng.below(produced.len())].clone())
            };
            inputs.push((
                format!("in{k}"),
                InputSpec {
                    source: src.map(|(oi, f)| (oi, f)),
                    need: *rng.pick(&Need::ALL),
                    uncertain: rng.below(10) == 0,
                },
            ));
        }
        let n_out = rng.below(3) + 1;
        let mut outputs = Vec::new();
        for _ in 0..n_out {
            let name = rng.pick(&FIELD_NAMES).to_string();
            let long_text = rng.boolean();
            let n_words = if long_text { rng.range(40, 80) } else { rng.range(1, 4) };
            let value = (0..n_words)
                .map(|_| *rng.pick(&WORDS))
                .collect::<Vec<_>>()
                .join(" ");
            let artifact = if rng.boolean() {
                Some(format!("art_{i}_{name}"))
            } else {
                None
            };
            outputs.push(Field::new(
                &name,
                rng.pick(&FIELD_TYPES),
                value,
                long_text,
                artifact,
            ));
            produced.push((i, name));
        }
        nodes.push(OpNode::new(rng.pick(&OP_TYPES), inputs, outputs));
    }
    let mut uncertain: Vec<FieldKey> = Vec::new();
    if !produced.is_empty() && rng.boolean() {
        let p = produced[rng.below(produced.len())].clone();
        uncertain.push(p);
    }
    let mut final_needs = Vec::new();
    for _ in 0..rng.below(3) {
        if !produced.is_empty() {
            let (oi, f) = produced[rng.below(produced.len())].clone();
            final_needs.push((oi, f, *rng.pick(&Need::ALL)));
        }
    }
    (TraceDAG::new(nodes, uncertain), final_needs)
}

/// Oracle obligations, written directly from the Sec. 3.3.1 spec.
fn oracle_obligations(
    dag: &TraceDAG,
    final_needs: &[(usize, String, Need)],
    step: usize,
) -> Vec<(FieldKey, Need)> {
    let mut obs = Vec::new();
    for j in step..dag.len() {
        for spec in dag.nodes[j].inputs.iter().map(|(_, s)| s) {
            if let Some(src) = &spec.source {
                if src.0 < step {
                    obs.push((src.clone(), spec.need));
                }
            }
        }
    }
    for (oi, f, n) in final_needs {
        if *oi < step {
            obs.push(((*oi, f.clone()), *n));
        }
    }
    for (oi, f) in &dag.uncertain {
        if *oi < step {
            obs.push(((*oi, f.clone()), Need::Unresolved));
        }
    }
    obs
}

fn oracle_requirements(obs: &[(FieldKey, Need)]) -> BTreeMap<FieldKey, Policy> {
    let mut caps: BTreeMap<FieldKey, u8> = BTreeMap::new();
    for (src, need) in obs {
        *caps.entry(src.clone()).or_insert(0) |= need.caps();
    }
    caps.into_iter()
        .map(|(k, c)| (k, policy_for_caps(c)))
        .collect()
}

#[test]
fn online_requirements_match_oracle_on_random_dags() {
    let mut rng = SplitMix64::new(20261109);
    for case in 0..2000 {
        let (dag, final_needs) = random_dag(&mut rng);
        let step = rng.below(dag.len() + 1);
        let obs = obligations_for_step(&dag, &final_needs, step);
        let oracle_obs = oracle_obligations(&dag, &final_needs, step);
        assert_eq!(obs, oracle_obs, "case {case} obligations");
        let got = online_requirements(&dag, &obs);
        let want = oracle_requirements(&obs);
        assert_eq!(got, want, "case {case} requirements");
        // Every requirement covers the union of its obligations' caps.
        for (src, pol) in &got {
            let mut union = 0u8;
            for (s, n) in &obs {
                if s == src {
                    union |= n.caps();
                }
            }
            assert_eq!(union & !pol.caps(), 0, "case {case} {src:?}");
        }
    }
}

#[test]
fn fusion_never_relaxes_random_requirement_sets() {
    let mut rng = SplitMix64::new(777);
    for _ in 0..2000 {
        let (dag, final_needs) = random_dag(&mut rng);
        let step = rng.below(dag.len() + 1);
        let obs = obligations_for_step(&dag, &final_needs, step);
        let online = online_requirements(&dag, &obs);
        // Random history proposal over the same fields.
        let mut proposal: BTreeMap<FieldKey, Policy> = BTreeMap::new();
        for (k, _) in &online {
            if rng.boolean() {
                proposal.insert(k.clone(), *rng.pick(&Policy::ALL));
            }
        }
        if rng.boolean() {
            proposal.insert((0, "ghost_field".to_string()), Policy::Exact);
        }
        let fused = fuse_policies(&online, Some(&proposal));
        for (k, pol) in &online {
            let f = fused.get(k).copied().unwrap_or(Policy::Drop);
            assert_eq!(pol.caps() & !f.caps(), 0, "history relaxed {k:?}");
            assert_eq!(f, oracle_fuse(*pol, proposal.get(k).copied().unwrap_or(Policy::Drop)));
        }
        // A proposal for a field with no online requirement is added, not dropped.
        if let Some(p) = proposal.get(&(0, "ghost_field".to_string())) {
            assert_eq!(fused.get(&(0, "ghost_field".to_string())), Some(p));
        }
    }
}

#[test]
fn projection_caps_follow_truth_table() {
    // Independent truth table over (policy, long_text, artifact present).
    let mut rng = SplitMix64::new(99);
    for _ in 0..2000 {
        let long_text = rng.boolean();
        let has_art = rng.boolean();
        let in_runtime = has_art && rng.boolean();
        let n_words = if long_text {
            rng.range(40, 80)
        } else {
            rng.range(1, 4)
        };
        let value = (0..n_words)
            .map(|_| *rng.pick(&WORDS))
            .collect::<Vec<_>>()
            .join(" ");
        let field = Field::new(
            "f",
            "PageText",
            value.clone(),
            long_text,
            if has_art { Some("art_x".to_string()) } else { None },
        );
        let policy = *rng.pick(&Policy::ALL);
        let mut runtime: HashMap<String, String> = HashMap::new();
        if in_runtime {
            runtime.insert("art_x".to_string(), value.clone());
        }
        let (text, fell_back) = project_field(&field, policy, &runtime);
        let caps = effective_caps(&field, policy, &runtime);
        let want_caps = match policy {
            Policy::Drop => 0,
            Policy::Exact => 0b111,
            Policy::Summary => {
                if long_text {
                    0b010
                } else {
                    0b011
                }
            }
            Policy::Pointer => {
                if in_runtime {
                    0b100
                } else {
                    0b111
                }
            }
        };
        assert_eq!(caps, want_caps, "policy {policy:?} long {long_text} art {has_art}/{in_runtime}");
        assert_eq!(fell_back, policy == Policy::Pointer && !in_runtime);
        match policy {
            Policy::Drop => assert!(text.is_none()),
            Policy::Exact => assert_eq!(text.as_deref(), Some(value.as_str())),
            Policy::Summary => {
                let t = text.unwrap();
                if long_text {
                    assert_eq!(t.split_whitespace().count(), 26, "stub is 25 tokens + marker");
                    assert!(t.ends_with("...[summary]"));
                } else {
                    assert_eq!(t, value);
                }
            }
            Policy::Pointer => {
                if in_runtime {
                    assert_eq!(text.as_deref(), Some("<ptr:art_x>"));
                } else {
                    assert_eq!(text.as_deref(), Some(value.as_str()));
                }
            }
        }
    }
}

// ------------------------------------------------------------- history
fn random_fragment(rng: &mut SplitMix64) -> waac_rs::waac::Fragment {
    let n_ops = rng.below(4) + 1;
    let ops: Vec<String> = (0..n_ops).map(|_| rng.pick(&OP_TYPES).to_string()).collect();
    let mut fields = Vec::new();
    for r in 0..n_ops {
        for _ in 0..rng.below(3) + 1 {
            fields.push((
                r,
                rng.pick(&FIELD_NAMES).to_string(),
                rng.pick(&FIELD_TYPES).to_string(),
            ));
        }
    }
    fields.sort();
    fields.dedup();
    let mut edges = Vec::new();
    for _ in 0..rng.below(3) {
        if n_ops >= 2 && !fields.is_empty() {
            let a = rng.below(fields.len());
            let b = rng.below(fields.len());
            let (ra, fa, _) = fields[a].clone();
            let (rb, fb, _) = fields[b].clone();
            if ra < rb {
                edges.push((ra, fa, rb, fb));
            }
        }
    }
    edges.sort();
    edges.dedup();
    let domain_words: Vec<String> = (0..rng.below(3) + 1)
        .map(|_| rng.pick(&WORDS).to_string())
        .collect();
    waac_rs::waac::Fragment {
        ops,
        fields,
        edges,
        domain_words,
    }
}

#[test]
fn history_match_properties_on_random_fragments() {
    let mut rng = SplitMix64::new(20261109);
    let mut lib = HistoryLibrary::new();
    let mut admitted: Vec<waac_rs::waac::Fragment> = Vec::new();
    for i in 0..40 {
        let frag = random_fragment(&mut rng);
        let mut pols: BTreeMap<(usize, String), Policy> = BTreeMap::new();
        for (r, n, _) in &frag.fields {
            pols.insert((*r, n.clone()), *rng.pick(&Policy::ALL));
        }
        if lib.add_fragment(frag.clone(), &pols, true, &format!("src-{i}")) {
            admitted.push(frag);
        }
    }
    assert!(!admitted.is_empty());
    // 2000 random queries: scores in [0, 1]; exact queries hit exact;
    // coarse only when ops + roles match; threshold respected.
    for _ in 0..2000 {
        let q = if rng.boolean() && !admitted.is_empty() {
            admitted[rng.below(admitted.len())].clone()
        } else {
            random_fragment(&mut rng)
        };
        let m = lib.match_fragment(&q, 0.5);
        if let Some((idx, score, kind)) = m {
            // Cosine range up to float rounding (see soak test note).
            assert!((-1e-9..=1.0 + 1e-9).contains(&score), "score {score}");
            let entry = &lib.entries[idx];
            if kind == "exact" {
                assert_eq!(canonical_key(&q), entry.key);
                assert_eq!(score, 1.0);
            } else {
                assert_eq!(kind, "coarse");
                assert_ne!(canonical_key(&q), entry.key);
                assert!(score >= 0.5);
                assert_eq!(entry.fragment.ops, q.ops, "coarse needs same ops");
                let mut want: Vec<(String, String)> = q
                    .fields
                    .iter()
                    .map(|(_, n, t)| (n.clone(), t.clone()))
                    .collect();
                want.sort();
                let mut got: Vec<(String, String)> = entry
                    .fragment
                    .fields
                    .iter()
                    .map(|(_, n, t)| (n.clone(), t.clone()))
                    .collect();
                got.sort();
                assert_eq!(want, got, "coarse needs same field roles");
            }
        } else {
            // No match at 0.5: either no compatible candidate, or the
            // best candidate scores below threshold.
            let m0 = lib.match_fragment(&q, 0.0);
            if let Some((_, score, _)) = m0 {
                assert!(score < 0.5, "best {score} should have matched");
            }
        }
    }
}

// ------------------------------------------------------------- soak
#[test]
fn soak_600_workflows_all_methods() {
    // 10x the Python test's corpus, fresh seeds: the mechanism's
    // invariants (WAAC == no-compression success, 100% gold retention)
    // must hold on every workflow, and every method must complete.
    let workflows = build_corpus(600, 424242);
    let dev = build_dev_corpus(20, 424243);
    let (lib, n_ok, n_bad) = build_history(&dev);
    assert_eq!(n_ok, 20);
    assert_eq!(n_bad, 0);
    let mut lookups = 0usize;
    let mut waac_ok = 0usize;
    let mut nocomp_ok = 0usize;
    let mut waac_tokens = 0usize;
    let mut nocomp_tokens = 0usize;
    let mut waac_per_wf: Vec<Vec<usize>> = Vec::new();
    for wf in &workflows {
        let r = run_workflow(wf, "waac_full", Some(&lib), None);
        let n = run_workflow(wf, "no_compression", Some(&lib), None);
        assert!((r.ret_min_step - 1.0).abs() < 1e-12, "wid {}", wf.wid);
        assert!((r.retention - 1.0).abs() < 1e-12, "wid {}", wf.wid);
        waac_ok += r.success as usize;
        nocomp_ok += n.success as usize;
        waac_tokens += r.tok_per_step.iter().sum::<usize>();
        nocomp_tokens += n.tok_per_step.iter().sum::<usize>();
        lookups += r.matches_exact + r.matches_coarse + r.matches_none;
        waac_per_wf.push(r.tok_per_step);
    }
    assert_eq!(waac_ok, 600);
    assert_eq!(nocomp_ok, 600);
    assert!(waac_tokens < nocomp_tokens / 2, "WAAC must stay far cheaper");
    assert!(lookups >= 5000, "lookups {lookups}");
    println!(
        "soak 600 workflows: waac_full {waac_ok}/600, no_compression {nocomp_ok}/600, \
         tokens waac {waac_tokens} vs no-compression {nocomp_tokens}, history lookups {lookups}"
    );
    // The other methods also run to completion without panicking, under
    // the same matched budgets the Python test uses.
    for method in ["waac_online", "keep_last_n", "keep_last_5", "text_importance", "summary_all"] {
        let mut done = 0usize;
        for (wf, toks) in workflows.iter().zip(waac_per_wf.iter()) {
            let budget = if method == "keep_last_n" || method == "text_importance" {
                Some(toks.as_slice())
            } else {
                None
            };
            let r = run_workflow(wf, method, Some(&lib), budget);
            done += r.steps_done;
        }
        assert!(done > 0, "{method} completed no steps");
    }
}

#[test]
fn soak_12000_history_lookups() {
    // History lookups at a scale the Python test never reaches.
    let dev = build_dev_corpus(20, 777);
    let (lib, _, _) = build_history(&dev);
    assert_eq!(lib.len(), 25);
    let mut rng = SplitMix64::new(0x5EED);
    let mut exact = 0usize;
    let mut coarse = 0usize;
    let mut none = 0usize;
    for i in 0..12000 {
        // Mix query shapes: verbatim library fragments (exact hits),
        // rewired ones (coarse), and fully random ones (mostly none).
        let frag = match i % 3 {
            0 => lib.entries[rng.below(lib.len())].fragment.clone(),
            1 => {
                let mut f = lib.entries[rng.below(lib.len())].fragment.clone();
                if !f.edges.is_empty() {
                    f.edges = f.edges[1..].to_vec();
                }
                f
            }
            _ => random_fragment(&mut rng),
        };
        match lib.match_fragment(&frag, 0.5) {
            Some((_, s, "exact")) => {
                assert_eq!(s, 1.0);
                exact += 1;
            }
            Some((_, s, "coarse")) => {
                // Cosine of two identical normalized vectors is 1.0 up
                // to float rounding (Python's own _cosine can return
                // 1.0000000000000002 as well); allow an epsilon.
                assert!((0.5 - 1e-9..=1.0 + 1e-9).contains(&s), "score {s}");
                coarse += 1;
            }
            Some((_, _, _)) => unreachable!(),
            None => none += 1,
        }
    }
    assert_eq!(exact + coarse + none, 12000);
    assert!(exact > 0, "random fragments should sometimes exact-hit");
    println!("soak 12000 history lookups: exact {exact}, coarse {coarse}, none {none}");
}

// ------------------------------------------------------------- threads
#[test]
fn threads_return_identical_results() {
    let workflows = Arc::new(build_corpus(60, 20261109));
    let dev = build_dev_corpus(20, 777);
    let lib = Arc::new(build_history(&dev).0);
    // Sequential baseline.
    let seq: Vec<(bool, f64, usize)> = workflows
        .iter()
        .map(|wf| {
            let r = run_workflow(wf, "waac_full", Some(&lib), None);
            (r.success, r.ret_min_step, r.tok_per_step.iter().sum())
        })
        .collect();
    // 8 threads over the same workflows + a fusion sweep each.
    let par: Vec<(Vec<(usize, bool, f64, usize)>, Policy)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..8 {
            let wfs = Arc::clone(&workflows);
            let lb = Arc::clone(&lib);
            handles.push(s.spawn(move || {
                let mut out = Vec::new();
                for wf in wfs.iter().skip(t).step_by(8) {
                    let r = run_workflow(wf, "waac_full", Some(&lb), None);
                    out.push((wf.wid, r.success, r.ret_min_step, r.tok_per_step.iter().sum::<usize>()));
                }
                // Fusion sweep: stateless, must agree across threads.
                let mut rng = SplitMix64::new(1000 + t as u64);
                let mut acc = Policy::Drop;
                for _ in 0..10000 {
                    acc = fuse(acc, *rng.pick(&Policy::ALL));
                }
                (out, acc)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // Reassemble in wid order and compare with sequential.
    let mut by_wid: BTreeMap<usize, (bool, f64, usize)> = BTreeMap::new();
    let mut accs = Vec::new();
    for (out, acc) in par {
        for (wid, s, m, t) in out {
            by_wid.insert(wid, (s, m, t));
        }
        accs.push(acc);
    }
    for (i, (s, m, t)) in seq.iter().enumerate() {
        assert_eq!(by_wid.get(&workflows[i].wid), Some(&(*s, *m, *t)));
    }
    // The fusion sweep is deterministic given the same seed stream per
    // thread only if the streams agree; here each thread used a different
    // seed, so instead assert the sweep result is a valid policy and
    // re-run one thread's stream sequentially for equality.
    let mut rng = SplitMix64::new(1000);
    let mut acc = Policy::Drop;
    for _ in 0..10000 {
        acc = fuse(acc, *rng.pick(&Policy::ALL));
    }
    assert_eq!(accs[0], acc);
    // History matching is pure: same fragment, same answer, every time.
    let frag = make_fragment(&workflows[0].dag, 0, 4, &domain_words(&workflows[0]));
    let first = lib.match_fragment(&frag, 0.5);
    for _ in 0..100 {
        assert_eq!(lib.match_fragment(&frag, 0.5).map(|(i, s, k)| (i, s, k)), first.map(|(i, s, k)| (i, s, k)));
    }
}

// ------------------------------------------------------------- corpus determinism
#[test]
fn corpus_rng_stream_is_stable_across_builds() {
    // Two independent PyRandom streams with the corpus seeds must agree
    // on every draw (guards against accidental state sharing).
    let mut a = PyRandom::new(20261109);
    let mut b = PyRandom::new(20261109);
    for _ in 0..1000 {
        assert_eq!(a.random(), b.random());
        assert_eq!(a.getrandbits(7), b.getrandbits(7));
    }
}
