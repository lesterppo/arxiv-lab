//! Unit tests for the WAAC Rust port — one test per mechanism.
//!
//! Pinned values come from the Python reference (`waac.py` /
//! `corpus.py` / `test_waac_cpu.py` in the session directory), run
//! 2026-11-09: MT19937 streams, SHA-256 vectors, workflow 0's shape,
//! and the fragment key for its first 4 ops.

use std::collections::{BTreeMap, HashMap};
use waac_rs::corpus::{build_corpus, build_dev_corpus, build_workflow, domain_words};
use waac_rs::executor::{build_history, run_workflow};
use waac_rs::mt19937::PyRandom;
use waac_rs::policy::{fuse, fuse_many, policy_satisfies, Need, Policy};
use waac_rs::py_sum;
use waac_rs::sha256::{sha256, sha256_hex};
use waac_rs::waac::{
    canonical_key, effective_caps, fuse_policies, make_fragment,
    obligations_for_step, online_requirements, project_context,
    project_field, Field, HistoryLibrary, InputSpec, OpNode, TraceDAG,
};

// ------------------------------------------------------------- MT19937
#[test]
fn mt19937_matches_python() {
    // Reference streams from CPython 3.12 `random.Random(seed)`.
    let cases: [(u64, [f64; 3]); 3] = [
        (
            20261109,
            [0.18367155676787084, 0.489810059146196, 0.6190994094165755],
        ),
        (
            777,
            [0.22933408950153078, 0.44559617334521107, 0.36859824937216046],
        ),
        (
            99,
            [0.40397807494366633, 0.20007544457494542, 0.17880232058661227],
        ),
    ];
    for (seed, expected) in cases {
        let mut r = PyRandom::new(seed);
        for e in expected {
            assert_eq!(r.random(), e, "seed {seed}");
        }
    }
    // choice / randint / shuffle / getrandbits, seed 20261109.
    let mut r = PyRandom::new(20261109);
    let _ = (0..3).map(|_| r.random()).count();
    let letters = ["a", "b", "c", "d", "e"];
    let picks: Vec<&str> = (0..5).map(|_| *r.choice(&letters)).collect();
    assert_eq!(picks, vec!["d", "c", "b", "a", "d"]);
    let ints: Vec<i64> = (0..5).map(|_| r.randint(9, 12)).collect();
    assert_eq!(ints, vec![9, 10, 12, 10, 11]);
    let mut x: Vec<i32> = (0..8).collect();
    r.shuffle(&mut x);
    assert_eq!(x, vec![0, 4, 3, 1, 6, 5, 7, 2]);
    let bits: Vec<u64> = (0..5).map(|_| r.getrandbits(4)).collect();
    assert_eq!(bits, vec![6, 3, 9, 9, 6]);
    // Same seed, same stream (determinism / clone).
    let mut a = PyRandom::new(4242);
    let mut b = a.clone();
    for _ in 0..100 {
        assert_eq!(a.random(), b.random());
    }
}

// ------------------------------------------------------------- SHA-256
#[test]
fn sha256_matches_python_hashlib() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // 1 MiB of 'a' (multi-block + padding edge).
    let big = vec![b'a'; 1_000_000];
    assert_eq!(
        sha256_hex(&big),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
    assert_eq!(sha256(b"abc").len(), 32);
}

// ------------------------------------------------------------- policies
#[test]
fn fuse_is_the_capability_union_lattice() {
    // Oracle: union of capability sets, mapped back to a policy.
    fn oracle(a: Policy, b: Policy) -> Policy {
        let caps = a.caps() | b.caps();
        Policy::ALL
            .iter()
            .copied()
            .filter(|p| caps & !p.caps() == 0)
            .min_by_key(|p| p.caps().count_ones())
            .unwrap()
    }
    for a in Policy::ALL {
        for b in Policy::ALL {
            assert_eq!(fuse(a, b), oracle(a, b), "fuse({a:?}, {b:?})");
            assert_eq!(fuse(a, b), fuse(b, a), "commutativity");
            assert_eq!(fuse(a, Policy::Drop), a, "DROP identity");
            assert_eq!(fuse(Policy::Drop, a), a, "DROP identity");
            // Never relaxes either input.
            assert_eq!(a.caps() & !fuse(a, b).caps(), 0);
            assert_eq!(b.caps() & !fuse(a, b).caps(), 0);
        }
    }
    assert_eq!(fuse(Policy::Summary, Policy::Pointer), Policy::Exact);
    assert_eq!(fuse_many([Policy::Summary, Policy::Pointer]), Policy::Exact);
    assert_eq!(
        fuse_many([Policy::Drop, Policy::Summary, Policy::Drop]),
        Policy::Summary
    );
    assert_eq!(fuse_many([]), Policy::Drop);
    // Associativity over all triples.
    for a in Policy::ALL {
        for b in Policy::ALL {
            for c in Policy::ALL {
                assert_eq!(fuse(fuse(a, b), c), fuse(a, fuse(b, c)));
            }
        }
    }
}

#[test]
fn policy_satisfies_table() {
    assert!(policy_satisfies(Policy::Exact, Need::Literal));
    assert!(policy_satisfies(Policy::Exact, Need::Semantic));
    assert!(policy_satisfies(Policy::Exact, Need::Recoverable));
    assert!(policy_satisfies(Policy::Exact, Need::Unresolved));
    assert!(policy_satisfies(Policy::Summary, Need::Semantic));
    assert!(!policy_satisfies(Policy::Summary, Need::Literal));
    assert!(!policy_satisfies(Policy::Summary, Need::Recoverable));
    assert!(policy_satisfies(Policy::Pointer, Need::Recoverable));
    assert!(!policy_satisfies(Policy::Pointer, Need::Semantic));
    assert!(!policy_satisfies(Policy::Drop, Need::Semantic));
    assert_eq!(Need::Literal.min_policy(), Policy::Exact);
    assert_eq!(Need::Semantic.min_policy(), Policy::Summary);
    assert_eq!(Need::Recoverable.min_policy(), Policy::Pointer);
    assert_eq!(Need::Unresolved.min_policy(), Policy::Exact);
}

// ------------------------------------------------------------- py_sum
#[test]
fn py_sum_is_neumaier() {
    // Naive summation gives 0.0 here; Neumaier (CPython sum) gives 2.0.
    assert_eq!(py_sum([1e16, 1.0, 1.0, -1e16]), 2.0);
    assert_eq!(py_sum([0.1; 10]), 1.0);
    assert_eq!(py_sum([]), 0.0);
}

// ------------------------------------------------------------- DAG
fn tiny_dag() -> TraceDAG {
    // op0 search -> page_text; op1 extract needs it semantically;
    // op2 verify needs it literally; op3 write needs title literally.
    let nodes = vec![
        OpNode::new(
            "search",
            vec![("query".to_string(), InputSpec::new(None, Need::Literal))],
            vec![Field::new("page_text", "PageText", "long page text here".into(), true, Some("art_page".into()))],
        ),
        OpNode::new(
            "extract",
            vec![("page_text".to_string(), InputSpec::new(Some((0, "page_text")), Need::Semantic))],
            vec![Field::new("title", "Title", "My Title".into(), false, None)],
        ),
        OpNode::new(
            "verify",
            vec![("page_text".to_string(), InputSpec::new(Some((0, "page_text")), Need::Literal))],
            vec![Field::new("check_result", "Status", "ok".into(), false, None)],
        ),
        OpNode::new(
            "write",
            vec![("title".to_string(), InputSpec::new(Some((1, "title")), Need::Literal))],
            vec![Field::new("artifact_path", "ArtifactPath", "/tmp/out.md".into(), false, None)],
        ),
    ];
    TraceDAG::new(nodes, vec![])
}

#[test]
fn dag_edges_from_argument_bindings() {
    let dag = tiny_dag();
    assert_eq!(dag.edges.len(), 3);
    assert_eq!(dag.edges[0], (0, "page_text".to_string(), 1, "page_text".to_string(), false));
    assert_eq!(dag.edges[1], (0, "page_text".to_string(), 2, "page_text".to_string(), false));
    assert_eq!(dag.edges[2], (1, "title".to_string(), 3, "title".to_string(), false));
    assert_eq!(dag.field(0, "page_text").ftype, "PageText");
}

#[test]
fn online_requirements_join_across_obligations() {
    let dag = tiny_dag();
    // At step 1: extract (semantic) + verify (literal) both outstanding
    // for page_text -> EXACT (join). Title is not yet produced.
    let final_needs = vec![(1, "title".to_string(), Need::Literal)];
    let obs = obligations_for_step(&dag, &final_needs, 1);
    let req = online_requirements(&dag, &obs);
    assert_eq!(req.get(&(0, "page_text".to_string())), Some(&Policy::Exact));
    assert!(!req.contains_key(&(1, "title".to_string())));
    // At step 2: extract done; verify (literal) + write's title need.
    let obs = obligations_for_step(&dag, &final_needs, 2);
    let req = online_requirements(&dag, &obs);
    assert_eq!(req.get(&(0, "page_text".to_string())), Some(&Policy::Exact));
    assert_eq!(req.get(&(1, "title".to_string())), Some(&Policy::Exact));
    // At step 3: only the write's title need remains; page_text is free.
    let obs = obligations_for_step(&dag, &final_needs, 3);
    let req = online_requirements(&dag, &obs);
    assert!(!req.contains_key(&(0, "page_text".to_string())));
    assert_eq!(req.get(&(1, "title".to_string())), Some(&Policy::Exact));
    // Semantic-only obligation -> SUMMARY.
    let obs = vec![((0, "page_text".to_string()), Need::Semantic)];
    let req = online_requirements(&dag, &obs);
    assert_eq!(req.get(&(0, "page_text".to_string())), Some(&Policy::Summary));
}

#[test]
fn uncertain_evidence_forces_exact() {
    let mut dag = tiny_dag();
    dag.uncertain = vec![(0, "page_text".to_string())];
    let obs = obligations_for_step(&dag, &[], 3);
    let req = online_requirements(&dag, &obs);
    assert_eq!(
        req.get(&(0, "page_text".to_string())),
        Some(&Policy::Exact),
        "unresolved evidence is retained EXACT even with no consumer"
    );
}

// ------------------------------------------------------------- fragments
#[test]
fn fragment_key_is_stable_and_wiring_sensitive() {
    let workflows = build_corpus(60, 20261109);
    let wf = &workflows[0];
    // Pinned from the Python reference.
    assert_eq!(wf.domain, "travel");
    assert_eq!(wf.topic, "kyoto in spring");
    assert_eq!(wf.skeleton, "two_source");
    assert_eq!(wf.n_ops, 9);
    assert_eq!(wf.verify_need, Need::Literal);
    assert_eq!(wf.write_need, Need::Recoverable);
    let frag = make_fragment(&wf.dag, 0, 4, &domain_words(wf));
    assert_eq!(canonical_key(&frag), "aacdc5c7643ccb1f");
    assert_eq!(frag.edges.len(), 2);
    // Same fragment -> same key; rewired -> different key.
    let frag2 = make_fragment(&wf.dag, 0, 4, &domain_words(wf));
    assert_eq!(canonical_key(&frag2), canonical_key(&frag));
    let mut rewired = frag.clone();
    rewired.edges = rewired.edges[1..].to_vec();
    assert_ne!(canonical_key(&rewired), canonical_key(&frag));
    assert_eq!(canonical_key(&frag).len(), 16);
    assert!(canonical_key(&frag).chars().all(|c| c.is_ascii_hexdigit()));
}

// ------------------------------------------------------------- history
#[test]
fn history_admission_rules() {
    let mut lib = HistoryLibrary::new();
    let workflows = build_corpus(60, 20261109);
    let wf = &workflows[0];
    let frag = make_fragment(&wf.dag, 0, 4, &domain_words(wf));
    let mut pols: BTreeMap<(usize, String), Policy> = BTreeMap::new();
    pols.insert((0, "result_url".to_string()), Policy::Exact);
    // Unverified admission is rejected without mutation.
    assert!(!lib.add_fragment(frag.clone(), &pols, false, "bad"));
    assert_eq!(lib.len(), 0);
    // Verified admission works; a second fragment with the same key
    // fuses policies conservatively (SUMMARY + EXACT -> EXACT).
    assert!(lib.add_fragment(frag.clone(), &pols, true, "dev-0"));
    assert_eq!(lib.len(), 1);
    let mut pols2: BTreeMap<(usize, String), Policy> = BTreeMap::new();
    pols2.insert((0, "result_url".to_string()), Policy::Summary);
    pols2.insert((0, "results_page".to_string()), Policy::Pointer);
    assert!(lib.add_fragment(frag.clone(), &pols2, true, "dev-1"));
    assert_eq!(lib.len(), 1, "same key aggregates, no new entry");
    let entry = &lib.entries[0];
    assert_eq!(
        entry.policies.get(&(0, "result_url".to_string())),
        Some(&Policy::Exact)
    );
    assert_eq!(
        entry.policies.get(&(0, "results_page".to_string())),
        Some(&Policy::Pointer)
    );
    assert_eq!(entry.sources, vec!["dev-0".to_string(), "dev-1".to_string()]);
    // Exact-key match wins over coarse.
    let m = lib.match_fragment(&frag, 0.5).unwrap();
    assert_eq!(m.2, "exact");
    assert_eq!(m.1, 1.0);
}

#[test]
fn history_coarse_match_and_no_match() {
    let dev = build_dev_corpus(20, 777);
    let (lib, n_ok, n_bad) = build_history(&dev);
    assert_eq!(n_ok, 20);
    assert_eq!(n_bad, 0);
    assert_eq!(lib.len(), 25);
    // Same structure, different wiring -> coarse at cosine 1.0.
    let entry = lib.entries.iter().find(|e| !e.fragment.edges.is_empty()).unwrap();
    let mut frag = entry.fragment.clone();
    frag.edges = frag.edges[1..].to_vec();
    assert_ne!(canonical_key(&frag), entry.key);
    let m = lib.match_fragment(&frag, 0.5).unwrap();
    assert_eq!(m.2, "coarse");
    assert!((m.1 - 1.0).abs() < 1e-12);
    // Different op sequence -> no match.
    let mut frag2 = entry.fragment.clone();
    frag2.ops = frag2.ops.iter().rev().cloned().collect();
    assert!(lib.match_fragment(&frag2, 0.5).is_none());
    // Empty library -> no match.
    assert!(HistoryLibrary::new()
        .match_fragment(&frag, 0.5)
        .is_none());
}

// ------------------------------------------------------------- projection
#[test]
fn projection_rules() {
    let runtime: HashMap<String, String> =
        [("art_1".to_string(), "FULL VALUE".to_string())]
            .into_iter()
            .collect();
    let empty: HashMap<String, String> = HashMap::new();
    // Long text SUMMARY: first 25 tokens + stub marker.
    let long_val = (0..60).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
    let long_field = Field::new("page_text", "PageText", long_val.clone(), true, Some("art_1".into()));
    let (text, fb) = project_field(&long_field, Policy::Summary, &runtime);
    let expected = (0..25)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ")
        + " ...[summary]";
    assert_eq!(text.as_deref(), Some(expected.as_str()));
    assert!(!fb);
    assert_eq!(text.unwrap().split_whitespace().count(), 26);
    // Short scalar SUMMARY: retained whole.
    let short = Field::new("title", "Title", "My Title".into(), false, None);
    assert_eq!(
        project_field(&short, Policy::Summary, &runtime).0.as_deref(),
        Some("My Title")
    );
    // EXACT keeps the value; DROP omits.
    assert_eq!(
        project_field(&long_field, Policy::Exact, &runtime).0.as_deref(),
        Some(long_val.as_str())
    );
    assert_eq!(project_field(&long_field, Policy::Drop, &runtime).0, None);
    // Valid pointer renders the reference; invalid falls back to EXACT.
    assert_eq!(
        project_field(&long_field, Policy::Pointer, &runtime),
        (Some("<ptr:art_1>".to_string()), false)
    );
    assert_eq!(
        project_field(&long_field, Policy::Pointer, &empty),
        (Some(long_val.clone()), true)
    );
    // Pointer with no artifact id at all also falls back.
    let no_art = Field::new("title", "Title", "T".into(), false, None);
    assert_eq!(
        project_field(&no_art, Policy::Pointer, &runtime).1,
        true
    );
}

#[test]
fn effective_caps_mirror_projection() {
    let runtime: HashMap<String, String> =
        [("art_1".to_string(), "FULL".to_string())].into_iter().collect();
    let empty: HashMap<String, String> = HashMap::new();
    let long_field = Field::new("page_text", "PageText", "a b c".into(), true, Some("art_1".into()));
    let short = Field::new("title", "Title", "T".into(), false, None);
    // SUMMARY of a short scalar also provides `literal`.
    assert_eq!(
        effective_caps(&short, Policy::Summary, &runtime),
        Need::Semantic.caps() | Need::Literal.caps()
    );
    // SUMMARY of long text provides only `semantic`.
    assert_eq!(
        effective_caps(&long_field, Policy::Summary, &runtime),
        Need::Semantic.caps()
    );
    // Valid pointer provides only `recoverable`; invalid -> EXACT caps.
    assert_eq!(
        effective_caps(&long_field, Policy::Pointer, &runtime),
        Need::Recoverable.caps()
    );
    assert_eq!(
        effective_caps(&long_field, Policy::Pointer, &empty),
        Policy::Exact.caps()
    );
    assert_eq!(effective_caps(&short, Policy::Drop, &runtime), 0);
}

#[test]
fn project_context_covers_visible_fields_only() {
    let dag = tiny_dag();
    let runtime: HashMap<String, String> = HashMap::new();
    let mut pols: BTreeMap<(usize, String), Policy> = BTreeMap::new();
    pols.insert((0, "page_text".to_string()), Policy::Exact);
    let (rendered, caps, fb) = project_context(&dag, &pols, &runtime, 1);
    assert_eq!(fb, 0);
    // Only op 0's fields are visible at step 1; page_text is EXACT,
    // and every other produced field defaults to DROP (caps 0, no text).
    assert_eq!(rendered.len(), 1);
    assert_eq!(rendered.get(&(0, "page_text".to_string())).unwrap(), "long page text here");
    assert_eq!(caps.get(&(0, "page_text".to_string())), Some(&0b111));
    // Step 0: nothing visible.
    let (rendered, _, _) = project_context(&dag, &pols, &runtime, 0);
    assert!(rendered.is_empty());
}

// ------------------------------------------------------------- fusion of policies
#[test]
fn fuse_policies_never_relaxes_online() {
    let mut online: BTreeMap<(usize, String), Policy> = BTreeMap::new();
    online.insert((0, "a".to_string()), Policy::Summary);
    online.insert((0, "b".to_string()), Policy::Exact);
    let mut proposal: BTreeMap<(usize, String), Policy> = BTreeMap::new();
    proposal.insert((0, "a".to_string()), Policy::Pointer);
    proposal.insert((0, "c".to_string()), Policy::Summary);
    let fused = fuse_policies(&online, Some(&proposal));
    // SUMMARY + POINTER -> EXACT; EXACT stays; new field added.
    assert_eq!(fused.get(&(0, "a".to_string())), Some(&Policy::Exact));
    assert_eq!(fused.get(&(0, "b".to_string())), Some(&Policy::Exact));
    assert_eq!(fused.get(&(0, "c".to_string())), Some(&Policy::Summary));
    // No proposal -> online unchanged.
    assert_eq!(fuse_policies(&online, None), online);
}

// ------------------------------------------------------------- corpus
#[test]
fn corpus_is_deterministic_and_shaped() {
    let a = build_corpus(60, 20261109);
    let b = build_corpus(60, 20261109);
    assert_eq!(a.len(), 60);
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.topic, y.topic);
        assert_eq!(x.n_ops, y.n_ops);
        assert_eq!(x.dag.len(), y.dag.len());
    }
    for wf in &a {
        assert!((7..=13).contains(&wf.n_ops), "wid {} n_ops {}", wf.wid, wf.n_ops);
        assert!(!wf.gold.is_empty());
        assert_eq!(wf.gold.len(), wf.n_ops + 1);
        // Gold at step 0 covers every consumer need; final step is
        // exactly the final-answer needs.
        assert_eq!(wf.gold[wf.n_ops].len(), wf.final_needs.len());
    }
    // Different seed -> different corpus.
    let c = build_corpus(60, 777);
    assert_ne!(a[0].topic, c[0].topic);
    // Dev corpus: separate seed, wids 1000+.
    let dev = build_dev_corpus(20, 777);
    assert_eq!(dev[0].wid, 1000);
    assert_eq!(dev[0].domain, "travel");
    assert_eq!(dev[0].topic, "bergen fjords");
    assert_eq!(dev[0].skeleton, "two_audit");
    assert_eq!(dev[0].n_ops, 10);
    // build_workflow with a fresh RNG reproduces workflow 0.
    let mut rng = PyRandom::new(20261109);
    let wf0 = build_workflow(&mut rng, 0);
    assert_eq!(wf0.topic, "kyoto in spring");
    assert_eq!(wf0.n_ops, 9);
}

// ------------------------------------------------------------- executor
#[test]
fn executor_waac_matches_no_compression_on_corpus() {
    let workflows = build_corpus(60, 20261109);
    let dev = build_dev_corpus(20, 777);
    let (lib, _, _) = build_history(&dev);
    let mut waac_tokens = 0usize;
    for wf in &workflows {
        let r = run_workflow(wf, "waac_full", Some(&lib), None);
        assert!(r.success, "waac_full must succeed on wid {}", wf.wid);
        assert!((r.ret_min_step - 1.0).abs() < 1e-12);
        assert!((r.retention - 1.0).abs() < 1e-12);
        waac_tokens += r.tok_per_step.iter().sum::<usize>();
        let n = run_workflow(wf, "no_compression", Some(&lib), None);
        assert!(n.success);
        assert!(r.tok_per_step.iter().sum::<usize>() < n.tok_per_step.iter().sum::<usize>());
    }
    // Pinned totals from the Python reference (exact integers).
    assert_eq!(waac_tokens, 72696);
    let mut nocomp_tokens = 0usize;
    for wf in &workflows {
        let n = run_workflow(wf, "no_compression", Some(&lib), None);
        nocomp_tokens += n.tok_per_step.iter().sum::<usize>();
    }
    assert_eq!(nocomp_tokens, 348074);
    // 607 total steps over the 60 workflows (sum of n_ops + 1).
    assert_eq!(workflows.iter().map(|w| w.n_ops + 1).sum::<usize>(), 607);
}
