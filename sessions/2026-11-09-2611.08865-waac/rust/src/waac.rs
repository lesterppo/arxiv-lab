//! WAAC mechanism (arXiv 2611.08865, Sec. 3.2-3.3 / App. A.1-A.3).
//!
//! Rust port of `waac.py` from the session: capability-set policies with
//! least-upper-bound fusion, typed Trace-to-DAG, online backward
//! dependency analysis, verified-only history library with TF-IDF cosine
//! coarse matching, conservative fusion, and context projection.
//! Std-only, deterministic, no network.

use crate::policy::{fuse, Need, Policy};
use crate::py_sum;
use crate::sha256::sha256_hex;
use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------- fields
/// One typed output field of an operation.
///
/// `long_text`: the value is long prose whose SUMMARY stub truncates it;
/// short scalar fields are retained whole by a SUMMARY stub.
/// `artifact_id`: key under which the full value is stored in the
/// runtime state S_t (None if not stored as an artifact).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub ftype: String,
    pub value: String,
    pub long_text: bool,
    pub artifact_id: Option<String>,
}

impl Field {
    pub fn new(
        name: &str,
        ftype: &str,
        value: String,
        long_text: bool,
        artifact_id: Option<String>,
    ) -> Self {
        Field {
            name: name.to_string(),
            ftype: ftype.to_string(),
            value,
            long_text,
            artifact_id,
        }
    }

    /// Whitespace-token count (Python `len(value.split())`).
    pub fn token_count(&self) -> usize {
        self.value.split_whitespace().count()
    }
}

/// One input binding of an operation node.
///
/// `source`: (op_idx, field_name) of the producer, or None for a task
/// literal. `need`: the fidelity the consumer demands. `uncertain`:
/// dependency evidence that could not be reliably resolved.
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub source: Option<(usize, String)>,
    pub need: Need,
    pub uncertain: bool,
}

impl InputSpec {
    pub fn new(source: Option<(usize, &str)>, need: Need) -> Self {
        InputSpec {
            source: source.map(|(i, f)| (i, f.to_string())),
            need,
            uncertain: false,
        }
    }

    pub fn uncertain(source: Option<(usize, &str)>, need: Need) -> Self {
        InputSpec {
            source: source.map(|(i, f)| (i, f.to_string())),
            need,
            uncertain: true,
        }
    }
}

/// One operation node v_i = (op_i, I_i, O_i).
///
/// Inputs keep insertion order (Python dict order); outputs keep
/// insertion order too — the corpus generator's field order matters for
/// the recency baselines and for fragment construction (which sorts by
/// name, as Python does).
#[derive(Clone, Debug)]
pub struct OpNode {
    pub op_type: String,
    pub inputs: Vec<(String, InputSpec)>,
    pub outputs: Vec<Field>,
}

impl OpNode {
    pub fn new(op_type: &str, inputs: Vec<(String, InputSpec)>, outputs: Vec<Field>) -> Self {
        OpNode {
            op_type: op_type.to_string(),
            inputs,
            outputs,
        }
    }

    pub fn input(&self, name: &str) -> Option<&InputSpec> {
        self.inputs.iter().find(|(n, _)| n == name).map(|(_, s)| s)
    }

    pub fn output(&self, name: &str) -> Option<&Field> {
        self.outputs.iter().find(|f| f.name == name)
    }
}

/// Field key: (producing op index, field name).
pub type FieldKey = (usize, String);

/// Typed Trace-to-DAG (Sec. 3.2).
///
/// Edges are built from explicit argument bindings: input I_j[q] naming
/// output O_i[p] as its source. `uncertain` lists (op_idx, field_name)
/// pairs whose dependency evidence could not be reliably resolved; the
/// online analysis retains those fields EXACT conservatively.
#[derive(Clone, Debug)]
pub struct TraceDAG {
    pub nodes: Vec<OpNode>,
    pub uncertain: Vec<FieldKey>,
    /// (src_op, src_field, dst_op, dst_field, uncertain)
    pub edges: Vec<(usize, String, usize, String, bool)>,
}

impl TraceDAG {
    pub fn new(nodes: Vec<OpNode>, uncertain: Vec<FieldKey>) -> Self {
        let mut edges = Vec::new();
        for (j, node) in nodes.iter().enumerate() {
            for (iname, spec) in &node.inputs {
                if let Some((si, sf)) = &spec.source {
                    edges.push((*si, sf.clone(), j, iname.clone(), spec.uncertain));
                }
            }
        }
        TraceDAG {
            nodes,
            uncertain,
            edges,
        }
    }

    pub fn field(&self, op_idx: usize, name: &str) -> &Field {
        self.nodes[op_idx]
            .output(name)
            .unwrap_or_else(|| panic!("no field {name} on op {op_idx}"))
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ------------------------------------------------------- online analysis
/// Outstanding obligations O_t at `step` (Sec. 3.3.1).
///
/// Pending tool inputs of ops >= step, final-answer requirements, plus
/// one unresolved-evidence obligation per uncertain field. Returns a
/// list of (source_key, need); source_key names a field produced by an
/// op < step (task literals carry no history field).
pub fn obligations_for_step(
    dag: &TraceDAG,
    final_needs: &[(usize, String, Need)],
    step: usize,
) -> Vec<(FieldKey, Need)> {
    let mut obs = Vec::new();
    for j in step..dag.len() {
        for spec in dag.nodes[j].inputs.iter().map(|(_, s)| s) {
            let Some(src) = &spec.source else { continue };
            if src.0 >= step {
                continue; // producer has not run yet
            }
            obs.push((src.clone(), spec.need));
        }
    }
    for (op_idx, fname, need) in final_needs {
        if *op_idx < step {
            obs.push(((*op_idx, fname.clone()), *need));
        }
    }
    for (op_idx, fname) in &dag.uncertain {
        if *op_idx < step {
            obs.push(((*op_idx, fname.clone()), Need::Unresolved));
        }
    }
    obs
}

/// Backward dependency analysis (Sec. 3.3.1).
///
/// Reverse-traverses the DAG from the outstanding obligations: each
/// obligation's consumer input resolves (via its data edge) to the
/// producer field, which inherits the minimum fidelity the obligation
/// demands. Requirements from several obligations join (fuse), so
/// satisfying one never invalidates another. Fields with no obligation
/// are absent (eligible for DROP).
pub fn online_requirements(
    _dag: &TraceDAG,
    obligations: &[(FieldKey, Need)],
) -> BTreeMap<FieldKey, Policy> {
    let mut req: BTreeMap<FieldKey, Policy> = BTreeMap::new();
    for (src, need) in obligations {
        let pol = need.min_policy();
        let cur = req.get(src).copied().unwrap_or(Policy::Drop);
        req.insert(src.clone(), fuse(cur, pol));
    }
    req
}

// ------------------------------------------------------------ fragments
/// Canonical local fragment: ops [start, end) with their fields and the
/// data edges internal to the window (Sec. 3.2.2 / A.2). Field roles are
/// (relative op index, field name, field type); instance values are
/// replaced by type placeholders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fragment {
    pub ops: Vec<String>,
    pub fields: Vec<(usize, String, String)>,
    pub edges: Vec<(usize, String, usize, String)>,
    pub domain_words: Vec<String>,
}

pub fn make_fragment(
    dag: &TraceDAG,
    start: usize,
    end: usize,
    domain_words: &[String],
) -> Fragment {
    let ops: Vec<String> = (start..end)
        .map(|i| dag.nodes[i].op_type.clone())
        .collect();
    let mut fields = Vec::new();
    for i in start..end {
        let mut outs: Vec<&Field> = dag.nodes[i].outputs.iter().collect();
        outs.sort_by(|a, b| a.name.cmp(&b.name));
        for f in outs {
            fields.push((i - start, f.name.clone(), f.ftype.clone()));
        }
    }
    let mut edges = Vec::new();
    for (si, sf, di, df, _unc) in &dag.edges {
        if start <= *si && *si < end && start <= *di && *di < end {
            edges.push((*si - start, sf.clone(), *di - start, df.clone()));
        }
    }
    edges.sort();
    let mut words: Vec<String> = domain_words.to_vec();
    words.sort();
    words.dedup();
    Fragment {
        ops,
        fields,
        edges,
        domain_words: words,
    }
}

/// Deterministic hash of the canonical fragment (Sec. 3.2.2).
/// First 16 hex chars of the SHA-256, as in Python.
pub fn canonical_key(fragment: &Fragment) -> String {
    let mut parts = String::new();
    parts.push_str("ops:");
    parts.push_str(&fragment.ops.join(","));
    parts.push_str("|fields:");
    let field_strs: Vec<String> = fragment
        .fields
        .iter()
        .map(|(r, n, t)| format!("{r}:{n}:{t}"))
        .collect();
    parts.push_str(&field_strs.join(","));
    parts.push_str("|edges:");
    let edge_strs: Vec<String> = fragment
        .edges
        .iter()
        .map(|(si, sf, di, df)| format!("{si}:{sf}>{di}:{df}"))
        .collect();
    parts.push_str(&edge_strs.join(","));
    sha256_hex(parts.as_bytes())[..16].to_string()
}

/// Tokens of the fragment's semantic description (Sec. 3.3.2): op types,
/// field names/types and domain words. Embeddings in the paper; TF-IDF
/// cosine over these tokens is the CPU proxy here.
pub fn semantic_tokens(fragment: &Fragment) -> Vec<String> {
    let mut toks: Vec<String> = fragment.ops.clone();
    for (_r, name, _t) in &fragment.fields {
        toks.push(name.clone());
    }
    for (_r, _n, ftype) in &fragment.fields {
        toks.push(ftype.clone());
    }
    toks.extend(fragment.domain_words.iter().cloned());
    toks
}

/// An insertion-ordered sparse vector: (token, weight) in first-occurrence
/// order of the token list. Order matters: Python dicts preserve insertion
/// order, and the cosine sum follows it, so float rounding must match.
pub type SparseVec = Vec<(String, f64)>;

fn tfidf_vector(tokens: &[String], idf: &HashMap<String, f64>) -> SparseVec {
    let mut vec: SparseVec = Vec::new();
    for t in tokens {
        if let Some(entry) = vec.iter_mut().find(|(k, _)| k == t) {
            entry.1 += 1.0;
        } else {
            vec.push((t.clone(), 1.0));
        }
    }
    for (t, v) in vec.iter_mut() {
        *v *= idf.get(t).copied().unwrap_or(0.0);
    }
    let norm = py_sum(vec.iter().map(|(_, v)| v * v)).sqrt();
    if norm > 0.0 {
        for (_, v) in vec.iter_mut() {
            *v /= norm;
        }
    }
    vec
}

fn cosine(a: &SparseVec, b: &SparseVec) -> f64 {
    // Python iterates the smaller dict (ties: the first argument), and
    // sums with Neumaier compensation (CPython 3.12 `sum()`).
    let (small, big) = if a.len() > b.len() { (b, a) } else { (a, b) };
    let products = small.iter().map(|(t, v)| {
        let bv = big
            .iter()
            .find(|(k, _)| k == t)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        v * bv
    });
    py_sum(products)
}

/// One admitted fragment + the policies observed for its fields.
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub key: String,
    pub fragment: Fragment,
    /// (relative op idx, field name) -> policy.
    pub policies: BTreeMap<(usize, String), Policy>,
    pub sources: Vec<String>,
}

/// Verified-successful trajectory fragments + their field policies.
///
/// Only trajectories that pass task-level verification are admitted
/// (Sec. 3.2.2 / 3.3). Records sharing a canonical key aggregate
/// policies by conservative join.
#[derive(Clone, Debug, Default)]
pub struct HistoryLibrary {
    pub by_key: HashMap<String, usize>,
    pub entries: Vec<HistoryEntry>,
}

impl HistoryLibrary {
    pub fn new() -> Self {
        HistoryLibrary::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Admit one checkpoint fragment. Returns true iff admitted
    /// (unverified fragments are rejected without mutation).
    pub fn add_fragment(
        &mut self,
        fragment: Fragment,
        policies: &BTreeMap<(usize, String), Policy>,
        verified: bool,
        source_id: &str,
    ) -> bool {
        if !verified {
            return false;
        }
        let key = canonical_key(&fragment);
        match self.by_key.get(&key).copied() {
            None => {
                let idx = self.entries.len();
                self.entries.push(HistoryEntry {
                    key: key.clone(),
                    fragment,
                    policies: policies.clone(),
                    sources: vec![source_id.to_string()],
                });
                self.by_key.insert(key, idx);
            }
            Some(idx) => {
                let entry = &mut self.entries[idx];
                for (role, pol) in policies {
                    let cur = entry.policies.get(role).copied().unwrap_or(Policy::Drop);
                    entry.policies.insert(role.clone(), fuse(cur, *pol));
                }
                if !entry.sources.iter().any(|s| s == source_id) {
                    entry.sources.push(source_id.to_string());
                }
            }
        }
        true
    }

    fn idf(&self) -> HashMap<String, f64> {
        let docs: Vec<Vec<String>> =
            self.entries.iter().map(|e| semantic_tokens(&e.fragment)).collect();
        let n = docs.len() as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        for toks in &docs {
            let mut seen: Vec<&String> = Vec::new();
            for t in toks {
                if !seen.contains(&t) {
                    seen.push(t);
                    *df.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }
        df.iter()
            .map(|(t, c)| {
                let c = *c as f64;
                (t.clone(), (1.0 + (n - c + 0.5) / (c + 0.5)).ln())
            })
            .collect()
    }

    /// Historical fragment matching (Sec. 3.3.2).
    ///
    /// Exact-key index first; otherwise, among candidates compatible in
    /// typed operation structure (same op-type sequence), field roles
    /// (same field-role multiset) and action categories, rank by TF-IDF
    /// cosine over semantic descriptions and take the best above
    /// `threshold`. Returns (entry index, score, kind) or None.
    pub fn match_fragment(
        &self,
        fragment: &Fragment,
        threshold: f64,
    ) -> Option<(usize, f64, &'static str)> {
        let key = canonical_key(fragment);
        if let Some(idx) = self.by_key.get(&key).copied() {
            return Some((idx, 1.0, "exact"));
        }
        if self.entries.is_empty() {
            return None;
        }
        let mut want_roles: Vec<(String, String)> = fragment
            .fields
            .iter()
            .map(|(_r, n, t)| (n.clone(), t.clone()))
            .collect();
        want_roles.sort();
        let mut cands: Vec<usize> = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            let frag = &e.fragment;
            if frag.ops != fragment.ops {
                continue;
            }
            let mut roles: Vec<(String, String)> = frag
                .fields
                .iter()
                .map(|(_r, n, t)| (n.clone(), t.clone()))
                .collect();
            roles.sort();
            if roles != want_roles {
                continue;
            }
            cands.push(i);
        }
        if cands.is_empty() {
            return None;
        }
        let idf = self.idf();
        let qvec = tfidf_vector(&semantic_tokens(fragment), &idf);
        let mut best: Option<usize> = None;
        let mut best_score = -1.0f64;
        for i in cands {
            let evec = tfidf_vector(&semantic_tokens(&self.entries[i].fragment), &idf);
            let s = cosine(&qvec, &evec);
            if s > best_score {
                best = Some(i);
                best_score = s;
            }
        }
        if best_score >= threshold {
            best.map(|i| (i, best_score, "coarse"))
        } else {
            None
        }
    }
}

// ------------------------------------------------------- fusion & project
/// Conservative fusion (Sec. 3.3.3): pi_t(f) = max(d_t(f), rho_hat(f)).
/// History can only ADD retention, never relax the online requirement.
pub fn fuse_policies(
    online: &BTreeMap<FieldKey, Policy>,
    history_proposal: Option<&BTreeMap<FieldKey, Policy>>,
) -> BTreeMap<FieldKey, Policy> {
    let mut out = online.clone();
    if let Some(proposal) = history_proposal {
        for (fkey, pol) in proposal {
            let cur = out.get(fkey).copied().unwrap_or(Policy::Drop);
            out.insert(fkey.clone(), fuse(cur, *pol));
        }
    }
    out
}

pub const SUMMARY_STUB_TOKENS: usize = 25;

/// Apply one policy to a field for the model-visible context (Sec. 3.3 /
/// App. A.3). Returns (rendered_text_or_None, fell_back).
///
/// EXACT: original value. SUMMARY: bounded stub -- short scalars are
/// retained whole, long text is truncated to the first
/// SUMMARY_STUB_TOKENS tokens. POINTER: opaque reference token if the
/// artifact is verifiably in the runtime state, else fall back to EXACT.
/// DROP: omitted from the visible context.
pub fn project_field(
    field: &Field,
    policy: Policy,
    runtime_state: &HashMap<String, String>,
) -> (Option<String>, bool) {
    match policy {
        Policy::Drop => (None, false),
        Policy::Exact => (Some(field.value.clone()), false),
        Policy::Summary => {
            if !field.long_text {
                (Some(field.value.clone()), false)
            } else {
                let toks: Vec<&str> = field.value.split_whitespace().collect();
                let stub: Vec<&str> = toks.iter().take(SUMMARY_STUB_TOKENS).copied().collect();
                (Some(format!("{} ...[summary]", stub.join(" "))), false)
            }
        }
        Policy::Pointer => {
            if let Some(aid) = &field.artifact_id {
                if runtime_state.contains_key(aid) {
                    return (Some(format!("<ptr:{aid}>")), false);
                }
            }
            (Some(field.value.clone()), true) // invalid reference -> EXACT
        }
    }
}

/// Capability set the projected representation of `field` actually
/// provides to a consumer. Mirrors `project_field`: a SUMMARY stub of a
/// short scalar retains the literal value, so it also provides
/// `literal`; a POINTER with an invalid reference falls back to EXACT
/// capabilities; DROP provides nothing.
pub fn effective_caps(
    field: &Field,
    policy: Policy,
    runtime_state: &HashMap<String, String>,
) -> u8 {
    match policy {
        Policy::Drop => 0,
        Policy::Exact => 0b111,
        Policy::Summary => {
            let mut caps = 0b010u8;
            if !field.long_text {
                caps |= 0b001;
            }
            caps
        }
        Policy::Pointer => {
            if let Some(aid) = &field.artifact_id {
                if runtime_state.contains_key(aid) {
                    return 0b100;
                }
            }
            0b111
        }
    }
}

/// Project all fields produced by ops < `step` under `policies`.
///
/// Returns (rendered, caps, n_fallbacks) with rendered mapping
/// (op_idx, field_name) -> visible text (absent = DROP) and caps mapping
/// the same key -> the capability set a consumer can actually use.
pub fn project_context(
    dag: &TraceDAG,
    policies: &BTreeMap<FieldKey, Policy>,
    runtime_state: &HashMap<String, String>,
    step: usize,
) -> (BTreeMap<FieldKey, String>, BTreeMap<FieldKey, u8>, usize) {
    let mut rendered: BTreeMap<FieldKey, String> = BTreeMap::new();
    let mut caps: BTreeMap<FieldKey, u8> = BTreeMap::new();
    let mut n_fallbacks = 0usize;
    for i in 0..step {
        for field in &dag.nodes[i].outputs {
            let pol = policies
                .get(&(i, field.name.clone()))
                .copied()
                .unwrap_or(Policy::Drop);
            let (text, fell_back) = project_field(field, pol, runtime_state);
            if fell_back {
                n_fallbacks += 1;
            }
            if let Some(t) = text {
                rendered.insert((i, field.name.clone()), t);
            }
            caps.insert(
                (i, field.name.clone()),
                effective_caps(field, pol, runtime_state),
            );
        }
    }
    (rendered, caps, n_fallbacks)
}
