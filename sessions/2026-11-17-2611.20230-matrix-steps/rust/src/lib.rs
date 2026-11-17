//! A matrix with steps: episodic and procedural memory (arXiv:2611.20230).
//!
//! Rust port of `arxiv_lab.memory.matrix_steps` (Python, stdlib + numpy).
//! Std only, no external crates, no network, no model calls.
//!
//! The Python module stays the primary verdict carrier; this port
//! cross-checks it. A failed Rust leg never overturns the Python verdict.

use std::collections::HashMap;

// ---------------------------------------------------------------- constants
/// Only a verification from a check, a test, a tool result or a person
/// counts. A model's statement never verifies an outcome.
pub const VERIFIED_SOURCES: [&str; 4] =
    ["external_check", "test", "tool_result", "person"];
pub const MODEL_CLAIM: &str = "model_claim";

/// Consolidation minimums (paper Sec. 4).
pub const MIN_EPISODES: usize = 5;
pub const MIN_SUCCESSES: usize = 3;
pub const MIN_COVERAGE: f64 = 0.60;

/// First-attempt evidence in fixed windows of 40 attempts/arm (Sec. 4).
pub const WINDOW_ATTEMPTS: usize = 40;

/// Writer limits per matrix type at d=1024 (cols/d -> max procedures).
pub fn matrix_limit(cols_over_dim: usize) -> Option<usize> {
    match cols_over_dim {
        1 => Some(200),
        4 => Some(500),
        8 => Some(1000),
        _ => None,
    }
}

/// Cosine below this means "no procedure found".
pub const FIND_COSINE_THRESHOLD: f64 = 0.50;

pub const ONLINE_ALPHA: f32 = 0.995;
pub const ONLINE_BETA_BASE: f32 = 0.25;
pub const REBUILD_ALPHA: f32 = 1.0;
pub const REBUILD_BETA_BASE: f32 = 1.0;
pub const REBUILD_PASSES: usize = 10;

pub const VOCAB_SIZE: usize = 3000;
pub const EMBED_DIM: usize = 1024;
pub const EMBED_SEED: u64 = 0x5EEDBEEF;

const SPLITMIX_C1: u64 = 0x9E3779B97F4A7C15;
const SPLITMIX_C2: u64 = 0xBF58476D1CE4E5B9;

const STOPWORDS: [&str; 23] = [
    "the", "a", "an", "on", "in", "at", "when", "to", "of", "for", "and",
    "is", "needs", "with", "by", "it", "its", "this", "that", "be", "are",
    "was", "were",
];

// ---------------------------------------------------------------- hashing
/// 64-bit FNV-1a hash of a str (paper Sec. 3.1: seeds the sign vector).
pub fn fnv1a_64(text: &str) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

/// SplitMix64 finalizer (pure integer ops; identical to the Python port).
pub fn splitmix64_finalizer(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

// ---------------------------------------------------------------- MT19937
/// Mersenne Twister (reference mt19937ar init_genrand / init_by_array).
/// Seeded by init_by_array over the two 32-bit halves of the FNV-1a hash,
/// exactly as in the Python port, so sign vectors agree bit for bit.
pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub fn from_seed_words(seed_words: &[u32]) -> Self {
        let mut mt = [0u32; 624];
        // init_genrand(19650218)
        mt[0] = 19650218;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        // init_by_array
        let key: Vec<u32> = seed_words.to_vec();
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = 624.max(key.len());
        while k > 0 {
            mt[i] = (mt[i]
                ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        for _ in 0..623 {
            mt[i] = (mt[i]
                ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
        }
        mt[0] = 0x80000000;
        Mt19937 { mt, idx: 624 }
    }

    /// Classic init_genrand path (used for the known test vector).
    pub fn from_int_seed(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.mt[i] & 0x80000000)
                | (self.mt[(i + 1) % 624] & 0x7FFFFFFF);
            self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                self.mt[i] ^= 0x9908B0DF;
            }
        }
        self.idx = 0;
    }

    /// One tempered 32-bit draw.
    pub fn uint32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= y >> 18;
        y
    }
}

/// Deterministic RNG on top of MT19937 (shared with the Python port).
pub struct Rng {
    mt: Mt19937,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Mirror the Python Rng: a str seed is FNV-1a hashed first (done by
        // the caller there); here the seed is the u64 value itself.
        let mt = if seed < (1 << 32) {
            Mt19937::from_int_seed(seed as u32)
        } else {
            Mt19937::from_seed_words(&[seed as u32, (seed >> 32) as u32])
        };
        Rng { mt }
    }

    pub fn from_str_seed(text: &str) -> Self {
        Rng::new(fnv1a_64(text))
    }

    pub fn uint32(&mut self) -> u32 {
        self.mt.uint32()
    }

    /// Uniform float in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        self.uint32() as f64 / 4294967296.0
    }

    /// Uniform int in [a, b] inclusive.
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        a + (self.uniform() * (b - a + 1) as f64) as i64
    }

    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        &seq[self.randint(0, seq.len() as i64 - 1) as usize]
    }

    pub fn shuffle<T: Clone>(&mut self, seq: &[T]) -> Vec<T> {
        let mut v = seq.to_vec();
        for i in (1..v.len()).rev() {
            let j = self.randint(0, i as i64) as usize;
            v.swap(i, j);
        }
        v
    }
}

// ---------------------------------------------------------------- embedder
fn is_stopword(tok: &str) -> bool {
    STOPWORDS.contains(&tok)
}

fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Deterministic bag-of-words hash embedder (paper Sec. 6): the hashing
/// trick -- each token hashes (FNV-1a) to one of `dim` buckets with a sign
/// from the hash's top bit; the text vector is the signed count vector,
/// L2-normalized.
#[derive(Clone)]
pub struct HashEmbedder {
    pub vocab_size: usize,
    pub dim: usize,
    pub seed: u64,
    cache: HashMap<String, Vec<f64>>,
}

impl HashEmbedder {
    pub fn new() -> Self {
        HashEmbedder {
            vocab_size: VOCAB_SIZE,
            dim: EMBED_DIM,
            seed: EMBED_SEED,
            cache: HashMap::new(),
        }
    }

    pub fn with_dim(dim: usize) -> Self {
        HashEmbedder {
            vocab_size: VOCAB_SIZE,
            dim,
            seed: EMBED_SEED,
            cache: HashMap::new(),
        }
    }

    pub fn embed(&mut self, text: &str) -> Vec<f64> {
        if let Some(v) = self.cache.get(text) {
            return v.clone();
        }
        let mut vec = vec![0.0f64; self.dim];
        for tok in tokenize(text) {
            if is_stopword(&tok) {
                continue;
            }
            let h = fnv1a_64(&tok);
            let b = (h % self.dim as u64) as usize;
            vec[b] += if h >> 63 == 1 { 1.0 } else { -1.0 };
        }
        let n: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 0.0 {
            for x in vec.iter_mut() {
                *x /= n;
            }
        }
        self.cache.insert(text.to_string(), vec.clone());
        vec
    }
}

/// Per-procedure sign vector s_p in {-1,+1}^width (paper Sec. 3.1).
/// 64-bit FNV-1a hash of the id seeds a Mersenne Twister; the low bit of
/// each draw gives one sign.
pub fn procedure_sign_vector(proc_id: &str, width: usize) -> Vec<f32> {
    let h = fnv1a_64(proc_id);
    let mut mt = Mt19937::from_seed_words(&[h as u32, (h >> 32) as u32]);
    (0..width)
        .map(|_| if mt.uint32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

// ---------------------------------------------------------------- matrix
/// Delta-rule associative matrix M in R^(dim x cols) (paper Eq. 1).
/// Stored row-major as a flat Vec<f32> of dim*cols.
pub struct DeltaMatrix {
    pub dim: usize,
    pub cols: usize,
    pub widening: String,
    pub seed: u64,
    pub m: Vec<f32>,
    proj: Option<Vec<f32>>, // cols x dim, Rademacher / sqrt(dim)
}

impl DeltaMatrix {
    pub fn new(dim: usize, cols: Option<usize>, widening: &str, seed: u64) -> Self {
        let cols = cols.unwrap_or(dim);
        let widening = if cols == dim {
            "none".to_string()
        } else {
            widening.to_string()
        };
        let mut proj = None;
        if cols != dim {
            let mut p = vec![0.0f32; cols * dim];
            for i in 0..cols {
                for j in 0..dim {
                    let z = splitmix64_finalizer(
                        seed
                            .wrapping_add((i as u64).wrapping_mul(SPLITMIX_C1))
                            .wrapping_add((j as u64).wrapping_mul(SPLITMIX_C2)),
                    );
                    p[i * dim + j] =
                        (if z & 1 == 1 { 1.0 } else { -1.0 }) / (dim as f32).sqrt();
                }
            }
            proj = Some(p);
        }
        DeltaMatrix {
            dim,
            cols,
            widening,
            seed,
            m: vec![0.0; dim * cols],
            proj,
        }
    }

    /// Widen a dim-vector key to `cols` columns (paper Sec. 3.1).
    pub fn widen(&self, k: &[f32]) -> Vec<f32> {
        if self.cols == self.dim {
            return k.to_vec();
        }
        let r = self.proj.as_ref().unwrap();
        if self.widening == "sign" {
            let mut w = vec![0.0f32; self.cols];
            for i in 0..self.cols {
                let mut s = 0.0f32;
                for j in 0..self.dim {
                    s += r[i * self.dim + j] * k[j];
                }
                w[i] = if s >= 0.0 { 1.0 } else { -1.0 } / (self.cols as f32).sqrt();
            }
            w
        } else {
            // "linear" control widening: phi(e) = R e / ||R e||
            let mut w = vec![0.0f32; self.cols];
            for i in 0..self.cols {
                let mut s = 0.0f32;
                for j in 0..self.dim {
                    s += r[i * self.dim + j] * k[j];
                }
                w[i] = s;
            }
            let n: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                for x in w.iter_mut() {
                    *x /= n;
                }
            }
            w
        }
    }

    /// One delta-rule write of key k (dim-vector) to value v.
    pub fn write(&mut self, k: &[f32], v: &[f32], alpha: f32, beta: f32) {
        let kw = self.widen(k);
        if alpha != 1.0 {
            for x in self.m.iter_mut() {
                *x *= alpha;
            }
        }
        // resid = v - M kw
        let mut resid = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut s = 0.0f32;
            for j in 0..self.cols {
                s += self.m[i * self.cols + j] * kw[j];
            }
            resid[i] = v[i] - s;
        }
        for i in 0..self.dim {
            let r = beta * resid[i];
            for j in 0..self.cols {
                self.m[i * self.cols + j] += r * kw[j];
            }
        }
    }

    /// Recall: v_hat = M q.
    pub fn read(&self, q: &[f32]) -> Vec<f32> {
        let qw = self.widen(q);
        let mut out = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut s = 0.0f32;
            for j in 0..self.cols {
                s += self.m[i * self.cols + j] * qw[j];
            }
            out[i] = s;
        }
        out
    }

    /// Rebuild from the write list: alpha=1, beta given, `passes` passes.
    pub fn rebuild(&mut self, pairs: &[(Vec<f32>, Vec<f32>)], passes: usize, beta: f32) {
        for x in self.m.iter_mut() {
            *x = 0.0;
        }
        for _ in 0..passes {
            for (k, v) in pairs {
                self.write(k, v, REBUILD_ALPHA, beta);
            }
        }
    }

    /// Writer refuses a scope larger than the type's limit (paper Sec. 3).
    pub fn check_limit(&self, n_procedures: usize) -> Result<(), String> {
        let ratio = self.cols / self.dim;
        if let Some(limit) = matrix_limit(ratio) {
            if n_procedures > limit {
                return Err(format!(
                    "scope of {} procedures exceeds the {}x{} matrix limit {}; split the scope",
                    n_procedures, self.dim, self.cols, limit
                ));
            }
        }
        Ok(())
    }
}

/// Cosine similarity between two float vectors.
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = na.sqrt() * nb.sqrt();
    if d > 0.0 {
        dot / d
    } else {
        0.0
    }
}

pub fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64) * (a[i] as f64);
        nb += (b[i] as f64) * (b[i] as f64);
    }
    let d = na.sqrt() * nb.sqrt();
    if d > 0.0 {
        dot / d
    } else {
        0.0
    }
}

pub fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

pub fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

// ---------------------------------------------------------------- records
/// One procedure step: a GOAL, not a motion (paper Sec. 3.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub action: String,
    pub precondition: String,
    pub done: String,
    pub tool: String,
    pub refines: Option<String>,
}

impl Step {
    pub fn new(action: &str) -> Self {
        Step {
            action: action.to_string(),
            precondition: String::new(),
            done: String::new(),
            tool: String::new(),
            refines: None,
        }
    }

    pub fn text(&self) -> &str {
        &self.action
    }
}

/// An ordered, addressable step record (paper Sec. 3.2).
#[derive(Clone, Debug)]
pub struct ProcedureRecord {
    pub id: String,
    pub name: String,
    pub goal: String,
    pub when_to_use: String,
    pub steps: Vec<Step>,
    pub attempts: usize,
    pub successes: usize,
    pub status: String, // active | revoked | demoted
    pub predecessor: Option<String>,
    pub episode_ids: Vec<String>,
}

impl ProcedureRecord {
    pub fn new(id: &str, name: &str, goal: &str, when_to_use: &str, steps: Vec<Step>) -> Self {
        ProcedureRecord {
            id: id.to_string(),
            name: name.to_string(),
            goal: goal.to_string(),
            when_to_use: when_to_use.to_string(),
            steps,
            attempts: 0,
            successes: 0,
            status: "active".to_string(),
            predecessor: None,
            episode_ids: Vec::new(),
        }
    }

    /// Key text: name + goal + when-to-use (paper Sec. 3.1).
    pub fn key_text(&self) -> String {
        format!("{} {} {}", self.name, self.goal, self.when_to_use)
    }

    /// Index text: name + goal + the actions of its steps.
    pub fn answer_text(&self) -> String {
        let actions: Vec<&str> = self.steps.iter().map(|s| s.text()).collect();
        format!("{} {} {}", self.name, self.goal, actions.join(" "))
    }
}

/// One recorded run (paper Sec. 4).
#[derive(Clone, Debug)]
pub struct Episode {
    pub id: String,
    pub scope: String,
    pub task_family: String,
    pub goal: String,
    pub steps: Vec<String>,
    pub outcome: String, // "success" | "failure" (as recorded)
    pub verification: String,
    pub agent_claimed_done: bool,
    pub tool_failed: bool,
    pub person_verdict: Option<String>,
    pub args: Vec<(String, String)>,
    pub condition: String,
    pub failure_mode: Option<String>,
}

impl Episode {
    pub fn new(
        id: &str, scope: &str, task_family: &str, goal: &str, steps: Vec<String>,
        outcome: &str, verification: &str,
    ) -> Self {
        Episode {
            id: id.to_string(),
            scope: scope.to_string(),
            task_family: task_family.to_string(),
            goal: goal.to_string(),
            steps,
            outcome: outcome.to_string(),
            verification: verification.to_string(),
            agent_claimed_done: false,
            tool_failed: false,
            person_verdict: None,
            args: Vec::new(),
            condition: String::new(),
            failure_mode: None,
        }
    }
}

/// The outcome an episode counts as, under the write gate (Sec. 4).
/// Returns None when the outcome is unverified (never counted, never
/// shown as an example).
pub fn recorded_outcome(ep: &Episode) -> Option<String> {
    if let Some(v) = &ep.person_verdict {
        return Some(v.clone());
    }
    if ep.tool_failed {
        return Some("failure".to_string());
    }
    if !VERIFIED_SOURCES.contains(&ep.verification.as_str()) {
        return None;
    }
    Some(ep.outcome.clone())
}

/// Consolidation confidence (s+1)/(n+2) (paper Sec. 4).
pub fn confidence(s: usize, n: usize) -> f64 {
    (s + 1) as f64 / (n + 2) as f64
}

/// A deterministic consolidation record.
#[derive(Clone, Debug)]
pub struct ConsolidatedRecord {
    pub steps: Vec<String>,
    pub fixed_args: Vec<(String, String)>,
    pub varied_args: Vec<(String, Vec<(String, usize)>)>,
    pub conditions: Vec<String>,
    pub failure_modes: Vec<(String, usize)>,
    pub confidence: f64,
    pub n_episodes: usize,
    pub n_successes: usize,
    pub top_sequence_share: f64,
    pub episode_ids: Vec<String>,
}

/// Deterministic consolidation (paper Sec. 4); no model call.
/// `episodes` must already be write-gated (verified only). Returns None
/// when the minimums are not met.
pub fn consolidate_episodes(
    episodes: &[Episode],
    min_episodes: usize,
    min_successes: usize,
    min_coverage: f64,
) -> Option<ConsolidatedRecord> {
    let counted: Vec<&Episode> = episodes
        .iter()
        .filter(|e| recorded_outcome(e).is_some())
        .collect();
    let n = counted.len();
    let successes: Vec<&Episode> = counted
        .iter()
        .filter(|e| recorded_outcome(e).as_deref() == Some("success"))
        .copied()
        .collect();
    let s = successes.len();
    if n < min_episodes || s < min_successes {
        return None;
    }
    let mut seq_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for e in &successes {
        *seq_counts.entry(e.steps.clone()).or_insert(0) += 1;
    }
    let (top_seq, top_count) = seq_counts
        .iter()
        .max_by_key(|(_, c)| **c)
        .map(|(k, v)| (k.clone(), *v))
        .unwrap();
    if top_count as f64 / s as f64 >= min_coverage {
        let mut fixed = Vec::new();
        let mut varied = Vec::new();
        if !successes.is_empty() {
            let mut keys: Vec<String> =
                successes[0].args.iter().map(|(k, _)| k.clone()).collect();
            keys.sort();
            for k in keys {
                let vals: Vec<Option<&String>> = successes
                    .iter()
                    .map(|e| e.args.iter().find(|(kk, _)| kk == &k).map(|(_, v)| v))
                    .collect();
                if vals.iter().all(|v| v == &vals[0]) {
                    if let Some(v) = vals[0] {
                        fixed.push((k, v.clone()));
                    }
                } else {
                    let mut dist: Vec<(String, usize)> = Vec::new();
                    for v in vals.iter().flatten() {
                        if let Some(e) = dist.iter_mut().find(|(vv, _)| vv == *v) {
                            e.1 += 1;
                        } else {
                            dist.push(((*v).clone(), 1));
                        }
                    }
                    varied.push((k, dist));
                }
            }
        }
        let mut conditions: Vec<String> = successes
            .iter()
            .filter(|e| !e.condition.is_empty())
            .map(|e| e.condition.clone())
            .collect();
        conditions.sort();
        conditions.dedup();
        let mut failures: Vec<(String, usize)> = Vec::new();
        for e in &counted {
            if recorded_outcome(e).as_deref() == Some("failure") {
                if let Some(fm) = &e.failure_mode {
                    if let Some(en) = failures.iter_mut().find(|(v, _)| v == fm) {
                        en.1 += 1;
                    } else {
                        failures.push((fm.clone(), 1));
                    }
                }
            }
        }
        Some(ConsolidatedRecord {
            steps: top_seq,
            fixed_args: fixed,
            varied_args: varied,
            conditions,
            failure_modes: failures,
            confidence: confidence(s, n),
            n_episodes: n,
            n_successes: s,
            top_sequence_share: top_count as f64 / s as f64,
            episode_ids: counted.iter().map(|e| e.id.clone()).collect(),
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------- memory
/// One erasure audit entry: names the record and the cause, never the person.
#[derive(Clone, Debug)]
pub struct AuditEntry {
    pub record: String,
    pub cause: String,
    pub action: String,
}

/// One scope's episodic + procedural memory (paper Sec. 3-4).
pub struct EpisodicMemory {
    pub scope: String,
    pub embedder: HashEmbedder,
    pub matrix: DeltaMatrix,
    pub episodes: HashMap<String, Episode>,
    pub records: HashMap<String, ProcedureRecord>,
    /// Insertion order of record ids (HashMap has none).
    pub record_order: Vec<String>,
    pub audit: Vec<AuditEntry>,
    pub default_id: Option<String>,
    pub rebuild_beta: f32,
}

impl EpisodicMemory {
    pub fn new(scope: &str, dim: usize, rebuild_beta: f32) -> Self {
        EpisodicMemory {
            scope: scope.to_string(),
            embedder: HashEmbedder::with_dim(dim),
            matrix: DeltaMatrix::new(dim, None, "sign", 777),
            episodes: HashMap::new(),
            records: HashMap::new(),
            record_order: Vec::new(),
            audit: Vec::new(),
            default_id: None,
            rebuild_beta,
        }
    }

    /// Store an episode; returns its counted outcome (None if unverified).
    pub fn record_episode(&mut self, ep: Episode) -> Option<String> {
        assert_eq!(ep.scope, self.scope, "episode scope does not match memory scope");
        let out = recorded_outcome(&ep);
        self.episodes.insert(ep.id.clone(), ep);
        out
    }

    /// Verified episodes of this scope only (never another scope's).
    pub fn verified_episodes(&self, task_family: Option<&str>) -> Vec<Episode> {
        let mut out: Vec<Episode> = self
            .episodes
            .values()
            .filter(|ep| ep.scope == self.scope)
            .filter(|ep| task_family.map_or(true, |f| ep.task_family == f))
            .filter(|ep| recorded_outcome(ep).is_some())
            .cloned()
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    pub fn add_record(&mut self, rec: ProcedureRecord) {
        if !self.records.contains_key(&rec.id) {
            self.record_order.push(rec.id.clone());
        }
        self.records.insert(rec.id.clone(), rec);
    }

    fn active_records(&self) -> Vec<ProcedureRecord> {
        self.record_order
            .iter()
            .filter_map(|id| self.records.get(id))
            .filter(|r| r.status == "active")
            .cloned()
            .collect()
    }

    /// Write list: one find association per procedure plus step chains.
    pub fn write_pairs(&mut self) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut pairs = Vec::new();
        for rec in self.active_records() {
            let key = to_f32(&self.embedder.embed(&rec.key_text()));
            let val = to_f32(&self.embedder.embed(&rec.answer_text()));
            pairs.push((key, val));
            let signs = procedure_sign_vector(&rec.id, self.matrix.cols);
            let step_vecs: Vec<Vec<f32>> = rec
                .steps
                .iter()
                .map(|s| to_f32(&self.embedder.embed(s.text())))
                .collect();
            for i in 0..step_vecs.len().saturating_sub(1) {
                let k = self.matrix.widen(&step_vecs[i]);
                let kb: Vec<f32> =
                    k.iter().zip(signs.iter()).map(|(a, b)| a * b).collect();
                pairs.push((kb, step_vecs[i + 1].clone()));
            }
        }
        pairs
    }

    pub fn rebuild_matrix(&mut self, passes: usize, enforce_limit: bool) {
        let active = self.active_records();
        if enforce_limit {
            self.matrix.check_limit(active.len()).unwrap();
        }
        let pairs = self.write_pairs();
        self.matrix.rebuild(&pairs, passes, self.rebuild_beta);
    }

    /// Find the procedure for a goal text. Returns (record id or None,
    /// best cosine). `min_cosine` is the no-match threshold; pass 0.0 for
    /// pure argmax, as the paper's capacity table reports "found".
    pub fn find(
        &mut self, query_text: &str, method: &str, min_cosine: f64,
    ) -> (Option<String>, f64) {
        let active = self.active_records();
        if active.is_empty() {
            return (None, 0.0);
        }
        let q = self.embedder.embed(query_text);
        let v_hat: Vec<f64> = match method {
            "matrix" => to_f64(&self.matrix.read(&to_f32(&q))),
            "exact" => q.clone(),
            _ => panic!("unknown find method"),
        };
        let mut best_i: Option<usize> = None;
        let mut best_c = -1.0f64;
        for (i, r) in active.iter().enumerate() {
            let cand = if method == "matrix" {
                self.embedder.embed(&r.answer_text())
            } else {
                self.embedder.embed(&r.key_text())
            };
            let c = cosine(&v_hat, &cand);
            if c > best_c {
                best_c = c;
                best_i = Some(i);
            }
        }
        if best_c < min_cosine {
            return (None, best_c);
        }
        (best_i.map(|i| active[i].id.clone()), best_c)
    }

    /// What comes after step i, from the records (source of truth).
    pub fn next_step(&self, proc_id: &str, step_index: usize) -> Option<Step> {
        let rec = self.records.get(proc_id)?;
        rec.steps.get(step_index + 1).cloned()
    }

    /// Next step decoded from the matrix (bound step key -> step i+1).
    /// Returns (step action text or None, best cosine).
    pub fn next_step_via_matrix(
        &mut self, proc_id: &str, step_index: usize,
    ) -> (Option<String>, f64) {
        let rec = match self.records.get(proc_id) {
            Some(r) => r.clone(),
            None => return (None, 0.0),
        };
        if step_index + 1 >= rec.steps.len() {
            return (None, 0.0);
        }
        let signs = procedure_sign_vector(proc_id, self.matrix.cols);
        let e_i = to_f32(&self.embedder.embed(rec.steps[step_index].text()));
        let k = self.matrix.widen(&e_i);
        let kb: Vec<f32> = k.iter().zip(signs.iter()).map(|(a, b)| a * b).collect();
        let v_hat = to_f64(&self.matrix.matvec(&kb));
        let mut best: Option<String> = None;
        let mut best_c = -1.0f64;
        for r in self.active_records() {
            for s in &r.steps {
                let cv = self.embedder.embed(s.text());
                let c = cosine(&v_hat, &cv);
                if c > best_c {
                    best_c = c;
                    best = Some(s.text().to_string());
                }
            }
        }
        (best, best_c)
    }

    /// Consolidate this scope's verified episodes of one family.
    pub fn consolidate(&self, task_family: &str) -> Option<ConsolidatedRecord> {
        let eps = self.verified_episodes(Some(task_family));
        consolidate_episodes(&eps, MIN_EPISODES, MIN_SUCCESSES, MIN_COVERAGE)
    }

    /// Erase episodes; recompute affected records; rebuild the matrix.
    pub fn erase_episodes(
        &mut self, episode_ids: &[String], cause: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut erased = Vec::new();
        for eid in episode_ids {
            if self.episodes.remove(eid).is_some() {
                erased.push(eid.clone());
            }
        }
        let mut affected: Vec<String> = Vec::new();
        let rec_ids: Vec<String> = self.record_order.clone();
        for rid in rec_ids {
            let mut drop = false;
            let mut correct = false;
            if let Some(rec) = self.records.get_mut(&rid) {
                let kept: Vec<String> = rec
                    .episode_ids
                    .iter()
                    .filter(|eid| self.episodes.contains_key(*eid))
                    .cloned()
                    .collect();
                if kept.len() != rec.episode_ids.len() {
                    affected.push(rid.clone());
                    rec.episode_ids = kept;
                    let eps: Vec<Episode> = rec
                        .episode_ids
                        .iter()
                        .filter_map(|eid| self.episodes.get(eid))
                        .cloned()
                        .collect();
                    match consolidate_episodes(
                        &eps, MIN_EPISODES, MIN_SUCCESSES, MIN_COVERAGE,
                    ) {
                        None => drop = true,
                        Some(new) => {
                            rec.attempts = new.n_episodes;
                            rec.successes = new.n_successes;
                            correct = true;
                        }
                    }
                }
            }
            if drop {
                self.records.remove(&rid);
                self.record_order.retain(|x| x != &rid);
                self.audit.push(AuditEntry {
                    record: rid.clone(),
                    cause: cause.to_string(),
                    action: "dropped below minimums".to_string(),
                });
            } else if correct {
                self.audit.push(AuditEntry {
                    record: rid.clone(),
                    cause: cause.to_string(),
                    action: "corrected from remaining episodes".to_string(),
                });
            }
        }
        self.rebuild_matrix(REBUILD_PASSES, true);
        (erased, affected)
    }

    pub fn set_default(&mut self, proc_id: Option<&str>, predecessor: Option<&str>) {
        if let Some(id) = proc_id {
            assert!(self.records.contains_key(id), "unknown procedure");
            if let Some(rec) = self.records.get_mut(id) {
                rec.predecessor = predecessor.map(|s| s.to_string());
            }
        }
        self.default_id = proc_id.map(|s| s.to_string());
    }

    /// Demotion needs no test (paper Sec. 4): a default whose real
    /// attempt ends in a protective stop loses its place at once, and the
    /// predecessor becomes default again. Returns the restored default.
    pub fn demote_on_guard_trip(&mut self, proc_id: &str) -> Option<String> {
        if self.default_id.as_deref() != Some(proc_id) {
            return None;
        }
        let prev = self.records.get(proc_id).and_then(|r| r.predecessor.clone());
        if let Some(rec) = self.records.get_mut(proc_id) {
            rec.status = "demoted".to_string();
        }
        self.default_id = prev.clone();
        self.rebuild_matrix(REBUILD_PASSES, true);
        prev
    }
}

// ---------------------------------------------------------------- promotion
/// One-sided p-value for candidate success rate > default (pooled z-test).
pub fn z_test_one_sided_p(x_cand: usize, n_cand: usize, x_def: usize, n_def: usize) -> f64 {
    if n_cand == 0 || n_def == 0 {
        return 1.0;
    }
    let p_c = x_cand as f64 / n_cand as f64;
    let p_d = x_def as f64 / n_def as f64;
    let p_pool = (x_cand + x_def) as f64 / (n_cand + n_def) as f64;
    let var = p_pool * (1.0 - p_pool) * (1.0 / n_cand as f64 + 1.0 / n_def as f64);
    if var <= 0.0 {
        return if p_c > p_d { 0.0 } else { 1.0 };
    }
    let z = (p_c - p_d) / var.sqrt();
    // Normal survival function via erf (Abramowitz-Stegun 7.1.26 is not
    // needed: use the error function from the standard library -- not
    // available in core, so use a rational approximation here).
    0.5 * (1.0 - erf(z / std::f64::consts::SQRT_2))
}

/// Error function (Abramowitz & Stegun 7.1.26, |eps| < 1.5e-7).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736)
            * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn promotion_window_counts(
    rng: &mut Rng, p_default: f64, p_candidate: f64, window: usize,
) -> (usize, usize, usize, usize) {
    let (mut n_d, mut n_c, mut x_d, mut x_c) = (0usize, 0usize, 0usize, 0usize);
    while n_d < window || n_c < window {
        if rng.bernoulli(0.5) {
            if n_c < window {
                n_c += 1;
                if rng.bernoulli(p_candidate) {
                    x_c += 1;
                }
            }
        } else if n_d < window {
            n_d += 1;
            if rng.bernoulli(p_default) {
                x_d += 1;
            }
        }
    }
    (x_d, n_d, x_c, n_c)
}

/// One two-stage promotion test on FIRST-attempt evidence (paper Sec. 4).
/// Returns (promoted, screen_p, confirm_p_or_None).
pub fn promotion_window(
    rng: &mut Rng, p_default: f64, p_candidate: f64, window: usize,
    alpha_screen: f64, alpha_confirm: f64,
) -> (bool, f64, Option<f64>) {
    let (x_d, n_d, x_c, n_c) = promotion_window_counts(rng, p_default, p_candidate, window);
    let p_screen = z_test_one_sided_p(x_c, n_c, x_d, n_d);
    if p_screen >= alpha_screen {
        return (false, p_screen, None);
    }
    let (x_d2, n_d2, x_c2, n_c2) =
        promotion_window_counts(rng, p_default, p_candidate, window);
    let p_confirm = z_test_one_sided_p(x_c2, n_c2, x_d2, n_d2);
    (p_confirm < alpha_confirm, p_screen, Some(p_confirm))
}

/// Control: the rule that keeps any higher score (paper Sec. 6).
pub fn naive_keep_count(
    rng: &mut Rng, p_default: f64, p_candidate: f64, n_windows: usize, window: usize,
) -> usize {
    let mut kept = 0;
    for _ in 0..n_windows {
        let x_d: usize = (0..window).map(|_| rng.bernoulli(p_default) as usize).sum();
        let x_c: usize = (0..window).map(|_| rng.bernoulli(p_candidate) as usize).sum();
        if x_c > x_d {
            kept += 1;
        }
    }
    kept
}

/// Run `n_windows` two-stage promotion tests; return (promoted, naive_kept).
pub fn run_promotion_windows(
    seed: u64, p_default: f64, p_candidate: f64, n_windows: usize, window: usize,
) -> (usize, usize) {
    let mut rng = Rng::new(seed);
    let mut promoted = 0;
    for _ in 0..n_windows {
        let (ok, _, _) =
            promotion_window(&mut rng, p_default, p_candidate, window, 0.025, 0.025);
        promoted += ok as usize;
    }
    let mut rng2 = Rng::new(seed + 1);
    let naive = naive_keep_count(&mut rng2, p_default, p_candidate, n_windows, window);
    (promoted, naive)
}

// Raw matvec M @ kw with an already-widened key (used for bound step keys,
// which Python's next_step_via_matrix also multiplies directly).
impl DeltaMatrix {
    pub fn matvec(&self, kw: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut s = 0.0f32;
            for j in 0..self.cols {
                s += self.m[i * self.cols + j] * kw[j];
            }
            out[i] = s;
        }
        out
    }
}

// ---------------------------------------------------------------- synthetic data
/// The paper's synthetic capacity data (Sec. 6) draws procedure texts as
/// random word sets from the 3000-word vocabulary.
pub fn synth_vocab() -> Vec<String> {
    (0..VOCAB_SIZE).map(|i| format!("w{:04}", i)).collect()
}

fn shared_step_pool() -> Vec<String> {
    let mut rng = Rng::new(999);
    let vocab = synth_vocab();
    (0..12)
        .map(|_| {
            (0..3)
                .map(|_| rng.choice(&vocab).clone())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Synthetic procedures (paper Sec. 6): 3-8 steps each, ~15% of step texts
/// shared between procedures. Texts are random word sets from the
/// 3000-word vocabulary, so distinct procedures have near-independent
/// embeddings -- the premise of the paper's capacity table.
pub fn generate_synthetic_procedures(
    n: usize, seed: u64, steps_min: usize, steps_max: usize, shared_frac: f64,
) -> Vec<ProcedureRecord> {
    let mut rng = Rng::new(seed);
    let vocab = synth_vocab();
    let shared_pool = shared_step_pool();
    let mut procs = Vec::new();
    for i in 0..n {
        let name: Vec<String> =
            (0..2).map(|_| rng.choice(&vocab).clone()).collect();
        let goal: Vec<String> =
            (0..6).map(|_| rng.choice(&vocab).clone()).collect();
        let when: Vec<String> =
            (0..4).map(|_| rng.choice(&vocab).clone()).collect();
        let n_steps = rng.randint(steps_min as i64, steps_max as i64) as usize;
        let mut steps = Vec::new();
        for _ in 0..n_steps {
            let action = if rng.uniform() < shared_frac {
                rng.choice(&shared_pool).clone()
            } else {
                (0..3)
                    .map(|_| rng.choice(&vocab).clone())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let mut s = Step::new(&action);
            s.tool = format!("tool_{}", rng.randint(1, 6));
            s.done = format!("{} done", action);
            steps.push(s);
        }
        procs.push(ProcedureRecord::new(
            &format!("proc_{:04}", i),
            &name.join(" "),
            &goal.join(" "),
            &when.join(" "),
            steps,
        ));
    }
    procs
}

/// Capacity experiment result (paper Sec. 6 capacity table).
#[derive(Clone, Debug)]
pub struct CapacityResult {
    pub n_procedures: usize,
    pub n_associations: usize,
    pub found_matrix: f64,
    pub found_exact: f64,
    pub next_step: f64,
    pub median_lookup_ms: f64,
}

/// Paper Sec. 6 capacity table: d=1024 square matrix, hash embedder.
/// "Found" is pure argmax, as the paper reports it.
pub fn capacity_experiment(
    n_procedures: usize, seed: u64, passes: usize, beta: f32, dim: usize,
) -> CapacityResult {
    let mut mem = EpisodicMemory::new("capacity", dim, beta);
    let procs = generate_synthetic_procedures(n_procedures, seed, 3, 8, 0.15);
    for p in &procs {
        mem.add_record(p.clone());
    }
    let pairs = mem.write_pairs();
    let n_associations = pairs.len();
    mem.matrix.rebuild(&pairs, passes, beta);

    let mut found = 0usize;
    let mut found_exact = 0usize;
    let mut times: Vec<f64> = Vec::new();
    for p in &procs {
        let t0 = std::time::Instant::now();
        let (rec, _) = mem.find(&p.goal, "matrix", 0.0);
        times.push(t0.elapsed().as_secs_f64());
        if rec.as_deref() == Some(p.id.as_str()) {
            found += 1;
        }
        let (rec_e, _) = mem.find(&p.goal, "exact", 0.0);
        if rec_e.as_deref() == Some(p.id.as_str()) {
            found_exact += 1;
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = 1000.0 * times[times.len() / 2];

    let mut n_steps = 0usize;
    let mut n_ok = 0usize;
    for p in &procs {
        for i in 0..p.steps.len().saturating_sub(1) {
            let (best, _) = mem.next_step_via_matrix(&p.id, i);
            n_steps += 1;
            if best.as_deref() == Some(p.steps[i + 1].text()) {
                n_ok += 1;
            }
        }
    }
    CapacityResult {
        n_procedures,
        n_associations,
        found_matrix: found as f64 / n_procedures as f64,
        found_exact: found_exact as f64 / n_procedures as f64,
        next_step: n_ok as f64 / n_steps.max(1) as f64,
        median_lookup_ms: median_ms,
    }
}

// ---------------------------------------------------------------- simulations
/// Hidden-criterion policy outcome (paper Sec. 6, Plan 4).
#[derive(Clone, Debug)]
pub struct PolicyOutcome {
    pub reported_done: f64,
    pub solved: f64,
    pub attempts_per_task: f64,
    pub damaging: usize,
}

#[derive(Clone, Debug)]
pub struct HiddenCriterionResult {
    pub no_memory: PolicyOutcome,
    pub self_judged_stops: PolicyOutcome,
    pub verified: PolicyOutcome,
    pub ceiling: f64,
    pub n_tasks: usize,
    pub n_situations: usize,
}

/// Paper Sec. 6, Plan 4: hidden arrival-speed criterion.
///
/// Mirrors the Python implementation exactly: ONE rng stream -- situations
/// and tasks are drawn first, then the three policies replay in order
/// (no_memory, self_judged_stops, verified), each drawing its attempts
/// from the continuing stream. The agent's own claim knows only "goal
/// reached"; the external check measures arrival speed.
pub fn simulate_hidden_criterion(
    seed: u64, n_tasks: usize, n_situations: usize, cap: usize,
) -> HiddenCriterionResult {
    let mut rng = Rng::new(seed);
    let speeds = [0.7f64, 0.9, 1.1, 1.3];
    let reach_p = [0.75f64, 0.85, 0.92, 0.97];
    let order = [3usize, 2, 1, 0];
    let mut situations: Vec<(f64, bool)> = Vec::new();
    for _ in 0..n_situations {
        let limit = 0.6 + rng.uniform() * 1.0;
        let reachable = rng.bernoulli(0.95);
        situations.push((limit, reachable));
    }
    let tasks: Vec<usize> = (0..n_tasks)
        .map(|_| rng.randint(0, n_situations as i64 - 1) as usize)
        .collect();
    let ceiling = tasks
        .iter()
        .filter(|sit| {
            let (lim, ok) = situations[**sit];
            ok && speeds.iter().any(|s| *s <= lim)
        })
        .count() as f64
        / n_tasks as f64;

    let mut run = |policy: &str| -> PolicyOutcome {
        let mut verified: HashMap<usize, usize> = HashMap::new();
        let (mut reported, mut solved, mut attempts, mut damaging) =
            (0usize, 0usize, 0usize, 0usize);
        for sit in &tasks {
            let sit = *sit;
            let methods: Vec<usize> = if policy == "verified" {
                if let Some(m) = verified.get(&sit) {
                    let mut v = vec![*m];
                    v.extend(order.iter().filter(|x| **x != *m));
                    v
                } else {
                    order.to_vec()
                }
            } else {
                order.to_vec()
            };
            for a in 0..cap {
                let m = methods[a % methods.len()];
                attempts += 1;
                let (limit, reachable) = situations[sit];
                let reached = reachable && rng.bernoulli(reach_p[m]);
                let safe = speeds[m] <= limit;
                let is_solved = reached && safe;
                if reached && !safe {
                    damaging += 1;
                }
                if policy == "self_judged_stops" {
                    if reached {
                        reported += 1;
                        if is_solved {
                            solved += 1;
                        }
                        break;
                    }
                } else if is_solved {
                    solved += 1;
                    reported += 1;
                    if policy == "verified" {
                        verified.insert(sit, m);
                    }
                    break;
                }
            }
        }
        PolicyOutcome {
            reported_done: reported as f64 / n_tasks as f64,
            solved: solved as f64 / n_tasks as f64,
            attempts_per_task: attempts as f64 / n_tasks as f64,
            damaging,
        }
    };
    let no_memory = run("no_memory");
    let self_judged_stops = run("self_judged_stops");
    let verified = run("verified");
    HiddenCriterionResult {
        no_memory,
        self_judged_stops,
        verified,
        ceiling,
        n_tasks,
        n_situations,
    }
}

/// Revisit-stream result (paper Sec. 6).
#[derive(Clone, Debug)]
pub struct RevisitResult {
    pub n_situations: usize,
    pub n_tasks: usize,
    pub solved_situations: usize,
    pub revisit_first_attempt: f64,
    pub revisits: usize,
    pub first_encounter_attempts: usize,
}

/// Paper Sec. 6: a stream of 4x tasks over N situations; one of the
/// methods fits each situation; only simulator-confirmed placements are
/// written. Reach is deterministic per (situation, method).
pub fn simulate_revisit_stream(
    seed: u64, n_situations: usize, repeats: usize, n_methods: usize,
) -> RevisitResult {
    let mut rng = Rng::new(seed);
    let mut correct: Vec<usize> = Vec::new();
    let mut reach: Vec<Vec<bool>> = Vec::new();
    for _ in 0..n_situations {
        let c = rng.randint(0, n_methods as i64 - 1) as usize;
        correct.push(c);
        let mut row = Vec::new();
        for m in 0..n_methods {
            row.push(m == c || rng.bernoulli(0.5));
        }
        reach.push(row);
    }
    let mut solved_once: HashMap<usize, usize> = HashMap::new();
    let (mut revisit_first, mut revisit_total) = (0usize, 0usize);
    let mut first_attempt_total = 0usize;
    for _ in 0..repeats {
        for sit in 0..n_situations {
            if let Some(m) = solved_once.get(&sit) {
                revisit_total += 1;
                if reach[sit][*m] {
                    revisit_first += 1;
                }
            } else {
                for m in 0..n_methods {
                    first_attempt_total += 1;
                    if reach[sit][m] && m == correct[sit] {
                        solved_once.insert(sit, m);
                        break;
                    }
                }
            }
        }
    }
    RevisitResult {
        n_situations,
        n_tasks: n_situations * repeats,
        solved_situations: solved_once.len(),
        revisit_first_attempt: revisit_first as f64 / revisit_total.max(1) as f64,
        revisits: revisit_total,
        first_encounter_attempts: first_attempt_total,
    }
}

/// Demotion-replay result (paper Sec. 6).
#[derive(Clone, Debug)]
pub struct DemotionResult {
    pub n_tasks: usize,
    pub n_stations: usize,
    pub solved_without_rule: f64,
    pub solved_with_rule: f64,
    pub trips_without_rule: usize,
    pub trips_with_rule: usize,
    pub demotions: usize,
    pub first_trip_task: Option<usize>,
}

/// Paper Sec. 6: default-demotion replay on the industrial arm.
pub fn simulate_demotion(
    seed: u64, n_tasks: usize, n_stations: usize, trip_station: usize,
    predecessor_solve_p: f64,
) -> DemotionResult {
    let mut rng = Rng::new(seed);
    let stations: Vec<usize> = (0..n_tasks)
        .map(|_| rng.randint(0, n_stations as i64 - 1) as usize)
        .collect();
    let pred_solves: Vec<bool> =
        (0..n_tasks).map(|_| rng.bernoulli(predecessor_solve_p)).collect();
    let run = |with_rule: bool| -> (f64, usize, usize, Option<usize>) {
        let mut default_bad = true;
        let (mut solved, mut trips, mut demotions) = (0usize, 0usize, 0usize);
        let mut first_trip: Option<usize> = None;
        for (t, st) in stations.iter().enumerate() {
            if default_bad {
                if *st == trip_station {
                    trips += 1;
                    if first_trip.is_none() {
                        first_trip = Some(t);
                    }
                    if with_rule {
                        default_bad = false;
                        demotions += 1;
                    }
                } else {
                    solved += pred_solves[t] as usize;
                }
            } else {
                solved += pred_solves[t] as usize;
            }
        }
        (solved as f64 / n_tasks as f64, trips, demotions, first_trip)
    };
    let (sw, tw, _, _) = run(false);
    let (sr, tr, dem, ft) = run(true);
    DemotionResult {
        n_tasks,
        n_stations,
        solved_without_rule: sw,
        solved_with_rule: sr,
        trips_without_rule: tw,
        trips_with_rule: tr,
        demotions: dem,
        first_trip_task: ft,
    }
}
