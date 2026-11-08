//! CORE mechanism port (arXiv 2611.07859): BM25 coarse retriever,
//! TF-IDF-cosine snippet extractor, Jaccard-based context manager, and
//! whitespace token counter. Faithful port of `core.py` in the parent
//! session directory; see `rust/README.md` for the divergence list.
//!
//! Stdlib only. All floating point is f64, matching Python's float64.

use std::collections::{HashMap, HashSet};

/// Stopwords removed for retrieval / snippet scoring — identical set to
/// `core.py`'s `STOPWORDS`.
pub const STOPWORDS: &[&str] = &[
    "which",
    "is",
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "who",
    "with",
    "by",
    "that",
    "it",
    "as",
    "are",
    "was",
    "were",
    "be",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "and",
    "or",
    "but",
    "from",
    "into",
    "their",
    "they",
    "this",
    "these",
    "those",
    "affiliated",
];

fn is_stopword(t: &str) -> bool {
    STOPWORDS.contains(&t)
}

/// Lowercase alphanumeric tokens. Mirrors `re.findall(r"[a-z0-9]+",
/// text.lower())`: a run of `[a-z0-9]` after lowercasing; any other
/// character breaks a token. Non-ASCII letters are not `[a-z0-9]` and
/// break tokens, exactly as in Python (see README divergence note).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let c = ch.to_ascii_lowercase();
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

/// Query/content tokens with stopwords removed.
pub fn content_tokens(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| !is_stopword(t))
        .collect()
}

/// Whitespace token count (the paper's cost metric, approximated).
pub fn count_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

// ---------------------------------------------------------------- BM25

/// Classic BM25 coarse retriever (CPU proxy for the paper's dense
/// retriever). `k1 = 1.5`, `b = 0.75`, `idf = ln(1 + (N - df + 0.5) /
/// (df + 0.5))`, exactly as in `core.py`.
///
/// Internal optimisation only: per-document term frequencies are
/// precomputed at construction instead of rebuilt on every `score` call
/// (as `core.py` does). The values are identical.
pub struct Bm25 {
    pub passages: Vec<String>,
    docs: Vec<Vec<String>>,
    doc_tfs: Vec<HashMap<String, usize>>,
    k1: f64,
    b: f64,
    n: usize,
    avgdl: f64,
    idf: HashMap<String, f64>,
}

impl Bm25 {
    pub fn new(passages: Vec<String>, k1: f64, b: f64) -> Self {
        let docs: Vec<Vec<String>> = passages.iter().map(|p| content_tokens(p)).collect();
        let n = docs.len();
        let avgdl = docs.iter().map(|d| d.len()).sum::<usize>() as f64 / n.max(1) as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        for d in &docs {
            for t in d.iter().collect::<HashSet<_>>() {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
        }
        let idf: HashMap<String, f64> = df
            .iter()
            .map(|(t, &c)| {
                (
                    t.clone(),
                    (1.0 + (n as f64 - c as f64 + 0.5) / (c as f64 + 0.5)).ln(),
                )
            })
            .collect();
        let doc_tfs: Vec<HashMap<String, usize>> = docs
            .iter()
            .map(|d| {
                let mut tf = HashMap::new();
                for t in d {
                    *tf.entry(t.clone()).or_insert(0) += 1;
                }
                tf
            })
            .collect();
        Self {
            passages,
            docs,
            doc_tfs,
            k1,
            b,
            n,
            avgdl,
            idf,
        }
    }

    pub fn score(&self, query_tokens: &[String], doc_idx: usize) -> f64 {
        let doc = &self.docs[doc_idx];
        if doc.is_empty() {
            return 0.0;
        }
        let tf = &self.doc_tfs[doc_idx];
        let dl = doc.len() as f64;
        let mut seen: HashSet<&str> = HashSet::new();
        let mut s = 0.0;
        // Python iterates set(query_tokens) in hash order; Rust iterates
        // query order deduplicated. The summed terms are identical; only
        // the float summation order differs (last-ulp level, see README).
        for t in query_tokens {
            if !seen.insert(t.as_str()) {
                continue;
            }
            let Some(&f) = tf.get(t.as_str()) else {
                continue;
            };
            let f = f as f64;
            let idf = self.idf.get(t.as_str()).copied().unwrap_or(0.0);
            s += idf * f * (self.k1 + 1.0)
                / (f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl));
        }
        s
    }

    /// Return `[(doc_idx, score)]` sorted by score desc, ties broken by
    /// lower doc index — same as `core.py`'s `sort(key=(-score, idx))`.
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f64)> {
        let qt = content_tokens(query);
        let mut scored: Vec<(usize, f64)> = (0..self.n).map(|i| (i, self.score(&qt, i))).collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }
}

// ------------------------------------------------------- snippet extractor

/// Token-similarity snippet extraction (CPU proxy for the late-interaction
/// reranker's token similarities): TF-IDF cosine between the query and
/// each sentence window of a passage; return the best window.
pub struct SnippetExtractor {
    vocab: HashMap<String, usize>,
    idf: Vec<f64>,
}

impl SnippetExtractor {
    pub fn new(all_sentences: &[String]) -> Self {
        let sent_token_lists: Vec<Vec<String>> =
            all_sentences.iter().map(|s| content_tokens(s)).collect();
        let mut vocab: HashMap<String, usize> = HashMap::new();
        for toks in &sent_token_lists {
            for t in toks {
                if !vocab.contains_key(t) {
                    vocab.insert(t.clone(), vocab.len());
                }
            }
        }
        let n = sent_token_lists.len() as f64;
        let mut df = vec![0.0f64; vocab.len()];
        for toks in &sent_token_lists {
            let mut seen: HashSet<&str> = HashSet::new();
            for t in toks {
                if seen.insert(t.as_str()) {
                    df[vocab[t.as_str()]] += 1.0;
                }
            }
        }
        // BM25-style positive idf, same formula as core.py.
        let idf: Vec<f64> = df
            .iter()
            .map(|&c| (1.0 + (n - c + 0.5) / (c + 0.5)).ln())
            .collect();
        Self { vocab, idf }
    }

    fn vector(&self, tokens: &[String]) -> Vec<f64> {
        let mut v = vec![0.0f64; self.vocab.len()];
        for t in tokens {
            if let Some(&idx) = self.vocab.get(t.as_str()) {
                v[idx] += 1.0;
            }
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x *= self.idf[i];
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }

    /// Cosine similarity between the query and one sentence (width-1
    /// window). Exposed for the property tests.
    pub fn sentence_cosine(&self, sentence: &str, query: &str) -> f64 {
        let qv = self.vector(&content_tokens(query));
        let wv = self.vector(&content_tokens(sentence));
        qv.iter().zip(wv.iter()).map(|(a, b)| a * b).sum()
    }

    /// Best consecutive `width`-sentence window by TF-IDF cosine.
    /// Returns (indices, text). Strict `>` keeps the first window on
    /// ties, as in `core.py`; initial best score is -1.0.
    pub fn best_window(
        &self,
        sentences: &[String],
        query: &str,
        width: usize,
    ) -> (Vec<usize>, String) {
        let qt = content_tokens(query);
        let qv = self.vector(&qt);
        let mut best_idx: Vec<usize> = Vec::new();
        let mut best_score = -1.0f64;
        // range(0, max(1, len(sentences) - width + 1))
        let n_starts = sentences.len().saturating_sub(width - 1).max(1);
        for start in 0..n_starts {
            let end = (start + width).min(sentences.len());
            let idxs: Vec<usize> = (start..end).collect();
            let mut toks = Vec::new();
            for &i in &idxs {
                toks.extend(content_tokens(&sentences[i]));
            }
            let wv = self.vector(&toks);
            let score: f64 = qv.iter().zip(wv.iter()).map(|(a, b)| a * b).sum();
            if score > best_score {
                best_score = score;
                best_idx = idxs;
            }
        }
        let text = best_idx
            .iter()
            .map(|&i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        (best_idx, text)
    }

    /// Ablation: first `width` sentences.
    pub fn first_window(&self, sentences: &[String], width: usize) -> (Vec<usize>, String) {
        let idxs: Vec<usize> = (0..width.min(sentences.len())).collect();
        let text = idxs
            .iter()
            .map(|&i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        (idxs, text)
    }
}

pub fn window_text(sentences: &[String], indices: &[usize]) -> String {
    indices
        .iter()
        .map(|&i| sentences[i].as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Jaccard overlap between two token sets. Empty/empty returns 1.0.
pub fn jaccard(a_tokens: &[String], b_tokens: &[String]) -> f64 {
    let a: HashSet<&str> = a_tokens.iter().map(|s| s.as_str()).collect();
    let b: HashSet<&str> = b_tokens.iter().map(|s| s.as_str()).collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(&b).count() as f64;
    let union = a.union(&b).count() as f64;
    inter / union
}

// ---------------------------------------------------------- context manager

/// Turn-2 action for a passage already seen in the trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmAction {
    Fresh,
    Suppress,
    Expand,
}

impl CmAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            CmAction::Fresh => "fresh",
            CmAction::Suppress => "suppress",
            CmAction::Expand => "expand",
        }
    }
}

/// Cross-turn context management (paper stage 3). Tracks snippets already
/// returned per passage. `seen` keeps insertion order (a `Vec`, matching
/// Python's dict) so `is_duplicate` scans in the same order.
pub struct ContextManager {
    threshold: f64,
    seen: Vec<(String, String)>,
}

impl ContextManager {
    pub fn new(jaccard_threshold: f64) -> Self {
        Self {
            threshold: jaccard_threshold,
            seen: Vec::new(),
        }
    }

    pub fn register(&mut self, passage_id: &str, snippet_text: &str) {
        // Python's dict assignment replaces the value for an existing key;
        // keep the same semantics (latest value wins).
        if let Some(slot) = self.seen.iter_mut().find(|(id, _)| id == passage_id) {
            slot.1 = snippet_text.to_string();
        } else {
            self.seen
                .push((passage_id.to_string(), snippet_text.to_string()));
        }
    }

    pub fn was_seen(&self, passage_id: &str) -> bool {
        self.seen.iter().any(|(id, _)| id == passage_id)
    }

    pub fn is_duplicate(&self, snippet_text: &str) -> bool {
        let toks = content_tokens(snippet_text);
        for (_, prev) in &self.seen {
            if jaccard(&toks, &content_tokens(prev)) >= self.threshold {
                return true;
            }
        }
        false
    }

    /// Return (action, text). Mirrors `core.py::turn2_decision` exactly,
    /// including the substring check `answer_sentence in prev` on the
    /// previously returned snippet text.
    pub fn turn2_decision(
        &self,
        passage_id: &str,
        fresh_text: &str,
        answer_sentence: &str,
    ) -> (CmAction, String) {
        let Some((_, prev)) = self.seen.iter().find(|(id, _)| id == passage_id) else {
            return (CmAction::Fresh, fresh_text.to_string());
        };
        if jaccard(&content_tokens(fresh_text), &content_tokens(prev)) >= self.threshold {
            return (CmAction::Suppress, String::new());
        }
        if prev.contains(answer_sentence) {
            return (CmAction::Suppress, String::new());
        }
        (CmAction::Expand, answer_sentence.to_string())
    }
}
