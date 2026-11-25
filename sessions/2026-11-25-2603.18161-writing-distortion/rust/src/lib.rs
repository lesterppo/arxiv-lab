//! Rust port of `writing_distortion_metrics.py` (arXiv:2603.18161v2,
//! "How LLMs Distort Our Written Language").
//!
//! Pure, deterministic CPU behaviour only. Model-backed / embedding
//! parts of the session (Qwen generation, judge embeddings) are OUT
//! of scope; see `README.md`.
//!
//! Mirrored exactly from the Python module:
//! - [`unigram_jsd`]: log base 2, add-alpha smoothing with
//!   `alpha = 1e-6` per vocabulary type, tokenization = lowercase
//!   alphanumeric words (`[a-z0-9]+`).
//! - [`pronoun_rate`] / [`adjective_rate`]: per 100 words, same
//!   wordlists / suffixes / lexicon (copied verbatim).
//! - [`keyword_stance`]: same PRO/ANTI wordlists (copied verbatim),
//!   whole-word/phrase counting, case-insensitive, labels at
//!   score >= 3 ("pro") / <= -3 ("anti") / else "mixed".
//! - [`pct_change`]: same zero guards (`0/0 -> 0.0`, `x/0 -> +inf`).
//! - [`mean_shift_alignment`]: shift vectors, L2-normalize, mean
//!   direction, mean cosine similarity; same error cases as Python
//!   (`ValueError` there, `Err(String)` here).
//!
//! Std only, no external crates.

use std::collections::BTreeMap;

pub mod mini_json;

/// Lowercase alphanumeric words of `text` (list of words).
///
/// Python: `re.compile(r"[a-z0-9]+").findall(text.lower())`.
/// Rust lowercases with Unicode `to_lowercase` (same as Python's
/// `str.lower()` for the inputs this module sees) and then keeps
/// runs of ASCII `[a-z0-9]`, exactly like the Python regex.
pub fn tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut words = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_ascii_alphanumeric() {
            cur.push(c);
        } else if !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

pub const PRONOUNS: &[&str] = &[
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "this", "that", "these",
    "those", "who", "whom", "whose", "myself", "ourselves",
    "yourself", "yourselves", "himself", "herself", "itself",
    "themselves", "someone", "anyone", "everyone", "nobody",
    "somebody", "everybody",
];

pub const ADJECTIVE_SUFFIXES: &[&str] = &[
    "ous", "ive", "ful", "less", "able", "ible", "al", "ic",
    "ary", "ory", "ant", "ent",
];

/// Small hand-picked lexicon of common adjectives not caught by the
/// suffix list. This is a PROXY, NOT LIWC. Copied verbatim from the
/// Python module (which lists "safe" twice in its set literal; a set
/// dedupes it, so it appears once here).
pub const ADJECTIVE_LEXICON: &[&str] = &[
    "safe", "dangerous", "good", "bad", "big", "small", "new", "old",
    "important", "different", "public", "personal", "human",
    "electric", "automatic", "future", "modern", "major", "main",
    "high", "low", "large", "great", "best", "worst", "easy", "hard",
    "fast", "slow", "strong", "weak", "free", "open", "full", "real",
    "true", "clear", "common", "general", "local", "national",
    "social", "economic", "political", "legal", "medical", "physical",
    "mental", "daily", "weekly", "annual", "central", "rural",
    "urban", "global", "severe", "serious", "significant",
    "potential", "possible", "likely", "certain", "ready", "aware",
    "careful", "happy", "sad", "young", "elderly", "disabled",
    "remote", "huge", "tiny", "heavy", "light", "dark", "bright",
    "quiet", "busy", "rich", "poor", "cheap", "costly",
];

/// Keyword stance wordlists -- copied verbatim from the Python
/// module / session spec; do not alter.
pub const PRO_WORDS: &[&str] = &[
    "safer", "safety", "reduce accidents", "fewer accidents",
    "reduce crashes", "save lives", "saves lives", "efficient",
    "improve traffic", "reduce traffic", "cost saving", "save time",
    "save money", "greater access", "convenient", "reliable",
    "reduce congestion", "less pollution", "benefit", "benefits",
    "improve", "improves",
];

pub const ANTI_WORDS: &[&str] = &[
    "dangerous", "kill", "killed", "death", "deaths", "accident",
    "accidents", "crash", "crashes", "risk", "risks", "risky",
    "unsafe", "unethical", "job loss", "jobs", "unemployment",
    "expensive", "hack", "hacking", "privacy", "liability",
    "unpredictable", "fail", "failure", "malfunction", "error",
    "errors",
];

/// Jensen-Shannon divergence of unigram distributions, log base 2.
///
/// Range 0..1; 0 for identical texts. Add-alpha smoothing
/// (alpha = 1e-6 per vocabulary type). Deterministic: the vocabulary
/// is summed in sorted order (see README divergence note D1: Python
/// sums over a set, whose order is hash-dependent; results agree to
/// within floating-point error, verified by the parity sweep).
pub fn unigram_jsd(text_a: &str, text_b: &str) -> f64 {
    let wa = tokenize(text_a);
    let wb = tokenize(text_b);
    let mut ca: BTreeMap<&str, f64> = BTreeMap::new();
    let mut cb: BTreeMap<&str, f64> = BTreeMap::new();
    for w in &wa {
        *ca.entry(w.as_str()).or_insert(0.0) += 1.0;
    }
    for w in &wb {
        *cb.entry(w.as_str()).or_insert(0.0) += 1.0;
    }
    // Vocabulary union, sorted (BTreeMap key order).
    let mut vocab: Vec<&str> = ca.keys().copied().collect();
    for k in cb.keys() {
        if !ca.contains_key(k) {
            vocab.push(k);
        }
    }
    vocab.sort_unstable();
    if vocab.is_empty() {
        return 0.0;
    }
    let alpha = 1e-6_f64;
    let na = wa.len() as f64 + alpha * vocab.len() as f64;
    let nb = wb.len() as f64 + alpha * vocab.len() as f64;
    let mut jsd = 0.0_f64;
    for w in vocab {
        let p = (ca.get(w).copied().unwrap_or(0.0) + alpha) / na;
        let q = (cb.get(w).copied().unwrap_or(0.0) + alpha) / nb;
        let m = 0.5 * (p + q);
        jsd += 0.5 * p * (p / m).log2() + 0.5 * q * (q / m).log2();
    }
    // Clamp tiny floating point excursions.
    jsd.clamp(0.0, 1.0)
}

/// Pronouns per 100 words. Returns (rate, count, n_words).
pub fn pronoun_rate(text: &str) -> (f64, usize, usize) {
    let words = tokenize(text);
    let n = words.len();
    let count = words.iter().filter(|w| PRONOUNS.contains(&w.as_str())).count();
    let rate = if n > 0 { 100.0 * count as f64 / n as f64 } else { 0.0 };
    (rate, count, n)
}

/// Adjective PROXY per 100 words. Returns (rate, count, n_words).
///
/// A word counts if it has length >= 5 and ends in one of
/// [`ADJECTIVE_SUFFIXES`], or if it is in [`ADJECTIVE_LEXICON`].
/// Words are counted at most once each (suffix match OR lexicon hit).
/// This is a suffix+lexicon proxy, NOT LIWC.
pub fn adjective_rate(text: &str) -> (f64, usize, usize) {
    let words = tokenize(text);
    let n = words.len();
    let mut count = 0_usize;
    for w in &words {
        if w.len() >= 5 && ADJECTIVE_SUFFIXES.iter().any(|s| w.ends_with(s)) {
            count += 1;
        } else if ADJECTIVE_LEXICON.contains(&w.as_str()) {
            count += 1;
        }
    }
    let rate = if n > 0 { 100.0 * count as f64 / n as f64 } else { 0.0 };
    (rate, count, n)
}

/// A character is a Python-regex "word character" for `\b` purposes:
/// alphanumeric (Unicode) or underscore.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Whole-word/phrase occurrences of `phrase` in lowercased text.
///
/// Mirrors Python `len(re.findall(r"\b" + re.escape(phrase) + r"\b",
/// text_lower))`: non-overlapping left-to-right matches, with a word
/// boundary required on both ends of the phrase. All phrases in the
/// PRO/ANTI lists start and end with an alphanumeric character, so
/// only the end boundaries matter (see README divergence note D3 for
/// the Unicode `\w` approximation).
fn count_phrase(text_lower: &str, phrase: &str) -> usize {
    let hay: Vec<char> = text_lower.chars().collect();
    let needle: Vec<char> = phrase.chars().collect();
    if needle.is_empty() || hay.len() < needle.len() {
        return 0;
    }
    let mut count = 0_usize;
    let mut i = 0_usize;
    while i + needle.len() <= hay.len() {
        if hay[i..i + needle.len()] == needle[..] {
            let before_ok = i == 0 || !is_word_char(hay[i - 1]);
            let after_ok = i + needle.len() == hay.len()
                || !is_word_char(hay[i + needle.len()]);
            if before_ok && after_ok {
                count += 1;
                i += needle.len();
                continue;
            }
        }
        i += 1;
    }
    count
}

/// Keyword stance on self-driving cars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stance {
    pub pro: usize,
    pub anti: usize,
    pub score: i64,
    pub label: &'static str,
}

pub fn keyword_stance(text: &str) -> Stance {
    let lowered = text.to_lowercase();
    let pro: usize = PRO_WORDS.iter().map(|p| count_phrase(&lowered, p)).sum();
    let anti: usize = ANTI_WORDS.iter().map(|a| count_phrase(&lowered, a)).sum();
    let score = pro as i64 - anti as i64;
    let label = if score >= 3 {
        "pro"
    } else if score <= -3 {
        "anti"
    } else {
        "mixed"
    };
    Stance { pro, anti, score, label }
}

/// `(after - before) / before * 100`; guards `before == 0`.
///
/// Returns 0.0 when both are 0, +inf when before is 0 and after is
/// nonzero (positive infinity regardless of the sign of `after`,
/// exactly as in Python).
pub fn pct_change(before: f64, after: f64) -> f64 {
    if before == 0.0 {
        if after == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (after - before) / before * 100.0
    }
}

fn normalize(vec: &[f64]) -> Option<Vec<f64>> {
    let norm = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        None
    } else {
        Some(vec.iter().map(|x| x / norm).collect())
    }
}

/// Homogenization measure over shift vectors, in [-1, 1].
///
/// Shifts are `b - a` per pair, L2-normalized; the mean direction is
/// the (normalized) mean of the normalized shifts. Returns the mean
/// cosine similarity of each normalized shift to the mean direction.
/// Identical shifts -> ~1.0; opposing shifts cancel the mean
/// direction -> 0.0.
///
/// Errors (Python raises `ValueError` for the same cases):
/// - `embeddings_a` and `embeddings_b` have unequal length
/// - either list is empty
/// - a paired pair has unequal dimension
/// - a shift vector is zero (cannot be normalized)
pub fn mean_shift_alignment(
    embeddings_a: &[Vec<f64>],
    embeddings_b: &[Vec<f64>],
) -> Result<f64, String> {
    if embeddings_a.len() != embeddings_b.len() {
        return Err(
            "embeddings_a and embeddings_b must have equal length".to_string(),
        );
    }
    if embeddings_a.is_empty() {
        return Err("empty embedding lists".to_string());
    }
    let mut unit: Vec<Vec<f64>> = Vec::with_capacity(embeddings_a.len());
    for (a, b) in embeddings_a.iter().zip(embeddings_b.iter()) {
        if a.len() != b.len() {
            return Err("paired vectors must have equal length".to_string());
        }
        let shift: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| y - x).collect();
        match normalize(&shift) {
            Some(u) => unit.push(u),
            None => return Err("zero shift vector cannot be normalized".to_string()),
        }
    }
    let dim = unit[0].len();
    let n = unit.len() as f64;
    let mut mean_dir: Vec<f64> = (0..dim)
        .map(|d| unit.iter().map(|u| u[d]).sum::<f64>() / n)
        .collect();
    mean_dir = match normalize(&mean_dir) {
        Some(m) => m,
        None => return Ok(0.0),
    };
    let cosines: Vec<f64> = unit
        .iter()
        .map(|u| u.iter().zip(mean_dir.iter()).map(|(x, y)| x * y).sum::<f64>())
        .collect();
    let mean_cos = cosines.iter().sum::<f64>() / cosines.len() as f64;
    Ok(mean_cos.clamp(-1.0, 1.0))
}
