//! GrepMem: lossless, index-free memory for long-horizon LLM agents.
//!
//! Rust port of `arxiv_lab.agent.grepmem` (arXiv:2611.25123 — Yang, Shi,
//! Zhao, 2026, "Forget Freely, Grep Later"). An append-only plain-text
//! log of the agent's complete history plus a regular-expression search
//! tool for on-demand recall, compared against lossy memory policies
//! (summarization, recency truncation, random masking) with a perfect
//! keyword QA oracle, so accuracy differences measure information
//! preservation only.
//!
//! The port is behaviorally identical to the Python module, including
//! the random streams: [`PyRandom`] is a faithful re-implementation of
//! CPython's `random.Random` (MT19937 + `init_by_array` seeding,
//! `random()`, `getrandbits`, `_randbelow`, `choice`, `shuffle`,
//! `sample`), so `generate_session` produces byte-identical sessions
//! to Python for the same seed and parameters. Verified against
//! CPython 3.12 values in `tests/unit.rs`.
//!
//! The only external dependency is the `regex` crate (same search
//! semantics as Python `re` for the patterns used here: literals,
//! `\b`, alternation, groups, `(.+)$`, case-insensitive flag).

use regex::Regex;
use std::collections::{HashMap, HashSet};

// ===========================================================================
// CPython-compatible RNG (MT19937, random.Random for integer seeds)
// ===========================================================================

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

/// A faithful port of CPython's `random.Random` restricted to the
/// operations this mechanism uses. Seeding: an integer seed is split
/// into little-endian 32-bit words and fed to MT19937 `init_by_array`
/// — exactly CPython's version-2 seeding for `int` (verified:
/// `random()` streams match `random.Random(seed)` for seeds 0..5 and
/// `getrandbits` matches for k = 3, 6, 32, 33, 40, 64).
pub struct PyRandom {
    mt: [u32; MT_N],
    idx: usize,
}

impl PyRandom {
    pub fn new(seed: u64) -> Self {
        let mut key = vec![(seed & 0xffff_ffff) as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        let mut r = Self {
            mt: [0; MT_N],
            idx: MT_N,
        };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..MT_N {
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.idx = MT_N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        let k = MT_N.max(key.len());
        for _ in 0..k {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                    .wrapping_mul(1_664_525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                self.mt[0] = self.mt[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..(MT_N - 1) {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                    .wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                self.mt[0] = self.mt[MT_N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
        self.idx = MT_N;
    }

    fn twist(&mut self) {
        for i in 0..MT_N {
            let x = (self.mt[i] & MT_UPPER_MASK)
                | (self.mt[(i + 1) % MT_N] & MT_LOWER_MASK);
            let mut xa = x >> 1;
            if x & 1 != 0 {
                xa ^= MT_MATRIX_A;
            }
            self.mt[i] = self.mt[(i + MT_M) % MT_N] ^ xa;
        }
        self.idx = 0;
    }

    pub fn gen_u32(&mut self) -> u32 {
        if self.idx >= MT_N {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython `random()`: `(a * 2**26 + b) / 2**53` with
    /// `a = genrand() >> 5`, `b = genrand() >> 6`.
    pub fn random(&mut self) -> f64 {
        let a = (self.gen_u32() >> 5) as f64;
        let b = (self.gen_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// CPython `getrandbits(k)`: the low `k` bits of the result are the
    /// top bits of successive 32-bit words, first word lowest. Exact
    /// for 0 < k <= 64.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k <= 64, "getrandbits only supports k <= 64 here");
        if k == 0 {
            return 0;
        }
        let mut result: u64 = 0;
        let mut done = 0u32;
        let mut shift = 0u32;
        while done < k {
            let take = (k - done).min(32);
            let word = self.gen_u32();
            let bits = (word >> (32 - take)) as u64;
            result |= bits << shift;
            shift += take;
            done += take;
        }
        result
    }

    /// CPython `_randbelow(n)`: `getrandbits(n.bit_length())` with
    /// rejection.
    pub fn randbelow(&mut self, n: usize) -> usize {
        assert!(n > 0, "randbelow of empty range");
        let k = 64 - (n as u64).leading_zeros();
        loop {
            let r = self.getrandbits(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// CPython `choice(seq)`.
    pub fn choice<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.randbelow(xs.len())]
    }

    /// CPython `shuffle(x)`: Fisher-Yates from the end.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.randbelow(i + 1);
            xs.swap(i, j);
        }
    }

    /// CPython `random.sample` over the index range `0..n`, returned in
    /// selection order (pool branch for small `n`, selection-set branch
    /// otherwise — same threshold as CPython: `setsize = 21`, plus
    /// `4 ** ceil(log(k*3, 4))` when `k > 5`).
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "sample larger than population");
        let mut setsize = 21usize;
        if k > 5 {
            let logv = ((k * 3) as f64).ln() / 4f64.ln();
            setsize += 4usize.pow(logv.ceil() as u32);
        }
        if n <= setsize {
            let mut pool: Vec<usize> = (0..n).collect();
            let mut result = Vec::with_capacity(k);
            for i in 0..k {
                let j = self.randbelow(n - i);
                result.push(pool[j]);
                pool[j] = pool[n - i - 1];
            }
            result
        } else {
            let mut selected: HashSet<usize> = HashSet::new();
            let mut result = Vec::with_capacity(k);
            for _ in 0..k {
                let mut j = self.randbelow(n);
                while selected.contains(&j) {
                    j = self.randbelow(n);
                }
                selected.insert(j);
                result.push(j);
            }
            result
        }
    }

    /// CPython `random.sample(xs, k)` over a slice, selection order.
    pub fn sample_items<T: Clone>(&mut self, xs: &[T], k: usize) -> Vec<T> {
        self.sample_indices(xs.len(), k)
            .iter()
            .map(|&i| xs[i].clone())
            .collect()
    }
}

// ===========================================================================
// Log
// ===========================================================================

pub const FACT_VERBS: [&str; 6] = ["is", "is in", "is now", "runs on", "uses", "costs"];

/// Append-only plain-text log of an agent's complete history. Nothing
/// is ever deleted or rewritten — the paper's "forget freely" half.
#[derive(Debug, Clone, Default)]
pub struct AppendOnlyLog {
    pub lines: Vec<String>,
}

impl AppendOnlyLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, line: String) {
        self.lines.push(line);
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Regular-expression search over the log (case-insensitive).
    /// Returns matching lines in log order. `max_lines` caps the
    /// returned lines (most recent matches win when capped).
    pub fn grep(&self, pattern: &str, max_lines: Option<usize>) -> Vec<String> {
        let rx = Regex::new(&format!("(?i)(?:{})", pattern))
            .unwrap_or_else(|e| panic!("bad grep pattern {:?}: {}", pattern, e));
        let mut hits: Vec<String> = self
            .lines
            .iter()
            .filter(|ln| rx.is_match(ln))
            .cloned()
            .collect();
        if let Some(cap) = max_lines {
            if hits.len() > cap {
                hits = hits[hits.len() - cap..].to_vec();
            }
        }
        hits
    }
}

// ===========================================================================
// Synthetic research sessions (deterministic task generator)
// ===========================================================================

/// (entity name, kind)
pub const ENTITIES: [(&str, &str); 10] = [
    ("Project Atlas", "project"),
    ("Project Beacon", "project"),
    ("Dr. Chen", "person"),
    ("Dr. Okafor", "person"),
    ("Lab 7", "place"),
    ("Server Node 3", "device"),
    ("Experiment 12", "experiment"),
    ("Experiment 27", "experiment"),
    ("Dataset Vega", "dataset"),
    ("Model Orion", "model"),
];

/// (attribute, verb, values)
pub struct AttrDef {
    pub name: &'static str,
    pub verb: &'static str,
    pub values: &'static [&'static str],
}

pub const ATTRIBUTES: [AttrDef; 6] = [
    AttrDef {
        name: "budget",
        verb: "is",
        values: &["$1.2M", "$2.4M", "$3.1M", "$0.8M", "$4.7M"],
    },
    AttrDef {
        name: "location",
        verb: "is in",
        values: &["Building 7", "Building 2", "Room 14", "Annex B", "Floor 3"],
    },
    AttrDef {
        name: "port",
        verb: "runs on",
        values: &["port 8080", "port 443", "port 9000", "port 5432"],
    },
    AttrDef {
        name: "temperature",
        verb: "is",
        values: &["37.5C", "21.0C", "42.3C", "18.9C"],
    },
    AttrDef {
        name: "status",
        verb: "is",
        values: &["active", "paused", "complete", "pending review"],
    },
    AttrDef {
        name: "lead",
        verb: "is led by",
        values: &["Dr. Chen", "Dr. Okafor", "Dr. Patel", "Dr. Kim"],
    },
];

/// Paraphrase map: attribute -> synonym used in the question text only.
pub const PARAPHRASE: [(&str, &str); 6] = [
    ("budget", "funding"),
    ("location", "whereabouts"),
    ("port", "endpoint"),
    ("temperature", "heat"),
    ("status", "state"),
    ("lead", "supervisor"),
];

pub const STOPWORDS: [&str; 19] = [
    "what", "is", "the", "of", "does", "have", "has", "for", "in", "on", "how", "much", "where",
    "who", "which", "a", "an", "to", "do",
];

pub const DISTRACTOR_TOPICS: [&str; 10] = [
    "team standup moved to 10:00",
    "coffee machine needs descaling",
    "weekly report draft saved",
    "calendar invite accepted for Friday",
    "VPN connection re-established",
    "notebook kernel restarted",
    "package update available for numpy",
    "meeting notes filed under Q3",
    "disk usage at 62 percent",
    "backup completed successfully",
];

pub const AGENT_ACTIONS: [&str; 5] = [
    "draft summary",
    "run analysis",
    "open file",
    "call tool",
    "review output",
];

pub const OBS_KINDS: [&str; 3] = ["OBS tool:web_search", "OBS tool:db_query", "OBS tool:api_call"];

#[derive(Debug, Clone)]
pub struct Question {
    pub text: String,
    pub entity: String,
    pub attr: String,
    pub answer: String,
    /// Step index of the fact in the log; `position = step / (n_lines - 1)`.
    pub step: usize,
    pub position: f64,
    pub paraphrased: bool,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub log: AppendOnlyLog,
    pub questions: Vec<Question>,
    /// Latest value per (entity, attr) pair.
    pub truth: HashMap<(String, String), String>,
}

fn fact_line(step: usize, entity: &str, attr: &str, verb: &str, value: &str, kind: &str) -> String {
    format!("[t{:04}] {} | {} {} {} {}", step, kind, entity, attr, verb, value)
}

fn distractor_line(rng: &mut PyRandom, step: usize) -> String {
    format!(
        "[t{:04}] CHAT | {}",
        step,
        rng.choice(&DISTRACTOR_TOPICS)
    )
}

/// Generate one synthetic long-horizon research session. Same draw
/// order as Python's `generate_session`, so the output is identical
/// for the same seed (same `PyRandom` stream).
pub fn generate_session(seed: u64, n_lines: usize, n_questions: usize) -> Session {
    generate_session_rates(seed, n_lines, n_questions, 0.3, 0.25)
}

pub fn generate_session_rates(
    seed: u64,
    n_lines: usize,
    n_questions: usize,
    update_rate: f64,
    paraphrase_rate: f64,
) -> Session {
    let mut rng = PyRandom::new(seed);
    let mut log = AppendOnlyLog::new();
    // Choose which (entity, attr) pairs carry facts, and their values.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (ei, _) in ENTITIES.iter().enumerate() {
        for (ai, _) in ATTRIBUTES.iter().enumerate() {
            if rng.random() < 0.55 {
                pairs.push((ei, ai));
            }
        }
    }
    rng.shuffle(&mut pairs);
    let mut truth: HashMap<(String, String), String> = HashMap::new();
    let mut fact_steps: Vec<(usize, usize, usize)> = Vec::new(); // (step, entity idx, attr idx)
    let n_facts = n_questions.max(n_lines / 12);
    let mut fact_positions = rng.sample_indices(n_lines, n_facts);
    fact_positions.sort_unstable();
    let mut fi = 0usize;
    for step in 0..n_lines {
        if fi < n_facts && step == fact_positions[fi] {
            fi += 1;
            let &(ei, ai) = rng.choice(&pairs);
            let entity = ENTITIES[ei].0;
            let attr_def = &ATTRIBUTES[ai];
            let key = (entity.to_string(), attr_def.name.to_string());
            let value = if truth.contains_key(&key) && rng.random() > update_rate {
                // Restatement of the current value (no update).
                truth[&key].clone()
            } else {
                rng.choice(attr_def.values).to_string()
            };
            truth.insert(key, value.clone());
            let kind = rng.choice(&OBS_KINDS);
            log.append(fact_line(step, entity, attr_def.name, attr_def.verb, &value, kind));
            fact_steps.push((step, ei, ai));
        } else if rng.random() < 0.45 {
            // Agent action line (non-factual trajectory content).
            let action = rng.choice(&AGENT_ACTIONS);
            log.append(format!("[t{:04}] ACT agent: {}", step, action));
        } else {
            log.append(distractor_line(&mut rng, step));
        }
    }
    // Questions: sample from facts actually in the log.
    let mut questions = Vec::new();
    rng.shuffle(&mut fact_steps);
    let denom = (n_lines.saturating_sub(1)).max(1) as f64;
    for &(step, ei, ai) in fact_steps.iter().take(n_questions) {
        let entity = ENTITIES[ei].0;
        let attr_def = &ATTRIBUTES[ai];
        let paraphrased = rng.random() < paraphrase_rate;
        let attr_word = if paraphrased {
            PARAPHRASE
                .iter()
                .find(|(a, _)| *a == attr_def.name)
                .map(|(_, p)| *p)
                .unwrap_or(attr_def.name)
        } else {
            attr_def.name
        };
        let text = if attr_def.name == "budget" {
            if paraphrased {
                format!("How much {} does {} have?", attr_word, entity)
            } else {
                format!("What is the {} of {}?", attr_def.name, entity)
            }
        } else if attr_def.name == "location" {
            if paraphrased {
                format!("What are the {} of {}?", attr_word, entity)
            } else {
                format!("What is the {} of {}?", attr_def.name, entity)
            }
        } else {
            format!("What is the {} of {}?", attr_word, entity)
        };
        questions.push(Question {
            text,
            entity: entity.to_string(),
            attr: attr_def.name.to_string(),
            answer: truth[&(entity.to_string(), attr_def.name.to_string())].clone(),
            step,
            position: step as f64 / denom,
            paraphrased,
        });
    }
    Session {
        log,
        questions,
        truth,
    }
}

// ===========================================================================
// Memory policies
// ===========================================================================

/// `re.findall(r"[a-z0-9]+", text.lower())` minus stopwords.
pub fn keywords(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        let lc = c.to_ascii_lowercase();
        if lc.is_ascii_alphanumeric() {
            cur.push(lc);
        } else if !cur.is_empty() {
            if !STOPWORDS.contains(&cur.as_str()) {
                words.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if !cur.is_empty() && !STOPWORDS.contains(&cur.as_str()) {
        words.push(cur);
    }
    words
}

fn word_pattern(token: &str) -> String {
    format!(r"\b{}\b", regex::escape(token))
}

pub fn policy_full(log: &AppendOnlyLog, _question: &Question) -> Vec<String> {
    log.lines.clone()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GrepMemOpts {
    pub max_lines: usize,
    pub entity_only: bool,
    pub attr_only: bool,
}

impl GrepMemOpts {
    pub fn new(max_lines: usize) -> Self {
        Self {
            max_lines,
            entity_only: false,
            attr_only: false,
        }
    }
}

/// GrepMem: regex search over the append-only log, on demand. One
/// pattern per keyword group (entity tokens, attribute token); matches
/// are unioned in log order and capped at `max_lines` (most recent
/// matches win). Duplicate lines are de-duplicated by content, as in
/// Python (`if ln not in seen`).
pub fn policy_grepmem(log: &AppendOnlyLog, question: &Question, opts: GrepMemOpts) -> Vec<String> {
    let kws = keywords(&question.text);
    let entity_tokens = keywords(&question.entity);
    let mut patterns: Vec<Regex> = Vec::new();
    if opts.attr_only {
        for t in kws.iter().filter(|w| !entity_tokens.contains(w)) {
            patterns.push(Regex::new(&format!("(?i){}", word_pattern(t))).unwrap());
        }
    } else if opts.entity_only {
        for t in &entity_tokens {
            patterns.push(Regex::new(&format!("(?i){}", word_pattern(t))).unwrap());
        }
    } else {
        for t in &entity_tokens {
            patterns.push(Regex::new(&format!("(?i){}", word_pattern(t))).unwrap());
        }
        for t in kws.iter().filter(|w| !entity_tokens.contains(w)) {
            patterns.push(Regex::new(&format!("(?i){}", word_pattern(t))).unwrap());
        }
    }
    let mut hits: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for ln in &log.lines {
        if patterns.iter().any(|rx| rx.is_match(ln)) && seen.insert(ln.clone()) {
            hits.push(ln.clone());
        }
    }
    if hits.len() > opts.max_lines {
        hits = hits[hits.len() - opts.max_lines..].to_vec();
    }
    hits
}

/// Lossy summarization baseline (SUPO-style). Each block of `block`
/// lines is replaced by its first line plus a seeded `retain` fraction
/// of its fact lines; the most recent block is kept whole.
pub fn policy_summary(
    log: &AppendOnlyLog,
    _question: &Question,
    block: usize,
    retain: f64,
    seed: u64,
) -> Vec<String> {
    let mut rng = PyRandom::new(seed);
    let mut out: Vec<String> = Vec::new();
    let lines = &log.lines;
    let mut start = 0usize;
    while start < lines.len() {
        let chunk = &lines[start..(start + block).min(lines.len())];
        if start + block >= lines.len() {
            out.extend(chunk.iter().cloned());
        } else {
            out.push(chunk[0].clone());
            let facts: Vec<String> = chunk[1..]
                .iter()
                .filter(|ln| ln.contains(" | ") && ln.contains("OBS"))
                .cloned()
                .collect();
            let keep = (facts.len() as f64 * retain) as usize;
            if keep > 0 {
                out.extend(rng.sample_items(&facts, keep));
            }
        }
        start += block;
    }
    out
}

/// Truncation / masking baseline: only the last `window` lines.
pub fn policy_recency(log: &AppendOnlyLog, _question: &Question, window: usize) -> Vec<String> {
    if log.lines.len() <= window {
        log.lines.clone()
    } else {
        log.lines[log.lines.len() - window..].to_vec()
    }
}

/// Random-masking baseline: a seeded `keep` fraction of lines.
pub fn policy_random_mask(
    log: &AppendOnlyLog,
    _question: &Question,
    keep: f64,
    seed: u64,
) -> Vec<String> {
    let mut rng = PyRandom::new(seed);
    log.lines
        .iter()
        .filter(|_| rng.random() < keep)
        .cloned()
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Full,
    GrepMem,
    Summary,
    Recency,
    RandomMask,
}

impl Policy {
    pub fn name(&self) -> &'static str {
        match self {
            Policy::Full => "full",
            Policy::GrepMem => "grepmem",
            Policy::Summary => "summary",
            Policy::Recency => "recency",
            Policy::RandomMask => "random_mask",
        }
    }

    pub fn all() -> [Policy; 5] {
        [
            Policy::Full,
            Policy::GrepMem,
            Policy::Summary,
            Policy::Recency,
            Policy::RandomMask,
        ]
    }
}

/// Apply a policy with the suite's default parameters (grepmem
/// `max_lines` is supplied by the caller for the budget sweep).
pub fn apply_policy(
    policy: Policy,
    log: &AppendOnlyLog,
    question: &Question,
    grepmem_max_lines: usize,
) -> Vec<String> {
    match policy {
        Policy::Full => policy_full(log, question),
        Policy::GrepMem => policy_grepmem(log, question, GrepMemOpts::new(grepmem_max_lines)),
        Policy::Summary => policy_summary(log, question, 25, 0.15, 0),
        Policy::Recency => policy_recency(log, question, 60),
        Policy::RandomMask => policy_random_mask(log, question, 0.20, 0),
    }
}

// ===========================================================================
// Deterministic QA oracle (perfect given the facts in context)
// ===========================================================================

/// Answer from the supplied context lines only. Anchors on the fact
/// structure `| <entity> <attr> <verb> <value>` so an entity mentioned
/// as another fact's *value* (e.g. "lead is led by Dr. Chen") cannot
/// false-match a question about that entity. Latest match wins.
pub fn answer_question(question: &Question, context: &[String]) -> Option<String> {
    let anchor = Regex::new(&format!(
        r"\| {} {} (?:is in|is now|is led by|is|runs on|uses|costs) (.+)$",
        regex::escape(&question.entity),
        regex::escape(&question.attr)
    ))
    .unwrap();
    let mut best: Option<String> = None;
    for ln in context {
        if let Some(m) = anchor.captures(ln) {
            best = Some(m[1].trim().to_string());
        }
    }
    best
}

// ===========================================================================
// Evaluation harness
// ===========================================================================

#[derive(Debug, Clone)]
pub struct QuestionEval {
    pub text: String,
    pub entity: String,
    pub attr: String,
    pub expected: String,
    pub position: f64,
    pub paraphrased: bool,
    pub answer: Option<String>,
    pub correct: bool,
    pub ctx: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EvalResult {
    pub policy: &'static str,
    pub accuracy: f64,
    pub n: usize,
    pub correct: usize,
    pub total_ctx_lines: usize,
    pub mean_context_lines: f64,
    /// (early, mid, late) accuracies.
    pub by_position: [f64; 3],
    pub by_position_n: [usize; 3],
    pub paraphrase_accuracy: f64,
    pub paraphrase_n: usize,
    pub per_question: Vec<QuestionEval>,
}

fn bucket(position: f64) -> usize {
    if position < 0.33 {
        0
    } else if position < 0.67 {
        1
    } else {
        2
    }
}

pub fn evaluate_policy_detailed<F>(policy_name: &'static str, sessions: &[Session], mut run: F) -> EvalResult
where
    F: FnMut(&AppendOnlyLog, &Question) -> Vec<String>,
{
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut total_ctx = 0usize;
    let mut pos_ok = [0usize; 3];
    let mut pos_n = [0usize; 3];
    let mut para_ok = 0usize;
    let mut para_n = 0usize;
    let mut per_question = Vec::new();
    for sess in sessions {
        for q in &sess.questions {
            let ctx = run(&sess.log, q);
            total_ctx += ctx.len();
            let ans = answer_question(q, &ctx);
            let ok = ans.as_deref() == Some(q.answer.as_str());
            correct += ok as usize;
            total += 1;
            let b = bucket(q.position);
            pos_ok[b] += ok as usize;
            pos_n[b] += 1;
            if q.paraphrased {
                para_ok += ok as usize;
                para_n += 1;
            }
            per_question.push(QuestionEval {
                text: q.text.clone(),
                entity: q.entity.clone(),
                attr: q.attr.clone(),
                expected: q.answer.clone(),
                position: q.position,
                paraphrased: q.paraphrased,
                answer: ans,
                correct: ok,
                ctx,
            });
        }
    }
    EvalResult {
        policy: policy_name,
        accuracy: if total > 0 { correct as f64 / total as f64 } else { 0.0 },
        n: total,
        correct,
        total_ctx_lines: total_ctx,
        mean_context_lines: if total > 0 {
            total_ctx as f64 / total as f64
        } else {
            0.0
        },
        by_position: [
            if pos_n[0] > 0 { pos_ok[0] as f64 / pos_n[0] as f64 } else { 0.0 },
            if pos_n[1] > 0 { pos_ok[1] as f64 / pos_n[1] as f64 } else { 0.0 },
            if pos_n[2] > 0 { pos_ok[2] as f64 / pos_n[2] as f64 } else { 0.0 },
        ],
        by_position_n: pos_n,
        paraphrase_accuracy: if para_n > 0 { para_ok as f64 / para_n as f64 } else { 0.0 },
        paraphrase_n: para_n,
        per_question,
    }
}

/// Run one policy over generated sessions; return accuracy + cost.
/// Mirrors Python's `evaluate_policy`.
pub fn evaluate_policy(
    policy: Policy,
    sessions: &[Session],
    grepmem_max_lines: usize,
) -> EvalResult {
    evaluate_policy_detailed(policy.name(), sessions, |log, q| {
        apply_policy(policy, log, q, grepmem_max_lines)
    })
}

pub struct SuiteResult {
    pub n_lines: usize,
    pub n_sessions: usize,
    pub n_questions: usize,
    pub results: Vec<EvalResult>, // in Policy::all() order
}

impl SuiteResult {
    pub fn get(&self, policy: Policy) -> &EvalResult {
        &self.results[policy as usize]
    }
}

/// Full comparison suite over generated sessions (one stream length).
pub fn run_suite(seeds: &[u64], n_lines: usize, n_questions: usize, grepmem_max_lines: usize) -> SuiteResult {
    let sessions: Vec<Session> = seeds
        .iter()
        .map(|&s| generate_session(s, n_lines, n_questions))
        .collect();
    let n_questions_total = sessions.iter().map(|s| s.questions.len()).sum();
    let mut results = Vec::new();
    for policy in Policy::all() {
        results.push(evaluate_policy(policy, &sessions, grepmem_max_lines));
    }
    SuiteResult {
        n_lines,
        n_sessions: sessions.len(),
        n_questions: n_questions_total,
        results,
    }
}

// ===========================================================================
// Session file format (line-based, v1) + JSON output helpers
// ===========================================================================

/// Session file format, one record per line:
/// ```text
/// # grepmem-session-v1 seed=7 n_lines=500 n_questions=20
/// L [t0000] OBS tool:web_search | Project Atlas budget is $1.2M
/// Q Project Atlas\tbudget\t$1.2M\t123\t0\tWhat is the budget of Project Atlas?
/// ```
/// `L` lines are log lines verbatim (they contain no tabs by
/// construction). `Q` fields are tab-separated: entity, attr, answer,
/// fact step (int), paraphrased (0/1), question text. The position is
/// derived on read as `step / (n_lines - 1)`, so no float formatting
/// ever enters the file.
pub fn session_to_text(sess: &Session, seed: u64, n_lines: usize, n_questions: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# grepmem-session-v1 seed={} n_lines={} n_questions={}\n",
        seed, n_lines, n_questions
    ));
    for ln in &sess.log.lines {
        out.push_str("L ");
        out.push_str(ln);
        out.push('\n');
    }
    for q in &sess.questions {
        out.push_str(&format!(
            "Q\t{}\t{}\t{}\t{}\t{}\t{}\n",
            q.entity,
            q.attr,
            q.answer,
            q.step,
            if q.paraphrased { 1 } else { 0 },
            q.text
        ));
    }
    out
}

pub fn session_from_text(text: &str) -> Result<Session, String> {
    let mut log = AppendOnlyLog::new();
    let mut questions = Vec::new();
    let mut truth: HashMap<(String, String), String> = HashMap::new();
    for (lineno, raw) in text.lines().enumerate() {
        if raw.starts_with('#') || raw.is_empty() {
            continue;
        }
        if let Some(rest) = raw.strip_prefix("L ") {
            log.append(rest.to_string());
        } else if let Some(rest) = raw.strip_prefix("Q\t") {
            let parts: Vec<&str> = rest.split('\t').collect();
            if parts.len() != 6 {
                return Err(format!("line {}: bad Q record ({} fields)", lineno + 1, parts.len()));
            }
            let step: usize = parts[3]
                .parse()
                .map_err(|_| format!("line {}: bad step", lineno + 1))?;
            let paraphrased = match parts[4] {
                "1" => true,
                "0" => false,
                _ => return Err(format!("line {}: bad paraphrased flag", lineno + 1)),
            };
            let denom = (log.len().saturating_sub(1)).max(1) as f64;
            // NOTE: position needs the final log length, which is only
            // known after the whole file is read; fix up below.
            questions.push(Question {
                text: parts[5].to_string(),
                entity: parts[0].to_string(),
                attr: parts[1].to_string(),
                answer: parts[2].to_string(),
                step,
                position: step as f64 / denom,
                paraphrased,
            });
            truth.insert((parts[0].to_string(), parts[1].to_string()), parts[2].to_string());
        } else {
            return Err(format!("line {}: unknown record", lineno + 1));
        }
    }
    // Fix up positions with the final log length.
    let denom = (log.len().saturating_sub(1)).max(1) as f64;
    for q in &mut questions {
        q.position = q.step as f64 / denom;
    }
    Ok(Session {
        log,
        questions,
        truth,
    })
}

/// JSON-escape a string (including the surrounding quotes).
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Format an f64 the way Python's `repr`/`json` does for the values we
/// emit (shortest round-trip; integral floats get `.0`).
pub fn fmt_f64(v: f64) -> String {
    if v.is_nan() || v.is_infinite() {
        return "null".to_string();
    }
    if v == v.trunc() && v.abs() < 1e16 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

fn json_str_list(xs: &[String]) -> String {
    let parts: Vec<String> = xs.iter().map(|s| json_escape(s)).collect();
    format!("[{}]", parts.join(", "))
}

/// Machine-readable JSON for one evaluation (same shape on both sides
/// of the parity sweep).
pub fn eval_to_json(res: &EvalResult) -> String {
    let mut qs = Vec::new();
    for q in &res.per_question {
        qs.push(format!(
            "{{\"text\": {}, \"entity\": {}, \"attr\": {}, \"expected\": {}, \"answer\": {}, \"correct\": {}, \"position\": {}, \"paraphrased\": {}, \"ctx_lines\": {}, \"ctx\": {}}}",
            json_escape(&q.text),
            json_escape(&q.entity),
            json_escape(&q.attr),
            json_escape(&q.expected),
            match &q.answer {
                Some(a) => json_escape(a),
                None => "null".to_string(),
            },
            q.correct,
            fmt_f64(q.position),
            q.paraphrased,
            q.ctx.len(),
            json_str_list(&q.ctx),
        ));
    }
    format!(
        "{{\"policy\": {}, \"accuracy\": {}, \"n\": {}, \"correct\": {}, \"mean_context_lines\": {}, \"total_ctx_lines\": {}, \"by_position\": {{\"early\": {}, \"mid\": {}, \"late\": {}}}, \"by_position_n\": [{}, {}, {}], \"paraphrase_accuracy\": {}, \"paraphrase_n\": {}, \"per_question\": [{}]}}",
        json_escape(res.policy),
        fmt_f64(res.accuracy),
        res.n,
        res.correct,
        fmt_f64(res.mean_context_lines),
        res.total_ctx_lines,
        fmt_f64(res.by_position[0]),
        fmt_f64(res.by_position[1]),
        fmt_f64(res.by_position[2]),
        res.by_position_n[0],
        res.by_position_n[1],
        res.by_position_n[2],
        fmt_f64(res.paraphrase_accuracy),
        res.paraphrase_n,
        qs.join(", "),
    )
}
