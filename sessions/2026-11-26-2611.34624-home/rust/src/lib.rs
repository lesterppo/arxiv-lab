//! HoMe: Toward Practical Harness Optimization of Memory Systems.
//!
//! arXiv:2611.34624 "HoMe: Toward Practical Harness Optimization of
//! Memory Systems for AI Assistants" (Agentscope-AI / ReMe).
//!
//! Rust port of `src/arxiv_lab/agent/home.py` (std only, no external
//! crates, no network). The Python version stays the primary verdict
//! carrier; this port mirrors its mechanism and is verified against
//! it (see `tests/home_tests.rs`, parity sweep).
//!
//! Deliberate divergences from Python (documented, as required):
//! * Entities / attributes / values are indices into fixed tables
//!   (`ENTITIES`, `ATTRIBUTES`, `VALUES`) instead of strings; the
//!   tables are in the same order as Python's, so parity dumps map
//!   1:1. Behaviour is identical.
//! * `Harness.recency` is stored as a percentage (u8: 0/25/50/100)
//!   instead of an f64, so `Harness` is `Eq + Hash` and candidate
//!   sets stay deterministic. Scoring converts to f64 identically.
//! * World generation uses a xorshift64 RNG, not Python's
//!   `random.Random` (Mersenne Twister); the *shapes* of generated
//!   worlds match Python's (same distributions, same call order --
//!   see `make_world`), but sampled values differ. Exact
//!   cross-language equality is the parity sweep's job
//!   (`tests/parity_cases.txt`, worlds dumped from Python).
//! * `build_memory` returns `Vec<Vec<Fact>>` indexed by case
//!   position instead of a `dict` keyed by case index.
//! * No `frozenset`/hash iteration anywhere: every iteration order is
//!   insertion order or an explicit sort, so results are identical
//!   across runs and processes (the CAPE determinism lesson).

pub const ENTITIES: [&str; 8] =
    ["ana", "ben", "cy", "dan", "eli", "fay", "gus", "hal"];
pub const ATTRIBUTES: [&str; 5] =
    ["office", "phone", "project", "manager", "city"];
/// Synonym of each attribute, same table as Python's `SYNONYM`.
pub const SYNONYMS: [&str; 5] =
    ["workplace", "telephone", "assignment", "supervisor", "town"];
pub const VALUES: [&str; 12] = ["alpha", "bravo", "charlie", "delta",
    "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"];

pub const N_SESSIONS: usize = 4;
pub const TOK_PROMPT: f64 = 20.0;
pub const TOK_PER_FACT: f64 = 12.0;
pub const TOK_FUZZY: f64 = 10.0;
pub const N_FORMAL: usize = 6;
pub const SMOKE_N: usize = 8;

// ---------------------------------------------------------------- RNG
/// xorshift64* -- deterministic, std-only. `below(n)` is `next % n`,
/// exactly as the Python replica in `../soak_py.py` implements it.
#[derive(Clone, Copy, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Zero seed would freeze xorshift; mix it like the Python side.
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }
    /// Fisher-Yates shuffle, in place.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }
    /// Sample `k` distinct items from `xs` (order: shuffled, first k).
    pub fn sample<T: Clone>(&mut self, xs: &[T], k: usize) -> Vec<T> {
        let mut pool = xs.to_vec();
        self.shuffle(&mut pool);
        pool.truncate(k);
        pool
    }
}

// ---------------------------------------------------------------- world
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fact {
    pub entity: usize,
    pub attribute: usize,
    pub value: usize,
    pub session: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    pub entity: usize,
    pub attribute: usize,
    /// The attribute named by the query (true attribute index).
    pub query_attribute: usize,
    /// True when the query names the attribute's synonym.
    pub query_synonym: bool,
    pub historical: bool,
    pub target_session: usize,
    pub gold: usize,
    pub history: Vec<Fact>,
}

/// Generate a world with the same shape as Python's `make_world`:
/// same distributions, same RNG call order. Sampled values differ
/// (xorshift vs Mersenne Twister); `../soak_py.py` replicates this
/// function call-for-call in Python for the soak cross-check.
pub fn make_world(seed: u64, n_cases: usize) -> Vec<Case> {
    let mut rng = Rng::new(seed);
    let mut cases = Vec::with_capacity(n_cases);
    for _ in 0..n_cases {
        let entity = rng.below(ENTITIES.len());
        let attr = rng.below(ATTRIBUTES.len());
        let n_revisions = [1usize, 1, 2][rng.below(3)];
        let mut sessions: Vec<usize> = rng.sample(&[0, 1, 2, 3], n_revisions + 1);
        sessions.sort_unstable();
        let vals: Vec<usize> = rng.sample(&(0..VALUES.len()).collect::<Vec<_>>(),
                                          n_revisions + 1);
        let mut facts: Vec<Fact> = sessions.iter().zip(vals.iter())
            .map(|(&s, &v)| Fact { entity, attribute: attr, value: v, session: s })
            .collect();
        // distractors: same attribute, other entities
        let n_dist = 1 + rng.below(3);
        for _ in 0..n_dist {
            let others: Vec<usize> = (0..ENTITIES.len())
                .filter(|&e| e != entity).collect();
            let other = others[rng.below(others.len())];
            facts.push(Fact {
                entity: other,
                attribute: attr,
                value: rng.below(VALUES.len()),
                session: rng.below(N_SESSIONS),
            });
        }
        // distractor: same entity, other attribute
        let other_attrs: Vec<usize> = (0..ATTRIBUTES.len())
            .filter(|&a| a != attr).collect();
        let other_attr = other_attrs[rng.below(other_attrs.len())];
        facts.push(Fact {
            entity,
            attribute: other_attr,
            value: rng.below(VALUES.len()),
            session: rng.below(N_SESSIONS),
        });
        let historical = rng.below(100) < 25;
        let (target, gold) = if historical {
            let ti = rng.below(sessions.len());
            (sessions[ti], vals[ti])
        } else {
            (N_SESSIONS - 1, *vals.last().unwrap())
        };
        let query_synonym = rng.below(100) < 30;
        cases.push(Case {
            entity,
            attribute: attr,
            query_attribute: attr,
            query_synonym,
            historical,
            target_session: target,
            gold,
            history: facts,
        });
    }
    cases
}

// ---------------------------------------------------------------- harness
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Revision {
    All,
    Latest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnswerRule {
    Top1,
    Majority,
}

/// A memory-system harness. `recency` is a percentage (0/25/50/100),
/// matching Python's 0.0/0.25/0.5/1.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Harness {
    pub top_k: usize,
    pub recency: u8,
    pub revision: Revision,
    pub fuzzy: bool,
    pub answer: AnswerRule,
}

impl Default for Harness {
    fn default() -> Self {
        Harness {
            top_k: 3,
            recency: 0,
            revision: Revision::All,
            fuzzy: false,
            answer: AnswerRule::Majority,
        }
    }
}

impl Harness {
    pub fn tokens(&self) -> f64 {
        TOK_PROMPT + self.top_k as f64 * TOK_PER_FACT
            + if self.fuzzy { TOK_FUZZY } else { 0.0 }
    }
}

/// All 256 harnesses, in the same enumeration order as Python's
/// `ALL_HARNESSES` (top_k, recency, revision, fuzzy, answer).
pub fn all_harnesses() -> Vec<Harness> {
    let mut out = Vec::with_capacity(256);
    for &k in &[1usize, 2, 3, 5] {
        for &r in &[0u8, 25, 50, 100] {
            for &v in &[Revision::All, Revision::Latest] {
                for &f in &[false, true] {
                    for &a in &[AnswerRule::Top1, AnswerRule::Majority] {
                        out.push(Harness { top_k: k, recency: r,
                                           revision: v, fuzzy: f, answer: a });
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------- memory
/// Session-level versioned memory, built once per revision policy and
/// reused across candidates sharing it. `revision == Latest` drops
/// superseded versions at construction time, so historical queries
/// can no longer be answered -- the genuine tradeoff against the
/// STALE failures `Latest` fixes.
pub fn build_memory(cases: &[Case], revision: Revision) -> Vec<Vec<Fact>> {
    cases.iter().map(|c| {
        let mut facts = c.history.clone();
        facts.sort_by_key(|f| (f.session, f.value));
        if revision == Revision::Latest {
            let mut kept: Vec<Fact> = Vec::new();
            for f in facts {
                let key = (f.entity, f.attribute);
                match kept.iter().position(|g: &Fact| (g.entity, g.attribute) == key) {
                    Some(i) => {
                        if f.session >= kept[i].session {
                            kept[i] = f;
                        }
                    }
                    None => kept.push(f),
                }
            }
            kept.sort_by_key(|f| (f.session, f.value));
            kept
        } else {
            facts
        }
    }).collect()
}

// ---------------------------------------------------------------- evaluate
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cause {
    Ok,
    Stale,
    Missing,
}

#[derive(Clone, Debug)]
pub struct Trace {
    pub case_index: usize,
    pub retrieved: Vec<Fact>,
    pub answer: Option<usize>,
    pub correct: bool,
    pub cause: Cause,
}

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub accuracy: f64,
    pub mean_tokens: f64,
    pub traces: Vec<Trace>,
    pub n: usize,
    pub constructions: usize,
}

/// Stratified smoke-test indices (the paper's adaptive sampling):
/// round-robin over the four (historical x synonym) buckets, so the
/// smoke set always covers the failure modes a candidate could break.
/// Deterministic. Identical to Python's `smoke_indices`.
pub fn smoke_indices(cases: &[Case], n: usize) -> Vec<usize> {
    let mut buckets: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (i, c) in cases.iter().enumerate() {
        let key = (if c.historical { 1 } else { 0 }) * 2
            + (if c.query_synonym { 1 } else { 0 });
        buckets[key].push(i);
    }
    let mut out: Vec<usize> = Vec::new();
    let order = [0usize, 2, 1, 3];
    while out.len() < n {
        let mut progressed = false;
        for &b in &order {
            if !buckets[b].is_empty() && out.len() < n {
                out.push(buckets[b].remove(0));
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    out.sort_unstable();
    out
}

fn score(fact: &Fact, case: &Case, h: &Harness) -> Option<f64> {
    let ent = if fact.entity == case.entity { 2.0 } else { 0.0 };
    let att = if fact.attribute == case.query_attribute {
        if case.query_synonym {
            if h.fuzzy { 1.2 } else { return None; }
        } else {
            2.0
        }
    } else {
        return None;
    };
    Some(ent + att + h.recency as f64 / 100.0 * (fact.session as f64 / (N_SESSIONS - 1) as f64))
}

fn retrieve(memory: &[Fact], case: &Case, h: &Harness) -> Vec<Fact> {
    let mut pool: Vec<Fact> = memory.to_vec();
    if case.historical {
        pool.retain(|f| f.session <= case.target_session);
    }
    let mut scored: Vec<(f64, usize, usize, Fact)> = Vec::new();
    for (pos, f) in pool.iter().enumerate() {
        if let Some(s) = score(f, case, h) {
            scored.push((s, pos, pos, *f));
        }
    }
    // Rank: score desc; ties break toward the latest session when
    // recency > 0, else insertion order (oldest first -> stale wins).
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap()
            .then(if h.recency > 0 {
                b.3.session.cmp(&a.3.session)
            } else {
                a.1.cmp(&b.1)
            })
            .then(a.2.cmp(&b.2))
    });
    scored.truncate(h.top_k);
    scored.into_iter().map(|t| t.3).collect()
}

fn answer(retrieved: &[Fact], h: &Harness) -> Option<usize> {
    if retrieved.is_empty() {
        return None;
    }
    if h.answer == AnswerRule::Top1 {
        return Some(retrieved[0].value);
    }
    let mut counts: Vec<(usize, usize)> = Vec::new(); // (value, count), first-appearance order
    for f in retrieved {
        match counts.iter_mut().find(|(v, _)| *v == f.value) {
            Some(e) => e.1 += 1,
            None => counts.push((f.value, 1)),
        }
    }
    let mut best = counts[0];
    for &c in &counts[1..] {
        if c.1 > best.1 {
            best = c;
        }
    }
    Some(best.0)
}

fn classify(case: &Case, answer: Option<usize>) -> Cause {
    match answer {
        None => Cause::Missing,
        Some(v) if v == case.gold => Cause::Ok,
        Some(_) => Cause::Stale,
    }
}

/// Evaluate one harness on all cases. `memory` is built per revision
/// policy and passed in (reuse); `constructions` counts builds
/// charged to this evaluation (0 when reused).
pub fn evaluate(h: &Harness, cases: &[Case], memory: &[Vec<Fact>],
                constructions: usize) -> EvalResult {
    let mut traces = Vec::with_capacity(cases.len());
    let mut correct = 0usize;
    for (i, case) in cases.iter().enumerate() {
        let retrieved = retrieve(&memory[i], case, h);
        let ans = answer(&retrieved, h);
        let ok = ans == Some(case.gold);
        if ok {
            correct += 1;
        }
        traces.push(Trace {
            case_index: i,
            retrieved,
            answer: ans,
            correct: ok,
            cause: classify(case, ans),
        });
    }
    EvalResult {
        accuracy: correct as f64 / cases.len() as f64,
        mean_tokens: h.tokens(),
        traces,
        n: cases.len(),
        constructions,
    }
}

pub fn evaluate_fresh(h: &Harness, cases: &[Case]) -> EvalResult {
    let memory = build_memory(cases, h.revision);
    evaluate(h, cases, &memory, 1)
}

// ------------------------------------------------------- failure analysis
/// Counts of non-OK failure causes, in deterministic (alphabetical)
/// cause order.
pub fn failure_breakdown(result: &EvalResult) -> Vec<(Cause, usize)> {
    let mut stale = 0usize;
    let mut missing = 0usize;
    for t in &result.traces {
        match t.cause {
            Cause::Stale => stale += 1,
            Cause::Missing => missing += 1,
            Cause::Ok => {}
        }
    }
    let mut out = Vec::new();
    if missing > 0 {
        out.push((Cause::Missing, missing));
    }
    if stale > 0 {
        out.push((Cause::Stale, stale));
    }
    out
}

pub fn dominant_cause(result: &EvalResult) -> Option<Cause> {
    let counts = failure_breakdown(result);
    if counts.is_empty() {
        return None;
    }
    // Highest count; ties broken alphabetically (Missing < Stale).
    let mut best = &counts[0];
    for c in &counts[1..] {
        if c.1 > best.1 {
            best = c;
        }
    }
    Some(best.0.clone())
}

/// Prioritized candidate edits for the dominant shared failure, each
/// targeting the localized component. Order matters: the first
/// candidate is the component's primary fix; the risky latest-only
/// fix is last.
pub fn localized_candidates(h: &Harness, cause: &Cause) -> Vec<Harness> {
    match cause {
        Cause::Stale => {
            let mut cands = Vec::new();
            if h.recency < 100 {
                cands.push(Harness { recency: 100, ..*h });
            }
            if h.answer != AnswerRule::Top1 {
                cands.push(Harness { answer: AnswerRule::Top1, ..*h });
            }
            if h.top_k != 1 {
                cands.push(Harness { top_k: 1, ..*h });
            }
            if h.revision != Revision::Latest {
                cands.push(Harness { revision: Revision::Latest, ..*h });
            }
            cands
        }
        Cause::Missing => {
            let mut cands = Vec::new();
            if !h.fuzzy {
                cands.push(Harness { fuzzy: true, ..*h });
            }
            if h.top_k < 5 {
                cands.push(Harness { top_k: 5, ..*h });
            }
            cands
        }
        Cause::Ok => Vec::new(),
    }
}

/// One random single-parameter edit (the unrestricted-editing
/// baseline the paper contrasts against).
pub fn random_edit(h: &Harness, rng: &mut Rng) -> Harness {
    match rng.below(5) {
        0 => {
            let opts: Vec<usize> = [1, 2, 3, 5].iter().cloned()
                .filter(|&k| k != h.top_k).collect();
            Harness { top_k: opts[rng.below(opts.len())], ..*h }
        }
        1 => {
            let opts: Vec<u8> = [0, 25, 50, 100].iter().cloned()
                .filter(|&r| r != h.recency).collect();
            Harness { recency: opts[rng.below(opts.len())], ..*h }
        }
        2 => Harness {
            revision: if h.revision == Revision::All { Revision::Latest } else { Revision::All },
            ..*h
        },
        3 => Harness { fuzzy: !h.fuzzy, ..*h },
        _ => Harness {
            answer: if h.answer == AnswerRule::Top1 { AnswerRule::Majority } else { AnswerRule::Top1 },
            ..*h
        },
    }
}

// ------------------------------------------------------------------- loop
#[derive(Clone, Debug)]
pub struct LoopResult {
    pub best: Harness,
    pub best_result: EvalResult,
    pub start_result: EvalResult,
    pub formal_evals: usize,
    pub smoke_evals: usize,
    pub smoke_rejects: usize,
    pub candidates_tried: usize,
    pub constructions: usize,
    pub history: Vec<(Harness, f64, f64)>,
}

fn better(a: &EvalResult, b: &EvalResult) -> bool {
    if a.accuracy != b.accuracy {
        return a.accuracy > b.accuracy;
    }
    a.mean_tokens < b.mean_tokens
}

/// Memory for one revision policy, built once and cached. The
/// returned memory covers the full `cases` slice; smoke evaluations
/// pass a prefix slice of it.
fn cached_memory(cases: &[Case], rev: Revision,
                 cache: &mut Vec<(Revision, Vec<Vec<Fact>>)>,
                 constructions: &mut usize) -> Vec<Vec<Fact>> {
    if let Some((_, m)) = cache.iter().find(|(r, _)| *r == rev) {
        return m.clone();
    }
    let m = build_memory(cases, rev);
    *constructions += 1;
    cache.push((rev, m.clone()));
    m
}

/// HoMe loop: full initial evaluation, then up to `n_formal` formal
/// candidate evaluations, then a cost phase on remaining budget.
/// Memory is built once per revision policy and reused. With gating,
/// each candidate is smoke-tested on SMOKE_N cases first; a smoke
/// regression rejects it without a formal evaluation. With
/// `localized == false` the candidates are random edits (baseline).
pub fn home_loop(cases: &[Case], start: &Harness, seed: u64,
                 n_formal: usize, gating: bool, localized: bool) -> LoopResult {
    let mut rng = Rng::new(seed.wrapping_add(0xA0761D6478BD642F));
    let mut mem_cache: Vec<(Revision, Vec<Vec<Fact>>)> = Vec::new();
    let mut constructions = 0usize;
    let start_result = {
        let m = cached_memory(cases, start.revision, &mut mem_cache,
                              &mut constructions);
        evaluate(start, cases, &m, 0)
    };
    let mut best = *start;
    let mut best_result = start_result.clone();
    let mut formal = 0usize;
    let mut smoke = 0usize;
    let mut rejects = 0usize;
    let mut tried = 0usize;
    let mut history = vec![(*start, start_result.accuracy, start_result.mean_tokens)];
    let sidx = smoke_indices(cases, SMOKE_N);
    let smoke_cases: Vec<Case> = sidx.iter().map(|&i| cases[i].clone()).collect();
    let mut tried_list: Vec<Harness> = vec![*start];
    // One attempt (smoke gate + optional formal eval) for a candidate.
    // Returns true when the candidate was adopted.
    macro_rules! attempt {
        ($cand:expr) => {{
            let cand: Harness = $cand;
            if formal >= n_formal || tried_list.contains(&cand) {
                false
            } else {
                tried_list.push(cand);
                tried += 1;
                let mut gated_out = false;
                if gating {
                    smoke += 1;
                    let full_c = cached_memory(cases, cand.revision, &mut mem_cache,
                                               &mut constructions);
                    let full_b = cached_memory(cases, best.revision, &mut mem_cache,
                                               &mut constructions);
                    // Smoke memory: the full memory's entries at the
                    // stratified smoke indices, re-keyed by position.
                    let sm_c: Vec<Vec<Fact>> = sidx.iter()
                        .map(|&i| full_c[i].clone()).collect();
                    let sm_b: Vec<Vec<Fact>> = sidx.iter()
                        .map(|&i| full_b[i].clone()).collect();
                    let s_cand = evaluate(&cand, &smoke_cases, &sm_c, 0);
                    let s_best = evaluate(&best, &smoke_cases, &sm_b, 0);
                    if s_cand.accuracy < s_best.accuracy - 1e-9 {
                        rejects += 1;
                        gated_out = true;
                    }
                }
                if gated_out {
                    false
                } else {
                    formal += 1;
                    let m = cached_memory(cases, cand.revision, &mut mem_cache,
                                          &mut constructions);
                    let res = evaluate(&cand, cases, &m, 0);
                    if better(&res, &best_result) {
                        best = cand;
                        best_result = res.clone();
                        history.push((cand, res.accuracy, res.mean_tokens));
                        true
                    } else {
                        false
                    }
                }
            }
        }};
    }
    while formal < n_formal {
        let cause = match dominant_cause(&best_result) {
            Some(c) => c,
            None => break,
        };
        let cands: Vec<Harness> = if localized {
            localized_candidates(&best, &cause)
        } else {
            vec![random_edit(&best, &mut rng)]
        };
        let mut progressed = false;
        for cand in cands {
            if attempt!(cand) {
                progressed = true;
                break; // re-attribute failures from the new best
            }
            if formal >= n_formal {
                break;
            }
        }
        if !progressed {
            break; // no candidate in this round improved anything
        }
    }
    // Cost phase: spend remaining formal evaluations on cheaper
    // harnesses that preserve accuracy. Each candidate is smoke-gated
    // and counts as a formal evaluation.
    for k in [1usize, best.top_k.saturating_sub(1)] {
        if formal >= n_formal || k < 1 || k >= best.top_k {
            continue;
        }
        let cand = Harness { top_k: k, ..best };
        if attempt!(cand) {
            break; // cheapest working k found; nothing cheaper to try
        }
    }
    LoopResult {
        best,
        best_result,
        start_result,
        formal_evals: formal,
        smoke_evals: smoke,
        smoke_rejects: rejects,
        candidates_tried: tried,
        constructions,
        history,
    }
}

/// Exhaustive optimum over the harness space.
pub fn oracle_best(cases: &[Case]) -> (Harness, EvalResult) {
    let mut best_h: Option<Harness> = None;
    let mut best_r: Option<EvalResult> = None;
    for h in all_harnesses() {
        let m = build_memory(cases, h.revision);
        let r = evaluate(&h, cases, &m, 0);
        let take = match &best_r {
            None => true,
            Some(br) => better(&r, br),
        };
        if take {
            best_h = Some(h);
            best_r = Some(r);
        }
    }
    (best_h.unwrap(), best_r.unwrap())
}
