//! Student-Aware Teaching (SAT) for LLM agents (arXiv:2611.13144) — Rust port
//! of the Python mechanism in `student_teaching.py` (same session folder).
//!
//! Line-for-line port of the pure logic: SplitMix64 RNG, task generation,
//! the student agent loop, and the teacher pipeline (profile / probes /
//! admission gate) plus the profile-free, profile-only and misleading
//! controls. No external crates, std only, no network.
//!
//! Deliberate divergences from Python (all documented here; anything else
//! is a bug):
//!  * `Teacher::probe` seeds probe tasks with a fixed constant instead of
//!    Python's `hash(lim) % 1000`: CPython string hashing is randomized
//!    per process, but the seed is unused — probe tasks are fixed per
//!    limitation, so the tasks (and outcomes) are identical either way.
//!  * Dict/set iteration order: Python `set` iteration is nondeterministic
//!    in principle, but the module only uses sets for membership and for
//!    counting (addition commutes), so results are order-independent.
//!    Rust uses BTreeMap/BTreeSet (sorted) and Vec (insertion order) where
//!    order is observable.
//!  * `f64` conversion of a u64 uses the same round-to-nearest-even rule in
//!    both languages, so `SplitMix64::random` streams match exactly.

use std::collections::{BTreeMap, BTreeSet, HashMap};

// ---------------------------------------------------------------- RNG

const MASK64: u64 = u64::MAX;

#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed & MASK64 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn randbelow(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        // Compare in u128: for n == 1 the limit is 2^64, which wraps to 0
        // as u64 and would loop forever (choice() on a 1-element slice
        // calls randbelow(1)). Python computes the same limit in bigint.
        let total: u128 = 1u128 << 64;
        let limit = total - (total % n as u128);
        loop {
            let v = self.next_u64();
            if (v as u128) < limit {
                return v % n;
            }
        }
    }

    pub fn random(&mut self) -> f64 {
        self.next_u64() as f64 / 18446744073709551616.0
    }

    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        &seq[self.randbelow(seq.len() as u64) as usize]
    }

    pub fn shuffle<T: Clone>(&mut self, seq: &[T]) -> Vec<T> {
        let mut v = seq.to_vec();
        let n = v.len();
        for i in (1..n).rev() {
            let j = self.randbelow((i + 1) as u64) as usize;
            v.swap(i, j);
        }
        v
    }
}

// ---------------------------------------------------------------- limitations

pub const LIMITATIONS: [&str; 5] = ["L1", "L2", "L3", "L4", "L5"];

pub fn skill_text(lim: &str) -> &'static str {
    match lim {
        "L1" => "Before reading a file, list its directory and confirm the file exists; if it is missing, use the task's fallback value instead of reading it.",
        "L2" => "When searching a file for a value, use substring matching on each line, never whole-line exact matching.",
        "L3" => "After writing a file, always re-read it fresh before using its value again; never reuse a value read before the write.",
        "L4" => "If a read fails with a transient error, retry it once before giving up on that goal.",
        "L5" => "Never do multi-digit multiplication in your head; always call the compute tool for it.",
        _ => panic!("unknown limitation {lim}"),
    }
}

pub fn fingerprint(lim: &str) -> &'static str {
    match lim {
        "L1" => "read_missing_error",
        "L2" => "search_miss",
        "L3" => "stale_value",
        "L4" => "no_retry_abandon",
        "L5" => "math_error",
        _ => panic!("unknown limitation {lim}"),
    }
}

pub fn lim_from_fingerprint(fp: &str) -> Option<&'static str> {
    LIMITATIONS.iter().copied().find(|l| fingerprint(l) == fp)
}

pub const BASE_BUDGET: i64 = 14;
// Calibration: a student that follows the correct procedure on every
// generated task needs 6-15 calls (measured: <=10 calls on 84% of tasks,
// 11 on 11.5%, 12-13 on 4.5%, 14-15 on ~0.2% — the rare 15-call tasks
// combine flaky retries on two distinct keys with a stale re-read).
// BASE_BUDGET=14 leaves the untaught student never budget-limited on the
// 200-task test set, a 3-skill SAT library (budget 11) rarely limited
// (~4.5% of tasks), and a 5-skill profile-free library (budget 9)
// visibly limited (~16% of tasks) — the paper's "a procedure that helps
// one model can exhaust the budget of another". Known limit: ~0.2% of
// tasks need 15 calls and are unsolvable even for correct behavior.

// ---------------------------------------------------------------- tasks

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Var(String),
    Lit(i64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Read { file: String, key: String, dest: String },
    Search { file: String, needle: String, dest: String },
    Compute { op: String, a: Operand, b: Operand, dest: String },
    Write { file: String, key: String, value: Operand },
    Answer { var: String },
}

#[derive(Clone, Debug)]
pub struct Task {
    pub ops: Vec<Op>,
    pub files: BTreeMap<String, BTreeMap<String, i64>>,
    pub notes: BTreeMap<String, Vec<String>>,
    pub missing: BTreeSet<(String, String)>,
    pub flaky: BTreeSet<String>,
    pub fallback: BTreeMap<(String, String), i64>,
    pub traps: BTreeSet<String>,
    pub gold: i64,
}

fn first_int_in_line(line: &str) -> Option<i64> {
    let spaced: String = line.chars().map(|c| if c == ':' { ' ' } else { c }).collect();
    for tok in spaced.split_whitespace() {
        if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
            return tok.parse::<i64>().ok();
        }
    }
    None
}

fn simulate_gold(
    ops: &[Op],
    files: &BTreeMap<String, BTreeMap<String, i64>>,
    notes: &BTreeMap<String, Vec<String>>,
    missing: &BTreeSet<(String, String)>,
    fallback: &BTreeMap<(String, String), i64>,
) -> i64 {
    let mut vals: BTreeMap<String, i64> = BTreeMap::new();
    let mut world: BTreeMap<String, BTreeMap<String, i64>> = files.clone();
    let resolve = |o: &Operand, vals: &BTreeMap<String, i64>| -> i64 {
        match o {
            Operand::Var(v) => vals[v],
            Operand::Lit(n) => *n,
        }
    };
    for op in ops {
        match op {
            Op::Read { file, key, dest } => {
                let v = if missing.contains(&(file.clone(), key.clone())) {
                    fallback[&(file.clone(), key.clone())]
                } else {
                    world[file][key]
                };
                vals.insert(dest.clone(), v);
            }
            Op::Search { file, needle, dest } => {
                let mut found = None;
                for line in &notes[file] {
                    if line.contains(needle.as_str()) {
                        found = first_int_in_line(line);
                        break;
                    }
                }
                vals.insert(dest.clone(), found.expect("search must match in gold sim"));
            }
            Op::Compute { op, a, b, dest } => {
                let av = resolve(a, &vals);
                let bv = resolve(b, &vals);
                vals.insert(dest.clone(), if op == "add" { av + bv } else { av * bv });
            }
            Op::Write { file, key, value } => {
                let v = resolve(value, &vals);
                world.entry(file.clone()).or_default().insert(key.clone(), v);
            }
            Op::Answer { var } => return vals[var],
        }
    }
    panic!("task has no answer op");
}

fn gen_task(rng: &mut SplitMix64, n_ops: usize, trap_p: f64) -> Task {
    let nfiles = rng.randbelow(3) + 2;
    let mut files: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    for i in 0..nfiles {
        let keys: Vec<&str> = if rng.random() < 0.5 { vec!["a", "b"] } else { vec!["a"] };
        let mut m = BTreeMap::new();
        for k in keys {
            m.insert(k.to_string(), rng.randbelow(9) as i64 + 1);
        }
        files.insert(format!("f{i}.txt"), m);
    }
    let mut notes: BTreeMap<String, Vec<String>> = BTreeMap::new();
    notes.insert(
        "notes.txt".to_string(),
        vec![
            format!("order total: {} units", rng.randbelow(90) + 10),
            format!("tracking code: {}", rng.randbelow(900) + 100),
            format!("discount rate: {} percent", rng.randbelow(20) + 1),
        ],
    );
    let mut traps: BTreeSet<String> = BTreeSet::new();
    for t in ["missing", "flaky", "stale", "search_sub", "big_mult"] {
        if rng.random() < trap_p {
            traps.insert(t.to_string());
        }
    }
    let mut missing: BTreeSet<(String, String)> = BTreeSet::new();
    let mut flaky: BTreeSet<String> = BTreeSet::new();
    let mut fallback: BTreeMap<(String, String), i64> = BTreeMap::new();
    if traps.contains("missing") {
        let f = format!("f{}.txt", rng.randbelow(nfiles));
        missing.insert((f.clone(), "a".to_string()));
        fallback.insert((f, "a".to_string()), rng.randbelow(9) as i64 + 1);
    }
    if traps.contains("flaky") {
        flaky.insert(format!("f{}.txt", rng.randbelow(nfiles)));
    }

    let mut ops: Vec<Op> = Vec::new();
    let mut vars: Vec<String> = Vec::new();
    let newvar = |vars: &mut Vec<String>| -> String {
        let v = format!("v{}", vars.len());
        vars.push(v.clone());
        v
    };

    let mut stale_file = String::new();
    let mut stale_key = String::new();
    if traps.contains("stale") {
        stale_file = format!("f{}.txt", rng.randbelow(nfiles));
        stale_key = "a".to_string();
        let d = newvar(&mut vars);
        ops.push(Op::Read { file: stale_file.clone(), key: stale_key.clone(), dest: d });
    }
    let n_body = n_ops - if traps.contains("stale") { 1 } else { 0 };
    for _ in 0..n_body {
        let kind = rng.randbelow(3);
        if kind == 0 {
            let f = format!("f{}.txt", rng.randbelow(nfiles));
            let keys: Vec<String> = files[&f].keys().cloned().collect();
            let k = rng.choice(&keys).clone();
            let d = newvar(&mut vars);
            ops.push(Op::Read { file: f, key: k, dest: d });
        } else if kind == 1 && traps.contains("search_sub") {
            let needle = rng.choice(&["total", "tracking", "discount"]).to_string();
            let d = newvar(&mut vars);
            ops.push(Op::Search { file: "notes.txt".to_string(), needle, dest: d });
        } else {
            let a: Operand = if !vars.is_empty() && rng.random() < 0.7 {
                Operand::Var(rng.choice(&vars).clone())
            } else {
                Operand::Lit(rng.randbelow(9) as i64 + 1)
            };
            if traps.contains("big_mult") && rng.random() < 0.6 {
                let b = Operand::Lit(rng.randbelow(5) as i64 + 6);
                let d = newvar(&mut vars);
                ops.push(Op::Compute { op: "mul".to_string(), a, b, dest: d });
            } else {
                let b = Operand::Lit(rng.randbelow(9) as i64 + 1);
                let cop = rng.choice(&["add", "mul"]).to_string();
                let d = newvar(&mut vars);
                ops.push(Op::Compute { op: cop, a, b, dest: d });
            }
        }
    }
    if traps.contains("stale") {
        let nv = rng.randbelow(9) as i64 + 1;
        ops.push(Op::Write {
            file: stale_file.clone(),
            key: stale_key.clone(),
            value: Operand::Lit(nv),
        });
        let d = newvar(&mut vars);
        ops.push(Op::Read { file: stale_file, key: stale_key, dest: d });
    }
    let last = vars.last().unwrap().clone();
    ops.push(Op::Answer { var: last });
    let gold = simulate_gold(&ops, &files, &notes, &missing, &fallback);
    Task { ops, files, notes, missing, flaky, fallback, traps, gold }
}

pub fn gen_tasks(seed: u64, n: usize) -> Vec<Task> {
    gen_tasks_window(seed, n, 4, 7, 0.4)
}

pub fn gen_tasks_window(seed: u64, n: usize, n_ops_lo: usize, n_ops_hi: usize, trap_p: f64) -> Vec<Task> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s = seed
            .wrapping_mul(1_000_003)
            .wrapping_add((i as u64).wrapping_mul(7919))
            .wrapping_add(17);
        let mut rng = SplitMix64::new(s);
        let n_ops = n_ops_lo + rng.randbelow((n_ops_hi - n_ops_lo + 1) as u64) as usize;
        out.push(gen_task(&mut rng, n_ops, trap_p));
    }
    out
}

pub fn gen_probe_task(lim: &str) -> Task {
    match lim {
        "L1" => {
            // Read a (file, key) pair that is "missing" although the file
            // is listed in the directory: a student that lists first uses
            // the fallback; a student that never lists fails. (Matches
            // the generator's notion of "missing".)
            let mut files = BTreeMap::new();
            let mut m0 = BTreeMap::new();
            m0.insert("a".to_string(), 5);
            files.insert("f0.txt".to_string(), m0);
            let mut m9 = BTreeMap::new();
            m9.insert("a".to_string(), 3);
            files.insert("f9.txt".to_string(), m9);
            let mut missing = BTreeSet::new();
            missing.insert(("f9.txt".to_string(), "a".to_string()));
            let mut fallback = BTreeMap::new();
            fallback.insert(("f9.txt".to_string(), "a".to_string()), 7);
            let mut traps = BTreeSet::new();
            traps.insert("missing".to_string());
            Task {
                ops: vec![
                    Op::Read { file: "f9.txt".to_string(), key: "a".to_string(), dest: "v0".to_string() },
                    Op::Answer { var: "v0".to_string() },
                ],
                files,
                notes: BTreeMap::new(),
                missing,
                flaky: BTreeSet::new(),
                fallback,
                traps,
                gold: 7,
            }
        }
        "L2" => {
            let mut notes = BTreeMap::new();
            notes.insert("notes.txt".to_string(), vec!["order total: 42 units".to_string()]);
            let mut traps = BTreeSet::new();
            traps.insert("search_sub".to_string());
            Task {
                ops: vec![
                    Op::Search { file: "notes.txt".to_string(), needle: "total".to_string(), dest: "v0".to_string() },
                    Op::Answer { var: "v0".to_string() },
                ],
                files: BTreeMap::new(),
                notes,
                missing: BTreeSet::new(),
                flaky: BTreeSet::new(),
                fallback: BTreeMap::new(),
                traps,
                gold: 42,
            }
        }
        "L3" => {
            let mut files = BTreeMap::new();
            let mut m = BTreeMap::new();
            m.insert("a".to_string(), 5);
            files.insert("f0.txt".to_string(), m);
            let mut traps = BTreeSet::new();
            traps.insert("stale".to_string());
            Task {
                ops: vec![
                    Op::Read { file: "f0.txt".to_string(), key: "a".to_string(), dest: "v0".to_string() },
                    Op::Write { file: "f0.txt".to_string(), key: "a".to_string(), value: Operand::Lit(9) },
                    Op::Read { file: "f0.txt".to_string(), key: "a".to_string(), dest: "v1".to_string() },
                    Op::Answer { var: "v1".to_string() },
                ],
                files,
                notes: BTreeMap::new(),
                missing: BTreeSet::new(),
                flaky: BTreeSet::new(),
                fallback: BTreeMap::new(),
                traps,
                gold: 9,
            }
        }
        "L4" => {
            let mut files = BTreeMap::new();
            let mut m = BTreeMap::new();
            m.insert("a".to_string(), 8);
            files.insert("f0.txt".to_string(), m);
            let mut flaky = BTreeSet::new();
            flaky.insert("f0.txt".to_string());
            let mut traps = BTreeSet::new();
            traps.insert("flaky".to_string());
            Task {
                ops: vec![
                    Op::Read { file: "f0.txt".to_string(), key: "a".to_string(), dest: "v0".to_string() },
                    Op::Answer { var: "v0".to_string() },
                ],
                files,
                notes: BTreeMap::new(),
                missing: BTreeSet::new(),
                flaky,
                fallback: BTreeMap::new(),
                traps,
                gold: 8,
            }
        }
        "L5" => {
            let mut files = BTreeMap::new();
            let mut m = BTreeMap::new();
            m.insert("a".to_string(), 6);
            files.insert("f0.txt".to_string(), m);
            let mut traps = BTreeSet::new();
            traps.insert("big_mult".to_string());
            Task {
                ops: vec![
                    Op::Read { file: "f0.txt".to_string(), key: "a".to_string(), dest: "v0".to_string() },
                    Op::Compute { op: "mul".to_string(), a: Operand::Var("v0".to_string()), b: Operand::Lit(7), dest: "v1".to_string() },
                    Op::Answer { var: "v1".to_string() },
                ],
                files,
                notes: BTreeMap::new(),
                missing: BTreeSet::new(),
                flaky: BTreeSet::new(),
                fallback: BTreeMap::new(),
                traps,
                gold: 42,
            }
        }
        _ => panic!("unknown limitation {lim}"),
    }
}

// ---------------------------------------------------------------- student

#[derive(Clone)]
pub struct StudentAgent {
    pub limitations: BTreeSet<String>,
    pub skills: BTreeSet<String>,
}

impl StudentAgent {
    pub fn new(limitations: &[&str], skills: &[&str]) -> Self {
        StudentAgent {
            limitations: limitations.iter().map(|s| s.to_string()).collect(),
            skills: skills.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn budget(&self) -> i64 {
        BASE_BUDGET - self.skills.len() as i64
    }

    fn has_lim(&self, lim: &str) -> bool {
        self.limitations.contains(lim) && !self.skills.contains(lim)
    }

    /// Returns (success, calls_used, fingerprints in order of occurrence).
    pub fn solve(&self, task: &Task) -> (bool, i64, Vec<String>) {
        let mut world: BTreeMap<String, BTreeMap<String, i64>> = task.files.clone();
        let mut calls: i64 = 0;
        let mut prints: Vec<String> = Vec::new();
        let mut vals: BTreeMap<String, i64> = BTreeMap::new();
        let mut cache: BTreeMap<(String, String), i64> = BTreeMap::new();
        let mut read_attempts: HashMap<(String, String), u32> = HashMap::new();
        let budget = self.budget();

        // spend(n): charge n calls; Ok iff still within budget.
        macro_rules! spend {
            ($n:expr) => {{
                calls += $n;
                if calls > budget {
                    return (false, calls, {
                        let mut p = prints.clone();
                        p.push("budget_exhausted".to_string());
                        p
                    });
                }
            }};
        }

        let mut listed: BTreeSet<String> = BTreeSet::new();
        if !self.has_lim("L1") {
            // correct student lists the directory first (1 call)
            calls += 1;
            if calls > budget {
                return (false, calls, vec!["budget_exhausted".to_string()]);
            }
            listed = world.keys().cloned().collect();
            for k in task.notes.keys() {
                listed.insert(k.clone());
            }
        }

        for op in &task.ops {
            match op {
                Op::Read { file, key, dest } => {
                    spend!(1);
                    if self.has_lim("L3") {
                        if let Some(v) = cache.get(&(file.clone(), key.clone())) {
                            vals.insert(dest.clone(), *v);
                            prints.push(fingerprint("L3").to_string());
                            continue;
                        }
                    }
                    // do_read
                    let key2 = (file.clone(), key.clone());
                    let res: Option<i64> = (|| {
                        if !world.contains_key(file) && !task.missing.contains(&key2) {
                            return None;
                        }
                        if task.missing.contains(&key2) {
                            if listed.contains(file) {
                                return Some(task.fallback[&key2]);
                            }
                            // L1: read directly -> error, retry, abandon
                            calls += 1;
                            if calls > budget {
                                return None;
                            }
                            calls += 1;
                            if calls > budget {
                                return None;
                            }
                            prints.push(fingerprint("L1").to_string());
                            return None;
                        }
                        let n = {
                            let e = read_attempts.entry(key2.clone()).or_insert(0);
                            *e += 1;
                            *e
                        };
                        if task.flaky.contains(file) && n == 1 {
                            calls += 1;
                            if calls > budget {
                                return None;
                            }
                            if self.has_lim("L4") {
                                prints.push(fingerprint("L4").to_string());
                                return None;
                            }
                            calls += 1;
                            if calls > budget {
                                return None;
                            }
                            return Some(world[file][key]);
                        }
                        Some(world[file][key])
                    })();
                    // NOTE: a budget failure inside do_read returns WITHOUT
                    // appending "budget_exhausted" (matches Python: the inner
                    // spends return False and the caller returns prints as-is).
                    match res {
                        None => return (false, calls, prints),
                        Some(v) => {
                            vals.insert(dest.clone(), v);
                            cache.insert(key2, v);
                        }
                    }
                }
                Op::Search { file, needle, dest } => {
                    spend!(1);
                    let found: Option<i64> = if self.has_lim("L2") {
                        let mut f = None;
                        for line in &task.notes[file] {
                            if line == needle {
                                f = first_int_in_line(line);
                                break;
                            }
                        }
                        if f.is_none() {
                            prints.push(fingerprint("L2").to_string());
                        }
                        f
                    } else {
                        let mut f = None;
                        for line in &task.notes[file] {
                            if line.contains(needle.as_str()) {
                                f = first_int_in_line(line);
                                break;
                            }
                        }
                        f
                    };
                    match found {
                        None => return (false, calls, prints),
                        Some(v) => {
                            vals.insert(dest.clone(), v);
                        }
                    }
                }
                Op::Compute { op, a, b, dest } => {
                    let av = match a {
                        Operand::Var(v) => vals[v],
                        Operand::Lit(n) => *n,
                    };
                    let bv = match b {
                        Operand::Var(v) => vals[v],
                        Operand::Lit(n) => *n,
                    };
                    if op == "mul" && bv > 5 && self.has_lim("L5") {
                        // mental math: deterministic wrong answer, no tool call
                        vals.insert(dest.clone(), av * bv + av);
                        prints.push(fingerprint("L5").to_string());
                        continue;
                    }
                    spend!(1);
                    vals.insert(dest.clone(), if op == "add" { av + bv } else { av * bv });
                }
                Op::Write { file, key, value } => {
                    spend!(1);
                    let v = match value {
                        Operand::Var(vv) => vals[vv],
                        Operand::Lit(n) => *n,
                    };
                    world.entry(file.clone()).or_default().insert(key.clone(), v);
                    // cache intentionally NOT invalidated (see Python docstring)
                }
                Op::Answer { var } => {
                    spend!(1);
                    return (vals[var] == task.gold, calls, prints);
                }
            }
        }
        let mut p = prints.clone();
        p.push("no_answer".to_string());
        (false, calls, p)
    }
}

pub fn run_batch(student: &StudentAgent, tasks: &[Task]) -> (f64, BTreeMap<String, usize>) {
    let mut ok = 0usize;
    let mut fp_counts: BTreeMap<String, usize> =
        LIMITATIONS.iter().map(|l| (l.to_string(), 0)).collect();
    for t in tasks {
        let (s, _, prints) = student.solve(t);
        ok += s as usize;
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for p in prints {
            if let Some(l) = lim_from_fingerprint(&p) {
                seen.insert(l.to_string());
            }
        }
        for l in seen {
            *fp_counts.get_mut(&l).unwrap() += 1;
        }
    }
    (ok as f64 / tasks.len() as f64, fp_counts)
}

// ---------------------------------------------------------------- teacher

pub struct Teacher {
    pub seed: u64,
    pub probe_n: usize,
    pub admit_n: usize,
    pub profile_threshold: f64,
    pub admit_threshold: f64,
    pub spurious_p: f64,
}

impl Default for Teacher {
    fn default() -> Self {
        Teacher {
            seed: 20261111,
            probe_n: 6,
            admit_n: 12,
            profile_threshold: 0.05,
            admit_threshold: 0.15,
            spurious_p: 0.15,
        }
    }
}

pub struct ProbeReport {
    pub confirmed: bool,
    pub fail_rate: f64,
}

pub struct AdmissionReport {
    pub admitted: bool,
    pub delta: f64,
    pub base_ok: usize,
    pub skilled_ok: usize,
    pub n: usize,
}

pub struct TeachReport {
    pub profile_rates: BTreeMap<String, f64>,
    pub hypotheses: Vec<String>,
    pub probes: BTreeMap<String, ProbeReport>,
    pub admissions: BTreeMap<String, AdmissionReport>,
}

fn trap_of_lim(lim: &str) -> &'static str {
    match lim {
        "L1" => "missing",
        "L2" => "search_sub",
        "L3" => "stale",
        "L4" => "flaky",
        "L5" => "big_mult",
        _ => panic!("unknown limitation {lim}"),
    }
}

impl Teacher {
    pub fn profile(&self, student: &StudentAgent, dev_tasks: &[Task]) -> (BTreeMap<String, f64>, Vec<String>) {
        let mut counts: BTreeMap<String, usize> =
            LIMITATIONS.iter().map(|l| (l.to_string(), 0)).collect();
        let mut rng = SplitMix64::new(self.seed);
        for t in dev_tasks {
            let (_, _, prints) = student.solve(t);
            let mut fps: BTreeSet<String> = BTreeSet::new();
            for p in &prints {
                if lim_from_fingerprint(p).is_some() {
                    fps.insert(p.clone());
                }
            }
            if !fps.is_empty() && rng.random() < self.spurious_p {
                let others: Vec<&str> = LIMITATIONS
                    .iter()
                    .copied()
                    .filter(|l| {
                        !fps.iter().any(|p| lim_from_fingerprint(p) == Some(l))
                    })
                    .collect();
                let other = rng.choice(&others);
                fps.insert(fingerprint(other).to_string());
            }
            for p in fps {
                if let Some(l) = lim_from_fingerprint(&p) {
                    *counts.get_mut(l).unwrap() += 1;
                }
            }
        }
        let rates: BTreeMap<String, f64> = LIMITATIONS
            .iter()
            .map(|l| (l.to_string(), counts[*l] as f64 / dev_tasks.len() as f64))
            .collect();
        let hyps: Vec<String> = LIMITATIONS
            .iter()
            .copied()
            .filter(|l| rates[*l] >= self.profile_threshold)
            .map(|l| l.to_string())
            .collect();
        (rates, hyps)
    }

    pub fn probe(&self, student: &StudentAgent, lim: &str) -> (bool, f64) {
        let mut fails = 0usize;
        for _ in 0..self.probe_n {
            let t = gen_probe_task(lim);
            let (s, _, prints) = student.solve(&t);
            if !s && prints.iter().any(|p| p == fingerprint(lim)) {
                fails += 1;
            }
        }
        let rate = fails as f64 / self.probe_n as f64;
        (rate >= 0.5, rate)
    }

    pub fn admit(&self, student: &StudentAgent, lim: &str, matched_tasks: &[Task]) -> AdmissionReport {
        let base = StudentAgent::new(
            &student.limitations.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &[],
        );
        let skilled = StudentAgent::new(
            &student.limitations.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &[lim],
        );
        let b_ok = matched_tasks.iter().filter(|t| base.solve(t).0).count();
        let s_ok = matched_tasks.iter().filter(|t| skilled.solve(t).0).count();
        let n = matched_tasks.len();
        let delta = if n > 0 { (s_ok as f64 - b_ok as f64) / n as f64 } else { 0.0 };
        AdmissionReport { admitted: delta >= self.admit_threshold, delta, base_ok: b_ok, skilled_ok: s_ok, n }
    }

    pub fn teach(&self, student: &StudentAgent, dev_tasks: &[Task]) -> (Vec<String>, TeachReport) {
        let (rates, hyps) = self.profile(student, dev_tasks);
        let mut report = TeachReport {
            profile_rates: rates,
            hypotheses: hyps.clone(),
            probes: BTreeMap::new(),
            admissions: BTreeMap::new(),
        };
        let mut library: Vec<String> = Vec::new();
        for lim in &hyps {
            let (confirmed, rate) = self.probe(student, lim);
            report.probes.insert(lim.clone(), ProbeReport { confirmed, fail_rate: rate });
            if !confirmed {
                continue;
            }
            let trap = trap_of_lim(lim);
            let matched: Vec<Task> = dev_tasks
                .iter()
                .filter(|t| t.traps.contains(trap))
                .take(self.admit_n)
                .cloned()
                .collect();
            let rep = self.admit(student, lim, &matched);
            let admitted = rep.admitted;
            report.admissions.insert(lim.clone(), rep);
            if admitted {
                library.push(lim.clone());
            }
        }
        (library, report)
    }
}

pub fn profile_free_teach() -> Vec<String> {
    LIMITATIONS.iter().map(|s| s.to_string()).collect()
}

pub fn profile_only_teach(student: &StudentAgent, dev_tasks: &[Task], seed: u64) -> (Vec<String>, Vec<String>, BTreeMap<String, f64>) {
    let teacher = Teacher { seed, ..Default::default() };
    let (rates, _) = teacher.profile(student, dev_tasks);
    let mut order: Vec<&str> = LIMITATIONS.to_vec();
    order.sort_by(|a, b| rates[*b].partial_cmp(&rates[*a]).unwrap());
    let top: Vec<String> = order.iter().take(2).map(|s| s.to_string()).collect();
    let mut library: Vec<String> = Vec::new();
    for lim in &top {
        let base = StudentAgent::new(
            &student.limitations.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &[],
        );
        let skilled = StudentAgent::new(
            &student.limitations.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &[lim.as_str()],
        );
        let b_ok = dev_tasks.iter().filter(|t| base.solve(t).0).count();
        let s_ok = dev_tasks.iter().filter(|t| skilled.solve(t).0).count();
        if (s_ok as f64 - b_ok as f64) / dev_tasks.len() as f64 >= 0.05 {
            library.push(lim.clone());
        }
    }
    (library, top, rates)
}

pub fn misleading_teach(
    student: &StudentAgent,
    dev_tasks: &[Task],
    wrong_lims: &[&str],
    seed: u64,
) -> (Vec<String>, BTreeMap<String, AdmissionReport>) {
    let teacher = Teacher { seed, ..Default::default() };
    let mut admitted: Vec<String> = Vec::new();
    let mut report: BTreeMap<String, AdmissionReport> = BTreeMap::new();
    for lim in wrong_lims {
        let trap = trap_of_lim(lim);
        let matched: Vec<Task> = dev_tasks
            .iter()
            .filter(|t| t.traps.contains(trap))
            .take(12)
            .cloned()
            .collect();
        let rep = teacher.admit(student, lim, &matched);
        let ok = rep.admitted;
        report.insert(lim.to_string(), rep);
        if ok {
            admitted.push(lim.to_string());
        }
    }
    (admitted, report)
}
