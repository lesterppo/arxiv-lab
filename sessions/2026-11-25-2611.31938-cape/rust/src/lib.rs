//! CAPE: Coordinating Calibration and Search for Tool-Use Policies.
//!
//! Rust port of `src/arxiv_lab/agent/cape.py` (arXiv:2611.31938, Wang,
//! Chen, Yang, Wang). Pure std, deterministic: every iteration over a
//! belief state is sorted (BTreeSet), matching the Python module's
//! sorted-iteration fix of 2026-11-25.
//!
//! Model (paper Sec. 3.1, 4.1-4.3): configurations are combinations of
//! parsing aspects; programs commit iff their encoding matches the
//! configuration on every aspect, else they return an error naming the
//! first mismatching aspect; probes reveal one aspect without
//! committing. A policy tree chooses actions from responses so far,
//! using at most `cap` calls per execution. Objective J(pi) =
//! (|S(pi)|, -M(pi)) lexicographically (completions first, then
//! fewest calls), subject to R0 subset S(pi), where R0 is what the
//! initial policy pi0 (programs in catalog order) completes.
//!
//! Deliberate divergences from Python (all behaviour-preserving):
//! * Configs are aspect-value *indices* (`Vec<u8>`) instead of string
//!   tuples; responses are an enum instead of strings. Partitioning
//!   only needs response equality, so values are identical.
//! * Belief states are `BTreeSet<Config>` (sorted iteration) where
//!   Python uses frozenset + sorted() at the decision points.
//! * `make_world` uses a xorshift64 RNG instead of Python's
//!   `random.Random` (Mersenne Twister): sampled config *sets* differ
//!   from Python's for the same seed, so `verify` asserts the same
//!   *properties* (optimal rate, certification, R0, EF gain) on the
//!   same shapes of worlds rather than identical numbers. Exact
//!   cross-language equality is checked by the parity sweep
//!   (`tests/parity_cases.txt`, generated from Python worlds).
//! * No `f64` anywhere except the acquisition ranking scores, as in
//!   Python.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

// ---------------------------------------------------------------------------
// Paper headline numbers (abstract / Table 1), for reference in tests.
// ---------------------------------------------------------------------------
pub const PAPER_OPTIMAL_RATE: f64 = 502.0 / 504.0;
pub const PAPER_TIME_SAVING: f64 = 0.220;
pub const PAPER_COMPLETION_GAIN_PP: (f64, f64) = (15.38, 34.66);

// ---------------------------------------------------------------------------
// World model
// ---------------------------------------------------------------------------

/// One parsing aspect, e.g. delimiter. Values are indexed 0..k.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Aspect {
    pub name: &'static str,
    pub values: &'static [&'static str],
}

pub const DEFAULT_ASPECTS: &[Aspect] = &[
    Aspect { name: "delim", values: &[",", ";", "\t"] },
    Aspect { name: "date", values: &["DMY", "MDY"] },
    Aspect { name: "quote", values: &["minimal", "all"] },
    Aspect { name: "missing", values: &["empty", "NA"] },
];

pub const EXTENDED_ASPECTS: &[Aspect] = &[
    Aspect { name: "delim", values: &[",", ";", "\t"] },
    Aspect { name: "date", values: &["DMY", "MDY"] },
    Aspect { name: "quote", values: &["minimal", "all"] },
    Aspect { name: "missing", values: &["empty", "NA"] },
    Aspect { name: "numeric", values: &["dot", "comma"] },
    Aspect { name: "header", values: &["yes", "no"] },
];

/// A configuration: one aspect-value index per aspect.
pub type Config = Vec<u8>;

/// An action: a program (encoding) or a probe
/// (primary aspect, primary value, reveal aspect).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Act {
    Prog(Config),
    Probe(usize, u8, usize),
}

/// A tool response. `Unknown(i)` is the optimistic per-config cell
/// marker ("?", h) of paper Sec. 4.2, keyed by the config's index.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Resp {
    Ok,
    Err(usize),
    Reveal(usize, u8),
    Unknown(usize),
}

#[derive(Clone, Debug)]
pub struct World {
    pub aspects: Vec<Aspect>,
    pub configs: Vec<Config>,
    pub programs: Vec<Config>,
    pub probes: Vec<(usize, u8, usize)>,
}

impl World {
    /// f(a, h) -> (response, commit). Deterministic.
    pub fn effect(&self, action: &Act, h: &Config) -> (Resp, bool) {
        match action {
            Act::Prog(enc) => {
                for (i, v) in enc.iter().enumerate() {
                    if h[i] != *v {
                        return (Resp::Err(i), false);
                    }
                }
                (Resp::Ok, true)
            }
            Act::Probe(pi, pval, ri) => {
                if h[*pi] != *pval {
                    return (Resp::Err(*pi), false);
                }
                (Resp::Reveal(*ri, h[*ri]), false)
            }
        }
    }

    /// Full catalog A = P u Q, programs first (catalog order).
    pub fn actions(&self) -> Vec<Act> {
        let mut acts: Vec<Act> = self.programs.iter().map(|p| Act::Prog(p.clone())).collect();
        acts.extend(self.probes.iter().map(|&(pi, pv, ri)| Act::Probe(pi, pv, ri)));
        acts
    }

    pub fn config_index(&self, h: &Config) -> usize {
        self.configs.iter().position(|c| c == h).expect("config not in world")
    }
}

fn product(aspects: &[Aspect]) -> Vec<Config> {
    let mut out: Vec<Config> = vec![vec![]];
    for a in aspects {
        let mut next = Vec::new();
        for prefix in &out {
            for v in 0..a.values.len() as u8 {
                let mut c = prefix.clone();
                c.push(v);
                next.push(c);
            }
        }
        out = next;
    }
    out
}

/// Tiny deterministic RNG (xorshift64star). Used only for Rust-native
/// world generation; parity cases come from Python worlds instead.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9e3779b97f4a7c15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

/// Compare configs the way Python's `sorted(V)` does: configs are
/// tuples of aspect *values* (strings), and Python sorts those strings
/// lexicographically. Value indices do NOT sort the same way (the delim
/// aspect values (",", ";", "\t") are indices 0,1,2 but sort as
/// "\t" < "," < ";"), so sorting by index diverges from Python and
/// changes acquisition order (found 2026-11-25: parity line 3 chose
/// target 0010 in Rust vs 2101 in Python at round 0). Every
/// order-sensitive iteration (missing / partition / optimistic cells /
/// world sampling) must use this order; BTreeSet<Config> storage keeps
/// index order and is used only for set operations.
/// Compare two configs the way Python's `sorted(V)` orders them, without
/// allocating a key per comparison (this runs in the hot partition loop).
fn cmp_config_str(aspects: &[Aspect], a: &Config, b: &Config) -> Ordering {
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let xs = aspects[i].values[*x as usize];
        let ys = aspects[i].values[*y as usize];
        match xs.cmp(ys) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    Ordering::Equal
}

/// Configs of a belief state in Python's `sorted(V)` order.
fn sorted_configs(world: &World, v: &BTreeSet<Config>) -> Vec<Config> {
    let mut out: Vec<Config> = v.iter().cloned().collect();
    out.sort_by(|a, b| cmp_config_str(&world.aspects, a, b));
    out
}

pub fn make_world(n_configs: usize, n_aspects: usize, seed: u64) -> World {
    let aspects: Vec<Aspect> = if n_aspects == 4 {
        DEFAULT_ASPECTS.to_vec()
    } else {
        EXTENDED_ASPECTS.to_vec()
    };
    let mut prod = product(&aspects);
    let mut rng = Rng::new(seed.wrapping_add(0xC0FFEE));
    let configs = if n_configs >= prod.len() {
        prod.clone()
    } else {
        // Partial Fisher-Yates sample, then sort (as Python does).
        for i in 0..n_configs {
            let j = i + rng.below(prod.len() - i);
            prod.swap(i, j);
        }
        let mut sample: Vec<Config> = prod[..n_configs].to_vec();
        sample.sort_by(|a, b| cmp_config_str(&aspects, a, b));
        sample
    };
    let mut probes = Vec::new();
    for pi in 0..aspects.len() {
        for ri in 0..aspects.len() {
            if ri == pi {
                continue;
            }
            for v in 0..aspects[pi].values.len() as u8 {
                probes.push((pi, v, ri));
            }
        }
    }
    let programs = product(&aspects);
    World { aspects, configs, programs, probes }
}

// ---------------------------------------------------------------------------
// Effect table (partial knowledge)
// ---------------------------------------------------------------------------

pub struct EffectTable<'a> {
    pub world: &'a World,
    pub known: BTreeMap<(Act, Config), (Resp, bool)>,
}

impl<'a> EffectTable<'a> {
    pub fn new(world: &'a World) -> Self {
        EffectTable { world, known: BTreeMap::new() }
    }
    pub fn acquire(&mut self, action: &Act, h: &Config) -> (Resp, bool) {
        let eff = self.world.effect(action, h);
        self.known.insert((action.clone(), h.clone()), eff.clone());
        eff
    }
    pub fn is_known(&self, action: &Act, h: &Config) -> bool {
        self.known.contains_key(&(action.clone(), h.clone()))
    }
    pub fn missing(&self, action: &Act, v: &BTreeSet<Config>) -> Vec<Config> {
        // Python's EffectTable.missing returns sorted(V): string order.
        let mut out: Vec<Config> =
            v.iter().filter(|h| !self.is_known(action, h)).cloned().collect();
        out.sort_by(|a, b| cmp_config_str(&self.world.aspects, a, b));
        out
    }
    pub fn closed(&self, action: &Act, v: &BTreeSet<Config>) -> bool {
        v.iter().all(|h| self.is_known(action, h))
    }
    pub fn complete(&self) -> bool {
        self.known.len() == self.world.actions().len() * self.world.configs.len()
    }
}

// ---------------------------------------------------------------------------
// Policies as trees; execution
// ---------------------------------------------------------------------------
// A policy tree node is Stop or Act(action, [(response, subtree)]).

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    Stop,
    Act(Act, Vec<(Resp, Node)>),
}

/// Value = (n_completed, -total_calls), lexicographic.
pub type Value = (usize, i64);

pub fn execute_policy(world: &World, mut tree: &Node, h: &Config, cap: usize) -> (bool, usize) {
    let mut calls = 0usize;
    loop {
        match tree {
            Node::Stop => return (false, calls),
            Node::Act(action, children) => {
                if calls >= cap {
                    return (false, calls);
                }
                let (resp, commit) = world.effect(action, h);
                calls += 1;
                if commit {
                    return (true, calls);
                }
                match children.iter().find(|(r, _)| *r == resp) {
                    Some((_, nxt)) => tree = nxt,
                    None => return (false, calls),
                }
            }
        }
    }
}

pub fn evaluate_policy(world: &World, tree: &Node, cap: usize) -> (BTreeSet<Config>, usize) {
    let mut done = BTreeSet::new();
    let mut total = 0usize;
    for h in &world.configs {
        let (ok, calls) = execute_policy(world, tree, h, cap);
        total += calls;
        if ok {
            done.insert(h.clone());
        }
    }
    (done, total)
}

/// Partition V by the (known) response of `action`.
/// Only valid when the action is closed on V (or the table is complete).
fn partition(
    table: &EffectTable,
    action: &Act,
    v: &BTreeSet<Config>,
) -> (BTreeSet<Config>, Vec<(Resp, BTreeSet<Config>)>) {
    let mut committed = BTreeSet::new();
    let mut cells: Vec<(Resp, BTreeSet<Config>)> = Vec::new();
    for h in sorted_configs(table.world, v) {
        let (resp, commit) = table.known.get(&(action.clone(), h.clone())).expect("not closed");
        if *commit {
            committed.insert(h.clone());
        } else {
            match cells.iter_mut().find(|(r, _)| *r == *resp) {
                Some((_, cell)) => {
                    cell.insert(h.clone());
                }
                None => cells.push((resp.clone(), {
                    let mut s = BTreeSet::new();
                    s.insert(h.clone());
                    s
                })),
            }
        }
    }
    (committed, cells)
}

fn union_cells(cells: &[(Resp, BTreeSet<Config>)]) -> BTreeSet<Config> {
    let mut u = BTreeSet::new();
    for (_, c) in cells {
        u.extend(c.iter().cloned());
    }
    u
}

// ---------------------------------------------------------------------------
// Exact oracle: full-knowledge DP over belief states (V, b)
// ---------------------------------------------------------------------------

pub fn exact_optimum(world: &World, cap: usize) -> (Option<Value>, Option<Node>) {
    exact_optimum_with(world, cap, &world.actions(), None)
}

pub fn exact_optimum_with(
    world: &World,
    cap: usize,
    actions: &[Act],
    table: Option<&EffectTable>,
) -> (Option<Value>, Option<Node>) {
    let mut owned;
    let table_ref: &EffectTable = match table {
        Some(t) => t,
        None => {
            owned = EffectTable::new(world);
            for a in actions {
                for h in &world.configs {
                    owned.acquire(a, h);
                }
            }
            &owned
        }
    };
    let (r0, _) = evaluate_policy(world, &initial_policy(world, cap), cap);
    let mut memo: HashMap<(BTreeSet<Config>, usize), (Option<Value>, Option<Node>)> = HashMap::new();
    solve_exact(world, table_ref, actions, &r0, &BTreeSet::from_iter(world.configs.iter().cloned()), cap, &mut memo)
}

#[allow(clippy::too_many_arguments)]
fn solve_exact(
    world: &World,
    table: &EffectTable,
    actions: &[Act],
    r0: &BTreeSet<Config>,
    v: &BTreeSet<Config>,
    b: usize,
    memo: &mut HashMap<(BTreeSet<Config>, usize), (Option<Value>, Option<Node>)>,
) -> (Option<Value>, Option<Node>) {
    let key = (v.clone(), b);
    if let Some(res) = memo.get(&key) {
        return res.clone();
    }
    let oblig: BTreeSet<Config> = r0.intersection(v).cloned().collect();
    let mut best: Option<(Value, Node)> = if oblig.is_empty() {
        Some(((0, 0), Node::Stop))
    } else {
        None
    };
    if b >= 1 {
        for a in actions {
            if !table.closed(a, v) {
                continue;
            }
            let (committed, cells) = partition(table, a, v);
            let mut child_results: Vec<(Resp, (Value, Node))> = Vec::new();
            let mut feasible = true;
            let mut comp = committed.len();
            let mut calls = v.len();
            for (resp, cell) in &cells {
                let sub = solve_exact(world, table, actions, r0, cell, b - 1, memo);
                match sub {
                    (Some(val), Some(tree)) => {
                        comp += val.0;
                        calls += (-val.1) as usize;
                        child_results.push((resp.clone(), (val, tree)));
                    }
                    _ => {
                        feasible = false;
                        break;
                    }
                }
            }
            if !feasible {
                continue;
            }
            let covered: BTreeSet<Config> =
                committed.union(&union_cells(&cells)).cloned().collect();
            if !oblig.is_subset(&covered) {
                continue;
            }
            let val = (comp, -(calls as i64));
            let tree = Node::Act(a.clone(), child_results.into_iter().map(|(r, (_, t))| (r, t)).collect());
            if best.as_ref().map_or(true, |(bv, _)| val > *bv) {
                best = Some((val, tree));
            }
        }
    }
    let out = match best {
        Some((val, tree)) => (Some(val), Some(tree)),
        None => (None, None),
    };
    memo.insert(key, out.clone());
    out
}

pub fn initial_policy(world: &World, cap: usize) -> Node {
    let progs: Vec<Act> = world.programs.iter().map(|p| Act::Prog(p.clone())).collect();
    let mut node = Node::Stop;
    for p in progs.iter().take(cap).rev() {
        let children: Vec<(Resp, Node)> = (0..world.aspects.len())
            .map(|i| (Resp::Err(i), node.clone()))
            .collect();
        node = Node::Act(p.clone(), children);
    }
    node
}

// ---------------------------------------------------------------------------
// CAPE compiler
// ---------------------------------------------------------------------------

pub struct CapeResult {
    pub tree: Node,
    pub value: Value,
    pub queries: usize,
    pub certified: bool,
    pub completed: BTreeSet<Config>,
    pub total_calls: usize,
    pub r0: BTreeSet<Config>,
    pub upper: Option<Value>,
    /// Ordered (action, config) pairs acquired in the loop, for audit /
    /// parity tracing (excludes the pi0 validation acquisitions).
    pub acquisitions: Vec<(Act, Config)>,
}

/// Optimistic response cells F_a^+ (paper Sec. 4.2).
fn optimistic_cells(
    world: &World,
    table: &EffectTable,
    action: &Act,
    v: &BTreeSet<Config>,
) -> (BTreeSet<Config>, Vec<(Resp, BTreeSet<Config>)>) {
    let mut committed = BTreeSet::new();
    let mut cells: Vec<(Resp, BTreeSet<Config>)> = Vec::new();
    let mut unknown: Vec<Config> = Vec::new();
    for h in sorted_configs(world, v) {
        match table.known.get(&(action.clone(), h.clone())) {
            Some((resp, commit)) => {
                if *commit {
                    committed.insert(h.clone());
                } else {
                    match cells.iter_mut().find(|(r, _)| *r == *resp) {
                        Some((_, cell)) => {
                            cell.insert(h.clone());
                        }
                        None => cells.push((resp.clone(), {
                            let mut s = BTreeSet::new();
                            s.insert(h.clone());
                            s
                        })),
                    }
                }
            }
            None => unknown.push(h.clone()),
        }
    }
    match action {
        Act::Prog(_) => {
            committed.extend(unknown);
        }
        Act::Probe(_, _, _) => {
            for h in unknown {
                let idx = world.config_index(&h);
                cells.push((Resp::Unknown(idx), {
                    let mut s = BTreeSet::new();
                    s.insert(h);
                    s
                }));
            }
        }
    }
    (committed, cells)
}

pub struct Cape<'a> {
    pub world: &'a World,
    pub cap: usize,
    pub budget: usize,
    pub table: EffectTable<'a>,
    pub queries: usize,
    pub r0: BTreeSet<Config>,
    b_cache: HashMap<(BTreeSet<Config>, usize), Option<Value>>,
    ua_cache: HashMap<(Act, BTreeSet<Config>, usize), Option<Value>>,
    u_cache: HashMap<(BTreeSet<Config>, usize), Option<Value>>,
    t2_cache: HashMap<BTreeSet<Config>, Option<Value>>,
}

impl<'a> Cape<'a> {
    pub fn new(world: &'a World, cap: usize, budget: usize) -> Self {
        let mut c = Cape {
            world,
            cap,
            budget,
            table: EffectTable::new(world),
            queries: 0,
            r0: BTreeSet::new(),
            b_cache: HashMap::new(),
            ua_cache: HashMap::new(),
            u_cache: HashMap::new(),
            t2_cache: HashMap::new(),
        };
        let (r0, _) = evaluate_policy(world, &initial_policy(world, cap), cap);
        c.r0 = r0;
        c.validate_initial();
        c
    }

    fn validate_initial(&mut self) {
        let progs: Vec<Act> = self.world.programs.iter().take(self.cap).map(|p| Act::Prog(p.clone())).collect();
        let mut remaining: Vec<Config> = self.world.configs.clone();
        for p in &progs {
            for h in remaining.clone() {
                let (_, commit) = self.table.acquire(p, &h);
                self.queries += 1;
                if commit {
                    remaining.retain(|x| *x != h);
                }
            }
        }
    }

    /// Witness: best feasible composition from closed actions only.
    pub fn witness(
        &self,
        v: &BTreeSet<Config>,
        b: usize,
        memo: &mut HashMap<(BTreeSet<Config>, usize), (Option<Value>, Option<Node>)>,
    ) -> (Option<Value>, Option<Node>) {
        let key = (v.clone(), b);
        if let Some(res) = memo.get(&key) {
            return res.clone();
        }
        let oblig: BTreeSet<Config> = self.r0.intersection(v).cloned().collect();
        let mut best: Option<(Value, Node)> = if oblig.is_empty() {
            Some(((0, 0), Node::Stop))
        } else {
            None
        };
        if b >= 1 {
            for a in self.world.actions() {
                if !self.table.closed(&a, v) {
                    continue;
                }
                let (committed, cells) = partition(&self.table, &a, v);
                let mut child_results: Vec<(Resp, (Value, Node))> = Vec::new();
                let mut feasible = true;
                let mut comp = committed.len();
                let mut calls = v.len();
                for (resp, cell) in &cells {
                    let sub = self.witness(cell, b - 1, memo);
                    match sub {
                        (Some(val), Some(tree)) => {
                            comp += val.0;
                            calls += (-val.1) as usize;
                            child_results.push((resp.clone(), (val, tree)));
                        }
                        _ => {
                            feasible = false;
                            break;
                        }
                    }
                }
                if !feasible {
                    continue;
                }
                let covered: BTreeSet<Config> =
                    committed.union(&union_cells(&cells)).cloned().collect();
                if !oblig.is_subset(&covered) {
                    continue;
                }
                let val = (comp, -(calls as i64));
                let tree = Node::Act(a.clone(), child_results.into_iter().map(|(r, (_, t))| (r, t)).collect());
                if best.as_ref().map_or(true, |(bv, _)| val > *bv) {
                    best = Some((val, tree));
                }
            }
        }
        let out = match best {
            Some((val, tree)) => (Some(val), Some(tree)),
            None => (None, None),
        };
        memo.insert(key, out.clone());
        out
    }

    /// Cheap upper value B(V, b), Eq. (5); b == 1 special-cased.
    fn bound_b(&mut self, v: &BTreeSet<Config>, b: usize) -> Option<Value> {
        let key = (v.clone(), b);
        if let Some(res) = self.b_cache.get(&key) {
            return *res;
        }
        let vset: BTreeSet<Config> = v.clone();
        let oblig: BTreeSet<Config> = self.r0.intersection(&vset).cloned().collect();
        let mut gbar: Vec<BTreeSet<Config>> = Vec::new();
        for a in self.world.actions() {
            if let Act::Prog(_) = a {
                let mut s = BTreeSet::new();
                for h in &vset {
                    match self.table.known.get(&(a.clone(), h.clone())) {
                        Some((_, commit)) => {
                            if *commit {
                                s.insert(h.clone());
                            }
                        }
                        None => {
                            s.insert(h.clone());
                        }
                    }
                }
                gbar.push(s);
            }
        }
        let union: BTreeSet<Config> = gbar.iter().flat_map(|s| s.iter().cloned()).collect();
        let out = if !oblig.is_subset(&union) {
            None
        } else if union.is_empty() {
            Some((0, 0))
        } else {
            let dbar = gbar.iter().map(|s| s.len()).max().unwrap_or(0);
            if b <= 1 {
                Some((dbar, -(vset.len() as i64)))
            } else {
                Some((union.len(), -((vset.len() + union.len() - dbar) as i64)))
            }
        };
        self.b_cache.insert(key, out);
        out
    }

    /// Upper value U_a(V, b), Eq. (6), over optimistic cells.
    fn action_upper(&mut self, a: &Act, v: &BTreeSet<Config>, b: usize, use_t2_children: bool) -> Option<Value> {
        let key = (a.clone(), v.clone(), b);
        if let Some(res) = self.ua_cache.get(&key) {
            return *res;
        }
        let (committed, cells) = optimistic_cells(self.world, &self.table, a, v);
        let oblig: BTreeSet<Config> = self.r0.intersection(v).cloned().collect();
        let covered: BTreeSet<Config> = committed.union(&union_cells(&cells)).cloned().collect();
        let out = if !oblig.is_subset(&covered) {
            None
        } else {
            let mut comp = committed.len();
            let mut negcalls = -(v.len() as i64);
            let mut ok = true;
            for cell in cells.iter().map(|(_, c)| c) {
                let sub = if b <= 1 {
                    Some((0, 0))
                } else {
                    self.upper(cell, b - 1, use_t2_children)
                };
                match sub {
                    Some(sv) => {
                        comp += sv.0;
                        negcalls += sv.1;
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                Some((comp, negcalls))
            } else {
                None
            }
        };
        self.ua_cache.insert(key, out);
        out
    }

    /// Upper value U(V, b): min(B, max_a U_a); b == 2 uses T2.
    fn upper(&mut self, v: &BTreeSet<Config>, b: usize, use_t2_children: bool) -> Option<Value> {
        let key = (v.clone(), b);
        if let Some(res) = self.u_cache.get(&key) {
            return *res;
        }
        let bval = self.bound_b(v, b);
        let out = match bval {
            None => None,
            Some(bv) => {
                if b <= 1 {
                    Some(bv)
                } else if b == 2 {
                    self.tighten2(v)
                } else {
                    let mut best: Option<Value> = None;
                    for a in self.world.actions() {
                        if let Some(ua) = self.action_upper(&a, v, b, use_t2_children) {
                            if best.map_or(true, |cur| ua > cur) {
                                best = Some(ua);
                            }
                        }
                    }
                    match best {
                        None => Some(bv),
                        Some(bestv) => Some(if bv < bestv { bv } else { bestv }),
                    }
                }
            }
        };
        self.u_cache.insert(key, out);
        out
    }

    /// T2(V): exact optimum of the relaxed two-call problem, Eq. (7).
    fn tighten2(&mut self, v: &BTreeSet<Config>) -> Option<Value> {
        let key = v.clone();
        if let Some(res) = self.t2_cache.get(&key) {
            return *res;
        }
        let oblig: BTreeSet<Config> = self.r0.intersection(v).cloned().collect();
        let bval = self.bound_b(v, 2);
        let out = match bval {
            None => None,
            Some(bv) => {
                let mut best: Option<Value> = if oblig.is_empty() { Some((0, 0)) } else { None };
                for a in self.world.actions() {
                    let (committed, cells) = optimistic_cells(self.world, &self.table, &a, v);
                    let covered: BTreeSet<Config> =
                        committed.union(&union_cells(&cells)).cloned().collect();
                    if !oblig.is_subset(&covered) {
                        continue;
                    }
                    let mut comp = committed.len();
                    let mut negcalls = -(v.len() as i64);
                    let mut ok = true;
                    for cell in cells.iter().map(|(_, c)| c) {
                        match self.bound_b(cell, 1) {
                            Some(sv) => {
                                comp += sv.0;
                                negcalls += sv.1;
                            }
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let val = (comp, negcalls);
                    if best.map_or(true, |cur| val > cur) {
                        best = Some(val);
                    }
                }
                match best {
                    None => Some(bv),
                    Some(bestv) => Some(if bv < bestv { bv } else { bestv }),
                }
            }
        };
        self.t2_cache.insert(key, out);
        out
    }

    /// States reached by the current executable policy, plus states
    /// along one best optimistic alternative from the root.
    fn frontier_states(&mut self, root_witness_tree: &Node) -> Vec<(BTreeSet<Config>, usize)> {
        let mut states: Vec<(BTreeSet<Config>, usize)> = Vec::new();
        fn walk(
            cape: &Cape,
            tree: &Node,
            v: &BTreeSet<Config>,
            b: usize,
            states: &mut Vec<(BTreeSet<Config>, usize)>,
        ) {
            if b == 0 {
                return;
            }
            if !states.iter().any(|(sv, sb)| sv == v && *sb == b) {
                states.push((v.clone(), b));
            }
            let (a, children) = match tree {
                Node::Act(a, children) => (a, children),
                _ => return,
            };
            if !cape.table.closed(a, v) {
                return;
            }
            let (_, cells) = partition(&cape.table, a, v);
            for (resp, cell) in &cells {
                if let Some((_, child)) = children.iter().find(|(r, _)| *r == *resp) {
                    walk(cape, child, cell, b - 1, states);
                }
            }
        }
        let root_v: BTreeSet<Config> = self.world.configs.iter().cloned().collect();
        walk(self, root_witness_tree, &root_v, self.cap, &mut states);
        // One best optimistic alternative from the root.
        let mut v = root_v;
        let mut b = self.cap;
        while b >= 2 {
            let mut best_a: Option<Act> = None;
            let mut best_u: Option<Value> = None;
            for a in self.world.actions() {
                if let Some(ua) = self.action_upper(&a, &v, b, false) {
                    if best_u.map_or(true, |cur| ua > cur) {
                        best_a = Some(a);
                        best_u = Some(ua);
                    }
                }
            }
            let best_a = match best_a {
                Some(a) => a,
                None => break,
            };
            if !states.iter().any(|(sv, sb)| sv == &v && *sb == b) {
                states.push((v.clone(), b));
            }
            let (_, cells) = optimistic_cells(self.world, &self.table, &best_a, &v);
            if cells.is_empty() {
                break;
            }
            // Largest optimistic cell; first on ties (Python max()).
            let mut largest: Option<&BTreeSet<Config>> = None;
            for (_, cell) in &cells {
                if largest.map_or(true, |cur| cell.len() > cur.len()) {
                    largest = Some(cell);
                }
            }
            v = largest.unwrap().clone();
            b -= 1;
            if !states.iter().any(|(sv, sb)| sv == &v && *sb == b) {
                states.push((v.clone(), b));
            }
        }
        states
    }

    /// Three-tier ranking of (action, state) acquisitions.
    /// Returns (tier, score, action, V, b), sorted by (tier, -score).
    fn ranked_candidates(
        &mut self,
        frontier: &[(BTreeSet<Config>, usize)],
    ) -> Vec<(u8, f64, Act, BTreeSet<Config>, usize)> {
        let mut cands: Vec<(u8, f64, Act, BTreeSet<Config>, usize)> = Vec::new();
        let mut wit_memo: HashMap<(BTreeSet<Config>, usize), (Option<Value>, Option<Node>)> = HashMap::new();
        for (v, b) in frontier {
            let (wit_val, _) = self.witness(v, *b, &mut wit_memo);
            for a in self.world.actions() {
                let missing = self.table.missing(&a, v);
                if missing.is_empty() {
                    continue;
                }
                let ua = match self.action_upper(&a, v, *b, false) {
                    Some(u) => u,
                    None => continue,
                };
                match wit_val {
                    None => {
                        let score = (self.r0.intersection(v).count() as f64) / (missing.len() as f64);
                        cands.push((1, score, a, v.clone(), *b));
                    }
                    Some(wv) => {
                        if ua.0 > wv.0 {
                            let score = ((ua.0 - wv.0) as f64) / (missing.len() as f64);
                            cands.push((2, score, a, v.clone(), *b));
                        } else if ua.0 == wv.0 && ua.1 > wv.1 {
                            let score = ((ua.1 - wv.1) as f64) / (missing.len() as f64);
                            cands.push((3, score, a, v.clone(), *b));
                        }
                    }
                }
            }
        }
        // Stable sort by (tier asc, score desc), as Python's sort on
        // (tier, -score) with a stable sort.
        cands.sort_by(|x, y| {
            x.0.cmp(&y.0)
                .then(y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        cands
    }

    pub fn run(mut self) -> CapeResult {
        let mut acquisitions: Vec<(Act, Config)> = Vec::new();
        while self.queries < self.budget {
            self.b_cache.clear();
            self.ua_cache.clear();
            self.u_cache.clear();
            self.t2_cache.clear();
            let mut memo: HashMap<(BTreeSet<Config>, usize), (Option<Value>, Option<Node>)> = HashMap::new();
            let root_v: BTreeSet<Config> = self.world.configs.iter().cloned().collect();
            let (root_val_opt, root_tree_opt) = self.witness(&root_v, self.cap, &mut memo);
            let (root_val, root_tree) = match (root_val_opt, root_tree_opt) {
                (Some(v), Some(t)) => (v, t),
                _ => break,
            };
            let upper = self.upper(&root_v, self.cap, false);
            if upper == Some(root_val) {
                break; // certified optimal
            }
            let frontier = self.frontier_states(&root_tree);
            let cands = self.ranked_candidates(&frontier);
            if cands.is_empty() {
                break;
            }
            let (_, _, action, v, _b) = &cands[0];
            let missing = self.table.missing(action, v);
            let required: Vec<Config> = missing.iter().filter(|h| self.r0.contains(*h)).cloned().collect();
            let target = if !required.is_empty() { required[0].clone() } else { missing[0].clone() };
            self.table.acquire(action, &target);
            self.queries += 1;
            acquisitions.push((action.clone(), target));
        }
        self.b_cache.clear();
        self.ua_cache.clear();
        self.u_cache.clear();
        self.t2_cache.clear();
        let mut memo: HashMap<(BTreeSet<Config>, usize), (Option<Value>, Option<Node>)> = HashMap::new();
        let root_v: BTreeSet<Config> = self.world.configs.iter().cloned().collect();
        let (mut root_val, mut root_tree) = self.witness(&root_v, self.cap, &mut memo);
        let upper = self.upper(&root_v, self.cap, false);
        let mut certified = matches!((&root_val, &upper), (Some(wv), Some(uv)) if wv == uv);
        if self.table.complete() {
            let (val, tree) = exact_optimum_with(self.world, self.cap, &self.world.actions(), Some(&self.table));
            if let (Some(vv), Some(tt)) = (val, tree) {
                root_val = Some(vv);
                root_tree = Some(tt);
                certified = true;
            }
        }
        let (completed, total_calls) =
            evaluate_policy(self.world, root_tree.as_ref().expect("witness tree"), self.cap);
        CapeResult {
            tree: root_tree.unwrap(),
            value: root_val.unwrap(),
            queries: self.queries,
            certified,
            completed,
            total_calls,
            r0: self.r0.clone(),
            upper,
            acquisitions,
        }
    }
}

pub fn compile_policy(world: &World, cap: usize, budget: usize) -> CapeResult {
    Cape::new(world, cap, budget).run()
}

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

pub fn program_only_policy(world: &World, cap: usize) -> (Option<Value>, Option<Node>) {
    let progs: Vec<Act> = world.programs.iter().map(|p| Act::Prog(p.clone())).collect();
    exact_optimum_with(world, cap, &progs, None)
}

/// Deterministic proxy for the paper's LLM error-feedback reviser.
/// (The Python original creates an unused `random.Random(seed)`; the
/// reviser itself is fully deterministic, so no RNG is needed here.)
pub fn error_feedback_policy(world: &World, cap: usize) -> (BTreeSet<Config>, usize) {
    let default: Config = world.aspects.iter().map(|_| 0u8).collect();
    let mut completed = BTreeSet::new();
    let mut total_calls = 0usize;
    for h in &world.configs {
        let mut enc = default.clone();
        let mut ok = false;
        for _ in 0..cap {
            total_calls += 1;
            let (resp, commit) = world.effect(&Act::Prog(enc.clone()), h);
            if commit {
                ok = true;
                break;
            }
            match resp {
                Resp::Err(idx) => {
                    let nvals = world.aspects[idx].values.len();
                    enc[idx] = (enc[idx] + 1) % (nvals as u8);
                }
                _ => break,
            }
        }
        if ok {
            completed.insert(h.clone());
        }
    }
    (completed, total_calls)
}
