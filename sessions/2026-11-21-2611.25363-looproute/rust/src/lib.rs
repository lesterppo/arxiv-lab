//! LoopRoute: loop-triggered routing for System-1 decision agents.
//!
//! Rust port of `arxiv_lab.agent.looproute` (arXiv:2611.25363 — Yu et al.,
//! 2026, "Stop Going in Circles: Loop-Triggered Routing for System 1
//! Decision Agents"). A fast decision policy acts by default; recurring
//! (state, action) pairs signal stalled execution and trigger a slow
//! assistant (here a BFS oracle planner), then control returns to the fast
//! policy. No training, no confidence calibration.
//!
//! The port is behaviorally identical to the Python module, including the
//! seeded task generation: [`PyRandom`] is a faithful re-implementation
//! of CPython's `random.Random` (MT19937 + `init_by_array` seeding,
//! `random()`, `getrandbits`, `_randbelow`, `choice`, `shuffle`,
//! `sample`), so `KitchenWorld::new(seed, ...)` generates the same task
//! as Python for the same seed. Verified against CPython 3.12 values in
//! `tests/unit.rs` and by the `gen-spec` parity check in
//! `tests/parity_sweep.py`.
//!
//! OUT OF SCOPE (not ported): the NVIDIA NIM model-backed leg — it needs
//! network access and a real LLM, so it stays Python-only. The oracle
//! assistant (`plan_to_goal`) is the deterministic stand-in used here.
//!
//! Python stays the primary verdict carrier; this port cross-checks it.
//! A failed Rust leg would never overturn the Python verdict.

use std::collections::{HashMap, HashSet, VecDeque};

// ===========================================================================
// CPython-compatible RNG (MT19937, random.Random for integer seeds)
// Adapted from the grepmem Rust leg (same repo, arXiv:2611.25123).
// ===========================================================================

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

/// A faithful port of CPython's `random.Random` restricted to the
/// operations this mechanism uses. Seeding: an integer seed is split
/// into little-endian 32-bit words and fed to MT19937 `init_by_array`
/// — exactly CPython's version-2 seeding for `int`.
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

    /// CPython `random()`: `(a * 2**26 + b) / 2**53`.
    pub fn random(&mut self) -> f64 {
        let a = (self.gen_u32() >> 5) as f64;
        let b = (self.gen_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// CPython `getrandbits(k)`: the low `k` bits of the result are the
    /// top bits of successive 32-bit words, first word lowest.
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
}

// ===========================================================================
// Environment: KitchenWorld
// ===========================================================================

pub const ROOMS: [&str; 6] = [
    "kitchen", "pantry", "garden", "cellar", "bedroom", "hallway",
];
/// Room indices: kitchen 0, pantry 1, garden 2, cellar 3, bedroom 4,
/// hallway 5. Adjacency lists keep Python's tuple order (BFS tie-breaks
/// and legal-action order depend on it).
pub const ADJ: [&[usize]; 6] = [
    &[1, 5, 2], // kitchen -> pantry, hallway, garden
    &[0, 3],    // pantry -> kitchen, cellar
    &[0, 5],    // garden -> kitchen, hallway
    &[1, 4],    // cellar -> pantry, bedroom
    &[3, 5],    // bedroom -> cellar, hallway
    &[0, 2, 4], // hallway -> kitchen, garden, bedroom
];

pub const INGREDIENTS: [&str; 10] = [
    "tomato", "onion", "garlic", "pepper", "carrot", "potato", "basil",
    "lemon", "mushroom", "chili",
];

pub const CONTAINER: &str = "chest";
pub const KEY: &str = "brass key";
/// Inventory bit reserved for the key (ingredient bits are 0..9).
pub const KEY_BIT: u16 = 1 << 10;

pub fn room_index(name: &str) -> Option<usize> {
    ROOMS.iter().position(|&r| r == name)
}

pub fn ingredient_index(name: &str) -> Option<usize> {
    INGREDIENTS.iter().position(|&i| i == name)
}

/// Shortest room path (inclusive) from `start` to `goal`, or `None`.
/// Mirrors Python `bfs_path`, including the early exit on discovering
/// the goal while expanding a room (same tie-breaking).
pub fn bfs_path(start: usize, goal: usize) -> Option<Vec<usize>> {
    if start == goal {
        return Some(vec![start]);
    }
    let mut prev: HashMap<usize, Option<usize>> = HashMap::new();
    prev.insert(start, None);
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(start);
    while let Some(cur) = q.pop_front() {
        for &nxt in ADJ[cur] {
            if !prev.contains_key(&nxt) {
                prev.insert(nxt, Some(cur));
                if nxt == goal {
                    let mut path = vec![goal];
                    while let Some(Some(p)) = prev.get(path.last().unwrap()) {
                        path.push(*p);
                    }
                    path.reverse();
                    return Some(path);
                }
                q.push_back(nxt);
            }
        }
    }
    None
}

/// Hashable env state — the router's loop key. Field order mirrors
/// Python's state tuple: (room, inventory, collected, chest_open,
/// key_taken, taken).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EnvState {
    pub room: usize,
    pub inventory: u16,
    pub collected: u16,
    pub chest_open: bool,
    pub key_taken: bool,
    pub taken: u16,
}

impl EnvState {
    /// Canonical string form, matching the Python side's rendering of
    /// the state tuple (used by the parity sweep).
    pub fn canonical(&self) -> String {
        let names = |bits: u16| -> String {
            let mut v: Vec<&str> = Vec::new();
            for i in 0..10 {
                if bits & (1 << i) != 0 {
                    v.push(INGREDIENTS[i]);
                }
            }
            if bits & KEY_BIT != 0 {
                v.push(KEY);
            }
            v.sort_unstable();
            v.join(",")
        };
        format!(
            "{}|inv={}|col={}|chest={}|key={}|taken={}",
            ROOMS[self.room],
            names(self.inventory),
            names(self.collected),
            self.chest_open as u8,
            self.key_taken as u8,
            names(self.taken),
        )
    }
}

/// Deterministic cooking task: collect the recipe ingredients (one
/// inside a locked chest that needs a key found elsewhere), then cook
/// in the kitchen.
pub struct KitchenWorld {
    pub seed: u64,
    /// Recipe ingredient indices, in recipe order.
    pub recipe: Vec<usize>,
    /// Room per recipe position (kitchen for the chest ingredient).
    pub placement: Vec<usize>,
    /// Per recipe position: is the ingredient inside the locked chest?
    pub in_chest: Vec<bool>,
    pub key_room: usize,
    pub locked: bool,
    // Mutable state.
    pub room: usize,
    pub inventory: u16,
    pub collected: u16,
    pub chest_open: bool,
    pub key_taken: bool,
    pub taken: u16,
    pub steps: usize,
}

impl KitchenWorld {
    /// Seeded task generation, identical to Python `KitchenWorld(seed,
    /// n_ingredients, locked)`: `sample` the recipe, `shuffle` the
    /// non-kitchen rooms, place ingredients, `choice` the key room.
    pub fn new(seed: u64, n_ingredients: usize, locked: bool) -> Self {
        assert!((1..=10).contains(&n_ingredients), "n_ingredients 1..=10");
        let mut rng = PyRandom::new(seed);
        let recipe = rng.sample_indices(INGREDIENTS.len(), n_ingredients);
        let mut rooms: Vec<usize> = vec![1, 2, 3, 4, 5];
        rng.shuffle(&mut rooms);
        let mut placement = vec![0usize; n_ingredients];
        let mut in_chest = vec![false; n_ingredients];
        for i in 0..n_ingredients {
            if locked && i == n_ingredients - 1 {
                placement[i] = 0; // kitchen
                in_chest[i] = true;
            } else {
                placement[i] = rooms[i % rooms.len()];
            }
        }
        let used: HashSet<usize> = placement.iter().cloned().collect();
        let candidates: Vec<usize> =
            rooms.iter().cloned().filter(|r| !used.contains(r)).collect();
        // Python: rng.choice([r for r in rooms if ...] or ["bedroom"]) —
        // the fallback still consumes one randbelow(1) draw.
        let key_room = if candidates.is_empty() {
            let _ = rng.randbelow(1);
            4 // bedroom
        } else {
            *rng.choice(&candidates)
        };
        let mut env = Self {
            seed,
            recipe,
            placement,
            in_chest,
            key_room,
            locked,
            room: 0,
            inventory: 0,
            collected: 0,
            chest_open: false,
            key_taken: false,
            taken: 0,
            steps: 0,
        };
        env.reset();
        env
    }

    /// Build an env from an explicit task spec (used by the parity
    /// sweep; seed is irrelevant here — see `clone_task`). The spec
    /// lists a `place` line per non-chest ingredient (recipe order)
    /// and a `chest <ing>` line for the chest ingredient, if any.
    pub fn from_spec(spec: &TaskSpec) -> Result<Self, String> {
        let mut recipe = Vec::new();
        let mut placement = Vec::new();
        let mut in_chest = Vec::new();
        for name in &spec.recipe {
            let ii = ingredient_index(name)
                .ok_or_else(|| format!("unknown ingredient {name}"))?;
            recipe.push(ii);
            if Some(name.as_str()) == spec.chest.as_deref() {
                placement.push(0); // kitchen
                in_chest.push(true);
            } else {
                let (_, room) = spec
                    .placement
                    .iter()
                    .find(|(ing, _)| ing == name)
                    .ok_or_else(|| format!("no placement for {name}"))?;
                placement.push(
                    room_index(room).ok_or_else(|| format!("unknown room {room}"))?,
                );
                in_chest.push(false);
            }
        }
        if spec.placement.len() + usize::from(spec.chest.is_some()) != spec.recipe.len() {
            return Err("spec placement/recipe length mismatch".to_string());
        }
        let key_room = room_index(&spec.key_room)
            .ok_or_else(|| format!("unknown key room {}", spec.key_room))?;
        let mut env = Self {
            seed: u64::MAX,
            recipe,
            placement,
            in_chest,
            key_room,
            locked: spec.locked,
            room: 0,
            inventory: 0,
            collected: 0,
            chest_open: false,
            key_taken: false,
            taken: 0,
            steps: 0,
        };
        env.reset();
        Ok(env)
    }

    /// A fresh env with the same task definition (used by
    /// `plan_to_goal` as its search probe). Python rebuilds the probe
    /// from the seed; copying the task fields is the same task, and it
    /// also works for from-spec envs, where no seed exists.
    pub fn clone_task(&self) -> Self {
        let mut env = Self {
            seed: self.seed,
            recipe: self.recipe.clone(),
            placement: self.placement.clone(),
            in_chest: self.in_chest.clone(),
            key_room: self.key_room,
            locked: self.locked,
            room: 0,
            inventory: 0,
            collected: 0,
            chest_open: false,
            key_taken: false,
            taken: 0,
            steps: 0,
        };
        env.reset();
        env
    }

    pub fn reset(&mut self) -> EnvState {
        self.room = 0;
        self.inventory = 0;
        self.collected = 0;
        self.chest_open = false;
        self.key_taken = false;
        self.taken = 0;
        self.steps = 0;
        self.state()
    }

    pub fn state(&self) -> EnvState {
        EnvState {
            room: self.room,
            inventory: self.inventory,
            collected: self.collected,
            chest_open: self.chest_open,
            key_taken: self.key_taken,
            taken: self.taken,
        }
    }

    pub fn set_state(&mut self, s: &EnvState) {
        self.room = s.room;
        self.inventory = s.inventory;
        self.collected = s.collected;
        self.chest_open = s.chest_open;
        self.key_taken = s.key_taken;
        self.taken = s.taken;
    }

    fn has(&self, bits: u16, ing: usize) -> bool {
        bits & (1 << ing) != 0
    }

    pub fn recipe_collected(&self) -> bool {
        self.recipe.iter().all(|&ing| self.has(self.collected, ing))
    }

    /// Human-readable state for the LLM assistant prompt (ported for
    /// fidelity; unused by the oracle leg).
    pub fn describe(&self) -> String {
        let here: Vec<&str> = self
            .recipe
            .iter()
            .enumerate()
            .filter(|(pos, _)| {
                self.placement[*pos] == self.room && !self.has(self.collected, self.recipe[*pos])
            })
            .map(|(_, &ing)| INGREDIENTS[ing])
            .collect();
        let mut lines = vec![
            format!("You are in the {}.", ROOMS[self.room]),
            format!(
                "Recipe (in order): {}.",
                self.recipe
                    .iter()
                    .map(|&ing| INGREDIENTS[ing])
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!(
                "Collected: {}.",
                {
                    let mut v: Vec<&str> = self
                        .recipe
                        .iter()
                        .filter(|&&ing| self.has(self.collected, ing))
                        .map(|&ing| INGREDIENTS[ing])
                        .collect();
                    v.sort_unstable();
                    if v.is_empty() {
                        "none".to_string()
                    } else {
                        v.join(", ")
                    }
                }
            ),
            format!(
                "Carrying: {}.",
                {
                    let mut v: Vec<&str> = Vec::new();
                    for i in 0..10 {
                        if self.has(self.inventory, i) {
                            v.push(INGREDIENTS[i]);
                        }
                    }
                    if self.inventory & KEY_BIT != 0 {
                        v.push(KEY);
                    }
                    v.sort_unstable();
                    if v.is_empty() {
                        "nothing".to_string()
                    } else {
                        v.join(", ")
                    }
                }
            ),
        ];
        if !here.is_empty() {
            lines.push(format!("Visible here: {}.", here.join(", ")));
        }
        if self.room == 0 && self.locked {
            lines.push(format!(
                "A locked chest is here. {}",
                if self.chest_open {
                    "It is open."
                } else {
                    "It is locked; it needs the brass key."
                }
            ));
        }
        if self.room == self.key_room && !self.key_taken {
            lines.push("A brass key is visible here.".to_string());
        }
        lines.join(" ")
    }

    /// Legal actions, in Python's order: go (adjacency order), take
    /// (recipe order), open chest, take key, cook.
    pub fn legal_actions(&self) -> Vec<String> {
        let mut acts = Vec::new();
        for &nxt in ADJ[self.room] {
            acts.push(format!("go {}", ROOMS[nxt]));
        }
        for (pos, &ing) in self.recipe.iter().enumerate() {
            if self.placement[pos] == self.room && !self.has(self.collected, ing) {
                if pos == self.recipe.len() - 1 && self.locked && !self.chest_open {
                    continue; // inside the locked chest
                }
                acts.push(format!("take {}", INGREDIENTS[ing]));
            }
        }
        if self.room == 0 && self.locked && !self.chest_open {
            acts.push("open chest".to_string());
        }
        if self.room == self.key_room && !self.key_taken {
            acts.push(format!("take {KEY}"));
        }
        if self.room == 0 && self.recipe_collected() {
            acts.push("cook".to_string());
        }
        acts
    }

    /// One step. Returns (observation, reward, done, ok) with the exact
    /// observation strings Python produces. Invalid actions leave the
    /// state unchanged and report ok=false.
    pub fn step(&mut self, action: &str) -> (String, i32, bool, bool) {
        self.steps += 1;
        let (verb, arg) = match action.split_once(' ') {
            Some((v, a)) => (v, a),
            None => (action, ""),
        };
        if verb == "go" {
            if let Some(dest) = room_index(arg) {
                if ADJ[self.room].contains(&dest) {
                    self.room = dest;
                    return (format!("You go to the {arg}."), 0, false, true);
                }
            }
            return ("Unknown action.".to_string(), 0, false, false);
        }
        if verb == "take" {
            if arg == KEY && self.room == self.key_room && !self.key_taken {
                self.key_taken = true;
                self.inventory |= KEY_BIT;
                return ("You take the brass key.".to_string(), 0, false, true);
            }
            if let Some(ing) = ingredient_index(arg) {
                if let Some(pos) = self.recipe.iter().position(|&i| i == ing) {
                    if self.placement[pos] == self.room && !self.has(self.collected, ing) {
                        if pos == self.recipe.len() - 1 && self.locked && !self.chest_open {
                            return ("The chest is locked.".to_string(), 0, false, false);
                        }
                        self.collected |= 1 << ing;
                        self.taken |= 1 << ing;
                        self.inventory |= 1 << ing;
                        return (format!("You take the {arg}."), 0, false, true);
                    }
                }
            }
            return ("You cannot take that here.".to_string(), 0, false, false);
        }
        if verb == "open" && arg == CONTAINER {
            if self.room != 0 {
                return ("There is no chest here.".to_string(), 0, false, false);
            }
            if self.chest_open {
                return ("The chest is already open.".to_string(), 0, false, false);
            }
            if self.inventory & KEY_BIT == 0 {
                return (
                    "The chest is locked. You need the brass key.".to_string(),
                    0,
                    false,
                    false,
                );
            }
            self.chest_open = true;
            return ("You unlock and open the chest.".to_string(), 0, false, true);
        }
        if verb == "cook" {
            if self.room == 0 && self.recipe_collected() {
                return ("You cook the meal. Task complete!".to_string(), 1, true, true);
            }
            return ("You cannot cook yet.".to_string(), 0, false, false);
        }
        ("Unknown action.".to_string(), 0, false, false)
    }
}

// ===========================================================================
// Fast policy (System-1 stand-in) and the LoopRoute router
// ===========================================================================

/// Greedy heuristic with a realistic System-1 flaw: when the next
/// needed ingredient is inside the locked chest and the key is not
/// held, it keeps trying `open chest` instead of detouring to fetch
/// the key — the (state, "open chest") pair therefore recurs.
pub struct FastPolicy;

impl FastPolicy {
    pub fn act(&self, env: &KitchenWorld) -> String {
        if env.recipe_collected() {
            if env.room == 0 {
                return "cook".to_string();
            }
            let path = bfs_path(env.room, 0).expect("kitchen reachable");
            return format!("go {}", ROOMS[path[1]]);
        }
        let nxt_pos = env
            .recipe
            .iter()
            .position(|&ing| !env.has(env.collected, ing))
            .expect("an uncollected ingredient exists");
        if nxt_pos == env.recipe.len() - 1 && env.locked && !env.chest_open {
            if env.room == 0 {
                return "open chest".to_string(); // the flaw: no key check
            }
            let path = bfs_path(env.room, 0).expect("kitchen reachable");
            return format!("go {}", ROOMS[path[1]]);
        }
        let room = env.placement[nxt_pos];
        if env.room == room {
            return format!("take {}", INGREDIENTS[env.recipe[nxt_pos]]);
        }
        let path = bfs_path(env.room, room).expect("room reachable");
        format!("go {}", ROOMS[path[1]])
    }
}

/// LoopRoute router (paper §3). Counts (state, action) pairs within an
/// episode. When the fast policy proposes a pair for the
/// `threshold`-th time, control passes to the assistant; after an
/// assist that pair's count resets, so a later recurrence can trigger
/// assistance again.
pub struct LoopRouter {
    pub threshold: usize,
    pub counts: HashMap<(EnvState, String), usize>,
    pub assists: usize,
}

impl LoopRouter {
    pub fn new(threshold: usize) -> Self {
        assert!(threshold >= 1, "threshold must be >= 1");
        Self {
            threshold,
            counts: HashMap::new(),
            assists: 0,
        }
    }

    pub fn reset(&mut self) {
        self.counts.clear();
        self.assists = 0;
    }

    /// True when this (state, action) pair has recurred to threshold.
    pub fn should_assist(&self, state: &EnvState, action: &str) -> bool {
        self.counts
            .get(&(*state, action.to_string()))
            .copied()
            .unwrap_or(0)
            >= self.threshold - 1
    }

    pub fn note(&mut self, state: &EnvState, action: &str) {
        *self
            .counts
            .entry((*state, action.to_string()))
            .or_insert(0) += 1;
    }

    /// Record an assist for (state, action); reset its count.
    pub fn assist(
        &mut self,
        state: &EnvState,
        action: &str,
        plan: Vec<String>,
    ) -> Vec<String> {
        self.assists += 1;
        self.counts.remove(&(*state, action.to_string()));
        plan
    }
}

// ===========================================================================
// Oracle planner (deterministic stand-in for the paper's LLM assistant)
// ===========================================================================

/// BFS over the full state space from `env.state()` to a cooked meal.
/// Returns a list of actions, or `[]` if unreachable. Mirrors Python
/// `plan_to_goal`, including the explicit goal-edge recording: cooking
/// does not change the state, so the final "cook" action must be
/// recorded explicitly or the reconstructed plan would omit it.
pub fn plan_to_goal(env: &KitchenWorld) -> Vec<String> {
    let start = env.state();
    let mut probe = env.clone_task();
    probe.set_state(&start);
    let mut prev: HashMap<EnvState, (Option<EnvState>, Option<String>)> =
        HashMap::new();
    prev.insert(start, (None, None));
    let mut q: VecDeque<EnvState> = VecDeque::new();
    q.push_back(start);
    let mut goal: Option<(EnvState, String)> = None;
    while let Some(cur) = q.pop_front() {
        probe.set_state(&cur);
        for act in probe.legal_actions() {
            probe.set_state(&cur);
            probe.steps = 0;
            let (_, _, done, _) = probe.step(&act);
            if done {
                goal = Some((cur, act));
                break;
            }
            let ns = probe.state();
            if prev.contains_key(&ns) {
                continue;
            }
            prev.insert(ns, (Some(cur), Some(act)));
            q.push_back(ns);
        }
        if goal.is_some() {
            break;
        }
    }
    let (goal_state, goal_action) = match goal {
        Some(g) => g,
        None => return Vec::new(),
    };
    let mut plan = vec![goal_action];
    let mut cur = goal_state;
    while let Some((Some(parent), Some(act))) = prev.get(&cur).cloned() {
        plan.push(act);
        cur = parent;
    }
    plan.reverse();
    plan
}

/// Deterministic BFS plan from the current state to the goal.
pub fn oracle_assistant(env: &KitchenWorld) -> Vec<String> {
    plan_to_goal(env)
}

// ===========================================================================
// Episode runner
// ===========================================================================

/// One step of an episode trajectory.
#[derive(Clone, Debug)]
pub struct TrajStep {
    pub state: EnvState,
    pub action: String,
    pub source: String,
    pub obs: String,
}

/// Outcome of one task run.
#[derive(Clone, Debug)]
pub struct EpisodeResult {
    pub success: bool,
    pub steps: usize,
    pub assists: usize,
    pub llm_calls: usize,
    pub trajectory: Vec<TrajStep>,
}

/// Run one task. `assistant: None` means no assistant is available
/// (a trigger then executes the fast action once and keeps counting).
/// Mirrors Python `run_episode` step for step, including the
/// empty-plan-assist branch.
pub fn run_episode(
    env: &mut KitchenWorld,
    policy: &FastPolicy,
    router: Option<&mut LoopRouter>,
    assistant: Option<fn(&KitchenWorld) -> Vec<String>>,
    max_steps: usize,
) -> EpisodeResult {
    env.reset();
    let mut router = router;
    if let Some(r) = router.as_mut() {
        r.reset();
    }
    let mut traj: Vec<TrajStep> = Vec::new();
    let mut plan: Vec<String> = Vec::new();
    let mut llm_calls = 0usize;
    while env.steps < max_steps {
        let state = env.state();
        let (action, source) = if !plan.is_empty() {
            (plan.remove(0), "plan".to_string())
        } else {
            let mut act = policy.act(env);
            let mut src = "fast".to_string();
            let mut done_early = false;
            let mut early_result: Option<EpisodeResult> = None;
            if let Some(r) = router.as_mut() {
                if r.should_assist(&state, &act) {
                    if let Some(assistant) = assistant {
                        llm_calls += 1;
                        let p = r.assist(&state, &act, assistant(env));
                        if !p.is_empty() {
                            plan = p;
                            act = plan.remove(0);
                            src = "assist".to_string();
                        } else {
                            // Assistant produced nothing: count the
                            // attempt, execute the fast action once
                            // and keep counting.
                            r.note(&state, &act);
                            let (obs, _, done, _) = env.step(&act);
                            traj.push(TrajStep {
                                state,
                                action: act.clone(),
                                source: src.clone(),
                                obs,
                            });
                            if done {
                                let assists =
                                    router.as_ref().map(|r| r.assists).unwrap_or(0);
                                early_result = Some(EpisodeResult {
                                    success: true,
                                    steps: env.steps,
                                    assists,
                                    llm_calls,
                                    trajectory: traj.clone(),
                                });
                            }
                            done_early = true;
                        }
                    }
                }
            }
            if done_early {
                if let Some(res) = early_result {
                    return res;
                }
                continue;
            }
            (act, src)
        };
        if let Some(r) = router.as_mut() {
            r.note(&state, &action);
        }
        let (obs, _, done, _) = env.step(&action);
        traj.push(TrajStep {
            state,
            action,
            source,
            obs,
        });
        if done {
            let assists = router.as_ref().map(|r| r.assists).unwrap_or(0);
            return EpisodeResult {
                success: true,
                steps: env.steps,
                assists,
                llm_calls,
                trajectory: traj,
            };
        }
    }
    let assists = router.as_ref().map(|r| r.assists).unwrap_or(0);
    EpisodeResult {
        success: false,
        steps: env.steps,
        assists,
        llm_calls,
        trajectory: traj,
    }
}

/// Aggregate over a set of episode results.
#[derive(Clone, Debug, PartialEq)]
pub struct Aggregate {
    pub n: usize,
    pub success_rate: f64,
    pub mean_steps: f64,
    pub mean_assists: f64,
    pub mean_llm_calls: f64,
}

fn aggregate(rs: &[EpisodeResult]) -> Aggregate {
    let n = rs.len();
    Aggregate {
        n,
        success_rate: rs.iter().filter(|r| r.success).count() as f64 / n as f64,
        mean_steps: rs.iter().map(|r| r.steps).sum::<usize>() as f64 / n as f64,
        mean_assists: rs.iter().map(|r| r.assists).sum::<usize>() as f64 / n as f64,
        mean_llm_calls: rs.iter().map(|r| r.llm_calls).sum::<usize>() as f64 / n as f64,
    }
}

/// Fast-only vs LoopRoute-oracle over seeded tasks. Task mix mirrors
/// Python `run_experiment`: even seeds have the locked chest (the fast
/// policy loops on them); odd seeds are unlocked.
pub fn run_experiment(seeds: &[u64], threshold: usize) -> (Aggregate, Aggregate) {
    let policy = FastPolicy;
    let mut fast = Vec::with_capacity(seeds.len());
    let mut routed = Vec::with_capacity(seeds.len());
    for &s in seeds {
        let locked = s % 2 == 0;
        let mut env = KitchenWorld::new(s, 3, locked);
        fast.push(run_episode(&mut env, &policy, None, None, 60));
        let mut router = LoopRouter::new(threshold);
        let mut env2 = KitchenWorld::new(s, 3, locked);
        routed.push(run_episode(
            &mut env2,
            &policy,
            Some(&mut router),
            Some(oracle_assistant),
            60,
        ));
    }
    (aggregate(&fast), aggregate(&routed))
}

// ===========================================================================
// Task specs (parity harness interchange format, line-based v1)
// ===========================================================================

/// An explicit task definition, shared verbatim between the Python and
/// Rust sides of the parity sweep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskSpec {
    /// Recipe ingredient names, in recipe order.
    pub recipe: Vec<String>,
    /// (ingredient, room) per non-chest ingredient, recipe order.
    pub placement: Vec<(String, String)>,
    /// Ingredient inside the locked chest, if any.
    pub chest: Option<String>,
    pub key_room: String,
    pub locked: bool,
}

/// Canonical spec text for an env (what `gen-spec` prints).
pub fn spec_of_env(env: &KitchenWorld) -> String {
    let mut out = String::new();
    out.push_str("# looproute-spec-v1\n");
    out.push_str(&format!("locked={}\n", if env.locked { 1 } else { 0 }));
    out.push_str(&format!(
        "recipe {}\n",
        env.recipe
            .iter()
            .map(|&ing| INGREDIENTS[ing])
            .collect::<Vec<_>>()
            .join(" ")
    ));
    for (pos, &ing) in env.recipe.iter().enumerate() {
        if env.in_chest[pos] {
            out.push_str(&format!("chest {}\n", INGREDIENTS[ing]));
        } else {
            out.push_str(&format!(
                "place {} {}\n",
                INGREDIENTS[ing],
                ROOMS[env.placement[pos]]
            ));
        }
    }
    out.push_str(&format!("key {}\n", ROOMS[env.key_room]));
    out
}

/// Parse canonical spec text back into a [`TaskSpec`].
pub fn parse_spec(text: &str) -> Result<TaskSpec, String> {
    let mut recipe: Vec<String> = Vec::new();
    let mut placement: Vec<(String, String)> = Vec::new();
    let mut chest: Option<String> = None;
    let mut key_room: Option<String> = None;
    let mut locked: Option<bool> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("locked=1") => locked = Some(true),
            Some("locked=0") => locked = Some(false),
            Some("recipe") => {
                recipe = parts.map(|s| s.to_string()).collect();
            }
            Some("place") => {
                let ing = parts.next().ok_or("place needs ingredient")?;
                let room = parts.next().ok_or("place needs room")?;
                placement.push((ing.to_string(), room.to_string()));
            }
            Some("chest") => {
                chest = Some(
                    parts
                        .next()
                        .ok_or("chest needs ingredient")?
                        .to_string(),
                );
            }
            Some("key") => {
                key_room = Some(
                    parts.next().ok_or("key needs room")?.to_string(),
                );
            }
            Some(other) => return Err(format!("unknown spec line: {other}")),
            None => {}
        }
    }
    Ok(TaskSpec {
        recipe,
        placement,
        chest,
        key_room: key_room.ok_or("spec missing key line")?,
        locked: locked.ok_or("spec missing locked line")?,
    })
}
