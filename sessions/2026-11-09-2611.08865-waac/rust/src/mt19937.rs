//! CPython-compatible Mersenne Twister (`random.Random`).
//!
//! The corpus generator (`corpus.py`) is seeded with
//! `random.Random(20261109)` / `random.Random(777)`, and the fusion
//! property check uses `random.Random(99)`. To reproduce the exact same
//! workflows (and hence the exact same measured token counts) the Rust
//! port must reproduce CPython's `random` bit-for-bit: MT19937 with
//! `init_by_array` seeding from the int seed's little-endian 32-bit
//! words, `random()` from two successive `genrand_uint32` outputs,
//! `_randbelow` rejection sampling, and the Fisher-Yates `shuffle`.
//!
//! Reference values (from CPython 3.12, seed 20261109):
//! `random()` -> 0.18367155676787084, 0.489810059146196, 0.6190994094165755;
//! see `tests/unit.rs::mt19937_matches_python`.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;

/// A `random.Random(seed)` equivalent for integer seeds.
#[derive(Clone, Debug)]
pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// Seed like `random.Random(seed)` for a non-negative integer seed.
    /// CPython converts the int to little-endian 32-bit words and calls
    /// `init_by_array`.
    pub fn new(seed: u64) -> Self {
        let mut key: Vec<u32> = vec![seed as u32];
        if seed > u32::MAX as u64 {
            key.push((seed >> 32) as u32);
        }
        let mut r = PyRandom {
            mt: [0; N],
            index: N + 1,
        };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for j in 1..N {
            self.mt[j] = 1812433253u32
                .wrapping_mul(self.mt[j - 1] ^ (self.mt[j - 1] >> 30))
                .wrapping_add(j as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, init_key: &[u32]) {
        self.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        let key_len = init_key.len();
        let mut k = N.max(key_len);
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525)))
            .wrapping_add(init_key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key_len {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                    .wrapping_mul(1566083941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x80000000;
    }

    fn genrand_u32(&mut self) -> u32 {
        if self.index >= N {
            // Twist.
            for kk in 0..N {
                let y = (self.mt[kk] & UPPER_MASK)
                    | (self.mt[(kk + 1) % N] & LOWER_MASK);
                let mut xa = self.mt[(kk + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    xa ^= MATRIX_A;
                }
                self.mt[kk] = xa;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// `random.random()`: float in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = self.genrand_u32() >> 5;
        let b = self.genrand_u32() >> 6;
        (a as f64 * 67108864.0 + b as f64) / 9007199254740992.0
    }

    /// `random.getrandbits(k)` for 0 < k <= 64.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k > 0 && k <= 64);
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        let words = (k + 31) / 32;
        let mut r: u64 = 0;
        for _ in 0..words {
            r = (r << 32) | self.genrand_u32() as u64;
        }
        r >> (words * 32 - k)
    }

    /// `random._randbelow(n)`: uniform int in [0, n).
    pub fn randbelow(&mut self, n: u32) -> u32 {
        assert!(n > 0);
        let k = 32 - n.leading_zeros();
        let mut r = self.getrandbits(k) as u32;
        while r >= n {
            r = self.getrandbits(k) as u32;
        }
        r
    }

    /// `random.choice(seq)`.
    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        assert!(!seq.is_empty());
        &seq[self.randbelow(seq.len() as u32) as usize]
    }

    /// `random.randint(a, b)` (inclusive).
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        assert!(a <= b);
        let width = (b - a + 1) as u32;
        a + self.randbelow(width) as i64
    }

    /// `random.shuffle(x)` (Fisher-Yates from the end).
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u32 + 1) as usize;
            x.swap(i, j);
        }
    }
}
