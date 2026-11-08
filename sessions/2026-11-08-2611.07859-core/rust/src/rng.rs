//! Deterministic xorshift64star RNG (stdlib-only).
//!
//! Divergence from Python: `corpus.py` uses `random.Random` (Mersenne
//! Twister). The Rust corpus therefore draws *different* entities than the
//! Python corpus. The templates, draw sequence structure, and mechanism
//! behaviour are identical; only the random stream differs. See
//! `rust/README.md` for the full divergence list.

/// xorshift64star generator. Seeded identically to the Python corpus
/// (`SEED = 20261108`) but producing a different stream by design.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform value in `[0, n)`.
    ///
    /// Divergence: uses modulo reduction (tiny bias for non-power-of-two
    /// `n`); Python's `randbelow` is rejection-based and unbiased. The
    /// bias is irrelevant for corpus generation and the parity sweep
    /// (which does not use this RNG at all).
    pub fn next_below(&mut self, n: usize) -> usize {
        assert!(n > 0, "next_below requires n > 0");
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates shuffle, same structure as Python's `random.shuffle`
    /// (draw `j` in `[0, i]` for `i` from `len-1` down to `1`).
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        let len = xs.len();
        for i in (1..len).rev() {
            let j = self.next_below(i + 1);
            xs.swap(i, j);
        }
    }
}
