//! WAAC (arXiv 2611.08865) — Rust port.
//!
//! Trajectory compression as field-level fidelity allocation: typed
//! Trace-to-DAG, online backward dependency analysis, verified-only
//! history library with TF-IDF coarse matching, conservative fusion,
//! and context projection. Std-only, deterministic, no network.
//!
//! Python reference: `../waac.py`, `../corpus.py`, `../test_waac_cpu.py`
//! in this session directory.

pub mod corpus;
pub mod executor;
pub mod json;
pub mod mt19937;
pub mod policy;
pub mod sha256;
pub mod waac;

/// CPython `sum()` over floats (Neumaier compensated summation, since
/// 3.12). Naive left-to-right addition differs from it in the last ulp
/// on the aggregate ratios, so every place the Python reference sums
/// floats must use this: `aggregate`, the TF-IDF norm, and the cosine.
pub fn py_sum(items: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for x in items {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}
