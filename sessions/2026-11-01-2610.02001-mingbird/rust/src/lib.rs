//! Rust port of the Mingbird agent-harness trio from
//! `lesterppo/arxiv-lab` (arXiv:2610.02001).
//!
//! - [`PrefillBudget`]: byte-level net-zero prefill budget.
//! - [`FinishGate`]: finish gate re-checking every success criterion
//!   against the current environment before accepting completion.
//! - [`LoopDetector`]: signature-level loop detection (exact-repeat
//!   streaks and short repeated cycles).
//!
//! Pure logic, no network, no model calls, no secrets. The Python
//! reference stays the specification: see `README.md` for the parity
//! notes and the `verify` subcommand of the `arxiv-harness` binary,
//! which re-runs the Python test's three experiments in Rust.

pub mod finish_gate;
pub mod jsonutil;
pub mod loop_detector;
pub mod prefill;

pub use finish_gate::FinishGate;
pub use loop_detector::LoopDetector;
pub use prefill::PrefillBudget;
