//! Signature-level loop detection (Mingbird, arXiv:2610.02001).
//!
//! Rust port of `arxiv_lab/harness/loop.py`. The signature is
//! `sha1("<tool>|<canonical args>")[:12]`, where canonical args are
//! Python `json.dumps(args, sort_keys=True, separators=(",", ":"))`.
//! [`crate::jsonutil::dumps_sorted`] reproduces that form, including
//! Python's `ensure_ascii` escaping, so signatures match the Python
//! implementation byte-for-byte.

use crate::jsonutil::dumps_sorted;
use serde_json::Value;
use sha1::{Digest, Sha1};

/// Detect exact-repeat streaks and short repeated action cycles.
pub struct LoopDetector {
    pub repeat_k: usize,
    pub max_cycle: usize,
    pub history: Vec<String>,
}

impl LoopDetector {
    pub fn new(repeat_k: usize, max_cycle: usize) -> Self {
        Self {
            repeat_k,
            max_cycle,
            history: Vec::new(),
        }
    }

    /// Stable fingerprint of one action.
    pub fn signature(tool: &str, args: &Value) -> String {
        let canon = dumps_sorted(args);
        let input = format!("{tool}|{canon}");
        let hash = Sha1::digest(input.as_bytes());
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        hex[..12].to_string()
    }

    /// Record one action. Returns a loop description string, or `None`.
    pub fn observe(&mut self, tool: &str, args: &Value) -> Option<String> {
        let sig = Self::signature(tool, args);
        self.history.push(sig.clone());
        let h = &self.history;
        // Exact-repeat streak (checked first, as in Python).
        if h.len() >= self.repeat_k && h[h.len() - self.repeat_k..].iter().all(|s| s == &sig) {
            return Some(format!("loop: signature {sig} repeated {}x", self.repeat_k));
        }
        // Cycle of length 2..=max_cycle repeated twice.
        for cyc in 2..=self.max_cycle {
            if h.len() >= 2 * cyc && h[h.len() - 2 * cyc..h.len() - cyc] == h[h.len() - cyc..] {
                return Some(format!("loop: {cyc}-step cycle repeated"));
            }
        }
        None
    }

    /// Clear the observed history (e.g. between tasks).
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for LoopDetector {
    fn default() -> Self {
        Self::new(3, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn streak_and_cycle_detection() {
        let mut d = LoopDetector::default();
        assert_eq!(d.observe("read", &json!({"path": "/a"})), None);
        assert_eq!(d.observe("read", &json!({"path": "/a"})), None);
        let hit = d.observe("read", &json!({"path": "/a"})).unwrap();
        assert!(hit.starts_with("loop: signature "), "{hit}");

        let mut d = LoopDetector::default();
        // 2-step cycle: a,b,a,b
        assert_eq!(d.observe("a", &json!({"x": 1})), None);
        assert_eq!(d.observe("b", &json!({"x": 2})), None);
        assert_eq!(d.observe("a", &json!({"x": 1})), None);
        assert_eq!(
            d.observe("b", &json!({"x": 2})).as_deref(),
            Some("loop: 2-step cycle repeated")
        );
    }

    #[test]
    fn signature_is_key_order_insensitive() {
        assert_eq!(
            LoopDetector::signature("t", &json!({"b": 1, "a": 2})),
            LoopDetector::signature("t", &json!({"a": 2, "b": 1}))
        );
    }
}
