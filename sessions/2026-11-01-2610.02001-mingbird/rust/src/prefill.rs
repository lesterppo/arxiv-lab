//! Byte-level net-zero prefill budget (Mingbird, arXiv:2610.02001).
//!
//! Rust port of `arxiv_lab/harness/prefill.py`. Behaviour mirrors the
//! Python version exactly for budgets >= 2 (the budgets used in practice;
//! the Python last-resort slice behaves oddly for budget 0/1, where this
//! port saturates instead of wrapping).

use crate::jsonutil::{decode_utf8_ignore, dumps};
use serde_json::Value;

/// Enforce a byte budget on the harness prefill (system text + tool schemas).
pub struct PrefillBudget {
    pub budget_bytes: usize,
}

impl PrefillBudget {
    pub fn new(budget_bytes: usize) -> Self {
        Self { budget_bytes }
    }

    /// Level 0 = full, 1 = drop examples, 2 = drop descriptions+examples.
    /// Works on a clone: the caller's schema values are never mutated.
    fn schema_bytes(schema: &Value, level: u8) -> String {
        let mut d = schema.clone();
        if let Value::Object(map) = &mut d {
            if level >= 1 {
                map.remove("examples");
                if let Some(Value::Object(params)) = map.get_mut("parameters") {
                    params.remove("examples");
                }
            }
            if level >= 2 {
                map.remove("description");
            }
        }
        dumps(&d)
    }

    /// Return `(prefill_text, bytes_used, compressed)`.
    ///
    /// Never exceeds `budget_bytes`: falls back to higher compression
    /// levels, and as a last resort truncates the system text.
    /// `bytes_used` is the byte length of the blob before the
    /// UTF-8-ignore decode, exactly as in the Python version.
    pub fn build_prefill(
        &self,
        system_text: &str,
        tool_schemas: &[Value],
    ) -> (String, usize, bool) {
        let join_parts = |system: &str, schemas: &[String]| -> String {
            let mut parts = vec![system.to_string()];
            parts.extend(schemas.iter().cloned());
            parts.join("\n")
        };

        let full_schemas: Vec<String> = tool_schemas
            .iter()
            .map(|s| Self::schema_bytes(s, 0))
            .collect();
        let full = join_parts(system_text, &full_schemas);
        if full.as_bytes().len() <= self.budget_bytes {
            return (full.clone(), full.as_bytes().len(), false);
        }
        for level in [1u8, 2u8] {
            let schemas: Vec<String> = tool_schemas
                .iter()
                .map(|s| Self::schema_bytes(s, level))
                .collect();
            let blob = join_parts(system_text, &schemas);
            if blob.as_bytes().len() <= self.budget_bytes {
                return (blob.clone(), blob.as_bytes().len(), true);
            }
        }
        // Last resort: truncate system text (never exceed budget).
        let head_len = (self.budget_bytes / 2).min(system_text.as_bytes().len());
        let head = &system_text.as_bytes()[..head_len];
        let rest_joined: Vec<String> = tool_schemas
            .iter()
            .map(|s| Self::schema_bytes(s, 2))
            .collect();
        let rest_full = rest_joined.join("\n");
        let rest_len = self
            .budget_bytes
            .saturating_sub(head.len())
            .saturating_sub(1)
            .min(rest_full.as_bytes().len());
        let rest = &rest_full.as_bytes()[..rest_len];
        let mut blob = Vec::with_capacity(head.len() + 1 + rest.len());
        blob.extend_from_slice(head);
        blob.push(b'\n');
        blob.extend_from_slice(rest);
        let bytes_used = blob.len();
        (decode_utf8_ignore(&blob), bytes_used, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schemas() -> Vec<Value> {
        vec![json!({
            "name": "read",
            "description": "Read a file. ",
            "parameters": {"path": {"type": "string"}},
            "examples": [{"in": {"path": "/a"}, "out": "x"}]
        })]
    }

    #[test]
    fn fits_without_compression() {
        let (text, used, compressed) = PrefillBudget::new(10000).build_prefill("sys", &schemas());
        assert!(!compressed);
        assert_eq!(used, text.as_bytes().len());
        assert!(text.contains("Read a file."));
    }

    #[test]
    fn compresses_and_never_exceeds_budget() {
        let big = vec![json!({
            "name": "read",
            "description": "x".repeat(500),
            "parameters": {"path": {"type": "string"}},
            "examples": [{"in": 1}]
        })];
        for budget in [600usize, 100, 50] {
            let (text, used, compressed) =
                PrefillBudget::new(budget).build_prefill("system text here", &big);
            assert!(compressed);
            assert!(used <= budget, "used {used} > budget {budget}");
            assert!(text.as_bytes().len() <= budget);
        }
    }

    #[test]
    fn does_not_mutate_caller_schemas() {
        let s = schemas();
        let before = dumps(&s[0]);
        PrefillBudget::new(10).build_prefill("sys", &s);
        assert_eq!(dumps(&s[0]), before);
    }
}
