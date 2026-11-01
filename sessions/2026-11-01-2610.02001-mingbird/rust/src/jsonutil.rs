//! JSON serialisation that matches Python's `json.dumps` for the inputs
//! this crate handles, so byte counts and signatures stay comparable
//! with the Python reference.
//!
//! Two differences from `serde_json::to_string` matter here:
//! 1. Python preserves dict insertion order; `serde_json` with the
//!    `preserve_order` feature does too, and this module serialises
//!    `Value` directly so that order is kept.
//! 2. Python's default `ensure_ascii=True` escapes every non-ASCII
//!    character as `\uXXXX` (with surrogate pairs above the BMP).
//!    `serde_json` emits raw UTF-8, which would change byte counts
//!    and SHA-1 signatures for non-ASCII input. This module escapes
//!    non-ASCII the Python way.

use serde_json::Value;

/// Python `json.dumps(v, separators=(",", ":"))` (compact form).
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_value(v, &mut out, false);
    out
}

/// Python `json.dumps(v)` with default separators `(", ", ": ")`.
/// Used only for the no-budget prefill byte count in `verify`, which
/// mirrors the Python test's `json.dumps(s)` call exactly.
pub fn dumps_default(v: &Value) -> String {
    let mut out = String::new();
    write_value(v, &mut out, true);
    out
}

/// Like [`dumps`], but object keys are sorted recursively first, matching
/// Python `json.dumps(v, sort_keys=True, separators=(",", ":"))`.
pub fn dumps_sorted(v: &Value) -> String {
    dumps(&sort_value(v))
}

fn sort_value(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), sort_value(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_value).collect()),
        other => other.clone(),
    }
}

fn write_value(v: &Value, out: &mut String, spaced: bool) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if spaced {
                        out.push(' ');
                    }
                }
                write_value(item, out, spaced);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if spaced {
                        out.push(' ');
                    }
                }
                write_string(k, out);
                out.push(':');
                if spaced {
                    out.push(' ');
                }
                write_value(val, out, spaced);
            }
            out.push('}');
        }
    }
}

/// Python `json.dumps` string escaping with `ensure_ascii=True`.
fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if (c as u32) > 0x7f => {
                let cp = c as u32;
                if cp > 0xffff {
                    let v = cp - 0x10000;
                    out.push_str(&format!(
                        "\\u{:04x}\\u{:04x}",
                        0xd800 + (v >> 10),
                        0xdc00 + (v & 0x3ff)
                    ));
                } else {
                    out.push_str(&format!("\\u{:04x}", cp));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Decode bytes as UTF-8, dropping invalid bytes, matching Python's
/// `bytes.decode("utf-8", "ignore")` used by the last-resort truncation
/// path in `PrefillBudget::build_prefill`.
pub fn decode_utf8_ignore(bytes: &[u8]) -> String {
    let mut out = String::new();
    let mut rest = bytes;
    while !rest.is_empty() {
        match std::str::from_utf8(rest) {
            Ok(s) => {
                out.push_str(s);
                break;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    // SAFETY: `valid_up_to` is a UTF-8 boundary by definition.
                    out.push_str(unsafe { std::str::from_utf8_unchecked(&rest[..valid]) });
                }
                let skip = e.error_len().unwrap_or(1);
                rest = &rest[valid + skip..];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compact_and_sorted_match_python_forms() {
        let v = json!({"b": 1, "a": {"z": "x\ny", "m": true}});
        assert_eq!(dumps(&v), r#"{"b":1,"a":{"z":"x\ny","m":true}}"#);
        assert_eq!(dumps_sorted(&v), r#"{"a":{"m":true,"z":"x\ny"},"b":1}"#);
        assert_eq!(
            dumps_default(&v),
            r#"{"b": 1, "a": {"z": "x\ny", "m": true}}"#
        );
    }

    #[test]
    fn non_ascii_is_escaped_like_python() {
        assert_eq!(dumps(&json!("café")), r#""caf\u00e9""#);
        assert_eq!(dumps(&json!("😀")), r#""\ud83d\ude00""#);
    }

    #[test]
    fn decode_ignore_drops_truncated_tail() {
        // "é" is 2 bytes; keep only the first byte, as a byte-level
        // truncation of the system text can do.
        assert_eq!(decode_utf8_ignore(&[0x61, 0xc3]), "a");
        assert_eq!(decode_utf8_ignore(b"ok\xff!"), "ok!");
    }
}
