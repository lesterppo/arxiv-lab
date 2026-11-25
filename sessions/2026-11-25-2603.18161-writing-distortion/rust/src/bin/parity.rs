//! `parity` bin: JSON-lines oracle for the Python-vs-Rust parity sweep.
//!
//! Reads one JSON object per line from stdin, computes the requested
//! metric with the Rust port, and prints one JSON object per line.
//! Protocol (see `parity.py` for the driver):
//!
//! ```text
//! {"op":"jsd","a":"...","b":"..."}            -> {"op":"jsd","value":<f64>}
//! {"op":"pronoun","text":"..."}               -> {"op":"pronoun","rate":..,"count":..,"n":..}
//! {"op":"adjective","text":"..."}             -> {"op":"adjective","rate":..,"count":..,"n":..}
//! {"op":"stance","text":"..."}                -> {"op":"stance","pro":..,"anti":..,"score":..,"label":".."}
//! {"op":"pct","before":..,"after":..}         -> {"op":"pct","value":<f64>}
//! {"op":"align","a":[[..]],"b":[[..]]}        -> {"op":"align","value":<f64>}
//!                                                or {"op":"align","error":"..."}
//! ```
//!
//! Floats are printed with Rust's `Display` (shortest round-trip
//! repr); non-finite values print as `Infinity` / `-Infinity` /
//! `NaN`, which Python's `json` module parses with its default
//! `allow_nan=True`.
//! Unknown ops / malformed lines print `{"error":"..."}`.

use std::io::{BufRead, Write};
use writing_distortion::mini_json::Value;
use writing_distortion::*;

fn json_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn fnum(v: f64) -> String {
    // Display gives the shortest string that round-trips. Non-finite
    // values are spelled the way Python's json module parses them
    // (Infinity / -Infinity / NaN); Rust's "inf" is not valid there.
    if v.is_nan() {
        "NaN".to_string()
    } else if v == f64::INFINITY {
        "Infinity".to_string()
    } else if v == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        format!("{}", v)
    }
}

fn get_str<'a>(v: &'a Value, k: &str) -> Result<&'a str, String> {
    v.get(k)
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("missing string field '{}'", k))
}

fn get_num(v: &Value, k: &str) -> Result<f64, String> {
    v.get(k)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("missing numeric field '{}'", k))
}

fn get_vecs(v: &Value, k: &str) -> Result<Vec<Vec<f64>>, String> {
    let arr = v
        .get(k)
        .and_then(|x| x.as_arr())
        .ok_or_else(|| format!("missing array field '{}'", k))?;
    let mut out = Vec::with_capacity(arr.len());
    for row in arr {
        let row = row
            .as_arr()
            .ok_or_else(|| format!("field '{}' must be an array of arrays", k))?;
        let mut r = Vec::with_capacity(row.len());
        for x in row {
            r.push(
                x.as_f64()
                    .ok_or_else(|| format!("field '{}' must contain numbers", k))?,
            );
        }
        out.push(r);
    }
    Ok(out)
}

fn handle(line: &str) -> String {
    let v = match mini_json::parse(line) {
        Ok(v) => v,
        Err(e) => return format!("{{\"error\":\"parse: {}\"}}", json_escape(&e)),
    };
    let op = match v.get("op").and_then(|x| x.as_str()) {
        Some(o) => o,
        None => return "{\"error\":\"missing op\"}".to_string(),
    };
    match op {
        "jsd" => match (get_str(&v, "a"), get_str(&v, "b")) {
            (Ok(a), Ok(b)) => format!("{{\"op\":\"jsd\",\"value\":{}}}", fnum(unigram_jsd(a, b))),
            _ => "{\"error\":\"jsd needs string a and b\"}".to_string(),
        },
        "pronoun" => match get_str(&v, "text") {
            Ok(t) => {
                let (rate, count, n) = pronoun_rate(t);
                format!(
                    "{{\"op\":\"pronoun\",\"rate\":{},\"count\":{},\"n\":{}}}",
                    fnum(rate),
                    count,
                    n
                )
            }
            _ => "{\"error\":\"pronoun needs string text\"}".to_string(),
        },
        "adjective" => match get_str(&v, "text") {
            Ok(t) => {
                let (rate, count, n) = adjective_rate(t);
                format!(
                    "{{\"op\":\"adjective\",\"rate\":{},\"count\":{},\"n\":{}}}",
                    fnum(rate),
                    count,
                    n
                )
            }
            _ => "{\"error\":\"adjective needs string text\"}".to_string(),
        },
        "stance" => match get_str(&v, "text") {
            Ok(t) => {
                let s = keyword_stance(t);
                format!(
                    "{{\"op\":\"stance\",\"pro\":{},\"anti\":{},\"score\":{},\"label\":\"{}\"}}",
                    s.pro, s.anti, s.score, s.label
                )
            }
            _ => "{\"error\":\"stance needs string text\"}".to_string(),
        },
        "pct" => match (get_num(&v, "before"), get_num(&v, "after")) {
            (Ok(b), Ok(a)) => format!(
                "{{\"op\":\"pct\",\"value\":{}}}",
                fnum(pct_change(b, a))
            ),
            _ => "{\"error\":\"pct needs numeric before and after\"}".to_string(),
        },
        "align" => match (get_vecs(&v, "a"), get_vecs(&v, "b")) {
            (Ok(a), Ok(b)) => match mean_shift_alignment(&a, &b) {
                Ok(val) => format!("{{\"op\":\"align\",\"value\":{}}}", fnum(val)),
                Err(e) => format!("{{\"op\":\"align\",\"error\":\"{}\"}}", json_escape(&e)),
            },
            _ => "{\"error\":\"align needs array fields a and b\"}".to_string(),
        },
        other => format!("{{\"error\":\"unknown op '{}'\"}}", json_escape(other)),
    }
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let _ = writeln!(out, "{}", handle(&line));
    }
    let _ = out.flush();
}
