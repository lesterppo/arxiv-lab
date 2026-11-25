//! `writing_distortion` verify binary.
//!
//! Mirrors the checkable assertions of `../test_writing_distortion_cpu.py`
//! (the Python CPU test, 16 checks) against the Rust port, using the
//! same `sample16.json` for the real-text checks. Usage:
//!
//! ```text
//! cargo run --release [-- ../sample16.json] [--json]
//! ```
//!
//! Default sample path is `../sample16.json` relative to the current
//! working directory (i.e. run from `rust/`). `--json` prints a JSON
//! summary object as the last line (for `results.json` generation).
//! Exit code 0 iff every check passes.

use std::process::ExitCode;
use writing_distortion::mini_json::Value;
use writing_distortion::*;

struct Check {
    name: &'static str,
    ok: bool,
    detail: String,
}

fn check(results: &mut Vec<Check>, name: &'static str, ok: bool, detail: String) {
    println!(
        "{}  {}{}",
        if ok { "PASS" } else { "FAIL" },
        name,
        if detail.is_empty() { String::new() } else { format!("  -- {}", detail) }
    );
    results.push(Check { name, ok, detail });
}

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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let want_json = args.iter().any(|a| a == "--json");
    let sample_path = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| "../sample16.json".to_string());

    let mut results: Vec<Check> = Vec::new();

    // 1. JSD of a text with itself is exactly 0.
    let t = "Self driving cars are safer and more efficient than human drivers.";
    let j = unigram_jsd(t, t);
    check(&mut results, "jsd_identical_is_zero", j == 0.0, format!("got {:?}", j));

    // 2. JSD in [0, 1] and > 0 for different texts.
    let a = "Self driving cars may seem too futuristic to believe.";
    let b = "Cooking rice requires water heat and patience every day.";
    let j = unigram_jsd(a, b);
    check(
        &mut results,
        "jsd_range_and_positive",
        j > 0.0 && j <= 1.0,
        format!("jsd={:.4}", j),
    );

    // 3. Determinism: repeat call identical.
    let j1 = unigram_jsd(a, b);
    let j2 = unigram_jsd(a, b);
    check(
        &mut results,
        "jsd_deterministic",
        j1 == j2,
        format!("{:.6} vs {:.6}", j1, j2),
    );

    // 4. pronoun_rate on a hand-computed example.
    let (rate, count, n) = pronoun_rate("I am happy. We are here.");
    check(
        &mut results,
        "pronoun_rate_hand_computed",
        count == 2 && n == 6 && (rate - 100.0 * 2.0 / 6.0).abs() < 1e-9,
        format!("rate={:.4} count={} n={}", rate, count, n),
    );

    // 5. pronoun_rate empty text guard.
    let (rate, count, n) = pronoun_rate("");
    check(
        &mut results,
        "pronoun_rate_empty",
        rate == 0.0 && count == 0 && n == 0,
        format!("got {:?}", (rate, count, n)),
    );

    // 6. adjective_rate: suffix words counted, plain words not.
    let (rate, count, n) = adjective_rate("beautiful wonderful dangerous cat dog run");
    check(
        &mut results,
        "adjective_rate_suffix_proxy",
        count == 3 && (rate - 50.0).abs() < 1e-9,
        format!("rate={:.4} count={} n={}", rate, count, n),
    );

    // 7-9, 14-16 need sample16.json.
    let sample = match std::fs::read_to_string(&sample_path) {
        Ok(s) => match mini_json::parse(&s) {
            Ok(v) => Some(v),
            Err(e) => {
                check(
                    &mut results,
                    "sample16_json_parse",
                    false,
                    format!("{}: {}", sample_path, e),
                );
                None
            }
        },
        Err(e) => {
            for name in [
                "adjective_rate_real_range_0_100",
                "keyword_stance_matches_sample16_all16",
                "keyword_stance_essay17_pro",
                "human_jsd_range_all16",
                "frontier_grammar_jsd_and_rates_sane",
                "determinism_on_real_text",
            ] {
                check(
                    &mut results,
                    name,
                    false,
                    format!("cannot read {}: {}", sample_path, e),
                );
            }
            None
        }
    };

    // 10. pct_change correctness incl. zero guard.
    let ok = pct_change(100.0, 150.0) == 50.0
        && pct_change(100.0, 50.0) == -50.0
        && pct_change(100.0, 100.0) == 0.0
        && pct_change(0.0, 0.0) == 0.0
        && pct_change(0.0, 5.0).is_infinite();
    check(&mut results, "pct_change_values", ok, "50/-50/0/0/inf expected".to_string());

    // 11. mean_shift_alignment: identical shifts -> ~1.0.
    let a_vecs = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 1.0]];
    let b_vecs = vec![vec![1.0, 0.0], vec![2.0, 2.0], vec![4.0, 1.0]];
    let val = mean_shift_alignment(&a_vecs, &b_vecs);
    check(
        &mut results,
        "mean_shift_identical_shifts_near_1",
        matches!(val, Ok(v) if (v - 1.0).abs() < 1e-9),
        format!("got {:?}", val),
    );

    // 12. mean_shift_alignment: orthogonal shifts -> ~0.7071.
    let val = mean_shift_alignment(
        &[vec![0.0, 0.0], vec![0.0, 0.0]],
        &[vec![1.0, 0.0], vec![0.0, 1.0]],
    );
    check(
        &mut results,
        "mean_shift_orthogonal_shifts",
        matches!(val, Ok(v) if (v - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9),
        format!("got {:?}", val),
    );

    // 13. mean_shift_alignment: opposing shifts cancel -> low (0.0).
    let val = mean_shift_alignment(
        &[vec![0.0, 0.0], vec![0.0, 0.0]],
        &[vec![1.0, 0.0], vec![-1.0, 0.0]],
    );
    check(
        &mut results,
        "mean_shift_opposing_shifts_low",
        matches!(val, Ok(v) if v < 0.1),
        format!("got {:?}", val),
    );

    if let Some(sample) = &sample {
        let essays: Vec<&Value> = sample
            .get("essays")
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().collect())
            .unwrap_or_default();
        let get_str = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();

        // 7. adjective_rate is a proxy, sane range on real essays.
        let mut ok = true;
        for e in &essays {
            let (r, _, _) = adjective_rate(&get_str(e, "draft1"));
            if !(0.0..=100.0).contains(&r) {
                ok = false;
            }
        }
        check(
            &mut results,
            "adjective_rate_real_range_0_100",
            ok,
            format!("n={}", essays.len()),
        );

        // 8. keyword_stance matches the recorded label AND counts for
        // all 16 sample essays on draft1.
        let mut bad: Vec<String> = Vec::new();
        for e in &essays {
            let got = keyword_stance(&get_str(e, "draft1"));
            let rec = e.get("keyword_stance");
            let (rpro, ranti, rlabel) = match rec {
                Some(r) => (
                    r.get("pro").and_then(|x| x.as_f64()).unwrap_or(-1.0) as i64,
                    r.get("anti").and_then(|x| x.as_f64()).unwrap_or(-1.0) as i64,
                    r.get("label").and_then(|x| x.as_str()).unwrap_or("?"),
                ),
                None => (-1, -1, "?"),
            };
            if got.label != rlabel || got.pro as i64 != rpro || got.anti as i64 != ranti {
                bad.push(format!(
                    "{} got pro={} anti={} label={} rec pro={} anti={} label={}",
                    get_str(e, "essay_id"), got.pro, got.anti, got.label, rpro, ranti, rlabel
                ));
            }
        }
        check(
            &mut results,
            "keyword_stance_matches_sample16_all16",
            bad.is_empty(),
            if bad.is_empty() { "16/16 match".to_string() } else { format!("mismatches={:?}", bad) },
        );

        // 9. keyword_stance: essay 17 is pro (known sample).
        let e17 = essays.iter().find(|e| get_str(e, "essay_id") == "17");
        let (ok9, detail9) = match e17 {
            Some(e) => {
                let got = keyword_stance(&get_str(e, "draft1"));
                (
                    got.label == "pro" && got.score >= 3,
                    format!("got pro={} anti={} score={} label={}", got.pro, got.anti, got.score, got.label),
                )
            }
            None => (false, "essay 17 not found".to_string()),
        };
        check(&mut results, "keyword_stance_essay17_pro", ok9, detail9);

        // 14. Human D1->D2 JSD in [0, 1] for all 16 sample essays.
        let mut ok = true;
        let mut worst = 0.0_f64;
        for e in &essays {
            let j = unigram_jsd(&get_str(e, "draft1"), &get_str(e, "draft2"));
            worst = worst.max(j);
            if !(0.0..=1.0).contains(&j) {
                ok = false;
            }
        }
        check(
            &mut results,
            "human_jsd_range_all16",
            ok,
            format!("max jsd={:.4}", worst),
        );

        // 15. Frontier grammar JSD in [0, 1] for all 16 x 3 models, and
        // pronoun/adjective rates in sane ranges on frontier texts.
        let mut ok = true;
        for e in &essays {
            if let Some(Value::Obj(cells)) = e
                .get("frontier_with_feedback")
                .and_then(|f| f.get("grammar"))
            {
                for (_, cell) in cells {
                    let ai = match cell {
                        Value::Str(s) => s.clone(),
                        v => v.get("ai_text").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                    };
                    let j = unigram_jsd(&get_str(e, "draft1"), &ai);
                    let (pr, _, _) = pronoun_rate(&ai);
                    let (ar, _, _) = adjective_rate(&ai);
                    if !(0.0..=1.0).contains(&j)
                        || !(0.0..=100.0).contains(&pr)
                        || !(0.0..=100.0).contains(&ar)
                    {
                        ok = false;
                    }
                }
            }
        }
        check(
            &mut results,
            "frontier_grammar_jsd_and_rates_sane",
            ok,
            "16x3 grammar cells".to_string(),
        );

        // 16. Determinism on real text (repeat JSD + stance identical).
        let e0 = &essays[0];
        let ok = unigram_jsd(&get_str(e0, "draft1"), &get_str(e0, "draft2"))
            == unigram_jsd(&get_str(e0, "draft1"), &get_str(e0, "draft2"))
            && keyword_stance(&get_str(e0, "draft1")) == keyword_stance(&get_str(e0, "draft1"));
        check(&mut results, "determinism_on_real_text", ok, String::new());
    }

    let n_pass = results.iter().filter(|c| c.ok).count();
    println!("\n{}/{} checks passed", n_pass, results.len());
    if want_json {
        let items: Vec<String> = results
            .iter()
            .map(|c| {
                format!(
                    "{{\"name\":\"{}\",\"ok\":{},\"detail\":\"{}\"}}",
                    json_escape(c.name),
                    c.ok,
                    json_escape(&c.detail)
                )
            })
            .collect();
        println!(
            "{{\"checks_passed\":{},\"checks_total\":{},\"checks\":[{}]}}",
            n_pass,
            results.len(),
            items.join(",")
        );
    }
    if n_pass == results.len() {
        ExitCode::from(0)
    } else {
        ExitCode::from(1)
    }
}
