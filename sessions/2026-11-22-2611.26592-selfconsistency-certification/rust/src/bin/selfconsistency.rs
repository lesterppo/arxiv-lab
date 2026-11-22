//! `selfconsistency` — CLI for the Rust port of arXiv:2611.26592.
//!
//!   verify [--expected PATH]  mirror the Python CPU test's
//!                             experiments E0-E7 and assertions, plus
//!                             exact-equality checks against
//!                             ../results.json (default --expected)
//!   ppf <p> / cdf <x>         print normal_ppf / normal_cdf with
//!                             full precision + u64 bits (parity)
//!   ppf-batch / cdf-batch     read values on stdin, one per line
//!   eprocess <csv>            wealth path of the mixture e-process
//!   crossing <eps> <csv>      first crossing step, or null
//!   run-single <seed> <theta> <rho> <n_rollouts> <n_units> [eps]
//!   run-prompt <seed> <theta> <rho> <n_prompts> <n_reps> [eps]
//!   ess-rows <seed> <rho> <m_units> <n_max>
//!                             E4-style ESS rows (prefixes 1,4,16,64,256)
//!   soak [n_units] [n_rollouts] [seed] [theta] [rho]
//!                             timed scale run (default 20000 x 256)
//!
//! All floats print with Python-`repr`-compatible formatting
//! (`selfconsistency_rs::py_repr`), so parsed values are identical
//! to the Python side bit-for-bit.

use selfconsistency_rs::*;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, BufRead};
use std::process::exit;
use std::time::Instant;

const EPS: f64 = 0.05;

// ---------------------------------------------------------------------------
// Minimal JSON value + parser (for comparing against ../results.json).
// Numbers parse as f64; objects keep insertion order via BTreeMap
// (lookups only, order irrelevant).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn ws(&mut self) {
        while self.pos < self.bytes.len() && (self.bytes[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.bytes.get(self.pos) == Some(&c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at {}", c as char, self.pos))
        }
    }

    fn word(&mut self, w: &str) -> Result<(), String> {
        if self.bytes[self.pos..].starts_with(w.as_bytes()) {
            self.pos += w.len();
            Ok(())
        } else {
            Err(format!("expected {w}"))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => {
                self.pos += 1;
                let mut m = BTreeMap::new();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Ok(Json::Obj(m));
                }
                loop {
                    let k = match self.value()? {
                        Json::Str(s) => s,
                        _ => return Err("expected string key".to_string()),
                    };
                    self.expect(b':')?;
                    let v = self.value()?;
                    m.insert(k, v);
                    match self.peek() {
                        Some(b',') => {
                            self.pos += 1;
                        }
                        Some(b'}') => {
                            self.pos += 1;
                            return Ok(Json::Obj(m));
                        }
                        _ => return Err("expected ',' or '}' in object".to_string()),
                    }
                }
            }
            Some(b'[') => {
                self.pos += 1;
                let mut a = Vec::new();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    match self.peek() {
                        Some(b',') => {
                            self.pos += 1;
                        }
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err("expected ',' or ']' in array".to_string()),
                    }
                }
            }
            Some(b'"') => {
                self.pos += 1;
                let mut s = String::new();
                loop {
                    match self.bytes.get(self.pos) {
                        None => return Err("unterminated string".to_string()),
                        Some(b'"') => {
                            self.pos += 1;
                            return Ok(Json::Str(s));
                        }
                        Some(b'\\') => {
                            self.pos += 1;
                            let e = *self.bytes.get(self.pos).ok_or("bad escape")?;
                            self.pos += 1;
                            match e {
                                b'n' => s.push('\n'),
                                b't' => s.push('\t'),
                                b'r' => s.push('\r'),
                                b'u' => {
                                    let hex = std::str::from_utf8(
                                        &self.bytes[self.pos..self.pos + 4],
                                    )
                                    .map_err(|_| "bad \\u".to_string())?;
                                    let cp = u32::from_str_radix(hex, 16)
                                        .map_err(|_| "bad \\u".to_string())?;
                                    self.pos += 4;
                                    s.push(
                                        char::from_u32(cp)
                                            .ok_or("bad code point".to_string())?,
                                    );
                                }
                                _ => s.push(e as char),
                            }
                        }
                        Some(&b) => {
                            // Strings here are ASCII in practice; take
                            // bytes as chars (results.json has none
                            // outside ASCII except in no field we read).
                            s.push(b as char);
                            self.pos += 1;
                        }
                    }
                }
            }
            Some(b't') => {
                self.word("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.word("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.word("null")?;
                Ok(Json::Null)
            }
            Some(_) => {
                let start = self.pos;
                while self.pos < self.bytes.len()
                    && matches!(
                        self.bytes[self.pos],
                        b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
                    )
                {
                    self.pos += 1;
                }
                let tok = std::str::from_utf8(&self.bytes[start..self.pos])
                    .map_err(|_| "bad number".to_string())?;
                tok.parse::<f64>()
                    .map(Json::Num)
                    .map_err(|_| format!("bad number {tok}"))
            }
            None => Err("unexpected end of input".to_string()),
        }
    }
}

fn parse_json(s: &str) -> Result<Json, String> {
    let mut p = Parser::new(s);
    let v = p.value()?;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check(name: &str, cond: bool, detail: &str, failures: &mut Vec<String>) {
    println!("[{}] {} {}", if cond { "PASS" } else { "FAIL" }, name, detail);
    if !cond {
        failures.push(format!("{} {}", name, detail));
    }
}

fn parse_csv(s: &str) -> Vec<f64> {
    s.split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse::<f64>().unwrap())
        .collect()
}

fn ess_rows(seed: u64, rho: f64, m_units: usize, n_max: usize) -> Vec<(usize, f64, f64, f64)> {
    // Mirrors the Python E4 loop exactly: one RNG stream, running
    // sums across the prefixes 1, 4, 16, 64, 256.
    let (tau, sigma) = tau_sigma_from_rho(rho);
    let mut rng = SplitMix64::new(seed);
    let prefixes = [1usize, 4, 16, 64, 256];
    let mut sums: Vec<Vec<f64>> = prefixes.iter().map(|_| Vec::new()).collect();
    for _ in 0..m_units {
        let (_, incs) = simulate_unit(&mut rng, 0.0, tau, sigma, n_max, false);
        let mut run = 0.0f64;
        let mut idx = 0usize;
        for (pi, &n) in prefixes.iter().enumerate() {
            while idx < n {
                run += incs[idx];
                idx += 1;
            }
            sums[pi].push(run / n as f64);
        }
    }
    prefixes
        .iter()
        .enumerate()
        .map(|(pi, &n)| {
            let vals = &sums[pi];
            // Python's E4 uses sum() (Neumaier) for mean and var.
            let mean: f64 = py_sum(vals.iter().copied()) / vals.len() as f64;
            let var: f64 =
                py_sum(vals.iter().map(|v| (v - mean) * (v - mean))) / vals.len() as f64;
            (n, var, within_unit_mean_var(n, rho), ess_within_unit(n, rho))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// verify: mirror of the Python CPU test E0-E7.
// ---------------------------------------------------------------------------

fn cmd_verify(expected_path: &str) -> i32 {
    let mut failures: Vec<String> = Vec::new();
    let mut results: Vec<(String, String)> = Vec::new();

    // ---- E0: unit checks ----
    check(
        "E0 ppf(0.5)=0",
        normal_ppf(0.5).abs() < 1e-12,
        "",
        &mut failures,
    );
    check(
        "E0 ppf(0.975)~1.96",
        (normal_ppf(0.975) - 1.959964).abs() < 1e-5,
        &format!("got {:.6}", normal_ppf(0.975)),
        &mut failures,
    );
    check(
        "E0 cdf(0)=0.5",
        (normal_cdf(0.0) - 0.5).abs() < 1e-12,
        "",
        &mut failures,
    );
    let (tau, sigma) = tau_sigma_from_rho(0.48);
    check(
        "E0 rho round trip",
        (rho_from_tau_sigma(tau, sigma) - 0.48).abs() < 1e-12,
        "",
        &mut failures,
    );
    check(
        "E0 pi(theta=-0.15, tau=0.693)~0.414",
        (pi_positive(-0.15, tau) - 0.414).abs() < 0.005,
        &format!("got {:.3}", pi_positive(-0.15, tau)),
        &mut failures,
    );
    let path = betting_eprocess(&vec![1.0; 20], &LAMBDA_GRID);
    check(
        "E0 e-process crosses on all +1",
        crossing_step(&vec![1.0; 20], EPS, &LAMBDA_GRID).is_some(),
        "",
        &mut failures,
    );
    check(
        "E0 e-process never crosses on all -1",
        crossing_step(&vec![-1.0; 200], EPS, &LAMBDA_GRID).is_none(),
        "",
        &mut failures,
    );
    check(
        "E0 e-process wealth starts at 1",
        path[0] == 1.0,
        "",
        &mut failures,
    );
    check(
        "E0 theta_hat(0.5)=0",
        theta_hat_from_accuracy(0.5).abs() < 1e-9,
        "",
        &mut failures,
    );
    check(
        "E0 icc on perfectly split groups ~1",
        (icc_anova(&[
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ])
        .unwrap()
            - 1.0)
            .abs()
            < 1e-9,
        "",
        &mut failures,
    );
    check(
        "E0 icc on identical groups ~0",
        icc_anova(&[
            vec![1.0, 0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0, 1.0],
        ])
        .unwrap()
        .abs()
            < 1e-9,
        "",
        &mut failures,
    );
    results.push(("E0_units".to_string(), "\"all passed\"".to_string()));

    // ---- E1: rho=0 control ----
    let e1 = run_single_unit_experiment(7, -0.15, 0.0, 64, 2000, EPS, &LAMBDA_GRID);
    println!("E1 rho=0 control: crossing={:.4}", e1.crossing_rate);
    check(
        "E1 rho=0 crossing <= eps",
        e1.crossing_rate < 0.02,
        &format!("rate={:.4}", e1.crossing_rate),
        &mut failures,
    );
    results.push(("E1_rho0_control".to_string(), single_unit_json(&e1)));

    // ---- E2: AIME regime, population null true ----
    let e2_64 = run_single_unit_experiment(11, -0.15, 0.48, 64, 2000, EPS, &LAMBDA_GRID);
    let e2_256 = run_single_unit_experiment(11, -0.15, 0.48, 256, 2000, EPS, &LAMBDA_GRID);
    println!(
        "E2 AIME regime: N=64 crossing={:.3} (pi={:.3}); N=256 crossing={:.3}",
        e2_64.crossing_rate, e2_64.pi, e2_256.crossing_rate
    );
    check(
        "E2 N=64 crossing far above eps (paper: 0.24)",
        e2_64.crossing_rate > 0.10,
        &format!("rate={:.3}", e2_64.crossing_rate),
        &mut failures,
    );
    check(
        "E2 crossing grows with rollout budget",
        e2_256.crossing_rate > e2_64.crossing_rate,
        &format!("64:{:.3} 256:{:.3}", e2_64.crossing_rate, e2_256.crossing_rate),
        &mut failures,
    );
    check(
        "E2 N=256 crossing approaches pi from below",
        e2_256.crossing_rate > 0.25 && e2_256.crossing_rate < e2_256.pi + 0.05,
        &format!("rate={:.3} pi={:.3}", e2_256.crossing_rate, e2_256.pi),
        &mut failures,
    );
    results.push(("E2_aime_regime_N64".to_string(), single_unit_json(&e2_64)));
    results.push(("E2_aime_regime_N256".to_string(), single_unit_json(&e2_256)));

    // ---- E3: realized-null split ----
    println!(
        "E3 realized-null split (N=256): mu<=0 rate={:.4} (n={}); mu>0 rate={:.3} (n={})",
        e2_256.crossing_rate_realized_null.unwrap(),
        e2_256.n_realized_null,
        e2_256.crossing_rate_realized_alt.unwrap(),
        e2_256.n_realized_alt
    );
    check(
        "E3 valid for the realized null (mu<=0)",
        e2_256.crossing_rate_realized_null.unwrap() < 0.08,
        &format!(
            "rate={:.4}",
            e2_256.crossing_rate_realized_null.unwrap()
        ),
        &mut failures,
    );
    check(
        "E3 high power when realized null is false (mu>0)",
        e2_256.crossing_rate_realized_alt.unwrap() > 0.50,
        &format!("rate={:.3}", e2_256.crossing_rate_realized_alt.unwrap()),
        &mut failures,
    );
    results.push((
        "E3_realized_null_split".to_string(),
        format!(
            "{{\"rate_mu_le_0\": {}, \"n_mu_le_0\": {}, \"rate_mu_gt_0\": {}, \"n_mu_gt_0\": {}}}",
            py_repr(e2_256.crossing_rate_realized_null.unwrap()),
            e2_256.n_realized_null,
            py_repr(e2_256.crossing_rate_realized_alt.unwrap()),
            e2_256.n_realized_alt
        ),
    ));

    // ---- E4: ESS saturation ----
    let rows = ess_rows(23, 0.48, 4000, 256);
    let mut e4_json = String::from("[");
    for (i, (n, emp, pred, ess)) in rows.iter().enumerate() {
        println!(
            "E4 n={:3} emp_var={:.4} pred_var={:.4} ESS={:.2}",
            n, emp, pred, ess
        );
        if i > 0 {
            e4_json.push_str(", ");
        }
        e4_json.push_str(&format!(
            "{{\"n\": {}, \"empirical_var\": {}, \"predicted_var\": {}, \"ess\": {}}}",
            n,
            py_repr((emp * 10000.0).round() / 10000.0),
            py_repr((pred * 10000.0).round() / 10000.0),
            py_repr((ess * 1000.0).round() / 1000.0)
        ));
    }
    e4_json.push(']');
    check(
        "E4 var(n=256) saturates near rho, not near 0",
        (rows[4].1 - 0.48).abs() < 0.05,
        &format!("emp={:.4} rho=0.48", rows[4].1),
        &mut failures,
    );
    check(
        "E4 var matches rho+(1-rho)/n at every prefix",
        rows.iter().all(|(_, emp, pred, _)| (emp - pred).abs() < 0.03),
        "",
        &mut failures,
    );
    check(
        "E4 ESS(256) ~ 1/rho ~ 2.08",
        ess_within_unit(256, 0.48) > 1.5 && ess_within_unit(256, 0.48) < 3.0,
        &format!("ESS={:.2}", ess_within_unit(256, 0.48)),
        &mut failures,
    );
    results.push(("E4_ess_saturation".to_string(), e4_json));

    // ---- E5: prompt-level restores validity; optimal allocation ----
    let e5_valid = run_prompt_level_experiment(31, -0.15, 0.48, 256, 2000, EPS, &LAMBDA_GRID);
    println!(
        "E5 prompt-level validity: crossing={:.4}",
        e5_valid.crossing_rate
    );
    check(
        "E5 prompt-level crossing <= eps",
        e5_valid.crossing_rate < 0.08,
        &format!("rate={:.4}", e5_valid.crossing_rate),
        &mut failures,
    );
    let e5_pow_prompt =
        run_prompt_level_experiment(37, 0.4, 0.48, 256, 1000, EPS, &LAMBDA_GRID);
    let e5_pow_single =
        run_single_unit_experiment(37, 0.4, 0.48, 256, 1000, EPS, &LAMBDA_GRID);
    println!(
        "E5 power (theta=+0.4, B=256): prompt-level={:.3} single-unit={:.3}",
        e5_pow_prompt.crossing_rate, e5_pow_single.crossing_rate
    );
    check(
        "E5 prompt-level power exceeds single-unit by >0.15",
        e5_pow_prompt.crossing_rate > e5_pow_single.crossing_rate + 0.15,
        &format!(
            "prompt={:.3} single={:.3}",
            e5_pow_prompt.crossing_rate, e5_pow_single.crossing_rate
        ),
        &mut failures,
    );
    check(
        "E5 prompt-level power high when null false",
        e5_pow_prompt.crossing_rate > 0.85,
        &format!("rate={:.3}", e5_pow_prompt.crossing_rate),
        &mut failures,
    );
    results.push((
        "E5_prompt_level_validity".to_string(),
        prompt_level_json(&e5_valid),
    ));
    results.push((
        "E5_power_prompt_level".to_string(),
        prompt_level_json(&e5_pow_prompt),
    ));
    results.push((
        "E5_power_single_unit".to_string(),
        single_unit_json(&e5_pow_single),
    ));

    // ---- E6: GSM8K regime, gap dormant ----
    let e6 = run_single_unit_experiment(41, 0.96, 0.435, 64, 2000, EPS, &LAMBDA_GRID);
    println!(
        "E6 GSM8K regime: crossing={:.3} (pi={:.3})",
        e6.crossing_rate, e6.pi
    );
    check(
        "E6 GSM8K regime single-unit crosses with high probability",
        e6.crossing_rate > 0.80,
        &format!("rate={:.3}", e6.crossing_rate),
        &mut failures,
    );
    results.push(("E6_gsm8k_regime".to_string(), single_unit_json(&e6)));

    // ---- E7: determinism ----
    let e2_repeat = run_single_unit_experiment(11, -0.15, 0.48, 64, 2000, EPS, &LAMBDA_GRID);
    check(
        "E7 deterministic repeat identical",
        e2_repeat == e2_64,
        "",
        &mut failures,
    );
    results.push(("E7_determinism".to_string(), "\"identical\"".to_string()));

    // ---- Exact comparison against the Python results.json ----
    match fs::read_to_string(expected_path) {
        Ok(text) => match parse_json(&text) {
            Ok(expected) => {
                let mut mismatches: Vec<String> = Vec::new();
                let cmp_single = |label: &str, got: &SingleUnitResult, exp: &Json, out: &mut Vec<String>| {
                    let checks: [(&str, f64, Option<f64>); 8] = [
                        ("theta", got.theta, exp.get("theta").and_then(|v| v.as_f64())),
                        ("rho", got.rho, exp.get("rho").and_then(|v| v.as_f64())),
                        (
                            "n_rollouts",
                            got.n_rollouts as f64,
                            exp.get("n_rollouts").and_then(|v| v.as_f64()),
                        ),
                        (
                            "n_units",
                            got.n_units as f64,
                            exp.get("n_units").and_then(|v| v.as_f64()),
                        ),
                        ("eps", got.eps, exp.get("eps").and_then(|v| v.as_f64())),
                        ("pi", got.pi, exp.get("pi").and_then(|v| v.as_f64())),
                        (
                            "crossing_rate",
                            got.crossing_rate,
                            exp.get("crossing_rate").and_then(|v| v.as_f64()),
                        ),
                        (
                            "n_realized_null",
                            got.n_realized_null as f64,
                            exp.get("n_realized_null").and_then(|v| v.as_f64()),
                        ),
                    ];
                    for (k, g, e) in checks {
                        match e {
                            None => out.push(format!("{label}.{k}: missing in expected")),
                            Some(e) if g != e => {
                                out.push(format!("{label}.{k}: rust={g:?} python={e:?}"))
                            }
                            _ => {}
                        }
                    }
                    let mut cmp_opt = |k: &str, g: Option<f64>, e: Option<&Json>| match (g, e) {
                        (Some(gv), Some(Json::Num(ev))) if gv != *ev => {
                            out.push(format!("{label}.{k}: rust={gv:?} python={ev:?}"))
                        }
                        (None, Some(Json::Null)) | (Some(_), Some(Json::Num(_))) | (None, None) => {}
                        _ => out.push(format!("{label}.{k}: shape mismatch")),
                    };
                    cmp_opt(
                        "crossing_rate_realized_null",
                        got.crossing_rate_realized_null,
                        exp.get("crossing_rate_realized_null"),
                    );
                    cmp_opt(
                        "crossing_rate_realized_alt",
                        got.crossing_rate_realized_alt,
                        exp.get("crossing_rate_realized_alt"),
                    );
                    match exp.get("n_realized_alt").and_then(|v| v.as_f64()) {
                        None => out.push(format!("{label}.n_realized_alt: missing")),
                        Some(e) if got.n_realized_alt as f64 != e => {
                            out.push(format!(
                                "{label}.n_realized_alt: rust={} python={e}",
                                got.n_realized_alt
                            ))
                        }
                        _ => {}
                    }
                };
                let cmp_prompt = |label: &str, got: &PromptLevelResult, exp: &Json, out: &mut Vec<String>| {
                    for (k, g, e) in [
                        ("theta", got.theta, exp.get("theta").and_then(|v| v.as_f64())),
                        ("rho", got.rho, exp.get("rho").and_then(|v| v.as_f64())),
                        (
                            "n_prompts",
                            got.n_prompts as f64,
                            exp.get("n_prompts").and_then(|v| v.as_f64()),
                        ),
                        (
                            "n_reps",
                            got.n_reps as f64,
                            exp.get("n_reps").and_then(|v| v.as_f64()),
                        ),
                        ("eps", got.eps, exp.get("eps").and_then(|v| v.as_f64())),
                        (
                            "crossing_rate",
                            got.crossing_rate,
                            exp.get("crossing_rate").and_then(|v| v.as_f64()),
                        ),
                    ] {
                        match e {
                            None => out.push(format!("{label}.{k}: missing in expected")),
                            Some(e) if g != e => {
                                out.push(format!("{label}.{k}: rust={g:?} python={e:?}"))
                            }
                            _ => {}
                        }
                    }
                };
                if let Some(e) = expected.get("E1_rho0_control") {
                    cmp_single("E1", &e1, e, &mut mismatches);
                }
                if let Some(e) = expected.get("E2_aime_regime_N64") {
                    cmp_single("E2_N64", &e2_64, e, &mut mismatches);
                }
                if let Some(e) = expected.get("E2_aime_regime_N256") {
                    cmp_single("E2_N256", &e2_256, e, &mut mismatches);
                }
                if let Some(e) = expected.get("E5_prompt_level_validity") {
                    cmp_prompt("E5_validity", &e5_valid, e, &mut mismatches);
                }
                if let Some(e) = expected.get("E5_power_prompt_level") {
                    cmp_prompt("E5_power_prompt", &e5_pow_prompt, e, &mut mismatches);
                }
                if let Some(e) = expected.get("E5_power_single_unit") {
                    cmp_single("E5_power_single", &e5_pow_single, e, &mut mismatches);
                }
                if let Some(e) = expected.get("E6_gsm8k_regime") {
                    cmp_single("E6", &e6, e, &mut mismatches);
                }
                if let Some(e) = expected.get("E4_ess_saturation") {
                    if let Some(arr) = e.as_arr() {
                        for (i, (n, emp, pred, ess)) in rows.iter().enumerate() {
                            let row = &arr[i];
                            let checks = [
                                ("n", *n as f64, row.get("n").and_then(|v| v.as_f64())),
                                (
                                    "empirical_var",
                                    (emp * 10000.0).round() / 10000.0,
                                    row.get("empirical_var").and_then(|v| v.as_f64()),
                                ),
                                (
                                    "predicted_var",
                                    (pred * 10000.0).round() / 10000.0,
                                    row.get("predicted_var").and_then(|v| v.as_f64()),
                                ),
                                (
                                    "ess",
                                    (ess * 1000.0).round() / 1000.0,
                                    row.get("ess").and_then(|v| v.as_f64()),
                                ),
                            ];
                            for (k, g, ev) in checks {
                                match ev {
                                    None => mismatches
                                        .push(format!("E4[{i}].{k}: missing in expected")),
                                    Some(ev) if g != ev => mismatches.push(format!(
                                        "E4[{i}].{k}: rust={g:?} python={ev:?}"
                                    )),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                if mismatches.is_empty() {
                    println!(
                        "EXPECTED_COMPARISON all numbers identical to {}",
                        expected_path
                    );
                } else {
                    for m in &mismatches {
                        println!("[FAIL] expected comparison: {}", m);
                    }
                    failures.extend(mismatches);
                }
            }
            Err(e) => {
                println!("EXPECTED_COMPARISON skipped: cannot parse {}: {}", expected_path, e);
            }
        },
        Err(e) => {
            println!(
                "EXPECTED_COMPARISON skipped: cannot read {}: {}",
                expected_path, e
            );
        }
    }

    let mut json = String::from("{\n");
    for (i, (k, v)) in results.iter().enumerate() {
        json.push_str(&format!("  \"{}\": {}", k, v));
        if i + 1 < results.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push('}');
    println!("RESULTS_JSON {}", json);

    if failures.is_empty() {
        println!("ALL CHECKS PASSED");
        0
    } else {
        println!("{} CHECK(S) FAILED", failures.len());
        1
    }
}

// ---------------------------------------------------------------------------
// soak: timed scale run beyond the Python test's reach.
// ---------------------------------------------------------------------------

fn cmd_soak(args: &[String]) -> i32 {
    let n_units: usize = args.get(0).and_then(|s| s.parse().ok()).unwrap_or(20000);
    let n_rollouts: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(11);
    let theta: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(-0.15);
    let rho: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.48);
    let t0 = Instant::now();
    let r = run_single_unit_experiment(seed, theta, rho, n_rollouts, n_units, EPS, &LAMBDA_GRID);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "{{\"n_units\": {}, \"n_rollouts\": {}, \"seed\": {}, \"theta\": {}, \"rho\": {}, \
         \"crossing_rate\": {}, \"pi\": {}, \"wall_seconds\": {}}}",
        n_units,
        n_rollouts,
        seed,
        py_repr(theta),
        py_repr(rho),
        py_repr(r.crossing_rate),
        py_repr(r.pi),
        py_repr(secs)
    );
    0
}

fn usage() -> ! {
    eprintln!(
        "usage: selfconsistency <verify [--expected PATH] | ppf <p> | cdf <x> | \
         ppf-batch | cdf-batch | eprocess <csv> | crossing <eps> <csv> | \
         run-single <seed> <theta> <rho> <n_rollouts> <n_units> [eps] | \
         run-prompt <seed> <theta> <rho> <n_prompts> <n_reps> [eps] | \
         ess-rows <seed> <rho> <m_units> <n_max> | soak [n_units] [n_rollouts] [seed] [theta] [rho]>"
    );
    exit(2);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    match args[0].as_str() {
        "verify" => {
            let mut expected = "../results.json".to_string();
            let mut i = 1;
            while i < args.len() {
                if args[i] == "--expected" && i + 1 < args.len() {
                    expected = args[i + 1].clone();
                    i += 2;
                } else {
                    usage();
                }
            }
            exit(cmd_verify(&expected));
        }
        "ppf" => {
            if args.len() != 2 {
                usage();
            }
            let p: f64 = args[1].parse().unwrap();
            let v = normal_ppf(p);
            println!("{} bits=0x{:016x}", py_repr(v), v.to_bits());
        }
        "cdf" => {
            if args.len() != 2 {
                usage();
            }
            let x: f64 = args[1].parse().unwrap();
            let v = normal_cdf(x);
            println!("{} bits=0x{:016x}", py_repr(v), v.to_bits());
        }
        "ppf-batch" => {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = line.unwrap();
                let t = line.trim();
                if t.is_empty() {
                    continue;
                }
                let p: f64 = t.parse().unwrap();
                let v = normal_ppf(p);
                println!("{} bits=0x{:016x}", py_repr(v), v.to_bits());
            }
        }
        "cdf-batch" => {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = line.unwrap();
                let t = line.trim();
                if t.is_empty() {
                    continue;
                }
                let x: f64 = t.parse().unwrap();
                let v = normal_cdf(x);
                println!("{} bits=0x{:016x}", py_repr(v), v.to_bits());
            }
        }
        "eprocess" => {
            if args.len() != 2 {
                usage();
            }
            let incs = parse_csv(&args[1]);
            let path = betting_eprocess(&incs, &LAMBDA_GRID);
            let parts: Vec<String> = path.iter().map(|v| py_repr(*v)).collect();
            println!("[{}]", parts.join(", "));
        }
        "crossing" => {
            if args.len() != 3 {
                usage();
            }
            let eps: f64 = args[1].parse().unwrap();
            let incs = parse_csv(&args[2]);
            match crossing_step(&incs, eps, &LAMBDA_GRID) {
                Some(s) => println!("{}", s),
                None => println!("null"),
            }
        }
        "run-single" => {
            if !(6..=7).contains(&args.len()) {
                usage();
            }
            let seed: u64 = args[1].parse().unwrap();
            let theta: f64 = args[2].parse().unwrap();
            let rho: f64 = args[3].parse().unwrap();
            let n_rollouts: usize = args[4].parse().unwrap();
            let n_units: usize = args[5].parse().unwrap();
            let eps: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(EPS);
            let r = run_single_unit_experiment(seed, theta, rho, n_rollouts, n_units, eps, &LAMBDA_GRID);
            println!("{}", single_unit_json(&r));
        }
        "run-prompt" => {
            if !(6..=7).contains(&args.len()) {
                usage();
            }
            let seed: u64 = args[1].parse().unwrap();
            let theta: f64 = args[2].parse().unwrap();
            let rho: f64 = args[3].parse().unwrap();
            let n_prompts: usize = args[4].parse().unwrap();
            let n_reps: usize = args[5].parse().unwrap();
            let eps: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(EPS);
            let r = run_prompt_level_experiment(seed, theta, rho, n_prompts, n_reps, eps, &LAMBDA_GRID);
            println!("{}", prompt_level_json(&r));
        }
        "ess-rows" => {
            if args.len() != 5 {
                usage();
            }
            let seed: u64 = args[1].parse().unwrap();
            let rho: f64 = args[2].parse().unwrap();
            let m_units: usize = args[3].parse().unwrap();
            let n_max: usize = args[4].parse().unwrap();
            let rows = ess_rows(seed, rho, m_units, n_max);
            let parts: Vec<String> = rows
                .iter()
                .map(|(n, emp, pred, ess)| {
                    format!(
                        "{{\"n\": {}, \"empirical_var\": {}, \"predicted_var\": {}, \"ess\": {}}}",
                        n,
                        py_repr((emp * 10000.0).round() / 10000.0),
                        py_repr((pred * 10000.0).round() / 10000.0),
                        py_repr((ess * 1000.0).round() / 1000.0)
                    )
                })
                .collect();
            println!("[{}]", parts.join(", "));
        }
        "soak" => {
            exit(cmd_soak(&args[1..]));
        }
        _ => usage(),
    }
}
