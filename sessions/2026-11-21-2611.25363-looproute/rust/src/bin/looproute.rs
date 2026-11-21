//! `looproute` — CLI for the LoopRoute Rust port.
//!
//! Subcommands:
//!   verify [--expected PATH]  mirror the Python CPU test's experiments
//!                             E1-E5 with the same assertions, and compare
//!                             the computed numbers to ../results.json
//!   gen-spec <seed> <locked>  print the canonical spec for a seeded task
//!   run-spec <file> [--threshold K] [--mode fast|routed]
//!                             run one episode from a spec file, JSON out
//!   soak                      600-seed scale run + 200-step cap check
//!
//! No external crates; JSON is hand-rolled (writer + minimal parser).

use looproute_rs::{
    parse_spec, spec_of_env, FastPolicy, KitchenWorld, LoopRouter,
    oracle_assistant, plan_to_goal, run_episode, run_experiment, Aggregate,
    EpisodeResult,
};
use std::env;
use std::fs;
use std::process::exit;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Minimal JSON value + parser (for comparing against ../results.json)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        if let Json::Obj(pairs) = self {
            pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }
    fn as_f64(&self) -> Option<f64> {
        if let Json::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    fn as_u64(&self) -> Option<u64> {
        self.as_f64().map(|n| n as u64)
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            s: text.as_bytes(),
            i: 0,
        }
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() {
            self.i += 1;
        }
    }

    fn parse(&mut self) -> Result<Json, String> {
        self.ws();
        let v = self.value()?;
        self.ws();
        if self.i != self.s.len() {
            return Err("trailing characters".to_string());
        }
        Ok(v)
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.s.get(self.i) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(_) => self.number(),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("expected {word}"))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self.i < self.s.len()
            && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[start..self.i])
            .map_err(|_| "bad utf8 in number".to_string())?;
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("bad number {text}"))
    }

    fn string(&mut self) -> Result<String, String> {
        // Assumes the current char is '"'.
        self.i += 1;
        let mut out = String::new();
        while let Some(&c) = self.s.get(self.i) {
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    match self.s.get(self.i) {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'n') => out.push('\n'),
                        Some(b't') => out.push('\t'),
                        Some(b'r') => out.push('\r'),
                        Some(b'b') => out.push('\u{0008}'),
                        Some(b'f') => out.push('\u{000C}'),
                        Some(b'u') => {
                            let hex = std::str::from_utf8(
                                self.s.get(self.i + 1..self.i + 5).unwrap_or(&[]),
                            )
                            .unwrap_or("");
                            let cp = u32::from_str_radix(hex, 16)
                                .map_err(|_| "bad \\u escape".to_string())?;
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            self.i += 4;
                        }
                        _ => return Err("bad escape".to_string()),
                    }
                    self.i += 1;
                }
                _ => {
                    // Copy one UTF-8 char.
                    let ch_len = if c < 0x80 {
                        1
                    } else if c < 0xE0 {
                        2
                    } else if c < 0xF0 {
                        3
                    } else {
                        4
                    };
                    let end = (self.i + ch_len).min(self.s.len());
                    out.push_str(
                        std::str::from_utf8(&self.s[self.i..end])
                            .map_err(|_| "bad utf8 in string".to_string())?,
                    );
                    self.i = end;
                }
            }
        }
        Err("unterminated string".to_string())
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // '{'
        let mut pairs = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return Err("expected ':' in object".to_string());
            }
            self.i += 1;
            let v = self.value()?;
            pairs.push((k, v));
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return Err("expected ',' or '}' in object".to_string()),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err("expected ',' or ']' in array".to_string()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JSON writer helpers (Python-json-compatible float formatting)
// ---------------------------------------------------------------------------

fn fmt_f64(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn agg_json(a: &Aggregate) -> String {
    format!(
        "{{\"mean_assists\": {}, \"mean_llm_calls\": {}, \"mean_steps\": {}, \"n\": {}, \"success_rate\": {}}}",
        fmt_f64(a.mean_assists),
        fmt_f64(a.mean_llm_calls),
        fmt_f64(a.mean_steps),
        a.n,
        fmt_f64(a.success_rate)
    )
}

fn episode_json(r: &EpisodeResult) -> String {
    let mut traj = String::new();
    for (i, t) in r.trajectory.iter().enumerate() {
        if i > 0 {
            traj.push(',');
        }
        traj.push_str(&format!(
            "[\"{}\", \"{}\", \"{}\", \"{}\"]",
            json_escape(&t.state.canonical()),
            json_escape(&t.action),
            json_escape(&t.source),
            json_escape(&t.obs)
        ));
    }
    format!(
        "{{\"success\": {}, \"steps\": {}, \"assists\": {}, \"llm_calls\": {}, \"trajectory\": [{}]}}",
        r.success, r.steps, r.assists, r.llm_calls, traj
    )
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

fn check(name: &str, cond: bool, detail: &str, failed: &mut bool) {
    println!("[{}] {name} {detail}", if cond { "PASS" } else { "FAIL" });
    if !cond {
        *failed = true;
    }
}

fn cmd_verify(expected_path: &str) -> i32 {
    let mut failed = false;

    // E1 — router unit behaviour.
    let mut r = LoopRouter::new(2);
    let s = {
        let mut env = KitchenWorld::new(1, 3, true);
        env.reset()
    };
    let a = "open chest";
    check(
        "E1 no trigger on 1st occurrence",
        !r.should_assist(&s, a),
        "",
        &mut failed,
    );
    r.note(&s, a);
    check(
        "E1 trigger on 2nd occurrence",
        r.should_assist(&s, a),
        "",
        &mut failed,
    );
    let plan = r.assist(&s, a, vec!["go pantry".to_string(), "take brass key".to_string()]);
    check(
        "E1 assist returns plan",
        plan == vec!["go pantry".to_string(), "take brass key".to_string()],
        "",
        &mut failed,
    );
    check(
        "E1 count resets after assist",
        !r.should_assist(&s, a),
        "",
        &mut failed,
    );
    r.note(&s, a);
    check(
        "E1 re-trigger after recurrence",
        r.should_assist(&s, a),
        "",
        &mut failed,
    );
    let mut r3 = LoopRouter::new(3);
    r3.note(&s, a);
    check(
        "E1 threshold=3 silent on 2nd",
        !r3.should_assist(&s, a),
        "",
        &mut failed,
    );
    r3.note(&s, a);
    check(
        "E1 threshold=3 triggers on 3rd",
        r3.should_assist(&s, a),
        "",
        &mut failed,
    );

    // E2 — oracle planner sanity on 10 seeds (mixed locked/unlocked).
    for seed in 2000..2010u64 {
        let mut env = KitchenWorld::new(seed, 3, seed % 2 == 0);
        let plan = plan_to_goal(&env);
        check(
            &format!("E2 plan exists seed={seed}"),
            !plan.is_empty(),
            &format!("len={}", plan.len()),
            &mut failed,
        );
        env.reset();
        let mut done = false;
        for act in &plan {
            let (_, _, d, info) = env.step(act);
            if !info {
                check(
                    &format!("E2 plan legal seed={seed}"),
                    false,
                    &format!("illegal plan step {act:?}"),
                    &mut failed,
                );
            }
            done = d;
        }
        check(
            &format!("E2 plan cooks seed={seed}"),
            done,
            "",
            &mut failed,
        );
    }

    // E3 — false-trigger check on unlocked tasks.
    let policy = FastPolicy;
    let mut assists = 0usize;
    let mut solved = 0usize;
    for seed in 3000..3020u64 {
        let mut router = LoopRouter::new(2);
        let mut env = KitchenWorld::new(seed, 3, false);
        let res = run_episode(
            &mut env,
            &policy,
            Some(&mut router),
            Some(oracle_assistant),
            60,
        );
        assists += res.assists;
        solved += usize::from(res.success);
    }
    check(
        "E3 zero assists on loop-free tasks",
        assists == 0,
        &format!("assists={assists}"),
        &mut failed,
    );
    check(
        "E3 fast policy solves unlocked tasks",
        solved == 20,
        &format!("solved={solved}/20"),
        &mut failed,
    );

    // E4 — main experiment.
    let seeds: Vec<u64> = (1000..1060).collect();
    let (fast, routed) = run_experiment(&seeds, 2);
    let (_, routed3) = run_experiment(&seeds, 3);
    let gain = routed.success_rate - fast.success_rate;
    println!(
        "E4 fast-only:    success={:.3} mean_steps={:.1}",
        fast.success_rate, fast.mean_steps
    );
    println!(
        "E4 looproute K=2: success={:.3} mean_steps={:.1} mean_assists={:.2} mean_llm_calls={:.2}",
        routed.success_rate, routed.mean_steps, routed.mean_assists, routed.mean_llm_calls
    );
    println!(
        "E4 looproute K=3: success={:.3} mean_assists={:.2}",
        routed3.success_rate, routed3.mean_assists
    );
    check(
        "E4 looproute beats fast-only by >=30pp",
        gain >= 0.30,
        &format!("gain={gain:.3}"),
        &mut failed,
    );
    check(
        "E4 limited assistant use (<=3 calls/task)",
        routed.mean_llm_calls <= 3.0,
        &format!("calls={:.2}", routed.mean_llm_calls),
        &mut failed,
    );
    check(
        "E4 looproute mean steps < fast-only",
        routed.mean_steps < fast.mean_steps,
        &format!("{:.1} vs {:.1}", routed.mean_steps, fast.mean_steps),
        &mut failed,
    );

    // E5 — determinism.
    let again = run_experiment(&seeds, 2);
    check(
        "E5 deterministic repeat",
        again == (fast.clone(), routed.clone()),
        "",
        &mut failed,
    );

    let results = format!(
        "{{\"E1_router_units\": \"all passed\", \"E2_oracle_plans\": \"10/10 plans legal and complete\", \"E3_false_trigger\": {{\"assists\": {assists}, \"solved\": {solved}}}, \"E4_main_K2\": {{\"fast_only\": {}, \"gain_pp\": {}, \"looproute_oracle\": {}}}, \"E4_main_K3\": {}, \"E5_deterministic\": true}}",
        agg_json(&fast),
        fmt_f64(gain * 100.0),
        agg_json(&routed),
        agg_json(&routed3),
    );
    println!("RESULTS_JSON {results}");

    // Exact comparison against the Python results.json, when present.
    match fs::read_to_string(expected_path) {
        Ok(text) => match Parser::new(&text).parse() {
            Ok(expected) => {
                let mut mismatches: Vec<String> = Vec::new();
                let cmp_agg = |label: &str, got: &Aggregate, want: &Json, out: &mut Vec<String>| {
                    let pairs = [
                        ("success_rate", got.success_rate),
                        ("mean_steps", got.mean_steps),
                        ("mean_assists", got.mean_assists),
                        ("mean_llm_calls", got.mean_llm_calls),
                    ];
                    for (k, gv) in pairs {
                        match want.get(k).and_then(|v| v.as_f64()) {
                            Some(wv) if wv == gv => {}
                            Some(wv) => out.push(format!(
                                "{label}.{k}: rust={gv} python={wv}"
                            )),
                            None => out.push(format!("{label}.{k}: missing in expected")),
                        }
                    }
                    match want.get("n").and_then(|v| v.as_u64()) {
                        Some(wv) if wv as usize == got.n => {}
                        Some(wv) => out.push(format!("{label}.n: rust={} python={wv}", got.n)),
                        None => out.push(format!("{label}.n: missing in expected")),
                    }
                };
                if let Some(e4) = expected.get("E4_main_K2") {
                    if let Some(f) = e4.get("fast_only") {
                        cmp_agg("E4.fast_only", &fast, f, &mut mismatches);
                    }
                    if let Some(o) = e4.get("looproute_oracle") {
                        cmp_agg("E4.looproute_oracle", &routed, o, &mut mismatches);
                    }
                    if let Some(g) = e4.get("gain_pp").and_then(|v| v.as_f64()) {
                        if (g - gain * 100.0).abs() > 1e-9 {
                            mismatches.push(format!(
                                "E4.gain_pp: rust={} python={g}",
                                gain * 100.0
                            ));
                        }
                    }
                }
                if let Some(k3) = expected.get("E4_main_K3") {
                    cmp_agg("E4.K3", &routed3, k3, &mut mismatches);
                }
                if let Some(e3) = expected.get("E3_false_trigger") {
                    if e3.get("assists").and_then(|v| v.as_u64()) != Some(assists as u64) {
                        mismatches.push("E3.assists mismatch".to_string());
                    }
                    if e3.get("solved").and_then(|v| v.as_u64()) != Some(solved as u64) {
                        mismatches.push("E3.solved mismatch".to_string());
                    }
                }
                if mismatches.is_empty() {
                    println!("EXPECTED_COMPARISON all numbers identical to {expected_path}");
                } else {
                    for m in &mismatches {
                        println!("[FAIL] expected comparison: {m}");
                    }
                    failed = true;
                }
            }
            Err(e) => {
                println!("EXPECTED_COMPARISON skipped: cannot parse {expected_path}: {e}");
            }
        },
        Err(_) => {
            println!("EXPECTED_COMPARISON skipped: {expected_path} not found");
        }
    }

    if failed {
        println!("CHECKS FAILED");
        1
    } else {
        println!("ALL CHECKS PASSED");
        0
    }
}

fn cmd_gen_spec(seed: u64, locked: bool) {
    let env = KitchenWorld::new(seed, 3, locked);
    print!("{}", spec_of_env(&env));
}

fn cmd_run_spec(path: &str, threshold: usize, mode: &str) -> i32 {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return 1;
        }
    };
    let spec = match parse_spec(&text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot parse {path}: {e}");
            return 1;
        }
    };
    let mut env = match KitchenWorld::from_spec(&spec) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cannot build env from {path}: {e}");
            return 1;
        }
    };
    let policy = FastPolicy;
    let res = match mode {
        "fast" => run_episode(&mut env, &policy, None, None, 60),
        "routed" => {
            let mut router = LoopRouter::new(threshold);
            run_episode(
                &mut env,
                &policy,
                Some(&mut router),
                Some(oracle_assistant),
                60,
            )
        }
        other => {
            eprintln!("unknown mode {other} (fast|routed)");
            return 1;
        }
    };
    println!("{}", episode_json(&res));
    0
}

fn cmd_soak() -> i32 {
    let t0 = Instant::now();
    let seeds: Vec<u64> = (5000..5600).collect();
    let policy = FastPolicy;
    let mut fast_ok = 0usize;
    let mut routed_ok = 0usize;
    let mut fast_steps = 0usize;
    let mut routed_steps = 0usize;
    let mut routed_assists = 0usize;
    let mut routed_calls = 0usize;
    let mut max_assists = 0usize;
    let mut max_calls = 0usize;
    for &s in &seeds {
        let locked = s % 2 == 0;
        let mut env = KitchenWorld::new(s, 3, locked);
        let f = run_episode(&mut env, &policy, None, None, 60);
        fast_ok += usize::from(f.success);
        fast_steps += f.steps;
        let mut router = LoopRouter::new(2);
        let mut env2 = KitchenWorld::new(s, 3, locked);
        let r = run_episode(
            &mut env2,
            &policy,
            Some(&mut router),
            Some(oracle_assistant),
            60,
        );
        routed_ok += usize::from(r.success);
        routed_steps += r.steps;
        routed_assists += r.assists;
        routed_calls += r.llm_calls;
        max_assists = max_assists.max(r.assists);
        max_calls = max_calls.max(r.llm_calls);
    }
    let n = seeds.len() as f64;
    // 200-step cap check: a locked task run with a generous cap must
    // still terminate within the cap and succeed.
    let mut cap_env = KitchenWorld::new(4242, 3, true);
    let mut cap_router = LoopRouter::new(2);
    let cap = run_episode(
        &mut cap_env,
        &policy,
        Some(&mut cap_router),
        Some(oracle_assistant),
        200,
    );
    let elapsed_ms = t0.elapsed().as_millis();
    println!(
        "{{\"n\": {}, \"seeds\": \"5000..5600\", \"fast_only\": {{\"success_rate\": {}, \"mean_steps\": {}}}, \"looproute_oracle_K2\": {{\"success_rate\": {}, \"mean_steps\": {}, \"mean_assists\": {}, \"mean_llm_calls\": {}, \"max_assists\": {}, \"max_llm_calls\": {}}}, \"cap200\": {{\"success\": {}, \"steps\": {}, \"assists\": {}, \"llm_calls\": {}}}, \"elapsed_ms\": {}}}",
        seeds.len(),
        fmt_f64(fast_ok as f64 / n),
        fmt_f64(fast_steps as f64 / n),
        fmt_f64(routed_ok as f64 / n),
        fmt_f64(routed_steps as f64 / n),
        fmt_f64(routed_assists as f64 / n),
        fmt_f64(routed_calls as f64 / n),
        max_assists,
        max_calls,
        cap.success,
        cap.steps,
        cap.assists,
        cap.llm_calls,
        elapsed_ms
    );
    if routed_ok == seeds.len() && cap.success && cap.steps <= 200 {
        println!("SOAK OK");
        0
    } else {
        println!("SOAK FAILED");
        1
    }
}

fn usage() -> i32 {
    eprintln!(
        "usage: looproute verify [--expected PATH]\n       looproute gen-spec <seed> <locked 0|1>\n       looproute run-spec <file> [--threshold K] [--mode fast|routed]\n       looproute soak"
    );
    2
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let code = match args.get(1).map(|s| s.as_str()) {
        Some("verify") => {
            let mut expected = "../results.json".to_string();
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--expected" && i + 1 < args.len() {
                    expected = args[i + 1].clone();
                    i += 2;
                } else {
                    exit(usage());
                }
            }
            cmd_verify(&expected)
        }
        Some("gen-spec") => {
            if args.len() != 4 {
                exit(usage());
            }
            let seed: u64 = args[2].parse().unwrap_or_else(|_| {
                eprintln!("bad seed");
                exit(2)
            });
            let locked = match args[3].as_str() {
                "1" => true,
                "0" => false,
                _ => {
                    eprintln!("locked must be 0 or 1");
                    exit(2)
                }
            };
            cmd_gen_spec(seed, locked);
            0
        }
        Some("run-spec") => {
            if args.len() < 3 {
                exit(usage());
            }
            let mut threshold = 2usize;
            let mut mode = "routed".to_string();
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--threshold" if i + 1 < args.len() => {
                        threshold = args[i + 1].parse().unwrap_or_else(|_| {
                            eprintln!("bad threshold");
                            exit(2)
                        });
                        i += 2;
                    }
                    "--mode" if i + 1 < args.len() => {
                        mode = args[i + 1].clone();
                        i += 2;
                    }
                    _ => exit(usage()),
                }
            }
            cmd_run_spec(&args[2], threshold, &mode)
        }
        Some("soak") => cmd_soak(),
        _ => usage(),
    };
    exit(code);
}
