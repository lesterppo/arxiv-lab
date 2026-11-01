//! Agent-facing CLI for the Mingbird harness trio.
//!
//! Usage:
//!   arxiv-harness prefill --budget 600 --system "..." --schemas-file schemas.json
//!   arxiv-harness loop --actions-file actions.json [--repeat-k 3] [--max-cycle 4]
//!   arxiv-harness finish --fs-file fs.json --criteria-file criteria.json
//!   arxiv-harness verify        # re-run the Python test's 3 experiments
//!
//! `prefill` / `loop` / `finish` speak JSON on stdout so other agents can
//! call this binary per turn without a Python interpreter. `verify` is the
//! live-test verification: it mirrors `tests/test_harness_cpu.py` in
//! `lesterppo/arxiv-lab` experiment for experiment, with the same
//! bloated schemas, tasks, scripted policies and assertions.

use arxiv_harness_rs::jsonutil::{dumps_default, dumps_sorted};
use arxiv_harness_rs::{FinishGate, LoopDetector, PrefillBudget};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Read;
use std::process::ExitCode;

const USAGE: &str = "usage: arxiv-harness <prefill|loop|finish|verify> [options]\n  prefill --budget N --system TEXT (--schemas-file F | --schemas JSON)\n  loop [--repeat-k 3] [--max-cycle 4] (--actions-file F | --actions JSON | stdin)\n  finish --fs-file F --criteria-file F\n  verify";

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let cmd = argv[0].as_str();
    let rest = &argv[1..];
    match cmd {
        "prefill" => run_prefill(rest),
        "loop" => run_loop(rest),
        "finish" => run_finish(rest),
        "verify" => run_verify(),
        "--help" | "-h" | "help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {other}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn flag(rest: &[String], name: &str) -> Option<String> {
    let mut i = 0;
    while i < rest.len() {
        let a = &rest[i];
        if let Some(v) = a.strip_prefix(&format!("--{name}=")) {
            return Some(v.to_string());
        }
        if a == &format!("--{name}") && i + 1 < rest.len() {
            return Some(rest[i + 1].clone());
        }
        i += 1;
    }
    None
}

fn read_input(file_flag: Option<String>, inline_flag: Option<String>) -> Result<String, String> {
    if let Some(path) = file_flag {
        std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))
    } else if let Some(inline) = inline_flag {
        Ok(inline)
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("cannot read stdin: {e}"))?;
        Ok(buf)
    }
}

fn run_prefill(rest: &[String]) -> ExitCode {
    let budget: usize = match flag(rest, "budget").and_then(|v| v.parse().ok()) {
        Some(b) => b,
        None => {
            eprintln!("prefill: --budget N is required");
            return ExitCode::from(2);
        }
    };
    let system = flag(rest, "system").unwrap_or_default();
    let raw = match read_input(flag(rest, "schemas-file"), flag(rest, "schemas")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("prefill: {e}");
            return ExitCode::from(2);
        }
    };
    let parsed: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("prefill: invalid schemas JSON: {e}");
            return ExitCode::from(2);
        }
    };
    let schemas = match parsed.as_array() {
        Some(a) => a.clone(),
        None => {
            eprintln!("prefill: schemas JSON must be an array");
            return ExitCode::from(2);
        }
    };
    let (prefill, bytes_used, compressed) =
        PrefillBudget::new(budget).build_prefill(&system, &schemas);
    println!(
        "{}",
        serde_json::json!({
            "prefill": prefill,
            "bytes_used": bytes_used,
            "compressed": compressed,
        })
    );
    ExitCode::SUCCESS
}

fn run_loop(rest: &[String]) -> ExitCode {
    let repeat_k: usize = flag(rest, "repeat-k")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let max_cycle: usize = flag(rest, "max-cycle")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let raw = match read_input(flag(rest, "actions-file"), flag(rest, "actions")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("loop: {e}");
            return ExitCode::from(2);
        }
    };
    let parsed: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("loop: invalid actions JSON: {e}");
            return ExitCode::from(2);
        }
    };
    let actions = match parsed.as_array() {
        Some(a) => a.clone(),
        None => {
            eprintln!("loop: actions JSON must be an array");
            return ExitCode::from(2);
        }
    };
    let mut det = LoopDetector::new(repeat_k, max_cycle);
    let mut out = Vec::new();
    for a in &actions {
        let tool = a.get("tool").and_then(|t| t.as_str()).unwrap_or("");
        let args = a
            .get("args")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        let sig = LoopDetector::signature(tool, &args);
        let hit = det.observe(tool, &args);
        out.push(serde_json::json!({"tool": tool, "signature": sig, "loop": hit}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Array(out)).unwrap()
    );
    ExitCode::SUCCESS
}

fn run_finish(rest: &[String]) -> ExitCode {
    let fs_raw = match read_input(flag(rest, "fs-file"), flag(rest, "fs")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("finish: {e}");
            return ExitCode::from(2);
        }
    };
    let crit_raw = match read_input(flag(rest, "criteria-file"), flag(rest, "criteria")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("finish: {e}");
            return ExitCode::from(2);
        }
    };
    let fs_val: Value = match serde_json::from_str(&fs_raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("finish: invalid fs JSON: {e}");
            return ExitCode::from(2);
        }
    };
    let mut fs: HashMap<String, String> = HashMap::new();
    if let Value::Object(map) = &fs_val {
        for (k, v) in map {
            if let Some(s) = v.as_str() {
                fs.insert(k.clone(), s.to_string());
            }
        }
    }
    let crit_val: Value = match serde_json::from_str(&crit_raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("finish: invalid criteria JSON: {e}");
            return ExitCode::from(2);
        }
    };
    let mut criteria: Vec<(String, Box<dyn Fn(&HashMap<String, String>) -> bool>)> = Vec::new();
    if let Value::Array(items) = &crit_val {
        for c in items {
            let name = c
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let path = c
                .get("path")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_string();
            if let Some(expected) = c.get("equals").and_then(|e| e.as_str()) {
                let expected = expected.to_string();
                criteria.push((
                    name,
                    Box::new(move |env: &HashMap<String, String>| {
                        env.get(&path).map(|v| v.as_str()) == Some(expected.as_str())
                    }),
                ));
            } else {
                criteria.push((
                    name,
                    Box::new(move |env: &HashMap<String, String>| env.contains_key(&path)),
                ));
            }
        }
    }
    let task = flag(rest, "task").unwrap_or_default();
    let gate = FinishGate::new(task, criteria);
    let (accepted, failed) = gate.review(&fs);
    println!(
        "{}",
        serde_json::json!({"accepted": accepted, "failed": failed})
    );
    ExitCode::SUCCESS
}

// ── verify: mirror of tests/test_harness_cpu.py ──────────────────────────

#[derive(Clone, Default)]
struct Env {
    fs: HashMap<String, String>,
}

impl Env {
    fn new() -> Self {
        let mut fs = HashMap::new();
        fs.insert("/data/nums.txt".to_string(), "3\n7\n11\n".to_string());
        fs.insert("/data/flag.txt".to_string(), "yes\n".to_string());
        Self { fs }
    }
    fn read(&self, path: &str) -> Option<&str> {
        self.fs.get(path).map(|s| s.as_str())
    }
    fn write(&mut self, path: &str, content: &str) {
        self.fs.insert(path.to_string(), content.to_string());
    }
}

fn bloated_schemas() -> Vec<Value> {
    let read_example = serde_json::json!({"in": {"path": "/a"}, "out": "x"});
    let write_example = serde_json::json!({"in": {"path": "/b", "content": "y"}, "out": "ok"});
    vec![
        serde_json::json!({
            "name": "read",
            "description": "Read a file from the virtual filesystem. ".repeat(40),
            "parameters": {"path": {"type": "string", "description": "Absolute path. ".repeat(30)}},
            "examples": vec![read_example; 25],
        }),
        serde_json::json!({
            "name": "write",
            "description": "Write content to a file, creating parents. ".repeat(40),
            "parameters": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Full text. ".repeat(30)}
            },
            "examples": vec![write_example; 25],
        }),
    ]
}

const SYSTEM: &str = "You are a file-task agent. Use tools.\n";

fn sum_crit(env: &Env) -> bool {
    env.read("/out/sum.txt") == Some("21\n")
}
fn log_crit(env: &Env) -> bool {
    env.read("/out/log.txt") == Some("done\n")
}
fn status_crit(env: &Env) -> bool {
    env.read("/out/status.txt") == Some("verified\n")
}

struct Task {
    text: &'static str,
    criteria: Vec<(&'static str, fn(&Env) -> bool)>,
}

fn sum_task() -> Task {
    Task {
        text: "Write the sum of the numbers in /data/nums.txt to /out/sum.txt",
        criteria: vec![("sum_artifact", sum_crit)],
    }
}

fn gate_task() -> Task {
    Task {
        text: "If /data/flag.txt exists, append the line 'done' to /out/log.txt; then write the word 'verified' to /out/status.txt",
        criteria: vec![("log_artifact", log_crit), ("status_artifact", status_crit)],
    }
}

enum Action {
    Tool(String, Value),
    Finish,
}

trait Policy {
    fn next(&mut self, env: &Env) -> Option<Action>;
}

struct BloatPolicy {
    stage: usize,
}
impl Policy for BloatPolicy {
    fn next(&mut self, env: &Env) -> Option<Action> {
        let a = match self.stage {
            0 => Action::Tool(
                "read".to_string(),
                serde_json::json!({"path": "/data/nums.txt"}),
            ),
            1 => {
                let nums: Vec<i64> = env
                    .read("/data/nums.txt")
                    .unwrap_or("")
                    .split_whitespace()
                    .filter_map(|x| x.parse().ok())
                    .collect();
                let sum: i64 = nums.iter().sum();
                Action::Tool(
                    "write".to_string(),
                    serde_json::json!({"path": "/out/sum.txt", "content": format!("{sum}\n")}),
                )
            }
            2 => Action::Finish,
            _ => return None,
        };
        self.stage += 1;
        Some(a)
    }
}

struct PrematurePolicy {
    stage: usize,
}
impl Policy for PrematurePolicy {
    fn next(&mut self, _env: &Env) -> Option<Action> {
        let a = match self.stage {
            0 => Action::Tool(
                "write".to_string(),
                serde_json::json!({"path": "/out/log.txt", "content": "done\n"}),
            ),
            1 => Action::Finish,
            _ => return None,
        };
        self.stage += 1;
        Some(a)
    }
}

struct LooperPolicy {
    n: usize,
}
impl Policy for LooperPolicy {
    fn next(&mut self, _env: &Env) -> Option<Action> {
        if self.n < 60 {
            self.n += 1;
            Some(Action::Tool(
                "read".to_string(),
                serde_json::json!({"path": "/data/nums.txt"}),
            ))
        } else if self.n == 60 {
            self.n += 1;
            Some(Action::Finish)
        } else {
            None
        }
    }
}

#[derive(Default)]
struct RunResult {
    done: bool,
    steps: usize,
    prefill: usize,
    loop_broken: bool,
    score: f64,
    gate_failed: Vec<String>,
}

fn run_policy(
    policy: &mut dyn Policy,
    task: &Task,
    use_budget: bool,
    use_gate: bool,
    use_loopdet: bool,
    budget_bytes: usize,
    max_steps: usize,
) -> RunResult {
    let mut env = Env::new();
    let budget = PrefillBudget::new(budget_bytes);
    let gate = FinishGate::new(
        task.text,
        task.criteria
            .iter()
            .map(|(name, f)| {
                let f = *f;
                (
                    name.to_string(),
                    Box::new(move |e: &Env| f(e)) as Box<dyn Fn(&Env) -> bool>,
                )
            })
            .collect(),
    );
    let mut loopdet = LoopDetector::default();
    let mut steps = 0usize;
    let mut finish_accepted = false;
    let mut loop_broken = false;
    let mut prefill_bytes: Option<usize> = None;

    while let Some(action) = policy.next(&env) {
        match action {
            Action::Finish => {
                if use_gate {
                    let (ok, failed) = gate.review(&env);
                    finish_accepted = ok;
                    if !ok {
                        let score = task.criteria.iter().filter(|(_, c)| c(&env)).count() as f64
                            / task.criteria.len() as f64;
                        return RunResult {
                            done: false,
                            steps,
                            prefill: 0,
                            loop_broken,
                            score,
                            gate_failed: failed,
                        };
                    }
                } else {
                    finish_accepted = true;
                }
                break;
            }
            Action::Tool(tool, args) => {
                if use_loopdet && loopdet.observe(&tool, &args).is_some() {
                    loop_broken = true;
                    break;
                }
                if steps == 0 {
                    prefill_bytes = Some(if use_budget {
                        let (_, used, _) = budget.build_prefill(SYSTEM, &bloated_schemas());
                        used
                    } else {
                        let joined = bloated_schemas()
                            .iter()
                            .map(dumps_default)
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("{SYSTEM}{joined}").as_bytes().len()
                    });
                }
                steps += 1;
                if steps >= max_steps {
                    break;
                }
                match tool.as_str() {
                    "read" => {
                        let _ = args
                            .get("path")
                            .and_then(|p| p.as_str())
                            .map(|p| env.read(p));
                    }
                    "write" => {
                        if let (Some(p), Some(c)) = (
                            args.get("path").and_then(|v| v.as_str()),
                            args.get("content").and_then(|v| v.as_str()),
                        ) {
                            env.write(p, c);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    let score =
        task.criteria.iter().filter(|(_, c)| c(&env)).count() as f64 / task.criteria.len() as f64;
    RunResult {
        done: finish_accepted,
        steps,
        prefill: prefill_bytes.unwrap_or(0),
        loop_broken,
        score,
        gate_failed: Vec::new(),
    }
}

fn run_verify() -> ExitCode {
    const CONTEXT: usize = 4096;
    let sum = sum_task();
    let gate = gate_task();

    let r_off = run_policy(
        &mut BloatPolicy { stage: 0 },
        &sum,
        false,
        false,
        false,
        600,
        60,
    );
    let r_on = run_policy(
        &mut BloatPolicy { stage: 0 },
        &sum,
        true,
        false,
        false,
        600,
        60,
    );
    let g_off = run_policy(
        &mut PrematurePolicy { stage: 0 },
        &gate,
        false,
        false,
        false,
        600,
        60,
    );
    let g_on = run_policy(
        &mut PrematurePolicy { stage: 0 },
        &gate,
        false,
        true,
        false,
        600,
        60,
    );
    let l_off = run_policy(
        &mut LooperPolicy { n: 0 },
        &sum,
        false,
        false,
        false,
        600,
        60,
    );
    let l_on = run_policy(
        &mut LooperPolicy { n: 0 },
        &sum,
        false,
        false,
        true,
        600,
        60,
    );

    // Rows mirror the Python test's table: (condition, measured, verdict).
    let rows: Vec<(String, String, String)> = vec![
        (
            "prefill OFF: bytes in context".to_string(),
            r_off.prefill.to_string(),
            if r_off.prefill > CONTEXT {
                "OVERFLOW".to_string()
            } else {
                "fits".to_string()
            },
        ),
        (
            "prefill ON : bytes in context".to_string(),
            r_on.prefill.to_string(),
            if r_on.prefill <= 600 {
                "fits".to_string()
            } else {
                "OVER BUDGET".to_string()
            },
        ),
        (
            "gate OFF: premature finish accepted?".to_string(),
            g_off.done.to_string(),
            format!("artifact score {:.2} (silently incomplete)", g_off.score),
        ),
        (
            "gate ON : premature finish accepted?".to_string(),
            g_on.done.to_string(),
            format!(
                "rejected, missing: {:?} (score if completed: {:.2})",
                g_on.gate_failed, g_on.score
            ),
        ),
        (
            "loopdet OFF: steps burned".to_string(),
            l_off.steps.to_string(),
            "ran to max_steps".to_string(),
        ),
        (
            "loopdet ON : steps burned".to_string(),
            l_on.steps.to_string(),
            if l_on.loop_broken {
                "broken".to_string()
            } else {
                "NOT broken".to_string()
            },
        ),
    ];

    println!("{:42} {:>12}  verdict", "condition", "measured");
    println!("{}", "-".repeat(90));
    for (cond, val, verdict) in &rows {
        println!("{cond:42} {val:>12}  {verdict}");
    }

    let checks = [
        r_off.prefill > CONTEXT,
        r_on.prefill <= 600,
        g_off.done && rows[2].2.starts_with("artifact score 0.50"),
        !g_on.done,
        l_off.steps == 60,
        l_on.steps <= 4,
    ];
    // Library-level regression: build_prefill must not mutate caller schemas.
    let before = dumps_sorted(&Value::Array(bloated_schemas()));
    PrefillBudget::new(600).build_prefill(SYSTEM, &bloated_schemas());
    let after = dumps_sorted(&Value::Array(bloated_schemas()));
    let checks = checks.iter().all(|c| *c) && before == after;

    if checks {
        println!("\nALL HARNESS CHECKS PASSED.");
        ExitCode::SUCCESS
    } else {
        eprintln!("\nMECHANISM CHECK FAILED");
        ExitCode::from(1)
    }
}
