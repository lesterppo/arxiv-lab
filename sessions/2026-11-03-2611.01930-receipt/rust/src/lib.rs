//! RECEIPT outcome model + waiting-prompt detector (arXiv 2611.01930).
//! Pure logic port of the Python module `arxiv_lab.agent.receipt`:
//! no subprocesses, no model, no network -- the live-subprocess part
//! of the Python test is out of scope for this port by design (see
//! rust/README.md). `simulate_outcome` must match Python exactly;
//! the parity sweep checks that on random cases.

/// (status, action_done, timed_out, false_success). status None = killed on timeout.
pub type Outcome = (Option<i32>, bool, bool, bool);

/// Harness input mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Closed,
    Open,
    Receipt,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "closed" => Some(Mode::Closed),
            "open" => Some(Mode::Open),
            "receipt" => Some(Mode::Receipt),
            _ => None,
        }
    }
}

pub fn simulate_outcome(guarded: bool, mode: Mode, policy_answers: bool) -> Outcome {
    if !guarded {
        return (Some(0), true, false, false);
    }
    match mode {
        // EOF on stdin: the program exits 0 having done nothing.
        Mode::Closed => (Some(0), false, false, true),
        // Nobody types: the process blocks until the harness timeout.
        Mode::Open => (None, false, true, false),
        Mode::Receipt => {
            if policy_answers {
                (Some(0), true, false, false)
            } else {
                (Some(0), false, false, false)
            }
        }
    }
}

/// Return the waiting question in `output`, or None.
///
/// Mirrors the Python detector: scan lines newest-first; a line is a
/// waiting prompt if, stripped, it ends with '?' or with a choice
/// marker ([y/n] / [yes/no], any case). Plain status output matches
/// neither.
pub fn extract_waiting_prompt(output: &str) -> Option<String> {
    for line in output.lines().rev() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let lower = t.to_lowercase();
        if t.ends_with('?') || lower.ends_with("[y/n]") || lower.ends_with("[yes/no]") {
            return Some(t.to_string());
        }
    }
    None
}
