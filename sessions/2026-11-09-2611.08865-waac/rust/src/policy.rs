//! Capability-set policies (WAAC Sec. 3.2 / App. A.1).
//!
//! Policies are capability sets: DROP {} < SUMMARY {semantic},
//! POINTER {recoverable} (incomparable), EXACT {literal, semantic,
//! recoverable}. `fuse` is the least upper bound, so
//! SUMMARY v POINTER = EXACT: both capabilities are required and no
//! single non-EXACT representation covers both.
//!
//! Capabilities are bit flags: literal = 1, semantic = 2,
//! recoverable = 4. The fuse mapping reproduces the Python reference
//! exactly, including its conservative EXACT for the
//! {semantic, recoverable} union.

/// A retention policy for one field.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Policy {
    Drop,
    Summary,
    Pointer,
    Exact,
}

impl Policy {
    pub const ALL: [Policy; 4] = [
        Policy::Drop,
        Policy::Summary,
        Policy::Pointer,
        Policy::Exact,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Policy::Drop => "DROP",
            Policy::Summary => "SUMMARY",
            Policy::Pointer => "POINTER",
            Policy::Exact => "EXACT",
        }
    }

    pub fn from_name(s: &str) -> Option<Policy> {
        match s {
            "DROP" => Some(Policy::Drop),
            "SUMMARY" => Some(Policy::Summary),
            "POINTER" => Some(Policy::Pointer),
            "EXACT" => Some(Policy::Exact),
            _ => None,
        }
    }

    /// Capability bitmask: literal = 1, semantic = 2, recoverable = 4.
    pub fn caps(self) -> u8 {
        match self {
            Policy::Drop => 0,
            Policy::Summary => 0b010,
            Policy::Pointer => 0b100,
            Policy::Exact => 0b111,
        }
    }
}

/// A consumer need kind (App. A.1).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Need {
    Literal,
    Semantic,
    Recoverable,
    Unresolved,
}

impl Need {
    pub const ALL: [Need; 4] = [
        Need::Literal,
        Need::Semantic,
        Need::Recoverable,
        Need::Unresolved,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Need::Literal => "literal",
            Need::Semantic => "semantic",
            Need::Recoverable => "recoverable",
            Need::Unresolved => "unresolved",
        }
    }

    pub fn from_name(s: &str) -> Option<Need> {
        match s {
            "literal" => Some(Need::Literal),
            "semantic" => Some(Need::Semantic),
            "recoverable" => Some(Need::Recoverable),
            "unresolved" => Some(Need::Unresolved),
            _ => None,
        }
    }

    /// Capability set the representation must provide.
    pub fn caps(self) -> u8 {
        match self {
            Need::Literal => 0b001,
            Need::Semantic => 0b010,
            Need::Recoverable => 0b100,
            Need::Unresolved => 0b111,
        }
    }

    /// Minimum policy that satisfies this need (App. A.1).
    pub fn min_policy(self) -> Policy {
        match self {
            Need::Literal => Policy::Exact,
            Need::Semantic => Policy::Summary,
            Need::Recoverable => Policy::Pointer,
            Need::Unresolved => Policy::Exact,
        }
    }
}

/// Least upper bound of two policies under the preservation order.
///
/// `fuse(SUMMARY, POINTER) == EXACT` (incomparable capabilities, neither
/// alone suffices); fuse is the capability union mapped back to a policy.
pub fn fuse(a: Policy, b: Policy) -> Policy {
    match a.caps() | b.caps() {
        0b111 => Policy::Exact,
        0b110 => Policy::Exact, // {semantic, recoverable} -> conservative EXACT
        0b010 => Policy::Summary,
        0b100 => Policy::Pointer,
        _ => Policy::Drop, // 0, and the unreachable {literal} unions
    }
}

/// Join of any number of policies (associative / commutative).
pub fn fuse_many(policies: impl IntoIterator<Item = Policy>) -> Policy {
    let mut out = Policy::Drop;
    for p in policies {
        out = fuse(out, p);
    }
    out
}

/// Does `policy` (as a bare representation) satisfy a need kind?
pub fn policy_satisfies(policy: Policy, need: Need) -> bool {
    need.caps() & !policy.caps() == 0
}
