//! Live extensive tests for the signed-manifest Rust port.
//! All run live with `cargo test`: known crypto vectors, unit tests for
//! each mechanism, property sweeps over random inputs, a 100k-case soak
//! checking the paper's invariants, and thread identity (stateless).

use signed_manifest::*;
use std::collections::HashSet;

// --- small deterministic PRNG (test-only) ---------------------------------
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// --- crypto vectors --------------------------------------------------------
#[test]
fn sha256_known_vectors() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex(&sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // 1,000,000 x 'a' (multi-block + padding edge).
    let big = vec![b'a'; 1_000_000];
    assert_eq!(
        hex(&sha256(&big)),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn hmac_sha256_rfc4231_case1() {
    // RFC 4231 test case 1: key = 0x0b * 20, data = "Hi There".
    let sig = hmac_sha256_hex(&[0x0b; 20], b"Hi There");
    assert_eq!(
        sig,
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    );
    // Different keys -> different signatures; tamper -> mismatch.
    let other = hmac_sha256_hex(b"other-key", b"Hi There");
    assert_ne!(sig, other);
}

// --- unit: extractor / predicates / verifier -------------------------------
#[test]
fn extractor_last_line_wins() {
    let text = build_task_text(&["env", "spend"], true);
    let reqs = extract_requirements(&text);
    let env = reqs.iter().find(|r| r.id == "env").unwrap();
    assert_eq!(env.allowed, Allowed::Set(&["production"]));
    assert!(reqs.iter().any(|r| r.id == "spend"));
}

#[test]
fn extractor_recovers_listed_requirements() {
    let text = build_task_text(&["env", "email", "payment", "logging"], false);
    let ids: HashSet<&str> = extract_requirements(&text).iter().map(|r| r.id).collect();
    assert_eq!(ids, ["env", "email", "payment", "logging"].into_iter().collect());
}

#[test]
fn predicates_accept_reject() {
    let ok = Decision {
        env: "staging".into(),
        email_scope: "internal".into(),
        spend_usd: 100,
        payment_approved: true,
        logging: true,
    };
    let bad = Decision {
        env: "production".into(),
        email_scope: "external".into(),
        spend_usd: 5000,
        payment_approved: false,
        logging: false,
    };
    for r in REQUIREMENTS {
        assert!(check_requirement(r, &ok), "{}", r.id);
        assert!(!check_requirement(r, &bad), "{}", r.id);
    }
}

#[test]
fn conformant_decision_satisfies_manifest() {
    let reqs = extract_requirements(&build_task_text(
        &["env", "email", "spend", "payment", "logging"],
        false,
    ));
    let d = Decision::conformant(&reqs);
    assert!(reqs.iter().all(|r| check_requirement(r, &d)));
    // ... and the injected manifest (production allowed) too.
    let reqs2 = extract_requirements(&build_task_text(&["env"], true));
    let d2 = Decision::conformant(&reqs2);
    assert!(reqs2.iter().all(|r| check_requirement(r, &d2)));
    assert_eq!(d2.env, "production");
}

#[test]
fn verifier_never_false_alarms() {
    let mut rng = Rng(99);
    for _ in 0..500 {
        let ids = random_ids(&mut rng, 4);
        let reqs = extract_requirements(&build_task_text(&ids, false));
        let d = Decision::conformant(&reqs);
        assert!(deterministic_verifier(&d, &reqs).is_empty());
    }
}

#[test]
fn sign_verify_roundtrip_and_tamper() {
    let reqs = extract_requirements(&build_task_text(&["env", "spend"], false));
    let text = build_task_text(&["env", "spend"], false);
    let sig = sign_manifest(&text, &reqs, MERGER_KEY);
    assert!(verify_manifest(&text, &reqs, &sig, MERGER_KEY));
    let mut tampered = reqs.clone();
    tampered[0].quote = "deploy only to production";
    assert!(!verify_manifest(&text, &tampered, &sig, MERGER_KEY));
    assert!(!verify_manifest(&text, &reqs, &sig, "wrong-key"));
}

// --- helpers ----------------------------------------------------------------
fn random_ids(rng: &mut Rng, k: usize) -> Vec<&'static str> {
    let mut ids: Vec<&'static str> = REQUIREMENTS.iter().map(|r| r.id).collect();
    for i in 0..k {
        let j = i + rng.below(ids.len() - i);
        ids.swap(i, j);
    }
    ids.truncate(k);
    ids
}

const FORMATS: [Format; 3] = [Format::FreeText, Format::JsonUnsigned, Format::SignedManifest];
const THREATS: [Threat; 6] = [
    Threat::None,
    Threat::Drop,
    Threat::Alter,
    Threat::Ignore,
    Threat::Mutate,
    Threat::Inject,
];

fn parse_out(s: &str) -> (bool, bool, bool) {
    let mut refused = false;
    let mut committed = false;
    let mut unsafe_ = false;
    for part in s.split('|') {
        if let Some(v) = part.strip_prefix("refused=") {
            refused = v == "1";
        } else if let Some(v) = part.strip_prefix("committed=") {
            committed = v == "1";
        } else if let Some(v) = part.strip_prefix("unsafe=") {
            unsafe_ = v == "1";
        }
    }
    (refused, committed, unsafe_)
}

// --- property sweeps + soak --------------------------------------------------
#[test]
fn property_signed_mutate_always_refused() {
    let mut rng = Rng(1234);
    for _ in 0..2000 {
        let k = 1 + rng.below(5);
        let ids = random_ids(&mut rng, k);
        let out = run_case_explicit(
            &ids,
            Format::SignedManifest,
            Threat::Mutate,
            rng.below(4),
            false,
        );
        let (refused, committed, unsafe_) = parse_out(&out);
        assert!(refused && !committed && !unsafe_, "{}", out);
    }
}

#[test]
fn property_signed_unsafe_only_via_inject() {
    // In this simulator a signed manifest commits an unsafe decision only
    // when the manifest itself was corrupted pre-signing (inject threat):
    // the signature authenticates the corrupted manifest.
    let mut rng = Rng(777);
    for _ in 0..5000 {
        let k = 1 + rng.below(5);
        let ids = random_ids(&mut rng, k);
        let threat = THREATS[rng.below(THREATS.len())];
        let out = run_case_explicit(
            &ids,
            Format::SignedManifest,
            threat,
            rng.below(4),
            threat == Threat::Inject,
        );
        let (_, _, unsafe_) = parse_out(&out);
        if unsafe_ {
            assert_eq!(threat, Threat::Inject, "{}", out);
        }
    }
}

#[test]
fn property_free_text_never_refuses() {
    let mut rng = Rng(55);
    for _ in 0..2000 {
        let k = 1 + rng.below(5);
        let ids = random_ids(&mut rng, k);
        let threat = THREATS[rng.below(THREATS.len())];
        let out = run_case_explicit(
            &ids,
            Format::FreeText,
            threat,
            rng.below(4),
            threat == Threat::Inject,
        );
        let (refused, _, _) = parse_out(&out);
        assert!(!refused, "{}", out);
    }
}

#[test]
fn soak_100k_no_panic_and_invariants_hold() {
    let mut rng = Rng(20261106);
    let mut counts = [0usize; 6]; // unsafe per (fmt) under attack
    for _ in 0..100_000 {
        let k = 1 + rng.below(5);
        let ids = random_ids(&mut rng, k);
        let fi = rng.below(3);
        let threat = THREATS[1 + rng.below(5)]; // attack threats only
        let out = run_case_explicit(
            &ids,
            FORMATS[fi],
            threat,
            rng.below(4),
            threat == Threat::Inject,
        );
        let (refused, committed, unsafe_) = parse_out(&out);
        assert!(!(refused && committed), "{}", out);
        assert!(!unsafe_ || committed, "{}", out);
        if unsafe_ {
            counts[fi] += 1;
        }
    }
    // Signed stays far below free-text under attack even at soak scale.
    assert!(counts[2] * 4 < counts[0], "{:?}", counts);
}

// --- thread identity (stateless) ---------------------------------------------
#[test]
fn thread_identity() {
    let cases: Vec<(Vec<&'static str>, Format, Threat, usize, bool)> = vec![
        (vec!["env", "spend"], Format::SignedManifest, Threat::Mutate, 0, false),
        (vec!["env", "email"], Format::SignedManifest, Threat::Inject, 1, true),
        (vec!["payment"], Format::JsonUnsigned, Threat::Alter, 0, false),
        (vec!["logging", "spend"], Format::FreeText, Threat::Drop, 1, false),
    ];
    let expected: Vec<String> = cases
        .iter()
        .map(|(ids, f, t, v, inj)| run_case_explicit(ids, *f, *t, *v, *inj))
        .collect();
    let mut handles = vec![];
    for _ in 0..8 {
        let cases = cases.clone();
        let expected = expected.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..2500 {
                for (i, (ids, f, t, v, inj)) in cases.iter().enumerate() {
                    let got = run_case_explicit(ids, *f, *t, *v, *inj);
                    assert_eq!(got, expected[i]);
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
