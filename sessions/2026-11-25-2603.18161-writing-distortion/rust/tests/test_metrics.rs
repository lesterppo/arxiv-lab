//! Unit tests for the Rust port of `writing_distortion_metrics.py`.
//! Mirrors the hand-computed expectations of the Python CPU test
//! plus edge cases for each function.

use writing_distortion::*;

const EPS: f64 = 1e-9;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS
}

// --- tokenize ---

#[test]
fn tokenize_lowercases_and_splits() {
    assert_eq!(tokenize("Hello, WORLD! 123 go."), vec!["hello", "world", "123", "go"]);
}

#[test]
fn tokenize_empty_and_punctuation_only() {
    assert!(tokenize("").is_empty());
    assert!(tokenize("... !!! ---").is_empty());
}

#[test]
fn tokenize_keeps_digits_inside_words() {
    assert_eq!(tokenize("abc123 def"), vec!["abc123", "def"]);
}

// --- unigram_jsd ---

#[test]
fn jsd_identical_is_exactly_zero() {
    let t = "Self driving cars are safer and more efficient than human drivers.";
    assert_eq!(unigram_jsd(t, t), 0.0);
    assert_eq!(unigram_jsd("", ""), 0.0);
}

#[test]
fn jsd_empty_vs_text() {
    // Smoothing makes an empty text uniform over the vocabulary, so
    // empty vs a text with each word once is exactly 0.0; empty vs a
    // skewed text is small but positive (Python: 0.09328454978990909).
    assert_eq!(unigram_jsd("", "cooking rice requires water"), 0.0);
    let j = unigram_jsd("", "a a a a a b");
    assert!(j > 0.0 && j < 0.2, "got {}", j);
    assert!(approx(j, 0.09328454978990909), "got {}", j);
}

#[test]
fn jsd_range_and_positive_for_different_texts() {
    let j = unigram_jsd(
        "Self driving cars may seem too futuristic to believe.",
        "Cooking rice requires water heat and patience every day.",
    );
    assert!(j > 0.0 && j <= 1.0, "got {}", j);
}

#[test]
fn jsd_symmetric_and_deterministic() {
    let (a, b) = ("the cat sat on the mat", "dogs bark loudly at night today");
    assert_eq!(unigram_jsd(a, b), unigram_jsd(a, b));
    assert!(approx(unigram_jsd(a, b), unigram_jsd(b, a)));
}

#[test]
fn jsd_single_word_swap_is_small() {
    // One word different out of ten: small but nonzero
    // (Python value 0.09999795257567473).
    let j = unigram_jsd(
        "one two three four five six seven eight nine ten",
        "one two three four five six seven eight nine eleven",
    );
    assert!(j > 0.0 && j < 0.11, "got {}", j);
    assert!(approx(j, 0.09999795257567473), "got {}", j);
}

// --- pronoun_rate ---

#[test]
fn pronoun_rate_hand_computed() {
    let (rate, count, n) = pronoun_rate("I am happy. We are here.");
    assert_eq!((count, n), (2, 6));
    assert!(approx(rate, 100.0 * 2.0 / 6.0));
}

#[test]
fn pronoun_rate_empty() {
    assert_eq!(pronoun_rate(""), (0.0, 0, 0));
}

#[test]
fn pronoun_rate_case_insensitive_and_all_pronouns() {
    let (rate, count, n) = pronoun_rate("I YOU He She They We");
    assert_eq!((count, n), (6, 6));
    assert!(approx(rate, 100.0));
}

#[test]
fn pronoun_rate_no_pronouns() {
    let (rate, count, n) = pronoun_rate("cars drive roads fast");
    assert_eq!((count, n), (0, 4));
    assert_eq!(rate, 0.0);
}

// --- adjective_rate ---

#[test]
fn adjective_rate_suffix_proxy_hand_computed() {
    let (rate, count, n) = adjective_rate("beautiful wonderful dangerous cat dog run");
    assert_eq!((count, n), (3, 6));
    assert!(approx(rate, 50.0));
}

#[test]
fn adjective_rate_short_suffix_words_not_counted() {
    // "able" (len 4) must NOT count despite the "able" suffix; the
    // Python rule requires len >= 5.
    let (_, count, _) = adjective_rate("able al ic");
    assert_eq!(count, 0);
    // "capable" (len 7, ends "able") counts; "national" counts twice
    // over via suffix only (counted once).
    let (_, count, _) = adjective_rate("capable national");
    assert_eq!(count, 2);
}

#[test]
fn adjective_rate_lexicon_words_count_once() {
    // "safe" is in the lexicon (len 4, no suffix hit) -> counts once.
    // "local" hits both suffix "al" and lexicon -> still counted once.
    let (_, count, n) = adjective_rate("safe local good");
    assert_eq!((count, n), (3, 3));
}

#[test]
fn adjective_rate_empty() {
    assert_eq!(adjective_rate(""), (0.0, 0, 0));
}

// --- keyword_stance ---

#[test]
fn stance_pro_threshold_exactly_three() {
    let s = keyword_stance("safer safety efficient");
    assert_eq!((s.pro, s.anti, s.score), (3, 0, 3));
    assert_eq!(s.label, "pro");
}

#[test]
fn stance_anti_threshold_exactly_minus_three() {
    let s = keyword_stance("dangerous crash risk");
    assert_eq!((s.pro, s.anti, s.score), (0, 3, -3));
    assert_eq!(s.label, "anti");
}

#[test]
fn stance_below_threshold_is_mixed() {
    let s = keyword_stance("safer efficient dangerous crash");
    assert_eq!((s.pro, s.anti, s.score), (2, 2, 0));
    assert_eq!(s.label, "mixed");
}

#[test]
fn stance_multiword_phrases_count() {
    // NOTE: each list entry is counted independently, so "reduce
    // accidents" counts as pro AND the bare word "accidents" inside
    // it counts as anti -- that is the Python behaviour too (verified
    // against the Python module), not a port bug.
    let s = keyword_stance("We can reduce accidents and save lives every day.");
    assert_eq!((s.pro, s.anti), (2, 1));
    let s = keyword_stance("Fear of job loss and unemployment grows.");
    assert_eq!((s.pro, s.anti), (0, 2));
}

#[test]
fn stance_whole_word_only() {
    // "safety" inside "safetynet" must not count; "kill" inside
    // "skill" must not count; "jobs" as a standalone word counts.
    let s = keyword_stance("safetynet skill jobs");
    assert_eq!((s.pro, s.anti), (0, 1));
}

#[test]
fn stance_case_insensitive() {
    let s = keyword_stance("SAFER Safety EFFICIENT");
    assert_eq!((s.pro, s.anti, s.score), (3, 0, 3));
    assert_eq!(s.label, "pro");
}

#[test]
fn stance_empty_is_mixed_zero() {
    let s = keyword_stance("");
    assert_eq!((s.pro, s.anti, s.score), (0, 0, 0));
    assert_eq!(s.label, "mixed");
}

#[test]
fn stance_overlapping_phrases_each_count() {
    // "save lives" and "saves lives" are distinct entries; a text
    // with both counts both.
    let s = keyword_stance("save lives saves lives save time");
    assert_eq!(s.pro, 3);
}

// --- pct_change ---

#[test]
fn pct_change_values() {
    assert_eq!(pct_change(100.0, 150.0), 50.0);
    assert_eq!(pct_change(100.0, 50.0), -50.0);
    assert_eq!(pct_change(100.0, 100.0), 0.0);
    assert_eq!(pct_change(0.0, 0.0), 0.0);
    assert!(pct_change(0.0, 5.0).is_infinite());
    // Python returns +inf even when after is negative.
    assert_eq!(pct_change(0.0, -5.0), f64::INFINITY);
}

// --- mean_shift_alignment ---

#[test]
fn align_identical_shifts_near_one() {
    let v = mean_shift_alignment(
        &[vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 1.0]],
        &[vec![1.0, 0.0], vec![2.0, 2.0], vec![4.0, 1.0]],
    )
    .unwrap();
    assert!(approx(v, 1.0), "got {}", v);
}

#[test]
fn align_orthogonal_shifts() {
    let v = mean_shift_alignment(
        &[vec![0.0, 0.0], vec![0.0, 0.0]],
        &[vec![1.0, 0.0], vec![0.0, 1.0]],
    )
    .unwrap();
    assert!(approx(v, std::f64::consts::FRAC_1_SQRT_2), "got {}", v);
}

#[test]
fn align_opposing_shifts_low() {
    let v = mean_shift_alignment(
        &[vec![0.0, 0.0], vec![0.0, 0.0]],
        &[vec![1.0, 0.0], vec![-1.0, 0.0]],
    )
    .unwrap();
    assert!(v < 0.1, "got {}", v);
}

#[test]
fn align_single_pair_is_one() {
    let v = mean_shift_alignment(&[vec![1.0, 1.0]], &[vec![4.0, 5.0]]).unwrap();
    assert!(approx(v, 1.0), "got {}", v);
}

#[test]
fn align_errors() {
    assert!(mean_shift_alignment(&[], &[]).is_err());
    assert!(mean_shift_alignment(&[vec![0.0]], &[vec![1.0], vec![2.0]]).is_err());
    assert!(mean_shift_alignment(&[vec![0.0, 0.0]], &[vec![1.0]]).is_err());
    // Zero shift vector cannot be normalized.
    assert!(mean_shift_alignment(&[vec![1.0, 1.0]], &[vec![1.0, 1.0]]).is_err());
}

#[test]
fn align_result_in_range() {
    let v = mean_shift_alignment(
        &[vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        &[vec![1.0, 1.0, 0.0], vec![2.0, 1.0, 1.0], vec![1.0, 2.0, 0.0]],
    )
    .unwrap();
    assert!((-1.0..=1.0).contains(&v), "got {}", v);
}
