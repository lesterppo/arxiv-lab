//! Seeded synthetic multi-step agent-workflow corpus for WAAC.
//!
//! Rust port of `corpus.py`. Each workflow is a typed Trace-to-DAG of
//! 6-14 ops (search / fetch / extract / compute / verify / write /
//! audit) over one of five domains, with cross-step dependencies,
//! distractor long-text fields, and final-answer obligations. The
//! generator records the gold required-field set per step from its own
//! data-flow graph.
//!
//! Deterministic: `PyRandom` (CPython-compatible MT19937) only. The RNG
//! call order mirrors `corpus.py` line by line — argument evaluation
//! order included (`randint` before the `_long_text` call that consumes
//! it) — so the Rust corpus is bit-identical to the Python one for the
//! same seed. Verified by the exact `results.json` match in `verify`.

use crate::mt19937::PyRandom;
use crate::policy::Need;
use crate::waac::{Field, FieldKey, InputSpec, OpNode, TraceDAG};
use std::collections::BTreeMap;

// ---------------------------------------------------------------- domains
pub struct Domain {
    pub name: &'static str,
    pub topics: [&'static str; 5],
    pub titles: [&'static str; 3],
    pub numbers: [&'static str; 5],
    pub number_label: &'static str,
    pub dates: [&'static str; 3],
    pub fillers: [&'static str; 8],
}

pub const DOMAINS: [Domain; 5] = [
    Domain {
        name: "cooking",
        topics: ["sourdough bread", "miso ramen", "thai green curry",
                 "sourdough pizza", "kimchi fried rice"],
        titles: ["{topic} baking guide", "{topic} recipe walkthrough",
                 "{topic} technique notes"],
        numbers: ["180", "200", "220", "165", "190"],
        number_label: "oven temperature in celsius",
        dates: ["2026-03-14", "2025-11-02", "2026-01-20"],
        fillers: [
            "The {topic} method rewards patience and a hot oven.",
            "Many cooks rest the dough overnight for deeper flavour.",
            "A heavy pot holds steady heat through the long bake.",
            "Seasoning is adjusted at the end, never at the start.",
            "The kitchen fills with a warm, yeasty smell as it bakes.",
            "Leftovers keep well and reheat gently the next day.",
            "A sharp knife and a steady hand make the prep faster.",
            "The recipe scales cleanly from two servings to twelve.",
        ],
    },
    Domain {
        name: "travel",
        topics: ["kyoto in spring", "lisbon weekend", "taipei food tour",
                 "bergen fjords", "chiang mai retreat"],
        titles: ["{topic} itinerary", "{topic} travel plan",
                 "{topic} visitor guide"],
        numbers: ["148", "96", "210", "175", "132"],
        number_label: "hotel price per night in dollars",
        dates: ["2026-04-10", "2026-05-22", "2026-09-05"],
        fillers: [
            "The {topic} route is best walked in the early morning.",
            "Trains run often and tickets can be bought at the station.",
            "Local markets open before sunrise and close by noon.",
            "A light jacket is enough even after sunset.",
            "The old town is compact and easy to explore on foot.",
            "Booking ahead saves both time and money in peak season.",
            "Street food stalls cluster near the night market entrance.",
            "The harbour view at dusk is worth the short climb.",
        ],
    },
    Domain {
        name: "finance",
        topics: ["AAPL quarterly earnings", "TSLA delivery numbers",
                 "NVDA data-center revenue", "MSFT cloud growth",
                 "AMZN retail margins"],
        titles: ["{topic} summary", "{topic} earnings brief",
                 "{topic} analyst notes"],
        numbers: ["184.5", "212.3", "96.8", "341.2", "157.9"],
        number_label: "closing price in dollars",
        dates: ["2026-10-30", "2026-07-31", "2026-04-29"],
        fillers: [
            "The {topic} figures beat the street consensus this quarter.",
            "Management guided cautiously for the next two quarters.",
            "Margins expanded as supply costs eased through the year.",
            "Analysts raised targets after the call with investors.",
            "Cash flow remained strong despite higher capital spending.",
            "The buyback program continued at a steady quarterly pace.",
            "Segment growth was broad based across all regions.",
            "Guidance assumes no change in the current tax regime.",
        ],
    },
    Domain {
        name: "research",
        topics: ["retrieval-augmented generation", "tool-using agents",
                 "context compression", "agentic workflows",
                 "long-horizon planning"],
        titles: ["{topic} survey", "{topic} reproduction notes",
                 "{topic} literature review"],
        numbers: ["42", "128", "17", "256", "89"],
        number_label: "citation count of the lead paper",
        dates: ["2026-02-11", "2025-12-03", "2026-06-18"],
        fillers: [
            "The {topic} literature has grown quickly in the last year.",
            "Several groups report similar gains with different setups.",
            "Ablations show the retrieval step matters more than scale.",
            "The benchmark suite covers both short and long horizons.",
            "Reproduction requires careful control of the prompt template.",
            "Open implementations lag the reported numbers by a small gap.",
            "The method transfers well to a second, harder benchmark.",
            "Future work points toward multi-agent coordination.",
        ],
    },
    Domain {
        name: "shopping",
        topics: ["trail running shoes", "noise-cancelling headphones",
                 "mechanical keyboards", "espresso machines",
                 "ultralight backpacks"],
        titles: ["{topic} comparison", "{topic} buying guide",
                 "{topic} shortlist"],
        numbers: ["129", "249", "89", "349", "179"],
        number_label: "best price found in dollars",
        dates: ["2026-08-15", "2026-03-08", "2026-10-01"],
        fillers: [
            "The {topic} market splits into budget and premium tiers.",
            "Build quality matters more than the headline specification.",
            "Most buyers keep the product for at least three years.",
            "Warranty terms vary widely between the major brands.",
            "The mid-range option wins on value for most people.",
            "Colour choices are limited at the lower price points.",
            "Accessories add roughly a tenth to the total cost.",
            "Stock is seasonal and sells out before the holidays.",
        ],
    },
];

const SKELETONS: [(&str, &[&str]); 5] = [
    ("single", &["search", "fetch", "extract", "compute", "verify", "write"]),
    ("audit", &["search", "fetch", "extract", "compute", "verify", "write", "audit"]),
    ("two_source", &["search", "fetch", "extract", "search", "fetch", "extract", "compute", "verify", "write"]),
    ("retry", &["search", "fetch", "fetch", "extract", "compute", "verify", "write"]),
    ("two_audit", &["search", "fetch", "extract", "search", "fetch", "extract", "compute", "verify", "write", "audit"]),
];
const EXTRA_OPS: [&str; 3] = ["compute2", "verify2", "audit"];

/// Python `str(float)`: shortest repr, with `.0` for integral floats
/// (Rust `{:?}` on f64 matches Python `repr` for the value range used
/// here; both are shortest-round-trip).
pub fn py_float_str(x: f64) -> String {
    format!("{x:?}")
}

fn long_text(
    rng: &mut PyRandom,
    domain: &Domain,
    topic: &str,
    key_sentence: &str,
    n_sentences: i64,
) -> String {
    // Long prose whose FIRST sentence carries the key fact, so a SUMMARY
    // stub (first ~25 tokens) retains the required semantic facts while
    // a literal consumer still needs the full text.
    let mut pool: Vec<String> = domain
        .fillers
        .iter()
        .map(|f| f.replace("{topic}", topic))
        .collect();
    rng.shuffle(&mut pool);
    let mut sents: Vec<String> = vec![key_sentence.to_string()];
    let take = (n_sentences - 1).max(0) as usize;
    sents.extend(pool.into_iter().take(take));
    sents.join(" ")
}

fn fld(name: &str, ftype: &str, value: String, long_text: bool, artifact_id: Option<String>) -> Field {
    Field::new(name, ftype, value, long_text, artifact_id)
}

fn in_named(name: &str, source: Option<(usize, &str)>, need: Need) -> (String, InputSpec) {
    (name.to_string(), InputSpec::new(source, need))
}

/// One built workflow: DAG + gold required-field sets + obligations.
#[derive(Debug)]
pub struct Workflow {
    pub wid: usize,
    pub domain: String,
    pub topic: String,
    pub skeleton: String,
    pub n_ops: usize,
    pub verify_need: Need,
    pub write_need: Need,
    pub dag: TraceDAG,
    pub final_needs: Vec<(usize, String, Need)>,
    /// gold[t][(op_idx, field)] = joined capability set demanded by all
    /// consumers at steps >= t plus the final answer.
    pub gold: Vec<BTreeMap<FieldKey, u8>>,
    pub uncertain: Vec<FieldKey>,
    pub task_text: String,
}

/// Domain words for history fragments: [domain] + topic words.
pub fn domain_words(wf: &Workflow) -> Vec<String> {
    let mut w = vec![wf.domain.clone()];
    w.extend(wf.topic.split_whitespace().map(|s| s.to_string()));
    w
}

fn add_fetch(
    rng: &mut PyRandom,
    nodes: &mut Vec<OpNode>,
    domain: &Domain,
    topic: &str,
    number: &str,
    date: &str,
    wid: usize,
    url_source: (usize, &str),
    failed: bool,
) -> usize {
    if failed {
        nodes.push(OpNode::new(
            "fetch",
            vec![in_named("url", Some(url_source), Need::Literal)],
            vec![fld("page_status", "Status", "error".to_string(), false, None)],
        ));
        return nodes.len() - 1;
    }
    let key = format!(
        "The full page on {topic} states the {} is {number} and the date is {date}.",
        domain.number_label
    );
    let n = rng.randint(9, 12);
    let page_text = long_text(rng, domain, topic, &key, n);
    let n2 = rng.randint(8, 10);
    let sidebar = long_text(
        rng,
        domain,
        topic,
        "The sidebar lists adverts and unrelated headlines.",
        n2,
    );
    nodes.push(OpNode::new(
        "fetch",
        vec![in_named("url", Some(url_source), Need::Literal)],
        vec![
            fld("page_text", "PageText", page_text, true,
                Some(format!("art_{wid}_page"))),
            fld("page_status", "Status", "200".to_string(), false, None),
            fld("sidebar_text", "PageText", sidebar, true,
                Some(format!("art_{wid}_sidebar"))),
        ],
    ));
    nodes.len() - 1
}

fn add_extract(
    rng: &mut PyRandom,
    nodes: &mut Vec<OpNode>,
    domain: &Domain,
    topic: &str,
    number: &str,
    date: &str,
    title: &str,
    wid: usize,
    page_source: (usize, &str),
    tag: &str,
) -> usize {
    let key = format!(
        "Report on {topic}: the {} is {number}, the date is {date}, and the title is {title}.",
        domain.number_label
    );
    let n = rng.randint(9, 12);
    let report_body = long_text(rng, domain, topic, &key, n);
    let n2 = rng.randint(8, 10);
    let raw_notes = long_text(
        rng,
        domain,
        topic,
        "Raw notes hold unedited observations and stray remarks.",
        n2,
    );
    nodes.push(OpNode::new(
        "extract",
        vec![in_named("page_text", Some(page_source), Need::Semantic)],
        vec![
            fld("title", "Title", title.to_string(), false, None),
            fld("key_number", "Number", number.to_string(), false, None),
            fld("fact_date", "Date", date.to_string(), false, None),
            fld("report_body", "ReportBody", report_body, true,
                Some(format!("art_{wid}_report{tag}"))),
            fld("raw_notes", "PageText", raw_notes, true,
                Some(format!("art_{wid}_notes{tag}"))),
        ],
    ));
    nodes.len() - 1
}

#[allow(clippy::too_many_arguments)]
pub fn build_workflow(rng: &mut PyRandom, wid: usize) -> Workflow {
    // RNG call order mirrors corpus.py exactly.
    let domain = rng.choice(&DOMAINS);
    let topic = rng.choice(&domain.topics).to_string();
    let title = rng.choice(&domain.titles).replace("{topic}", &topic);
    let number = rng.choice(&domain.numbers).to_string();
    let date = rng.choice(&domain.dates).to_string();
    let verify_need = *rng.choice(&[Need::Literal, Need::Literal, Need::Semantic]);
    let write_need = *rng.choice(&[
        Need::Literal,
        Need::Literal,
        Need::Semantic,
        Need::Recoverable,
    ]);
    let (skel_name, skel) = *rng.choice(&SKELETONS);
    let mut ops: Vec<&str> = skel.to_vec();
    let n_extra = *rng.choice(&[0i64, 0, 1, 1, 2, 3]);
    for _ in 0..n_extra {
        if ops.len() >= 14 {
            break;
        }
        ops.push(rng.choice(&EXTRA_OPS));
    }

    let mut nodes: Vec<OpNode> = Vec::new();

    // -- op: search ------------------------------------------------------
    let key_sentence = format!(
        "The {} for {topic} is {number}, recorded on {date}.",
        domain.number_label
    );
    let n = rng.randint(9, 12);
    let results_page = long_text(rng, domain, &topic, &key_sentence, n);
    let n2 = rng.randint(8, 10);
    let related = long_text(
        rng,
        domain,
        &topic,
        "Related pages cover neighbouring topics and background reading.",
        n2,
    );
    nodes.push(OpNode::new(
        "search",
        vec![in_named("query", None, Need::Literal)],
        vec![
            fld("results_page", "PageText", results_page, true,
                Some(format!("art_{wid}_results"))),
            fld("result_url", "URL",
                format!("https://example.com/{}/{wid}", domain.name),
                false, None),
            fld("related_links", "PageText", related, true,
                Some(format!("art_{wid}_links"))),
        ],
    ));
    let search_i = nodes.len() - 1;

    let mut fetch_i: Option<usize> = None;
    let mut fetch2_i: Option<usize> = None;
    let mut extract_i: Option<usize> = None;
    let mut extract2_i: Option<usize> = None;
    let mut compute_i: Option<usize> = None;
    let mut verify_i: Option<usize> = None;
    let mut write_i: Option<usize> = None;
    let mut search2_i: Option<usize> = None;
    let mut seen_fetch = 0usize;
    let mut seen_extract = 0usize;
    let mut seen_search = 0usize;

    for op in ops {
        match op {
            "search" => {
                seen_search += 1;
                if seen_search == 1 {
                    continue; // first search already added
                }
                let n = rng.randint(9, 12);
                let rp = long_text(rng, domain, &topic, &key_sentence, n);
                let n2 = rng.randint(8, 10);
                let rl = long_text(
                    rng,
                    domain,
                    &topic,
                    "More related pages and background reading.",
                    n2,
                );
                nodes.push(OpNode::new(
                    "search",
                    vec![in_named("query", None, Need::Literal)],
                    vec![
                        fld("results_page", "PageText", rp, true,
                            Some(format!("art_{wid}_results2"))),
                        fld("result_url", "URL",
                            format!("https://example.com/{}/{wid}/b", domain.name),
                            false, None),
                        fld("related_links", "PageText", rl, true,
                            Some(format!("art_{wid}_links2"))),
                    ],
                ));
                search2_i = Some(nodes.len() - 1);
            }
            "fetch" => {
                seen_fetch += 1;
                if skel_name == "retry" && seen_fetch == 1 {
                    fetch_i = Some(add_fetch(
                        rng, &mut nodes, domain, &topic, &number, &date, wid,
                        (search_i, "result_url"), true,
                    ));
                } else if seen_fetch == 1 {
                    fetch_i = Some(add_fetch(
                        rng, &mut nodes, domain, &topic, &number, &date, wid,
                        (search_i, "result_url"), false,
                    ));
                } else if seen_fetch == 2 && skel_name == "retry" {
                    fetch_i = Some(add_fetch(
                        rng, &mut nodes, domain, &topic, &number, &date, wid,
                        (search_i, "result_url"), false,
                    ));
                } else {
                    fetch2_i = Some(add_fetch(
                        rng, &mut nodes, domain, &topic, &number, &date, wid,
                        (search2_i.unwrap(), "result_url"), false,
                    ));
                }
            }
            "extract" => {
                seen_extract += 1;
                if seen_extract == 1 {
                    extract_i = Some(add_extract(
                        rng, &mut nodes, domain, &topic, &number, &date, &title,
                        wid, (fetch_i.unwrap(), "page_text"), "",
                    ));
                } else {
                    extract2_i = Some(add_extract(
                        rng, &mut nodes, domain, &topic, &number, &date, &title,
                        wid, (fetch2_i.unwrap(), "page_text"), "2",
                    ));
                }
            }
            "compute" => {
                let mut inputs = vec![
                    in_named("key_number", Some((extract_i.unwrap(), "key_number")), Need::Literal),
                    in_named("fact_date", Some((extract_i.unwrap(), "fact_date")), Need::Literal),
                ];
                if let Some(e2) = extract2_i {
                    inputs.push(in_named(
                        "key_number2",
                        Some((e2, "key_number")),
                        Need::Literal,
                    ));
                }
                let computed = if number.contains('.') {
                    py_float_str(number.parse::<f64>().unwrap() * 2.0)
                } else {
                    (number.parse::<i64>().unwrap() * 2).to_string()
                };
                nodes.push(OpNode::new(
                    "compute",
                    inputs,
                    vec![
                        fld("computed_value", "Number", computed, false, None),
                        fld("check_flag", "Status", "pass".to_string(), false, None),
                    ],
                ));
                compute_i = Some(nodes.len() - 1);
            }
            "verify" => {
                let status_src = match fetch2_i {
                    Some(f2) => (f2, "page_status"),
                    None => (fetch_i.unwrap(), "page_status"),
                };
                nodes.push(OpNode::new(
                    "verify",
                    vec![
                        in_named("computed_value", Some((compute_i.unwrap(), "computed_value")), Need::Literal),
                        in_named("page_status", Some(status_src), Need::Literal),
                        in_named("page_text", Some((fetch_i.unwrap(), "page_text")), verify_need),
                    ],
                    vec![
                        fld("check_result", "Status", "ok".to_string(), false, None),
                        fld("verified_quote", "Quote",
                            format!("{} is {number}", domain.number_label),
                            false, None),
                    ],
                ));
                verify_i = Some(nodes.len() - 1);
            }
            "write" => {
                let body_src = match extract2_i {
                    Some(e2) => (e2, "report_body"),
                    None => (extract_i.unwrap(), "report_body"),
                };
                nodes.push(OpNode::new(
                    "write",
                    vec![
                        in_named("report_body", Some(body_src), write_need),
                        in_named("title", Some((extract_i.unwrap(), "title")), Need::Literal),
                        in_named("computed_value", Some((compute_i.unwrap(), "computed_value")), Need::Literal),
                    ],
                    vec![
                        fld("artifact_path", "ArtifactPath",
                            format!("/workspace/artifacts/report_{wid:03}.md"),
                            false, None),
                        fld("write_status", "Status", "ok".to_string(), false, None),
                    ],
                ));
                write_i = Some(nodes.len() - 1);
            }
            "compute2" => {
                nodes.push(OpNode::new(
                    "compute2",
                    vec![
                        in_named("computed_value", Some((compute_i.unwrap(), "computed_value")), Need::Literal),
                        in_named("key_number", Some((extract_i.unwrap(), "key_number")), Need::Literal),
                    ],
                    vec![fld(
                        "refined_value",
                        "Number",
                        py_float_str(number.parse::<f64>().unwrap() * 3.0),
                        false,
                        None,
                    )],
                ));
            }
            "verify2" => {
                nodes.push(OpNode::new(
                    "verify2",
                    vec![
                        in_named("computed_value", Some((compute_i.unwrap(), "computed_value")), Need::Literal),
                        in_named("page_text", Some((fetch_i.unwrap(), "page_text")), verify_need),
                    ],
                    vec![fld("check_result", "Status", "ok".to_string(), false, None)],
                ));
            }
            "audit" => {
                nodes.push(OpNode::new(
                    "audit",
                    vec![
                        in_named("artifact_path", Some((write_i.unwrap(), "artifact_path")), Need::Literal),
                        in_named("write_status", Some((write_i.unwrap(), "write_status")), Need::Literal),
                    ],
                    vec![fld("audit_result", "Status", "ok".to_string(), false, None)],
                ));
            }
            other => panic!("unknown op {other}"),
        }
    }
    // Silence unused warnings for indices kept for parity with Python.
    let _ = (verify_i, write_i);

    let n_ops = nodes.len();

    // Final-answer obligations: specific short-scalar fields, literal.
    let final_needs: Vec<(usize, String, Need)> = vec![
        (extract_i.unwrap(), "title".to_string(), Need::Literal),
        (compute_i.unwrap(), "computed_value".to_string(), Need::Literal),
        (write_i.unwrap(), "artifact_path".to_string(), Need::Literal),
        (write_i.unwrap(), "write_status".to_string(), Need::Literal),
    ];

    // Uncertain dependency evidence (Sec. 3.2): with prob. 0.15 the
    // auxiliary extractor proposes that raw_notes may be needed later,
    // but the relation cannot be reliably resolved -> the online
    // analysis must retain it EXACT conservatively. It is NOT a real
    // consumer, so gold excludes it.
    let mut uncertain: Vec<FieldKey> = Vec::new();
    if rng.random() < 0.15 {
        uncertain.push((extract_i.unwrap(), "raw_notes".to_string()));
    }

    let dag = TraceDAG::new(nodes, uncertain.clone());

    // Gold required-field sets per step, from the generator's data flow.
    let mut gold: Vec<BTreeMap<FieldKey, u8>> = Vec::new();
    for t in 0..=n_ops {
        let mut need: BTreeMap<FieldKey, u8> = BTreeMap::new();
        for j in t..n_ops {
            for spec in dag.nodes[j].inputs.iter().map(|(_, s)| s) {
                let Some(src) = &spec.source else { continue };
                if src.0 >= t {
                    continue;
                }
                let e = need.entry(src.clone()).or_insert(0);
                *e |= spec.need.caps();
            }
        }
        for (oi, fname, nd) in &final_needs {
            if *oi < t {
                let e = need.entry((*oi, fname.clone())).or_insert(0);
                *e |= nd.caps();
            }
        }
        gold.push(need);
    }

    Workflow {
        wid,
        domain: domain.name.to_string(),
        topic: topic.clone(),
        skeleton: skel_name.to_string(),
        n_ops,
        verify_need,
        write_need,
        dag,
        final_needs,
        gold,
        uncertain,
        task_text: format!(
            "Research {topic}: find the page, extract the {} and the date, verify the quote, write the report and give the final answer with the title, the computed value, the artifact path and the write status.",
            domain.number_label
        ),
    }
}

/// ~60 seeded workflows (6-14 ops each).
pub fn build_corpus(n_workflows: usize, seed: u64) -> Vec<Workflow> {
    let mut rng = PyRandom::new(seed);
    (0..n_workflows).map(|wid| build_workflow(&mut rng, wid)).collect()
}

/// Separate-seed development workflows for the history library.
pub fn build_dev_corpus(n_workflows: usize, seed: u64) -> Vec<Workflow> {
    let mut rng = PyRandom::new(seed);
    (0..n_workflows)
        .map(|wid| build_workflow(&mut rng, 1000 + wid))
        .collect()
}
