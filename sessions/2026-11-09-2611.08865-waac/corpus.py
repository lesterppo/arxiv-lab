"""Seeded synthetic multi-step agent-workflow corpus for WAAC.

Each workflow is a typed Trace-to-DAG of 6-14 ops (search / fetch /
extract / compute / verify / write / audit) over one of five domains.
Ops consume typed fields produced by earlier ops -- including fields
consumed several steps later (cross-step dependencies) -- and long
distractor fields that are never consumed. Final-answer obligations
require specific short-scalar fields. The generator records the gold
required-field set per step: for every field produced before step t,
the joined capability set demanded by all consumers at steps >= t plus
the final answer, derived from the generator's own data-flow graph.

Deterministic: `random.Random(seed)` only. No network, no model.
"""

import random

from waac import Field, OpNode, TraceDAG

# ---------------------------------------------------------------- domains
_DOMAINS = {
    "cooking": {
        "topics": ["sourdough bread", "miso ramen", "thai green curry",
                   "sourdough pizza", "kimchi fried rice"],
        "titles": ["{topic} baking guide", "{topic} recipe walkthrough",
                   "{topic} technique notes"],
        "numbers": ["180", "200", "220", "165", "190"],
        "number_label": "oven temperature in celsius",
        "dates": ["2026-03-14", "2025-11-02", "2026-01-20"],
        "fillers": [
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
    "travel": {
        "topics": ["kyoto in spring", "lisbon weekend", "taipei food tour",
                   "bergen fjords", "chiang mai retreat"],
        "titles": ["{topic} itinerary", "{topic} travel plan",
                   "{topic} visitor guide"],
        "numbers": ["148", "96", "210", "175", "132"],
        "number_label": "hotel price per night in dollars",
        "dates": ["2026-04-10", "2026-05-22", "2026-09-05"],
        "fillers": [
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
    "finance": {
        "topics": ["AAPL quarterly earnings", "TSLA delivery numbers",
                   "NVDA data-center revenue", "MSFT cloud growth",
                   "AMZN retail margins"],
        "titles": ["{topic} summary", "{topic} earnings brief",
                   "{topic} analyst notes"],
        "numbers": ["184.5", "212.3", "96.8", "341.2", "157.9"],
        "number_label": "closing price in dollars",
        "dates": ["2026-10-30", "2026-07-31", "2026-04-29"],
        "fillers": [
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
    "research": {
        "topics": ["retrieval-augmented generation", "tool-using agents",
                   "context compression", "agentic workflows",
                   "long-horizon planning"],
        "titles": ["{topic} survey", "{topic} reproduction notes",
                   "{topic} literature review"],
        "numbers": ["42", "128", "17", "256", "89"],
        "number_label": "citation count of the lead paper",
        "dates": ["2026-02-11", "2025-12-03", "2026-06-18"],
        "fillers": [
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
    "shopping": {
        "topics": ["trail running shoes", "noise-cancelling headphones",
                   "mechanical keyboards", "espresso machines",
                   "ultralight backpacks"],
        "titles": ["{topic} comparison", "{topic} buying guide",
                   "{topic} shortlist"],
        "numbers": ["129", "249", "89", "349", "179"],
        "number_label": "best price found in dollars",
        "dates": ["2026-08-15", "2026-03-08", "2026-10-01"],
        "fillers": [
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
}

# op-type skeletons: (name, n_ops). Extras are appended afterwards.
_SKELETONS = [
    ("single", ["search", "fetch", "extract", "compute", "verify", "write"]),
    ("audit", ["search", "fetch", "extract", "compute", "verify", "write",
               "audit"]),
    ("two_source", ["search", "fetch", "extract", "search", "fetch",
                    "extract", "compute", "verify", "write"]),
    ("retry", ["search", "fetch", "fetch", "extract", "compute", "verify",
               "write"]),
    ("two_audit", ["search", "fetch", "extract", "search", "fetch",
                   "extract", "compute", "verify", "write", "audit"]),
]
_EXTRA_OPS = ["compute2", "verify2", "audit"]


def _long_text(rng, domain, topic, key_sentence, n_sentences, wid):
    """Long prose whose FIRST sentence carries the key fact, so a SUMMARY
    stub (first ~25 tokens) retains the required semantic facts while a
    literal consumer still needs the full text."""
    pool = [f.format(topic=topic) for f in domain["fillers"]]
    rng.shuffle(pool)
    sents = [key_sentence] + pool[:max(0, n_sentences - 1)]
    return " ".join(sents)


def _field(rng, name, ftype, value, long_text=False, artifact_id=None):
    return Field(name, ftype, value, long_text=long_text,
                 artifact_id=artifact_id)


def _input(source, need, uncertain=False):
    return {"source": source, "need": need, "uncertain": uncertain}


def build_workflow(rng, wid):
    """Build one workflow dict with keys: wid, domain, topic, dag,
    final_needs [(op_idx, field, need)], gold [per-step dict
    {(op_idx, field): frozenset(caps)}], uncertain fields, flags."""
    domain_name = rng.choice(list(_DOMAINS))
    domain = _DOMAINS[domain_name]
    topic = rng.choice(domain["topics"])
    title = rng.choice(domain["titles"]).format(topic=topic)
    number = rng.choice(domain["numbers"])
    date = rng.choice(domain["dates"])
    verify_need = rng.choice(["literal", "literal", "semantic"])
    write_need = rng.choice(["literal", "literal", "semantic", "recoverable"])
    skel_name, skel = rng.choice(_SKELETONS)
    ops = list(skel)
    # Append 0-3 extra ops to vary length (6-14 ops total).
    n_extra = rng.choice([0, 0, 1, 1, 2, 3])
    for _ in range(n_extra):
        if len(ops) >= 14:
            break
        ops.append(rng.choice(_EXTRA_OPS))

    nodes = []
    idx = {}  # op role -> op index, for wiring inputs

    def add(op_type, inputs, outputs):
        nodes.append(OpNode(op_type, inputs, outputs))
        return len(nodes) - 1

    # -- op: search ------------------------------------------------------
    key_sentence = ("The %s for %s is %s, recorded on %s."
                    % (domain["number_label"], topic, number, date))
    results_page = _long_text(rng, domain, topic, key_sentence,
                              rng.randint(9, 12), wid)
    search_i = add("search", {"query": _input(None, "literal")}, {
        "results_page": _field(rng, "results_page", "PageText", results_page,
                               long_text=True,
                               artifact_id="art_%d_results" % wid),
        "result_url": _field(rng, "result_url", "URL",
                             "https://example.com/%s/%d"
                             % (domain_name, wid)),
        "related_links": _field(
            rng, "related_links", "PageText",
            _long_text(rng, domain, topic,
                       "Related pages cover neighbouring topics and "
                       "background reading.", rng.randint(8, 10), wid),
            long_text=True, artifact_id="art_%d_links" % wid),
    })
    idx["search"] = search_i

    # -- op: fetch -------------------------------------------------------
    def add_fetch(url_source, first=False, failed=False):
        nonlocal nodes
        if failed:
            return add("fetch", {"url": _input(url_source, "literal")}, {
                "page_status": _field(rng, "page_status", "Status", "error"),
            })
        page_text = _long_text(
            rng, domain, topic,
            "The full page on %s states the %s is %s and the date is %s."
            % (topic, domain["number_label"], number, date),
            rng.randint(9, 12), wid)
        return add("fetch", {"url": _input(url_source, "literal")}, {
            "page_text": _field(rng, "page_text", "PageText", page_text,
                                long_text=True,
                                artifact_id="art_%d_page" % wid),
            "page_status": _field(rng, "page_status", "Status", "200"),
            "sidebar_text": _field(
                rng, "sidebar_text", "PageText",
                _long_text(rng, domain, topic,
                           "The sidebar lists adverts and unrelated "
                           "headlines.", rng.randint(8, 10), wid),
                long_text=True, artifact_id="art_%d_sidebar" % wid),
        })

    # -- op: extract -----------------------------------------------------
    def add_extract(page_source, tag):
        report_body = _long_text(
            rng, domain, topic,
            "Report on %s: the %s is %s, the date is %s, and the title "
            "is %s." % (topic, domain["number_label"], number, date, title),
            rng.randint(9, 12), wid)
        return add("extract",
                   {"page_text": _input(page_source, "semantic")}, {
                       "title": _field(rng, "title", "Title", title),
                       "key_number": _field(rng, "key_number", "Number",
                                            number),
                       "fact_date": _field(rng, "fact_date", "Date", date),
                       "report_body": _field(
                           rng, "report_body", "ReportBody", report_body,
                           long_text=True,
                           artifact_id="art_%d_report%s" % (wid, tag)),
                       "raw_notes": _field(
                           rng, "raw_notes", "PageText",
                           _long_text(rng, domain, topic,
                                      "Raw notes hold unedited observations "
                                      "and stray remarks.",
                                      rng.randint(8, 10), wid),
                           long_text=True,
                           artifact_id="art_%d_notes%s" % (wid, tag)),
                   })

    fetch_i = fetch2_i = extract_i = extract2_i = None
    compute_i = verify_i = write_i = None
    seen = {"fetch": 0, "extract": 0, "search": 0}
    for op in ops:
        if op == "search":
            seen["search"] += 1
            if seen["search"] == 1:
                continue  # first search already added
            si = add("search", {"query": _input(None, "literal")}, {
                "results_page": _field(
                    rng, "results_page", "PageText",
                    _long_text(rng, domain, topic, key_sentence,
                               rng.randint(9, 12), wid),
                    long_text=True,
                    artifact_id="art_%d_results2" % wid),
                "result_url": _field(rng, "result_url", "URL",
                                     "https://example.com/%s/%d/b"
                                     % (domain_name, wid)),
                "related_links": _field(
                    rng, "related_links", "PageText",
                    _long_text(rng, domain, topic,
                               "More related pages and background reading.",
                               rng.randint(8, 10), wid),
                    long_text=True, artifact_id="art_%d_links2" % wid),
            })
            idx["search2"] = si
        elif op == "fetch":
            seen["fetch"] += 1
            if skel_name == "retry" and seen["fetch"] == 1:
                fetch_i = add_fetch((search_i, "result_url"), failed=True)
            elif seen["fetch"] == 1:
                fetch_i = add_fetch((search_i, "result_url"))
            elif seen["fetch"] == 2 and skel_name == "retry":
                fetch_i = add_fetch((search_i, "result_url"))
            else:
                fetch2_i = add_fetch((idx["search2"], "result_url"))
        elif op == "extract":
            seen["extract"] += 1
            if seen["extract"] == 1:
                extract_i = add_extract((fetch_i, "page_text"), "")
            else:
                extract2_i = add_extract((fetch2_i, "page_text"), "2")
        elif op == "compute":
            inputs = {"key_number": _input((extract_i, "key_number"),
                                           "literal"),
                      "fact_date": _input((extract_i, "fact_date"),
                                          "literal")}
            if extract2_i is not None:
                # Two-source comparison: second key number, long-range.
                inputs["key_number2"] = _input(
                    (extract2_i, "key_number"), "literal")
            computed = str(float(number) * 2) if "." in number \
                else str(int(number) * 2)
            compute_i = add("compute", inputs, {
                "computed_value": _field(rng, "computed_value", "Number",
                                         computed),
                "check_flag": _field(rng, "check_flag", "Status", "pass"),
            })
        elif op == "verify":
            # Long-range: page_text from the early fetch, status from the
            # (possibly second) fetch, computed value from compute.
            status_src = (fetch2_i, "page_status") \
                if fetch2_i is not None else (fetch_i, "page_status")
            verify_i = add("verify", {
                "computed_value": _input((compute_i, "computed_value"),
                                         "literal"),
                "page_status": _input(status_src, "literal"),
                "page_text": _input((fetch_i, "page_text"), verify_need),
            }, {
                "check_result": _field(rng, "check_result", "Status", "ok"),
                "verified_quote": _field(rng, "verified_quote", "Quote",
                                         "%s is %s" % (domain["number_label"],
                                                       number)),
            })
        elif op == "write":
            body_src = (extract2_i, "report_body") \
                if extract2_i is not None else (extract_i, "report_body")
            write_i = add("write", {
                "report_body": _input(body_src, write_need),
                "title": _input((extract_i, "title"), "literal"),
                "computed_value": _input((compute_i, "computed_value"),
                                         "literal"),
            }, {
                "artifact_path": _field(
                    rng, "artifact_path", "ArtifactPath",
                    "/workspace/artifacts/report_%03d.md" % wid),
                "write_status": _field(rng, "write_status", "Status", "ok"),
            })
        elif op == "compute2":
            # Long-range: key_number from the early extract, consumed here.
            add("compute2", {
                "computed_value": _input((compute_i, "computed_value"),
                                         "literal"),
                "key_number": _input((extract_i, "key_number"), "literal"),
            }, {
                "refined_value": _field(rng, "refined_value", "Number",
                                        str(float(number) * 3)),
            })
        elif op == "verify2":
            add("verify2", {
                "computed_value": _input((compute_i, "computed_value"),
                                         "literal"),
                "page_text": _input((fetch_i, "page_text"), verify_need),
            }, {
                "check_result": _field(rng, "check_result", "Status", "ok"),
            })
        elif op == "audit":
            add("audit", {
                "artifact_path": _input((write_i, "artifact_path"),
                                        "literal"),
                "write_status": _input((write_i, "write_status"), "literal"),
            }, {
                "audit_result": _field(rng, "audit_result", "Status", "ok"),
            })

    n_ops = len(nodes)

    # Final-answer obligations: specific short-scalar fields, literal.
    final_needs = [
        (extract_i, "title", "literal"),
        (compute_i, "computed_value", "literal"),
        (write_i, "artifact_path", "literal"),
        (write_i, "write_status", "literal"),
    ]

    # Uncertain dependency evidence (Sec. 3.2): with prob. 0.15 the
    # auxiliary extractor proposes that raw_notes may be needed later,
    # but the relation cannot be reliably resolved -> the online
    # analysis must retain it EXACT conservatively. It is NOT a real
    # consumer, so gold excludes it.
    uncertain = []
    if rng.random() < 0.15:
        uncertain.append((extract_i, "raw_notes"))

    dag = TraceDAG(nodes, uncertain=uncertain)

    # Gold required-field sets per step, from the generator's data flow.
    # gold[t][(op_idx, field)] = joined capability set demanded by all
    # consumers at steps >= t plus the final answer.
    from waac import NEED_CAPS
    gold = []
    for t in range(n_ops + 1):
        need = {}
        for j in range(t, n_ops):
            for spec in nodes[j].inputs.values():
                src = spec.get("source")
                if src is None or src[0] >= t:
                    continue
                need[src] = need.get(src, frozenset()) | NEED_CAPS[spec["need"]]
        for (oi, fname, nd) in final_needs:
            if oi < t:
                need[(oi, fname)] = need.get((oi, fname), frozenset()) \
                    | NEED_CAPS[nd]
        gold.append(need)

    return {
        "wid": wid,
        "domain": domain_name,
        "topic": topic,
        "skeleton": skel_name,
        "n_ops": n_ops,
        "verify_need": verify_need,
        "write_need": write_need,
        "dag": dag,
        "final_needs": final_needs,
        "gold": gold,
        "uncertain": uncertain,
        "task_text": (
            "Research %s: find the page, extract the %s and the date, "
            "verify the quote, write the report and give the final answer "
            "with the title, the computed value, the artifact path and "
            "the write status." % (topic, domain["number_label"])),
    }


def build_corpus(n_workflows=60, seed=20261109):
    """~60 seeded workflows (6-14 ops each)."""
    rng = random.Random(seed)
    return [build_workflow(rng, wid) for wid in range(n_workflows)]


def build_dev_corpus(n_workflows=20, seed=777):
    """Separate-seed development workflows for the history library."""
    rng = random.Random(seed)
    return [build_workflow(rng, 1000 + wid) for wid in range(n_workflows)]
