#!/usr/bin/env python3
"""Python-vs-Rust parity sweep for the WAAC Rust port.

Generates seeded random cases with the Python reference (`waac.py`),
runs the Rust `waac-parity` binary on them, and compares every output
exactly. 0 mismatches is required for a parity VERIFIED claim.

Case kinds (see `src/bin/parity.rs` for the Rust side):
  fuse          -- fuse(a, b) for all 16 pairs + 500 random pairs
  fuse_many     -- fuse_many of 200 random triples
  satisfies     -- policy_satisfies(policy, need), 200 random
  dag           -- random DAG: online requirements + fusion + projection
  project_field -- random field x policy x runtime
  fragment      -- canonical key, semantic tokens, history match
  make_fragment -- fragment construction + key from a random DAG

Floats (TF-IDF cosine scores) are compared with exact equality: the
Rust port reproduces CPython's Neumaier `sum()` and libm `log`, so
scores must match bit-for-bit, not approximately.

Run:  python3 tests/parity_check.py [--binary PATH] [--cases N]
Writes: /tmp/waac_parity_cases.json, /tmp/waac_parity_out.json
Exit 0 iff mismatches == 0.
"""

import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.dirname(os.path.dirname(HERE))  # sessions/2026-11-09-2611.08865-waac/
sys.path.insert(0, SESSION)

from waac import (  # noqa: E402
    POLICIES, NEED_CAPS, Field, HistoryLibrary, TraceDAG, canonical_key,
    effective_caps, fuse, fuse_many, fuse_policies, make_fragment,
    obligations_for_step, online_requirements, policy_caps,
    policy_satisfies, project_context, project_field, semantic_tokens,
)

RUST_DIR = os.path.join(SESSION, "rust")
DEFAULT_BINARY = os.path.join(RUST_DIR, "target", "debug", "waac-parity")

CAP_BIT = {"literal": 1, "semantic": 2, "recoverable": 4}


def caps_mask(caps):
    m = 0
    for c in caps:
        m |= CAP_BIT[c]
    return m


OP_TYPES = ["search", "fetch", "extract", "compute", "verify", "write", "audit"]
FIELD_NAMES = ["page_text", "title", "key_number", "report_body", "url",
               "status", "value", "notes"]
FIELD_TYPES = ["PageText", "Title", "Number", "ReportBody", "URL", "Status"]
WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta",
         "lambda", "kappa", "mu", "nu", "xi"]
NEEDS = ["literal", "semantic", "recoverable", "unresolved"]


def random_field(rng, name=None):
    name = name if name is not None else rng.choice(FIELD_NAMES)
    long_text = rng.random() < 0.5
    n_words = rng.randint(40, 80) if long_text else rng.randint(1, 4)
    value = " ".join(rng.choice(WORDS) for _ in range(n_words))
    artifact = ("art_%s" % name) if rng.random() < 0.5 else None
    return {
        "name": name,
        "ftype": rng.choice(FIELD_TYPES),
        "value": value,
        "long_text": long_text,
        "artifact_id": artifact,
    }


def random_dag_json(rng):
    n_ops = rng.randint(2, 8)
    nodes = []
    produced = []
    for i in range(n_ops):
        inputs = []
        for k in range(rng.randint(0, 2)):
            src = None
            if produced and rng.random() < 0.5:
                src = list(rng.choice(produced))
            inputs.append({
                "name": "in%d" % k,
                "source": src,
                "need": rng.choice(NEEDS),
                "uncertain": rng.random() < 0.1,
            })
        outputs = []
        # Field names are unique per op (Python stores outputs in a dict,
        # so a duplicate name would silently overwrite; the corpus never
        # produces them).
        for f in [random_field(rng, name)
                  for name in rng.sample(FIELD_NAMES, rng.randint(1, 3))]:
            outputs.append(f)
            produced.append([i, f["name"]])
        nodes.append({
            "op_type": rng.choice(OP_TYPES),
            "inputs": inputs,
            "outputs": outputs,
        })
    uncertain = []
    if produced and rng.random() < 0.5:
        uncertain.append(list(rng.choice(produced)))
    final_needs = []
    for _ in range(rng.randint(0, 2)):
        if produced:
            oi, fn_ = rng.choice(produced)
            final_needs.append([oi, fn_, rng.choice(NEEDS)])
    return {
        "nodes": nodes,
        "uncertain": uncertain,
        "final_needs": final_needs,
    }


def dag_from_json(dj):
    nodes = []
    for nv in dj["nodes"]:
        inputs = {}
        for iv in nv["inputs"]:
            src = tuple(iv["source"]) if iv["source"] is not None else None
            inputs[iv["name"]] = {
                "source": src,
                "need": iv["need"],
                "uncertain": iv["uncertain"],
            }
        outputs = {}
        for ov in nv["outputs"]:
            outputs[ov["name"]] = Field(
                ov["name"], ov["ftype"], ov["value"],
                long_text=ov["long_text"], artifact_id=ov["artifact_id"])
        from waac import OpNode
        nodes.append(OpNode(nv["op_type"], inputs, outputs))
    uncertain = [tuple(u) for u in dj["uncertain"]]
    return TraceDAG(nodes, uncertain=uncertain)


def random_fragment(rng):
    n_ops = rng.randint(1, 4)
    ops = [rng.choice(OP_TYPES) for _ in range(n_ops)]
    fields = set()
    for r in range(n_ops):
        for _ in range(rng.randint(1, 3)):
            fields.add((r, rng.choice(FIELD_NAMES), rng.choice(FIELD_TYPES)))
    fields = sorted(fields)
    edges = set()
    for _ in range(rng.randint(0, 2)):
        if n_ops >= 2 and fields:
            a = rng.choice(fields)
            b = rng.choice(fields)
            if a[0] < b[0]:
                edges.add((a[0], a[1], b[0], b[1]))
    domain_words = sorted({rng.choice(WORDS) for _ in range(rng.randint(1, 3))})
    return {
        "ops": ops,
        "fields": [list(f) for f in fields],
        "edges": [list(e) for e in sorted(edges)],
        "domain_words": domain_words,
    }


def frag_obj(fj):
    return {
        "ops": list(fj["ops"]),
        "fields": [tuple(f) for f in fj["fields"]],
        "edges": [tuple(e) for e in fj["edges"]],
        "domain_words": list(fj["domain_words"]),
    }


def sorted_kv(d):
    return sorted(([k[0], k[1], v] for k, v in d.items()), key=lambda r: (r[0], r[1]))


def reference(case):
    """Compute the expected output for one case with waac.py."""
    kind = case["kind"]
    if kind == "fuse":
        f = fuse(case["a"], case["b"])
        return {"kind": "fuse", "result": f, "caps": caps_mask(policy_caps(f))}
    if kind == "fuse_many":
        return {"kind": "fuse_many", "result": fuse_many(case["policies"])}
    if kind == "satisfies":
        return {"kind": "satisfies",
                "result": policy_satisfies(case["policy"], case["need"])}
    if kind == "dag":
        dag = dag_from_json(case["dag"])
        final_needs = [(o, f, n) for o, f, n in case["dag"]["final_needs"]]
        step = case["step"]
        obs = obligations_for_step(dag, final_needs, step)
        online = online_requirements(dag, obs)
        proposal = {}
        if case.get("proposal") is not None:
            for k, p in case["proposal"].items():
                oi, fname = k.split("|", 1)
                proposal[(int(oi), fname)] = p
        fused = fuse_policies(online, proposal if proposal else None)
        runtime = dict(case.get("runtime") or {})
        rendered, caps, n_fb = project_context(dag, fused, runtime, step)
        return {
            "kind": "dag",
            "online": sorted_kv({k: v for k, v in online.items()}),
            "fused": sorted_kv({k: v for k, v in fused.items()}),
            "rendered": sorted_kv(rendered),
            "caps": sorted_kv({k: caps_mask(v) for k, v in caps.items()}),
            "n_fallbacks": n_fb,
        }
    if kind == "project_field":
        fj = case["field"]
        field = Field(fj["name"], fj["ftype"], fj["value"],
                      long_text=fj["long_text"], artifact_id=fj["artifact_id"])
        runtime = dict(case.get("runtime") or {})
        text, fb = project_field(field, case["policy"], runtime)
        caps = effective_caps(field, case["policy"], runtime)
        return {"kind": "project_field", "text": text, "fell_back": fb,
                "caps": caps_mask(caps)}
    if kind == "fragment":
        frag = frag_obj(case["fragment"])
        key = canonical_key(frag)
        toks = semantic_tokens(frag)
        lib = HistoryLibrary()
        for e in case["entries"]:
            ef = frag_obj(e["fragment"])
            pols = {}
            for k, p in e["policies"].items():
                rel, fname = k.split("|", 1)
                pols[(int(rel), fname)] = p
            lib.add_fragment(ef, pols, e["verified"], e["source_id"])
        m = lib.match(frag)
        if m is None:
            mi, ms, mk = None, None, None
        else:
            entry, score, kind2 = m
            mi = lib.entries.index(entry)
            ms, mk = score, kind2
        return {"kind": "fragment", "key": key, "tokens": toks,
                "library_entries": len(lib), "match_index": mi,
                "match_score": ms, "match_kind": mk}
    if kind == "make_fragment":
        dag = dag_from_json(case["dag"])
        frag = make_fragment(dag, case["start"], case["end"],
                             case["domain_words"])
        return {
            "kind": "make_fragment",
            "fragment": {
                "ops": frag["ops"],
                "fields": [list(f) for f in frag["fields"]],
                "edges": [list(e) for e in frag["edges"]],
                "domain_words": frag["domain_words"],
            },
            "key": canonical_key(frag),
        }
    raise ValueError("unknown kind %r" % kind)


def gen_cases(seed=20261109):
    rng = random.Random(seed)
    cases = []
    for a in POLICIES:
        for b in POLICIES:
            cases.append({"kind": "fuse", "a": a, "b": b})
    for _ in range(500):
        cases.append({"kind": "fuse", "a": rng.choice(POLICIES),
                      "b": rng.choice(POLICIES)})
    for _ in range(200):
        cases.append({"kind": "fuse_many",
                      "policies": [rng.choice(POLICIES) for _ in range(3)]})
    for _ in range(200):
        cases.append({"kind": "satisfies", "policy": rng.choice(POLICIES),
                      "need": rng.choice(NEEDS)})
    for _ in range(300):
        dj = random_dag_json(rng)
        step = rng.randint(0, len(dj["nodes"]))
        proposal = {}
        for nv_i, nv in enumerate(dj["nodes"]):
            for ov in nv["outputs"]:
                if rng.random() < 0.3:
                    proposal["%d|%s" % (nv_i, ov["name"])] = rng.choice(POLICIES)
        runtime = {}
        for nv in dj["nodes"]:
            for ov in nv["outputs"]:
                if ov["artifact_id"] and rng.random() < 0.5:
                    runtime[ov["artifact_id"]] = ov["value"]
        cases.append({"kind": "dag", "dag": dj, "step": step,
                      "proposal": proposal or None, "runtime": runtime})
    for _ in range(300):
        fj = random_field(rng)
        runtime = {}
        if fj["artifact_id"] and rng.random() < 0.5:
            runtime[fj["artifact_id"]] = fj["value"]
        cases.append({"kind": "project_field", "field": fj,
                      "policy": rng.choice(POLICIES), "runtime": runtime})
    for _ in range(200):
        frag = random_fragment(rng)
        entries = []
        for i in range(rng.randint(0, 8)):
            ef = random_fragment(rng)
            if rng.random() < 0.25:
                ef = frag  # sometimes an exact-key entry
            pols = {}
            for (r, n, _t) in ef["fields"]:
                pols["%d|%s" % (r, n)] = rng.choice(POLICIES)
            entries.append({
                "fragment": ef, "policies": pols,
                "verified": rng.random() < 0.85,
                "source_id": "src-%d" % i,
            })
        cases.append({"kind": "fragment", "fragment": frag,
                      "entries": entries})
    for _ in range(200):
        dj = random_dag_json(rng)
        n = len(dj["nodes"])
        start = rng.randint(0, n - 1)
        end = rng.randint(start + 1, n)
        cases.append({"kind": "make_fragment", "dag": dj, "start": start,
                      "end": end,
                      "domain_words": sorted({rng.choice(WORDS) for _ in range(3)})})
    return cases


def main():
    binary = DEFAULT_BINARY
    args = sys.argv[1:]
    if "--binary" in args:
        binary = args[args.index("--binary") + 1]
    seed = 20261109
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    cases = gen_cases(seed)
    cases_path = "/tmp/waac_parity_cases.json"
    out_path = "/tmp/waac_parity_out.json"
    with open(cases_path, "w") as f:
        json.dump({"cases": cases}, f)
    r = subprocess.run([binary, cases_path, out_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("parity binary failed:", r.stderr)
        return 2
    with open(out_path) as f:
        got = json.load(f)["results"]
    assert len(got) == len(cases), (len(got), len(cases))
    mismatches = 0
    by_kind = {}
    for i, (case, actual) in enumerate(zip(cases, got)):
        want = reference(case)
        if want != actual:
            mismatches += 1
            by_kind[case["kind"]] = by_kind.get(case["kind"], 0) + 1
            if mismatches <= 5:
                print("MISMATCH case %d kind=%s" % (i, case["kind"]))
                for k in sorted(set(want) | set(actual)):
                    if want.get(k) != actual.get(k):
                        print("  %s: python=%r rust=%r" % (k, want.get(k), actual.get(k)))
    print("parity sweep: %d cases, %d mismatches %s"
          % (len(cases), mismatches, by_kind))
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
