"""WAAC: Workflow-Aware Agent Trajectory Compression (CPU replication).

Replication of arXiv 2611.08865 (Zhuang, Liu & Ren, 2026) -- trajectory
compression as field-level fidelity allocation:

  * Typed Trace-to-DAG (Sec. 3.2): node v_i = (op_i, I_i, O_i); edge
    e_ij^{p->q} when output field O_i[p] is consumed as input field
    I_j[q]. A tool result may hold an artifact path, a status value and a
    long report body; a later op may need the path and status exactly
    while the body is accessed via the stored artifact.
  * Online backward dependency analysis (Sec. 3.3.1): from outstanding
    obligations O_t (pending tool inputs, explicit planned ops, final-answer
    requirements, verification checks), reverse-traverse the DAG; each field
    gets a minimum fidelity Req(f, s) per obligation; requirements join
    across obligations (satisfying one must not invalidate another);
    unresolved requirements trigger conservative EXACT retention.
    Requirements are recomputed as execution proceeds.
  * Historical fragment matching (Sec. 3.3.2): library of
    verified-successful trajectories only. Exact-key index on the canonical
    representation first; eligibility requires compatible typed operation
    structure, field roles and action categories; eligible candidates are
    ranked by semantic similarity (CPU proxy: TF-IDF cosine over op / field
    / domain tokens -- the paper uses embeddings). No compatible candidate
    above threshold -> online requirements alone.
  * Conservative fusion (Sec. 3.3.3): final policy pi_t(f) is the least
    upper bound under the preservation order EXACT > SUMMARY, EXACT >
    POINTER, with SUMMARY and POINTER incomparable capabilities (SUMMARY
    exposes bounded semantic info, POINTER supports access to an external
    artifact) -- if both are required and neither alone suffices, the join
    is EXACT. History can only ADD retention, never relax the online
    requirement. If history is empty / no match, pi = online requirement.
  * Four policies (Sec. 3.2 / App. A.1): EXACT, SUMMARY, POINTER, DROP.
    A pointer requires a valid reference, else it falls back to EXACT.
    Original events and artifacts stay in the runtime state S_t.

Stdlib only. Deterministic (no randomness anywhere in this module).
"""

import hashlib
import math

# ---------------------------------------------------------------- policies
# Policies are capability sets (App. A.1): SUMMARY provides {semantic},
# POINTER provides {recoverable}, EXACT provides {literal, semantic,
# recoverable}, DROP provides {}. The join of two policies is the union of
# their capabilities; SUMMARY v POINTER = {semantic, recoverable} = EXACT,
# because no single non-EXACT representation covers both capabilities.
POLICIES = ("DROP", "SUMMARY", "POINTER", "EXACT")

_POLICY_CAPS = {
    "DROP": frozenset(),
    "SUMMARY": frozenset({"semantic"}),
    "POINTER": frozenset({"recoverable"}),
    "EXACT": frozenset({"literal", "semantic", "recoverable"}),
}

# Need kind -> minimum policy that satisfies it (App. A.1).
NEED_TO_POLICY = {
    "literal": "EXACT",       # literal / type-sensitive execution & verification
    "semantic": "SUMMARY",    # bounded semantic facts suffice
    "recoverable": "POINTER",  # external artifact recovery suffices
    "unresolved": "EXACT",    # unresolved schema/dependency evidence -> EXACT
}

# Need kind -> capability set the representation must provide.
NEED_CAPS = {
    "literal": frozenset({"literal"}),
    "semantic": frozenset({"semantic"}),
    "recoverable": frozenset({"recoverable"}),
    "unresolved": frozenset({"literal", "semantic", "recoverable"}),
}


def policy_caps(policy):
    """Capability set of a policy."""
    return _POLICY_CAPS[policy]


def fuse(a, b):
    """Least upper bound of two policies under the preservation order.

    fuse(SUMMARY, POINTER) == EXACT (incomparable capabilities, neither
    alone suffices); fuse is the capability union mapped back to a policy.
    """
    caps = _POLICY_CAPS[a] | _POLICY_CAPS[b]
    if caps == frozenset({"literal", "semantic", "recoverable"}):
        return "EXACT"
    if caps == frozenset({"semantic", "recoverable"}):
        # Both capabilities required and neither representation alone
        # suffices -> conservative EXACT (Sec. 3.3.3 / App. A.1).
        return "EXACT"
    if caps == frozenset({"semantic"}):
        return "SUMMARY"
    if caps == frozenset({"recoverable"}):
        return "POINTER"
    return "DROP"


def fuse_many(policies):
    """Join of any number of policies (associative / commutative)."""
    out = "DROP"
    for p in policies:
        out = fuse(out, p)
    return out


def policy_satisfies(policy, need):
    """Does `policy` (as a bare representation) satisfy a need kind?"""
    return NEED_CAPS[need] <= _POLICY_CAPS[policy]


# ---------------------------------------------------------------- fields
class Field:
    """One typed output field of an operation.

    `long_text`: the value is long prose (a page / report body) whose
    SUMMARY stub truncates it; short scalar fields are retained whole by
    a SUMMARY stub. `artifact_id`: key under which the full value is
    stored in the runtime state S_t (None if not stored as an artifact).
    """

    def __init__(self, name, ftype, value, long_text=False, artifact_id=None):
        self.name = name
        self.ftype = ftype
        self.value = str(value)
        self.long_text = bool(long_text)
        self.artifact_id = artifact_id

    def token_count(self):
        return len(self.value.split())

    def __repr__(self):  # pragma: no cover - debugging aid
        return "Field(%s:%s)" % (self.name, self.ftype)


class OpNode:
    """One operation node v_i = (op_i, I_i, O_i).

    inputs: {name: {"source": (op_idx, field_name) or None for a task
             literal, "need": "literal"|"semantic"|"recoverable"|
             "unresolved", "uncertain": bool}}
    outputs: {name: Field}
    """

    def __init__(self, op_type, inputs, outputs):
        self.op_type = op_type
        self.inputs = dict(inputs)
        self.outputs = dict(outputs)


class TraceDAG:
    """Typed Trace-to-DAG (Sec. 3.2).

    Edges are built from explicit argument bindings: input I_j[q] naming
    output O_i[p] as its source. `uncertain` lists (op_idx, field_name)
    pairs whose dependency evidence could not be reliably resolved; the
    paper marks those fields for conservative EXACT retention.
    """

    def __init__(self, nodes, uncertain=None):
        self.nodes = list(nodes)
        self.uncertain = list(uncertain or [])
        self.edges = []  # (src_op, src_field, dst_op, dst_field, uncertain)
        for j, node in enumerate(self.nodes):
            for iname, spec in node.inputs.items():
                src = spec.get("source")
                if src is None:
                    continue
                self.edges.append((src[0], src[1], j, iname,
                                   bool(spec.get("uncertain", False))))

    def field(self, op_idx, name):
        return self.nodes[op_idx].outputs[name]

    def __len__(self):
        return len(self.nodes)


# ------------------------------------------------------- online analysis
def obligations_for_step(dag, final_needs, step):
    """Outstanding obligations O_t at `step` (Sec. 3.3.1).

    Pending tool inputs of ops >= step, final-answer requirements, plus
    one unresolved-evidence obligation per uncertain field (conservative
    retention). Returns a list of (source_key, need) with
    source_key = (op_idx, field_name) naming a field produced by an op
    < step, or None for a task literal (no history field involved).
    """
    obs = []
    for j in range(step, len(dag)):
        for spec in dag.nodes[j].inputs.values():
            src = spec.get("source")
            if src is None:
                continue
            if src[0] >= step:
                # Producer has not run yet: no history field to retain.
                continue
            obs.append((src, spec.get("need", "unresolved")))
    for (op_idx, fname, need) in final_needs:
        if op_idx < step:
            obs.append(((op_idx, fname), need))
    for (op_idx, fname) in dag.uncertain:
        if op_idx < step:
            obs.append(((op_idx, fname), "unresolved"))
    return obs


def online_requirements(dag, obligations):
    """Backward dependency analysis (Sec. 3.3.1).

    Reverse-traverses the DAG from the outstanding obligations: each
    obligation's consumer input resolves (via its data edge) to the
    producer field, which inherits the minimum fidelity Req(f, s) that
    the obligation demands. Requirements from several obligations join
    (fuse), so satisfying one never invalidates another. Fields with no
    obligation are absent from the result (eligible for DROP).
    """
    req = {}
    for (src, need) in obligations:
        if src is None:
            continue
        pol = NEED_TO_POLICY.get(need, "EXACT")
        req[src] = fuse(req.get(src, "DROP"), pol)
    return req


# ------------------------------------------------------------ fragments
def make_fragment(dag, start, end, domain_words):
    """Canonical local fragment: ops [start, end) with their fields and
    the data edges internal to the window (Sec. 3.2.2 / A.2). Field roles
    are (relative op index, field name, field type); instance values are
    replaced by type placeholders, so the fragment is comparable across
    task instances."""
    ops = [dag.nodes[i].op_type for i in range(start, end)]
    fields = []
    for i in range(start, end):
        for fname, f in sorted(dag.nodes[i].outputs.items()):
            fields.append((i - start, fname, f.ftype))
    edges = []
    for (si, sf, di, df, _unc) in dag.edges:
        if start <= si < end and start <= di < end:
            edges.append((si - start, sf, di - start, df))
    return {
        "ops": ops,
        "fields": fields,
        "edges": sorted(edges),
        "domain_words": sorted(set(domain_words)),
    }


def canonical_key(fragment):
    """Deterministic hash of the canonical fragment (Sec. 3.2.2)."""
    parts = [
        "ops:" + ",".join(fragment["ops"]),
        "fields:" + ",".join("%d:%s:%s" % f for f in fragment["fields"]),
        "edges:" + ",".join("%d:%s>%d:%s" % e for e in fragment["edges"]),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def semantic_tokens(fragment):
    """Tokens of the fragment's semantic description (Sec. 3.3.2): op
    types, field names/types and domain words. Embeddings in the paper;
    TF-IDF cosine over these tokens is the CPU proxy here."""
    toks = list(fragment["ops"])
    toks += [name for (_r, name, _t) in fragment["fields"]]
    toks += [ftype for (_r, _n, ftype) in fragment["fields"]]
    toks += list(fragment["domain_words"])
    return toks


def _tfidf_vector(tokens, idf):
    vec = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    for t in vec:
        vec[t] *= idf.get(t, 0.0)
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        for t in vec:
            vec[t] /= norm
    return vec


def _cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


class HistoryLibrary:
    """Verified-successful trajectory fragments + their field policies.

    Only trajectories that pass task-level verification are admitted
    (Sec. 3.2.2 / 3.3): `add_fragment(..., verified=False)` is rejected.
    Records sharing a canonical key aggregate policies by conservative
    join (the maximum-fidelity policy observed for that field).
    """

    def __init__(self):
        self.by_key = {}   # canonical key -> entry
        self.entries = []  # entries in admission order

    def __len__(self):
        return len(self.entries)

    def add_fragment(self, fragment, policies, verified, source_id):
        """Admit one checkpoint fragment. `policies` maps
        (rel_op_idx, field_name) -> policy. Returns True if admitted."""
        if not verified:
            return False
        key = canonical_key(fragment)
        entry = self.by_key.get(key)
        if entry is None:
            entry = {
                "key": key,
                "fragment": fragment,
                "policies": dict(policies),
                "sources": [source_id],
            }
            self.by_key[key] = entry
            self.entries.append(entry)
        else:
            for role, pol in policies.items():
                entry["policies"][role] = fuse(
                    entry["policies"].get(role, "DROP"), pol)
            if source_id not in entry["sources"]:
                entry["sources"].append(source_id)
        return True

    def _idf(self):
        docs = [semantic_tokens(e["fragment"]) for e in self.entries]
        n = len(docs)
        df = {}
        for toks in docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        return {t: math.log(1 + (n - c + 0.5) / (c + 0.5))
                for t, c in df.items()}

    def match(self, fragment, threshold=0.5):
        """Historical fragment matching (Sec. 3.3.2).

        Exact-key index first; otherwise, among candidates compatible in
        typed operation structure (same op-type sequence), field roles
        (same field-role multiset) and action categories, rank by TF-IDF
        cosine over semantic descriptions and take the best above
        `threshold`. Returns (entry, score, kind) or None.
        """
        key = canonical_key(fragment)
        if key in self.by_key:
            return (self.by_key[key], 1.0, "exact")
        if not self.entries:
            return None
        want_roles = sorted((name, ftype)
                            for (_r, name, ftype) in fragment["fields"])
        cands = []
        for e in self.entries:
            frag = e["fragment"]
            if frag["ops"] != fragment["ops"]:
                continue
            roles = sorted((name, ftype)
                           for (_r, name, ftype) in frag["fields"])
            if roles != want_roles:
                continue
            cands.append(e)
        if not cands:
            return None
        idf = self._idf()
        qvec = _tfidf_vector(semantic_tokens(fragment), idf)
        best, best_score = None, -1.0
        for e in cands:
            evec = _tfidf_vector(semantic_tokens(e["fragment"]), idf)
            s = _cosine(qvec, evec)
            if s > best_score:
                best, best_score = e, s
        if best_score >= threshold:
            return (best, best_score, "coarse")
        return None


# ------------------------------------------------------- fusion & project
def fuse_policies(online, history_proposal):
    """Conservative fusion (Sec. 3.3.3): pi_t(f) = max(d_t(f), rho_hat(f)).

    History can only ADD retention, never relax the online requirement.
    Fields with no online requirement and no proposal are absent (DROP).
    """
    out = dict(online)
    for fkey, pol in (history_proposal or {}).items():
        out[fkey] = fuse(out.get(fkey, "DROP"), pol)
    return out


SUMMARY_STUB_TOKENS = 25


def project_field(field, policy, runtime_state):
    """Apply one policy to a field for the model-visible context
    (Sec. 3.3 / App. A.3). Returns (rendered_text_or_None, fell_back).

    EXACT: original value. SUMMARY: bounded stub -- short scalars are
    retained whole, long text is truncated to the first
    SUMMARY_STUB_TOKENS tokens. POINTER: opaque reference token if the
    artifact is verifiably in the runtime state, else fall back to EXACT.
    DROP: omitted from the visible context.
    """
    if policy == "DROP":
        return None, False
    if policy == "EXACT":
        return field.value, False
    if policy == "SUMMARY":
        if not field.long_text:
            return field.value, False
        toks = field.value.split()
        return " ".join(toks[:SUMMARY_STUB_TOKENS]) + " ...[summary]", False
    if policy == "POINTER":
        if field.artifact_id is not None and field.artifact_id in runtime_state:
            return "<ptr:%s>" % field.artifact_id, False
        return field.value, True  # invalid reference -> fall back to EXACT
    raise ValueError("unknown policy %r" % (policy,))


def effective_caps(field, policy, runtime_state):
    """Capability set the projected representation of `field` actually
    provides to a consumer. Mirrors project_field: a SUMMARY stub of a
    short scalar retains the literal value, so it also provides
    `literal`; a POINTER with an invalid reference falls back to EXACT
    capabilities; DROP provides nothing."""
    if policy == "DROP":
        return frozenset()
    if policy == "EXACT":
        return _POLICY_CAPS["EXACT"]
    if policy == "SUMMARY":
        caps = {"semantic"}
        if not field.long_text:
            caps.add("literal")
        return frozenset(caps)
    if policy == "POINTER":
        if field.artifact_id is not None and field.artifact_id in runtime_state:
            return frozenset({"recoverable"})
        return _POLICY_CAPS["EXACT"]
    raise ValueError("unknown policy %r" % (policy,))


def project_context(dag, policies, runtime_state, step):
    """Project all fields produced by ops < `step` under `policies`.

    Returns (rendered, caps, n_fallbacks) with rendered mapping
    (op_idx, field_name) -> visible text (absent = DROP) and caps mapping
    the same key -> the capability set a consumer can actually use.
    """
    rendered, caps = {}, {}
    n_fallbacks = 0
    for i in range(step):
        for fname, field in dag.nodes[i].outputs.items():
            pol = policies.get((i, fname), "DROP")
            text, fell_back = project_field(field, pol, runtime_state)
            if fell_back:
                n_fallbacks += 1
            if text is not None:
                rendered[(i, fname)] = text
            caps[(i, fname)] = effective_caps(field, pol, runtime_state)
    return rendered, caps, n_fallbacks
