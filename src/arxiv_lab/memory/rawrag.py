"""Raw-RAG vs transformative agentic memory (arXiv:2610.30755).

Paper claim implemented: a naive RAG baseline that retrieves over the *raw*
conversation history (dense retrieval + rerank) matches or beats complex
agentic memory systems that mutate / structurally reorganize memory, because
higher *coverage* of the necessary evidence (answer values plus the relation
and dependency paths needed to derive them) offsets lower *utilization*.
Even complete access to a transformed store leaves an accuracy gap versus
raw evidence.

This module operationalizes the claim as a deterministic, seeded CPU test:
a synthetic multi-session conversation world with ground-truth evidence
turns, four memory arms (raw-RAG, extractive-summary memory, full-summary
store, full-raw store), a TF-IDF retriever, and a deterministic oracle
reader. Metrics per arm: answer exact-match, evidence coverage, utilization
(EM / coverage).

Stdlib + numpy only.
"""

import math
import random
import re
from collections import Counter

# ------------------------------------------------------------------ corpus

NAMES = ["Ava Chen", "Ben Rao", "Cara Kim", "Dev Okafor", "Eli Silva",
         "Fay Novak", "Gus Haddad", "Hana Tanaka"]
CITIES = ["Portland", "Austin", "Kyoto", "Lisbon", "Toronto", "Seoul",
          "Berlin", "Oslo"]
HOBBIES = ["rock climbing", "sourdough baking", "birdwatching",
           "amateur radio", "pottery", "open-water swimming",
           "chess boxing", "urban sketching"]
PETS = [("cat", "Miso"), ("dog", "Biscuit"), ("parrot", "Kiwi"),
        ("rabbit", "Pip"), ("gecko", "Ziggy"), ("hamster", "Noodle")]
DOCTORS = [("Alvarez", "St. Mercy"), ("Osei", "Harborview"),
           ("Lindqvist", "Northgate"), ("Park", "Cedar Sinai"),
           ("Nasser", "Riverside"), ("Ito", "Bayfront")]
EMPLOYERS = ["Nimbus Labs", "Copperfield", "Driftwell", "Halcyon",
             "Vantage Co", "Bluefin", "Northloop", "Quartzline",
             "Ferrous", "Lumen_arc", "Driftmark", "Cobalt Works",
             "Asterfield", "Kestrel", "Moonshot", "Ironpine"]

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text):
    return _TOKEN.findall(text.lower())


class Turn:
    __slots__ = ("tid", "session", "speaker", "text")

    def __init__(self, tid, session, speaker, text):
        self.tid = tid
        self.session = session
        self.speaker = speaker
        self.text = text


class Question:
    __slots__ = ("qid", "kind", "text", "person", "answer", "evidence",
                 "keys")

    def __init__(self, qid, kind, text, person, answer, evidence, keys):
        self.qid = qid
        self.kind = kind            # 'single' | 'multi' | 'temporal'
        self.text = text
        self.person = person        # full name string
        self.answer = answer        # expected answer string
        self.evidence = evidence    # tuple of required turn tids
        self.keys = keys            # tuple of (person, fact-key) required


class ConversationWorld:
    """Seeded multi-session conversation world with ground-truth evidence."""

    def __init__(self, seed, n_persons=6, n_sessions=8, turns_per_session=14):
        rng = random.Random(seed)
        self.persons = rng.sample(NAMES, n_persons)
        self.facts = {}
        cities = rng.sample(CITIES, n_persons)
        hobbies = rng.sample(HOBBIES, n_persons)
        pets = rng.sample(PETS, n_persons)
        docs = rng.sample(DOCTORS, n_persons)
        emps = rng.sample(EMPLOYERS, 2 * n_persons)
        for i, name in enumerate(self.persons):
            self.facts[name] = {
                "city": cities[i],
                "hobby": hobbies[i],
                "pet_kind": pets[i][0],
                "pet": pets[i][1],
                "doctor": "Dr. " + docs[i][0],
                "hospital": docs[i][1],
                "employer_prev": emps[2 * i],
                "employer_curr": emps[2 * i + 1],
            }
        self.turns = []
        tid = 0
        # fact turns are generated first so their tids are stable; each fact
        # is stated once per world (some facts restated in later sessions)
        fact_turns = []  # (person, key, text)
        for name in self.persons:
            f = self.facts[name]
            first = name.split()[0]
            fact_turns += [
                (name, "city", f"I live in {f['city']} now, moved here last spring."),
                (name, "doctor", f"My doctor is {f['doctor']}."),
                (name, "hospital", f"{f['doctor']} practices at {f['hospital']}."),
                (name, "temporal",
                 f"I used to work at {f['employer_prev']} before joining {f['employer_curr']}."),
                (name, "employer", f"I work at {f['employer_curr']} these days."),
                (name, "hobby", f"I spend most weekends {f['hobby']}."),
                (name, "pet", f"My {f['pet_kind']} is named {f['pet']}."),
            ]
        rng.shuffle(fact_turns)
        # round-robin so every fact is placed exactly once
        buckets = [[] for _ in range(n_sessions)]
        for i, ft in enumerate(fact_turns):
            buckets[i % n_sessions].append(ft)
        self._fact_tid = {}  # (person, key) -> tid (remapped after shuffle)
        distractor_pool = [
            "The weather has been strange lately, don't you think?",
            "I finally finished that book you recommended.",
            "Traffic on the bridge was awful this morning.",
            "Have you tried the new cafe downtown yet?",
            "My plants are somehow still alive, a small miracle.",
            "I keep meaning to call my sister back.",
        ]
        for s in range(n_sessions):
            session_facts = buckets[s]
            session_texts = []
            for (name, key, text) in session_facts:
                session_texts.append(("user", f"{name.split()[0]}: {text}"))
            # distractors, some with cross-entity attribute mentions
            n_dist = turns_per_session - len(session_texts)
            for _ in range(n_dist):
                other = rng.choice(self.persons)
                of = self.facts[other]
                kind = rng.random()
                if kind < 0.25:
                    t = (f"{other.split()[0]} also likes {of['hobby']}, "
                         f"small world.")
                elif kind < 0.5:
                    t = f"I visited {of['city']} once, lovely place."
                elif kind < 0.7:
                    t = rng.choice(distractor_pool)
                else:
                    t = (f"{other.split()[0]} mentioned {of['employer_curr']} "
                         f"is hiring again.")
                session_texts.append(("user", t))
            rng.shuffle(session_texts)
            for speaker, text in session_texts:
                self.turns.append(Turn(len(self.turns), s, speaker, text))
        # remap fact tids to the final shuffled positions
        self._fact_tid = {}
        for t in self.turns:
            for name in self.persons:
                f = self.facts[name]
                first = name.split()[0]
                if t.text.startswith(first + ":"):
                    body = t.text[len(first) + 2:]
                    for key, marker in (
                            ("city", f"I live in {f['city']}"),
                            ("doctor", f"My doctor is {f['doctor']}"),
                            ("hospital", f"practices at {f['hospital']}"),
                            ("temporal", f"I used to work at {f['employer_prev']}"),
                            ("employer", f"I work at {f['employer_curr']}"),
                            ("hobby", f"weekends {f['hobby']}"),
                            ("pet", f"named {f['pet']}")):
                        if marker in body:
                            self._fact_tid[(name, key)] = t.tid
        self.questions = self._make_questions(rng)

    # ------------------------------------------------------------- questions
    def _make_questions(self, rng):
        qs = []
        qid = 0

        def ev(*keys_person):
            return tuple(self._fact_tid[k] for k in keys_person)

        for name in self.persons:
            f = self.facts[name]
            first = name.split()[0]
            qs.append(Question(qid, "single",
                               f"What city does {first} live in?",
                               name, f["city"], ev((name, "city")),
                               ((name, "city"),)))
            qid += 1
            qs.append(Question(qid, "single",
                               f"What is the name of {first}'s {f['pet_kind']}?",
                               name, f["pet"], ev((name, "pet")),
                               ((name, "pet"),)))
            qid += 1
            qs.append(Question(qid, "multi",
                               f"Which hospital does {first}'s doctor practice at?",
                               name, f["hospital"],
                               ev((name, "doctor"), (name, "hospital")),
                               ((name, "doctor"), (name, "hospital"))))
            qid += 1
            qs.append(Question(qid, "temporal",
                               f"Where did {first} work before {f['employer_curr']}?",
                               name, f["employer_prev"],
                               ev((name, "temporal")),
                               ((name, "temporal"),)))
            qid += 1
        rng.shuffle(qs)
        for i, q in enumerate(qs):
            q.qid = i
        return qs


# ------------------------------------------------------------------ retrieval

def _tfidf_matrix(texts):
    """Return (vocab, idf, rows) with L2-normalized TF-IDF rows (numpy)."""
    import numpy as np
    docs = [_tok(t) for t in texts]
    df = Counter()
    for d in docs:
        for tok in set(d):
            df[tok] += 1
    vocab = {t: i for i, t in enumerate(df)}
    n = len(docs)
    idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in df])
    rows = np.zeros((n, len(vocab)))
    for i, d in enumerate(docs):
        tf = Counter(d)
        for tok, c in tf.items():
            rows[i, vocab[tok]] = c * idf[vocab[tok]]
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


def tfidf_retrieve(chunks, query, k):
    """Rank chunk indices by TF-IDF cosine similarity to query (desc)."""
    import numpy as np
    mat = _tfidf_matrix(chunks + [query])
    q = mat[-1]
    sims = mat[:-1] @ q
    order = np.argsort(-sims, kind="stable")
    return [int(i) for i in order[:k]]


# ------------------------------------------- recency-window transformation

def window_chunks(world, per_person=4):
    """Fixed-budget forgetting: keep each person's `per_person` most recent
    fact turns, drop the rest. A transparent, widely used memory-management
    policy (recency window); the information loss is the budget itself, not
    a scoring-function artifact.

    Methods note: extractive summarization was tried first (TextRank, then
    TF-IDF salience with and without length normalization), but every
    variant's keep/drop pattern was dominated by scoring artifacts —
    TextRank anti-selected proper-noun-dense fact turns (8/42 kept, seed 0);
    unnormalized salience favored long distractors; mean-normalized salience
    favored short sentences and dropped city/employer/temporal facts (16/42).
    The recency window replaces all of that with a budget rule anyone can
    audit.
    """
    rev = _rev_map(world)
    by_person = {}
    for t in world.turns:
        if t.tid in rev:
            name, _ = rev[t.tid]
            by_person.setdefault(name, []).append(t)
    keep = set()
    for name, turns in by_person.items():
        turns = sorted(turns, key=lambda t: t.tid, reverse=True)
        keep |= {t.tid for t in turns[:per_person]}
    out = []
    for t in world.turns:
        if t.tid in keep:
            out.append((t.text, frozenset([t.tid]), {rev[t.tid]}))
    return out


# ------------------------------------------------------------------ reader

def _mentions(text, name):
    low = text.lower()
    return name.lower() in low or name.split()[0].lower() in low


def oracle_answer(world, question, context):
    """Deterministic reader.

    context: list of (text, sources, keys); keys = set of (person, fact-key)
    the chunk is derived from. Returns (correct, covered) where covered =
    every required (person, fact-key) is present in the context.
    """
    f = world.facts[question.person]
    have = set()
    for (_, _, keys) in context:
        have |= keys
    covered = all(k in have for k in question.keys)
    if not covered:
        return False, False
    texts = [t for (t, _, _) in context]
    if question.kind == "single":
        ok = any(_mentions(t, question.person) and question.answer in t
                 for t in texts)
    elif question.kind == "multi":
        hop1 = [t for t in texts
                if _mentions(t, question.person) and f["doctor"] in t]
        hop2 = [t for t in texts
                if f["doctor"] in t and question.answer in t]
        ok = bool(hop1) and bool(hop2)
    else:  # temporal: the 'used to ... before' cue must be present
        ok = any(_mentions(t, question.person)
                 and question.answer in t
                 and ("used to" in t.lower() or "before" in t.lower())
                 for t in texts)
    return ok, True


# ------------------------------------------------------------------ arms

def _rev_map(world):
    return {tid: (name, key) for (name, key), tid in world._fact_tid.items()}


def _chunks_raw(world):
    rev = _rev_map(world)
    out = []
    for t in world.turns:
        keys = {rev[t.tid]} if t.tid in rev else set()
        out.append((t.text, frozenset([t.tid]), keys))
    return out


# ------------------------------------------- mutable profile store (mutation)

_PATTERNS = [
    ("city", re.compile(r"^[A-Za-z]+: I live in (.+?) now,"), "lives in {}"),
    ("doctor", re.compile(r"^[A-Za-z]+: My doctor is (Dr\. [A-Za-z]+)\."),
     "doctor is {}"),
    ("hospital", re.compile(r"^[A-Za-z]+: (Dr\. [A-Za-z]+) practices at (.+)\."),
     None),  # stored on the doctor card
    ("employer", re.compile(r"^[A-Za-z]+: I work at (.+?) these days\."),
     "works at {}"),
    ("temporal", re.compile(r"^[A-Za-z]+: I used to work at (.+?) before "
                            r"joining (.+?)\."), None),
    ("hobby", re.compile(r"^[A-Za-z]+: I spend most weekends (.+?)\."),
     "spends weekends {}"),
    ("pet", re.compile(r"^[A-Za-z]+: My (\w+) is named (\w+)\."),
     None),
]


def extract_profile_facts(turn, first_to_person):
    """Deterministic fact extractor (the profile system's write path).

    Returns list of (person, key, value, tid). The temporal rule keeps only
    the *current* employer: standard mutable-memory overwrite semantics —
    history is not retained. This is the genuine cost of the mutation op,
    not an injected handicap.
    """
    m0 = re.match(r"^([A-Za-z]+):", turn.text)
    if not m0 or m0.group(1) not in first_to_person:
        return []
    person = first_to_person[m0.group(1)]
    out = []
    for key, pat, _ in _PATTERNS:
        m = pat.match(turn.text)
        if not m:
            continue
        if key == "hospital":
            out.append((m.group(1), "hospital", m.group(2), turn.tid))
        elif key == "temporal":
            # mutation: overwrite with current employer; the previous
            # employer is not stored (history loss is the op's real semantics)
            out.append((person, "employer", m.group(2), turn.tid))
        elif key == "pet":
            out.append((person, "pet", f"{m.group(1)} named {m.group(2)}",
                        turn.tid))
        else:
            out.append((person, key, m.group(1), turn.tid))
    return out


class ProfileStore:
    """Per-entity profile cards with overwrite (mutation) write semantics."""

    def __init__(self):
        self.cards = {}   # person -> {key: (value, tid)}
        self.doctors = {}  # "Dr. X" -> (hospital, tid)

    def write(self, person, key, value, tid):
        if key == "hospital":
            self.doctors[person] = (value, tid)  # person slot holds doctor name
        else:
            self.cards.setdefault(person, {})[key] = (value, tid)

    def chunks(self):
        """Render cards as retrieval chunks: (text, sources, keys)."""
        out = []
        for person, attrs in self.cards.items():
            for key, (value, tid) in attrs.items():
                if key == "city":
                    text = f"{person} lives in {value}."
                elif key == "doctor":
                    text = f"{person}'s doctor is {value}."
                elif key == "employer":
                    text = f"{person} works at {value}."
                elif key == "hobby":
                    text = f"{person} spends weekends {value}."
                elif key == "pet":
                    text = f"{person}'s {value}."
                else:
                    continue
                out.append((text, frozenset([tid]), {(person, key)}))
        for doctor, (hospital, tid) in self.doctors.items():
            out.append((f"{doctor} practices at {hospital}.",
                        frozenset([tid]), {(doctor, "hospital")}))
        return out


def _chunks_profile(world):
    first_to_person = {p.split()[0]: p for p in world.persons}
    store = ProfileStore()
    for t in world.turns:  # chronological: later writes overwrite earlier ones
        for (person, key, value, tid) in extract_profile_facts(t,
                                                               first_to_person):
            store.write(person, key, value, tid)
    # doctor-card keys are stored under the doctor name; remap the
    # (person, 'hospital') requirement onto the matching doctor's card key
    chunks = []
    for (text, sources, keys) in store.chunks():
        remapped = set()
        for (who, key) in keys:
            if key == "hospital":
                for person in world.persons:
                    if world.facts[person]["doctor"] == who:
                        remapped.add((person, "hospital"))
            else:
                remapped.add((who, key))
        chunks.append((text, sources, remapped))
    return chunks


def evaluate_profile(world, chunks):
    """Profile arm with entity-keyed lookup (the system's native access
    pattern — profile stores are queried by entity, not by text retrieval).

    Context for a question about person P: P's whole card + P's doctor's
    card. Retrieval quality is therefore not a confound; what is measured
    is purely what the mutation write semantics preserved.
    """
    ems, covs, utils = [], [], []
    kind_em = Counter()
    kind_n = Counter()
    for q in world.questions:
        ctx = [c for c in chunks
               if any(who == q.person for (who, _) in c[2])]
        correct, covered = oracle_answer(world, q, ctx)
        ems.append(float(correct))
        covs.append(float(covered))
        if covered:
            utils.append(float(correct))
        kind_em[q.kind] += float(correct)
        kind_n[q.kind] += 1
    import numpy as np
    return {
        "em": float(np.mean(ems)),
        "coverage": float(np.mean(covs)),
        "utilization": float(np.mean(utils)) if utils else 0.0,
        "n": len(ems),
        "kind_em": {kk: kind_em[kk] / kind_n[kk] for kk in kind_n},
    }


def evaluate_arm(world, chunks, k, full=False):
    """Run every question against one memory arm.

    Returns dict with em, coverage, utilization means and per-kind EM.
    """
    ems, covs, utils = [], [], []
    kind_em = Counter()
    kind_n = Counter()
    for q in world.questions:
        if full:
            ctx = chunks
        else:
            idx = tfidf_retrieve([c[0] for c in chunks], q.text, k)
            ctx = [chunks[i] for i in idx]
        correct, covered = oracle_answer(world, q, ctx)
        ems.append(float(correct))
        covs.append(float(covered))
        if covered:
            utils.append(float(correct))
        kind_em[q.kind] += float(correct)
        kind_n[q.kind] += 1
    import numpy as np
    return {
        "em": float(np.mean(ems)),
        "coverage": float(np.mean(covs)),
        "utilization": float(np.mean(utils)) if utils else 0.0,
        "n": len(ems),
        "kind_em": {kk: kind_em[kk] / kind_n[kk] for kk in kind_n},
    }


def run_setting(seed, k=8, per_person=4):
    """One seeded world, all five arms. Returns {arm: metrics}."""
    world = ConversationWorld(seed)
    raw = _chunks_raw(world)
    win = window_chunks(world, per_person=per_person)
    prof = _chunks_profile(world)
    return {
        "raw_rag": evaluate_arm(world, raw, k),
        "window_rag": evaluate_arm(world, win, k),
        "window_full": evaluate_arm(world, win, k, full=True),
        "profile_rag": evaluate_profile(world, prof),
        "raw_full": evaluate_arm(world, raw, k, full=True),
    }


def compare(seeds, k=8, per_person=4):
    """Aggregate run_setting over seeds -> {arm: {metric: (mean, std)}}."""
    import numpy as np
    per_arm = {}
    for seed in seeds:
        res = run_setting(seed, k=k, per_person=per_person)
        for arm, m in res.items():
            per_arm.setdefault(arm, []).append(m)
    agg = {}
    for arm, ms in per_arm.items():
        agg[arm] = {}
        for key in ("em", "coverage", "utilization"):
            vals = np.array([m[key] for m in ms])
            agg[arm][key] = (float(vals.mean()), float(vals.std(ddof=1)))
        kinds = {}
        for kk in ms[0]["kind_em"]:
            vals = np.array([m["kind_em"][kk] for m in ms])
            kinds[kk] = (float(vals.mean()), float(vals.std(ddof=1)))
        agg[arm]["kind_em"] = kinds
    return agg


if __name__ == "__main__":
    import json
    agg = compare(list(range(8)))
    print(json.dumps(agg, indent=1))
