"""A matrix with steps: episodic and procedural memory for tool-using agents
(arXiv:2611.20230, Paulhart & Schuller).

Paper claim implemented: a memory with two parts -- a delta-rule
associative matrix M (Eq. 1: M <- alpha*M + beta*(v - M k) k^T) as the
FINDER, and ordered, addressable STEP RECORDS as the source of truth --
plus a write gate admitting only externally verified outcomes, deterministic
consolidation, an asymmetric (two-stage, first-attempt) rule for changing a
default procedure, instant demotion on a protective stop, and erasure by
rebuilding the matrix from the remaining records.

Stdlib + numpy only. No model calls, no network at import time. The
embedder is a deterministic bag-of-words hash embedder over a fixed
3000-word vocabulary, as in the paper's synthetic capacity runs.
"""

import math
import re

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is the single optional dep
    np = None

__all__ = [
    "fnv1a_64",
    "MT19937",
    "Rng",
    "HashEmbedder",
    "procedure_sign_vector",
    "DeltaMatrix",
    "Step",
    "ProcedureRecord",
    "Episode",
    "EpisodicMemory",
    "VERIFIED_SOURCES",
    "recorded_outcome",
    "confidence",
    "consolidate_episodes",
    "z_test_one_sided_p",
    "promotion_window",
    "run_promotion_windows",
    "naive_keep_count",
    "generate_synthetic_procedures",
    "capacity_experiment",
    "simulate_hidden_criterion",
    "simulate_revisit_stream",
    "simulate_demotion",
    "MIN_EPISODES",
    "MIN_SUCCESSES",
    "MIN_COVERAGE",
    "WINDOW_ATTEMPTS",
    "MATRIX_LIMITS",
    "FIND_COSINE_THRESHOLD",
    "ONLINE_ALPHA",
    "ONLINE_BETA_BASE",
    "REBUILD_PASSES",
]

# ---------------------------------------------------------------- constants
# Paper Sec. 4: only a verification from a check, a test, a tool result or a
# person counts. A model's statement never verifies an outcome.
VERIFIED_SOURCES = ("external_check", "test", "tool_result", "person")
MODEL_CLAIM = "model_claim"

# Paper Sec. 4: consolidation minimums.
MIN_EPISODES = 5
MIN_SUCCESSES = 3
MIN_COVERAGE = 0.60

# Paper Sec. 4: first-attempt evidence in fixed windows of 40 attempts/arm.
WINDOW_ATTEMPTS = 40

# Paper Sec. 3: writer limits per matrix type at d=1024 (cols/d -> max procs).
MATRIX_LIMITS = {1: 200, 4: 500, 8: 1000}

# Cosine below this means "no procedure found" (recall returns nothing).
FIND_COSINE_THRESHOLD = 0.50

# Paper Sec. 3: online defaults vs rebuild defaults.
ONLINE_ALPHA = 0.995
ONLINE_BETA_BASE = 0.25
REBUILD_ALPHA = 1.0
REBUILD_BETA_BASE = 1.0
REBUILD_PASSES = 10

# Embedder constants (paper Sec. 6: 3000-word vocabulary, d=1024).
VOCAB_SIZE = 3000
EMBED_DIM = 1024
EMBED_SEED = 0x5EEDBEEF
_SPLITMIX_C1 = 0x9E3779B97F4A7C15
_SPLITMIX_C2 = 0xBF58476D1CE4E5B9
_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Function words carry no procedure content but appear in almost every
# text; leaving them in makes all embeddings share one large common
# direction (cos ~0.45 between unrelated keys) and drowns the matrix
# recall in crosstalk. The hash embedder is over content words.
STOPWORDS = frozenset(
    "the a an on in at when to of for and is needs with by it its this that "
    "be are was were has have had do does did".split())


def _need_numpy():
    if np is None:  # pragma: no cover
        raise ImportError("numpy is required for the matrix/embedder parts "
                          "of arxiv_lab.memory.matrix_steps")


# ---------------------------------------------------------------- hashing
def fnv1a_64(text):
    """64-bit FNV-1a hash of a str (paper Sec. 3.1: seeds the sign vector)."""
    h = 0xCBF29CE484222325
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & _MASK64
    return h


def _splitmix64_finalizer(x):
    """SplitMix64 finalizer (pure integer ops; identical in the Rust port)."""
    x &= _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


# ---------------------------------------------------------------- MT19937
class MT19937:
    """Mersenne Twister (reference mt19937ar init_genrand / init_by_array).

    Paper Sec. 3.1: a 64-bit FNV-1a hash of the procedure id seeds a
    Mersenne Twister; the low bit of each draw gives one sign of the
    per-procedure sign vector. Seeding here is init_by_array over the two
    32-bit halves of that hash, so the Python and Rust ports agree exactly.
    """

    N = 624
    M = 397
    _MATRIX_A = 0x9908B0DF
    _UPPER_MASK = 0x80000000
    _LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed_words):
        if isinstance(seed_words, int):
            seed_words = [seed_words & _MASK32]
        self._mt = [0] * self.N
        self._idx = self.N
        # init_genrand(19650218)
        self._mt[0] = 19650218
        for i in range(1, self.N):
            self._mt[i] = (1812433253 * (self._mt[i - 1]
                                         ^ (self._mt[i - 1] >> 30)) + i) & _MASK32
        # init_by_array
        key = [w & _MASK32 for w in seed_words]
        i, j = 1, 0
        k = max(self.N, len(key))
        for _ in range(k):
            self._mt[i] = ((self._mt[i]
                            ^ ((self._mt[i - 1] ^ (self._mt[i - 1] >> 30))
                               * 1664525)) + key[j] + j) & _MASK32
            i += 1
            j += 1
            if i >= self.N:
                self._mt[0] = self._mt[self.N - 1]
                i = 1
            if j >= len(key):
                j = 0
        for _ in range(self.N - 1):
            self._mt[i] = ((self._mt[i]
                            ^ ((self._mt[i - 1] ^ (self._mt[i - 1] >> 30))
                               * 1566083941)) - i) & _MASK32
            i += 1
            if i >= self.N:
                self._mt[0] = self._mt[self.N - 1]
                i = 1
        self._mt[0] = 0x80000000

    @classmethod
    def from_int_seed(cls, seed):
        """Classic init_genrand path (used for the known test vector)."""
        obj = cls.__new__(cls)
        obj._mt = [0] * cls.N
        obj._idx = cls.N
        obj._mt[0] = seed & _MASK32
        for i in range(1, cls.N):
            obj._mt[i] = (1812433253 * (obj._mt[i - 1]
                                        ^ (obj._mt[i - 1] >> 30)) + i) & _MASK32
        return obj

    def _twist(self):
        mt = self._mt
        for i in range(self.N):
            y = ((mt[i] & self._UPPER_MASK)
                 | (mt[(i + 1) % self.N] & self._LOWER_MASK))
            mt[i] = mt[(i + self.M) % self.N] ^ (y >> 1)
            if y & 1:
                mt[i] ^= self._MATRIX_A
        self._idx = 0

    def uint32(self):
        """One tempered 32-bit draw."""
        if self._idx >= self.N:
            self._twist()
        y = self._mt[self._idx]
        self._idx += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & _MASK32


class Rng:
    """Deterministic RNG on top of MT19937 (shared with the Rust port)."""

    def __init__(self, seed):
        if isinstance(seed, str):
            seed = fnv1a_64(seed)
        self._mt = MT19937.from_int_seed(seed & _MASK32) \
            if isinstance(seed, int) and seed < (1 << 32) else \
            MT19937([seed & _MASK32, (seed >> 32) & _MASK32])

    def uint32(self):
        return self._mt.uint32()

    def uniform(self):
        """Uniform float in [0, 1)."""
        return self.uint32() / 4294967296.0

    def randint(self, a, b):
        """Uniform int in [a, b] inclusive."""
        return a + int(self.uniform() * (b - a + 1))

    def bernoulli(self, p):
        return self.uniform() < p

    def choice(self, seq):
        return seq[self.randint(0, len(seq) - 1)]

    def shuffle(self, seq):
        seq = list(seq)
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(0, i)
            seq[i], seq[j] = seq[j], seq[i]
        return seq


# ---------------------------------------------------------------- embedder
def _word_component(vocab_index, dim_index, seed=EMBED_SEED):
    """One component of a vocabulary word's vector, in [-1, 1).

    SplitMix64 finalizer over (seed + vocab*C1 + dim*C2); identical
    arithmetic in the Rust port, so widened keys agree across languages.
    """
    z = (seed + vocab_index * _SPLITMIX_C1 + dim_index * _SPLITMIX_C2) & _MASK64
    z = _splitmix64_finalizer(z)
    return z / 9223372036854775808.0 - 1.0


class HashEmbedder:
    """Deterministic bag-of-words hash embedder (paper Sec. 6).

    The hashing trick: each token hashes (FNV-1a) to one of `dim`
    buckets with a sign from the hash's top bit; the text vector is the
    signed count vector, L2-normalized. Texts that share no words are
    (near-)orthogonal, which is what the paper's synthetic capacity runs
    rely on; a dense random-vector embedder leaves a ~0.03 cross-talk
    per key pair that drowns matrix recall once ~1000 associations are
    stored (measured while building this module).
    """

    def __init__(self, vocab_size=VOCAB_SIZE, dim=EMBED_DIM, seed=EMBED_SEED):
        self.vocab_size = vocab_size
        self.dim = dim
        self.seed = seed
        self._text_cache = {}

    def embed(self, text):
        """L2-normalized embedding of `text` (numpy float64 array).

        Cached per text: the same text always embeds to the same vector,
        and repeated lookups (decoding scans) are dict hits.
        """
        _need_numpy()
        vec = self._text_cache.get(text)
        if vec is not None:
            return vec
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in _TOKEN_RE.findall(text.lower()):
            if tok in STOPWORDS:
                continue
            h = fnv1a_64(tok)
            vec[h % self.dim] += 1.0 if (h >> 63) else -1.0
        n = float(np.linalg.norm(vec))
        if n > 0.0:
            vec = vec / n
        self._text_cache[text] = vec
        return vec


def procedure_sign_vector(proc_id, width):
    """Per-procedure sign vector s_p in {-1,+1}^width (paper Sec. 3.1).

    64-bit FNV-1a hash of the id seeds a Mersenne Twister; the low bit of
    each draw gives one sign.
    """
    h = fnv1a_64(proc_id)
    mt = MT19937([h & _MASK32, (h >> 32) & _MASK32])
    return [1 if (mt.uint32() & 1) else -1 for _ in range(width)]


# ---------------------------------------------------------------- matrix
class DeltaMatrix:
    """Delta-rule associative matrix M in R^(dim x cols) (paper Eq. 1).

    Write: M <- alpha*M + beta*(v - M k) k^T. Read: v_hat = M q.
    Keys are widened to `cols` columns before writing (signs of a fixed
    random projection) unless cols == dim, in which case keys pass through.
    """

    def __init__(self, dim=EMBED_DIM, cols=None, widening="sign", seed=777):
        _need_numpy()
        self.dim = dim
        self.cols = cols if cols is not None else dim
        self.widening = widening if self.cols != dim else "none"
        self.seed = seed
        self.M = np.zeros((dim, self.cols), dtype=np.float32)
        self._proj = None
        if self.cols != dim:
            # Fixed Rademacher projection R in R^(cols x dim), entries
            # +/- 1/sqrt(dim); built from SplitMix64 so the Rust port agrees.
            ii = np.arange(self.cols, dtype=np.uint64)[:, None]
            jj = np.arange(dim, dtype=np.uint64)[None, :]
            z = (np.uint64(seed)
                 + ii * np.uint64(_SPLITMIX_C1) + jj * np.uint64(_SPLITMIX_C2))
            z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
            z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
            z = z ^ (z >> np.uint64(31))
            signs = np.where((z & np.uint64(1)) == np.uint64(0), -1.0, 1.0)
            self._proj = (signs / math.sqrt(dim)).astype(np.float32)

    def widen(self, k):
        """Widen a dim-vector key to `cols` columns (paper Sec. 3.1)."""
        k = np.asarray(k, dtype=np.float32)
        if self.cols == self.dim:
            return k
        if self.widening == "sign":
            w = np.sign(self._proj @ k)
            w[w == 0] = 1.0
            return (w / math.sqrt(self.cols)).astype(np.float32)
        # "linear" control widening: phi(e) = R e / ||R e||
        w = self._proj @ k
        n = float(np.linalg.norm(w))
        return (w / n).astype(np.float32) if n > 0 else w

    def write(self, k, v, alpha=REBUILD_ALPHA, beta=REBUILD_BETA_BASE):
        """One delta-rule write of key k (dim-vector) to value v."""
        kw = self.widen(k)
        v = np.asarray(v, dtype=np.float32)
        if alpha != 1.0:
            self.M *= np.float32(alpha)
        resid = v - self.M @ kw
        self.M += (np.float32(beta) * np.outer(resid, kw)).astype(np.float32)

    def read(self, q):
        """Recall: v_hat = M q."""
        return self.M @ self.widen(q)

    def rebuild(self, pairs, passes=REBUILD_PASSES,
                alpha=REBUILD_ALPHA, beta=REBUILD_BETA_BASE):
        """Rebuild from the write list: alpha=1, beta=1, 10 passes."""
        self.M[:] = 0.0
        for _ in range(passes):
            for k, v in pairs:
                self.write(k, v, alpha=alpha, beta=beta)

    def check_limit(self, n_procedures):
        """Writer refuses a scope larger than the type's limit (paper Sec. 3)."""
        ratio = self.cols // self.dim
        limit = MATRIX_LIMITS.get(ratio)
        if limit is not None and n_procedures > limit:
            raise ValueError(
                "scope of %d procedures exceeds the %dx%d matrix limit %d; "
                "split the scope" % (n_procedures, self.dim, self.cols, limit))
        return True


# ---------------------------------------------------------------- records
class Step:
    """One procedure step: a GOAL, not a motion (paper Sec. 3.2)."""

    def __init__(self, action, precondition="", done="", tool="", refines=None):
        self.action = action
        self.precondition = precondition
        self.done = done
        self.tool = tool
        self.refines = refines

    def text(self):
        return self.action

    def to_dict(self):
        return {"action": self.action, "precondition": self.precondition,
                "done": self.done, "tool": self.tool,
                "refines": self.refines}


class ProcedureRecord:
    """An ordered, addressable step record (paper Sec. 3.2)."""

    def __init__(self, proc_id, name, goal, when_to_use, steps,
                 status="active", predecessor=None):
        self.id = proc_id
        self.name = name
        self.goal = goal
        self.when_to_use = when_to_use
        self.steps = list(steps)
        self.attempts = 0
        self.successes = 0
        self.status = status  # active | revoked | demoted
        self.predecessor = predecessor
        self.episode_ids = []

    def key_text(self):
        """Key text: name + goal + when-to-use (paper Sec. 3.1)."""
        return "%s %s %s" % (self.name, self.goal, self.when_to_use)

    def answer_text(self):
        """Index text: name + goal + the actions of its steps."""
        return "%s %s %s" % (self.name, self.goal,
                             " ".join(s.text() for s in self.steps))

    def to_dict(self):
        return {"id": self.id, "name": self.name, "goal": self.goal,
                "when_to_use": self.when_to_use,
                "steps": [s.to_dict() for s in self.steps],
                "attempts": self.attempts, "successes": self.successes,
                "status": self.status, "predecessor": self.predecessor,
                "episode_ids": list(self.episode_ids)}


class Episode:
    """One recorded run (paper Sec. 4)."""

    def __init__(self, ep_id, scope, task_family, goal, steps,
                 outcome, verification, agent_claimed_done=False,
                 tool_failed=False, person_verdict=None, args=None,
                 condition="", failure_mode=None):
        self.id = ep_id
        self.scope = scope
        self.task_family = task_family
        self.goal = goal
        self.steps = tuple(steps)  # exact step sequence (texts)
        self.outcome = outcome  # "success" | "failure" (as recorded)
        self.verification = verification  # see VERIFIED_SOURCES / MODEL_CLAIM
        self.agent_claimed_done = agent_claimed_done
        self.tool_failed = tool_failed
        self.person_verdict = person_verdict  # overrides the outcome
        self.args = dict(args or {})
        self.condition = condition
        self.failure_mode = failure_mode


def recorded_outcome(ep):
    """The outcome an episode counts as, under the write gate (Sec. 4).

    - A reply saying "done" over a failed tool call is recorded as the
      failure it was, whatever the agent claimed.
    - A person's later verdict overrides the recorded outcome.
    - A model-only-vouched outcome is unverified: returns None (never
      counted, never shown as an example).
    """
    if ep.person_verdict is not None:
        return ep.person_verdict
    if ep.tool_failed:
        return "failure"
    if ep.verification not in VERIFIED_SOURCES:
        return None
    return ep.outcome


def confidence(s, n):
    """Consolidation confidence (s+1)/(n+2) (paper Sec. 4)."""
    return (s + 1) / (n + 2)


def consolidate_episodes(episodes, min_episodes=MIN_EPISODES,
                         min_successes=MIN_SUCCESSES,
                         min_coverage=MIN_COVERAGE):
    """Deterministic consolidation (paper Sec. 4); no model call.

    `episodes` must already be write-gated (verified only). Returns a
    record dict, or None when the minimums are not met: at least 5 counted
    episodes, 3 successes, and the most frequent exact step sequence
    covering at least 60% of the successes.
    """
    counted = [e for e in episodes if recorded_outcome(e) is not None]
    n = len(counted)
    successes = [e for e in counted if recorded_outcome(e) == "success"]
    s = len(successes)
    if n < min_episodes or s < min_successes:
        return None
    seq_counts = {}
    for e in successes:
        seq_counts[e.steps] = seq_counts.get(e.steps, 0) + 1
    top_seq, top_count = max(seq_counts.items(), key=lambda kv: kv[1])
    if top_count / s < min_coverage:
        return None
    # Fixed args: identical across all successes; varied-arg distribution.
    fixed, varied = {}, {}
    if successes:
        keys = sorted(successes[0].args.keys())
        for k in keys:
            vals = [e.args.get(k) for e in successes]
            if all(v == vals[0] for v in vals):
                fixed[k] = vals[0]
            else:
                dist = {}
                for v in vals:
                    dist[v] = dist.get(v, 0) + 1
                varied[k] = dist
    conditions = sorted({e.condition for e in successes if e.condition})
    failures = {}
    for e in counted:
        if recorded_outcome(e) == "failure" and e.failure_mode:
            failures[e.failure_mode] = failures.get(e.failure_mode, 0) + 1
    return {"steps": list(top_seq), "fixed_args": fixed,
            "varied_args": varied, "conditions": conditions,
            "failure_modes": failures, "confidence": confidence(s, n),
            "n_episodes": n, "n_successes": s,
            "top_sequence_share": top_count / s,
            "episode_ids": [e.id for e in counted]}


# ---------------------------------------------------------------- memory
class EpisodicMemory:
    """One scope's episodic + procedural memory (paper Sec. 3-4).

    One matrix per scope; the matrix is the finder, the records are the
    source of truth for order, counts and deletion.
    """

    def __init__(self, scope, dim=EMBED_DIM, cols=None, widening="sign",
                 seed=777, embedder=None, rebuild_beta=REBUILD_BETA_BASE):
        self.scope = scope
        self.embedder = embedder or HashEmbedder(dim=dim)
        self.matrix = DeltaMatrix(dim=dim, cols=cols, widening=widening,
                                  seed=seed)
        # Beta used by rebuilds from records. The paper specifies beta=1;
        # beta=0.25 x 10 passes is the stable iterative route to the same
        # least-squares solution (see capacity_experiment) and is what the
        # tests use when they need every procedure retrievable.
        self.rebuild_beta = rebuild_beta
        self.episodes = {}   # id -> Episode
        self.records = {}    # id -> ProcedureRecord
        self.audit = []      # erasure audit entries (record + cause, never person)
        self.default_id = None

    # -- episodes / write gate -------------------------------------------
    def record_episode(self, ep):
        """Store an episode; returns its counted outcome (None if unverified).

        Unverified episodes are stored (the run happened) but never count
        as evidence and are never shown as examples.
        """
        if ep.scope != self.scope:
            raise ValueError("episode scope %r does not match memory scope %r"
                             % (ep.scope, self.scope))
        self.episodes[ep.id] = ep
        return recorded_outcome(ep)

    def verified_episodes(self, task_family=None):
        """Verified episodes of this scope only (never another scope's)."""
        out = []
        for ep in self.episodes.values():
            if ep.scope != self.scope:
                continue
            if task_family is not None and ep.task_family != task_family:
                continue
            if recorded_outcome(ep) is not None:
                out.append(ep)
        return out

    # -- procedures / matrix ----------------------------------------------
    def add_record(self, rec):
        self.records[rec.id] = rec

    def write_pairs(self, records=None):
        """Write list: one find association per procedure plus step chains."""
        pairs = []
        for rec in (records if records is not None else self.records.values()):
            if rec.status != "active":
                continue
            key = self.embedder.embed(rec.key_text())
            val = self.embedder.embed(rec.answer_text())
            pairs.append((key, val))
            signs = procedure_sign_vector(rec.id, self.matrix.cols)
            step_vecs = [self.embedder.embed(s.text()) for s in rec.steps]
            for i in range(len(step_vecs) - 1):
                k = self.matrix.widen(step_vecs[i])
                k = (k * np.asarray(signs, dtype=np.float32)).astype(np.float32)
                pairs.append((k, step_vecs[i + 1]))
        return pairs

    def rebuild_matrix(self, passes=REBUILD_PASSES, enforce_limit=True,
                       beta=None):
        """Rebuild the finder from the active records (paper Sec. 3.1).

        `enforce_limit=False` is for capacity measurement only: the
        paper's Sec. 6 capacity table deliberately exceeds the writer's
        limit to find where recall degrades.
        """
        active = [r for r in self.records.values() if r.status == "active"]
        if enforce_limit:
            self.matrix.check_limit(len(active))
        self.matrix.rebuild(self.write_pairs(active), passes=passes,
                            beta=self.rebuild_beta if beta is None else beta)

    def find(self, query_text, method="matrix", min_cosine=None):
        """Find the procedure for a goal text.

        Returns (record_or_None, best_cosine). The matrix decodes Mq
        against stored value embeddings; the exact scan compares the query
        with every stored key. `min_cosine` is the no-match threshold
        (default FIND_COSINE_THRESHOLD): below it, recall returns nothing.
        Pass min_cosine=0 for pure argmax, as the paper's capacity table
        reports "found".
        """
        if min_cosine is None:
            min_cosine = FIND_COSINE_THRESHOLD
        active = [r for r in self.records.values() if r.status == "active"]
        if not active:
            return None, 0.0
        q = self.embedder.embed(query_text)
        if method == "matrix":
            v_hat = self.matrix.read(q)
            cands = [self.embedder.embed(r.answer_text()) for r in active]
        elif method == "exact":
            v_hat = q
            cands = [self.embedder.embed(r.key_text()) for r in active]
        else:
            raise ValueError("unknown find method %r" % method)
        best_i, best_c = -1, -1.0
        vh = np.asarray(v_hat, dtype=np.float64)
        nrm = float(np.linalg.norm(vh))
        for i, c in enumerate(cands):
            cv = np.asarray(c, dtype=np.float64)
            denom = nrm * float(np.linalg.norm(cv))
            cos = float(vh @ cv / denom) if denom > 0 else 0.0
            if cos > best_c:
                best_c, best_i = cos, i
        if best_c < min_cosine:
            return None, best_c
        return active[best_i], best_c

    def next_step(self, proc_id, step_index):
        """What comes after step i, from the records (source of truth)."""
        rec = self.records.get(proc_id)
        if rec is None or step_index + 1 >= len(rec.steps):
            return None
        return rec.steps[step_index + 1]

    def next_step_via_matrix(self, proc_id, step_index):
        """Next step decoded from the matrix (bound step key -> step i+1)."""
        rec = self.records.get(proc_id)
        if rec is None or step_index + 1 >= len(rec.steps):
            return None, 0.0
        signs = procedure_sign_vector(proc_id, self.matrix.cols)
        e_i = self.embedder.embed(rec.steps[step_index].text())
        k = self.matrix.widen(e_i)
        k = (k * np.asarray(signs, dtype=np.float32)).astype(np.float32)
        v_hat = self.matrix.M @ k
        # Decode against every step embedding of every active record.
        best, best_c = None, -1.0
        vh = np.asarray(v_hat, dtype=np.float64)
        nrm = float(np.linalg.norm(vh))
        for r in self.records.values():
            if r.status != "active":
                continue
            for s in r.steps:
                cv = np.asarray(self.embedder.embed(s.text()), dtype=np.float64)
                denom = nrm * float(np.linalg.norm(cv))
                cos = float(vh @ cv / denom) if denom > 0 else 0.0
                if cos > best_c:
                    best_c, best = cos, s
        return best, best_c

    # -- consolidation ------------------------------------------------------
    def consolidate(self, task_family):
        """Consolidate this scope's verified episodes of one family."""
        return consolidate_episodes(self.verified_episodes(task_family))

    # -- erasure ------------------------------------------------------------
    def erase_episodes(self, episode_ids, cause):
        """Erase episodes; recompute affected records; rebuild the matrix.

        A decided record is corrected from the erased episode's outcome
        and dropped below the minimums. The audit entry names the record
        and the cause, never the person. Erasure wins over retention.
        """
        erased = [self.episodes.pop(eid, None) for eid in episode_ids]
        erased = [e for e in erased if e is not None]
        affected = set()
        for rec in list(self.records.values()):
            kept = [eid for eid in rec.episode_ids if eid in self.episodes]
            if len(kept) != len(rec.episode_ids):
                affected.add(rec.id)
                rec.episode_ids = kept
                eps = [self.episodes[eid] for eid in kept]
                # Recompute the record from the remaining episodes.
                new = consolidate_episodes(eps) if eps else None
                if new is None:
                    self.audit.append({"record": rec.id, "cause": cause,
                                       "action": "dropped below minimums"})
                    del self.records[rec.id]
                else:
                    rec.attempts = new["n_episodes"]
                    rec.successes = new["n_successes"]
                    self.audit.append({"record": rec.id, "cause": cause,
                                       "action": "corrected from remaining "
                                                 "episodes"})
        self.rebuild_matrix()
        return {"erased": [e.id for e in erased],
                "affected_records": sorted(affected),
                "audit": list(self.audit)}

    # -- defaults: promotion / demotion --------------------------------------
    def set_default(self, proc_id, predecessor=None):
        if proc_id is not None and proc_id not in self.records:
            raise ValueError("unknown procedure %r" % proc_id)
        self.default_id = proc_id
        if proc_id is not None:
            self.records[proc_id].predecessor = predecessor

    def demote_on_guard_trip(self, proc_id):
        """Demotion needs no test (paper Sec. 4): a default whose real
        attempt ends in a protective stop loses its place at once, and the
        predecessor becomes default again."""
        rec = self.records.get(proc_id)
        if rec is None or self.default_id != proc_id:
            return None
        rec.status = "demoted"
        prev = rec.predecessor
        self.default_id = prev
        self.rebuild_matrix()
        return prev


# ---------------------------------------------------------------- promotion test
def z_test_one_sided_p(x_cand, n_cand, x_def, n_def):
    """One-sided p-value for candidate success rate > default (pooled z)."""
    if n_cand == 0 or n_def == 0:
        return 1.0
    p_c = x_cand / n_cand
    p_d = x_def / n_def
    p_pool = (x_cand + x_def) / (n_cand + n_def)
    var = p_pool * (1.0 - p_pool) * (1.0 / n_cand + 1.0 / n_def)
    if var <= 0:
        return 0.0 if p_c > p_d else 1.0
    z = (p_c - p_d) / math.sqrt(var)
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def promotion_window(rng, p_default, p_candidate, window=WINDOW_ATTEMPTS,
                     alpha_screen=0.025, alpha_confirm=0.025):
    """One two-stage promotion test on FIRST-attempt evidence (paper Sec. 4).

    A window is `window` first attempts per arm; the procedure for each
    task is chosen by a recorded coin. Screen, then confirmation on fresh
    situations. Returns (promoted, screen_p, confirm_p_or_None).
    """
    return _promotion_window_exact(rng, p_default, p_candidate, window,
                                   alpha_screen, alpha_confirm)


def _promotion_window_exact(rng, p_default, p_candidate, window,
                            alpha_screen=0.025, alpha_confirm=0.025):
    def arm_counts():
        n_d = n_c = x_d = x_c = 0
        while n_d < window or n_c < window:
            if rng.bernoulli(0.5):
                if n_c < window:
                    n_c += 1
                    x_c += rng.bernoulli(p_candidate)
            else:
                if n_d < window:
                    n_d += 1
                    x_d += rng.bernoulli(p_default)
        return x_d, window, x_c, window

    x_d, n_d, x_c, n_c = arm_counts()
    p_screen = z_test_one_sided_p(x_c, n_c, x_d, n_d)
    if p_screen >= alpha_screen:
        return False, p_screen, None
    x_d2, n_d2, x_c2, n_c2 = arm_counts()
    p_confirm = z_test_one_sided_p(x_c2, n_c2, x_d2, n_d2)
    return (p_confirm < alpha_confirm), p_screen, p_confirm


def naive_keep_count(rng, p_default, p_candidate, n_windows=200,
                     window=WINDOW_ATTEMPTS):
    """Control: the rule that keeps any higher score (paper Sec. 6).

    One window per trial; keep the candidate when its window score beats
    the default's. Returns the number of windows that kept the candidate.
    """
    kept = 0
    for _ in range(n_windows):
        x_d = sum(rng.bernoulli(p_default) for _ in range(window))
        x_c = sum(rng.bernoulli(p_candidate) for _ in range(window))
        if x_c > x_d:
            kept += 1
    return kept


def run_promotion_windows(seed, p_default, p_candidate, n_windows=200,
                          window=WINDOW_ATTEMPTS):
    """Run `n_windows` two-stage promotion tests; return counts."""
    rng = Rng(seed)
    promoted = 0
    for _ in range(n_windows):
        ok, _, _ = _promotion_window_exact(rng, p_default, p_candidate,
                                           window)
        promoted += bool(ok)
    rng2 = Rng(seed + 1)
    naive = naive_keep_count(rng2, p_default, p_candidate, n_windows, window)
    return {"n_windows": n_windows, "window": window,
            "promoted": promoted, "naive_kept": naive}


# The paper's synthetic capacity data (Sec. 6) draws procedure texts as
# random word sets from the 3000-word vocabulary -- near-independent
# vectors, with only ~15% of step texts shared. Templated natural-language
# texts break that premise: verbs/objects/contexts repeat across
# procedures, so unrelated keys share tokens (mean cross-cosine ~0.45
# with stopwords, ~0.09 without) and the matrix recall drowns in
# common-mode crosstalk. The generator below follows the paper: random
# word sets from a fixed 3000-word synthetic vocabulary.
_SYNTH_VOCAB = ["w%04d" % i for i in range(VOCAB_SIZE)]
_SHARED_STEP_POOL = None  # built once, lazily, from a fixed seed


def _shared_step_pool():
    global _SHARED_STEP_POOL
    if _SHARED_STEP_POOL is None:
        rng = Rng(999)
        _SHARED_STEP_POOL = [" ".join(rng.choice(_SYNTH_VOCAB)
                                      for _ in range(3))
                             for _ in range(12)]
    return _SHARED_STEP_POOL


def generate_synthetic_procedures(n, seed=20261117, steps_min=3, steps_max=8,
                                  shared_frac=0.15):
    """Synthetic procedures (paper Sec. 6): 3-8 steps each, ~15% of step
    texts shared between procedures.

    Texts are random word sets from the 3000-word vocabulary, so distinct
    procedures have near-independent embeddings -- the premise of the
    paper's capacity table. `name`/`goal`/`when_to_use` are those word
    sets; the record id ("proc_NNNN") is the human-readable handle.
    """
    rng = Rng(seed)
    shared_pool = _shared_step_pool()
    procs = []
    for i in range(n):
        name = " ".join(rng.choice(_SYNTH_VOCAB) for _ in range(2))
        goal = " ".join(rng.choice(_SYNTH_VOCAB) for _ in range(6))
        when = " ".join(rng.choice(_SYNTH_VOCAB) for _ in range(4))
        n_steps = rng.randint(steps_min, steps_max)
        steps = []
        for _ in range(n_steps):
            if rng.uniform() < shared_frac:
                action = rng.choice(shared_pool)
            else:
                action = " ".join(rng.choice(_SYNTH_VOCAB)
                                  for _ in range(3))
            steps.append(Step(action, tool="tool_%s" % rng.randint(1, 6),
                              done="%s done" % action))
        procs.append(ProcedureRecord("proc_%04d" % i, name, goal, when, steps))
    return procs


def capacity_experiment(n_procedures, seed=20261117, passes=REBUILD_PASSES,
                        beta=REBUILD_BETA_BASE, dim=EMBED_DIM):
    """Paper Sec. 6 capacity table: d=1024 square matrix, hash embedder.

    Returns procedure-finding accuracy (matrix vs exact scan), next-step
    accuracy, and median lookup time in ms. "Found" is pure argmax, as
    the paper reports it; the writer limit is bypassed (the table
    deliberately measures past it). `beta`/`passes` select the rebuild:
    the paper's table used beta=1, 1 pass; beta=0.25 x 10 passes is the
    stable iterative route to the same least-squares solution (step 1
    exceeds 2/lambda_max(K K^T) once ~1000 near-orthogonal keys are
    stored, so beta=1 oscillates instead of converging).
    """
    _need_numpy()
    mem = EpisodicMemory("capacity", dim=dim)
    procs = generate_synthetic_procedures(n_procedures, seed=seed)
    for p in procs:
        mem.add_record(p)
    mem.matrix.rebuild(mem.write_pairs(), passes=passes, beta=beta)

    import time
    found = found_exact = 0
    times = []
    for p in procs:
        t0 = time.perf_counter()
        rec, _ = mem.find(p.goal, method="matrix", min_cosine=0.0)
        times.append(time.perf_counter() - t0)
        if rec is not None and rec.id == p.id:
            found += 1
        rec_e, _ = mem.find(p.goal, method="exact", min_cosine=0.0)
        if rec_e is not None and rec_e.id == p.id:
            found_exact += 1
    times.sort()
    median_ms = 1000.0 * times[len(times) // 2]

    # Next-step accuracy: decode the bound step key against every step.
    n_steps = n_ok = 0
    for p in procs:
        for i in range(len(p.steps) - 1):
            best, _ = mem.next_step_via_matrix(p.id, i)
            n_steps += 1
            if best is not None and best.text() == p.steps[i + 1].text():
                n_ok += 1
    return {"n_procedures": n_procedures,
            "n_associations": len(mem.write_pairs()),
            "found_matrix": found / n_procedures,
            "found_exact": found_exact / n_procedures,
            "next_step": n_ok / max(n_steps, 1),
            "median_lookup_ms": median_ms}


# ---------------------------------------------------------------- simulations
def simulate_hidden_criterion(seed=7, n_tasks=200, n_situations=40, cap=6):
    """Paper Sec. 6, Plan 4: hidden arrival-speed criterion.

    Each situation has a hidden speed limit; an attempt reaches the goal
    with a method-dependent probability and damages the part when its
    arrival speed exceeds the limit. The agent's own claim knows only
    "goal reached". Policies replay the same outcome stream:
      no_memory, self_judged_stops, verified (ours).
    """
    rng = Rng(seed)
    speeds = (0.7, 0.9, 1.1, 1.3)
    reach_p = (0.75, 0.85, 0.92, 0.97)
    order = (3, 2, 1, 0)  # fastest first, as a naive agent tries
    situations = []
    for _ in range(n_situations):
        # Hidden speed limit; ~5% of situations have a goal no method
        # reaches at all (the paper's ~21% unsolvable tasks).
        limit = 0.6 + rng.uniform() * 1.0
        reachable = rng.bernoulli(0.95)
        situations.append((limit, reachable))
    n_solvable = sum(1 for lim, ok in situations
                     if ok and any(s <= lim for s in speeds))
    # Ceiling at task level: share of tasks in a solvable situation.
    tasks = [rng.randint(0, n_situations - 1) for _ in range(n_tasks)]
    ceiling = sum(1 for sit in tasks
                  if situations[sit][1]
                  and any(s <= situations[sit][0] for s in speeds)) / n_tasks

    def attempt(sit, method):
        limit, reachable = situations[sit]
        reached = reachable and rng.bernoulli(reach_p[method])
        safe = speeds[method] <= limit
        return {"reached": reached, "solved": reached and safe,
                "damaging": reached and not safe}

    def run(policy):
        verified = {}  # situation -> verified method
        reported = solved = attempts = damaging = 0
        for sit in tasks:
            if policy == "verified" and sit in verified:
                methods = [verified[sit]] + [m for m in order
                                             if m != verified[sit]]
            else:
                methods = list(order)
            done = False
            for a in range(cap):
                m = methods[a % len(methods)]
                attempts += 1
                out = attempt(sit, m)
                if out["damaging"]:
                    damaging += 1
                if policy == "self_judged_stops":
                    if out["reached"]:
                        reported += 1
                        solved += bool(out["solved"])
                        done = True
                        break
                else:
                    if out["solved"]:
                        solved += 1
                        reported += 1
                        if policy == "verified":
                            verified[sit] = m
                        done = True
                        break
            if policy != "self_judged_stops" and not done:
                pass  # task ends unsolved at the cap; not reported done
            if policy == "self_judged_stops" and not done:
                pass
        return {"reported_done": reported / n_tasks,
                "solved": solved / n_tasks,
                "attempts_per_task": attempts / n_tasks,
                "damaging": damaging}

    out = {p: run(p) for p in ("no_memory", "self_judged_stops", "verified")}
    out["ceiling"] = ceiling
    out["n_tasks"] = n_tasks
    out["n_situations"] = n_situations
    return out


def simulate_revisit_stream(seed=11, n_situations=1000, repeats=4,
                            n_methods=4):
    """Paper Sec. 6: a stream of 4x tasks over N situations; one of the
    methods fits each situation (drawn at random, so only memory can know
    it); only simulator-confirmed placements are written.

    Returns the first-attempt solve rate on revisits of solved situations.
    Reach is deterministic per (situation, method) here, so a verified
    method solves every revisit at the first attempt.
    """
    rng = Rng(seed)
    correct = {}
    reach = {}
    for sit in range(n_situations):
        correct[sit] = rng.randint(0, n_methods - 1)
        for m in range(n_methods):
            # The correct method always reaches safely; others reach with
            # p=0.5 but never solve (wrong placement).
            reach[(sit, m)] = (m == correct[sit]) or rng.bernoulli(0.5)
    solved_once = {}
    revisit_first = revisit_total = 0
    first_attempt_total = first_attempt_solved = 0
    for rep in range(repeats):
        for sit in range(n_situations):
            if sit in solved_once:
                revisit_total += 1
                m = solved_once[sit]
                if reach[(sit, m)]:
                    revisit_first += 1
            else:
                for m in range(n_methods):
                    first_attempt_total += 1
                    if reach[(sit, m)] and m == correct[sit]:
                        solved_once[sit] = m
                        first_attempt_solved += 1
                        break
    return {"n_situations": n_situations, "n_tasks": n_situations * repeats,
            "solved_situations": len(solved_once),
            "revisit_first_attempt": revisit_first / max(revisit_total, 1),
            "revisits": revisit_total,
            "first_encounter_attempts": first_attempt_total}


def simulate_demotion(seed=13, n_tasks=200, n_stations=4,
                      trip_station=3, predecessor_solve_p=0.97):
    """Paper Sec. 6: default-demotion replay on the industrial arm.

    The bad default solves every station except `trip_station`, where its
    real attempt trips a protective stop and ends the task. With the
    demotion rule the predecessor becomes default at once.
    Returns solved shares with and without the rule.
    """
    rng = Rng(seed)
    stations = [rng.randint(0, n_stations - 1) for _ in range(n_tasks)]
    # One pre-drawn predecessor outcome per task, so the with/without runs
    # replay the same outcome table (paired comparison, paper Sec. 6).
    pred_solves = [rng.bernoulli(predecessor_solve_p) for _ in range(n_tasks)]

    def run(with_rule):
        default_bad = True
        solved = trips = demotions = 0
        first_trip_task = None
        for t, st in enumerate(stations):
            if default_bad:
                if st == trip_station:
                    trips += 1
                    if first_trip_task is None:
                        first_trip_task = t
                    if with_rule:
                        default_bad = False
                        demotions += 1
                    # task ends unsolved on a guard trip
                else:
                    solved += pred_solves[t]
            else:
                solved += pred_solves[t]
        return {"solved": solved / n_tasks, "trips": trips,
                "demotions": demotions, "first_trip_task": first_trip_task}

    without = run(False)
    with_rule = run(True)
    return {"n_tasks": n_tasks, "n_stations": n_stations,
            "solved_without_rule": without["solved"],
            "solved_with_rule": with_rule["solved"],
            "trips_without_rule": without["trips"],
            "trips_with_rule": with_rule["trips"],
            "demotions": with_rule["demotions"],
            "first_trip_task": with_rule["first_trip_task"]}
