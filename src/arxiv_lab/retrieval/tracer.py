"""TRACER: Trajectory-Aware Tool Retrieval for LLM Agents.

arXiv:2610.21267 — "TRACER: Trajectory-Aware Tool Retrieval for LLM Agents".

Paper mechanism implemented here:

* **Trajectory trie**: observed tool-call sequences are indexed in a prefix
  tree. For any history prefix, each trie node stores how often each tool
  continued that prefix -> an *empirical continuation prior*.
* **Contextual prior**: a semantic score for each tool from the current step
  context (cosine similarity between a step embedding and tool description
  embeddings; in a live harness this comes from an embedding model).
* **Learned blend**: final ranking = ``w * empirical + (1 - w) * contextual``
  with ``w`` chosen on a held-out split to maximize top-1 selection
  accuracy. Only the top-k tools are exposed to the agent.
* **Update online**: new observed sequences are inserted as the agent runs.

Paper claims: on a mixed retail/telecom environment, +4.7 tool-selection
points over full-registry selection while exposing just 5 of 128 tools
(~80% fewer input tokens per step); average +6-10 points in 21/24
model-environment pairs.

Requires numpy (optional dependency): ``pip install numpy``.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

__all__ = [
    "TrajectoryTrie",
    "learn_blend_weight",
    "tracer_retrieve",
    "tracer_top1_accuracy",
    "cosine_similarity",
]

_DEFAULT_K = 5
_DEFAULT_SMOOTHING = 1.0


def _need_numpy():  # keep public entry points numpy-guarded
    if np is None:
        raise ImportError("numpy is required for arxiv_lab.retrieval.tracer")


def cosine_similarity(a, b):
    """Row-wise cosine similarity of matrix ``a`` against vector ``b``."""
    _need_numpy()
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b)
    denom = np.where(denom == 0.0, 1e-12, denom)
    return (a @ b) / denom


class TrajectoryTrie:
    """Prefix tree over observed tool-call sequences.

    ``insert(sequence)`` records one observed tool-call trajectory.
    ``continuation_probs(prefix, n_tools)`` returns a probability vector
    over all ``n_tools`` tools: Laplace-smoothed empirical continuation
    frequencies after ``prefix``. Unseen prefixes fall back to the root
    (marginal) continuation distribution.
    """

    def __init__(self, smoothing=_DEFAULT_SMOOTHING):
        self.smoothing = smoothing
        self.root = {}
        self.n_sequences = 0

    def insert(self, sequence):
        node = self.root
        for tool in sequence:
            node = node.setdefault(tool, {})
        node.setdefault(None, 0)
        node[None] += 1
        self.n_sequences += 1

    def _find(self, prefix):
        node = self.root
        for tool in prefix:
            if tool not in node:
                return self.root  # unseen prefix -> marginal fallback
            node = node[tool]
        return node

    def continuation_probs(self, prefix, n_tools):
        """Smoothed P(next tool | prefix) as a length-``n_tools`` vector."""
        _need_numpy()
        node = self._find(tuple(prefix))
        counts = np.full(n_tools, self.smoothing, dtype=float)
        total = self.smoothing * n_tools
        for key, child in node.items():
            if key is None:
                continue
            if 0 <= key < n_tools:
                c = sum(self._iter_terminals(child))
                counts[key] += c
                total += c
        return counts / total

    @staticmethod
    def _iter_terminals(node):
        if None in node:
            yield node[None]
        for key, child in node.items():
            if key is not None:
                yield from TrajectoryTrie._iter_terminals(child)


def tracer_retrieve(prefix, context_vec, tool_vecs, trie, k=_DEFAULT_K,
                     w_empirical=0.5):
    """Rank tools by the TRACER blend and return the top-``k`` indices.

    score = w_empirical * empirical_prior + (1 - w_empirical) * contextual_prior
    with the contextual prior = softmax(cosine(context, tool_vecs)).
    """
    _need_numpy()
    n_tools = len(tool_vecs)
    empirical = trie.continuation_probs(prefix, n_tools)
    context_vec = np.asarray(context_vec, dtype=float)
    tool_vecs = np.asarray(tool_vecs, dtype=float)
    sims = cosine_similarity(tool_vecs, context_vec)
    sims = sims - sims.max()
    contextual = np.exp(sims) / np.exp(sims).sum()
    scores = w_empirical * empirical + (1.0 - w_empirical) * contextual
    order = np.argsort(-scores, kind="stable")
    return order[:k].tolist(), scores


def learn_blend_weight(train_steps, tool_vecs, trie, grid=None):
    """Grid-search ``w_empirical`` on training steps (maximize top-1 acc).

    ``train_steps``: iterable of (prefix, context_vec, true_tool). The trie
    used for scoring is the caller's (already built); each step's own
    sequence must NOT be in the trie (caller splits train/test trajectories
    before insertion) to avoid self-leakage.
    """
    _need_numpy()
    if grid is None:
        grid = [round(x * 0.1, 1) for x in range(11)]
    best_w, best_acc = 0.5, -1.0
    for w in grid:
        correct = 0
        for prefix, context_vec, true_tool in train_steps:
            top1 = tracer_retrieve(prefix, context_vec, tool_vecs, trie,
                                   k=len(tool_vecs), w_empirical=w)[0][0]
            correct += (top1 == true_tool)
        acc = correct / max(1, len(train_steps))
        if acc > best_acc:
            best_acc, best_w = acc, w
    return best_w, best_acc


def tracer_top1_accuracy(steps, tool_vecs, trie, k=_DEFAULT_K,
                         w_empirical=0.5, baseline_full_registry=True):
    """Compare agent tool-selection accuracy with/without the TRACER front-end.

    Full-registry regime: the agent picks by contextual-softmax argmax over
    all tools. TRACER regime: the agent is shown only the blended top-``k``
    retrieval and takes its rank-1 tool (the learned blend *is* the ranking
    the paper's front-end returns; "ranks continuations using a learned
    blend of empirical and contextual priors").

    Returns a dict with per-regime accuracy and mean tools exposed per step.
    """
    _need_numpy()
    n_tools = len(tool_vecs)
    tool_vecs = np.asarray(tool_vecs, dtype=float)
    results = {"tracer": {"correct": 0, "exposed": []},
               "full": {"correct": 0, "exposed": []}}
    for prefix, context_vec, true_tool in steps:
        sims = cosine_similarity(tool_vecs, np.asarray(context_vec, float))
        if baseline_full_registry:
            pick_full = int(np.argmax(sims))
            results["full"]["correct"] += (pick_full == true_tool)
            results["full"]["exposed"].append(n_tools)
        retrieved, _scores = tracer_retrieve(prefix, context_vec, tool_vecs,
                                             trie, k=k,
                                             w_empirical=w_empirical)
        pick_tracer = retrieved[0]
        results["tracer"]["correct"] += (pick_tracer == true_tool)
        results["tracer"]["exposed"].append(len(retrieved))
    n = max(1, len(steps))
    return {
        "tracer_accuracy": results["tracer"]["correct"] / n,
        "full_accuracy": (results["full"]["correct"] / n
                          if baseline_full_registry else None),
        "tracer_mean_exposed": float(np.mean(results["tracer"]["exposed"])),
        "full_mean_exposed": float(np.mean(results["full"]["exposed"])),
        "n_steps": n,
    }
