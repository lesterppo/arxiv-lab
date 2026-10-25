"""SYNAPSE: Separate Field Scoring Protects Unused Tools When Agents Pool
Tool-Use Experience.

arXiv:2610.30867 — "Separate Field Scoring Protects Unused Tools When
Agents Pool Tool-Use Experience".

Paper claims implemented here (as a deterministic synthetic-embedding
simulation, not a rerun on the paper's catalogs/embedders):

* C1. Pooling usage examples into the retrieval index as extra entries for
  used tools raises Hit@5 for those tools but LOWERS it for held-out tools
  (tools with no usage at all), by a large margin on two of three catalogs
  and with both embedders tested.
* C2. A control with the same number of equally long entries filled with
  OTHER tools' text lowers held-out Hit@5 about as much, but gives little
  or none of the used-tool gain — so the held-out harm is an index-crowding
  effect, while the used-tool gain needs usage about the right tool.
* C3. SYNAPSE — scoring usage as its OWN FIELD, separate from the
  description field — protects held-out tools (held-out Hit@5 above the
  naive-pooling level, on StableToolBench 0.785 vs a slot-reserving router's
  0.683) while retaining the used-tool gain; the protection persists when
  usage is replaced with other tools' text.

Simulation design (all choices stated, so the result is auditable rather
than a fitted curve):

* 240 tools in 24 confusable groups of 10. Each tool has an intent vector
  (group centroid weighted 2.0 + tool-specific unit vector weighted 1.0, so
  same-group intents correlate ~0.8 and retrieval is hard but not random).
* A tool's description vector and its usage centroid are two INDEPENDENT
  noisy observations of the same intent (unit-noise weight 0.90 each); a
  query is a third such observation. This is what makes the field
  combination meaningful: averaging two fields reduces score variance,
  while taking the max over many index entries inflates it.
* Naive pooling: each used tool contributes 4 usage-example entries to the
  index and a tool's score is the max over its entries — extra entries are
  extra chances for a competitor to spike above the gold tool.
* SYNAPSE: each tool contributes exactly one candidate; a used tool's score
  is the mean of its description-field similarity and its usage-field
  similarity (usage centroid over its 4 examples); a held-out tool's score
  is its description similarity alone. No tool ever gets extra index slots.
* Shuffled controls permute the usage bundles among used tools, keeping
  entry counts identical and breaking only the usage-to-tool match.
* Constant-offset control (+0.15 to every used tool) isolates "used tools
  just get a boost" from field averaging.

Requires numpy (optional dependency): ``pip install numpy``.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

__all__ = [
    "Catalog",
    "N_QUERIES_PER_CLASS",
    "N_TOOLS",
    "build_catalog",
    "hit_at_k",
    "run_protocol",
]

# Module-level constants are numpy-free (plain tuples/ints).
N_TOOLS = 240
N_TOPICS = 24
GROUP_SIZE = 10
DIM = 64
N_USED = 120
USAGE_EXAMPLES = 4
N_QUERIES_PER_CLASS = 500
K = 5
OFFSET = 0.15
SEED = 20261025


def _need_numpy():
    if np is None:  # pragma: no cover - optional dependency
        raise ImportError("numpy is required for arxiv_lab.retrieval.synapse")


class Catalog:
    """Synthetic tool catalog with description and usage fields."""

    def __init__(self, desc, intent, used, usage_examples, usage_centroid):
        self.desc = desc
        self.intent = intent
        self.used = used
        self.usage_examples = usage_examples
        self.usage_centroid = usage_centroid


def _unit(rng, rows, dim=DIM):
    vec = rng.normal(size=(rows, dim))
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def build_catalog(seed=SEED):
    """Build the synthetic catalog (deterministic for a fixed seed)."""
    _need_numpy()
    rng = np.random.default_rng(seed)
    centroids = _unit(rng, N_TOPICS)
    intent = np.repeat(centroids, GROUP_SIZE, axis=0) * 2.0 + _unit(rng, N_TOOLS)
    intent = intent / np.linalg.norm(intent, axis=1, keepdims=True)
    desc = intent + _unit(rng, N_TOOLS) * 0.90
    desc = desc / np.linalg.norm(desc, axis=1, keepdims=True)
    used = np.zeros(N_TOOLS, dtype=bool)
    used[:N_USED] = True
    rng.shuffle(used)
    usage_examples = {}
    usage_centroid = {}
    for tool in np.where(used)[0]:
        examples = intent[tool] + _unit(rng, USAGE_EXAMPLES) * 0.90
        examples = examples / np.linalg.norm(examples, axis=1, keepdims=True)
        usage_examples[int(tool)] = examples
        centroid = examples.mean(axis=0)
        usage_centroid[int(tool)] = centroid / np.linalg.norm(centroid)
    return Catalog(desc, intent, used, usage_examples, usage_centroid)


def _shuffled_usage(catalog, rng):
    """Permute usage bundles among used tools (counts unchanged)."""
    used_tools = [int(t) for t in np.where(catalog.used)[0]]
    perm = rng.permutation(used_tools)
    examples = {t: catalog.usage_examples[int(p)] for t, p in zip(used_tools, perm)}
    centroid = {t: catalog.usage_centroid[int(p)] for t, p in zip(used_tools, perm)}
    return examples, centroid


def hit_at_k(scores, golds, k=K):
    """Fraction of queries whose gold tool ranks in the top-k scores."""
    _need_numpy()
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return float(np.mean([golds[i] in top[i] for i in range(len(golds))]))


def run_protocol(seed=SEED, n_queries=N_QUERIES_PER_CLASS):
    """Run base / naive / SYNAPSE / shuffled / offset retrieval conditions.

    Returns a dict of condition -> {"all": ..., "held": ..., "used": ...}
    Hit@5 values, measured on one shared query draw per condition (queries
    are re-drawn per condition from the same rng stream, as in the scratch
    harness; the canonical seed's values are asserted in the CPU test).
    """
    _need_numpy()
    catalog = build_catalog(seed)
    rng = np.random.default_rng(seed + 1)
    # Advance the rng past catalog construction identically for every
    # condition by re-seeding per condition below; instead, draw one query
    # set up front and share it across conditions (cleaner + comparable).
    held_golds = rng.choice(np.where(~catalog.used)[0], size=n_queries, replace=True)
    used_golds = rng.choice(np.where(catalog.used)[0], size=n_queries, replace=True)
    golds = np.concatenate([held_golds, used_golds])
    is_used = np.concatenate(
        [np.zeros(n_queries, dtype=bool), np.ones(n_queries, dtype=bool)]
    )
    queries = catalog.intent[golds] + _unit(rng, len(golds)) * 0.90
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    desc_sim = queries @ catalog.desc.T

    shuffled_examples, shuffled_centroid = _shuffled_usage(catalog, rng)

    def score(condition, examples, centroid):
        if condition == "base":
            return desc_sim
        if condition == "naive":
            scores = desc_sim.copy()
            for tool, entries in examples.items():
                scores[:, tool] = np.maximum(
                    scores[:, tool], (queries @ entries.T).max(axis=1)
                )
            return scores
        if condition == "synapse":
            scores = desc_sim.copy()
            for tool, center in centroid.items():
                scores[:, tool] = 0.5 * desc_sim[:, tool] + 0.5 * (queries @ center)
            return scores
        if condition == "offset":
            scores = desc_sim.copy()
            scores[:, catalog.used] += OFFSET
            return scores
        raise ValueError(f"unknown condition: {condition}")

    conditions = {
        "base": ("base", {}, {}),
        "naive": ("naive", catalog.usage_examples, catalog.usage_centroid),
        "naive_shuffled": (
            "naive",
            shuffled_examples,
            shuffled_centroid,
        ),
        "synapse": ("synapse", catalog.usage_examples, catalog.usage_centroid),
        "synapse_shuffled": (
            "synapse",
            shuffled_examples,
            shuffled_centroid,
        ),
        "offset": ("offset", {}, {}),
    }
    results = {}
    for name, (mode, examples, centroid) in conditions.items():
        scores = score(mode, examples, centroid)
        results[name] = {
            "all": hit_at_k(scores, golds),
            "held": hit_at_k(scores[~is_used], golds[~is_used]),
            "used": hit_at_k(scores[is_used], golds[is_used]),
        }
    return results
