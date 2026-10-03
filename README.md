# arxiv-lab

Portable CPU implementations of validated arXiv-paper findings, built for
local AI agents. Each module reproduces one core mechanism from a recent
arXiv paper — the same mechanisms that were validated as minimal runnable
experiments by the daily read-and-test runs of
[lesterppo/arxiv-gem-digest](https://github.com/lesterppo/arxiv-gem-digest)
(see its `experiments/YYYY-MM-DD/` folders for the original experiment
records). This repo is the *library* form of those findings: clean imports,
stable APIs, and a test suite.

Runs on **general Linux** with Python ≥ 3.10. Standard library only;
`numpy` is optional (needed by `council`, `memory`, `sampling` — those
modules raise a clear `ImportError` without it, and their tests skip
gracefully). No credentials, no network calls in the library itself —
model backends take keys only from env vars or explicit arguments and never
log them.

## Quickstart

```bash
git clone https://github.com/lesterppo/arxiv-lab.git
cd arxiv-lab
python3 tests/test_harness_cpu.py   # no install needed; tests add src/ to sys.path
```

Optional: `pip install numpy` enables the council/memory/sampling modules.

```python
import sys; sys.path.insert(0, "src")

# 1. Harness: keep a small model's tool prefill inside a byte budget
from arxiv_lab.harness import PrefillBudget, FinishGate, LoopDetector
prefill, n_bytes, compressed = PrefillBudget(2048).build_prefill(system, tools)

# 2. Council: calibrated multi-agent voting robust to adversarial agents
from arxiv_lab.council import bda_fit
posterior, reliabilities = bda_fit(n_answers=3, obs=typed_moves)

# 3. Memory: unbiased memory-utility estimates via retrieval intervention
from arxiv_lab.memory import MemoryStore, cmp_balanced_design
store = MemoryStore(rng); ate, valid = cmp_balanced_design(store, rng)

# 4. Text: model-free entropy-rate diversity scoring
from arxiv_lab.text import text_entropy_rate
h = text_entropy_rate(open("corpus.txt").read())   # low h -> collapsed/repetitive

# 5. Sampling: forward-KL mode-covering fits
from arxiv_lab.sampling import snis_weights, forward_kl_fit
a, w, ess = snis_weights(rng, 200_000, alpha=1.0)
mu, sg, pi = forward_kl_fit(a, w, n_comp=2)

# 6. Backends: one chat interface for NVIDIA NIM / DeepSeek / OpenAI / Ollama
from arxiv_lab.backends import OpenAICompatClient
reply = OpenAICompatClient.for_preset("nvidia").chat(
    "nvidia/llama-3.1-8b-instruct",
    [{"role": "user", "content": "Hello"}])
```

## Module map

| Module | Paper | What it implements |
|---|---|---|
| `arxiv_lab.harness` | [2610.02001](https://arxiv.org/abs/2610.02001) Mingbird | `PrefillBudget` (byte-level net-zero prefill budget), `FinishGate` (re-reads task before accepting completion), `LoopDetector` (signature-level loop detection) |
| `arxiv_lab.council` | [2610.02005](https://arxiv.org/abs/2610.02005) Bayesian Dialectical Argumentation | Typed-move annotator model + EM (`bda_fit`); per-agent reliabilities; adversaries inverted, not just outvoted |
| `arxiv_lab.memory` | [2610.02070](https://arxiv.org/abs/2610.02070) Causal Memory Policy | Retrieval-intervention utility estimation (`cmp_balanced_design`, SNIPW); fixes the positivity violation of store-level intervention |
| `arxiv_lab.text` | [2610.01493](https://arxiv.org/abs/2610.01493) No Model Required | Kontoyiannis entropy-rate estimator `h_k` from match-length statistics — model-free text-diversity scoring |
| `arxiv_lab.sampling` | [2610.02198](https://arxiv.org/abs/2610.02198) FERPO | Forward-KL mode-covering policy fits via SNIS (`snis_weights`, `forward_kl_fit`) vs reverse-KL mode seeking |
| `arxiv_lab.backends` | — | `OpenAICompatClient`: OpenAI-compatible `/chat/completions` over stdlib urllib; presets for `openai`, `nvidia`, `deepseek`, `ollama` (+ `base_url` override for vLLM etc.) |

## Tests

```bash
for t in tests/test_*_cpu.py; do python3 "$t"; done   # deterministic, seeded, <30s total
python3 tests/test_backends_live.py                    # skipped unless an API key env var is set
```

| Test file | Covers | Needs |
|---|---|---|
| `test_harness_cpu.py` | prefill budget / finish gate / loop detection vs scripted failure modes | stdlib |
| `test_text_cpu.py` | h_k calibration, collapse tracking, diversity filtering | stdlib |
| `test_council_cpu.py` | BDA calibration + adversarial robustness + agent inversion | numpy (else skip) |
| `test_memory_cpu.py` | positivity violation + CMP unbiasedness + AUC gain | numpy (else skip) |
| `test_sampling_cpu.py` | SNIS ESS vs KL temperature; forward-KL mode coverage | numpy (else skip) |
| `test_backends_live.py` | one tiny chat call per configured backend | API key env var (else skip) |

## Design notes

- These are **mechanism-level** reproductions: faithful to each paper's core
  claim, small enough to run on CPU in seconds — not full paper replications
  (which need real models and GPU runs).
- Findings flow one way: `arxiv-gem-digest` experiments validate →
  `arxiv-lab` implements. Experiment notes live in the digest repo; this repo
  holds only the cleaned-up library + tests.
- Portability is a hard requirement: no VM-specific paths, no credentials in
  code, no network at import time.
