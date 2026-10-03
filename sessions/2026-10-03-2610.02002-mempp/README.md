# Session 2026-10-03 — Mem++ (arXiv:2610.02002)

**Article:** "Mem++: Non-Destructive Memory for Long-Term Organizational LLM Agents"
([abs](https://arxiv.org/abs/2610.02002))

**Why picked:** most prospective *agent* article in the 2026-09-30→10-03
window (AGENT 4/5). Directly implementable on this VM, and it attacks the
exact weakness of the memory in Peter's local stack: DeepWorld's
`MemoryManager.compress_memory()` drops old fragments at write time — the
practice Mem++ argues destroys answerability of versioned questions.

**Paper claim:** shift from write-time distillation to read-time selection —
store every document whole with date+author (no model at write time); at
read time, filter to `date <= as_of`, fuse lexical (BM25) + semantic ranks
(RRF), return whole documents and let the answering model choose. OrgMemBench:
+8.0–13.1 pts over the strongest baseline.

## What was built (agent track, this VM)

- Implementation: `src/arxiv_lab/memory/mempp.py` — `MemStore` (non-destructive,
  no model at write), `retrieve()` (time filter → BM25 + TF-IDF-cosine RRF
  fusion → whole docs, newest-first ties). `embed_fn` hook for real dense
  embeddings.
- Test: `tests/test_memory_mempp_cpu.py` — OrgMemBench-style versioned
  decisions (budget 50k→70k→60k, retention 30d→90d), 6 as-of questions.

## Results (CPU, deterministic)

| | Mem++ | write-time-distillation baseline |
|---|---|---|
| as-of accuracy | **6/6** | 3/6 (misses exactly the backdated questions) |
| time filter | enforced (`date <= as_of`) | n/a (superseded versions destroyed) |
| model at write | none | n/a |

**Verdict: SUPPORTED.** Read-time selection answers every versioned question;
the baseline fails precisely where the paper says it must. Honest limits:
toy corpus (6 docs); semantic channel is TF-IDF cosine here, not dense
embeddings; the paper's +8–13 pt number is on OrgMemBench, not reproduced.

## Integration note

DeepWorld's `v4/engine/memory_manager.py` still compresses at write time.
`mempp.MemStore` is a drop-in non-destructive companion; wiring its
read-time `retrieve()` into the agent tick loop is queued (working-copy
change, needs Peter's go-ahead like the harness integration).
