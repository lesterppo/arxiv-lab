# 2026-11-05 — FlyHash for LLM Agent Memory (arXiv:2611.05791)

- Article: https://arxiv.org/abs/2611.05791 — "FlyHash for LLM Agent Memory:
  An Empirical Validation and a New Application"
- Track: agent. Scores: AGENT=5, TUNE=1.
- Why picked: sparse random projection + winner-take-all memory codes are
  pure logic, directly reusable in a local agent-memory harness, and the
  paper itself asks three practitioner questions (quality, cost, labels)
  that a deterministic CPU test can answer separately.
- Repo fit: arxiv-lab memory library (not graduated — see verdict).

## What was built

`flyhash.py` (stdlib + numpy): FlyHash codes (each of `m` Kenyon cells
sums `s` randomly sampled input coordinates with Gaussian weights; top-`k`
activations win), SimHash codes (dense Gaussian hyperplanes, sign) at the
same bit budget `m`, exact-cosine ground truth, recall@10, sparse-set
storage accounting, and inverted-index candidate retrieval.
`test_flyhash_cpu.py` runs the full experiment with assertions.

## Results — VERDICT: NEGATIVE

Matched-bit protocol (both codes `m = 512` bits, `k = 26`, `s = 10`),
recall@10 against exact cosine neighbours, 10 seeds each:

| dataset | FlyHash | SimHash |
|---|---|---|
| clustered Gaussian (d=48, 1500 db / 150 queries) | 0.2225 | 0.5343 |
| sparse binary bag vectors (d=200) | 0.2588 | 0.3942 |

FlyHash lost on **every** seed in both datasets. Non-circular labels
(Q3): TF-IDF memories labelled by fact identity (labels from fact ids,
not from embedding neighbours), 5 seeds — fact recall FlyHash 0.7023 vs
SimHash 0.8990. The quality advantage did not reproduce, and it did not
survive the paper's own Q3 framing in this setup.

Parameter sweep (did not rescue FlyHash): `m` 512→2048→8192, `s` 10→50,
`k` 5%→50% of `m`, Gaussian vs binary projection weights, noise 0.15/0.35.
FlyHash never led; the gap only closed at `k = 50%` of `m` (0.688 vs
0.699), where the code is no longer sparse. Fixing `k = 26` and expanding
`m` alone did not help (0.42–0.44 at `m` 2048/8192 vs SimHash-512 0.699):
quality scales with the number of active cells, not with expansion.

Two secondary findings, both consistent with the paper's own caveats:
- Fixed-`k` identity: Hamming distance = `2k − 2·overlap`, so Hamming and
  overlap rankings coincide exactly for fixed-`k` codes — they are not two
  independent experiments.
- Q2 cost: sparse storage is genuinely smaller (FlyHash 52 bytes vs
  SimHash 64 bytes vs dense float32 192 bytes at d=48), but at `m=512,
  k=26` inverted-index pruning barely prunes (mean candidate fraction
  0.8135), so the sparse-set comparison does not rescue latency at
  ordinary scale — the paper's "narrower practical case" in miniature.

## Honest limits

- The paper's four real datasets and exact protocol are not recoverable
  from the abstract alone; this is a negative on synthetic reproductions
  (clustered Gaussian, sparse binary, TF-IDF fact memories), not a
  refutation of the paper's specific real-data numbers. A real-embedding
  rerun (e.g. on the paper's released implementation/datasets) could
  still overturn this verdict.
- The synthetic tasks may understate FlyHash: SimHash is a strong dense
  baseline, and the clustered task's signal-to-noise ratio punishes
  sparse sampling. The sweep across SNR, `m`, `s`, `k` bounds that
  objection but does not eliminate it.

## Rust second stage

None — standing rule (2026-11-02): no Rust leg when the Python verdict
is NEGATIVE. The mechanism is not ported, and the verdict stands on the
Python test above.

## Reproduce

```bash
cd ~/workspace/arxiv-lab-build/sessions/2026-11-05-2611.05791-flyhash
python3 test_flyhash_cpu.py
```
