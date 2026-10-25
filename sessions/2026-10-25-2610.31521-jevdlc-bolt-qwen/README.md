# 2610.31521 Jev-DLC — head-to-head: Qwen3.5-9B as frozen scorer

Second scorer run of the Jev-DLC test (see sibling session
`2026-10-25-2610.31521-jevdlc-bolt/` for the Bolt run and the paper
summary). Run at Peter's go-ahead after the question "is Bolt better
than Qwen 9B for arXiv tests": the only honest answer is the same
stream on both scorers. Everything is identical to the Bolt run except
the scorer model: same seed (20261025), same 8 users, same hidden
rankings, same slates, same contexts, same channel constants
(α=0.5, count weight 2.0, η=0.25, σ=0.1), same predict-then-update
protocol, temp 0, think off.

## Scorer differences (measured, not assumed)

- Bolt: first-token logprobs are sharp — top letter near 0, others
  −3 to −13. The frozen prior is stubborn: a history prompt barely
  moves it.
- Qwen3.5-9B: first-token logprobs are flat — top letter −0.93,
  second −1.49 (probe values, identical on both VMs). The prior is
  easily moved by prompt context.
- Qwen tokenizer emits both `A` and ` A` in the top 20; the driver
  takes the max per letter (the best single-token form, the same
  quantity the Bolt run measured).

## Results (user-macro accuracy, 192/192 events, 0 scorer errors)

| slice | Bolt frozen | Bolt Recent-4 | Bolt Jev-DLC | Qwen frozen | Qwen Recent-4 | Qwen Jev-DLC |
|---|---|---|---|---|---|---|
| overall (n=192) | 30.7% | 36.5% | 37.5% | 27.6% | 71.4% | 65.6% |
| depth ≥ 2 (n=176) | 31.3% | 36.4% | 38.6% | 27.8% | 73.3% | 69.3% |
| depth ≥ 4 (n=160) | 31.3% | 37.5% | 39.4% | 28.1% | 73.8% | 70.6% |
| depth ≥ 8 (n=128) | 33.6% | 37.5% | 43.0% | 29.7% | 70.3% | 76.6% |

Qwen per-user overall: Jev-DLC [0.583, 0.750, 0.375, 0.542, 0.875,
0.708, 0.750, 0.667]; Recent-4 [0.625, 0.667, 0.667, 0.708, 1.000,
0.708, 0.708, 0.625] — Jev-DLC wins 3 users, loses 4, ties 1.

**Verdict for the Qwen run: INCONCLUSIVE** by the pre-registered rule
(Jev-DLC 65.6% < Recent-4 71.4% overall and at depth ≥ 2, but >
frozen 27.6%, so not NEGATIVE).

## Interpretation

- As frozen scorers, Bolt and Qwen are comparable (30.7% vs 27.6%).
- As feedback learners they are not: Qwen's in-context learning is
  far stronger. Recent-4 history replay jumps to 71.4% on Qwen vs
  36.5% on Bolt, and Jev-DLC channels lift Qwen from 27.6% to 65.6%
  (+38 points) while lifting Bolt only 30.7% to 37.5% (+7 points).
- So the value of Jev-DLC's compact state depends on the scorer.
  On Bolt (stubborn prior), channels beat history replay at every
  depth. On Qwen (flat prior, strong in-context learning), plain
  history replay beats the channels — except at depth ≥ 8, where
  channels overtake replay (+6.3 points), matching the paper's depth
  story: a 4-item history window saturates, accumulated state keeps
  compounding.
- Practical answer to "which is better for arXiv tests": for the
  highest accuracy on a feedback-learning test, Qwen + Recent-4
  (71.4%) crushes Bolt + Jev-DLC (37.5%). For demonstrating
  Jev-DLC's value over history replay, Bolt is the better test bed,
  because its stubborn prior makes the channels' contribution
  visible. For any test needing weight updates, Qwen wins by default:
  Bolt cannot be trained on a T4 (see the Bolt session README).
- Absolute levels are not comparable to the paper's TCPA-Bench
  numbers (44.83% / 42.40%): this is a synthetic stream with
  deterministic rankings, which in-context learning exploits far
  more easily. The valid comparison is within-stream, scorer vs
  scorer, which is what this run is.

## Run history (honest log of failures)

1. Second Colab session on kyu008008 failed: free Colab allows one
   GPU per account (`gpu-unavailable`). No account switch was made
   at that point, to protect the live Bolt supervisor.
2. First Qwen run scored 0/192: every scorer call raised
   `KeyError('')`. Cause: Ollama's top-20 includes an empty-token
   entry, and `'' in "ABCD"` is True in Python (empty string is a
   substring of every string), so the max() lookup crashed. Fixed
   to tuple membership `tok in ("A", "B", "C", "D")`.
3. The rerun was killed mid-stream when the original kyu008008
   session was reclaimed (~00:07–00:12 HKT, 2026-10-26); its VM-side
   outputs were lost. Bolt was redeployed on cyc236de after Peter
   authorized automatic use of cyc236de/cyc236hk (superseding the
   old never-touch rule), new tunnel
   `https://skirt-recognized-passive-airlines.trycloudflare.com`.
4. The Qwen pull + probe + stream in this record ran on the new
   cyc236de VM (Bolt stopped, VRAM 0, then Qwen loaded; probe
   values identical to the old VM). Host-side runtime restarts
   killed the supervisor process twice; it was restarted by hand
   each time and picked up the live session with no redeploy.
5. After this run, Bolt was restored on the same VM: `ollama stop`
   Qwen, chat request to the Bolt variant answered "4", VRAM
   7,859 MiB, supervisor heartbeating.

## Files

- `jevdlc_qwen.py` — driver (identical stream, Qwen scorer, fixed
  token handling)
- `jevdlc_qwen_events.csv` — per-event predictions (192 rows)
- `jevdlc_qwen_summary.json` — user-macro accuracies + verdict
