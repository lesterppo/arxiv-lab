# Run log — YYYY-MM-DD

## Fetch
- Source: api | rss | oai (+ fallback note)
- Window: last N days; fetched M papers, K in-window, C covered skipped

## Picks
| # | arXiv ID | Title | AGENT | TUNE | Track | Why picked (repo fit) |
|---|----------|-------|-------|------|-------|------------------------|
| 1 |          |       |       |      | agent |                        |
| 2 |          |       |       |      | training |                     |

## Near-misses
| arXiv ID | Title | AGENT | TUNE | Why not picked |
|----------|-------|-------|------|----------------|
|          |       |       |      |                |

## Agent track
- Module: `src/arxiv_lab/<area>/<thing>.py`
- CPU test: `tests/test_<area>_cpu.py` — n/n pass
- Key numbers:
- Verdict: VERIFIED | SUPPORTED | NEGATIVE | INCONCLUSIVE

## Training track
- GPU: session name / hardware
- Key numbers:
- Verdict: VERIFIED | SUPPORTED | PARTIAL | NEGATIVE | INCONCLUSIVE | BLOCKED
- Resume state (if PARTIAL/BLOCKED):

## Push
- Files pushed: N; verified via workflow/verify_push.py: N/N match remote

## Notes / blocked
