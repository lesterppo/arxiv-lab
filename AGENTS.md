# AGENTS.md — arxiv-lab operator notes

## Non-negotiables
- **No secrets in this repo.** API keys enter only via env vars
  (`NVIDIA_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`) or explicit
  arguments to `OpenAICompatClient`. The backends module never prints/logs
  keys — keep it that way in every edit.
- **Portable Linux, always.** No VM-specific paths, no `~/.local` assumptions,
  no authd/surrogate code, no network at import time. stdlib only; numpy is
  the single optional dependency.
- Findings flow one way: `lesterppo/arxiv-gem-digest` experiments validate,
  `arxiv-lab` implements. Don't copy experiment READMEs/RESULTS.md here.

## Running tests
```bash
cd ~/workspace/arxiv-lab-build   # or wherever this repo is checked out
for t in tests/test_*_cpu.py; do python3 "$t"; done
python3 tests/test_backends_live.py   # skipped unless an API key env var is set
```
Tests insert `src/` on `sys.path` themselves — no install needed. The numpy
tests skip gracefully when numpy is absent; keep the guard when editing them.

## Editing modules
- Keep public names stable (`PrefillBudget`, `FinishGate`, `LoopDetector`,
  `bda_fit`, `cmp_balanced_design`, `entropy_rate`, `snis_weights`, ...).
- Every module docstring cites its arXiv ID + the one-line claim it
  implements. New modules from new papers follow the same pattern:
  `src/arxiv_lab/<area>/<thing>.py` + `tests/test_<area>_cpu.py`.
- numpy-guarded modules: import numpy in try/except, keep module-level
  constants numpy-free (plain tuples), call `_need_numpy()` at the top of
  every public function.

## Publishing
Target repo: `lesterppo/arxiv-lab` (created separately by the parent agent —
do not create it from here). Push via the normal git flow once the GitHub
repo exists.
