# arxiv-lab daily read-and-test — replicable workflow (v3, 2026-10-15)

This document lets **any AI agent, on any machine**, replicate the daily arXiv
screen-and-test job from this repo alone and continue the task. It is the
portable form of the operator's cron protocol; keep it in sync when the
protocol changes.

## 0. Purpose

Every day: screen the newest arXiv cs.AI papers, pick the 1–2 most
prospective **testable** articles on AI agents and model training/fine-tuning,
run **live tests**, and bank one session per tested article in this repo —
a growing library of **verified mechanisms with honest verdicts** (including
clean negatives). Fit every step to that purpose: no theater runs, no silent
skips, no unverified pushes.

## 1. Prerequisites

- Python ≥ 3.10. Workflow scripts (`workflow/`) are **stdlib only**.
  Library tests need `numpy` only for some modules (they skip gracefully).
- GitHub write access to this repo: `gh` CLI authenticated, or
  `GITHUB_TOKEN` env var for `workflow/verify_push.py`.
- Network access to arXiv (`export.arxiv.org`, `arxiv.org`,
  `rss.arxiv.org`, `oaipmh.arxiv.org`).
- Optional, training track: a ~16 GB GPU (reference: free Colab T4) and
  whatever remote-execution tooling you prefer.
- Optional, model-backed validation: any OpenAI-compatible chat endpoint;
  the key enters via **env var only**, never files, never logs.

## 2. Schedule and idempotency

- Run **once daily** (reference schedule: 12:20 Asia/Hong_Kong).
- **First step, every run:** list `sessions/` for entries starting with
  today's date (`YYYY-MM-DD` in the operator's local timezone). If any exist,
  a same-day run already happened — do NOT re-run; summarize the day from
  the session READMEs and stop.
- An explicit operator instruction to run overrides the check, but **never
  re-test an article that already has a session** — pick different papers.

## 3. Verdict vocabulary (controlled — exactly one per tested article)

- **VERIFIED** — core claim reproduced end-to-end with measured numbers.
- **SUPPORTED** — reproduced directionally / at bounded scale; stated limits.
- **PARTIAL** — some claims tested, remainder blocked; resume state recorded.
- **NEGATIVE** — honest negative: claim did not reproduce under a fair test.
- **INCONCLUSIVE** — the test design could not discriminate (ceiling effect,
  broken operationalization, …); say exactly why.
- **BLOCKED** — no fair test was possible (infra/auth/reclamation); scripts
  staged, resume state recorded.

A PARTIAL/BLOCKED training track is a valid day outcome, never a run failure.

## 4. Steps

### 4.1 Fetch — `workflow/fetch_arxiv.py`

```bash
python3 workflow/fetch_arxiv.py --days 3 --max 200 --out papers.json
```

Three-tier ladder, first source that yields in-window papers wins:

1. `api` — arXiv API `search_query=cat:cs.AI`, `sortBy=submittedDate`
   descending, `max_results=200`, hosts `export.arxiv.org` → `arxiv.org`;
   filter locally by Atom `published` to the last `--days` days.
   **Never** use a `submittedDate:[...TO...]` range query (HTTP 500).
2. `rss` — `https://rss.arxiv.org/rss/cs.AI`. Announcement-fresh, but
   legitimately **empty on Sat/Sun**.
3. `oai` — OAI-PMH `ListRecords`, `metadataPrefix=arXiv`, `set=cs:cs`,
   `from`/`until` = window; follow `resumptionToken`; drop records whose
   `created` date is > 30 days old (metadata churn).

Retry ladder on 429/5xx with exponential backoff, honour `Retry-After`.
Etiquette: ≥ 3 s between calls, `User-Agent:
arxiv-gem-digest/1.0 (+https://github.com/lesterppo/arxiv-gem-digest)`.
The script prints the serving source — **tag the day's report with it**
(e.g. `(OAI fallback)`). If every tier errors, exit 2: record the outage as
BLOCKED, never a green "nothing new".

### 4.2 Dedup — `workflow/covered_ids.py`

```bash
python3 workflow/covered_ids.py --remote --out covered_ids.txt
```

Builds the covered arXiv-id set from `sessions/` directory names **and**
the repo's remote tree (ids normalized, version suffix stripped), one id per
line. **Skip covered ids before reading abstracts.**

### 4.3 Screen + score

Read titles/abstracts yourself; score each candidate **AGENT** and **TUNE**
1–5:

- **AGENT** — applicability to building/improving a *local AI-agent
  harness* (tool calling, memory, agent loops, multi-agent orchestration,
  evals, web/software agents, code gen). 5 = directly reusable on a local
  harness.
- **TUNE** — applicability to *small open-weights model fine-tune/training*
  that fits a single free Colab T4 (16 GB VRAM) or small local GPU.
  5 = practical on a free T4. A method needing a GPU cluster is TUNE-low —
  never force it into a T4 recipe.

Pick the **1 or 2 most prospective TESTABLE** articles — prefer one agent +
one training article when both clear the bar (**AGENT/TUNE ≥ 4**).
Tie-breakers, in order: **(a)** testable within the run budget (CPU or one
GPU session), **(b)** repo fit (a fine-tune method adaptable to a daily
QLoRA harness is strongly TUNE-relevant; agent mechanisms that generalize
beyond one benchmark are strongly AGENT-relevant), **(c)** novelty vs
near-misses. Record **near-misses** (id, title, AGENT/TUNE, one line why not
picked) in the run log so a later day can reconsider without re-screening.

### 4.4 Live test — agent track

Implement the paper's core mechanism as a module under
`src/arxiv_lab/<area>/` following `AGENTS.md` (stdlib + numpy only,
portable, no secrets, no network at import; docstring cites the arXiv ID +
one-line claim). Write a **deterministic CPU test** under `tests/`
reproducing the core claim with measured numbers and assertions.
Model-backed validation via an OpenAI-compatible endpoint is allowed;
keep any credential shim out of the repo.

### 4.5 Live test — training/fine-tuning track

Run on a ~16 GB GPU (reference: Colab T4). Reference recipe: a ~9B model at
Q4_K_M (~7 GB VRAM) + NF4 QLoRA (r=16/alpha=32, bf16, tiny data, few steps).
Colab quirk, if you use Colab: `pip uninstall -y torchao` right after pip
installs (Colab ships torchao 0.10; peft needs > 0.16, else ImportError at
`get_peft_model`). **Never fake a GPU run** — with no GPU, stage the
self-contained script and report BLOCKED.

**Carry-over rule:** the training track is an independent phase. If the GPU
run is reclaimed, OOMs, or is still executing at report time: stage the
scripts + CPU checks, record verdict PARTIAL or BLOCKED with the **exact
resume state** (what completed, script paths, how to resume), and report —
the day still succeeds.

### 4.6 Record — one session per tested article

`sessions/YYYY-MM-DD-<arxivid>-<slug>/README.md` containing: article link,
why picked, what was built, results + **exactly one controlled VERDICT**,
honest limits — plus the article-specific code/tests/GPU scripts.
Mechanisms that come out VERIFIED/SUPPORTED graduate to
`src/arxiv_lab/<area>/` + a CPU test.

Also keep a **per-day run log** (see `workflow/runlog_template.md`): screen
summary, picks, near-misses, per-track verdicts + key numbers, push
verification count.

### 4.7 Push

Push the day's session files + graduated `src/`/`tests/` files to this repo.
**Secrets-grep every file first** (`workflow/verify_push.py` does this);
abort the push on any hit. Never commit secrets, credentials, tokens, or
credential shims.

### 4.8 Verify — `workflow/verify_push.py`

```bash
GITHUB_TOKEN=... python3 workflow/verify_push.py
```

Compares local git blob-shas against the repo's recursive tree API;
re-push anything missing inside the run and report the **verified file
count**. Never report "push successful" from a push call alone.

### 4.9 Report

The day's picks (id, title, AGENT/TUNE, track), one controlled verdict each
with key measured numbers, session paths, serving fetch source, and anything
blocked/carried over with resume state. If arXiv is unreachable everywhere
or nothing testable clears AGENT/TUNE ≥ 4, still record the day: screen
notes + why nothing was tested, verdict BLOCKED — **never silently skip**.

## 5. Repo conventions (summary of AGENTS.md)

- **No secrets in this repo, ever.** Keys via env vars only.
- Portable Linux; stdlib only (+ numpy optional); no network at import time.
- Keep public API names stable; numpy-guarded modules degrade gracefully.
- Findings flow one way: experiments validate → `src/arxiv_lab/` implements.
