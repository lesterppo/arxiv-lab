#!/usr/bin/env python3
"""Verify local files against the GitHub repo tree (git blob-sha) + secrets scan.

Usage:
    GITHUB_TOKEN=... python3 verify_push.py [--repo owner/name] [paths...]

Default paths when none given: WORKFLOW.md, workflow/, sessions/, src/, tests/.
Compares local blob-shas with the repo's recursive tree API and lists
missing/differing files. Also scans for secrets/credentials.

Exit codes: 0 = all verified clean, 1 = diffs found (listed), 2 = secrets found.
Stdlib only.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request

TEXT_EXT = (".py", ".md", ".json", ".csv", ".txt", ".yaml", ".yml")
SKIP_DIRS = {"__pycache__", ".git", "node_modules"}
SECRET_PAT = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|passwd|bearer|hsurr:|ghp_|gho_|"
    r"sk-|AKIA)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-./+]{8,}")
ALLOW = ("enter", "via env", "never", "example", "placeholder", "your_", "<",
         "pad_token", "eos_token", "bos_token", "unk_token", "mask_token",
         "tokenizer", "autotokenizer", "tokens", "ntoken", "none_token",
         "_token =")


def blob_sha(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def collect(paths, root):
    out = {}
    for p in paths:
        ap = os.path.join(root, p) if not os.path.isabs(p) else p
        if os.path.isfile(ap):
            if ap.endswith(TEXT_EXT):
                out[os.path.relpath(ap, root)] = ap
        elif os.path.isdir(ap):
            for dp, dn, fn in os.walk(ap):
                dn[:] = [d for d in dn if d not in SKIP_DIRS]
                for f in fn:
                    if f.endswith(TEXT_EXT):
                        fp = os.path.join(dp, f)
                        out[os.path.relpath(fp, root)] = fp
        else:
            print(f"warning: path not found: {p}", file=sys.stderr)
    return out


def secrets_check(files):
    bad = []
    for rel, p in files.items():
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in SECRET_PAT.finditer(txt):
            line = txt[max(0, m.start() - 80):m.end() + 40]
            if any(k in line.lower() for k in ALLOW):
                continue
            bad.append((rel, line.strip()[:100]))
            break
    return bad


def remote_tree(repo, token):
    url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
    req = urllib.request.Request(
        url, headers={"User-Agent": "arxiv-lab-workflow",
                      "Authorization": f"Bearer {token}",
                      "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        tree = json.load(r)["tree"]
    return {t["path"]: t["sha"] for t in tree if t.get("type") == "blob"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="lesterppo/arxiv-lab")
    ap.add_argument("--root", default=".")
    ap.add_argument("paths", nargs="*",
                    default=["WORKFLOW.md", "workflow",
                             "sessions", "src", "tests"])
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN env var is required")

    files = collect(args.paths, args.root)
    print(f"local files: {len(files)}")
    bad = secrets_check(files)
    if bad:
        print("SECRET CHECK FAILED:")
        for rel, line in bad:
            print(f"  {rel}: {line}")
        return 2
    print("secrets check clean")

    remote = remote_tree(args.repo, token)
    print(f"remote blobs: {len(remote)}")
    diffs = [(rel, "missing on remote" if rel not in remote else
              f"sha differs local={blob_sha(p)[:7]} "
              f"remote={remote[rel][:7]}")
             for rel, p in sorted(files.items())
             if remote.get(rel) != blob_sha(p)]
    if diffs:
        print(f"UNVERIFIED ({len(diffs)}):")
        for rel, why in diffs:
            print(f"  {rel}: {why}")
        return 1
    print(f"verified clean: {len(files)} files match remote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
