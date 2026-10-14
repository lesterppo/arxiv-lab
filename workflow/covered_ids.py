#!/usr/bin/env python3
"""Build the covered arXiv-id set: sessions/ dir names + optional remote tree.

Usage:
    python3 covered_ids.py [--root .] [--remote] [--repo owner/name]
                           [--out covered_ids.txt]

--remote needs GITHUB_TOKEN env var. Ids are normalized (version stripped),
one per line, sorted.
"""
import argparse
import json
import os
import re
import sys
import urllib.request

DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-(\d{4}\.\d{4,5}(?:v\d+)?)-")


def norm_id(pid):
    pid = pid.strip()
    if "v" in pid:
        head, tail = pid.rsplit("v", 1)
        if tail.isdigit():
            return head
    return pid


def local_ids(root):
    ids = set()
    sdir = os.path.join(root, "sessions")
    if not os.path.isdir(sdir):
        return ids
    for name in os.listdir(sdir):
        m = DIR_RE.match(name)
        if m:
            ids.add(norm_id(m.group(1)))
    return ids


def remote_ids(repo):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("--remote needs GITHUB_TOKEN env var")
    url = (f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1")
    req = urllib.request.Request(
        url, headers={"User-Agent": "arxiv-lab-workflow",
                      "Authorization": f"Bearer {token}",
                      "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        tree = json.load(r)["tree"]
    ids = set()
    for t in tree:
        if t.get("type") != "blob":
            continue
        m = re.match(r"^sessions/\d{4}-\d{2}-\d{2}-"
                     r"(\d{4}\.\d{4,5}(?:v\d+)?)-", t["path"])
        if m:
            ids.add(norm_id(m.group(1)))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--repo", default="lesterppo/arxiv-lab")
    ap.add_argument("--out", default="covered_ids.txt")
    args = ap.parse_args()

    ids = local_ids(args.root)
    print(f"local sessions: {len(ids)} ids", file=sys.stderr)
    if args.remote:
        rids = remote_ids(args.repo)
        print(f"remote tree: {len(rids)} ids", file=sys.stderr)
        ids |= rids
    with open(args.out, "w") as f:
        for pid in sorted(ids):
            f.write(pid + "\n")
    print(f"wrote {len(ids)} covered ids -> {args.out}")


if __name__ == "__main__":
    main()
