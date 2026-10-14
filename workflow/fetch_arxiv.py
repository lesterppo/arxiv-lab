#!/usr/bin/env python3
"""Three-tier arXiv cs.AI fetch: api -> rss -> oai. Stdlib only.

Usage:
    python3 fetch_arxiv.py --days 3 --max 200 --out papers.json

Writes a JSON list of {"id","title","summary","published"} for papers in the
window. Prints the serving source to stdout. Exits 2 if every tier errored.

Etiquette: >=3s between calls, UA header below. Never use a
submittedDate:[...TO...] range query on the API (HTTP 500).
"""
import argparse
import datetime
import email.utils
import json
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

UA = "arxiv-gem-digest/1.0 (+https://github.com/lesterppo/arxiv-gem-digest)"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
OAI = {"oai": "http://www.openarchives.org/OAI/2.0/",
       "arx": "http://arxiv.org/OAI/arXiv/"}


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_with_retry(url, tries=3):
    last = None
    for i in range(tries):
        try:
            return http_get(url)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                try:
                    wait = int(e.headers.get("Retry-After", 0)) or (2 ** i * 3)
                except ValueError:
                    wait = 2 ** i * 3
                time.sleep(wait)
                continue
            raise
        except Exception as e:  # transient socket/DNS errors
            last = e
            time.sleep(2 ** i * 3)
    raise last


def norm_id(raw):
    """'http://arxiv.org/abs/2610.18522v2' -> '2610.18522'."""
    pid = raw.strip().rsplit("/abs/", 1)[-1]
    if "v" in pid:
        head, tail = pid.rsplit("v", 1)
        if tail.isdigit():
            return head
    return pid


def collapse(text):
    return " ".join((text or "").split())


def tier_api(max_results, cutoff):
    """Newest cs.AI via the arXiv API, filtered locally by published date."""
    errs = []
    for host in ("export.arxiv.org", "arxiv.org"):
        try:
            url = (f"https://{host}/api/query?search_query=cat:cs.AI&start=0"
                   f"&max_results={max_results}&sortBy=submittedDate"
                   f"&sortOrder=descending")
            root = ET.fromstring(get_with_retry(url))
            papers = []
            for e in root.findall("a:entry", ATOM):
                pid = norm_id(e.find("a:id", ATOM).text)
                title = collapse(e.find("a:title", ATOM).text)
                summary = collapse(e.find("a:summary", ATOM).text)
                pub_txt = e.find("a:published", ATOM).text
                dt = datetime.datetime.fromisoformat(
                    pub_txt.replace("Z", "+00:00"))
                if dt >= cutoff:
                    papers.append({"id": pid, "title": title,
                                   "summary": summary, "published": pub_txt})
            return papers, f"api ({host})"
        except Exception as e:
            errs.append(f"{host}: {e}")
            time.sleep(3)
    raise RuntimeError("api tier failed: " + " | ".join(errs))


def tier_rss(cutoff):
    """arXiv cs.AI RSS. Announcement-fresh; legitimately empty on Sat/Sun."""
    root = ET.fromstring(get_with_retry("https://rss.arxiv.org/rss/cs.AI"))
    papers = []
    for item in root.iter("item"):
        title = collapse(item.findtext("title"))
        link = (item.findtext("link") or "").strip()
        desc = collapse(item.findtext("description"))
        pub_txt = (item.findtext("pubDate") or "").strip()
        try:
            dt = email.utils.parsedate_to_datetime(pub_txt)
        except (TypeError, ValueError):
            continue
        if dt >= cutoff:
            papers.append({"id": norm_id(link), "title": title,
                           "summary": desc, "published": pub_txt})
    return papers, "rss"


def tier_oai(cutoff):
    """OAI-PMH ListRecords on set cs:cs with from/until window."""
    base = "https://oaipmh.arxiv.org/oai"
    frm = cutoff.date().isoformat()
    until = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    url = (f"{base}?verb=ListRecords&metadataPrefix=arXiv"
           f"&set=cs:cs&from={frm}&until={until}")
    papers, now = [], datetime.datetime.now(datetime.timezone.utc)
    while url:
        root = ET.fromstring(get_with_retry(url))
        for rec in root.findall("oai:ListRecords/oai:record", OAI):
            header = rec.find("oai:header", OAI)
            if header is not None and header.get("status") == "deleted":
                continue
            meta = rec.find("oai:metadata/arx:arXiv", OAI)
            if meta is None:
                continue
            pid = norm_id(meta.find("arx:id", OAI).text)
            title = collapse(meta.find("arx:title", OAI).text)
            abstract = collapse(meta.find("arx:abstract", OAI).text)
            created = (meta.find("arx:created", OAI).text or "").strip()
            try:
                cdt = datetime.datetime.fromisoformat(created).replace(
                    tzinfo=datetime.timezone.utc)
            except ValueError:
                continue
            if (now - cdt).days > 30:
                continue  # metadata churn, not a new paper
            if cdt >= cutoff:
                papers.append({"id": pid, "title": title,
                               "summary": abstract, "published": created})
        tok = root.find("oai:ListRecords/oai:resumptionToken", OAI)
        if tok is not None and tok.text and tok.text.strip():
            url = (f"{base}?verb=ListRecords&resumptionToken="
                   f"{tok.text.strip()}")
            time.sleep(3)
        else:
            url = None
    return papers, "oai"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--out", default="papers.json")
    args = ap.parse_args()

    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(days=args.days))
    tiers = [("api", lambda: tier_api(args.max, cutoff)),
             ("rss", lambda: tier_rss(cutoff)),
             ("oai", lambda: tier_oai(cutoff))]
    errors = []
    for name, fn in tiers:
        try:
            papers, source = fn()
        except Exception as e:
            print(f"tier {name} errored: {e}", file=sys.stderr)
            errors.append(name)
            time.sleep(3)
            continue
        if papers:
            with open(args.out, "w") as f:
                json.dump(papers, f, indent=1)
            print(f"source={source} papers={len(papers)} out={args.out}")
            return 0
        print(f"tier {name}: 0 in-window papers, trying next tier",
              file=sys.stderr)
        time.sleep(3)
    print(f"all tiers failed or empty (errors: {errors}); refusing to "
          f"report a green 'nothing new'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
