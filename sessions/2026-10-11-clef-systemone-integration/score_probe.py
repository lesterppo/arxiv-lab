#!/usr/bin/env python3
"""Probe which 'criteria' shape the /v1/systemone 'score' type accepts."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:11434"
STATE = "Production database latency p99 is 4.2s, error rate 2.1%."

SHAPES = {
    "array_of_strings": ["0: no urgency", "0.5: moderate", "1: drop everything"],
    "array_of_numbers": [0, 0.5, 1],
    "array_of_objects": [
        {"score": 0, "description": "no urgency"},
        {"score": 1, "description": "drop everything"},
    ],
    "no_criteria": None,
}

for label, criteria in SHAPES.items():
    q = {"type": "score", "instructions": "Rate response urgency 0-1."}
    if criteria is not None:
        q["criteria"] = criteria
    body = json.dumps({"model": "clef-flash:latest", "state": STATE,
                       "questions": {"urgency": q}}).encode()
    req = urllib.request.Request(BASE + "/v1/systemone", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read().decode())
        print(f"{label}: HTTP {resp.status} ->",
              json.dumps(data["answers"]["urgency"])[:200])
    except urllib.error.HTTPError as e:
        print(f"{label}: HTTP {e.code} ->", e.read().decode()[:160])
    except Exception as e:  # noqa: BLE001
        print(f"{label}: {type(e).__name__}: {e}")
