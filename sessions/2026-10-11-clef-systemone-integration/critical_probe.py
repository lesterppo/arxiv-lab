#!/usr/bin/env python3
"""Re-probe: 'Payments fully down across all regions.' -> severity? (got low)"""
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
for _p in (_here, "/content", os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
if "systemone" in sys.modules:
    del sys.modules["systemone"]
from systemone import SystemoneClient  # noqa: E402

SEV_Q = {"type": "choice", "instructions": "Incident severity?",
         "criteria": {"low": "minor, no customer impact",
                      "medium": "degraded for some customers",
                      "high": "critical outage, checkout down"}}
c = SystemoneClient(model="clef-flash:latest",
                    base_url="http://localhost:11434", timeout=300)
for i in range(3):
    t0 = time.time()
    ans = c.decide("Payments fully down across all regions.", {"sev": SEV_Q})
    s = ans["sev"]
    print(f"run{i}: {time.time()-t0:.1f}s choice={s['choice']} "
          f"probs={ {k: round(v, 3) for k, v in s['probabilities'].items()} }",
          flush=True)
