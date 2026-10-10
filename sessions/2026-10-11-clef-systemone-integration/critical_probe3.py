#!/usr/bin/env python3
"""Isolate the short-phrase misfire: micro-variants."""
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
variants = {
    "orig": "Payments fully down across all regions.",
    "no_period": "Payments fully down across all regions",
    "with_are": "Payments are fully down across all regions.",
    "checkout_subj": "Checkout fully down across all regions.",
    "payments_outage": "Payments outage across all regions.",
}
for label, state in variants.items():
    t0 = time.time()
    ans = c.decide(state, {"sev": SEV_Q})
    s = ans["sev"]
    print(f"{label}: {time.time()-t0:.1f}s choice={s['choice']} "
          f"high={s['probabilities']['high']:.3f} "
          f"low={s['probabilities']['low']:.3f}", flush=True)
