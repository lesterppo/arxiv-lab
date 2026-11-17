#!/usr/bin/env python3
"""Run the Python-vs-Rust parity sweep and report mismatches.

For each case, runs the Python oracle (tests/parity_gen.py) and the
Rust binary (`matrix-steps parity ...`), parses both JSON outputs and
compares. Discrete outputs (ids, texts, counts, decisions) must match
exactly; float vectors (embeddings, matrix checksums) compare with a
tolerance, since numpy and Rust sum in different orders. Exits non-zero
on any mismatch.
"""
import json
import subprocess
import sys

RUST_DIR = "/home/hatch/workspace/arxiv-lab-build/sessions/2026-11-17-2611.20230-matrix-steps/rust"
PY = ["python3", f"{RUST_DIR}/tests/parity_gen.py"]
RS = [f"{RUST_DIR}/target/release/matrix-steps", "parity"]

CASES = [
    ("rng", ["20261117", "20"], "rng seed=20261117 x20 draws"),
    ("rng", ["0", "20"], "rng seed=0 x20 draws"),
    ("rng", ["1", "20"], "rng seed=1 x20 draws"),
    ("rng", ["42", "20"], "rng seed=42 x20 draws"),
    ("rng", ["9223372036854775807", "20"], "rng seed=2^63-1 x20 draws"),
    ("embed", ["change the tyre on the bicycle"], "embed short text"),
    ("embed", ["w0001 w0002 w0003 w0004 w0005 w0006 w0007 w0008 w0009 w0010"],
     "embed 10-word text"),
    ("embed", ["the a an on in at"], "embed stopwords-only"),
    ("sign", ["proc_0001", "64"], "sign proc_0001 x64"),
    ("sign", ["proc_0042", "1024"], "sign proc_0042 x1024"),
    ("sign", ["other-id", "8"], "sign other-id x8"),
    ("matrix", ["20261117", "10", "10", "0.25"], "matrix 10 procs 10 passes"),
    ("matrix", ["7", "20", "10", "0.25"], "matrix 20 procs 10 passes"),
    ("matrix", ["99", "10", "1", "1.0"], "matrix 10 procs 1 pass beta=1"),
    ("consolidate", ["20261117", "12"], "consolidate seed=20261117 n=12"),
    ("consolidate", ["5", "4"], "consolidate below minimums n=4"),
    ("consolidate", ["5", "30"], "consolidate seed=5 n=30"),
    ("promotion", ["101", "0.5", "0.5", "200"], "promotion identical 0.5/0.5"),
    ("promotion", ["101", "0.3", "0.6", "200"], "promotion better 0.3->0.6"),
    ("promotion", ["7", "0.4", "0.4", "50"], "promotion identical 0.4/0.4 x50"),
    ("hidden", ["7"], "hidden criterion seed=7"),
    ("hidden", ["123"], "hidden criterion seed=123"),
    ("revisit", ["11", "100", "4"], "revisit 100 situations x4"),
    ("revisit", ["3", "50", "4"], "revisit 50 situations x4"),
    ("demotion", ["13"], "demotion seed=13"),
    ("demotion", ["99"], "demotion seed=99"),
]

FLOAT_TOL = {
    "vec": 1e-12,
    "norm": 1e-12,
    "matrix_checksum": 1e-2,
    "ceiling": 1e-12,
}


def deep_diff(a, b, path="$"):
    """Return a list of (path, a, b) differences (first few)."""
    diffs = []
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            tol = 0.0
            for key, t in FLOAT_TOL.items():
                if key in path:
                    tol = t
            if abs(float(a) - float(b)) <= tol:
                return diffs
            diffs.append((path, a, b))
            return diffs
    if type(a) is not type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == b:
            return diffs
        diffs.append((path, a, b))
        return diffs
    if isinstance(a, dict):
        if set(a) != set(b):
            diffs.append((path + " keys", sorted(a), sorted(b)))
        for k in a:
            if k in b:
                diffs.extend(deep_diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((path + " len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(deep_diff(x, y, f"{path}[{i}]"))
            if len(diffs) >= 5:
                return diffs
    else:
        if a != b:
            diffs.append((path, a, b))
    return diffs


def main():
    mismatches = 0
    for kind, args, label in CASES:
        py = subprocess.run(PY + [kind] + args, capture_output=True, text=True)
        rs = subprocess.run(RS + [kind] + args, capture_output=True, text=True)
        if py.returncode != 0:
            print(f"FAIL {label}: python error: {py.stderr[:300]}")
            mismatches += 1
            continue
        if rs.returncode != 0:
            print(f"FAIL {label}: rust error: {rs.stderr[:300]}")
            mismatches += 1
            continue
        try:
            a = json.loads(py.stdout)
            b = json.loads(rs.stdout)
        except json.JSONDecodeError as e:
            print(f"FAIL {label}: JSON parse error: {e}")
            print("  py:", py.stdout[:200])
            print("  rs:", rs.stdout[:200])
            mismatches += 1
            continue
        diffs = deep_diff(a, b)
        if diffs:
            print(f"FAIL {label}: {len(diffs)} difference(s)")
            for p, x, y in diffs[:5]:
                print(f"  {p}: py={x!r} rs={y!r}")
            mismatches += 1
        else:
            print(f"ok   {label}")
    print(f"\nparity sweep: {len(CASES) - mismatches}/{len(CASES)} cases match, "
          f"{mismatches} mismatch(es)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
