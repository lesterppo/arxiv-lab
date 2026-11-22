#!/usr/bin/env python3
"""Python-vs-Rust parity sweep for the self-consistency Rust port.

Four case kinds, all compared exactly (bit-for-bit where the value
is a pure function of libm routines; parsed-float equality for
JSON experiment results, which round-trips through repr):
- ppf:  normal_ppf over 5,000 random uniforms + edge values
        (Rust `ppf-batch` prints repr + u64 bits; bits must match
        CPython's exactly — this is the bit-exactness claim).
- cdf:  normal_cdf over 5,000 random points + edge values, same
        bit-exact comparison (exercises the libm `erf` FFI).
- eproc: 200 random +/-1 increment sequences (lengths 1..64) plus
         all-+1 / all--1: full wealth paths compared exactly, and
         crossing steps compared exactly.
- exps:  the full simulation experiments of the Python CPU test
         (E1, E2 N=64/N=256, E5 validity + power, E6) plus ESS
         rows, run on both sides with identical seeds and compared
         field-for-field against ../results.json values.

Exits non-zero on any mismatch. Builds the Rust binary first if
it is missing (CARGO_TARGET_DIR keeps target/ out of the session
dir).
"""
import json
import os
import random
import struct
import subprocess
import sys

RUST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_DIR = os.path.dirname(RUST_DIR)
sys.path.insert(0, os.path.join(SESSION_DIR, "..", "..", "src"))

from arxiv_lab.agent.selfconsistency import (  # noqa: E402
    LAMBDA_GRID,
    SplitMix64,
    betting_eprocess,
    crossing_step,
    run_prompt_level_experiment,
    run_single_unit_experiment,
    normal_cdf,
    normal_ppf,
    simulate_unit,
    tau_sigma_from_rho,
)

TARGET_DIR = os.environ.get("CARGO_TARGET_DIR", "/tmp/selfconsistency-rs-target")
RS_BIN = os.environ.get(
    "SELFCONSISTENCY_RS_BIN", os.path.join(TARGET_DIR, "release", "selfconsistency"))


def ensure_binary():
    if os.path.exists(RS_BIN):
        return
    print("building Rust binary ...")
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = TARGET_DIR
    subprocess.run(["cargo", "build", "--release", "--offline"],
                   cwd=RUST_DIR, env=env, check=True)


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def run_rs(args, stdin_text=None):
    out = subprocess.run([RS_BIN] + args, input=stdin_text,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main():
    ensure_binary()
    mismatches = []
    counts = {}

    # ---- ppf bit-exactness sweep ----
    rng = random.Random(20261122)
    p_vals = [rng.random() for _ in range(5000)]
    p_vals += [1e-12, 1e-9, 1e-6, 0.02425, 0.024249999, 0.97575, 0.975750001,
               0.5, 0.999999, 1 - 1e-9, 1 - 1e-12, 0.1, 0.9, 1e-300]
    stdin_text = "\n".join(repr(p) for p in p_vals) + "\n"
    lines = run_rs(["ppf-batch"], stdin_text).splitlines()
    assert len(lines) == len(p_vals), (len(lines), len(p_vals))
    n_bad = 0
    worst_ulp = 0
    for p, line in zip(p_vals, lines):
        py_v = normal_ppf(p)
        rs_repr, rs_bits = line.split(" bits=")
        rs_v = float(rs_repr)
        if bits(rs_v) != bits(py_v):
            n_bad += 1
            # ulp distance, for the report
            d = abs(struct.unpack("<q", struct.pack("<Q", bits(rs_v)))[0]
                    - struct.unpack("<q", struct.pack("<Q", bits(py_v)))[0])
            worst_ulp = max(worst_ulp, d)
            if n_bad <= 5:
                mismatches.append(f"ppf({p!r}): rust={rs_v!r} python={py_v!r} ulp={d}")
        if rs_bits != "0x%016x" % bits(rs_v):
            mismatches.append(f"ppf({p!r}): bits field inconsistent with repr")
    counts["ppf_cases"] = len(p_vals)
    counts["ppf_mismatches"] = n_bad
    counts["ppf_worst_ulp"] = worst_ulp
    print(f"ppf sweep: {len(p_vals)} cases, {n_bad} mismatches, worst ulp {worst_ulp}")

    # ---- cdf bit-exactness sweep ----
    x_vals = [rng.uniform(-8, 8) for _ in range(5000)]
    x_vals += [0.0, -0.0, 1.0, -1.0, 1.959963984540054, 8.0, -8.0, 0.5, 3.090232304709404]
    stdin_text = "\n".join(repr(x) for x in x_vals) + "\n"
    lines = run_rs(["cdf-batch"], stdin_text).splitlines()
    assert len(lines) == len(x_vals), (len(lines), len(x_vals))
    n_bad = 0
    worst_ulp = 0
    for x, line in zip(x_vals, lines):
        py_v = normal_cdf(x)
        rs_repr, rs_bits = line.split(" bits=")
        rs_v = float(rs_repr)
        if bits(rs_v) != bits(py_v):
            n_bad += 1
            d = abs(struct.unpack("<q", struct.pack("<Q", bits(rs_v)))[0]
                    - struct.unpack("<q", struct.pack("<Q", bits(py_v)))[0])
            worst_ulp = max(worst_ulp, d)
            if n_bad <= 5:
                mismatches.append(f"cdf({x!r}): rust={rs_v!r} python={py_v!r} ulp={d}")
    counts["cdf_cases"] = len(x_vals)
    counts["cdf_mismatches"] = n_bad
    counts["cdf_worst_ulp"] = worst_ulp
    print(f"cdf sweep: {len(x_vals)} cases, {n_bad} mismatches, worst ulp {worst_ulp}")

    # ---- e-process wealth paths on fixed increment sequences ----
    seqs = []
    for _ in range(200):
        n = rng.randint(1, 64)
        seqs.append([1.0 if rng.random() < 0.5 else -1.0 for _ in range(n)])
    seqs.append([1.0] * 64)
    seqs.append([-1.0] * 64)
    seqs.append([1.0] * 200)
    n_bad = 0
    for incs in seqs:
        py_path = betting_eprocess(incs)
        rs_path = json.loads(run_rs(["eprocess", ",".join(repr(v) for v in incs)]))
        if len(rs_path) != len(py_path):
            n_bad += 1
            mismatches.append(f"eprocess len: {incs[:5]}... rust {len(rs_path)} python {len(py_path)}")
            continue
        for i, (a, b) in enumerate(zip(rs_path, py_path)):
            if bits(a) != bits(b):
                n_bad += 1
                mismatches.append(
                    f"eprocess step {i} {incs[:5]}...: rust={a!r} python={b!r}")
                break
        py_cross = crossing_step(incs, eps=0.05)
        rs_cross = run_rs(["crossing", "0.05", ",".join(repr(v) for v in incs)])
        rs_cross = None if rs_cross == "null" else int(rs_cross)
        if rs_cross != py_cross:
            n_bad += 1
            mismatches.append(f"crossing {incs[:5]}...: rust={rs_cross} python={py_cross}")
    counts["eprocess_cases"] = len(seqs)
    counts["eprocess_mismatches"] = n_bad
    print(f"eprocess sweep: {len(seqs)} sequences, {n_bad} mismatches")

    # ---- full simulation experiments, identical seeds ----
    with open(os.path.join(SESSION_DIR, "results.json")) as fh:
        expected = json.load(fh)
    exp_cases = [
        ("run-single", [7, -0.15, 0.0, 64, 2000], expected["E1_rho0_control"]),
        ("run-single", [11, -0.15, 0.48, 64, 2000], expected["E2_aime_regime_N64"]),
        ("run-single", [11, -0.15, 0.48, 256, 2000], expected["E2_aime_regime_N256"]),
        ("run-prompt", [31, -0.15, 0.48, 256, 2000], expected["E5_prompt_level_validity"]),
        ("run-prompt", [37, 0.4, 0.48, 256, 1000], expected["E5_power_prompt_level"]),
        ("run-single", [37, 0.4, 0.48, 256, 1000], expected["E5_power_single_unit"]),
        ("run-single", [41, 0.96, 0.435, 64, 2000], expected["E6_gsm8k_regime"]),
    ]
    n_bad = 0
    for kind, params, exp in exp_cases:
        got = json.loads(run_rs([kind] + [repr(p) for p in params]))
        for k, v in exp.items():
            if got.get(k) != v:
                n_bad += 1
                mismatches.append(f"{kind} {params} field {k}: rust={got.get(k)!r} python={v!r}")
    counts["experiment_cases"] = len(exp_cases)
    counts["experiment_mismatches"] = n_bad
    print(f"experiment sweep: {len(exp_cases)} cases, {n_bad} mismatches")

    # ---- ESS rows (E4), identical seed ----
    rs_rows = json.loads(run_rs(["ess-rows", "23", "0.48", "4000", "256"]))
    py_rows = expected["E4_ess_saturation"]
    n_bad = 0
    if len(rs_rows) != len(py_rows):
        n_bad += 1
        mismatches.append("ess-rows length mismatch")
    else:
        for a, b in zip(rs_rows, py_rows):
            for k in b:
                if a[k] != b[k]:
                    n_bad += 1
                    mismatches.append(f"ess-rows n={b['n']} {k}: rust={a[k]!r} python={b[k]!r}")
    counts["ess_rows_cases"] = len(py_rows)
    counts["ess_rows_mismatches"] = n_bad
    print(f"ess-rows sweep: {len(py_rows)} rows, {n_bad} mismatches")

    # ---- extra random experiments (fresh seeds, both sides live) ----
    n_bad = 0
    n_extra = 0
    for seed in [101, 202, 303]:
        for theta, rho, n_roll, n_units in [(-0.3, 0.3, 32, 800),
                                            (0.2, 0.6, 128, 800),
                                            (-0.15, 0.9, 64, 800)]:
            py = run_single_unit_experiment(seed, theta, rho, n_roll, n_units)
            got = json.loads(run_rs(["run-single", str(seed), repr(theta),
                                     repr(rho), str(n_roll), str(n_units)]))
            n_extra += 1
            for k, v in py.items():
                if got.get(k) != v:
                    n_bad += 1
                    mismatches.append(
                        f"extra run-single seed={seed} theta={theta} rho={rho} "
                        f"field {k}: rust={got.get(k)!r} python={v!r}")
        for theta, rho in [(-0.15, 0.7), (0.5, 0.2)]:
            py = run_prompt_level_experiment(seed, theta, rho, 128, 500)
            got = json.loads(run_rs(["run-prompt", str(seed), repr(theta),
                                     repr(rho), "128", "500"]))
            n_extra += 1
            for k, v in py.items():
                if got.get(k) != v:
                    n_bad += 1
                    mismatches.append(
                        f"extra run-prompt seed={seed} theta={theta} rho={rho} "
                        f"field {k}: rust={got.get(k)!r} python={v!r}")
    counts["extra_experiment_cases"] = n_extra
    counts["extra_experiment_mismatches"] = n_bad
    print(f"extra experiments: {n_extra} cases, {n_bad} mismatches")

    total_mismatches = (counts["ppf_mismatches"] + counts["cdf_mismatches"]
                        + counts["eprocess_mismatches"]
                        + counts["experiment_mismatches"]
                        + counts["ess_rows_mismatches"]
                        + counts["extra_experiment_mismatches"])
    counts["total_mismatches"] = total_mismatches
    total_cases = (counts["ppf_cases"] + counts["cdf_cases"]
                   + counts["eprocess_cases"] + counts["experiment_cases"]
                   + counts["ess_rows_cases"] + counts["extra_experiment_cases"])
    counts["total_cases"] = total_cases
    print(f"PARITY total: {total_cases} cases, {total_mismatches} mismatches")
    for m in mismatches[:20]:
        print("MISMATCH:", m)
    with open("/tmp/selfconsistency-parity.json", "w") as fh:
        json.dump(counts, fh, indent=2)
    sys.exit(1 if total_mismatches else 0)


if __name__ == "__main__":
    main()
