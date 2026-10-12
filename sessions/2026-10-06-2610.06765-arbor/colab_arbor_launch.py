#!/usr/bin/env python3
"""Launcher for the ARBOR Colab arm with SIGKILL-tolerant load retries.

Context (session 2026-10-06-2610.06765-arbor): the Qwen3-8B
from_pretrained loader is OOM-killed (SIGKILL — uncatchable in-process)
by Colab's ~12GB system-RAM cgroup on ~4 of 5 attempts, even though the
same load succeeded once the same day. Each attempt therefore runs in a
child process; this parent retries up to MAX_ATTEMPTS times with page-cache
drops between attempts. Success = child exit 0 AND the results file exists.

Usage on the Colab VM (files already uploaded to /content/):
    python3 /content/colab_arbor_launch.py
Runs detached via: colab.py exec_detach -s <session> -f colab_arbor_launch.py
"""
import os
import subprocess
import sys
import time

SCRIPT = "/content/colab_arbor.py"
OUT = "/content/arbor_results_arbor.json"
LOG = "/content/arbor_launch.log"
MAX_ATTEMPTS = 5
RETRY_PAUSE_S = 20


def drop_caches():
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass


def attempt_ok():
    return os.path.exists(OUT) and os.path.getsize(OUT) > 0


def main():
    for i in range(1, MAX_ATTEMPTS + 1):
        print(f"[launcher] attempt {i}/{MAX_ATTEMPTS}", flush=True)
        drop_caches()
        p = subprocess.run(
            [sys.executable, SCRIPT, "--arm", "arbor", "--out", OUT])
        if p.returncode == 0 and attempt_ok():
            print(f"[launcher] SUCCESS on attempt {i}", flush=True)
            return 0
        print(f"[launcher] attempt {i} failed (rc={p.returncode}), "
              f"pausing {RETRY_PAUSE_S}s", flush=True)
        time.sleep(RETRY_PAUSE_S)
    print("[launcher] ALL ATTEMPTS FAILED", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
