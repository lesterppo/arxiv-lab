"""Build the self-contained Colab driver from elite_common.py + body.

Usage: python3 build_driver.py   (run in this session dir)
Output: colab_elitereplay_9b.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = '''# colab_elitereplay_9b.py — EliteReplay (2610.22290) training track, Colab T4.
# SELF-CONTAINED: the bootstrap block below installs deps itself (run with python3).
# Writes metrics to /content (elitereplay_metrics.csv, elitereplay_results.json,
# elitereplay_archive.json, elitereplay_status.txt).

# ================= bootstrap: pip installs (Colab T4) =================
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/huggingface/transformers.git", "peft", "bitsandbytes", "trl", "hf_transfer",
         "datasets", "accelerate", "huggingface_hub"], check=True)
# Colab ships torchao 0.10; peft needs >0.16 -> remove it right after installs.
# hf_transfer env: set in driver body before model load for fast weights download.
_sp.run([_sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
del _sp, _sys
# ================= end bootstrap =================

'''

with open(os.path.join(HERE, "elite_common.py")) as f:
    common = f.read()
with open(os.path.join(HERE, "colab_driver_body.py")) as f:
    body = f.read()

# strip the body module docstring's duplicate imports guard: body is appended as-is
out = HEADER + "\n# ================= elite_common (embedded, tested) =================\n" + common \
    + "\n# ================= driver body =================\n" + body
with open(os.path.join(HERE, "colab_elitereplay_9b.py"), "w") as f:
    f.write(out)
print("wrote colab_elitereplay_9b.py", len(out), "bytes")
