#!/bin/bash
# Mid-run downloader for lora-reasoning-1114d (v2): snapshots /content result
# files every 10 min so a reclamation does not lose the run's work.
# Stops when colab_out_v4/train-run.json records a terminal outcome.
set -u
cd ~/workspace/arxiv-lab-build/sessions/2026-11-14-2611.17072-lora-reasoning
mkdir -p colab_out_v4/midrun
REG=~/workspace/goals/arxiv-daily-read-and-test/hidden_files/train-1114d-registry.json
for i in $(seq 1 30); do
  sleep 600
  for f in lora_reasoning_metrics.csv lora_reasoning_results.json traces.jsonl; do
    python3 ~/workspace/hermes-colab-cli/colab.py download -s lora-reasoning-1114d \
      --registry-file "$REG" "/content/$f" "colab_out_v4/midrun/$f" >/dev/null 2>&1
  done
  echo "[midrun-v4] snapshot $i at $(date -u +%H:%M:%S) UTC" >> colab_out_v4/midrun/log.txt
  if [ -f colab_out_v4/train-run.json ] && grep -qE '"outcome": "(DONE|DIED|RECLAIMED|TIMEOUT|PROVISION_FAILED)"' colab_out_v4/train-run.json; then
    echo "[midrun-v4] terminal outcome recorded, stopping" >> colab_out_v4/midrun/log.txt
    break
  fi
done
