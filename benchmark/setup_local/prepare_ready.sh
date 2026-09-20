#!/usr/bin/env bash
# Prepare (raw -> BSS) every dataset whose raw data is already on disk.
# CPU/IO only -- no GPU, so it is safe to run while a trainer holds the GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in oxford seven_scenes vbr kitti; do
  echo "=================== prepare $cfg ==================="
  THREADS=8 ./bench.sh prepare "configs/$cfg.yaml"
done
echo "=================== all prepares done ==================="
