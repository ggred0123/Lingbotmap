#!/usr/bin/env bash
# Full 10-scene run of the keyframe-seam re-anchor on oxford_long, once the
# one-scene validation finishes.
set -u
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [rk] $*"; }
while ps -eo args | grep -q '[r]un.py --config configs/rk_oxlong'; do sleep 120; done
log "starting full rk_oxlong run"
GPU=1 THREADS=8 ./benchmark/bench.sh run configs/rk_oxlong.yaml 2>&1 \
  | grep -a --line-buffered -E "Combination \(|seams|Successful:|Total failed|rror|Traceback"
log "evaluate"
GPU=1 ./benchmark/bench.sh evaluate configs/rk_oxlong.yaml 2>&1 | grep -aE "Total success|Total failed|rror"
log "RK_OXLONG_DONE"
