#!/usr/bin/env bash
# Start the oxford_long re-anchor rescue test as soon as the v7f cell frees card 1.
set -u
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [queue] $*"; }
log "waiting for spine_oxlong_v7f to finish"
while ps -eo args | grep -q '[r]un.py --config configs/spine_oxlong_v7f'; do sleep 180; done
log "v7f done -- evaluating it, then starting the re-anchor sweep"
GPU=1 ./benchmark/bench.sh evaluate configs/spine_oxlong_v7f.yaml 2>&1 | grep -aE "Total success|Total failed|rror"
GPU=1 THREADS=8 ./benchmark/bench.sh run configs/ra_oxlong.yaml 2>&1 \
  | grep -aE "Combination \(|seams|Successful:|Total failed|already complete|rror|Traceback"
GPU=1 ./benchmark/bench.sh evaluate configs/ra_oxlong.yaml 2>&1 | grep -aE "Total success|Total failed|rror"
log "RA_OXLONG_DONE"
