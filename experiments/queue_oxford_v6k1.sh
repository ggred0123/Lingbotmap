#!/usr/bin/env bash
# oxford (stride 12) K=1 for v6b / v6c / v6d -- taking over from the kaist-bispl-ym
# run that died at 01:17 with v6c at 6/10.  gpu0 takes v6b, gpu1 takes v6d then
# v6c's remaining four.  Per-(scene,method) .complete.json means neither half
# redoes finished work and a kill costs only the scene in flight.
set -u
cd "$(dirname "$0")/../benchmark"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
F="Combination \(|Scenes to process|Scene \(|Successful:|Total failed|already complete|rror|Traceback"

GPU=0 THREADS=8 ./bench.sh run configs/oxford_v6k1_a.yaml 2>&1 | grep -aE "$F" | sed -u 's/^/[gpu0] /' & A=$!
GPU=1 THREADS=8 ./bench.sh run configs/oxford_v6k1_b.yaml 2>&1 | grep -aE "$F" | sed -u 's/^/[gpu1] /' & B=$!
wait $A; log "gpu0 done"
wait $B; log "gpu1 done"

log "=== EVALUATE base_k1 + v6b/v6c/v6d ==="
GPU=1 ./bench.sh evaluate configs/oxford_v6k1_all.yaml 2>&1 \
  | grep -aE "Total success|Total failed|rror|Traceback"
log "OXFORD_V6K1_DONE"
