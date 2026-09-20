#!/usr/bin/env bash
# oxford_long at K=1 -- the matched low-K point the v6 bench was still missing.
#
# Split by METHOD across the two cards rather than by scene: base_k1 on gpu0,
# the distilled pair on gpu1.  K=1 over 3840 frames is ~12x the keyframe count
# of the 'auto' pass (K=12), so a single-card run is ~20 h and a split one ~10 h.
#
# Both halves share bench_ws/oxford_long.  run.py skips any (scene, method) that
# already carries .complete.json, so a kill costs at most the scene in flight and
# re-running this script resumes.
set -u
cd "$(dirname "$0")/../benchmark"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "=== RUN gpu0: base_k1 ==="
GPU=0 THREADS=8 ./bench.sh run configs/oxford_long_k1_a.yaml 2>&1 \
  | grep -aE "Combination \(|Scenes to process|Scene \(|Successful:|Total failed|already complete|rror|Traceback" \
  | sed -u 's/^/[gpu0] /' &
A=$!

log "=== RUN gpu1: sd_v6as600_auto + sd_v6as600_k1 ==="
GPU=1 THREADS=8 ./bench.sh run configs/oxford_long_k1_b.yaml 2>&1 \
  | grep -aE "Combination \(|Scenes to process|Scene \(|Successful:|Total failed|already complete|rror|Traceback" \
  | sed -u 's/^/[gpu1] /' &
B=$!

wait $A; log "gpu0 RUN done"
wait $B; log "gpu1 RUN done"

# One evaluate over the FULL four-method config, so the k1 pair is scored in the
# same table as the auto pair it is meant to be compared against.
log "=== EVALUATE (4 methods) ==="
GPU=1 ./bench.sh evaluate configs/v6bench_oxford_long.yaml 2>&1 \
  | grep -aE "Total success|Total failed|rror|Traceback"
log "OXFORD_K1_DONE"
