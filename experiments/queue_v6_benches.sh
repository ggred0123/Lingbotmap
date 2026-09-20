#!/usr/bin/env bash
# Five held-out benchmarks for the v6 mixture, queued behind whatever is on the
# card now.  Split so the two long ones do not land on the same GPU: VBR is
# 84k frames and KITTI 50k, against ~6k for the other three combined.
#
# ★ NONE OF THESE FIVE IS IN THE TRAINING CORPUS.  KITTI and VKITTI2 were kept
# out of the v6 mixture for exactly this reason; ScanNet/Replica/DL3DV are in it
# but none of them is a benchmark here.
set -u
cd "$(dirname "$0")/../benchmark"
G=$1; shift
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
log "waiting for the card to clear"
while pgrep -f "chain_v5_kcurve[.]sh|oxford_v6_k" >/dev/null 2>&1; do sleep 30; done
log "clear -- starting $*"
for B in "$@"; do
  log "=== RUN  $B on gpu$G ==="
  GPU=$G THREADS=8 ./bench.sh run "configs/v6bench_${B}.yaml" 2>&1 \
    | grep -aE "Combination \(|Successful:|Total failed|rror|Traceback"
  log "=== EVAL $B ==="
  GPU=$G ./bench.sh evaluate "configs/v6bench_${B}.yaml" 2>&1 \
    | grep -aE "Total success|Total failed|rror|Traceback"
  log "BENCH_DONE $B"
done
log "V6BENCH_QUEUE_DONE gpu$G"
