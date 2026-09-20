#!/usr/bin/env bash
# Wait for the running K=3 Oxford chain to clear this GPU, then run K=1 and K=28
# on it.  K=1 = 1.71 m keyframe baseline, K=28 = 44.17 m (outside the training
# regime -- docs/self-distill-ver5.md reports it, never gates on it).
#
# The wait pattern matches the K=3 CONFIG names, not the chain script: this
# script runs chain_oxford_v5.sh itself, so waiting on the script name would
# make each GPU's queue block on the other's chain and serialise the two halves.
set -u
cd "$(dirname "$0")/.."
G=$1; shift
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
log "waiting for the K=3 chain (configs oxford_v5_g?.yaml) to finish"
while pgrep -f "oxford_v5_g[0-9][.]yaml" > /dev/null 2>&1; do sleep 20; done
log "K=3 chain clear -- starting $*"
for CFG in "$@"; do
  experiments/chain_oxford_v5.sh "$G" "$CFG"
done
log "OXFORD_K_QUEUE_DONE gpu$G"
