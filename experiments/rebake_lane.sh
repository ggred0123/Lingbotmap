#!/usr/bin/env bash
# One GPU, two shards back to back.  The user took all four shards onto this
# node, so each card runs its pair in sequence rather than two bakes racing for
# one card's SMs.
#   experiments/rebake_lane.sh <gpu> <shard> <shard>
set -u
cd "$(dirname "$0")/.."
G=$1; shift
for S in "$@"; do
  echo "[$(date '+%m-%d %H:%M:%S')] === shard $S -> gpu$G ==="
  SHARD=$S NSHARD=4 GPU=$G experiments/rebake_all_L48.sh
done
echo "[$(date '+%m-%d %H:%M:%S')] LANE_DONE gpu$G ($*)"
