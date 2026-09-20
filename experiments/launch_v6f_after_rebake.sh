#!/usr/bin/env bash
# v6f = v6c's corpus and banks, with the SHALLOW END ADDED to the horizon grid.
#
# v6c supervised 320/960/1920/3840 only, so a K=1 student was never corrected
# below cache depth 320 -- the regime every <=320-frame benchmark (eth3d 38,
# 7-Scenes 200, neural_rgbd 234) and every K=1 evaluation actually runs in.
# v6d probed that end with 96/192/384/768 but DROPPED the long end to get there.
# v6f keeps both: eight horizons, dealt by h_deck as 3+3+3+3 shallow and
# 2+2+2+2 deep, i.e. 60% / 40% of steps.
#
# ★ fixed_horizon=1 IS NOT OPTIONAL HERE.  With redraw (v6c's setting) the step
# share follows rollout LENGTH, and h_deck's docstring measured what that does
# on v6a: a uniform draw realised 15.7/26.4/35.0/20.4%, and only 69 of 802
# correction steps landed below depth 320.  Fixing the horizon per stream makes
# the share exactly (streams at h)/pool, which is the whole point of adding the
# shallow horizons at all.
#
# Waits for the L=48 re-bake of all seven non-MCD corpora (shards 0-1 on
# kaist-bispl-ym, 2-3 here) before starting, because a run launched early would
# silently train the unfinished corpora against their L=240 banks.
set -u
cd "$(dirname "$0")/.."
NEED=${NEED:-331}
MAXWAIT=${MAXWAIT:-43200}          # 12 h
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

count(){ ls -d labels/{slowtv,dl3dv,scannet,replica,dynamicreplica,unrealstereo4k,paralleldomain4d}_*_L48/index.json 2>/dev/null | wc -l; }

log "waiting for $NEED non-MCD _L48 banks (now $(count))"
t=0; last=-1
while [ "$(count)" -lt "$NEED" ] && [ "$t" -lt "$MAXWAIT" ]; do
  n=$(count); [ "$n" != "$last" ] && { log "  $n/$NEED"; last=$n; }
  sleep 60; t=$(( t + 60 ))
done
n=$(count)
if [ "$n" -lt "$NEED" ]; then
  log "TIMEOUT at $n/$NEED after ${t}s -- NOT launching.  Re-run this script once the re-bake finishes."
  exit 1
fi
log "re-bake complete ($n banks) -- launching v6f"

CUDA_VISIBLE_DEVICES=0,1 NAME=v6f STEPS=1250 NPROC=2 THREADS=16 \
  BANK_SUFFIX=_L48 MCD_ONLY=0 \
  HORIZONS="96 192 384 768 320 960 1920 3840" \
  P_IDENTITY=0.35 FIXED_HORIZON=1 \
  experiments/launch_v6.sh
log "V6F_LAUNCHED"
