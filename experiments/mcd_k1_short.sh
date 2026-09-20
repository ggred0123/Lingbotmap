#!/usr/bin/env bash
# MCD kth_day_06 at K=1 over a ladder of sequence lengths.
#
# The full-length K=1 runs could not separate any checkpoint (all within 3% of
# base) while the same weights differ by 16% at K=28.  Two candidate causes were
# confounded there: the 64-keyframe cache window spans only 6.4 s at K=1, and the
# 3D-RoPE frame axis is driven to 8894 positions.  Truncating the sequence moves
# the second without moving the first, so a length ladder separates them.
#
# ★ NOT COMPARABLE ACROSS LIMITS.  mcd_single_run.py fits one global Sim(3) over
# whatever frames are loaded, so each limit has its own alignment.  Only the
# base-vs-checkpoint gap WITHIN a limit means anything.
#
# ★ S>=240 or the 12 windows collapse (edges=linspace(0,S,13), rows need b-a>=20).
#
#   experiments/mcd_k1_short.sh <gpu> <ckpt-name> <limit>...
set -u
cd "$(dirname "$0")/.."
GPU=$1; NAME=$2; shift 2
BASE=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt
case "$NAME" in
  base) CK=$BASE ;;
  a3s*)   CK=bench_ckpt/sd_a3_step${NAME#a3s}.pt ;;
  a4s*)   CK=bench_ckpt/sd_a4_step${NAME#a4s}.pt ;;
  a2pcs*) CK=bench_ckpt/sd_a2pc_step${NAME#a2pcs}.pt ;;
  *) echo "unknown $NAME"; exit 2 ;;
esac
[ -f "$CK" ] || { echo "MISSING $CK"; exit 2; }
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
for L in "$@"; do
  OUT=experiments/results/mcd06_K1_L${L}_${NAME}.json
  [ -f "$OUT" ] && { log "have $OUT"; continue; }
  log "=== $NAME  K=1  limit=$L ==="
  python experiments/mcd_single_run.py --ckpt "$CK" \
    --frames data/mcd/kth_day_06/frames_10hz \
    --calib data/mcd/calib/hhs_calib.yaml --K 1 --limit "$L" --out "$OUT" \
    2>&1 | grep -av "it/s\]" | tail -5
done
log "SHORT_DONE $NAME"
