#!/usr/bin/env bash
# MCD kth_day_06 over a ladder of sequence lengths at a GIVEN K.
#
# Generalises mcd_k1_short.sh (which hardcodes K=1, the one K where every
# checkpoint sits within 3% of base and nothing separates).  Output names carry
# the K so they never collide with the existing mcd06_K1_L* files.
#
# ★ NOT COMPARABLE ACROSS LIMITS.  mcd_single_run.py fits one global Sim(3) over
# whatever frames are loaded, so each limit has its own alignment.  Only the
# base-vs-checkpoint gap WITHIN a limit means anything.
# ★ L>=240 or the 12 windows collapse (edges=linspace(0,S,13), rows need b-a>=20).
#
#   experiments/mcd_short_sweep.sh <gpu> <K> <ckpt-name> <limit>...
set -u
cd "$(dirname "$0")/.."
GPU=$1; K=$2; NAME=$3; shift 3
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 LINGBOT_THREADS=8
BASE=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt
case "$NAME" in
  base) CK=$BASE ;;
  a3s*)   CK=bench_ckpt/sd_a3_step${NAME#a3s}.pt ;;
  a4s*)   CK=bench_ckpt/sd_a4_step${NAME#a4s}.pt ;;
  a2pcs*) CK=bench_ckpt/sd_a2pc_step${NAME#a2pcs}.pt ;;
  v5as*|v5bs*|v5cs*) CK=bench_ckpt/sd_${NAME:0:3}_step${NAME#v5?s}.pt ;;
  v6as*) CK=bench_ckpt/sd_v6a_step${NAME#v6as}.pt ;;
  *) echo "unknown $NAME"; exit 2 ;;
esac
[ -f "$CK" ] || { echo "MISSING $CK"; exit 2; }
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
for L in "$@"; do
  OUT=experiments/results/mcd06_K${K}_L${L}_${NAME}.json
  [ -f "$OUT" ] && { log "have $OUT"; continue; }
  log "=== $NAME  K=$K  limit=$L ==="
  python experiments/mcd_single_run.py --ckpt "$CK" \
    --frames data/mcd/kth_day_06/frames_10hz \
    --calib data/mcd/calib/hhs_calib.yaml --K "$K" --limit "$L" --out "$OUT" \
    2>&1 | grep -av "it/s\]" | tail -5
done
log "SHORT_DONE $NAME K=$K"
