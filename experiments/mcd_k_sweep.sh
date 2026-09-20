#!/usr/bin/env bash
# MCD kth_day_06 (held out of all training) at a given K, for several checkpoints.
#   experiments/mcd_k_sweep.sh 1 base a3s50 a3s300 a4s300 a2pcs300
set -u
cd "$(dirname "$0")/.."
K=$1; shift
BASE=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt
BC=bench_ckpt
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
for name in "$@"; do
  case "$name" in
    base) CK=$BASE ;;
    a3s*)   CK=$BC/sd_a3_step${name#a3s}.pt ;;
    a4s*)   CK=$BC/sd_a4_step${name#a4s}.pt ;;
    a2pcs*) CK=$BC/sd_a2pc_step${name#a2pcs}.pt ;;
    v5as*|v5bs*|v5cs*) CK=$BC/sd_${name:0:3}_step${name#v5?s}.pt ;;
    v6as*)  CK=$BC/sd_v6a_step${name#v6as}.pt ;;
    v6bs*|v6cs*|v6ds*) CK=$BC/sd_${name:0:3}_step${name#v6?s}.pt ;;
    *) log "unknown $name"; continue ;;
  esac
  [ -f "$CK" ] || { log "MISSING $CK"; continue; }
  OUT=experiments/results/mcd06_K${K}_${name}.json
  [ -f "$OUT" ] && { log "have $OUT"; continue; }
  log "=== $name  K=$K ==="
  python experiments/mcd_single_run.py --ckpt "$CK" \
    --frames data/mcd/kth_day_06/frames_10hz \
    --calib data/mcd/calib/hhs_calib.yaml --K "$K" --out "$OUT" 2>&1 | grep -av "it/s\]" | tail -6
done
log "SWEEP_DONE K=$K"
