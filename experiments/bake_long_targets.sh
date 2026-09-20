#!/usr/bin/env bash
# Corpus-wide long-target bake: L=96 / stride=48 / POSE ONLY, then stitch.
# docs/long_supervision_design.md section 6, docs/long_supervision_plan.md Stage 1b.
#
# ★ POSE ONLY.  L_long reads rot/dir/scale and never depth, and stride=48 labels
# every raw frame twice -- with depth this bake would DOUBLE labels/ (+426 GB) to
# feed one term.  Pose-only it is ~3.7 kB per run: the whole corpus fits in tens
# of megabytes.
#
# ★ ONLY LONG SEQUENCES.  A Delta=319 pair needs 80 + 319 + 48 = 447 frames, so
# dl3dv (324-337) and replica (400) cannot contribute one and are skipped.
#
#   experiments/bake_long_targets.sh 0 mcd slowtv            # gpu 0, two legs
#   experiments/bake_long_targets.sh 1 scannet unrealstereo4k paralleldomain4d dynamicreplica
#   STITCH_ONLY=1 experiments/bake_long_targets.sh 0 all     # chain step only
set -u
cd "$(dirname "$0")/.."
B=/NHNHOME/WORKSPACE/26msit001_A
G=$1; shift
export CUDA_VISIBLE_DEVICES=$G
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export L=96 STRIDE=48 POSE_ONLY=1 SUFFIX=_L96s48 BURN_IN=${BURN_IN:-72}
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

if [ "${STITCH_ONLY:-0}" != "1" ]; then
  for job in "$@"; do
    log "BAKE BEGIN $job"
    case "$job" in
      mcd)              bash experiments/build_banks.sh ;;
      slowtv)           DATASET=slowtv ROOT=data/slow_tv SUBDIR=frames_10hz N=0 bash experiments/build_banks_generic.sh ;;
      scannet)          DATASET=scannet ROOT=$B/V-LAB/Datasets/scannet/scannet/train SUBDIR=color N=30 bash experiments/build_banks_generic.sh ;;
      unrealstereo4k)   DATASET=unrealstereo4k ROOT=$B/jinhyeok/dataset/unrealstereo4k SUBDIR=images N=0 bash experiments/build_banks_generic.sh ;;
      paralleldomain4d) DATASET=paralleldomain4d ROOT=$B/jinhyeok/dataset/paralleldomain4d SUBDIR=images N=50 bash experiments/build_banks_generic.sh ;;
      dynamicreplica)   DATASET=dynamicreplica ROOT=$B/jinhyeok/dataset/dynamicreplica SUBDIR=images N=75 bash experiments/build_banks_generic.sh ;;
      *) log "unknown leg $job" ;;
    esac
    log "BAKE DONE $job"
  done
fi

log "STITCH BEGIN"
n_ok=0; n_bad=0
for d in labels/*_L96s48; do
  [ -f "$d/index.json" ] || continue
  out="${d%_L96s48}_long"
  [ -f "$out/index.json" ] && continue
  if python experiments/stitch_bank.py "$d" --out "$out" > "experiments/logs/stitch_$(basename "$d").log" 2>&1; then
    n_ok=$(( n_ok + 1 ))
  else
    n_bad=$(( n_bad + 1 )); log "STITCH-FAIL $(basename "$d")"; rm -rf "$out"
  fi
done
log "STITCH DONE  ok=$n_ok fail=$n_bad"
log "BAKE_LONG_COMPLETE"
