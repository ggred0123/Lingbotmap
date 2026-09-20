#!/usr/bin/env bash
# Teacher banks for the v6 mixture.  One card per leg; each leg is a corpus.
# Budgets (DRY=1) were checked first: 1002 runs / ~193 GB / ~233 min of teacher.
set -u
cd "$(dirname "$0")/.."
B=/NHNHOME/WORKSPACE/26msit001_A
G=$1; shift
export CUDA_VISIBLE_DEVICES=$G
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
for job in "$@"; do
  case "$job" in
    paralleldomain4d) DATASET=paralleldomain4d ROOT=$B/jinhyeok/dataset/paralleldomain4d SUBDIR=images N=50  experiments/build_banks_generic.sh ;;
    dl3dv)            DATASET=dl3dv            ROOT=$B/jinhyeok/dataset/dl3dv_wai        SUBDIR=images N=150 experiments/build_banks_generic.sh ;;
    dynamicreplica)   DATASET=dynamicreplica   ROOT=$B/jinhyeok/dataset/dynamicreplica   SUBDIR=images N=75  experiments/build_banks_generic.sh ;;
    scannet)          DATASET=scannet          ROOT=$B/V-LAB/Datasets/scannet/scannet/train SUBDIR=color N=30 experiments/build_banks_generic.sh ;;
    replica)          DATASET=replica          ROOT=$B/jinhyeok/dataset/replica_wai      SUBDIR=images N=0   experiments/build_banks_generic.sh ;;
    unrealstereo4k)   DATASET=unrealstereo4k   ROOT=$B/jinhyeok/dataset/unrealstereo4k   SUBDIR=images N=0   experiments/build_banks_generic.sh ;;
    *) log "unknown job $job" ;;
  esac
done
log "V6_BANKS_DONE gpu$G"
