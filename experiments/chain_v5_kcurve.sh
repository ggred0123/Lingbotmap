#!/usr/bin/env bash
# MCD kth_day_06 K curve for the v5 cells, to sit alongside base / a3s300.
# Phase 1 is the decisive A-vs-B comparison; phase 2 adds C.
#   GPU is passed in; each leg is one card.
set -u
cd "$(dirname "$0")/.."
G=$1; shift
export CUDA_VISIBLE_DEVICES=$G
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 LINGBOT_THREADS=8
for spec in "$@"; do            # spec = name:K,K,K
  name=${spec%%:*}; ks=${spec#*:}
  for K in ${ks//,/ }; do experiments/mcd_k_sweep.sh "$K" "$name"; done
done
echo "[$(date '+%H:%M:%S')] KCURVE_DONE gpu$G"
