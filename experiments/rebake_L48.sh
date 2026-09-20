#!/usr/bin/env bash
# Re-bake the MCD training scenes at L=48 instead of the L=240 every bank on disk
# was built with.
#
# WHY.  experiments/teacher_L_sweep.py scored the SAME target frames against GT
# with banks built at L = 240 / 96 / 48 / 28 / 12.  A run writes its i-th label
# with the teacher at depth 80+i, so L sets how deep the teacher gets before it
# restarts, and the label error follows it almost linearly:
#
#     Oxford stride-12, median label error vs GT (m)
#       L=240   5.232  1.896  1.902      <- what v5a..v6a all trained against
#       L= 96   1.709  0.541  0.972
#       L= 48   0.838  0.266  0.442
#       L= 28   0.525  0.239  0.333      <- unusable, see below
#
# ★ L=48 IS THE FLOOR, NOT THE OPTIMUM.  next_valid_window (trainer.py:168)
# requires the whole S-frame window to sit inside one run, so L < S = 48 yields
# zero trainable windows.  L=28 and L=12 are better labels that no window can
# read.  Going below 48 means changing --S, which changes the loss's pair count
# and breaks comparability with every run so far.
set -u
cd "$(dirname "$0")/.."
G=$1; shift
CK=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
for S in "$@"; do
  OUT=labels/${S}_L48
  [ -f "$OUT/index.json" ] && { log "have $OUT"; continue; }
  N=$(python3 -c "import json;print(json.load(open('labels/$S/index.json'))['n_frames_available'])")
  USE=$(( N < 8000 ? N : 8000 ))
  log "BUILD ${S}_L48  (span [0,$USE))"
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 python -m lingbot_map.train.label_bank \
    --ckpt "$CK" --frames "data/mcd/$S/frames_10hz" --out "$OUT" \
    --span 0 "$USE" --burn_in 72 --L 48 \
    >> "experiments/logs/rebake_${S}_L48.log" 2>&1
  [ -f "$OUT/index.json" ] && log "DONE  ${S}_L48 ($(du -sh $OUT | cut -f1))" || log "FAIL  ${S}_L48"
done
log "REBAKE_DONE gpu$G"
