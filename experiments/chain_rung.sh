#!/usr/bin/env bash
# Score one rung of the scene-count ladder (docs/coverage-corpus-plan.md §3)
# once its trainer has finished: Oxford K=1 at steps 150/200/250/275 (5 min each;
# teasup swung +0.03 -> +0.19 between steps 200 and 275, so one step is not a
# read-out), then the MCD hold-out distance ladder at the same steps (11 min
# each).  Both hold-outs are the same
# sequences for every rung, so the rungs are paired.
#
#   experiments/chain_rung.sh <arm> <gpu> [steps="150 200 250 275"]
set -u
cd "$(dirname "$0")/.."
ARM=$1; GPU=$2; STEPS=${3:-"150 200 250 275"}
log(){ echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S')] [$ARM] $*"; }
finished(){ python3 -c "import json,sys; d=json.load(open('experiments/results/train_$ARM.json')); sys.exit(0 if d.get('wall_s') else 1)" 2>/dev/null; }
log "waiting for $ARM"
until finished; do sleep 120; done
while ps -eo args | grep -q "[-]-wandb_name $ARM "; do sleep 30; done
log "$ARM finished -- Oxford K=1 (steps $STEPS) on card $GPU"
ARM=$ARM STEPS_K1="$STEPS" FORCE_GPU=$GPU BENCH_GPU_MEM_FRACTION=0.27 experiments/score_k1_c0off.sh 2>&1 | sed -u "s/^/  /"
CK=""; for s in $STEPS; do [ -f bench_ckpt/sd_${ARM}_step$s.pt ] && CK="$CK ${ARM}_s$s=bench_ckpt/sd_${ARM}_step$s.pt"; done
log "MCD hold-out ladder:$CK"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 python3 experiments/mcd_distance_ladder.py --ckpts $CK \
  > experiments/logs/mcd_dist_$ARM.log 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python3 experiments/mcd_distance_score.py > /dev/null 2>&1
log "CHAIN_DONE $ARM"
