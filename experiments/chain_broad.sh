#!/usr/bin/env bash
# After the broad arm finishes: Oxford K=1 ladder, MCD hold-out distance ladder,
# decomposition, reports.  docs/gtabs-plan.md §13.
set -u
cd "$(dirname "$0")/.."
PY=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
B=bench_ckpt
log(){ echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S')] [broad] $*"; }
finished(){ python3 -c "import json,sys; d=json.load(open('experiments/results/train_broad.json')); sys.exit(0 if d.get('wall_s') else 1)" 2>/dev/null; }
log "waiting for broad"
until finished; do sleep 120; done
while ps -eo args | grep -q "[-]-wandb_name broad "; do sleep 30; done
log "broad finished -- Oxford ladder on card 0"
ARM=broad STEPS_K1="25 50 75 100 150 200 250 275" FORCE_GPU=0 BENCH_GPU_MEM_FRACTION=0.27 experiments/score_k1_c0off.sh 2>&1 | sed -u "s/^/  /"
log "MCD hold-out ladder for broad s100/s200/s275 on card 0"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 python3 experiments/mcd_distance_ladder.py \
  --ckpts broad_s100=$B/sd_broad_step100.pt broad_s200=$B/sd_broad_step200.pt broad_s275=$B/sd_broad_step275.pt \
  > experiments/logs/mcd_dist_broad.log 2>&1
log "decomposition + reports"
$PY experiments/drift_decomposition.py --runs gtctrl gtscale gtpaper teasup broad c0off --workers 8 > experiments/logs/drift_decomp_broad.log 2>&1
python3 experiments/gtabs_wandb_fill.py broad > /dev/null 2>&1
python3 experiments/gtabs_report.py --arms gtctrl gtscale gtpaper teasup broad > /dev/null 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python3 experiments/mcd_distance_score.py > /dev/null 2>&1
log "CHAIN_DONE broad"
