#!/usr/bin/env bash
# The gtabs cell end to end, docs/gtabs-plan.md §6-§7, on two cards:
#
#   card 0: gtctrl (started by hand) -> score gtctrl -> gtpaper -> score gtpaper
#   card 1: gtscale (started by hand once lambda is known) -> score gtscale
#
# This script only does the parts that WAIT: it never launches a treated arm
# (their lambda comes from the probes), it scores an arm when its trainer has
# finished AND its card is free, then refreshes the decomposition and report.
#
#   setsid nohup experiments/chain_gtabs.sh > experiments/logs/chain_gtabs.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
GRID=${GRID:-"25 50 75 100 150 200 250 275"}
PY=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
log(){ echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S')] [gtabs] $*"; }

finished(){ python3 -c "import json,sys; d=json.load(open('experiments/results/train_$1.json')); sys.exit(0 if d.get('wall_s') else 1)" 2>/dev/null; }
trainer_on_gpu(){ # any trainer whose CUDA_VISIBLE_DEVICES is $1 (the run file exports it)
  for f in experiments/logs/.run_*.sh; do
    n=$(basename "$f" .sh); n=${n#.run_}
    grep -q "CUDA_VISIBLE_DEVICES=$1\$" "$f" 2>/dev/null || continue
    ps -eo args | grep -q "[-]-wandb_name $n " && return 0
  done
  return 1
}
wait_arm(){ # $1 arm $2 gpu -- until finished and the card is free
  log "waiting for $1 (card $2)"
  while :; do
    if finished "$1" && ! trainer_on_gpu "$2"; then log "$1 finished, card $2 free"; return 0; fi
    sleep 120
  done
}
score(){ # $1 arm $2 gpu
  log "scoring $1 on card $2"
  ARM=$1 STEPS_K1="$GRID" FORCE_GPU=$2 BENCH_GPU_MEM_FRACTION=0.27 experiments/score_k1_c0off.sh 2>&1 | sed -u "s/^/  /"
  log "decomposition + report"
  $PY experiments/drift_decomposition.py --runs gtctrl gtscale gtpaper --workers 8 > experiments/logs/drift_decomp_gtabs.log 2>&1
  python3 experiments/gtabs_report.py > /dev/null 2>&1 && log "report refreshed (docs/gtabs-result.md)"
}

case "${1:-all}" in
  ctrl)  wait_arm gtctrl 0; score gtctrl 0 ;;
  scale) wait_arm gtscale 1; score gtscale 1 ;;
  paper) wait_arm gtpaper 0; score gtpaper 0 ;;
  all)
    wait_arm gtctrl 0; score gtctrl 0
    # gtpaper is launched by hand after this point (card 0 is free now)
    wait_arm gtscale 1; score gtscale 1
    ;;
esac
log "CHAIN_DONE ${1:-all}"
