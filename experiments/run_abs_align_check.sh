#!/usr/bin/env bash
# Alignment pre-check for L_abs -- docs/gtabs-plan.md §5-3-4: cos(g_new, g_GT)
# on drifted theta_0 windows, against the GT banks.  Four windows are enough to
# catch an implementation error (the plan's criterion: the scale terms must
# align better than A1PC's +0.09, abs_pos must be ~1 against the run-gauge ATE).
#   GPU=1 experiments/run_abs_align_check.sh
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}; MODE=${MODE:-paper}; FIT=${FIT:-prefix}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
V=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
SPECS=${SPECS:-"kth_day_10 512 1|tuhh_day_02 992 28"}
IFS='|' read -ra LIST <<< "$SPECS"
for spec in "${LIST[@]}"; do
  set -- $spec; sc=$1; t0=$2; K=$3
  o=experiments/results/termalign_abs_${MODE}_${FIT}_${sc}_t${t0}_K${K}.json
  [ -f "$o" ] && { echo "[align] have $o"; continue; }
  echo "[align] $(date +%H:%M) $sc t0=$t0 K=$K"
  while ps -eo args | grep -q "[t]erm_grad_probe.py\|[t]erm_align_probe.py"; do sleep 15; done   # one probe per card
  $V experiments/term_align_probe.py --scene $sc --bank labels/${sc}_gt --t0 $t0 --K $K \
     --abs_mode $MODE --abs_fit $FIT --out "$o" > experiments/logs/termalign_abs_${sc}_t${t0}_K${K}.log 2>&1
done
echo "[align] DONE"
