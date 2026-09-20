#!/usr/bin/env bash
# Launch a training run detached from whatever shell started it.
#
# ★ nohup IS NOT ENOUGH.  It suppresses SIGHUP, but a harness or terminal that
# tears down its process GROUP takes the job with it regardless -- which is
# exactly how the first A1/A2 pair died at step ~50 with no checkpoint.  tmux
# puts the run in its own session and its own process group, so nothing upstream
# can reach it, and it stays attachable for a live look.
#
#   experiments/launch_train.sh <name> [extra trainer args...]
#
#   experiments/launch_train.sh a1 --preset A1 --l2sp 1e-3
#   experiments/launch_train.sh a2 --preset A1 --lam_fresh 1.0 --l2sp 0
#
# Watch:   tmux attach -t train_a1        (detach with ctrl-b d)
# Status:  tmux ls
# Stop:    tmux kill-session -t train_a1
set -eu
cd "$(dirname "$0")/.."

NAME=$1; shift
GPU=${GPU:-0}
CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
STEPS=${STEPS:-1200}
# CPU budget per process.  72 cores here, two concurrent runs.
# ★ OMP_NUM_THREADS IS NOT THE BUDGET.  torch sizes its INTEROP pool from the
# core count independently of OMP, and load_and_preprocess_images adds a
# 16-thread pool -- measured, one trainer with OMP=32 ran 100 threads.  Two of
# those on 72 cores is the thrash docs/phase1-plan.md section 5.2 warns about.
# The trainer's _cap_threads() pins interop as well; this sets the rest.
THREADS=${THREADS:-16}

TRAIN_SCENES="kth_day_10 kth_night_01 kth_night_04 kth_night_05 \
tuhh_day_02 tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 tuhh_night_09"

SC=""
for s in $TRAIN_SCENES; do
  [ -f "labels/$s/index.json" ] || { echo "missing bank: labels/$s"; exit 1; }
  SC="$SC $s:data/mcd/$s/frames_10hz:labels/$s"
done

LOG=experiments/logs/train_$NAME.log
mkdir -p experiments/logs ckpt_train

# Resume automatically from the highest step checkpoint this name has written.
RESUME=""
LAST=$(ls -1 ckpt_train/${NAME}.step*.pt 2>/dev/null \
       | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
if [ -n "${LAST:-}" ]; then
  RESUME="--resume $LAST"
  echo "resuming from $LAST"
fi

tmux kill-session -t "train_$NAME" 2>/dev/null || true
tmux new-session -d -s "train_$NAME" \
  "CUDA_VISIBLE_DEVICES=$GPU LINGBOT_THREADS=$THREADS OMP_NUM_THREADS=$THREADS \
   MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
   python -u -m lingbot_map.train.trainer \
     --ckpt $CKPT --scene $SC \
     --steps $STEPS --pool 10 --S 48 --K 28 \
     --lr 1e-5 --warmup 50 --wd 0.05 --clip 1.0 \
     --probe_max 6 --probe_every 75 --save_every 50 \
     --gt_calib data/mcd/calib/hhs_calib.yaml --gt_sensor d455b_color \
     --wandb 1 --wandb_name $NAME --wandb_group mcd10 --wandb_tags mcd10 $NAME \
     --save ckpt_train/${NAME}.pt --out experiments/results/train_$NAME.json \
     $RESUME $* 2>&1 | tee -a $LOG"

sleep 3
tmux ls | sed 's/^/  /'
echo "log: $LOG"
