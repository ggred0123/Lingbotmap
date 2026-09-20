#!/usr/bin/env bash
# One experiment across BOTH GPUs (DDP).  Each rank walks its own half of the
# scenes, so an optimizer step averages WORLD windows from WORLD different
# sequences -- which halves the scene-to-scene variance that makes the
# single-GPU prequential curve oscillate.
#
#   experiments/launch_ddp.sh a2 --preset A1 --lam_fresh 1.0 --l2sp 0
set -eu
cd "$(dirname "$0")/.."
NAME=$1; shift
CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
STEPS=${STEPS:-1200}
THREADS=${THREADS:-16}
NPROC=${NPROC:-2}

TRAIN_SCENES="kth_day_10 kth_night_01 kth_night_04 kth_night_05 \
tuhh_day_02 tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 tuhh_night_09"
SC=""
for s in $TRAIN_SCENES; do
  [ -f "labels/$s/index.json" ] || { echo "missing bank: labels/$s"; exit 1; }
  SC="$SC $s:data/mcd/$s/frames_10hz:labels/$s"
done

RESUME=""
LAST=$(ls -1 ckpt_train/${NAME}.step*.pt 2>/dev/null \
       | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
[ -n "${LAST:-}" ] && { RESUME="--resume $LAST"; echo "resuming from $LAST"; }

LOG=experiments/logs/train_$NAME.log
mkdir -p experiments/logs ckpt_train
tmux kill-session -t "train_$NAME" 2>/dev/null || true
tmux new-session -d -s "train_$NAME" \
  "PYTHONUNBUFFERED=1 LINGBOT_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
   OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
   torchrun --nproc_per_node=$NPROC --master_port=\$((29500 + RANDOM % 1000)) \
     -m lingbot_map.train.trainer \
     --ckpt $CKPT --scene $SC \
     --steps $STEPS --pool 10 --S 48 --K 28 \
     --lr 1e-5 --warmup 50 --wd 0.05 --clip 1.0 \
     --probe_max 6 --probe_every 75 --save_every 50 \
     --gt_calib data/mcd/calib/hhs_calib.yaml --gt_sensor d455b_color \
     --wandb 1 --wandb_name $NAME --wandb_group mcd10 --wandb_tags mcd10 ddp $NAME \
     --save ckpt_train/${NAME}.pt --out experiments/results/train_$NAME.json \
     $RESUME $* 2>&1 | tee -a $LOG"
sleep 3; tmux ls | sed 's/^/  /'; echo "log: $LOG"
