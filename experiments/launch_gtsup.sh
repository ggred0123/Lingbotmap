#!/usr/bin/env bash
# One arm of the GT-supervision cell.  Same loss, same policy, same scenes --
# the ONLY difference between the two arms is which bank the labels come from.
#
#   experiments/launch_gtsup.sh gtsup  _gt    # GT poses
#   experiments/launch_gtsup.sh teasup ""     # base-model pseudo labels
#
# ★ DDP OVER BOTH CARDS, ARMS RUN SEQUENTIALLY -- and not by preference.  The
# first attempt ran the arms concurrently, one card each, to halve the clock.
# That needs --pool 10 so each stream still gets 1250/10 = 125 advances (under
# DDP --pool is the TOTAL, so v6i's 20 was already 10 per rank).  The trainer
# refuses:
#
#     --pool 10 x P(K=1)=0.2500 = 2.5000 is not an integer, so the per-stream
#     assignment cannot realise --k_dist.  The smallest pool that matches this
#     mixture is 20.
#
# and --pool 20 on one rank halves the rollout depth instead -- "horizons [3840]
# exceed the reachable age 1950".  Rollout depth is the axis this project is
# about, so it is the one that is preserved: 2 ranks, --pool 20, 1250 steps,
# byte-for-byte v6i's shape.  The arms then sit on the same axes as v6i, s0off
# and s0on rather than only against each other.
set -eu
cd "$(dirname "$0")/.."
NAME=$1; SUF=${2:-}; shift 2 || true
CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
STEPS=${STEPS:-1250}
THREADS=${THREADS:-16}
NPROC=${NPROC:-2}
POOL=${POOL:-20}
PROBE_MAX=${PROBE_MAX:-6}
SAVE_EVERY=${SAVE_EVERY:-50}
# ★ GPU pins the card (CUDA_VISIBLE_DEVICES) so two 1-rank arms can share the
# node, one card each -- the gtabs cell (docs/gtabs-plan.md) runs gtctrl and
# gtscale side by side.  Empty = torchrun's default, i.e. card 0.  Only
# meaningful with NPROC=1; a 2-rank launch wants both cards visible.
GPU=${GPU:-}

TRAIN_SCENES="kth_day_10 kth_night_01 kth_night_04 kth_night_05 \
tuhh_day_02 tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 tuhh_night_09"
SC=""
for s in $TRAIN_SCENES; do
  [ -f "labels/${s}${SUF}/index.json" ] || { echo "missing bank: labels/${s}${SUF}"; exit 1; }
  SC="$SC $s:data/mcd/$s/frames_10hz:labels/${s}${SUF}:mcd"
done

RESUME=""
LAST=$(ls -1 ckpt_train/${NAME}.step*.pt 2>/dev/null \
       | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
[ -n "${LAST:-}" ] && { RESUME="--resume $LAST"; echo "resuming from $LAST"; }

LOG=experiments/logs/train_$NAME.log
mkdir -p experiments/logs ckpt_train
# ★ setsid, NOT tmux.  A login-session cleanup reaped the tmux SERVER here once
# and took a 540-step run with it; launch_clean.sh was moved off tmux for that
# reason and this script had been left behind.  The runner tees to $LOG, so
# nothing tmux was providing is lost.  The invocation goes to a file because the
# scene list does not fit on a command line.
RUN=experiments/logs/.run_$NAME.sh
cat > "$RUN" <<EOF
#!/usr/bin/env bash
export PYTHONUNBUFFERED=1 LINGBOT_THREADS=$THREADS OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export MALLOC_ARENA_MAX=2
$( [ -n "$GPU" ] && echo "export CUDA_VISIBLE_DEVICES=$GPU" )
cd "$(pwd)"
torchrun --nproc_per_node=$NPROC --master_port=\$((29500 + RANDOM % 1000)) \
     -m lingbot_map.train.trainer \
     --ckpt $CKPT --scene $SC \
     --steps $STEPS --pool $POOL --S 48 --K 28 \
     --preset A1PC --p_identity 0.35 --sampler unified --k_dist v5 \
     --lam_fresh 1.0 --fresh_mode walk --fresh_pool 1 \
     --horizons 320 960 1920 3840 \
     --dataset_weights mcd:100 --sampler_seed 0 \
     --lr 1e-5 --warmup 50 --wd 0.05 --clip 1.0 \
     --probe_max $PROBE_MAX --probe_every 75 --save_every $SAVE_EVERY \
     --gt_calib data/mcd/calib/hhs_calib.yaml --gt_sensor d455b_color \
     --wandb 1 --wandb_name $NAME --wandb_group gtsup --wandb_tags gtsup $NAME \
     --save ckpt_train/${NAME}.pt --out experiments/results/train_$NAME.json \
     $RESUME $* 2>&1 | tee -a $LOG
EOF
chmod +x "$RUN"
setsid nohup "$RUN" >/dev/null 2>&1 &
sleep 5
# Bracket, so the pattern cannot match this shell's own command line: pgrep -f
# has reported a dead trainer alive and pkill -f has killed the caller here.
if ps -eo args | grep -q "[-]-wandb_name $NAME"; then
  echo "started $NAME (detached)  log: $LOG"
else
  echo "FAILED to start $NAME -- see $LOG"; tail -5 "$LOG" 2>/dev/null; exit 1
fi
