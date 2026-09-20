#!/usr/bin/env bash
# Why does descending this objective RAISE it?
#
# s0off is the cleanest case in the project: identity branch only, so every step
# takes a fresh anchor (raw_frame_age = 0) and the rollout depth is constant.
# Its training loss still went 0.0441 -> 0.2502 by step 250, on a stationary
# sample, with all five terms rising together and none traded off.  These runs
# split the three candidate causes, one flag at a time.
#
#   experiments/launch_descent.sh dlr  --lr 1e-6        # (a) step size
#   experiments/launch_descent.sh dmot --lam_motion 0   # (b) L_motion
#
# ★ THE BASELINE IS FREE.  Everything here is copied out of train_s0off.json's
# own meta -- the same 338 scene specs, the same pool/rank, the same batch, the
# same seed -- so s0off's steps 0-300 ARE the control and no third run is
# needed.  Change exactly one flag per arm or that stops being true.
set -eu
cd "$(dirname "$0")/.."
NAME=$1; shift
CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
STEPS=${STEPS:-300}
THREADS=${THREADS:-16}
NPROC=${NPROC:-2}

# KEEP filters the corpus by dataset.  experiments/corpus_order_audit.py found
# 65% of the sampling weight was never a video -- paralleldomain4d cycles 19
# views of one instant, unrealstereo4k alternates eyes, scannet is sorted
# lexicographically, dynamicreplica concatenates left then right -- and the
# banks were baked in that same order, so nothing downstream ever noticed.
# KEEP=mcd,slowtv,dl3dv,replica is the subset that really is video.
KEEP=${KEEP:-}
read -r WEIGHTS SC <<EOF
$(KEEP="$KEEP" python3 -c "
import json, os
m = json.load(open('experiments/results/train_s0off.json'))['meta']
keep = [k for k in os.environ.get('KEEP', '').split(',') if k]
W = {'mcd': 10, 'slowtv': 10, 'dl3dv': 10, 'scannet': 10, 'replica': 5,
     'dynamicreplica': 10, 'unrealstereo4k': 15, 'paralleldomain4d': 30}
sc = [s for s in m['scene'] if not keep or s.split(':')[-1] in keep]
ds = sorted({s.split(':')[-1] for s in sc})
print(','.join(f'{d}:{W[d]}' for d in ds), ' '.join(sc))
")
EOF
[ -n "$SC" ] || { echo "could not read s0off's scene spec"; exit 1; }
echo "corpus: $(echo $SC | wc -w) scenes   weights: $WEIGHTS"

LOG=experiments/logs/train_$NAME.log
RUN=experiments/logs/.run_$NAME.sh
mkdir -p experiments/logs ckpt_train

# * THE SCENE SPEC IS 338 ENTRIES, so it cannot be passed inline -- tmux answers
# "command too long".  Write the whole invocation to a file and run the file.
cat > "$RUN" <<EOF
#!/usr/bin/env bash
export PYTHONUNBUFFERED=1 LINGBOT_THREADS=$THREADS OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
cd "$(pwd)"
torchrun --nproc_per_node=$NPROC --master_port=\$((29500 + RANDOM % 1000)) \
  -m lingbot_map.train.trainer \
  --ckpt $CKPT --scene $SC \
  --steps $STEPS --pool 20 --S 48 --K 28 \
  --preset A1PC --p_identity 1.0 --sampler unified --k_dist v5 \
  --lam_fresh 1.0 --fresh_mode walk --fresh_pool 1 \
  --horizons 320 960 1920 3840 \
  --dataset_weights $WEIGHTS \\
  --sampler_seed 0 \
  --lr 1e-5 --warmup 50 --wd 0.05 --clip 1.0 \
  --probe_max 6 --probe_every 75 --save_every 50 \
  --wandb 1 --wandb_name $NAME --wandb_group descent --wandb_tags descent $NAME \
  --save ckpt_train/${NAME}.pt --out experiments/results/train_$NAME.json \
  $* 2>&1 | tee -a $LOG
EOF
chmod +x "$RUN"
tmux kill-session -t "train_$NAME" 2>/dev/null || true
tmux new-session -d -s "train_$NAME" "$RUN"
sleep 3; tmux ls | sed 's/^/  /'; echo "log: $LOG"
