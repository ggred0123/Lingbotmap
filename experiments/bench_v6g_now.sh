#!/usr/bin/env bash
# Bench the v6g checkpoints that have piled up while the other node's evaluation
# pipeline sat idle (traj.json last written 14:39, twelve checkpoints since).
#
# ★ RUNS ALONGSIDE THE v6g TRAINING ON PURPOSE.  The trainer has already reserved
# its share of both cards and PyTorch never returns cached blocks, so a bench that
# does not fit simply fails to allocate -- it cannot take the training down with
# it.  GPU 0 is the one with headroom.
#
# ★ STRIP FIRST, ON CPU.  ckpt_train/*.pt is 11 GB of weights+optimiser+stream
# snapshots; the bench only reads the weights.
set -u
cd "$(dirname "$0")/.."
STEPS=${STEPS:-"250 300 350 400 450 500 550 600 650 700 750 800"}
GPU=${GPU:-0}
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

for s in $STEPS; do
  src=ckpt_train/v6g.step$s.pt; dst=bench_ckpt/sd_v6g_step$s.pt
  [ -f "$src" ] || { log "skip s$s -- no checkpoint"; continue; }
  if [ ! -f "$dst" ]; then
    log "strip s$s"
    python3 experiments/_strip_optim.py "$src" "$dst" >/dev/null || { log "STRIP FAILED s$s"; continue; }
  fi
  cfg=benchmark/configs/methods/sd_v6gs${s}_k1.yaml
  [ -f "$cfg" ] || sed "s|sd_v6g_step50\.pt|sd_v6g_step${s}.pt|" \
      benchmark/configs/methods/sd_v6gs50_k1.yaml > "$cfg"
done
log "strip done: $(ls -1 bench_ckpt/sd_v6g_step*.pt | wc -l) checkpoints ready"

# one config listing every v6g k1 method plus the baseline, so evaluate scores
# them all into the same table run.py skips whatever already has .complete.json
M=$(ls -1 benchmark/configs/methods/sd_v6gs*_k1.yaml | sed 's|.*/||;s|\.yaml||' \
    | sort -t s -k3 -n | sed 's/^/  - /')
cat > benchmark/configs/oxford_v6g_k1.yaml <<YAML
# Oxford stride-12 at K=1 for the whole v6g checkpoint sweep.
workspace: /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford
datasets:
  - oxford
methods:
  - base_k1
$M
evaluation:
  traj: {enable: true, vis: false}
  auc: {enable: true, vis: false, aggregation: both}
  depth: {enable: false}
  points: {enable: false}
YAML
log "=== RUN on gpu$GPU ==="
GPU=$GPU THREADS=8 ./benchmark/bench.sh run configs/oxford_v6g_k1.yaml 2>&1 \
  | grep -aE "Combination \(|Scenes to process|Scene \(|Successful:|Total failed|already complete|rror|Traceback|CUDA out of memory"
log "=== EVALUATE ==="
GPU=$GPU ./benchmark/bench.sh evaluate configs/oxford_v6g_k1.yaml 2>&1 \
  | grep -aE "Total success|Total failed|rror|Traceback"
log "V6G_BENCH_DONE"
