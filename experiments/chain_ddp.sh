#!/usr/bin/env bash
# Wait for the running evaluation, verify DDP with a short smoke, then launch the
# real run.  The smoke is not ceremony: every crash this session appeared during
# SETUP (scene mapping, build_model, pool), never mid-training, so a 3-step run
# that reaches the first optimizer step has cleared every failure seen so far.
set -u
cd "$(dirname "$0")/.."

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for evaluation to finish"
while pgrep -f "[a]te_vs_distance" >/dev/null; do sleep 60; done
log "evaluation done"
sleep 20

SC=""
for s in kth_day_10 kth_night_01 kth_night_04 kth_night_05 tuhh_day_02 \
         tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 tuhh_night_09; do
  SC="$SC $s:data/mcd/$s/frames_10hz:labels/$s"
done

log "DDP smoke: 3 steps, both ranks, full 10-scene corpus"
LINGBOT_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
torchrun --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
  -m lingbot_map.train.trainer \
  --ckpt /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt \
  --scene $SC --preset A1 --lam_fresh 1.0 --l2sp 0 \
  --steps 3 --pool 10 --probe_max 1 --probe_every 0 --wandb 0 \
  --out /tmp/_ddp_smoke.json > experiments/logs/ddp_smoke.log 2>&1
RC=$?

if [ $RC -ne 0 ] || ! grep -qE "^\s+\[\s*[0-9]+\]" experiments/logs/ddp_smoke.log; then
  log "SMOKE FAILED (rc=$RC) -- not launching the real run"
  grep -av "it/s\]" experiments/logs/ddp_smoke.log | tail -25
  exit 1
fi
log "smoke passed:"
grep -aE "^\s+\[\s*[0-9]+\]|ddp\]" experiments/logs/ddp_smoke.log | tail -5

sleep 20
log "launching A2 (L_fresh + no L2-SP) on both GPUs"
experiments/launch_ddp.sh a2 --preset A1 --lam_fresh 1.0 --l2sp 0
log "done -- tmux attach -t train_a2"
