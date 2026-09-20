#!/usr/bin/env bash
# Stop train_a3 once its step-N checkpoint is fully written.
#
# save_every is 50, so the trainer writes ckpt_train/a3.step300.pt and then keeps
# going.  Killing the session the instant the file appears can truncate an
# 11.5 GB write, so wait for the size to stop changing first -- and only then
# kill, which lets chain_a3 (waiting on the session) take over.
#
#   tmux new-session -d -s stop_a3 "experiments/stop_a3_at.sh 300"
set -u
cd "$(dirname "$0")/.."
STEP=${1:-300}
CK="ckpt_train/a3.step${STEP}.pt"
LOG=experiments/logs/chain_a3.log
log() { echo "[$(date '+%F %T')] [stopper] $*" | tee -a "$LOG"; }

log "waiting for $CK"
while [ ! -f "$CK" ]; do
  tmux has-session -t train_a3 2>/dev/null || { log "train_a3 died before step $STEP"; exit 1; }
  sleep 30
done

log "$CK appeared, waiting for the write to settle"
prev=-1
while :; do
  cur=$(stat -c %s "$CK" 2>/dev/null || echo 0)
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 10000000000 ]; then break; fi
  prev=$cur
  sleep 20
done
log "$CK stable at $cur bytes"

sleep 10
tmux kill-session -t train_a3 2>/dev/null && log "killed train_a3 at step $STEP" \
  || log "train_a3 already gone"
