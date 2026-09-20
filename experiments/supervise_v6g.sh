#!/usr/bin/env bash
# Own the whole chain to completion: re-bake -> verify -> train v6g, retrying
# each stage on failure.  Safe to kill and restart; it re-derives state from disk.
#
# ★ RELAUNCHING IS THE RESUME.  launch_v6.sh picks the highest
# ckpt_train/<NAME>.step*.pt and passes it as --resume, and the trainer restores
# weights, optimiser, step counter and every stream's (scene, K, horizon, age).
# With --save_every 50 a crash costs at most 50 steps, so "start it again" is the
# whole recovery procedure -- no special-casing per failure mode.
#
# ★ BUT A DETERMINISTIC FAILURE MUST NOT LOOP.  A NaN abort, a missing bank or an
# OOM at a fixed step reproduces on every attempt, and relaunching would burn the
# card forever while looking busy.  So an attempt that ends at the SAME step the
# previous one did counts as no progress, and two of those in a row stop the
# chain with the log to read.
set -u
cd "$(dirname "$0")/.."
NAME=${NAME:-v6g}
STEPS=${STEPS:-1250}
NEED=${NEED:-331}
MAXWAIT=${MAXWAIT:-64800}          # 18 h for the re-bake
MAX_TRAIN_TRIES=${MAX_TRAIN_TRIES:-8}
LOG=experiments/logs/supervise_$NAME.log
exec >> "$LOG" 2>&1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [sup] $*"; }

banks(){ ls -d labels/{slowtv,dl3dv,scannet,replica,dynamicreplica,unrealstereo4k,paralleldomain4d}_*_L48/index.json 2>/dev/null | wc -l; }
step(){ ls -1 ckpt_train/${NAME}.step*.pt 2>/dev/null \
        | sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }

# ── stage 1: re-bake ─────────────────────────────────────────────────────────
# This node owns shards 0-1, but the shard plan is re-derived from whatever is
# still unbaked, so restarting them also picks up work the other node's shards
# never got to.  One node can finish alone if the other dies.
log "stage 1: re-bake, $(banks)/$NEED"
t=0; last=-1
while [ "$(banks)" -lt "$NEED" ] && [ "$t" -lt "$MAXWAIT" ]; do
  n=$(banks); [ "$n" != "$last" ] && { log "  banks $n/$NEED"; last=$n; }
  for s in 0 1; do
    if ! tmux has-session -t "rebake$s" 2>/dev/null; then
      log "  shard$s not running -- (re)starting on gpu$s"
      tmux new-session -d -s "rebake$s" \
        "SHARD=$s NSHARD=4 GPU=$s experiments/rebake_all_L48.sh 2>&1 | tee -a experiments/logs/rebake_shard$s.log"
    fi
  done
  sleep 120; t=$(( t + 120 ))
done
[ "$(banks)" -lt "$NEED" ] && { log "TIMEOUT: re-bake stuck at $(banks)/$NEED"; exit 1; }
log "stage 1 done: $(banks) banks"

# ── stage 2: verify ──────────────────────────────────────────────────────────
# A bank interrupted mid-build still has an index.json listing every run, so the
# trainer opens it and only raises inside LabelBank._load minutes later, after
# every scene is mapped.  Drop the broken ones and let the shards refill them.
for pass in 1 2 3; do
  if experiments/verify_banks_L48.sh; then log "stage 2 done: banks verify clean"; break; fi
  [ "$pass" = 3 ] && { log "VERIFY still failing after 3 passes -- stopping"; exit 1; }
  log "stage 2 pass $pass: dropping broken banks and refilling"
  # ★ PULL THE PATH BY SHAPE, NOT BY FIELD NUMBER.  The two failure lines do not
  # agree on column: "BROKEN   labels/x -- ..." puts it in $2 while "WRONG L
  # labels/x -- ..." puts it in $3, because "WRONG L" is two words.  Taking $3
  # blindly fed `rm -rf --` to the shell, so the broken bank was never cleared,
  # verify failed all three passes and the chain stopped with the cards idle.
  experiments/verify_banks_L48.sh 2>&1 \
    | awk '/BROKEN|WRONG L/{for(i=1;i<=NF;i++) if($i ~ /^labels\//) print $i}' | while read -r d; do
    pgrep -af "label_bank" | grep -q "$(basename "$d")" || { log "  rm $d"; rm -rf "$d"; }
  done
  for s in 0 1; do
    tmux has-session -t "rebake$s" 2>/dev/null || tmux new-session -d -s "rebake$s" \
      "SHARD=$s NSHARD=4 GPU=$s experiments/rebake_all_L48.sh 2>&1 | tee -a experiments/logs/rebake_shard$s.log"
  done
  while tmux has-session -t rebake0 2>/dev/null || tmux has-session -t rebake1 2>/dev/null; do sleep 60; done
done

# ── stage 3: train, retrying ─────────────────────────────────────────────────
prev=-1; stalls=0
for try in $(seq 1 "$MAX_TRAIN_TRIES"); do
  s=$(step); s=${s:-0}
  if [ "$s" -ge "$STEPS" ]; then log "stage 3 done: $NAME at step $s"; break; fi
  log "stage 3 attempt $try: at step $s / $STEPS"
  CUDA_VISIBLE_DEVICES=0,1 NAME=$NAME STEPS=$STEPS NPROC=2 THREADS=16 \
    BANK_SUFFIX=_L48 MCD_ONLY=0 \
    HORIZONS="320 960 1920 3840" \
    P_IDENTITY=0.35 FIXED_HORIZON=0 \
    experiments/launch_v6.sh || log "  launcher returned non-zero"
  sleep 90
  while tmux has-session -t "train_$NAME" 2>/dev/null; do sleep 120; done
  now=$(step); now=${now:-0}
  log "  attempt $try ended at step $now (was $s)"
  if [ "$now" -le "$prev" ] || [ "$now" -le "$s" ]; then
    stalls=$(( stalls + 1 ))
    log "  NO PROGRESS ($stalls/2).  Last lines of experiments/logs/train_$NAME.log:"
    tail -25 "experiments/logs/train_$NAME.log" | sed 's/^/        /'
    [ "$stalls" -ge 2 ] && { log "STOPPING: two attempts made no progress -- this is deterministic, read the log"; exit 1; }
  else
    stalls=0
  fi
  prev=$now
done

s=$(step); s=${s:-0}
if [ "$s" -ge "$STEPS" ]; then log "CHAIN COMPLETE: $NAME step $s"; else log "GAVE UP at step $s after $MAX_TRAIN_TRIES attempts"; fi
