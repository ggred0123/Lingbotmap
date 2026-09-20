#!/usr/bin/env bash
# Keep the BROAD-corpus arm (docs/gtabs-plan.md §13: gtctrl's recipe on the
# 214-scene clean corpus, teacher labels) alive on its card until it finishes.
# Same restart policy as watchdog_gtabs.sh; the launcher is launch_clean.sh.
#
#   experiments/watchdog_broad.sh <arm> <gpu> [trainer flags...]
#   e.g.  setsid nohup experiments/watchdog_gtabs.sh gtscale 1 \
#             --abs_mode scale --abs_fit prefix --lam_trans_scale X --lam_depth_scale Y &
#
# One loop per arm, detached (setsid), not cron: it polls every 5 minutes, and
# it is the same restart policy for every arm of the cell -- the plan's
# "same save_every, same restart policy" -- so the control and the treated arms
# see identical checkpointing.
#
# ★ COMPLETION IS wall_s IN THE RESULTS JSON, never a checkpoint name
# (watchdog_gtmag.sh: --steps 275 with save_every 10 never writes step275.pt).
# ★ save_every STAYS ON THE 25-GRID.  The scoring grid is 25 50 75 ... 275, so
# a derived interval must divide 25: it is 25 while a cycle survives >= 50
# steps and 5 otherwise (a livelock guard in the spirit of watchdog_teasup.sh,
# without breaking the grid with a 10/15/20).
# ★ THE TREATED ARMS MUST RUN THE FLAGS THEY WERE LAUNCHED WITH ON EVERY
# RESTART, so the flags live in this process's argv, not in a file that could
# be edited under it.
#
# STOP with:  touch experiments/logs/.<arm>.stop
set -u
cd "$(dirname "$0")/.."
ARM=$1; GPU=$2; shift 2
STOP=experiments/logs/.$ARM.stop
COUNT=experiments/logs/.$ARM.restarts
WLOG=experiments/logs/watchdog_$ARM.log
TARGET=${TARGET:-275}
POLL=${POLL:-300}
mkdir -p experiments/logs
rm -f "$STOP"; echo 0 > "$COUNT"

ckpt_step(){ ls ckpt_train/$ARM.step*.pt 2>/dev/null | sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
log_step(){ grep -aoE '^ *\[ *[0-9]+\]' experiments/logs/train_$ARM.log 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1; }
finished(){ python3 -c "import json,sys; d=json.load(open('experiments/results/train_$ARM.json')); sys.exit(0 if d.get('wall_s') else 1)" 2>/dev/null; }
# Bracket so the pattern cannot match this script's own command line.
alive(){ ps -eo args | grep -q "[-]-wandb_name $ARM "; }

echo "$(TZ=Asia/Seoul date '+%F %H:%M') watchdog up: arm=$ARM gpu=$GPU flags=$*" >> "$WLOG"
while :; do
  [ -f "$STOP" ] && { echo "$(TZ=Asia/Seoul date '+%F %H:%M') stop file -- exiting" >> "$WLOG"; exit 0; }
  if finished; then
    echo "$(TZ=Asia/Seoul date '+%F %H:%M') $ARM finished (wall_s present) -- exiting" >> "$WLOG"
    touch "$STOP"; exit 0
  fi
  if ! alive; then
    CK=$(ckpt_step); CK=${CK:-0}; LS=$(log_step); LS=${LS:-0}
    SURV=$(( LS - CK ))
    if [ "$SURV" -gt 0 ] && [ "$SURV" -lt 50 ]; then SE=5; else SE=25; fi
    N=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 )); echo "$N" > "$COUNT"
    if [ "$N" -gt 20 ]; then
      echo "$(TZ=Asia/Seoul date '+%F %H:%M') giving up after $N launches" >> "$WLOG"; touch "$STOP"; exit 1
    fi
    echo "$(TZ=Asia/Seoul date '+%F %H:%M') launch #$N  ckpt=$CK log=$LS survival=$SURV -> save_every=$SE" >> "$WLOG"
    GPU=$GPU NPROC=1 POOL=20 FRESH_POOL=1 PROBE_MAX=1 SAVE_EVERY=$SE STEPS=$TARGET THREADS=${THREADS:-8} \
      experiments/launch_clean.sh "$ARM" 0.35 "$@" >> "$WLOG" 2>&1
  fi
  sleep "$POLL"
done
