#!/usr/bin/env bash
# Keep one v6 training run alive to completion.  NAME picks the run; every other
# knob is passed through to launch_v6.sh by the caller's environment.
#
#   NAME=v6g STEPS=1250 NPROC=2 BANK_SUFFIX=_L48 MCD_ONLY=0 \
#   HORIZONS="320 960 1920 3840" P_IDENTITY=0.35 FIXED_HORIZON=0 \
#   experiments/watchdog_train.sh
#
# ★ WHY NOT supervise_v6g.sh.  That one waits on `tmux has-session -t train_$NAME`
# to decide an attempt is over, but launch_v6.sh backgrounds torchrun with
# setsid, never tmux -- so the wait returned instantly, every attempt looked like
# it ended at the step it started at, and the chain declared a deterministic
# failure at 12:03 while the run it had just started went on to step 200.
#
# ★ LIVENESS IS POLLED, NOT READ.  Every loss on this project was SIGKILL or
# SIGTERM, which leaves no line to grep and just stops the log.
#
# ★ THE PID COMES OUT OF "pid=NNN  log: ..." BY SHAPE.  Taking everything after
# the '=' stored the log path too, every kill -0 failed, and a second 1250-step
# run was launched onto the same two cards two minutes after the first.
set -u
cd "$(dirname "$0")/.."
NAME=${NAME:?set NAME}
# Which launcher this watchdog keeps alive.  v7 (L_long) uses launch_v7.sh, which
# pins v6i's policy and adds the long knobs.
LAUNCH=${LAUNCH:-experiments/launch_v6.sh}
TICK=${TICK:-120}
MAXRESTART=${MAXRESTART:-5}
PIDF=experiments/logs/wd_${NAME}.pid
say(){ echo "[wd:$NAME $(date '+%m-%d %H:%M:%S')] $*"; }
alive(){ [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
laststep(){ ls -1 ckpt_train/${NAME}.step*.pt 2>/dev/null | sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }

start(){
  # ★ ':-' WOULD DEFEAT AN EXPLICITLY EMPTY BANK_SUFFIX.  The untagged (L=240)
  # bake is selected by BANK_SUFFIX="", and ':-' substitutes the default for an
  # empty value as well as an unset one -- so a v7 run, which REQUIRES the L=240
  # banks, would have had its L=48 banks checked instead.  '-' only substitutes
  # when unset.
  if ! python3 experiments/check_banks.py --suffix "${BANK_SUFFIX-_L48}" > experiments/logs/bankcheck_$NAME.log 2>&1; then
    say "BANKS-BROKEN -- refusing to launch"; grep -v '^checked' experiments/logs/bankcheck_$NAME.log | head -5 | sed 's/^/       /'; return 1
  fi
  $LAUNCH >> experiments/logs/${NAME}_launch.log 2>&1
  local p; p=$(grep -a '^pid=' experiments/logs/${NAME}_launch.log | tail -1 | sed 's/^pid=\([0-9][0-9]*\).*/\1/')
  case "$p" in ''|*[!0-9]*) say "PIDPARSE failed -- not tracking"; return 1 ;; esac
  echo "$p" > "$PIDF"
  say "LAUNCH pid=$p  resume from step $(laststep)"
}

say "watchdog up.  tick ${TICK}s, max ${MAXRESTART} restarts"
n=0
while true; do
  if [ -f "ckpt_train/${NAME}.pt" ]; then say "DONE -- ckpt_train/${NAME}.pt present.  exiting."; exit 0; fi
  if pgrep -f "wandb_name $NAME" >/dev/null 2>&1; then
    alive || { pgrep -f 'torchrun.*nproc_per_node' | head -1 > "$PIDF"; say "ADOPT pid=$(cat $PIDF) (running but untracked)"; }
  elif ! alive; then
    if [ "$n" -ge "$MAXRESTART" ]; then say "GIVEUP after $n restarts at step $(laststep) -- needs a human"; exit 1; fi
    [ "$n" -gt 0 ] && say "RESTART (process gone at step $(laststep))"
    n=$(( n + 1 )); start || true
  fi
  sleep "$TICK"
done
