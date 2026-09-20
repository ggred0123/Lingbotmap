#!/usr/bin/env bash
# Two 300-step diagnostics, run back to back on both cards.
# Baseline is s0off's own steps 0-300 -- same corpus, same batch, same seed.
set -u
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [descent] $*"; }
train_step(){ grep -aoE '^ *\[ *[0-9]+\]' "experiments/logs/train_$1.log" 2>/dev/null | tail -1 | tr -dc 0-9; }
train_done(){
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys
try: d = json.load(open(f"experiments/results/train_{sys.argv[1]}.json"))
except Exception: sys.exit(1)
sys.exit(0 if d.get("wall_s") else 1)
PY
}
alive(){ tmux has-session -t "train_$1" 2>/dev/null; }
no_trainer(){ [ "$(ps -eo args | grep -c '[l]ingbot_map.train.trainer')" = "0" ]; }

wait_arm(){
  local n=$1 t=0
  log "waiting for $n"
  while [ "$t" -lt 86400 ]; do
    train_done "$n" && { log "$n finished"; return 0; }
    alive "$n" || { log "ABORT: $n died at step $(train_step "$n")"; return 1; }
    sleep 60; t=$((t+60))
    [ $((t % 900)) -eq 0 ] && log "  ... $n at step $(train_step "$n")"
  done
  log "TIMEOUT on $n"; return 1
}

run_arm(){ # $1=name  rest=flags
  local n=$1; shift
  train_done "$n" && { log "$n already done"; return 0; }
  alive "$n" || experiments/launch_descent.sh "$n" "$@"
  wait_arm "$n" || return 1
  for _ in $(seq 30); do no_trainer && break; sleep 20; done
  python3 experiments/mk_spine_report.py >/dev/null 2>&1
}

log "=== (a) step size: lr 1e-5 -> 1e-6 ==="
run_arm dlr --lr 1e-6 || exit 1
log "=== (b) L_motion off: lam_motion 0.9 -> 0 ==="
run_arm dmot --lam_motion 0 || exit 1
log "DESCENT_DONE"
