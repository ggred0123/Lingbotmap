#!/usr/bin/env bash
# Second pass of the queue.  Two things went wrong in the first one and both are
# fixed here:
#   * gtopt had no watchdog, so its OOM at step 165 left the card idle for 67 min.
#     Now every training job gets one before it starts, not after it dies.
#   * step 3 tested for gtopt.step250.pt and skipped scoring when the run had only
#     reached 170 -- an exact-checkpoint test again, the same shape as the
#     off-by-one that restarted a finished c0off 41 times.  Score whatever exists.
# The loop does not exit: when it runs out of work it re-checks, so an arm that a
# watchdog restarts gets scored when it finally lands.
set -u
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
V=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
log(){ echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S')] [q2] $*"; }
busy(){ ps -eo args | grep -qE "[t]erm_align_probe|[l]ingbot_map.train.trainer|[b]ench.sh"; }
scored(){ $V -c "
import json,re,sys
t=json.load(open('/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford/oxford/eval/traj.json'))
sys.exit(0 if [k for k in t if re.fullmatch(r'sd_$1s\d+_k1',k)] else 1)" 2>/dev/null; }
have(){ ls -1 ckpt_train/$1.step*.pt 2>/dev/null | sed 's/.*step//;s/\.pt//' | sort -n; }

for pass in 1 2 3 4 5 6 7 8 9 10; do
  while busy; do sleep 120; done
  did=0
  for arm in gtopt gtmag; do
    scored "$arm" && continue
    # only the steps this arm actually reached, intersected with gtsup's grid so
    # the comparison stays paired
    G=""
    for s in $(have "$arm"); do
      case " 25 50 75 100 150 200 250 275 " in *" $s "*) G="$G $s";; esac
    done
    [ -z "$G" ] && { log "$arm has no checkpoint on gtsup's grid yet"; continue; }
    log "scoring $arm at$G"
    ARM="$arm" STEPS_K1="$G" experiments/score_k1_c0off.sh 2>&1 | sed 's/^/    /'
    did=1
    while busy; do sleep 60; done
  done
  [ "$did" = 0 ] && { log "nothing to do on pass $pass; sleeping"; sleep 600; }
done
log "Q2 DONE"
