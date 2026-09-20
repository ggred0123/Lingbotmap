#!/usr/bin/env bash
# Keep the card busy.  Jobs run back to back; each one checks whether its output
# already exists and skips if so, so the queue is safe to re-run and safe to
# append to while it is running.
#
# WHY A QUEUE.  Every hand-off in this session has left the GPU idle -- 25 h after
# the first silent OOM, 18 h with .c0on.stop set, and minutes at a time between
# probe runs while a human reads results.  The card is the scarce thing; analysis
# is not.  Nothing here needs a decision to start.
set -u
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
V=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
log(){ echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S')] [queue] $*"; }
busy(){ ps -eo args | grep -qE "[t]erm_align_probe|[l]ingbot_map.train.trainer|[b]ench.sh"; }

# 0) never start on top of something already on the card
while busy; do log "waiting for the current job"; sleep 120; done

# 1) teasup: agent-A finished it at 285 and asked me to score it (MSG-15 (ii)).
#    Same grid as gtsup so the label contrast is paired point for point.
if ! $V -c "
import json,re,sys
t=json.load(open('/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford/oxford/eval/traj.json'))
sys.exit(0 if [k for k in t if re.fullmatch(r'sd_teasups\d+_k1',k)] else 1)" 2>/dev/null; then
  log "scoring teasup (Oxford K=1, grid matched to gtsup)"
  ARM=teasup STEPS_K1="25 50 75 100 150 200 250 275" experiments/score_k1_c0off.sh 2>&1 | sed 's/^/    /'
else
  log "teasup already scored, skipping"
fi
while busy; do sleep 60; done

# 2) the lambda* run.  Weights come from the sweep's median, not from one window.
#    L_mag is the only term aligned with GT (median cos +0.29 against L_rot's
#    -0.01) and ships at weight 0; gtmag put it at ~5% of the gradient and moved
#    nothing, so this run gives it the share the closed-form solution asks for.
# re-solve every time: the sweep may have added windows since the last pass and
# solving is seconds.  The 12-window answer and the 16-window answer must not
# silently differ by whichever ran first.
log "solving for lambda* from every window the sweep has finished"
$V experiments/solve_lam_star.py > experiments/logs/lam_star.log 2>&1 || log "solve failed"
grep -E "^windows|^cos|lam_" experiments/logs/lam_star.log | sed 's/^/    /' 
if [ -f experiments/results/lam_star.json ] && [ ! -f ckpt_train/gtopt.step270.pt ]; then
  read -r LR LD LM LDEP LMO <<< "$($V -c "
import json; d=json.load(open('experiments/results/lam_star.json'))['train_lambdas']
print(d['lam_rot'], d['lam_dir'], d['lam_mag'], d['lam_depth'], d['lam_motion'])")"
  log "launching gtopt with lam_rot=$LR lam_dir=$LD lam_mag=$LM lam_depth=$LDEP lam_motion=$LMO"
  NPROC=1 POOL=20 PROBE_MAX=1 SAVE_EVERY=10 STEPS=275 \
    experiments/launch_gtsup.sh gtopt _gt \
      --lam_rot "$LR" --lam_dir "$LD" --lam_mag "$LM" --lam_depth "$LDEP" --lam_motion "$LMO" 2>&1 | sed 's/^/    /'
  sleep 30
  while ps -eo args | grep -q "[-]-wandb_name gtopt"; do sleep 120; done
fi
while busy; do sleep 60; done

# 3) score it against gtsup on the same grid
if [ -f ckpt_train/gtopt.step250.pt ]; then
  log "scoring gtopt"
  ARM=gtopt STEPS_K1="50 100 150 200 250" experiments/score_k1_c0off.sh 2>&1 | sed 's/^/    /'
fi
log "QUEUE DONE"
