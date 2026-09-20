#!/usr/bin/env bash
# Never exits.  gpu_queue2.sh ran out of passes at 03:55 and the card sat idle for
# nine hours; that is the third idle gap in this session and every one of them
# cost more than the work would have.  This loop just re-checks forever.
set -u
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
V=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
log(){ echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S')] [busy] $*"; }
busy(){ ps -eo args | grep -qE "[t]erm_align_probe|[l]ong_grad_probe|[l]ingbot_map.train.trainer|[b]ench.sh"; }
scored(){ $V -c "
import json,re,sys
t=json.load(open('/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford/oxford/eval/traj.json'))
sys.exit(0 if [k for k in t if re.fullmatch(r'sd_$1s\d+_k1',k)] else 1)" 2>/dev/null; }

while :; do
  while busy; do sleep 120; done

  # 1) calibrate lam_long in the drifted regime.  long_supervision_result.md fact 1:
  #    a share measured at frozen theta_0 is ~40x too large, so this reads it off a
  #    checkpoint that has actually moved -- gtsup's step 275.
  if [ ! -f experiments/results/longshare_gtsup275.json ]; then
    log "calibrating lam_long against gtsup.step275"
    timeout 2400 $V experiments/long_grad_probe.py \
      --ckpt ckpt_train/gtsup.step275.pt --frames data/mcd/kth_day_10/frames_10hz \
      --bank labels/kth_day_10_gt --t0 512 --target_share 0.25 \
      --gt_calib data/mcd/calib/hhs_calib.yaml --gt_sensor d455b_color \
      --out experiments/results/longshare_gtsup275.json 2>&1 | tail -25 | sed 's/^/    /'
    while busy; do sleep 60; done
  fi

  # 2) the one candidate the null-space test passed that has never been trained
  #    with v7d's two defects fixed (--long_weight_norm, --long_scale_gauge_norm).
  if [ ! -f experiments/results/train_gtlong.json ] && [ -f experiments/results/longshare_gtsup275.json ]; then
    LL=$($V -c "
import json
d=json.load(open('experiments/results/longshare_gtsup275.json'))
for k in ('lam_for_target_share','lam_target','lam_long_target','lam_at_target'):
    if k in d: print(d[k]); break
else: print(1.0)" 2>/dev/null || echo 1.0)
    log "launching gtlong with lam_long=$LL (+ the v7d fixes)"
    NPROC=1 POOL=20 PROBE_MAX=1 SAVE_EVERY=10 STEPS=275 \
      experiments/launch_gtsup.sh gtlong _gt \
        --lam_long "$LL" --long_deltas 48 96 192 \
        --long_weight_norm 1 --long_scale_gauge_norm 1 --long_vlocal grad 2>&1 | sed 's/^/    /'
    sleep 30
    while busy; do sleep 120; done
  fi

  # 3) score anything trained that has not been scored, on gtsup's grid
  did=0
  for arm in gtlong gtopt gtmag; do
    scored "$arm" && continue
    G=""
    for s in $(ls -1 ckpt_train/$arm.step*.pt 2>/dev/null | sed 's/.*step//;s/\.pt//' | sort -n); do
      case " 25 50 75 100 150 200 250 275 " in *" $s "*) G="$G $s";; esac
    done
    [ -z "$G" ] && continue
    log "scoring $arm at$G"
    ARM="$arm" STEPS_K1="$G" experiments/score_k1_c0off.sh 2>&1 | tail -6 | sed 's/^/    /'
    did=1
    while busy; do sleep 60; done
  done
  [ "$did" = 0 ] && { log "idle pass -- nothing left; re-checking in 10 min"; sleep 600; }
done
