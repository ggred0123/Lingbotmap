#!/usr/bin/env bash
# section 10's lam calibration, done as a DISTRIBUTION rather than one window.
#
# ★ THE ALT TERM NEEDS ITS OWN KNOB, AND IT ALREADY HAS ONE.  --long_terms and
# --long_alt_terms are disjoint (trainer.py refuses otherwise), so lam_rot/lam_dir
# reach only the in-run criterion and lam_scale reaches only the stitched one.
# No new argument is needed to move the alt term alone.
#
# ★ ONE ROLLOUT PER WINDOW, NOT ONE PER lam_scale.  The alt criterion is exactly
# linear in lam_scale, so the probe derives the whole sweep from a single pair of
# gradients.  Re-running the probe per lam_scale would re-roll a 500-frame prefix
# for a number that is available in closed form.
#
# ★ WINDOWS MUST SATISFY BOTH off >= 192 AND aoff >= 272.  The first makes the
# in-run ladder fire in full; the second is the only way Delta=319 exists at all,
# and it is a property of the STITCHED offset, which is not the logged `offset`.
#
#   experiments/lam_scale_sweep.sh                  # v7f.step50, 8 windows
#   CKPT=ckpt_train/v6i.step50.pt experiments/lam_scale_sweep.sh
set -u
cd "$(dirname "$0")/.."
CKPT=${CKPT:-ckpt_train/v7f.step50.pt}
TAG=${TAG:-$(basename "$CKPT" .pt)}
SWEEP=${SWEEP:-"1 3 10 30 100 300"}
log(){ echo "[$(date '+%H:%M:%S')] [lamsweep $TAG] $*"; }

# scene:frames:t0  -- one per corpus, two windows each
CASES=(
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:512"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:992"
  "slowtv_00000:data/slow_tv/00000/frames_10hz:512"
  "slowtv_00000:data/slow_tv/00000/frames_10hz:992"
  "tuhh_day_02:data/mcd/tuhh_day_02/frames_10hz:512"
  "tuhh_day_02:data/mcd/tuhh_day_02/frames_10hz:992"
  "unrealstereo4k_00000:/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/unrealstereo4k/00000/images:512"
  "scannet_scene0005_00:/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/scannet/scannet/train/scene0005_00/color:512"
)

run_one(){ # $1=gpu $2=case
  local gpu=$1 c=$2
  local scene="${c%%:*}"; local rest="${c#*:}"
  local frames="${rest%:*}"; local t0="${rest##*:}"
  local out="experiments/results/lamsweep_${TAG}_${scene}_t${t0}.json"
  [ -f "$out" ] && { log "skip $scene t0=$t0 (exists)"; return; }
  log "gpu$gpu  $scene t0=$t0"
  CUDA_VISIBLE_DEVICES=$gpu python experiments/long_grad_probe.py \
    --ckpt "$CKPT" --frames "$frames" --bank "labels/$scene" \
    --long_terms rot,dir \
    --alt_bank "labels/${scene}_long" --long_alt_terms scale \
    --long_alt_deltas 48 96 192 319 \
    --lam_long 0.05 --target_share 0.25 --lam_scale_sweep $SWEEP \
    --t0 "$t0" --K 1 --out "$out" \
    > "experiments/logs/lamsweep_${TAG}_${scene}_t${t0}.log" 2>&1 \
    || log "FAILED $scene t0=$t0 -- see the log"
}

# split the case list across the two cards
A=(); B=(); i=0
for c in "${CASES[@]}"; do
  if [ $(( i % 2 )) -eq 0 ]; then A+=("$c"); else B+=("$c"); fi
  i=$(( i + 1 ))
done
( for c in "${A[@]}"; do run_one 0 "$c"; done ) & PA=$!
( for c in "${B[@]}"; do run_one 1 "$c"; done ) & PB=$!
wait $PA; wait $PB
log "LAMSWEEP_DONE"
