#!/usr/bin/env bash
# docs2/C.md -- which long TERM conflicts, and does the conflict agree with GT?
#
# ★ THE AGGREGATE cos(local, long) CANNOT ANSWER THIS.  rot and dir come from the
# in-run bank and scale from the stitched track, and the GT audit puts those two
# sources at opposite ends: stitched rotation error 3.96-13.57 deg against 1.12
# in-run, while stitched scale bias is the one that is near zero.  A single
# cosine averages a term that is probably harmful with one that is probably the
# whole point.  Every window here reports |g| and cos per term.
#
# ★ AND cos(local, long) ALONE CANNOT DECIDE ANYTHING EITHER.  A term that fights
# local is fine if it agrees with GT (that is the correction we want) and fatal if
# it does not.  On MCD the probe also takes the gradient of the window's GT ATE,
# which is the only direction here that is not a teacher's opinion, and C.md's
# table is read off cos(term, GT):
#
#   local-  GT+   necessary conflict     keep
#   local-  GT-   harmful target         mask this condition
#   local+  GT-   imitating a wrong teacher together   revisit loss/target
#   local+  GT+   safe reinforcement     lam candidate
#
# ★ THE WINDOW LIST IS BALANCED, NOT RANDOM (C.md).  Five axes: corpus,
# shallow/deep history, low/high K, stitched seam PASS/FAIL, and -- for eight
# pairs -- the SAME raw window with only K changed, which is the only way to see
# K's effect without the scene changing underneath it.
#
#   experiments/c_conflict_sweep.sh
#   CKPT=ckpt_train/v6i.step50.pt TAG=v6i experiments/c_conflict_sweep.sh
set -u
cd "$(dirname "$0")/.."
CKPT=${CKPT:-ckpt_train/v7f.step50.pt}
TAG=${TAG:-v7f}
CALIB=data/mcd/calib/hhs_calib.yaml
log(){ echo "[$(date '+%H:%M:%S')] [C $TAG] $*"; }

# scene : frames : t0 : K : seam_median_m : GT?
CASES=(
  # ── GT scenes (MCD).  These carry C.md's decisive measurement. ────────────
  "kth_night_04:data/mcd/kth_night_04/frames_10hz:512:1:0.00002:gt"
  "kth_night_04:data/mcd/kth_night_04/frames_10hz:512:28:0.00002:gt"
  "kth_night_04:data/mcd/kth_night_04/frames_10hz:992:1:0.00002:gt"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:512:1:0.00138:gt"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:512:28:0.00138:gt"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:992:1:0.00138:gt"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:992:28:0.00138:gt"
  "tuhh_day_02:data/mcd/tuhh_day_02/frames_10hz:512:1:0.00364:gt"
  "tuhh_day_02:data/mcd/tuhh_day_02/frames_10hz:512:28:0.00364:gt"
  "tuhh_day_02:data/mcd/tuhh_day_02/frames_10hz:992:1:0.00364:gt"
  "tuhh_day_04:data/mcd/tuhh_day_04/frames_10hz:512:1:0.00274:gt"
  "tuhh_day_04:data/mcd/tuhh_day_04/frames_10hz:512:12:0.00274:gt"
  # ── seam PASS, no GT ──────────────────────────────────────────────────────
  "slowtv_00006:data/slow_tv/00006/frames_10hz:512:1:0.00000:-"
  "slowtv_00006:data/slow_tv/00006/frames_10hz:512:28:0.00000:-"
  "scannet_scene0147_01:/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/scannet/scannet/train/scene0147_01/color:512:1:0.00070:-"
  "scannet_scene0147_01:/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/scannet/scannet/train/scene0147_01/color:512:28:0.00070:-"
  "dynamicreplica_009850-3_obj_source:/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/dynamicreplica/009850-3_obj_source/images:512:1:0.00133:-"
  # ── seam FAIL, no GT ──────────────────────────────────────────────────────
  "unrealstereo4k_00000:/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/unrealstereo4k/00000/images:512:1:0.00819:-"
  "unrealstereo4k_00000:/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/unrealstereo4k/00000/images:512:28:0.00819:-"
  "paralleldomain4d_scene_000037:/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/paralleldomain4d/scene_000037/images:512:1:0.02510:-"
  "paralleldomain4d_scene_000037:/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/paralleldomain4d/scene_000037/images:512:28:0.02510:-"
  "paralleldomain4d_scene_000010:/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/paralleldomain4d/scene_000010/images:512:1:0.01743:-"
)

run_one(){ # $1=gpu $2=case
  local gpu=$1 c=$2
  local scene frames t0 K seam gt
  IFS=: read -r scene frames t0 K seam gt <<< "$c"
  local out="experiments/results/cconf_${TAG}_${scene}_t${t0}_K${K}.json"
  [ -f "$out" ] && { log "skip $scene t$t0 K$K"; return; }
  local gtarg=""
  [ "$gt" = "gt" ] && gtarg="--gt_calib $CALIB"
  log "gpu$gpu  $scene t0=$t0 K=$K seam=$seam ${gt}"
  CUDA_VISIBLE_DEVICES=$gpu python experiments/long_grad_probe.py \
    --ckpt "$CKPT" --frames "$frames" --bank "labels/$scene" \
    --long_terms rot,dir \
    --alt_bank "labels/${scene}_long" --long_alt_terms scale \
    --long_alt_deltas 48 96 192 319 \
    --lam_long 0.05 --target_share 0.25 $gtarg \
    --t0 "$t0" --K "$K" --out "$out" \
    > "experiments/logs/cconf_${TAG}_${scene}_t${t0}_K${K}.log" 2>&1 \
    || log "FAILED $scene t$t0 K$K"
}

A=(); B=(); i=0
for c in "${CASES[@]}"; do
  if [ $(( i % 2 )) -eq 0 ]; then A+=("$c"); else B+=("$c"); fi
  i=$(( i + 1 ))
done
log "${#CASES[@]} windows: ${#A[@]} on card0, ${#B[@]} on card1"
( for c in "${A[@]}"; do run_one 0 "$c"; done ) & PA=$!
( for c in "${B[@]}"; do run_one 1 "$c"; done ) & PB=$!
wait $PA; wait $PB
log "CSWEEP_DONE"
