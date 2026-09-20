#!/usr/bin/env bash
# Does tilting A1PC's all-pairs rotation loss toward long gaps point at GT?
#
# ★ ONLY THE GT WINDOWS ARE HERE.  cos(band, local) can be measured anywhere, but
# it cannot decide anything -- docs2/C.md's table needs cos(term, GT), and GT
# exists only on MCD.  The window list is the GT half of c_conflict_sweep.sh, so
# every row here sits beside an existing cos(local, GT) from the C experiment
# rather than being a new, separately-sampled population.
#
#   experiments/gap_weight_sweep.sh
#   CKPT=ckpt_train/v6i.step50.pt TAG=v6i experiments/gap_weight_sweep.sh
set -u
cd "$(dirname "$0")/.."
CKPT=${CKPT:-ckpt_train/v7f.step50.pt}
TAG=${TAG:-v7f}
CALIB=data/mcd/calib/hhs_calib.yaml
log(){ echo "[$(date '+%H:%M:%S')] [span $TAG] $*"; }

# scene : frames : t0 : K
CASES=(
  "kth_night_04:data/mcd/kth_night_04/frames_10hz:512:1"
  "kth_night_04:data/mcd/kth_night_04/frames_10hz:512:28"
  "kth_night_04:data/mcd/kth_night_04/frames_10hz:992:1"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:512:1"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:512:28"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:992:1"
  "kth_day_10:data/mcd/kth_day_10/frames_10hz:992:28"
  "tuhh_day_02:data/mcd/tuhh_day_02/frames_10hz:512:1"
  "tuhh_day_02:data/mcd/tuhh_day_02/frames_10hz:512:28"
  "tuhh_day_02:data/mcd/tuhh_day_02/frames_10hz:992:1"
  "tuhh_day_04:data/mcd/tuhh_day_04/frames_10hz:512:1"
  "tuhh_day_04:data/mcd/tuhh_day_04/frames_10hz:512:12"
)

run_one(){ # $1=gpu $2=case
  local gpu=$1 c=$2 scene frames t0 K
  IFS=: read -r scene frames t0 K <<< "$c"
  local out="experiments/results/span_${TAG}_${scene}_t${t0}_K${K}.json"
  [ -f "$out" ] && { log "skip $scene t$t0 K$K"; return; }
  log "gpu$gpu  $scene t0=$t0 K=$K"
  CUDA_VISIBLE_DEVICES=$gpu python -u experiments/span_probe.py \
    --ckpt "$CKPT" --frames "$frames" --bank "labels/$scene" \
    --alt_bank "labels/${scene}_long" \
    --t0 "$t0" --K "$K" --gt_calib "$CALIB" --out "$out" \
    > "experiments/logs/span_${TAG}_${scene}_t${t0}_K${K}.log" 2>&1 \
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
log "SPAN_DONE"
