#!/usr/bin/env bash
# SlowTV -> lingbot_map training layout.
#
# Produces data/slow_tv/<seq>/frames_10hz/%06d.png : one CONTINUOUS 10 Hz run per
# video, matching data/mcd/<scene>/frames_10hz.
#
# Why continuous, and why not the upstream preprocessing:
#
#   The slowtv_monodepth pipeline exists to feed a photometric-reprojection
#   monodepth model.  It keeps 100 of every 250 frames (`data_scale = 4`), i.e.
#   10 s of 10 Hz video then a 15 s hole, which is right for that model -- it only
#   ever needs adjacent frame PAIRS -- and it also runs COLMAP per sequence to
#   recover the camera intrinsics that a reprojection loss needs.
#
#   lingbot_map needs neither.  Its labels are teacher outputs over a CONTIGUOUS
#   span: labels/<scene>/index.json shows burn_in=72 + L=240, so one run needs 312
#   consecutive frames = 31 s at 10 Hz.  A 100-frame block cannot hold even one.
#   And the loss is teacher-vs-student depth, so no intrinsics are involved.
#
#   So: no decimation, no COLMAP.  Just continuous frames.
#
# Naming: both label_bank.py and trainer.py index frames through image_names(),
# which is `sorted(listdir)` filtered to image extensions -- so a zero-padded
# counter is all that is required; MCD's `_<sec>_<nsec>` suffix is not parsed.
#
#   experiments/slowtv_frames.sh 0 39            # every video, full length
#   DUR=900 experiments/slowtv_frames.sh 0 19    # cap at 15 min each, first half
#
# One ffmpeg at a time on purpose: PNG export is capped per NODE by its single
# Lustre client (measured 47 PNG/s at N=1, 43 at N=4, 34 at N=16 -- more workers
# only split the same ceiling).  Use a second NODE to go faster, not more workers.
set -uo pipefail
FROM=${1:?usage: slowtv_frames.sh <from_idx> <to_idx>}
TO=${2:?usage: slowtv_frames.sh <from_idx> <to_idx>}
ROOT=${ROOT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill/data/slow_tv}
CONDA=${CONDA:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/miniconda3}
SKIP=${SKIP:-300}          # drop the first 5 min (intros/titles), as upstream did
SKIP_END=${SKIP_END:-300}  # and the last 5 min (outros/credits)
DUR=${DUR:-0}              # seconds per video; 0 = the whole video minus the trims
FPS=${FPS:-10}

source "$CONDA/etc/profile.d/conda.sh"; conda activate slowtv || exit 1

for i in $(seq "$FROM" "$TO"); do
  seq_id=$(printf '%05d' "$i")
  vid="$ROOT/videos/$seq_id.mp4"
  out="$ROOT/$seq_id/frames_10hz"
  [ -f "$vid" ] || { echo "[frames] $seq_id: no video, skip"; continue; }

  # per-video length when DUR=0: everything between the two trims
  if [ "$DUR" -eq 0 ]; then
    total=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$vid")
    dur=$(awk -v t="$total" -v a="$SKIP" -v b="$SKIP_END" 'BEGIN{d=t-a-b; print (d>0?int(d):0)}')
  else
    dur=$DUR
  fi
  WANT=$((dur * FPS))
  [ "$WANT" -lt 312 ] && { echo "[frames] $seq_id: only $WANT frames after trim (<312, one run needs that); skip"; continue; }

  have=$(ls "$out" 2>/dev/null | wc -l)
  if [ "$have" -ge "$WANT" ]; then
    echo "[frames] $seq_id: already has $have frames, skip"; continue
  fi
  # a short/partial dir is redone rather than resumed: a half-written run would be
  # indistinguishable from a complete one to image_names(), and every label offset
  # is computed from that listing.
  [ "$have" -gt 0 ] && { echo "[frames] $seq_id: partial ($have), redoing"; rm -rf "$out"; }

  mkdir -p "$out"
  echo "[frames] $seq_id: $dur s @ ${FPS}Hz from t=$SKIP -> ~$WANT frames  ($(date '+%F %T'))"
  ffmpeg -v error -ss "$SKIP" -t "$dur" -i "$vid" -vf "fps=$FPS" -start_number 0 "$out/%06d.png"
  got=$(ls "$out" | wc -l)
  echo "[frames] $seq_id: wrote $got frames"
  [ "$got" -lt $((WANT * 9 / 10)) ] && echo "[frames] $seq_id: WARNING only $got of $WANT expected"
done
echo "[frames] range $FROM-$TO done ($(date '+%F %T'))"
