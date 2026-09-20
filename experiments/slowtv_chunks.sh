#!/usr/bin/env bash
# One 8,000-frame (800 s @ 10 Hz) CHUNK of a SlowTV video as its own scene --
# docs/coverage-corpus-plan.md §3-2 (rung 45) and §5 (phase 2).
#
# WHY CHUNKS AND NOT THE WHOLE VIDEO.  A SlowTV video is 1-8 h = 30k-180k
# frames; the bank and the float32 frame cache cover the first SPAN=8000 frames
# of a scene (build_banks_slowtv.sh: "SPAN is a CAP"), so a video contributes
# 13 minutes of itself however long it is.  Mapping the whole video would be
# 240 GB of cache per video.  Cutting different time ranges into separate scene
# directories gives the sampler more places in the video at the same 8k-frame
# cost each, and needs nothing new downstream: a chunk is a frames_10hz dir
# like any other.
#
# POSITIONS: the trimmed video [SKIP, total-SKIP_END] (same 5-min trims as
# slowtv_frames.sh) is divided so that chunk k of NK starts at
#   SKIP + k * (usable - CHUNK_S) / (NK - 1),  k = 0..NK-1.
# k=0 starts at SKIP, i.e. it IS the existing data/slow_tv/<vid>/frames_10hz's
# first 8000 frames -- so for an already-extracted video ask for k >= 1.
#
#   experiments/slowtv_chunks.sh <vid> <k> [<k> ...]      -> data/slow_tv/<vid>c<k>/frames_10hz
#   NK=4 experiments/slowtv_chunks.sh 00031 0 1 2 3
#
# One ffmpeg at a time: PNG export is capped by this node's single Lustre
# client (~47 PNG/s), more workers only share the ceiling (slowtv_frames.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
VID=${1:?usage: slowtv_chunks.sh <vid> <k> [k...]}; shift
ROOT=${ROOT:-data/slow_tv}
SKIP=${SKIP:-300}; SKIP_END=${SKIP_END:-300}
CHUNK_S=${CHUNK_S:-800}; FPS=${FPS:-10}; NK=${NK:-4}
WANT=$(( CHUNK_S * FPS ))
vid="$ROOT/videos/$VID.mp4"
[ -f "$vid" ] || { echo "[chunk] $VID: no video at $vid"; exit 1; }
total=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$vid")
usable=$(awk -v t="$total" -v a="$SKIP" -v b="$SKIP_END" 'BEGIN{print int(t-a-b)}')
[ "$usable" -ge "$CHUNK_S" ] || { echo "[chunk] $VID: usable $usable s < chunk $CHUNK_S s"; exit 1; }
for k in "$@"; do
  start=$(awk -v s="$SKIP" -v u="$usable" -v c="$CHUNK_S" -v k="$k" -v n="$NK" 'BEGIN{print int(s + k*(u-c)/(n-1))}')
  out="$ROOT/${VID}c${k}/frames_10hz"
  have=$(ls "$out" 2>/dev/null | grep -c '\.png$')
  if [ "$have" -ge "$WANT" ]; then echo "[chunk] ${VID}c$k: already has $have frames, skip"; continue; fi
  [ "$have" -gt 0 ] && { echo "[chunk] ${VID}c$k: partial ($have), redoing"; rm -rf "$out"; }
  mkdir -p "$out"
  echo "[chunk] ${VID}c$k: t=$start s (+$CHUNK_S s) of $usable s usable -> $WANT frames  ($(date '+%F %T'))"
  ffmpeg -v error -ss "$start" -t "$CHUNK_S" -i "$vid" -vf "fps=$FPS" -start_number 0 "$out/%06d.png"
  got=$(ls "$out" | grep -c '\.png$')
  echo "[chunk] ${VID}c$k: wrote $got frames  ($(date '+%F %T'))"
  [ "$got" -lt "$WANT" ] && echo "[chunk] ${VID}c$k: WARNING only $got of $WANT"
  printf '{"video": "%s", "k": %d, "nk": %d, "start_s": %d, "chunk_s": %d, "fps": %d}\n' \
    "$VID" "$k" "$NK" "$start" "$CHUNK_S" "$FPS" > "$ROOT/${VID}c${k}/chunk.json"
done
