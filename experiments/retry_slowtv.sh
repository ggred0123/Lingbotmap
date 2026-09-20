#!/usr/bin/env bash
# Re-runs download_slowtv.sh on a backoff until all 40 videos are present.
#
# Why this exists: after ~11 videos / 33 GB from one IP, YouTube started
# answering every extraction with "Sign in to confirm you're not a bot" on both
# web_embedded and android -- the two clients that still serve full streams.
# That flag is IP/session based and decays on its own, so the cure is patience
# rather than credentials: logging in to bulk-fetch 112 GB is what actually
# risks an account.
#
# Runs *as* tmux session dl_slowtv, so preprocess_slowtv.sh's existing
# `while tmux has-session -t dl_slowtv` wait needs no change: it now waits for
# the whole retry campaign instead of a single pass.
#
#   INTERVAL=7200  seconds between attempts (default 2 h)
#   START_DELAY=1800  wait before the first attempt (the IP is hot right now)
#   MAX_ATTEMPTS=36   give up after this many (default ~3 days at 2 h)
set -uo pipefail

ROOT=${ROOT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill/data/slow_tv}
HERE=$(cd "$(dirname "$0")" && pwd)
INTERVAL=${INTERVAL:-7200}
START_DELAY=${START_DELAY:-1800}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-36}
WANT=$(grep -c . "$ROOT/splits/categories.txt")

count() { ls "$ROOT/videos"/*.mp4 2>/dev/null | wc -l; }

echo "[retry] $(date '+%F %T')  have $(count)/$WANT; first attempt in $((START_DELAY/60)) min"
sleep "$START_DELAY"

for a in $(seq 1 "$MAX_ATTEMPTS"); do
  have=$(count)
  if [ "$have" -ge "$WANT" ]; then
    echo "[retry] $(date '+%F %T')  all $WANT videos present -- done"; exit 0
  fi

  # failed.txt is append-only, so truncate per attempt to keep it a current
  # picture rather than a union of every attempt ever made.
  : > "$ROOT/failed.txt"

  echo "[retry] ===== attempt $a/$MAX_ATTEMPTS  $(date '+%F %T')  have $have/$WANT ====="
  "$HERE/download_slowtv.sh"
  now=$(count)
  echo "[retry] attempt $a finished: $have -> $now / $WANT"

  if [ "$now" -ge "$WANT" ]; then
    echo "[retry] $(date '+%F %T')  all $WANT videos present -- done"; exit 0
  fi
  if [ "$now" -gt "$have" ]; then
    echo "[retry] progress made; the block is lifting"
  else
    echo "[retry] no progress; still flagged"
  fi
  echo "[retry] sleeping $((INTERVAL/60)) min"
  sleep "$INTERVAL"
done

echo "[retry] gave up after $MAX_ATTEMPTS attempts with $(count)/$WANT videos"
exit 1
