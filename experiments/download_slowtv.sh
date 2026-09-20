#!/usr/bin/env bash
# SlowTV (jspenmar/slowtv_monodepth) -- 40 YouTube videos, 138 h, 1280x720, ~128 GB.
#
# Differences from the upstream api/data/download/slow_tv.sh, and why:
#
#   * Upstream downloads the whole list in one yt-dlp call with
#     `-o "videos/%(autonumber)s.%(ext)s"`.  autonumber counts *successful*
#     downloads, so one unavailable video silently shifts every later index --
#     and the shipped split files address sequences by index (00000 00001 ...,
#     paired with splits/categories.txt line by line).  Here each URL is fetched
#     separately into its own line-numbered name, so a failure leaves a hole
#     instead of corrupting the mapping.
#
#   * Upstream pins format 136 (720p30 AVC), which the clients below do serve
#     for all 40 videos.  The fallback chain exists for the day one of them
#     stops: it keeps 1280x720 and prefers the lowest frame rate, and never
#     drops below 720p -- asking for fps<=30 without pinning the height silently
#     selects 480p on some videos, which would change the training data.  Frame
#     rate itself is harmless, since export_slow_tv.py extracts at 10 fps.
#
#   * The system yt-dlp (2024.04) fails against YouTube's current API, so this
#     uses .venv-dl, an isolated venv holding a current yt-dlp plus a node
#     binary (nodejs-wheel-binaries) for JavaScript signature deciphering.
#
#   * YouTube now cuts anonymous downloads off at ~10 MB with HTTP 403 on the
#     default player clients (android_vr, tv_embedded, mediaconnect all die
#     there; web/mweb/ios are served storyboard images only, i.e. they want a
#     proof-of-origin token).  web_embedded and android are the two clients that
#     still serve full streams here, and with them every video offers format 136
#     -- the exact 720p30 AVC stream the paper used.
#
#   * --throttled-rate 2M.  YouTube throttles a long-running stream: measured
#     9.5 MB/s at the start of a session and 0.6 MB/s a few hours in, while a
#     *fresh* stream opened at the same moment still ran at 10.8 MB/s.  So the
#     limit is per stream, not per host, and re-extracting the URL clears it --
#     which is what this flag does when the rate drops below the threshold.
#     Without it the 112 GB takes ~40 h instead of ~4 h.
#
# Resumable: partial files continue, finished ones are skipped.  Failures are
# appended to failed.txt and do not stop the run.
#
#   experiments/download_slowtv.sh              # download everything
#   ROOT=/somewhere/slow_tv experiments/download_slowtv.sh
set -uo pipefail

ROOT=${ROOT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill/data/slow_tv}
YTDLP=${YTDLP:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-dl/bin/yt-dlp}
NODE=${NODE:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-dl/lib/python3.12/site-packages/nodejs_wheel/bin/node}
CLIENTS=${CLIENTS:-web_embedded,android}
RAW=https://raw.githubusercontent.com/jspenmar/slowtv_monodepth/main/api/data/splits/slow_tv/splits
FORMAT='136/bv*[height=720][fps<=30]/bv*[height=720]/bv*[height<=720]'

# info/ is deliberately NOT inside videos/: export_slow_tv.py lists that dir
# with io.get_files() (unfiltered, not the .mp4-filtered get_vid_files()), so a
# sidecar per video makes len(video_files) 80 vs 40 categories -- and sorts as
# 00000.info, 00000.mp4, ... which mispairs every sequence with its category.
mkdir -p "$ROOT/videos" "$ROOT/splits" "$ROOT/info"

# Split metadata: urls.txt and categories.txt are what map videos to sequences;
# the *_files.txt lists are what a training run reads.
for f in urls.txt categories.txt all/train_files.txt all/val_files.txt \
         driving/train_files.txt driving/val_files.txt \
         natural/train_files.txt natural/val_files.txt \
         underwater/train_files.txt underwater/val_files.txt; do
  dst="$ROOT/splits/$f"
  [ -s "$dst" ] && continue
  mkdir -p "$(dirname "$dst")"
  echo "[slowtv] fetching splits/$f"
  curl -sSfL "$RAW/$f" -o "$dst" || echo "[slowtv] WARNING: could not fetch $f"
done

n_urls=$(grep -c . "$ROOT/splits/urls.txt")
n_cats=$(grep -c . "$ROOT/splits/categories.txt")
if [ "$n_urls" != "$n_cats" ]; then
  echo "[slowtv] ABORT: urls.txt ($n_urls) and categories.txt ($n_cats) disagree"
  exit 1
fi
echo "[slowtv] $n_urls videos -> $ROOT/videos"

i=0
# `|| [ -n "$url" ]`: splits/urls.txt ships with no trailing newline, so a
# plain `while read` returns non-zero on the 40th URL and drops it -- 39/40
# videos, and preprocess_slowtv.sh then aborts on the 40-category check
# forever.  Exactly the index-alignment failure this script exists to avoid.
while read -r url || [ -n "$url" ]; do
  [ -z "$url" ] && continue
  idx=$(printf '%05d' "$i")
  i=$((i + 1))

  # Skip only when a finished file exists (a leftover .part means resume).
  done_file=""
  for ext in mp4 webm mkv; do
    [ -f "$ROOT/videos/$idx.$ext" ] && [ ! -f "$ROOT/videos/$idx.$ext.part" ] && done_file="$idx.$ext"
  done
  if [ -n "$done_file" ]; then
    echo "[slowtv] $idx already downloaded ($done_file)"
    continue
  fi

  echo "[slowtv] $idx <- $url"
  "$YTDLP" --no-playlist --no-overwrites --continue \
           --retries 10 --fragment-retries 10 --concurrent-fragments 4 \
           --sleep-requests 1 --throttled-rate 2M \
           --js-runtimes "node:$NODE" \
           --extractor-args "youtube:player_client=$CLIENTS" \
           -f "$FORMAT" \
           -o "$ROOT/videos/$idx.%(ext)s" \
           --print-to-file "%(id)s %(format_id)s %(width)sx%(height)s %(fps)s" "$ROOT/info/$idx.info" \
           "$url" \
    || { echo "[slowtv] FAILED $idx $url"; echo "$idx $url" >> "$ROOT/failed.txt"; }
done < "$ROOT/splits/urls.txt"

echo "[slowtv] downloaded $(ls "$ROOT/videos" | grep -cE '\.(mp4|webm|mkv)$')/$n_urls videos, "\
"$(du -sh "$ROOT/videos" | cut -f1)"
[ -s "$ROOT/failed.txt" ] && echo "[slowtv] failures listed in $ROOT/failed.txt (re-run to retry)"
echo "[slowtv] done"
