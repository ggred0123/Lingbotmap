#!/usr/bin/env bash
# Two gates before committing v5c to a long run:
#   (a) K=4 step curve -- does the low-K collapse improve or worsen with steps?
#   (b) short-L sweep at K=12 -- at what distance does v5c start beating base?
set -u
cd "$(dirname "$0")/.."
G=$1; shift
for job in "$@"; do
  case "$job" in
    step:*) experiments/chain_v5_kcurve.sh "$G" "v5cs${job#step:}:4" ;;
    shortL:*) experiments/mcd_short_sweep.sh "$G" 12 "${job#shortL:}" 960 1920 3840 ;;
  esac
done
echo "[$(date '+%H:%M:%S')] GATE_DONE gpu$G"
