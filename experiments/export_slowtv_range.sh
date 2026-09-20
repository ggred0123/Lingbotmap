#!/usr/bin/env bash
# Export a *range* of SlowTV sequences, one at a time.
#
# Why one at a time: PNG export throughput is capped per NODE, not per core.
# Measured on kaist-bispl-ym-cpu (72 cores, 2.2 TB RAM, all idle):
#   N=1 -> 47 PNG/s   N=4 -> 43   N=8 -> 38   N=16 -> 34   N=32 -> ~22 and falling
# Aggregate is flat-to-declining because every worker shares the node's single
# Lustre client; adding workers only splits the same ~47 PNG/s more ways and
# makes per-directory inserts worse as the dirs fill. So NPROC>1 is a pessimisation.
#
# The way to go faster is more NODES (each has its own Lustre client), which is
# what the range is for:
#   node A:  experiments/export_slowtv_range.sh 0 19
#   node B:  experiments/export_slowtv_range.sh 20 39
#
# Sequences already exported are skipped (io.has_contents), so this is resumable.
# NOTE: splits/*_files.txt are appended to per sequence; if two nodes run
# concurrently, regenerate them at the end with experiments/regen_slowtv_splits.py.
set -uo pipefail
FROM=${1:?usage: export_slowtv_range.sh <from_idx> <to_idx>}
TO=${2:?usage: export_slowtv_range.sh <from_idx> <to_idx>}
REPO=${REPO:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/tools/slowtv_monodepth}
CONDA=${CONDA:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/miniconda3}

source "$CONDA/etc/profile.d/conda.sh"; conda activate slowtv || exit 1
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
for i in $(seq "$FROM" "$TO"); do
  echo "[export] ===== idx $i  ($(date '+%F %T')) ====="
  python api/data/preprocess/export_slow_tv.py --idx "$i" || echo "[export] idx $i FAILED"
done
echo "[export] range $FROM-$TO done ($(date '+%F %T'))"
