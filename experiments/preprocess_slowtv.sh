#!/usr/bin/env bash
# SlowTV preprocessing: waits for the download, then runs the upstream
# api/data/preprocess/export_slow_tv.py -- frame export at 10 fps, COLMAP
# intrinsics, decimation and split generation.
#
# Environment: the `slowtv` conda env (conda-forge only -- the defaults channel
# needs a Terms of Service acceptance, which is the user's to give).  It holds
# COLMAP plus the handful of Python packages the preprocessing touches; the
# repo's docker/environment.yml is a full 2023 training environment and is not
# needed to preprocess.
#
# Two things this checks before starting, because export_slow_tv.py cannot:
#
#   * All 40 videos present.  The script asserts len(categories) == len(videos)
#     and otherwise pairs sequences with the wrong category labels.
#   * Every file is .mp4.  get_vid_files() filters on that suffix, so a .webm
#     fallback would be silently dropped from the corpus -- and from the
#     category pairing above.
#
#   experiments/preprocess_slowtv.sh            # wait for download, then run
#   NPROC=8 experiments/preprocess_slowtv.sh    # fewer workers (default 12)
#   NOWAIT=1 experiments/preprocess_slowtv.sh   # skip the wait, run now
set -uo pipefail

DATA=${DATA:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill/data/slow_tv}
REPO=${REPO:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/tools/slowtv_monodepth}
CONDA=${CONDA:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/miniconda3}
ENV=${ENV:-slowtv}
# The trainer holds ~32 threads on 72 cores; leave it room.
NPROC=${NPROC:-12}

if [ -z "${NOWAIT:-}" ]; then
  echo "[prep] waiting for the download to finish..."
  while tmux has-session -t dl_slowtv 2>/dev/null; do sleep 120; done
fi

n_mp4=$(ls "$DATA/videos"/*.mp4 2>/dev/null | wc -l)
n_other=$(ls "$DATA/videos" 2>/dev/null | grep -cE '\.(webm|mkv)$')
n_all=$(ls "$DATA/videos" 2>/dev/null | wc -l)
n_cat=$(grep -c . "$DATA/splits/categories.txt")

echo "[prep] videos: $n_mp4 mp4, $n_other non-mp4, $n_all entries total; categories: $n_cat"
# export_slow_tv.py pairs io.get_files(videos/) -- every entry, unfiltered -- with
# categories.txt line by line.  So ANY extra file in videos/ (a .info sidecar, a
# stray .part) both breaks its len() assert and shifts the pairing.  Check the
# total, not just the mp4 count: that is what upstream actually sees.
if [ "$n_all" != "$n_mp4" ]; then
  echo "[prep] ABORT: videos/ holds $n_all entries but only $n_mp4 .mp4 -- export_slow_tv.py"
  echo "       lists the directory unfiltered, so the extras mispair every sequence."
  ls "$DATA/videos" | grep -v '\.mp4$' | sed 's/^/         /' | head
  exit 1
fi
if [ "$n_other" != "0" ]; then
  echo "[prep] ABORT: non-mp4 files present -- get_vid_files() ignores them, which would"
  echo "       shift every sequence against splits/categories.txt.  Re-download those."
  exit 1
fi
if [ "$n_mp4" != "$n_cat" ]; then
  echo "[prep] ABORT: $n_mp4 videos vs $n_cat categories.  export_slow_tv.py asserts these match."
  [ -s "$DATA/failed.txt" ] && { echo "       failed downloads:"; sed 's/^/         /' "$DATA/failed.txt"; }
  echo "       Re-run experiments/download_slowtv.sh (it resumes) before preprocessing."
  exit 1
fi

[ -f "$REPO/PATHS.yaml" ] || { echo "[prep] ABORT: $REPO/PATHS.yaml missing (it points src.paths at $DATA)"; exit 1; }

# shellcheck disable=SC1091
source "$CONDA/etc/profile.d/conda.sh"
conda activate "$ENV" || { echo "[prep] ABORT: conda env '$ENV' not found"; exit 1; }
command -v colmap >/dev/null || { echo "[prep] ABORT: colmap not on PATH inside '$ENV'"; exit 1; }
echo "[prep] $(colmap -h 2>&1 | head -1)  |  $(ffmpeg -version | head -1 | cut -d' ' -f1-3)"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "[prep] export_slow_tv.py --n-proc $NPROC   (frames @10fps, COLMAP intrinsics, decimate, splits)"
python api/data/preprocess/export_slow_tv.py --n-proc "$NPROC"
rc=$?

echo "[prep] exit=$rc"
echo "[prep] sequences: $(find "$DATA" -maxdepth 1 -mindepth 1 -type d ! -name videos ! -name splits ! -name colmap | wc -l)"
echo "[prep] frames on disk: $(du -sh "$DATA" 2>/dev/null | cut -f1)"
exit $rc
