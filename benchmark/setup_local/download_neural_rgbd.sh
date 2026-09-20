#!/usr/bin/env bash
# Neural RGB-D (Azinovic et al.) -- 7.8 GB zip, no preprocessing needed:
# the archive's {scene}/{images,depth,poses.txt,focal.txt} layout is exactly what
# benchmark/datasets/neural_rgbd.py expects.
set -euo pipefail
DEST=${DEST:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/data_bench}
URL=https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip
mkdir -p "$DEST"
cd "$DEST"

if [ ! -d neural_rgbd_data ]; then
  echo "[nrgbd] downloading $(basename $URL)"
  # Never combine -c with -O.  wget then restarts at offset 0 *without*
  # truncating, so a retry overwrites the head and leaves the previous
  # attempt's tail in place -- which is how the first 4.47 GB partial got
  # corrupted.  basename($URL) is already neural_rgbd_data.zip, so -O was
  # redundant; dropping it makes -c a real byte-range resume.
  wget -c --progress=dot:giga "$URL"
  [ -f neural_rgbd_data.zip ] || { echo "[nrgbd] expected neural_rgbd_data.zip, got: $(ls)"; exit 1; }
  echo "[nrgbd] extracting"
  mkdir -p neural_rgbd_data
  unzip -q -o neural_rgbd_data.zip -d neural_rgbd_data
  # the zip may or may not carry a top-level dir; flatten if it does
  inner=$(ls neural_rgbd_data)
  if [ "$(echo "$inner" | wc -l)" = "1" ] && [ -d "neural_rgbd_data/$inner" ] \
     && [ ! -d "neural_rgbd_data/$inner/images" ]; then
    mv neural_rgbd_data/"$inner"/* neural_rgbd_data/ && rmdir neural_rgbd_data/"$inner"
  fi
  rm -f neural_rgbd_data.zip
fi
echo "[nrgbd] scenes:"; ls neural_rgbd_data | sed 's/^/  /'
