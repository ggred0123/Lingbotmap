#!/usr/bin/env bash
# ETH3D high-res multi-view (training split) for benchmark/datasets/eth3d.py.
#
# Two ingredients, both from the official site:
#   multi_view_training_dslr_undistorted.7z  (5.9 GB) -- undistorted DSLR images
#                                                        + COLMAP calibration
#   {scene}_dslr_depth.7z                    (per scene) -- GT depth, raw float32
#                                                          binary with a .JPG name
# There is no combined depth archive, so depth is fetched scene by scene.
# convert_eth3d.py then repacks this into the 'custom_undistorted' layout the
# adapter reads (images/custom_undistorted, ground_truth_depth/custom_undistorted,
# custom_undistorted_cam/*.npz).
set -euo pipefail
DEST=${DEST:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/data_bench}
RAW="$DEST/eth3d_raw"
BASE=https://www.eth3d.net/data

# The 11 scenes benchmark/datasets/eth3d.py actually evaluates
# (SCENES minus DA3_FILTER_SCENES = meadow, terrace).
SCENES="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"

mkdir -p "$RAW"
cd "$RAW"

echo "[eth3d] undistorted images + calibration"
wget -c --progress=dot:giga -O multi_view_training_dslr_undistorted.7z \
     "$BASE/multi_view_training_dslr_undistorted.7z"
7z x -y -bso0 -bsp0 multi_view_training_dslr_undistorted.7z

for s in $SCENES; do
  echo "[eth3d] depth: $s"
  wget -c --progress=dot:giga -O "${s}_dslr_depth.7z" "$BASE/${s}_dslr_depth.7z"
  7z x -y -bso0 -bsp0 "${s}_dslr_depth.7z"
done

echo "[eth3d] raw layout:"
ls "$RAW" | sed 's/^/  /'
echo "[eth3d] now run:  python setup_local/convert_eth3d.py"
