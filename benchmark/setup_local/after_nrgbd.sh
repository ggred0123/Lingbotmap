#!/usr/bin/env bash
# Sees the Neural RGB-D download through to a prepared BSS workspace.
#
# The first version of this watched `tmux has-session -t dl_nrgbd` and treated
# the session ending as "download finished".  That was wrong: the tmux session
# died while its wget kept running as an orphan, so this script decided the
# download was over, found no extracted directory, and exited -- leaving a live
# download nobody was waiting on.  So wait on the *file*, not the session, and
# resume the download (wget -c) whenever it stops short.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/data_bench
ZIP="$DATA/neural_rgbd_data.zip"
DIR="$DATA/neural_rgbd_data"
SIZE=7785287298                      # content-length advertised by the TUM host
PY=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
TRIES=${TRIES:-20}

for attempt in $(seq 1 "$TRIES"); do
  [ -d "$DIR" ] && break

  # Never start a second wget on the same file: wait out whatever is running.
  while pgrep -f "neural_rgbd_data.zip" > /dev/null; do sleep 60; done

  have=$( [ -f "$ZIP" ] && stat -c %s "$ZIP" || echo 0 )
  echo "[after_nrgbd] attempt $attempt: $((have / 1000000)) MB of $((SIZE / 1000000)) MB"
  ./setup_local/download_neural_rgbd.sh || echo "[after_nrgbd] download attempt $attempt ended early"
done

if [ ! -d "$DIR" ]; then
  echo "[after_nrgbd] giving up after $TRIES attempts -- the TUM host keeps dropping the transfer."
  exit 1
fi
echo "[after_nrgbd] scenes: $(ls "$DIR" | tr '\n' ' ')"

echo "[after_nrgbd] adapter smoke check"
PYTHONPATH="$(cd .. && pwd)" "$PY" - <<'PYEOF'
import logging, yaml
logging.disable(logging.INFO)
from benchmark.core.registry import ClassLoader
cfg = yaml.safe_load(open('configs/datasets/neural_rgbd.yaml'))
kwargs = {k[1:]: v for k, v in cfg.items() if k.startswith('_')}
ds = ClassLoader.load_dataset(cfg['dataset'])(raw_data_root=cfg['raw_data_root'], **kwargs)
scenes = ds.get_scenes()
frames = ds.get_frame_list(scenes[0])
d = ds.load_frame_data(scenes[0], frames[0])
print(f"OK  {len(scenes)} scenes, first='{scenes[0]}' {len(frames)} frames, "
      f"rgb={d['rgb'].shape}, keys={sorted(k for k in d if k != 'timestamp')}")
PYEOF
[ $? -ne 0 ] && { echo "[after_nrgbd] adapter check failed -- inspect the extracted layout before preparing"; exit 1; }

echo "[after_nrgbd] prepare"
THREADS=8 ./bench.sh prepare configs/neural_rgbd.yaml
echo "[after_nrgbd] done"
