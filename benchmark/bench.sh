#!/usr/bin/env bash
# Run a benchmark phase with the bench venv.
#
# This machine has no conda envs -- prepare/run/evaluate all execute under
# .venv-bench (system-site-packages, so it reuses the container's torch and adds
# opencv / open3d / evo / OpenEXR / plyfile / trimesh on top).  Method configs
# therefore carry no `env:` field and run.py executes the model in-process.
#
#   benchmark/bench.sh prepare  configs/oxford.yaml [--debug]
#   benchmark/bench.sh run      configs/oxford.yaml
#   benchmark/bench.sh evaluate configs/oxford.yaml
#   benchmark/bench.sh all      configs/oxford.yaml       # the three in order
#   GPU=1 benchmark/bench.sh run configs/oxford.yaml      # pick a device
set -euo pipefail
cd "$(dirname "$0")"

PY=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
PHASE=${1:-}
[ -n "$PHASE" ] || { echo "usage: bench.sh <prepare|run|evaluate|all|report> <config> [flags...]" >&2; exit 2; }
shift

# Training occupies both GPUs; default the benchmark to GPU 1 and keep its CPU
# footprint small so it does not fight the trainer for the 72 cores.
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export OMP_NUM_THREADS=${THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"   # so `import lingbot_map` resolves

case "$PHASE" in
  prepare|run|evaluate) exec "$PY" "$PHASE.py" --config "$@" ;;
  report)               exec "$PY" report.py --workspace "$@" ;;
  all)
    CFG=$1; shift
    "$PY" prepare.py  --config "$CFG" "$@"
    "$PY" run.py      --config "$CFG" "$@"
    "$PY" evaluate.py --config "$CFG" "$@"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
