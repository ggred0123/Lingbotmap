#!/usr/bin/env bash
# lambda for L_abs from the GRADIENT share -- docs/gtabs-plan.md §2-4 / §5-3-3.
# term_grad_probe.py --abs_mode paper on the same drifted windows the termalign
# sweep used (4 GT scenes x t0 {512, 992} x K {1, 28}), against the GT banks,
# at theta_0.  ★ ONE PROBE PER CARD: the caching allocator holds ~150 GB after
# the first backward, so a second probe on the same card OOMs (measured).
#   GPU=1 experiments/run_abs_grad_sweep.sh
#   python3 experiments/abs_lam_solve.py      -> experiments/results/abs_grad_share.json
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
FIT=${FIT:-prefix}
CKPT=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
for sc in kth_day_10 kth_night_04 tuhh_day_02 tuhh_day_04; do
  for K in 1 28; do
    o=experiments/results/tg_abs_${sc}_K${K}.json
    [ -f "$o" ] && { echo "[sweep] have $o"; continue; }
    echo "[sweep] $(date +%H:%M) $sc K=$K"
    while ps -eo args | grep -q "[t]erm_grad_probe.py"; do sleep 20; done   # the card is shared with nobody
    python3 experiments/term_grad_probe.py --ckpt $CKPT --frames data/mcd/$sc/frames_10hz \
      --probes labels/${sc}_gt:512 labels/${sc}_gt:992 --K $K --preset A1PC \
      --abs_mode paper --abs_fit $FIT --out "$o" > experiments/logs/tg_abs_${sc}_K${K}.log 2>&1
  done
done
echo "[sweep] DONE"
