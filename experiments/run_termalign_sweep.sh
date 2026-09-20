#!/usr/bin/env bash
# lambda* from one window is a coincidence; from a dozen it is a prescription.
# Same windows gap_weight_probe.py used on Sep 3, so the two measurements sit on
# the same axes: 4 GT scenes x t0 {512, 992} x K {1, 28}.
set -u
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
V=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench/bin/python
for sc in kth_day_10 kth_night_04 tuhh_day_02 tuhh_day_04; do
  for t0 in 512 992; do
    for K in 1 28; do
      o=experiments/results/termalign_${sc}_t${t0}_K${K}.json
      [ -f "$o" ] && { echo "[sweep] have $o"; continue; }
      echo "[sweep] $(date +%H:%M) $sc t0=$t0 K=$K"
      timeout 1800 $V experiments/term_align_probe.py --scene "$sc" --t0 "$t0" --K "$K" \
        --out "$o" 2>&1 | grep -E "cos |ceiling|L_mag|L_rot|no run|Error|Traceback" | sed 's/^/    /'
    done
  done
done
echo "[sweep] DONE"
