#!/usr/bin/env bash
# Generate one fresh-teacher label bank per TRAINING scene, as extraction makes
# them available.  Polls, so it can be launched while mcd_extract.py is still
# working through the remaining sequences.
#
# Corpus split (docs: v4 section 3 data hygiene, section 7 eval scenes):
#   TRAIN            kth_day_10, kth_night_{01,04,05}, tuhh_* (6)
#   EVAL in-domain   kth_day_06, kth_day_09     <- never trained on
#   EVAL held-out    ntu_* (6)                  <- different site/platform/calib
#
# A scene is only picked up once meta.npz exists: its absence means extraction is
# still writing PNGs, and a bank built against a growing directory would index
# frames that shift underneath it.
set -u
cd "$(dirname "$0")/.."

CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
BURN_IN=${BURN_IN:-72}
L=${L:-240}
# See build_banks_generic.sh: L is a property of the BANK, and the loop skips a
# scene whose index.json exists, so a re-bake at a different L must land on a
# different path.  SUFFIX=_L48 -> labels/kth_day_10_L48.
SUFFIX=${SUFFIX:-}
# Long-target bake knobs; see build_banks_generic.sh and
# docs/long_supervision_design.md section 6.
STRIDE=${STRIDE:-}
POSE_ONLY=${POSE_ONLY:-}
K_T=${K_T:-}
POLL=${POLL:-120}

TRAIN_SCENES="kth_day_10 kth_night_01 kth_night_04 kth_night_05 \
tuhh_day_02 tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 tuhh_night_09"

log() { echo "[$(date +%H:%M:%S)] $*"; }

done_count=0
while :; do
  pending=0
  for s in $TRAIN_SCENES; do
    d=data/mcd/$s/frames_10hz
    out=labels/$s$SUFFIX
    [ -f "$out/index.json" ] && continue          # already built
    if [ ! -f "$d/meta.npz" ]; then pending=$((pending+1)); continue; fi

    # Cap the span at the GT-contiguous range, not the PNG count: frames outside
    # it have no GT row, and the probe's scorer refuses such a window rather than
    # silently measuring a shifted one.
    read -r LO HI < <(python experiments/gt_span.py "$d" --verbose)
    n=$((HI-LO))
    if [ "$n" -lt 400 ]; then
      log "SKIP $s -- only $n GT-backed frames, too short for anchor+burn-in+L"
      continue
    fi
    log "BUILD $s  (GT span [$LO,$HI) = $n frames)"
    python -m lingbot_map.train.label_bank \
      --ckpt "$CKPT" --frames "$d" --out "$out" \
      --span "$LO" "$HI" --burn_in "$BURN_IN" --L "$L" \
      ${STRIDE:+--stride "$STRIDE"} ${POSE_ONLY:+--pose_only} \
      ${K_T:+--teacher_interval "$K_T"} \
      >> experiments/logs/bank_$s$SUFFIX.log 2>&1
    if [ -f "$out/index.json" ]; then
      sz=$(du -sh "$out" | cut -f1)
      log "DONE  $s -> $out ($sz)"
      done_count=$((done_count+1))
    else
      log "FAIL  $s -- see experiments/logs/bank_$s$SUFFIX.log"
      tail -5 experiments/logs/bank_$s$SUFFIX.log | sed 's/^/        /'
    fi
  done

  built=$(ls -d labels/*/index.json 2>/dev/null | wc -l)
  if [ "$pending" -eq 0 ]; then
    log "ALL TRAIN SCENES BUILT ($done_count this session, $built banks on disk)"
    break
  fi
  log "waiting: $pending scene(s) still extracting; sleeping ${POLL}s"
  sleep "$POLL"
done
