#!/usr/bin/env bash
# Frozen K=1 teacher banks for SlowTV -- docs/self-distill-ver5.md step 3.
#
# WHY THIS IS NOT build_banks.sh.  That script calls experiments/gt_span.py to
# find each scene's GT-contiguous range, and gt_span.py reads meta.npz.  SlowTV
# has no GT and no meta.npz -- it never will, it is monocular YouTube video --
# so there is no span to look up.  v5 says so directly: "pass --span 0
# <frame_count> to label_bank directly instead".
#
# Nothing in the distillation objective needs GT: the frozen base model supplies
# pose, depth and confidence targets, and every loss term is gauge-invariant.
# What SlowTV loses is only the GT PROBE, which is metric-only anyway -- so a
# SlowTV run is trained without --gt_calib and scored on MCD holdout instead.
#
# ★ THE TEACHER PASS IS THE COST OF THIS STEP, so it is budgeted out loud.
# Each run is 8 anchor + 72 burn-in + L supervised forwards for L supervised
# frames, i.e. (80 + L)/L ~ 1.33 forwards per label at L=240.  Measured forward
# cost on this box is ~50 ms, so one 8000-frame span is ~10.6k forwards ~ 9 min
# and 33 runs x 195 MB ~ 6.4 GB on disk.  Nine sequences: ~1.5 h and ~58 GB.
#
# SPAN is a CAP, not a target.  SlowTV holds ~944k frames against MCD's ~60k, and
# banking all of it would be 13 h of teacher and 767 GB -- for a corpus that a
# 50/50 dataset-balanced sampler will revisit 15x less often per frame than MCD
# anyway (v5 "Dataset Sampling").  Raise SPAN when the sampler is shown to be
# starved of SlowTV variety, not before.
#
#   experiments/build_banks_slowtv.sh              # all 9 downloaded sequences
#   SPAN=16000 experiments/build_banks_slowtv.sh 00000 00003
set -u
cd "$(dirname "$0")/.."

CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
BURN_IN=${BURN_IN:-72}
L=${L:-240}
# See build_banks_generic.sh: L is a property of the BANK, and the loop skips a
# scene whose index.json exists, so a re-bake at a different L must land on a
# different path.  SUFFIX=_L48 -> labels/slowtv_00000_L48.
SUFFIX=${SUFFIX:-}
SPAN=${SPAN:-8000}
ROOT=${ROOT:-data/slow_tv}
DRY=${DRY:-0}

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ "$#" -gt 0 ]; then
  SEQS="$*"
else
  SEQS=$(ls -d "$ROOT"/[0-9][0-9][0-9][0-9][0-9] 2>/dev/null | xargs -n1 basename | tr '\n' ' ')
fi
[ -n "${SEQS// /}" ] || { log "no sequences under $ROOT"; exit 1; }

log "sequences: $SEQS"
log "span cap $SPAN, L=$L, burn_in=$BURN_IN, suffix ${SUFFIX:-<none>}, ckpt=$CKPT"

# ---------------------------------------------------------------- budget first
TOT_F=0; TOT_R=0
for s in $SEQS; do
  d="$ROOT/$s/frames_10hz"
  [ -d "$d" ] || { log "SKIP $s -- no $d"; continue; }
  n=$(find "$d" -maxdepth 1 -name '*.png' -o -maxdepth 1 -name '*.jpg' | wc -l)
  use=$(( n < SPAN ? n : SPAN ))
  # runs are tiled from frame (burn_in + scale_frames) = 80
  usable=$(( use - BURN_IN - 8 ))
  runs=$(( usable > 0 ? (usable + L - 1) / L : 0 ))
  TOT_F=$(( TOT_F + use )); TOT_R=$(( TOT_R + runs ))
  log "  $s: $n frames, banking $use -> $runs runs"
done
log "TOTAL: $TOT_F frames, $TOT_R runs, ~$(( TOT_R * 195 / 1000 )) GB, ~$(( TOT_F * 133 / 100 * 50 / 1000 / 60 )) min of teacher"
[ "$DRY" = "1" ] && { log "DRY=1, stopping before any GPU work"; exit 0; }

if [ ! -e /dev/nvidiactl ]; then
  log "FATAL no GPU on this node (/dev/nvidiactl absent) -- the teacher pass needs one"
  exit 1
fi

mkdir -p experiments/logs labels

# ------------------------------------------------------------------- teacher
built=0
for s in $SEQS; do
  d="$ROOT/$s/frames_10hz"
  out="labels/slowtv_$s$SUFFIX"
  [ -d "$d" ] || continue
  [ -f "$out/index.json" ] && { log "have $out"; continue; }
  n=$(find "$d" -maxdepth 1 -name '*.png' -o -maxdepth 1 -name '*.jpg' | wc -l)
  use=$(( n < SPAN ? n : SPAN ))
  if [ "$use" -lt 400 ]; then
    log "SKIP $s -- only $use frames, too short for anchor+burn-in+L"
    continue
  fi
  log "BUILD slowtv_$s  (span [0,$use) of $n)"
  # --span 0 <cap>: no GT, so no gt_span.py.  See the header.
  python -m lingbot_map.train.label_bank \
    --ckpt "$CKPT" --frames "$d" --out "$out" \
    --span 0 "$use" --burn_in "$BURN_IN" --L "$L" \
    >> "experiments/logs/bank_slowtv_$s.log" 2>&1
  if [ -f "$out/index.json" ]; then
    log "DONE  slowtv_$s -> $out ($(du -sh "$out" | cut -f1))"
    built=$(( built + 1 ))
    # The trainer reads a float32 memmap when one exists and otherwise loads the
    # whole span resident.  Cache exactly the banked range, not the video.
    log "CACHE slowtv_$s frames"
    python experiments/cache_frames.py "$s" --root "$ROOT" \
      --bank_prefix slowtv_ --bank_suffix "$SUFFIX" \
      >> "experiments/logs/bank_slowtv_$s$SUFFIX.log" 2>&1 \
      || log "WARN cache_frames failed for $s -- the trainer will load resident"
  else
    log "FAIL  slowtv_$s -- see experiments/logs/bank_slowtv_$s.log"
    tail -5 "experiments/logs/bank_slowtv_$s.log" | sed 's/^/        /'
  fi
done

log "ALL DONE ($built built this session, $(ls -d labels/slowtv_*/index.json 2>/dev/null | wc -l) SlowTV banks on disk)"
