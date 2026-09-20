#!/usr/bin/env bash
# Frozen K=1 teacher banks for any unlabelled video corpus.
#
# WHY THIS IS NOT build_banks_slowtv.sh.  That script hardcodes SlowTV's layout
# (data/slow_tv/<5-digit>/frames_10hz) and its labels/slowtv_ prefix.  The v6
# mixture adds six more corpora with a <root>/<scene>/images layout, so the
# layout is parameterised here instead of copied six times.  Everything else is
# identical: no GT is needed (the teacher makes pose/depth/conf targets and every
# loss term is gauge-invariant -- see build_banks_slowtv.sh's header), so
# --span 0 <cap> goes straight to label_bank with no gt_span.py lookup.
#
# ★ SCENES SHORTER THAN burn_in + 8 + L (= 320 at L=240) PRODUCE ZERO RUNS.
# That is what rules out MVS-Synth (100 frames), SAIL-VOS 3D (69-183) and
# HyperSim (98) regardless of how well their motion band fits.
#
#   DATASET=dl3dv ROOT=/path/to/dl3dv_wai N=150 experiments/build_banks_generic.sh
#   DATASET=scannet ROOT=... SUBDIR=color N=30 experiments/build_banks_generic.sh
#   DRY=1 ... # budget only, no GPU work
set -u
cd "$(dirname "$0")/.."

DATASET=${DATASET:?set DATASET, e.g. DATASET=dl3dv}
ROOT=${ROOT:?set ROOT, the directory holding the scene directories}
SUBDIR=${SUBDIR:-images}
CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
BURN_IN=${BURN_IN:-72}
L=${L:-240}
# ★ SUFFIX NAMES THE BAKE, and it is what makes a re-bake possible at all.  The
# output path is keyed on the scene alone, and the loop below SKIPS a scene whose
# index.json exists ("have $out"), so `L=48 build_banks_generic.sh` against an
# existing labels/ tree builds nothing and says so 300 times.  L is a property of
# the bank, not of the scene, so it belongs in the path: SUFFIX=_L48 writes
# labels/dl3dv_<scene>_L48 beside the L=240 bake instead of over it.
# cache_frames.py --bank_suffix and launch_v6.sh BANK_SUFFIX read the same name.
SUFFIX=${SUFFIX:-}
# ── long-target bake knobs (docs/long_supervision_design.md section 6) ───────
# STRIDE < L gives the overlap the seam Sim(3) is fitted on; POSE_ONLY drops
# depth/conf (3.5 kB per run instead of 78 MB) because L_long never reads them;
# K_T raises the per-run supervised span, (budget - sf - B) * K_t, which is the
# stitching-free route to an in-run Delta=319 pair.
STRIDE=${STRIDE:-}
POSE_ONLY=${POSE_ONLY:-}
K_T=${K_T:-}
SPAN=${SPAN:-8000}
N=${N:-0}                    # 0 = every scene that qualifies
MINF=$(( BURN_IN + 8 + L ))  # below this a scene yields no runs at all
DRY=${DRY:-0}

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ "$#" -gt 0 ]; then
  SEQS="$*"
else
  # Scan in listing order and stop at N qualifying scenes.  Scanning all 10395
  # DL3DV scenes to pick 150 costs more Lustre round-trips than the banking.
  SEQS=""; cnt=0
  for s in $(ls "$ROOT" 2>/dev/null); do
    d="$ROOT/$s/$SUBDIR"; [ -d "$d" ] || continue
    n=$(ls "$d" 2>/dev/null | grep -icE "\.(png|jpg|jpeg)$")
    [ "$n" -lt "$MINF" ] && continue
    SEQS="$SEQS $s"; cnt=$(( cnt + 1 ))
    [ "$N" -gt 0 ] && [ "$cnt" -ge "$N" ] && break
  done
fi
[ -n "${SEQS// /}" ] || { log "no qualifying scenes under $ROOT (need >= $MINF frames in $SUBDIR/)"; exit 1; }

log "dataset=$DATASET root=$ROOT subdir=$SUBDIR"
log "span cap $SPAN, L=$L, burn_in=$BURN_IN, min frames $MINF, suffix ${SUFFIX:-<none>}"

# ---------------------------------------------------------------- budget first
TOT_F=0; TOT_R=0; NS=0
for s in $SEQS; do
  d="$ROOT/$s/$SUBDIR"; [ -d "$d" ] || { log "SKIP $s -- no $d"; continue; }
  n=$(ls "$d" 2>/dev/null | grep -icE "\.(png|jpg|jpeg)$")
  use=$(( n < SPAN ? n : SPAN ))
  usable=$(( use - BURN_IN - 8 ))
  runs=$(( usable > 0 ? (usable + L - 1) / L : 0 ))
  TOT_F=$(( TOT_F + use )); TOT_R=$(( TOT_R + runs )); NS=$(( NS + 1 ))
done
# ★ BOTH TERMS SCALE WITH L.  195 MB/run is the L=240 figure, so quote bytes per
# LABELLED FRAME instead; and the teacher does (8+B+L)/L forwards per label, which
# is 1.33 at L=240 but 2.67 at L=48.
log "TOTAL: $NS scenes, $TOT_F frames, $TOT_R runs, ~$(( TOT_F * 81 / 100 / 1000 )) GB, ~$(( TOT_F * (8 + BURN_IN + L) / L * 50 / 1000 / 60 )) min of teacher"
[ "$DRY" = "1" ] && { log "DRY=1, stopping before any GPU work"; exit 0; }

[ -e /dev/nvidiactl ] || { log "FATAL no GPU on this node -- the teacher pass needs one"; exit 1; }
mkdir -p experiments/logs labels

# ------------------------------------------------------------------- teacher
built=0
for s in $SEQS; do
  d="$ROOT/$s/$SUBDIR"
  out="labels/${DATASET}_${s}${SUFFIX}"
  [ -d "$d" ] || continue
  [ -f "$out/index.json" ] && { log "have $out"; continue; }
  # ★ CLAIM THE DIRECTORY ATOMICALLY BEFORE BUILDING.  rebake_all_L48.sh plans
  # its shards from the set of scenes not yet baked, so a shard started later
  # gets a DIFFERENT split from one started earlier -- the "deterministic and
  # disjoint" property only holds if all shards start from the same state.  In
  # practice they do not (a shard dies and is restarted), and then two of them
  # are handed the same scene, write the same run files, and the bank that
  # survives is a mix of two teachers with no error anywhere.  mkdir fails if the
  # directory exists, so it is the claim: whoever wins builds it.
  # A directory with no index.json and no writes for 15 min is a dead claim --
  # a build writes a run every ~6 s -- so reclaim it rather than stranding it.
  if [ -d "$out" ] && [ ! -f "$out/index.json" ]; then
    if [ -z "$(find "$out" -maxdepth 0 -mmin -15 2>/dev/null)" ]; then
      log "reclaiming stale partial $out"; rm -rf "$out"
    else
      log "in flight on another shard, skipping: $out"; continue
    fi
  fi
  mkdir "$out" 2>/dev/null || { log "claimed by another shard: $out"; continue; }
  n=$(ls "$d" 2>/dev/null | grep -icE "\.(png|jpg|jpeg)$")
  use=$(( n < SPAN ? n : SPAN ))
  if [ "$use" -lt "$MINF" ]; then
    log "SKIP $s -- only $use frames, too short for anchor+burn-in+L"
    continue
  fi
  lg="experiments/logs/bank_${DATASET}_${s}.log"
  log "BUILD ${DATASET}_${s}  (span [0,$use) of $n)"
  python -m lingbot_map.train.label_bank \
    --ckpt "$CKPT" --frames "$d" --out "$out" \
    --span 0 "$use" --burn_in "$BURN_IN" --L "$L" \
    ${STRIDE:+--stride "$STRIDE"} ${POSE_ONLY:+--pose_only} \
    ${K_T:+--teacher_interval "$K_T"} >> "$lg" 2>&1
  if [ -f "$out/index.json" ]; then
    log "DONE  ${DATASET}_${s} -> $out ($(du -sh "$out" | cut -f1))"
    built=$(( built + 1 ))
    # CACHE_FLAGS=--uint8 writes the 4x smaller cache (cache_frames.py --uint8);
    # DROP_FRAMES=1 deletes the source images once bank AND cache exist -- for
    # SlowTV chunks (slowtv_chunks.sh) the PNGs are 10 GB per scene of pure
    # transient, re-creatable from the mp4 in 40 s.
    python experiments/cache_frames.py "$s" --root "$ROOT" \
      --frames_subdir "$SUBDIR" --bank_prefix "${DATASET}_" \
      --bank_suffix "$SUFFIX" ${CACHE_FLAGS:-} >> "$lg" 2>&1 \
      || log "WARN cache_frames failed for $s -- the trainer will load resident"
    if [ "${DROP_FRAMES:-0}" = "1" ] && [ -f "$d/_cache_518_14.npy" ]; then
      nc=$(python3 -c "import numpy as np,sys; print(np.load('$d/_cache_518_14.npy', mmap_mode='r').shape[0])" 2>/dev/null || echo 0)
      if [ "$nc" -ge "$use" ]; then
        find "$d" -maxdepth 1 -name '*.png' -delete && log "DROPPED $use PNGs of $s (cache has $nc frames)"
      else
        log "KEEP PNGs of $s -- cache has $nc < $use frames"
      fi
    fi
  else
    log "FAIL  ${DATASET}_${s} -- see $lg"
    tail -5 "$lg" | sed 's/^/        /'
  fi
done
log "ALL DONE $DATASET ($built built, $(ls -d labels/${DATASET}_*/index.json 2>/dev/null | wc -l) banks on disk)"
