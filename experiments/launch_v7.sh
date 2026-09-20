#!/usr/bin/env bash
# v7: v6i's policy, unchanged, plus L_long.  docs/long_supervision_plan.md Stage 7.
#
# ★ EVERYTHING EXCEPT THE LONG TERM IS PINNED TO v6i.  The control cell for this
# experiment is v6i itself, so corpus weights, K deck, horizons, p_identity,
# lam_fresh, pool and seed all have to be the v6i values rather than
# launch_v6.sh's defaults -- those differ (mcd:30 / unrealstereo4k:10 /
# paralleldomain4d:15 against v6i's 10 / 15 / 30) and a corpus change would be
# confounded with the term under test.
#
#   CELL=D experiments/launch_v7.sh              # rot + dir + scale
#   CELL=B experiments/launch_v7.sh              # angular only
#   CELL=C experiments/launch_v7.sh              # scale only
#   CELL=A experiments/launch_v7.sh              # control (lam_long 0) -- a v6i re-run
#   LAM_LONG=0.3 CELL=D experiments/launch_v7.sh
set -eu
cd "$(dirname "$0")/.."

CELL=${CELL:-D}
case "$CELL" in
  A) TERMS=""            ; LAM_LONG=${LAM_LONG:-0} ;;
  B) TERMS="rot,dir"     ; LAM_LONG=${LAM_LONG:-1.0} ;;
  C) TERMS="scale"       ; LAM_LONG=${LAM_LONG:-1.0} ;;
  D) TERMS="rot,dir,scale"; LAM_LONG=${LAM_LONG:-1.0} ;;
  *) echo "CELL must be A|B|C|D"; exit 1 ;;
esac

export NAME=${NAME:-v7$(echo "$CELL" | tr 'A-Z' 'a-z')}
export STEPS=${STEPS:-1250}
export NPROC=${NPROC:-2}
export POOL=${POOL:-20}
export P_IDENTITY=${P_IDENTITY:-0.35}
export HORIZONS=${HORIZONS:-320 960 1920 3840}
export BANK_SUFFIX=${BANK_SUFFIX:-}          # ★ MUST stay empty: _L48 runs are
                                             # L=48, one window per run, so no
                                             # long pair can exist and the term
                                             # would be silently masked out.
export DATASET_WEIGHTS=${DATASET_WEIGHTS:-mcd:10,slowtv:10,dl3dv:10,scannet:10,replica:5,dynamicreplica:10,unrealstereo4k:15,paralleldomain4d:30}

LONG_ARGS=""
if [ "$LAM_LONG" != "0" ]; then
  LONG_ARGS="--lam_long $LAM_LONG --long_deltas ${LONG_DELTAS:-48 96 192} --long_terms $TERMS"
  [ -n "${LONG_LAM_DELTA:-}" ] && LONG_ARGS="$LONG_ARGS --long_lam_delta $LONG_LAM_DELTA"
  [ -n "${LONG_TAU:-}" ]       && LONG_ARGS="$LONG_ARGS --long_tau $LONG_TAU"
  # ── the stitched long-target bank ────────────────────────────────────────
  # ★ THIS IS WHAT THE DESIGN ASKED FOR AND v7d DID NOT USE.  v7d took its long
  # targets from the LOCAL L=240 bank, where a rung only fits at deep window
  # offsets -- Delta=192 only at offset 192, teacher depth 272 -- so Delta and
  # teacher depth were entangled and the scale bias tracked depth, not Delta
  # (-0.082 at depth 128, -0.263 at 272).  The stitched track takes each L96
  # run's FIRST 48 frames, so every window sits at depth 80-127: the confound is
  # gone by construction, and Delta=319 becomes reachable at all.
  [ -n "${LONG_BANK_SUFFIX:-}" ] && LONG_ARGS="$LONG_ARGS --long_bank_suffix $LONG_BANK_SUFFIX"
  [ -n "${LONG_ALT_DELTAS:-}" ]  && LONG_ARGS="$LONG_ARGS --long_alt_deltas $LONG_ALT_DELTAS"
  [ -n "${LONG_ALT_TERMS:-}" ]   && LONG_ARGS="$LONG_ARGS --long_alt_terms $LONG_ALT_TERMS"
  [ -n "${LONG_MAX_DEPTH:-}" ]   && LONG_ARGS="$LONG_ARGS --long_max_teacher_depth $LONG_MAX_DEPTH"
  [ -n "${LONG_VLOCAL:-}" ]      && LONG_ARGS="$LONG_ARGS --long_vlocal $LONG_VLOCAL"
  [ -n "${LONG_GAUGE_NORM:-}" ]  && LONG_ARGS="$LONG_ARGS --long_scale_gauge_norm $LONG_GAUGE_NORM"
  [ -n "${LONG_WEIGHT_NORM:-}" ] && LONG_ARGS="$LONG_ARGS --long_weight_norm $LONG_WEIGHT_NORM"
fi

# ★ THE GT PROBE, ON BY DEFAULT.  v6i and v7d both ran with it OFF, so every
# mid-training judgement rested on the teacher-imitation probe -- which improved
# markedly while the GT metric did not.  Metric only: GT never enters the loss.
# MCD windows carry it; the rest of the corpus simply has none.
GT_CALIB=${GT_CALIB:-data/mcd/calib/hhs_calib.yaml}
[ -n "$GT_CALIB" ] && LONG_ARGS="$LONG_ARGS --gt_calib $GT_CALIB"

echo "=== v7 cell $CELL -> $NAME ==="
echo "  long     lam=$LAM_LONG terms='${TERMS:-none}' deltas='${LONG_DELTAS:-48 96 192}'"
echo "  source   local bank + alt='${LONG_BANK_SUFFIX:-none}' for delta='${LONG_ALT_DELTAS:-none}'  max_depth=${LONG_MAX_DEPTH:-none}"
echo "  control  v6i (same policy, lam_long 0)"
exec env WANDB_GROUP=v7 experiments/launch_v6.sh $LONG_ARGS \
  --wandb_group v7 --wandb_tags v7 long "cell$CELL" "$@"
