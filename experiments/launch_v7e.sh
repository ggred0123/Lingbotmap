#!/usr/bin/env bash
# v7e: the cell-D long loss the design actually specifies -- rot/dir from the
# in-run L=240 bank, scale from the stitched L96s48 track, ladder out to 319.
# docs/long_supervision_design.md sections 5, 6.4, 10 and 11.
#
# ★ WHAT SEPARATES THIS FROM v7d.  v7d ran with LONG_BANK_SUFFIX unset, so every
# term -- scale included -- came from the local L=240 bank.  There a rung only
# fits at deep window offsets (Delta=192 only at offset 192, teacher depth 272),
# so Delta and teacher depth were entangled and the scale target carried the
# teacher's depth-driven bias (-0.082 at depth 128 against -0.263 at 272) rather
# than a trajectory relation.  The stitched track re-anchors every 48 frames, so
# every window sits at teacher depth 80-127 and Delta=319 exists at all.
#
# ★ THE CELL LETTERS IN launch_v7.sh PREDATE THE SOURCE SPLIT.  Its CELL=D means
# "--long_terms rot,dir,scale", i.e. all three from the LOCAL bank -- exactly
# what the GT audit says is wrong for scale (in-run bias -0.245 against the
# stitched +0.07).  The design's cell D is
#     --long_terms rot,dir   +   --long_alt_terms scale
# which is CELL=B here PLUS the alt bank.  Passing CELL=D would silently re-run
# v7d under a new name.
#
# ★ LAM_LONG IS NOT launch_v7.sh's 1.0.  Section 10 measured lam=2 at steps
# 30/35: the long term took 98.9% / 99.9% of the gradient, and the lam giving it
# a 25% share was 0.101 / 0.032.  Starting at 1.0 buries local A1PC and breaks
# the rpe_trans / K28 gains the experiment is supposed to PRESERVE.  0.05 sits in
# that measured band -- but section 10's procedure is to re-derive it from the
# gradient share at step 30-50 against the local-only control (v6i: |g| 19.49 at
# step 30, 8.81 at step 35) and rescale.  Do not trust this number past the probe.
#
#   experiments/launch_v7e.sh                          # watchdog, 1250 steps
#   STEPS=60 NAME=v7e_probe experiments/launch_v7e.sh  # gradient-share probe only
#   LAM_LONG=0.03 experiments/launch_v7e.sh            # after the probe
set -u
cd "$(dirname "$0")/.."

export NAME=${NAME:-v7e}
export STEPS=${STEPS:-1250}
export NPROC=${NPROC:-2}
export LAUNCH=${LAUNCH:-experiments/launch_v7.sh}

# ── the long term ────────────────────────────────────────────────────────────
export CELL=${CELL:-B}                                  # -> --long_terms rot,dir
export LAM_LONG=${LAM_LONG:-0.05}
export LONG_DELTAS="${LONG_DELTAS:-48 96 192}"           # in-run rungs, L=240 bank

# ★ THE ONE FLAG THAT DECIDES WHETHER ANY OF THIS REACHES THE LOSS.  Empty (the
# launch_v7.sh default) leaves Scene.long_bank None, Delta=319 is never built,
# and scale silently falls back to the depth-biased in-run target -- no error,
# no warning, just v7d again.  184 scenes have a track; dl3dv and replica have
# none by construction (a Delta=319 pair needs 80+319+48 = 447 frames).
export LONG_BANK_SUFFIX=${LONG_BANK_SUFFIX:-_long}
export LONG_ALT_DELTAS="${LONG_ALT_DELTAS:-48 96 192 319}"
export LONG_ALT_TERMS=${LONG_ALT_TERMS:-scale}

# ── pinned to v6i, which IS the control cell ─────────────────────────────────
# ★ BANK_SUFFIX MUST BE EXPLICITLY EMPTY, NOT UNSET.  watchdog_train.sh checks
# "${BANK_SUFFIX-_L48}" -- '-' so that an intentionally empty value survives --
# and the _L48 bake is L=48, one window per run, so no long pair can exist and
# the entire term would be masked out without a single error line.
export BANK_SUFFIX=""
export POOL=${POOL:-20}
export P_IDENTITY=${P_IDENTITY:-0.35}

# ★ THE HORIZON DECK IS NOT THE DELTA RANGE (section 11).  W=320 is the ladder's
# reach; the rollout horizon is separate.  With --horizons 320 alone the stream
# stops at t=368 and only 17 partial Delta=319 pairs ever form -- the rung dies
# quietly.  The full deck keeps 3/4 of streams able to supply it, with all 48
# pairs per window alive from t >= 416.
export HORIZONS="${HORIZONS:-320 960 1920 3840}"

cat <<EOF
=== v7e -> $NAME ===
  long      lam=$LAM_LONG  local terms=rot,dir deltas='$LONG_DELTAS'
            alt bank='$LONG_BANK_SUFFIX' terms=$LONG_ALT_TERMS deltas='$LONG_ALT_DELTAS'
  policy    v6i pinned: pool=$POOL p_identity=$P_IDENTITY horizons='$HORIZONS' bank='(L=240)'
  control   ckpt_train/v6i.pt (already trained, lam_long 0)
  steps     $STEPS on $NPROC ranks
  log       experiments/logs/train_$NAME.log
EOF

exec experiments/watchdog_train.sh
