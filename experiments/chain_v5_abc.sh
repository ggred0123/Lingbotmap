#!/usr/bin/env bash
# Run the v5 ablation cells A -> B -> C, one at a time, each on both GPUs.
#
# ★ STEPS=300, NOT the launcher's default 1200.  Every reference point we compare
# against is a step-300 checkpoint (a3.step300, a4.step300, a2pc.step300) and the
# plan's own matrix asks for "matched optimization steps".  300 also keeps the
# three cells inside ~18 h instead of ~72 h.  launch_v5.sh auto-resumes from the
# highest ckpt_train/v5<c>.step*.pt, so extending any cell later is one command.
#
# ★ POOL=19 FOR C ONLY.  C carries 19 scenes and the launcher defaults to 10, so
# 9 scenes would never be visited -- the "corpus silently shrinks" failure the
# pool warning exists for (ver5-implementation.md section 12).  A and B have 10
# scenes and keep the default.
#
#   tmux new-session -d -s chain_v5 experiments/chain_v5_abc.sh
set -u
cd "$(dirname "$0")/.."
LOG=experiments/logs/chain_v5.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] [chain] $*"; }

# ★ STEPS AND POOL ARE NOW COUPLED.  K is assigned per stream from a deck, so
# POOL must be a multiple of the mixture's denominator (20 for ver5), and the
# deepest reachable age is steps*(1-p_identity)/(POOL/NPROC)*S -- 1250 steps is
# what puts max(--horizons)=3840 inside it.  The old STEPS=300 / POOL=19 pair
# did neither: it drew 19 rollouts (effective sample size 11, so 44% K=12 and no
# K=8 at all) and reached age 1008, leaving the 1920 and 3840 horizons inert.
# See docs/v5c-training-and-review.md B2/B3.
STEPS=${STEPS:-1250}

for RUN in A B C; do
  NAME=v5${RUN,,}
  CK=ckpt_train/${NAME}.step${STEPS}.pt
  if [ -f "$CK" ]; then log "SKIP $RUN -- $CK already exists"; continue; fi

  # POOL is the same for every cell now -- it is the mixture's denominator, not
  # a per-cell scene-count decision, and A/B/C are only comparable at one value.
  POOL_ARG=""

  log "launching $RUN (steps=$STEPS ${POOL_ARG:-pool=default})"
  env $POOL_ARG STEPS=$STEPS experiments/launch_v5.sh "$RUN" || {
    log "LAUNCH FAILED for $RUN -- stopping the chain"; exit 1; }

  sleep 30
  while tmux has-session -t "train_$NAME" 2>/dev/null; do sleep 60; done
  log "train_$NAME session ended"

  if [ ! -f "$CK" ]; then
    log "FAILED $RUN -- no $CK on disk.  Last lines of its log:"
    tail -20 "experiments/logs/train_${NAME}.log" | sed 's/^/        /'
    log "stopping the chain so B/C do not run against a broken A"
    exit 1
  fi
  log "DONE $RUN -> $CK  ($(du -h "$CK" | cut -f1))"
done
log "CHAIN COMPLETE -- A, B, C all at step $STEPS"
