#!/usr/bin/env bash
# The GPU half of the v5 verification.  The CPU half already runs:
#
#   experiments/test_state_mixture.py   the draws (K, horizon, branch, dataset)
#   experiments/test_rollout_pool.py    the pool state machine, incl. resume
#   experiments/test_gca_mask.py        the mask predicate at K in {1,4,28}
#
# None of those touch a checkpoint, so none of them can tell you that a mixed-K
# rollout actually attends the cache it thinks it does.  This does: a handful of
# real steps per configuration, on one GPU, against the real bank.
#
# ★ WHAT IT IS ACTUALLY CHECKING is that _check_prefix_consistency stays quiet.
# That predicate (aggregator/stream.py) recomputes the cached keyframe count as
# ceil((t0 - sf) / K) and raises when the live cache disagrees.  It is the ONLY
# guard against a mask built for a rollout that did not happen -- every other
# mismatch shows up as a shape error, this one would train happily on the wrong
# context.  A mixed-K run that ever changed K mid-rollout would trip it.  So a
# clean multi-K smoke IS the correctness statement.
#
#   experiments/smoke_v5.sh                # all four checks, ~15 min
#   STEPS=4 experiments/smoke_v5.sh split  # just one
set -u
cd "$(dirname "$0")/.."

CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
STEPS=${STEPS:-8}
SCENE=${SCENE:-kth_day_10}
WHICH=${*:-split mixed unified resume}

if [ ! -e /dev/nvidiactl ]; then
  echo "FATAL no GPU on this node"; exit 1
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${FREE:-0}" -lt 75000 ]; then
  echo "WARNING only ${FREE} MiB free on GPU 0; a step peaks near 61 GB at S=48."
  echo "        Wait for the card, or lower --S (which changes what is tested)."
fi

SC="$SCENE:data/mcd/$SCENE/frames_10hz:labels/$SCENE:mcd"
[ -f "labels/$SCENE/index.json" ] || { echo "no bank labels/$SCENE"; exit 1; }
mkdir -p experiments/logs experiments/results/_smoke

# ★ --pool MUST BE ABLE TO EXPRESS THE K MIXTURE.  K is assigned per STREAM from
# a deck now (trainer.k_deck), so --pool x P(K) has to be an integer for every K
# or the trainer refuses to start.  The ver5 mixture needs pool 20, which is far
# too much roll-in for a smoke, so the smoke uses a 4-point mixture over 4
# streams: one stream each at K in {1, 4, 12, 28}, deterministically.  That is
# STRONGER coverage than the old --pool 2 with a random draw -- both extremes are
# guaranteed to run, which is the point (_check_prefix_consistency must stay
# quiet at every sampled K).
SMOKE_K=${SMOKE_K:-"1:25,4:25,12:25,28:25"}
COMMON="--ckpt $CKPT --scene $SC --steps $STEPS --pool 4 --S 48 \
  --preset A1PC --lr 1e-6 --warmup 1 --probe_every 0 --probe_max 1 --wandb 0"

run() {  # $1=label  rest=flags
  local label=$1; shift
  local out=experiments/results/_smoke/v5_$label.json
  local log=experiments/logs/smoke_v5_$label.log
  echo "=== $label ==="
  # shellcheck disable=SC2086
  if PYTHONUNBUFFERED=1 LINGBOT_THREADS=16 OMP_NUM_THREADS=16 \
     python -m lingbot_map.train.trainer $COMMON --out "$out" "$@" > "$log" 2>&1; then
    grep -E "^\s+\[[ 0-9]+\] (ID|COR)" "$log" | tail -4 | sed 's/^/    /'
    echo "    OK -> $out"
  else
    echo "    FAILED -- $log"
    tail -20 "$log" | sed 's/^/    /'
    return 1
  fi
}

FAIL=0
for w in $WHICH; do
  case "$w" in
    # 1. the pre-v5 path, untouched.  Every v5 default is off, so this is the
    #    control: if it breaks, the refactor broke something that has nothing to
    #    do with the mixture.
    split)
      run split --K 28 --lam_fresh 1.0 --fresh_mode walk || FAIL=1 ;;

    # 2. mixed K with the split sampler.  Exercises per-stream K and the horizon
    #    reset without changing the branch structure -- so a failure here is the
    #    rollout schedule and not the objective.  --horizons 320 makes a reset
    #    fire within STEPS instead of never.
    mixed)
      run mixed --k_dist "$SMOKE_K" --horizons 320 --lam_fresh 1.0 --fresh_mode walk || FAIL=1 ;;

    # 3. the v5 objective: one branch per step.  Watch the ID/COR column -- both
    #    must appear, and the identity rows must show K1 with age 0.
    unified)
      run unified --k_dist "$SMOKE_K" --horizons 320 960 \
        --sampler unified --p_identity 0.5 --lam_fresh 1.0 --fresh_mode walk || FAIL=1 ;;

    # 4. resume.  This is the one that cost a run before: a restart must CONTINUE
    #    each rollout at its own K, not redraw one over a prefix rolled at the
    #    old one.  Wrong, it trips _check_prefix_consistency on the first step
    #    after the restart -- after the pool has spent minutes rolling in.
    resume)
      rm -f experiments/results/_smoke/_rs.step*.pt
      run resume_a --k_dist "$SMOKE_K" --horizons 960 --sampler unified --p_identity 0.4 \
        --lam_fresh 1.0 --fresh_mode walk \
        --save experiments/results/_smoke/_rs.pt --save_every $((STEPS / 2)) || FAIL=1
      LAST=$(ls -1 experiments/results/_smoke/_rs.step*.pt 2>/dev/null | tail -1)
      if [ -z "${LAST:-}" ]; then
        echo "    no checkpoint written -- cannot test resume"; FAIL=1
      else
        # ★ --steps MUST EXCEED THE CHECKPOINT'S STEP, or the resumed loop is
        # `range(step0, steps)` = empty and the run exits having proved only
        # that the pool could be rebuilt.  The moment that matters is the FIRST
        # STEP AFTER the restore: that is when the masked window is built
        # against the rebuilt prefix and _check_prefix_consistency gets to
        # disagree.  A zero-step resume passes without testing anything.
        run resume_b --k_dist "$SMOKE_K" --horizons 960 --sampler unified --p_identity 0.4 \
          --lam_fresh 1.0 --fresh_mode walk --resume "$LAST" \
          --steps $((STEPS + 3)) || FAIL=1
        grep -E "^\[resume\]" experiments/logs/smoke_v5_resume_b.log | sed 's/^/    /'
        if ! grep -qE "^\s+\[[ 0-9]+\] (ID|COR)" experiments/logs/smoke_v5_resume_b.log; then
          echo "    NO STEP RAN AFTER THE RESUME -- nothing was tested"; FAIL=1
        fi
      fi ;;
    *) echo "unknown check: $w"; FAIL=1 ;;
  esac
done

echo
if [ "$FAIL" = "0" ]; then
  echo "ALL SMOKE CHECKS PASSED -- no prefix-consistency error at any sampled K"
else
  echo "SMOKE FAILED -- see experiments/logs/smoke_v5_*.log"
fi
exit "$FAIL"
