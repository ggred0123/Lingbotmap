#!/usr/bin/env bash
# Grind teasup forward to TARGET, restarting it after each cgroup OOM.
#
# teasup is gtsup's control: same loss, same p_identity 0.35 policy, same ten MCD
# scenes, and the ONLY difference is that labels come from the teacher bank
# instead of labels/<scene>_gt.  Scored against gtsup on Oxford, the MCD->Oxford
# domain shift is common to both arms and cancels in the difference.
#
# ★ save_every IS DERIVED, NOT FIXED.  Twice now a fixed interval has livelocked
# this exact setup: c0on wrote every 50 steps but survived 35 (7 restarts, 2 h,
# zero progress), and gtsup wrote every 25 but survived 20 at depth (19 restarts,
# 7.5 h, zero progress).  The window is not constant -- anon climbs with rollout
# age, age is restored from the checkpoint, so each resume starts nearer the cap
# and survives less than the last.  A fixed interval is therefore guaranteed to
# fall outside the window eventually.  This measures the last cycle's survival
# (max step reached minus the checkpoint it resumed from) and writes at half of
# it, so the interval shrinks as the window does.
#
# STOP IT with:  touch experiments/logs/.teasup.stop
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
TARGET=${TARGET:-275}          # match gtsup's depth so the two are scored at the same steps
STOP=experiments/logs/.teasup.stop
COUNT=experiments/logs/.teasup.restarts
WLOG=experiments/logs/watchdog_teasup.log
[ -f "$STOP" ] && exit 0
# Bracket: pgrep -f has matched its own command line and reported a dead trainer
# alive here, and pkill -f has killed the calling shell.
ps -eo args | grep -q "[l]ingbot_map.train.trainer" && exit 0

ckpt_step(){ ls ckpt_train/teasup.step*.pt 2>/dev/null \
  | sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
log_step(){ grep -aoE '^ *\[ *[0-9]+\]' experiments/logs/train_teasup.log 2>/dev/null \
  | tr -dc '0-9\n' | sort -n | tail -1; }

CK=$(ckpt_step); CK=${CK:-0}
LS=$(log_step);  LS=${LS:-0}
if [ "$LS" -ge "$TARGET" ] 2>/dev/null; then
  echo "$(TZ=Asia/Seoul date '+%F %H:%M') reached step $LS >= $TARGET -- stopping" >> "$WLOG"
  touch "$STOP"; exit 0
fi

# Survival of the cycle that just died: how far past its resume point it got.
SURV=$(( LS - CK ))
if [ "$SURV" -gt 0 ] 2>/dev/null; then
  SE=$(( SURV / 2 / 5 * 5 ))
  [ "$SE" -lt 5 ] && SE=5
  [ "$SE" -gt 25 ] && SE=25
else
  SE=25                                   # first launch, no history yet
fi

N=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$COUNT"
if [ "$N" -gt 25 ]; then
  echo "$(TZ=Asia/Seoul date '+%F %H:%M') giving up after $N restarts" >> "$WLOG"
  touch "$STOP"; exit 1
fi
echo "$(TZ=Asia/Seoul date '+%F %H:%M') restart #$N  ckpt=$CK log=$LS survival=$SURV -> save_every=$SE" >> "$WLOG"
NPROC=1 POOL=20 PROBE_MAX=1 SAVE_EVERY=$SE experiments/launch_gtsup.sh teasup "" >> "$WLOG" 2>&1
