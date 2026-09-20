#!/usr/bin/env bash
# Grind gtsup forward to TARGET, restarting it after each cgroup OOM.
#
# gtsup is the GT-label arm: p_identity 0.35 means 65% of steps roll the student,
# so anon climbs with rollout depth (163.9 GB at step 25 -> 187.6 GB at step 75,
# ~1.3 GB/min) until it hits this container's 200 GB cap and the kernel SIGKILLs
# it -- no traceback, nothing in the log, which is what a cgroup OOM looks like
# from inside.  About 80 steps per cycle at probe_max 3, ~130 at probe_max 1.
#
# ★ SAVE_EVERY MUST STAY BELOW THE SURVIVAL WINDOW.  The c0on watchdog livelocked
# for seven restarts and two hours because save_every was 50 while the arm only
# survived 35 steps: nothing after the resume point was ever written, so every
# cycle replayed the same 35 steps.  25 is inside gtsup's window with room.
#
# ★ PROBE_MAX 1 FROM STEP 75 ON.  It buys ~22 GB and cannot touch the trained
# weights -- run_probes sits behind the is_main gate and appears nowhere in the
# loss, backward or optimizer path.  What it does change is the probe SERIES:
# three scored windows before step 75, one after.  The series is in-domain and
# has already been shown to move independently of the benchmark, so it is the
# cheapest thing here to spend.  Noted so nobody reads the discontinuity as data.
#
# STOP IT with:  touch experiments/logs/.gtsup.stop
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
TARGET=${TARGET:-300}
STOP=experiments/logs/.gtsup.stop
COUNT=experiments/logs/.gtsup.restarts
[ -f "$STOP" ] && exit 0
# Bracket: pgrep -f matched its own command line here and reported a dead trainer
# alive, and pkill -f killed the calling shell twice.
ps -eo args | grep -q "[l]ingbot_map.train.trainer" && exit 0

STEP=$(grep -aoE '^ *\[ *[0-9]+\]' experiments/logs/train_gtsup.log 2>/dev/null \
       | tr -dc '0-9\n' | sort -n | tail -1)
if [ "${STEP:-0}" -ge "$TARGET" ] 2>/dev/null; then
  echo "$(TZ=Asia/Seoul date '+%F %H:%M') reached step $STEP >= $TARGET -- stopping" \
    >> experiments/logs/watchdog_gtsup.log
  touch "$STOP"; exit 0
fi

N=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$COUNT"
if [ "$N" -gt 20 ]; then
  echo "$(TZ=Asia/Seoul date '+%F %H:%M') giving up after $N restarts" >> experiments/logs/watchdog_gtsup.log
  touch "$STOP"; exit 1
fi
echo "$(TZ=Asia/Seoul date '+%F %H:%M') restart #$N from step ${STEP:-0}" >> experiments/logs/watchdog_gtsup.log
NPROC=1 POOL=20 PROBE_MAX=1 SAVE_EVERY=25 experiments/launch_gtsup.sh gtsup _gt \
  >> experiments/logs/watchdog_gtsup.log 2>&1
