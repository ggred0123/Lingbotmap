#!/usr/bin/env bash
# Restart c0on when it is not running.  Cron-parented, so it survives this shell.
#
# WHY THIS EXISTS.  c0on is the on-policy arm: its streams roll, so anon grows with
# rollout depth (177 GB at step 100 -> 194.6 GB at step 160) until it trips the
# container's 200 GB cap and the kernel SIGKILLs it -- four times so far, each with
# no traceback, which is what a cgroup OOM looks like from inside.  The identity arm
# does not roll and sits flat at ~162 GB, so this is structural, not a bug: it is
# the difference the 2x2 exists to measure.  Until the box is bigger, the arm has to
# grind forward in ~110-step chunks.  --save_every 50 caps each loss at 50 steps and
# launch_clean.sh re-attaches the newest checkpoint, so restarting is the whole fix.
#
# SAVE_EVERY=25, NOT the default 50.  With 50 this watchdog livelocked: resume at
# step 150, survive ~35 steps, OOM at ~185, and the next write was at 200 -- so
# nothing after 150 was ever saved and seven restarts over two hours made zero net
# progress (watchdog.log 16:53-18:55, all "from step 185", ckpts still 50/100/150).
# "at most save_every steps are lost" holds only while the interval is SHORTER than
# the survival window; here 50 > 35 and the arm could not advance at all.  25 puts
# the next write at 175, inside the window.  Shorter still (10) would be safer per
# cycle but each write is an 11.4 GB serialisation spike against a ~20 GB margin.
#
# STOP IT with:  touch experiments/logs/.c0on.stop
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
STOP=experiments/logs/.c0on.stop
COUNT=experiments/logs/.c0on.restarts
[ -f "$STOP" ] && exit 0

# Bracket so the pattern cannot match this script's own command line -- pgrep -f
# and pkill -f have already killed this shell once and misreported a live trainer
# as dead twice in this project.
ps -eo args | grep -q "[l]ingbot_map.train.trainer" && exit 0

STEP=$(grep -aoE '^ *\[ *[0-9]+\]' experiments/logs/train_c0on.log 2>/dev/null \
       | tr -dc '0-9\n' | sort -n | tail -1)
# ★ ASK FOR THE CHECKPOINT, NOT THE LOG.  The trainer logs steps 0-indexed, so a
# finished 1250-step run's last line is 1249 and "STEP >= 1250" is never true --
# but it names the file (step+1), so step1250.pt exists exactly when the run is
# done.  agent-B's c0off watchdog had this test and restarted a COMPLETED run 41
# times, each restart resuming from step1250.pt with nothing left to do and
# writing an empty train_c0off.json over the real one.  The weights survived; the
# training history did not.
[ -f ckpt_train/c0on.step1250.pt ] && exit 0

N=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$COUNT"
if [ "$N" -gt 40 ]; then                                # runaway guard
  echo "$(TZ=Asia/Seoul date '+%F %H:%M') [c0on@$(hostname -s)] giving up after $N restarts" >> experiments/logs/watchdog.log
  touch "$STOP"; exit 1
fi
echo "$(TZ=Asia/Seoul date '+%F %H:%M') [c0on@$(hostname -s)] restart #$N from step ${STEP:-0}" >> experiments/logs/watchdog.log
NPROC=1 POOL=20 PROBE_MAX=1 FRESH_POOL=5 SAVE_EVERY=25 experiments/launch_clean.sh c0on 0.0 \
  >> experiments/logs/watchdog.log 2>&1
