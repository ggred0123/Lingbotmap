#!/usr/bin/env bash
# Restart c0off when it is not running.  Cron-parented, so it survives this shell.
#
# WHY THIS EXISTS.  c0off is the OFF-policy arm, and it fails differently from c0on.
# Its streams never roll (p_identity=1.0 -> "reachable age 0", every step logs
# age=0), so anon does NOT climb: the floor sits flat at 159-170 GB.  What moves is
# the PEAK, and the peaks line up exactly with checkpoint writes -- 16:18, 16:36,
# 16:53 gave 172.2, 178.5, 179.4 GB -- because serialising 11.4 GB of model plus
# optimizer is a transient allocation on top of the floor.  Against a 200 GB cap
# that leaves ~20 GB, so this arm does not drift into the ceiling the way c0on does;
# it dies only if a spike grazes it.  The exposure is concentrated: probe_every 75
# and save_every 50 coincide at lcm 150, so steps 150/300/450/... take a probe pass
# and an 11.4 GB write in the same window.  Those are the steps to lose, and they
# are the reason this watchdog exists rather than a config cut -- cutting further
# would have to come out of training, and this arm is not the one that needs it.
#
# STOP IT with:  touch experiments/logs/.c0off.stop
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
STOP=experiments/logs/.c0off.stop
COUNT=experiments/logs/.c0off.restarts
[ -f "$STOP" ] && exit 0

# Bracket so the pattern cannot match this script's own command line -- pgrep -f
# and pkill -f have already killed this shell once and misreported a live trainer
# as dead twice in this project.
ps -eo args | grep -q "[l]ingbot_map.train.trainer" && exit 0

STEP=$(grep -aoE '^ *\[ *[0-9]+\]' experiments/logs/train_c0off.log 2>/dev/null \
       | tr -dc '0-9\n' | sort -n | tail -1)
[ "${STEP:-0}" -ge 1250 ] 2>/dev/null && exit 0        # finished, nothing to do

N=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$COUNT"
if [ "$N" -gt 40 ]; then                                # runaway guard
  echo "$(TZ=Asia/Seoul date '+%F %H:%M') giving up after $N restarts" >> experiments/logs/watchdog.log
  touch "$STOP"; exit 1
fi
echo "$(TZ=Asia/Seoul date '+%F %H:%M') restart #$N from step ${STEP:-0}" >> experiments/logs/watchdog.log
NPROC=1 POOL=20 PROBE_MAX=3 FRESH_POOL=5 experiments/launch_clean.sh c0off 1.0 \
  >> experiments/logs/watchdog.log 2>&1
