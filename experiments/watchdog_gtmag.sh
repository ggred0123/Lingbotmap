#!/usr/bin/env bash
# Restart gtmag when it is not running.  Cron-parented, so it survives this shell.
#
# WHY THIS EXISTS.  gtmag is gtsup with --lam_mag 1.0 and nothing else changed, so
# it inherits gtsup's memory profile -- and gtsup livelocked for 7.5 h overnight:
# save_every 25 against a ~20-step survival window meant every restart resumed from
# the same checkpoint and banked nothing, 19 times.  gtmag runs save_every 10 so
# each cycle lands at least one checkpoint inside the window.  Checkpoint frequency
# does not enter training, so this does not break the pairing with gtsup -- only
# --lam_mag differs, which is the whole point of the run.
#
# ★ COMPLETION IS THE RESULTS JSON, NOT A CHECKPOINT NAME.  --steps 275 with
# save_every 10 never writes step275.pt (275 is not a multiple of 10), so testing
# for that file would loop forever -- the same shape as the off-by-one that
# restarted a FINISHED c0off 41 times overnight and left its results json empty.
# wall_s is written once, on a clean exit; chain_spine_clean.sh uses the same test.
#
# STOP IT with:  touch experiments/logs/.gtmag.stop
set -u
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
STOP=experiments/logs/.gtmag.stop
COUNT=experiments/logs/.gtmag.restarts
[ -f "$STOP" ] && exit 0

# Bracket so the pattern cannot match this script's own command line -- pgrep -f
# and pkill -f have killed the caller and misreported a live trainer as dead in
# this project, four times between the two nodes.
ps -eo args | grep -q "[l]ingbot_map.train.trainer" && exit 0

python3 -c 'import json,sys; d=json.load(open("experiments/results/train_gtmag.json")); sys.exit(0 if d.get("wall_s") else 1)' 2>/dev/null && exit 0

STEP=$(grep -aoE '^ *\[ *[0-9]+\]' experiments/logs/train_gtmag.log 2>/dev/null \
       | tr -dc '0-9\n' | sort -n | tail -1)
N=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$COUNT"
if [ "$N" -gt 40 ]; then
  echo "$(TZ=Asia/Seoul date '+%F %H:%M') [gtmag@$(hostname -s)] giving up after $N restarts" >> experiments/logs/watchdog.log
  touch "$STOP"; exit 1
fi
echo "$(TZ=Asia/Seoul date '+%F %H:%M') [gtmag@$(hostname -s)] restart #$N from step ${STEP:-0}" >> experiments/logs/watchdog.log
NPROC=1 POOL=20 PROBE_MAX=1 SAVE_EVERY=10 STEPS=275 \
  experiments/launch_gtsup.sh gtmag _gt --lam_mag 1.0 >> experiments/logs/watchdog.log 2>&1
