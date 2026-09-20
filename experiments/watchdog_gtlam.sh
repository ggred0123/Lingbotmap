#!/usr/bin/env bash
# gtlam = gtsup with the loss weights term_grad_probe says equalise GRADIENT
# contribution, instead of A1PC's, which were set from loss VALUES.
#
# What the probe measured at theta_0, FAR (t0=5248), preset A1PC:
#     term      loss share   grad share
#     L_rot        14.1%       18.2%
#     L_dir        41.2%        7.6%     loud but weak
#     L_motion     35.7%       23.9%
#     L_depth       9.0%       50.3%     <- half the gradient, a twelfth of the loss
# and the terms partly cancel: ||sum lam_i g_i|| is 62% of sum ||lam_i g_i||,
# with L_rot on a negative cosine against ALL four others.
#
# L_depth sends NOTHING to camera_head, so half the update was flowing into the
# depth path while rpe_rot never moved on any benchmark and AUC_03 fell on all of
# them.  That is the trade we measured -- local alignment bought with global
# precision -- and this is the knob for it.
#
# lam* (equal gradient contribution, L_dir=1):
#     L_rot 3.31   L_dir 1   L_mag 0.976   L_motion 0.151   L_depth 0.136
# scaled to keep lam_dir at A1PC's 1.9, and lam_mag held at 0 because A1 zeroes it
# on purpose (L_mag fits away the pose scale that L_motion-depth exists to teach):
#     --lam_rot 6.29 --lam_dir 1.9 --lam_mag 0.0 --lam_motion 0.287 --lam_depth 0.258
#
# Everything else matches gtsup exactly so the two are comparable point for point.
# STOP IT with:  touch experiments/logs/.gtlam.stop
cd /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
TARGET=${TARGET:-275}
STOP=experiments/logs/.gtlam.stop
COUNT=experiments/logs/.gtlam.restarts
WLOG=experiments/logs/watchdog_gtlam.log
[ -f "$STOP" ] && exit 0
ps -eo args | grep -q "[l]ingbot_map.train.trainer" && exit 0

ckpt_step(){ ls ckpt_train/gtlam.step*.pt 2>/dev/null | sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
log_step(){ grep -aoE '^ *\[ *[0-9]+\]' experiments/logs/train_gtlam.log 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1; }
CK=$(ckpt_step); CK=${CK:-0}; LS=$(log_step); LS=${LS:-0}
if [ "$LS" -ge "$TARGET" ] 2>/dev/null; then
  echo "$(TZ=Asia/Seoul date '+%F %H:%M') reached step $LS >= $TARGET -- stopping" >> "$WLOG"
  touch "$STOP"; exit 0
fi
SURV=$(( LS - CK ))
if [ "$SURV" -gt 0 ] 2>/dev/null; then SE=$(( SURV / 2 / 5 * 5 )); [ "$SE" -lt 5 ] && SE=5; [ "$SE" -gt 25 ] && SE=25
else SE=25; fi
N=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 )); echo "$N" > "$COUNT"
if [ "$N" -gt 25 ]; then echo "$(TZ=Asia/Seoul date '+%F %H:%M') giving up after $N restarts" >> "$WLOG"; touch "$STOP"; exit 1; fi
echo "$(TZ=Asia/Seoul date '+%F %H:%M') restart #$N  ckpt=$CK log=$LS survival=$SURV -> save_every=$SE" >> "$WLOG"
NPROC=1 POOL=20 PROBE_MAX=1 SAVE_EVERY=$SE experiments/launch_gtsup.sh gtlam _gt \
  --lam_rot 6.29 --lam_dir 1.9 --lam_mag 0.0 --lam_motion 0.287 --lam_depth 0.258 >> "$WLOG" 2>&1
