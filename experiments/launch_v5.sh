#!/usr/bin/env bash
# The docs/self-distill-ver5.md "Minimum Ablation Matrix", one cell per launch.
#
#   Run  Training data   Student policy              Weight-space constraint
#    A   MCD             fixed K=28                  none
#    B   MCD             mixed-K                     none
#    C   MCD + SlowTV    mixed-K                     none
#    D   MCD             mixed-K                     L2-SP
#
#   experiments/launch_v5.sh A            # or B, C, D
#   STEPS=600 experiments/launch_v5.sh B --lr 1e-5
#
# ★ A IS A CONTROL, NOT A RE-RUN OF a3.  A and B differ in exactly two flags --
# --k_dist and --horizons -- so "A vs B measures reduction of keyframe-policy
# specialization" measures that and not the sampler rewrite that came with it.
# Both therefore use the unified sampler.  The historical fixed-K=28 baseline is
# ckpt_train/a3.step300.pt and stays available for reference; it is not this
# comparison's control, because it also differs in loss branch structure.
#
# ★ EVERY CELL USES THE SAME LOSS.  v5: "Keep the current best distillation loss
# and its component weights unchanged for the first mixed-K experiment.  Changing
# the objective, data distribution, and cache policy simultaneously would make
# the source of any improvement impossible to identify."
set -eu
cd "$(dirname "$0")/.."

RUN=${1:?usage: $0 <A|B|C|D> [extra trainer flags]}
shift || true

CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}

# ★ POOL=20 IS NOT A TUNING CHOICE, IT IS THE MIXTURE'S DENOMINATOR.  K is now
# assigned per STREAM from a deck (trainer.k_deck) instead of drawn per rollout,
# because the per-rollout draw's variance is set by the ROLLOUT count: 300 steps
# produced 19 rollouts, an effective sample size of 11, and the realised
# step-weighted mixture came out 44% K=12 with K=8 never drawn at all (v5b drew
# no K=1, K=4 or K=16).  With a deck the share is exactly (streams at K)/pool --
# zero variance -- but that requires pool*P(K) to be integral for every K, i.e. a
# multiple of lcm(4, 10) = 20 for the ver5 mixture.  The trainer refuses to start
# otherwise and names the smallest pool that works.
POOL=${POOL:-20}

# ★ STEPS=1250 IS SET BY THE HORIZON GRID, not by taste.  A stream gets
# steps*(1-p_identity)/(POOL/NPROC) turns and advances S=48 frames per turn, so
# the deepest reachable age is
#     1250 * 0.65 / 10 * 48 = 3900   >= max(--horizons) = 3840
# Below ~1231 the 3840 rollouts never reach their horizon and simply stop
# mid-walk -- which is what happened at STEPS=300 (max age 1008, so 1920 and
# 3840 were inert for 10 of 19 rollouts).  The trainer now warns when a horizon
# is unreachable; this is the value that keeps it quiet.
STEPS=${STEPS:-1250}
THREADS=${THREADS:-16}
NPROC=${NPROC:-2}
PRESET=${PRESET:-A1PC}
# ckpt is 11.5 GB each (weights + optimiser); 1250/125 = 10 of them = 115 GB.
SAVE_EVERY=${SAVE_EVERY:-125}
PROBE_EVERY=${PROBE_EVERY:-125}
# ★ TAG EXISTS SO A RERUN DOES NOT RESUME THE OLD RUN.  The auto-resume below
# picks the highest ckpt_train/<NAME>.step*.pt, so relaunching C after the
# per-rollout-K run would restore step-300 weights AND its 9-stream pool -- a
# different mixture wearing the new run's name, and it would silently overwrite
# the checkpoints docs/v5c-training-and-review.md is written against.
# TAG=2 -> v5c2.
TAG=${TAG:-}

MCD_SCENES="kth_day_10 kth_night_01 kth_night_04 kth_night_05 \
tuhh_day_02 tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 tuhh_night_09"

# ── the cell ─────────────────────────────────────────────────────────────────
MIXED="--k_dist v5 --horizons 320 960 1920 3840"
FIXED="--k_dist fixed --K 28 --horizons 0"
case "$RUN" in
  A) POLICY="$FIXED"; USE_SLOWTV=0; L2SP=0    ;;
  B) POLICY="$MIXED"; USE_SLOWTV=0; L2SP=0    ;;
  C) POLICY="$MIXED"; USE_SLOWTV=1; L2SP=0    ;;
  # l2sp1e2 and l2sp3e3 both scored 1.894-1.896 on the MCD K=28 per-window
  # metric against a base of 2.386, so the coefficient is not the sensitive
  # knob; 1e-2 is the stronger of the two that was measured.
  D) POLICY="$MIXED"; USE_SLOWTV=0; L2SP=${L2SP:-1e-2} ;;
  *) echo "RUN must be A, B, C or D"; exit 1 ;;
esac

# ── scenes ───────────────────────────────────────────────────────────────────
# ★ ORDERED IN BLOCKS OF $NPROC, NOT ALTERNATING.  The trainer partitions scenes
# as specs[rank::world], so a strictly alternating mcd,slowtv,mcd,slowtv list
# hands rank 0 every MCD scene and rank 1 every SlowTV scene -- each rank then
# holds one corpus and --dataset_weights cannot be honoured on either.  Blocks of
# `world` invert that: rank r takes one scene from each block, so every rank sees
# both corpora.  The trainer warns if this is got wrong; the ordering here is
# what stops the warning from firing.
SC=$(python3 - "$NPROC" "$USE_SLOWTV" $MCD_SCENES <<'PY'
import os, sys
world = max(1, int(sys.argv[1])); use_slowtv = sys.argv[2] == "1"
mcd = [(s, f"data/mcd/{s}/frames_10hz", f"labels/{s}", "mcd") for s in sys.argv[3:]]
slowtv = []
if use_slowtv:
    for d in sorted(os.listdir("labels")):
        if d.startswith("slowtv_") and os.path.exists(f"labels/{d}/index.json"):
            seq = d[len("slowtv_"):]
            slowtv.append((d, f"data/slow_tv/{seq}/frames_10hz", f"labels/{d}", "slowtv"))
    if not slowtv:
        sys.exit("no SlowTV banks under labels/slowtv_* -- "
                 "run experiments/build_banks_slowtv.sh first")
out, i = [], 0
while i * world < max(len(mcd), len(slowtv)):
    lo, hi = i * world, (i + 1) * world
    out += mcd[lo:hi] + slowtv[lo:hi]
    i += 1
for name, frames, bank, ds in out:
    if not os.path.exists(f"{bank}/index.json"):
        sys.exit(f"missing bank: {bank}")
    if not os.path.isdir(frames):
        sys.exit(f"missing frames: {frames}")
    print(f"{name}:{frames}:{bank}:{ds}")
PY
) || { echo "$SC"; exit 1; }
SC=$(echo "$SC" | tr '\n' ' ')

DSW=""
[ "$USE_SLOWTV" = "1" ] && DSW="--dataset_weights ${DATASET_WEIGHTS:-mcd:50,slowtv:50}"

# GT probe is MCD-only (SlowTV has no poses).  Metric-only either way -- GT never
# enters the loss -- so run C simply loses the probe and is scored on MCD holdout
# by the eval grid instead.
GT=""
[ "$USE_SLOWTV" = "0" ] && GT="--gt_calib data/mcd/calib/hhs_calib.yaml --gt_sensor d455b_color"

NAME=v5${RUN,,}${TAG}
RESUME=""
LAST=$(ls -1 ckpt_train/${NAME}.step*.pt 2>/dev/null \
       | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
[ -n "${LAST:-}" ] && { RESUME="--resume $LAST"; echo "resuming from $LAST"; }

LOG=experiments/logs/train_$NAME.log
mkdir -p experiments/logs ckpt_train
echo "=== run $RUN -> $NAME ==="
echo "  policy   $POLICY"
echo "  l2sp     $L2SP"
echo "  scenes   $(echo "$SC" | wc -w)  ${DSW:-(single corpus)}"
echo "  sampler  unified p_identity=0.35 lam_fresh=1.0 fresh_mode=walk"
echo "  pool     $POOL total ($((POOL / NPROC))/rank)   steps $STEPS   -> reachable age \
$(python3 -c "print(int($STEPS*0.65/($POOL/$NPROC)*48))")"
# ★ CHECK THE DECK HERE, NOT 13 MINUTES IN.  The trainer refuses a --pool that
# cannot express --k_dist, but only after it has built the model and mapped every
# scene.  Under tmux that failure is invisible until someone reads the log, so
# resolve it in the launcher where DRY=1 can see it too.
python3 - "$POOL" "$NPROC" $POLICY <<'PYCHK' || exit 1
import os, sys
sys.path.insert(0, os.getcwd())
from lingbot_map.train.trainer import k_deck, parse_k_dist, K_DIST_V5
pool, world, rest = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
spec = rest[rest.index("--k_dist") + 1] if "--k_dist" in rest else "fixed"
kd = parse_k_dist(",".join(f"{k}:{w}" for k, w in K_DIST_V5) if spec == "v5" else spec)
if pool % world:
    sys.exit(f"[deck] --pool {pool} is not divisible by NPROC {world}")
deck = k_deck(kd, pool, 28)                       # raises SystemExit with the fix
if kd:
    c = {k: deck.count(k) for k in sorted(set(deck))}
    print("  deck     " + "  ".join(f"K{k}x{v} ({v / pool:.0%})" for k, v in c.items()))
    print(f"  rank0    {sorted(deck[0::world])}")
    print(f"  rank1    {sorted(deck[1::world])}" if world > 1 else "")
PYCHK

[ "${DRY:-0}" = "1" ] && { echo "DRY=1, not launching"; exit 0; }

tmux kill-session -t "train_$NAME" 2>/dev/null || true
tmux new-session -d -s "train_$NAME" \
  "PYTHONUNBUFFERED=1 LINGBOT_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
   OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
   torchrun --nproc_per_node=$NPROC --master_port=\$((29500 + RANDOM % 1000)) \
     -m lingbot_map.train.trainer \
     --ckpt $CKPT --scene $SC \
     --steps $STEPS --pool $POOL --S 48 \
     --preset $PRESET --l2sp $L2SP \
     $POLICY $DSW \
     --sampler unified --p_identity 0.35 --lam_fresh 1.0 --fresh_mode walk \
     --lr 1e-5 --warmup 50 --wd 0.05 --clip 1.0 \
     --probe_max 6 --probe_every $PROBE_EVERY --save_every $SAVE_EVERY \
     $GT \
     --wandb 1 --wandb_name $NAME --wandb_group v5 --wandb_tags v5 $RUN ddp \
     --save ckpt_train/${NAME}.pt --out experiments/results/train_$NAME.json \
     $RESUME $* 2>&1 | tee -a $LOG"
sleep 3; tmux ls | sed 's/^/  /'; echo "log: $LOG"
