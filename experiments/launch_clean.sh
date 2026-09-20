#!/usr/bin/env bash
# One arm of the spine 2x2, re-run on a corpus that is actually video.
#
#   experiments/launch_clean.sh c0off 1.0    # identity branch only  (off-policy)
#   experiments/launch_clean.sh c0on  0.0    # correction branch only (on-policy)
#
# ★ WHY THE ORIGINAL 2x2 DOES NOT COUNT.  experiments/corpus_order_audit.py found
# 65% of the sampling weight was on sequences that are not videos, and the banks
# were baked through the same wrong order so nothing downstream noticed.  The
# correction branch was hit hardest because it ROLLS the student through those
# frames: 56% of s0on's windows were on broken data, 28% of them on
# unrealstereo4k, where consecutive frames alternate left/right eye and the
# median rollout depth was 720 frames -- a zigzag with zero net motion, which is
# a mundane way to teach exactly the scale collapse the experiment reported.
#
# ★ TWO DELIBERATE CHANGES from v6i's config, both forced by what was measured:
#
#   corpus       paralleldomain4d and dynamicreplica are GONE, not fixed.  Keep
#                one view and they hold 50 and 300 frames per scene against the
#                320 a run needs (burn_in 72 + 8 + L 240), so they yield zero
#                runs.  They were 40 of the 100 weight.
#   POOL         PARAMETERISED, BUT DO NOT LOWER IT -- 20 IS THE FLOOR.  --pool is
#                the TOTAL stream count across ranks (trainer.py: n_pool =
#                pool // world), so moving from DDP 2-card to NPROC=1 doubled
#                what ONE process holds: rank 0 used to carry the probes plus 10
#                streams, world=1 carries the probes plus all 20.  Measured that
#                is 80 GB of host snapshots + 67 GB of probes, and the first
#                training step then walks into this container's 200 GB cgroup cap
#                -- SIGKILL from the kernel, no traceback, nothing in the log.  It
#                happened twice at the same line (memory.events oom_kill 1 -> 2).
#                POOL=10 LOOKS like the fix and is not: k_deck() raises SystemExit
#                because pool x P(K=1)=0.25 = 2.5 is not an integer, and the 10%
#                rungs need a multiple of 10, so the smallest pool that realises
#                --k_dist v5 is 20.  Both arms died on this before it was
#                understood.  The memory lever is --probe_max (probe states are
#                built unconditionally at cand[:max(1, probe_max)] and held for
#                the whole run, ~11 GB each), NOT --pool.
#
#   SAVE_EVERY   the checkpoint interval has to be SHORTER than how long the arm
#                survives, or the watchdog livelocks.  Observed on c0on: it
#                resumes from step150.pt, reaches ~185, is OOM-killed, and
#                restarts from step150.pt again -- three times, 20 minutes a
#                cycle, newest checkpoint still 150.  save_every 50 puts the next
#                write at 200, which is 15 steps past where the arm dies, so the
#                run can never bank its progress.  Lowering it to 25 puts a write
#                at 175, inside the ~35-step window, and each cycle then advances.
#                Checkpoint frequency does not enter training -- it can differ
#                between arms without touching the 2x2, exactly like PROBE_MAX.
#                It is not free: each write is 11.4 GB and a transient ~12 GB of
#                anon, so it buys progress by adding spikes to an arm that is
#                already climbing.  It is a splint, not a fix; the fix is a box
#                that fits the on-policy arm.
#   PROBE_MAX    --pool cannot be lowered: trainer.py:440 requires pool x P(K) to be
#                an integer for every K in --k_dist v5, whose denominators have
#                lcm 20, so 20 is the SMALLEST legal pool ("--pool 10 x P(K=1)
#                = 2.5000 is not an integer" killed both arms at 15:15 / 15:25,
#                exitcode 1, no OOM).  The memory therefore has to come from
#                somewhere that is not the corpus.  Measured on gpu2:
#                80 GB (pool 20) + 67 GB (probe 6) = 147 GB anon, and the run
#                then died BUILDING the fresh walks -- the "[fresh] N dense
#                walk(s)" line never printed, and world=2 logged 94.4 GB for
#                that step, so the true peak was heading for 241 GB against a
#                200 GB cgroup cap.  probe_max 6 -> 3 frees ~33 GB and touches
#                NOTHING in training: the probe is gated by `is_main` at
#                trainer.py:2863 and never appears in a loss, backward or
#                optimizer path -- it only coarsens the Gate 6 read-out.
#   FRESH_POOL   10 -> 5 frees a further ~47 GB.  Unlike --pool this has no
#                legality guard (n_streams=args.fresh_pool, no check).  It is a
#                real cost -- the smaller the value the longer consecutive steps
#                sit on one scene, which is what the 1 -> 10 change above fixed
#                -- but 5 is far from the pathological 1.
#   fresh_pool   1 -> 10.  The flag's own help says 1 "makes consecutive steps
#                sweep the depth range in order"; measured, that put s0off on a
#                single scene for ~100 steps at a stretch (step 700-799 was 100%
#                dl3dv against a requested 10%) and made its loss curve a record
#                of which scene it sat on rather than of learning.
set -eu
cd "$(dirname "$0")/.."
NAME=$1; PID=$2; shift 2 || true
CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
STEPS=${STEPS:-1250}
THREADS=${THREADS:-16}
NPROC=${NPROC:-2}
POOL=${POOL:-20}
FRESH_POOL=${FRESH_POOL:-10}
PROBE_MAX=${PROBE_MAX:-6}
SAVE_EVERY=${SAVE_EVERY:-50}
KEEP=${KEEP:-mcd,slowtv,dl3dv,replica,scannet,unrealstereo4k}
# GPU pins the card for a 1-rank arm (docs/gtabs-plan.md §13 'broad' arm); empty = card 0
GPU=${GPU:-}

# MANIFEST=<file> replaces the s0off-derived corpus with an explicit list of
# scene specs, one per line (docs/coverage-corpus-plan.md §3: the scene-count
# ladder needs corpora that are subsets/supersets chosen by hand).  Weights are
# then SCENE-UNIFORM -- each dataset weighted by how many scenes it contributes
# -- unless WEIGHTS is given explicitly.
MANIFEST=${MANIFEST:-}
read -r WEIGHTS SC <<EOF
$(KEEP="$KEEP" MANIFEST="$MANIFEST" WEIGHTS="${WEIGHTS:-}" python3 -c "
import json, os, collections
mf = os.environ['MANIFEST']
if mf:
    sc = [l.strip() for l in open(mf) if l.strip() and not l.startswith('#')]
else:
    m = json.load(open('experiments/results/train_s0off.json'))['meta']
    keep = [k for k in os.environ['KEEP'].split(',') if k]
    sc = [s for s in m['scene'] if s.split(':')[-1] in keep]
missing = [s for s in sc if not os.path.exists(os.path.join(s.split(':')[2], 'index.json'))]
if missing:
    raise SystemExit('missing banks: ' + ', '.join(x.split(':')[0] for x in missing[:5]))
W = {'mcd': 10, 'slowtv': 10, 'dl3dv': 10, 'scannet': 10, 'replica': 5,
     'dynamicreplica': 10, 'unrealstereo4k': 15, 'paralleldomain4d': 30}
ds = sorted({s.split(':')[-1] for s in sc})
if os.environ['WEIGHTS']:
    w = os.environ['WEIGHTS']
elif mf:
    cnt = collections.Counter(s.split(':')[-1] for s in sc)
    w = ','.join(f'{d}:{cnt[d]}' for d in ds)
else:
    w = ','.join(f'{d}:{W[d]}' for d in ds)
print(w, ' '.join(sc))
")
EOF
[ -n "$SC" ] || { echo "empty corpus"; exit 1; }
echo "corpus: $(echo $SC | wc -w) scenes   weights: $WEIGHTS   p_identity: $PID   manifest: ${MANIFEST:-<s0off meta>}"

# ★ RESUME IS NOT OPTIONAL HERE.  c0off reached step 540 and then vanished with
# no error, no OOM, no reboot -- the tmux SERVER was gone too, which is what a
# login-session cleanup looks like.  --resume restores weights, optimizer state,
# the step counter and every stream's (scene, position), so the arm continues
# instead of restarting a ten-hour run from zero.
RESUME=""
LAST=$(ls -1 ckpt_train/${NAME}.step*.pt 2>/dev/null \
       | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
[ -n "${LAST:-}" ] && { RESUME="--resume $LAST"; echo "resuming from $LAST"; }

LOG=experiments/logs/train_$NAME.log
RUN=experiments/logs/.run_$NAME.sh
mkdir -p experiments/logs ckpt_train

# 214 scene specs will not fit on a tmux command line ("command too long"),
# so the whole invocation goes to a file and tmux runs the file.
cat > "$RUN" <<EOF
#!/usr/bin/env bash
export PYTHONUNBUFFERED=1 LINGBOT_THREADS=$THREADS OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export MALLOC_ARENA_MAX=2
$( [ -n "$GPU" ] && echo "export CUDA_VISIBLE_DEVICES=$GPU" )
cd "$(pwd)"
torchrun --nproc_per_node=$NPROC --master_port=\$((29500 + RANDOM % 1000)) \
  -m lingbot_map.train.trainer \
  --ckpt $CKPT --scene $SC \
  --steps $STEPS --pool $POOL --S 48 --K 28 \
  --preset A1PC --p_identity $PID --sampler unified --k_dist v5 \
  --lam_fresh 1.0 --fresh_mode walk --fresh_pool $FRESH_POOL \
  --horizons 320 960 1920 3840 \
  --dataset_weights $WEIGHTS --sampler_seed 0 \
  --lr 1e-5 --warmup 50 --wd 0.05 --clip 1.0 \
  --probe_max $PROBE_MAX --probe_every 75 --save_every $SAVE_EVERY \
  --wandb 1 --wandb_name $NAME --wandb_group spine_clean --wandb_tags spine_clean $NAME \
  --save ckpt_train/${NAME}.pt --out experiments/results/train_$NAME.json \
  $RESUME $* 2>&1 | tee -a $LOG
EOF
chmod +x "$RUN"
# ★ setsid, NOT tmux.  The first attempt ran under tmux and the whole server was
# reaped when the login session ended, taking a 540-step run with it.  setsid
# puts the trainer in its own session and process group, so nothing addressed at
# this shell's session reaches it.  The runner already tees to $LOG, so there is
# nothing tmux was providing that is lost.
setsid nohup "$RUN" >/dev/null 2>&1 &
sleep 5
if ps -eo args | grep -q "[-]-wandb_name $NAME"; then
  echo "started $NAME (detached)  log: $LOG"
else
  echo "FAILED to start $NAME -- see $LOG"; tail -5 "$LOG" 2>/dev/null; exit 1
fi
