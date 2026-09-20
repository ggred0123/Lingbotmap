#!/usr/bin/env bash
# v6: the eight-corpus mixture.  One cell, launched like launch_v5.sh's C but
# with the corpus list generalised and the visual-baseline band widened.
#
# WHY EIGHT CORPORA.  v5c (MCD + SlowTV) improved Oxford transfer 30% at every K
# and MCD global ATE at K>=8, but doubled MCD final-window ATE at K=1 and K=4 and
# had the worst scale drift of any cell.  Both corpora sit at the SLOW end --
# measured with experiments/flow_baseline.py, median LK flow at the model's own
# 518-px input width, frames i -> i+1:
#
#     dynamicreplica 0.7 | slowtv 0.6-2.2 | dl3dv 2.1 | scannet 4.3
#     replica 11.5 | MCD 11.6 | unrealstereo4k 25.1 | paralleldomain4d 81.0
#
# Oxford's benchmark K=8..28 lands at 77-109 px and christ-church-05 at 59 px
# from K=1 -- outside anything MCD+SlowTV covers (max 43.7 at K=28), which is
# where every checkpoint including the base collapses.  ParallelDomain-4D is the
# only corpus here that reaches that band, hence its weight.
#
# ★ KITTI AND VIRTUAL-KITTI2 ARE DELIBERATELY ABSENT.  benchmark/configs/
# kitti_504x280.yaml evaluates on KITTI, and VKITTI2 is a synthetic re-render of
# the same scenes, so either one in training contaminates that benchmark.
#
# ★ MVS-Synth, SAIL-VOS 3D and HyperSim are absent for a different reason: their
# sequences are 69-183 frames, below the burn_in + 8 + L = 320 floor a bank run
# needs.  They would have filled the 12-25 px gap; nothing else on disk does.
#
#   experiments/launch_v6.sh                    # STEPS=1200
#   STEPS=300 experiments/launch_v6.sh          # short control against v5c
#   DRY=1 experiments/launch_v6.sh              # print the corpus and stop
set -eu
cd "$(dirname "$0")/.."

CKPT=${CKPT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt}
STEPS=${STEPS:-1200}
THREADS=${THREADS:-16}
NPROC=${NPROC:-2}
PRESET=${PRESET:-A1PC}
# --pool is TOTAL streams across ranks and each stream holds a ~6 GB host
# snapshot, so it cannot scale with a 344-scene corpus.  It does not have to:
# every stream reset re-draws its scene from the whole corpus by dataset weight
# (trainer.py:435), so --pool bounds CONCURRENT streams, not reachable scenes.
# The trainer's "N scenes will never be visited" warning describes the initial
# `starts` assignment only and is expected here.
# ★ 20, NOT AN ARBITRARY NUMBER.  k_deck() requires pool * P(K) to be an integer
# for every K, and the ver5 mixture (25/10/10/10/10/10/25) has common denominator
# lcm(4, 10) = 20 -- 24 is rejected at startup (P(K=2)*24 = 2.4).  Going up to 40
# would halve the per-stream turns and put horizon 1920 out of reach as well.
POOL=${POOL:-20}
NAME=${NAME:-v6a}
B=/NHNHOME/WORKSPACE/26msit001_A

MCD_SCENES="kth_day_10 kth_night_01 kth_night_04 kth_night_05 \
tuhh_day_02 tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 tuhh_night_09"

WEIGHTS=${DATASET_WEIGHTS:-mcd:30,slowtv:10,dl3dv:10,scannet:10,replica:5,dynamicreplica:10,unrealstereo4k:10,paralleldomain4d:15}

# ── v6b/v6c knobs ────────────────────────────────────────────────────────────
# BANK_SUFFIX picks which bake of the MCD banks to train against: "" is the
# L=240 set every run through v6a used, "_L48" the re-bake whose labels score
# 5-7x closer to GT (experiments/teacher_L_sweep.py).  MCD_ONLY drops the other
# seven corpora so the L comparison is not confounded by them.
BANK_SUFFIX=${BANK_SUFFIX:-}
MCD_ONLY=${MCD_ONLY:-0}
HORIZONS=${HORIZONS:-320 960 1920 3840}
P_IDENTITY=${P_IDENTITY:-0.35}
FIXED_HORIZON=${FIXED_HORIZON:-0}

# The trainer refuses to start when --dataset_weights names a corpus no scene
# carries (it would otherwise balance nothing, silently), so MCD_ONLY has to
# narrow the weights too rather than only the scene list.
[ "$MCD_ONLY" = "1" ] && WEIGHTS=mcd:100

SC=$(BANK_SUFFIX="$BANK_SUFFIX" MCD_ONLY="$MCD_ONLY" python3 - "$NPROC" "$B" $MCD_SCENES <<'PY'
import os, re, sys

world = max(1, int(sys.argv[1]))
B = sys.argv[2]
mcd_scenes = sys.argv[3:]

# (dataset, bank prefix, frames-dir template).  {s} is the scene id as it appears
# after the prefix in labels/.
CORPORA = [
    ("mcd",              "",                 "data/mcd/{s}/frames_10hz"),
    ("slowtv",           "slowtv_",          "data/slow_tv/{s}/frames_10hz"),
    ("dl3dv",            "dl3dv_",           B + "/jinhyeok/dataset/dl3dv_wai/{s}/images"),
    ("scannet",          "scannet_",         B + "/V-LAB/Datasets/scannet/scannet/train/{s}/color"),
    ("replica",          "replica_",         B + "/jinhyeok/dataset/replica_wai/{s}/images"),
    ("dynamicreplica",   "dynamicreplica_",  B + "/jinhyeok/dataset/dynamicreplica/{s}/images"),
    ("unrealstereo4k",   "unrealstereo4k_",  B + "/jinhyeok/dataset/unrealstereo4k/{s}/images"),
    ("paralleldomain4d", "paralleldomain4d_", B + "/jinhyeok/dataset/paralleldomain4d/{s}/images"),
]

SUFFIX = os.environ.get("BANK_SUFFIX", "")
MCD_ONLY = os.environ.get("MCD_ONLY", "0") == "1"
#: a bake tag, e.g. the '_L48' of labels/dl3dv_<scene>_L48
_TAGGED = re.compile(r"_L\d+$")


def banks_for(ds, prefix):
    if ds == "mcd":
        # SUFFIX selects the bake; the frames are the same either way.
        return [(s, f"labels/{s}{SUFFIX}") for s in mcd_scenes]
    if MCD_ONLY:
        return []
    out = []
    for d in sorted(os.listdir("labels")):
        if not d.startswith(prefix) or not os.path.exists(f"labels/{d}/index.json"):
            continue
        # 'dynamicreplica_' also starts with... nothing else here, but 'replica_'
        # is a strict prefix of nothing while 'scannet_' is unambiguous; guard
        # anyway so a future 'replica_hd_' cannot be swallowed by 'replica_'.
        if any(d.startswith(p2) and len(p2) > len(prefix)
               for _, p2, _ in CORPORA if p2):
            continue
        # ★ THE SUFFIX SELECTS THE BAKE, AND THE EMPTY CASE MUST EXCLUDE THE
        # TAGGED ONES.  L is a property of the bank, so a re-bake at L=48 lands
        # on labels/<prefix><scene>_L48 beside the L=240 tree.  Matching on the
        # prefix alone would then put BOTH bakes of every scene in one --scene
        # list -- the same frames supervised twice at two different teacher
        # depths, silently doubling the corpus and mixing the two.  So an empty
        # BANK_SUFFIX means "the untagged bake" rather than "anything".
        if SUFFIX:
            if not d.endswith(SUFFIX):
                continue
            rest = d[len(prefix):-len(SUFFIX)]
        else:
            if _TAGGED.search(d):
                continue
            rest = d[len(prefix):]
        out.append((rest, f"labels/{d}"))
    return out

groups = []
for ds, prefix, tmpl in CORPORA:
    items = []
    for s, bank in banks_for(ds, prefix):
        frames = tmpl.format(s=s)
        if not os.path.isdir(frames):
            continue
        items.append((f"{ds}_{s}" if ds != "mcd" else s, frames, bank, ds))
    # ★ TRUNCATE TO A MULTIPLE OF `world`.  The trainer partitions specs[rank::
    # world]; a corpus contributing a partial block shifts every later corpus in
    # that block onto the wrong rank, and a rank that ends up missing a corpus
    # cannot honour --dataset_weights (trainer.py:1264 warns and carries on).
    # Dropping <= world-1 scenes per corpus is the cheap way to keep alignment.
    items = items[: len(items) - (len(items) % world)]
    if items:
        groups.append((ds, items))

if not groups:
    sys.exit("no banks found -- run experiments/chain_v6_banks.sh first")

longest = max(len(v) for _, v in groups)
out = []
for i in range(0, (longest + world - 1) // world):
    lo, hi = i * world, (i + 1) * world
    for _, items in groups:
        out += items[lo:hi]

sys.stderr.write("corpus: " + ", ".join(f"{d}x{len(v)}" for d, v in groups) +
                 f"  total {len(out)}\n")
for name, frames, bank, ds in out:
    print(f"{name}:{frames}:{bank}:{ds}")
PY
) || { echo "$SC"; exit 1; }
SC=$(echo "$SC" | tr '\n' ' ')

LOG=experiments/logs/train_$NAME.log
mkdir -p experiments/logs ckpt_train

RESUME=""
LAST=$(ls -1 ckpt_train/${NAME}.step*.pt 2>/dev/null \
       | sed 's/.*step\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)
[ -n "${LAST:-}" ] && { RESUME="--resume $LAST"; echo "resuming from $LAST"; }

echo "=== v6 -> $NAME ==="
echo "  policy   --k_dist v5 --horizons $HORIZONS  fixed_horizon=$FIXED_HORIZON"
echo "  banks    mcd${BANK_SUFFIX:-（L=240)}  mcd_only=$MCD_ONLY  p_identity=$P_IDENTITY"
echo "  scenes   $(echo "$SC" | wc -w)"
echo "  weights  $WEIGHTS"
echo "  pool     $POOL  steps $STEPS  nproc $NPROC"
# The GT probe is MCD-only and metric-only -- GT never enters the loss
# (trainer.py:612) -- so a multi-corpus run simply goes without it, exactly as
# v5c did, and is scored on the MCD holdout by the eval grid instead.
[ "${DRY:-0}" = "1" ] && { echo "DRY=1, not launching"; exit 0; }

# ★ setsid nohup, NOT tmux.  The v5 chain lost a finished eval sweep when the
# tmux server died mid-run; a 1200-step run is ~10 h and cannot afford that.
setsid nohup env PYTHONUNBUFFERED=1 LINGBOT_THREADS=$THREADS \
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
  torchrun --nproc_per_node=$NPROC --master_port=$((29500 + RANDOM % 1000)) \
    -m lingbot_map.train.trainer \
    --ckpt $CKPT --scene $SC \
    --steps $STEPS --pool $POOL --S 48 \
    --preset $PRESET --l2sp 0 \
    --k_dist v5 --horizons $HORIZONS --fixed_horizon $FIXED_HORIZON \
    --dataset_weights "$WEIGHTS" \
    --sampler unified --p_identity $P_IDENTITY --lam_fresh 1.0 --fresh_mode walk \
    --lr 1e-5 --warmup 50 --wd 0.05 --clip 1.0 \
    --probe_max 6 --probe_every 75 --save_every 50 \
    --wandb 1 --wandb_name $NAME --wandb_group v6 --wandb_tags v6 mixture ddp \
    --save ckpt_train/${NAME}.pt --out experiments/results/train_$NAME.json \
    $RESUME "$@" >> "$LOG" 2>&1 < /dev/null &
echo "pid=$!  log: $LOG"
