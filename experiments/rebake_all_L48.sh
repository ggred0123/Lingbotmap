#!/usr/bin/env bash
# Re-bake EVERY training corpus at L=48, sharded across GPUs and nodes.
#
# WHY.  experiments/rebake_L48.sh re-baked the 10 MCD scenes only, so v6c/v6e
# trained against a MIXED corpus: MCD at L=48 and the other seven corpora still
# at L=240.  A run writes its i-th label with the teacher at depth 80+i, so L is
# the teacher's drift budget, and teacher_L_sweep.py measured the cost of the
# difference on the SAME target frames (Oxford stride-12, median label error vs
# GT, m):  L=240 -> 5.232,  L=96 -> 1.709,  L=48 -> 0.838.  Mixing the two bakes
# means 97% of the corpus is supervised by labels 6x worse than the 3% that is
# not, and no ablation can separate "more corpora" from "worse labels".
#
# ★ L=48 IS THE FLOOR.  next_valid_window (trainer.py:168) requires a whole
# S=48 window inside one run, so L < S yields zero trainable windows.
#
# ★ SHARDS ARE DETERMINISTIC AND DISJOINT so two nodes can bake into the same
# Lustre labels/ tree at once.  Assignment is longest-processing-time-first over
# a cost model calibrated on the MCD re-bake (below), seeded by a stable sort --
# same inputs, same split, on either node.  build_banks_generic.sh also skips
# any bank whose index.json already exists, so an overlap would waste nothing.
#
#   DRY=1 experiments/rebake_all_L48.sh              # plan for all shards
#   SHARD=0 NSHARD=4 GPU=0 experiments/rebake_all_L48.sh
set -u
cd "$(dirname "$0")/.."

NSHARD=${NSHARD:-4}
SHARD=${SHARD:-0}
GPU=${GPU:-0}
DRY=${DRY:-0}
B=/NHNHOME/WORKSPACE/26msit001_A

# Cost model from experiments/logs/rebake_*_L48.log: a run costs
# (8 anchor + 72 burn-in + 48 supervised) = 128 teacher frames at ~58 ms, and
# 48 * 0.812 MB of labels.  Note this is 2x the per-supervised-frame cost of an
# L=240 bake -- the burn-in is paid five times as often.
PLAN=$(NSHARD="$NSHARD" B="$B" python3 - <<'PY'
import json, math, os, sys

B = os.environ["B"]; NSHARD = int(os.environ["NSHARD"])
CORPORA = [  # mirrors launch_v6.sh CORPORA, minus mcd (already re-baked)
    ("slowtv",           "slowtv_",           "data/slow_tv/{s}/frames_10hz",                        "frames_10hz"),
    ("dl3dv",            "dl3dv_",            B + "/jinhyeok/dataset/dl3dv_wai/{s}/images",          "images"),
    ("scannet",          "scannet_",          B + "/V-LAB/Datasets/scannet/scannet/train/{s}/color", "color"),
    ("replica",          "replica_",          B + "/jinhyeok/dataset/replica_wai/{s}/images",        "images"),
    ("dynamicreplica",   "dynamicreplica_",   B + "/jinhyeok/dataset/dynamicreplica/{s}/images",     "images"),
    ("unrealstereo4k",   "unrealstereo4k_",   B + "/jinhyeok/dataset/unrealstereo4k/{s}/images",     "images"),
    ("paralleldomain4d", "paralleldomain4d_", B + "/jinhyeok/dataset/paralleldomain4d/{s}/images",   "images"),
]
ROOT = {d: t.replace("/{s}/" + sub, "") for d, _, t, sub in CORPORA}
SUB = {d: sub for d, _, _, sub in CORPORA}
PFX = {d: p for d, p, _, _ in CORPORA}

# The L=240 banks define the corpus: baking exactly those scenes keeps the L=48
# corpus scene-for-scene identical, so the only variable is teacher depth.
jobs = []
for ds, prefix, tmpl, _ in CORPORA:
    for d in sorted(os.listdir("labels")):
        if not d.startswith(prefix) or d.endswith("_L48"):
            continue
        if any(d.startswith(p2) and len(p2) > len(prefix) for p2 in PFX.values()):
            continue
        if not os.path.exists(f"labels/{d}/index.json"):
            continue
        s = d[len(prefix):]
        if not os.path.isdir(tmpl.format(s=s)):
            continue
        if os.path.exists(f"labels/{d}_L48/index.json"):
            continue                      # already baked -- costs nothing
        n = json.load(open(f"labels/{d}/index.json")).get("n_frames_available", 0)
        runs = max(0, math.ceil((min(n, 8000) - 80) / 48))
        if runs:
            jobs.append((runs, ds, s))

# LPT: hand the most expensive scene to the emptiest shard.  Balances to within
# one scene's cost even though scene costs span 4 -> 166 runs.
jobs.sort(key=lambda j: (-j[0], j[1], j[2]))
load = [0] * NSHARD
bins = [[] for _ in range(NSHARD)]
for runs, ds, s in jobs:
    i = min(range(NSHARD), key=lambda k: (load[k], k))
    load[i] += runs; bins[i].append((ds, s))

for i, b in enumerate(bins):
    sys.stderr.write(f"shard {i}: {len(b):>4} scenes, {load[i]:>5} runs, "
                     f"~{load[i]*48*0.812/1000:>5.0f} GB, ~{load[i]*128*0.058/3600:>4.1f} GPU-h\n")
sys.stderr.write(f"TOTAL  : {len(jobs):>4} scenes, {sum(load):>5} runs, "
                 f"~{sum(load)*48*0.812/1000:>5.0f} GB, ~{sum(load)*128*0.058/3600:>4.1f} GPU-h\n")
for i, b in enumerate(bins):
    for ds, s in b:
        print(f"{i}\t{ds}\t{ROOT[ds]}\t{SUB[ds]}\t{s}")
PY
) || { echo "planning failed"; exit 1; }

[ "$DRY" = "1" ] && { echo "$PLAN" | awk -F'\t' '{c[$1"\t"$2]++} END{for(k in c) print k"\t"c[k]}' | sort; exit 0; }

log() { echo "[$(date +%H:%M:%S)] $*"; }
log "shard $SHARD/$NSHARD on gpu $GPU"
for ds in $(echo "$PLAN" | awk -F'\t' -v s="$SHARD" '$1==s{print $2}' | sort -u); do
  root=$(echo "$PLAN" | awk -F'\t' -v s="$SHARD" -v d="$ds" '$1==s&&$2==d{print $3; exit}')
  sub=$(echo "$PLAN"  | awk -F'\t' -v s="$SHARD" -v d="$ds" '$1==s&&$2==d{print $4; exit}')
  scenes=$(echo "$PLAN" | awk -F'\t' -v s="$SHARD" -v d="$ds" '$1==s&&$2==d{print $5}' | tr '\n' ' ')
  n=$(echo $scenes | wc -w)
  log "=== $ds: $n scenes ==="
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    DATASET=$ds ROOT="$root" SUBDIR="$sub" L=48 SUFFIX=_L48 \
    experiments/build_banks_generic.sh $scenes
done
log "SHARD_DONE $SHARD"
