#!/usr/bin/env bash
# Re-bake the two banks whose frame ORDER changed when image_names was fixed.
#
#   scannet          lexicographic "0,1,10,100" -> numeric.  Same 1159 frames,
#                    different order, so the old bank labels the wrong frames.
#   unrealstereo4k   cam0/cam1 alternating -> cam0 only.  2000 -> 1000 frames.
#
# paralleldomain4d (950 -> 50 frames) and dynamicreplica (600 -> 300) fall below
# burn_in + 8 + L = 320 once a single view is kept, so they produce zero runs and
# leave the corpus rather than being re-baked.
#
# ★ THE FRAME CACHE MUST DIE FIRST.  cache_frames.build() skips when the existing
# memmap has shape[0] >= len(names), and both defects either shorten the list
# (unrealstereo4k) or only reorder it (scannet) -- so a stale cache is silently
# reused and the whole fix is undone with no error anywhere.
set -u
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [refix] $*"; }

SCANNET_ROOT=/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/scannet/scannet/train
US4K_ROOT=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/unrealstereo4k

SC_SEQS=$(cat /tmp/seqs_scannet.txt)
US_SEQS=$(cat /tmp/seqs_unrealstereo4k.txt)

log "=== 1. drop stale frame caches ==="
n=0
for s in $SC_SEQS; do rm -f "$SCANNET_ROOT/$s/color/_cache_"*.npy && n=$((n+1)); done
log "  scannet: $n scenes cleared"
n=0
for s in $US_SEQS; do rm -f "$US4K_ROOT/$s/images/_cache_"*.npy && n=$((n+1)); done
log "  unrealstereo4k: $n scenes cleared"

log "=== 2. move the wrong banks aside ==="
mkdir -p labels/.dirty
for s in $SC_SEQS; do
  [ -d "labels/scannet_$s" ] && mv "labels/scannet_$s" "labels/.dirty/scannet_$s" 2>/dev/null
done
for s in $US_SEQS; do
  [ -d "labels/unrealstereo4k_$s" ] && mv "labels/unrealstereo4k_$s" "labels/.dirty/unrealstereo4k_$s" 2>/dev/null
done
log "  moved $(ls labels/.dirty | wc -l) banks to labels/.dirty"

log "=== 3. re-bake scannet (30 scenes) ==="
DATASET=scannet ROOT=$SCANNET_ROOT SUBDIR=color \
  experiments/build_banks_generic.sh $SC_SEQS 2>&1 | grep -aE "TOTAL|DONE|FAIL|SKIP|ALL DONE"

log "=== 4. re-bake unrealstereo4k (8 scenes) ==="
DATASET=unrealstereo4k ROOT=$US4K_ROOT SUBDIR=images \
  experiments/build_banks_generic.sh $US_SEQS 2>&1 | grep -aE "TOTAL|DONE|FAIL|SKIP|ALL DONE"

log "=== 5. verify ==="
python3 experiments/corpus_order_audit.py 2>&1 | grep -v "^\[frames\]" | tail -12
log "REFIX_DONE"
