#!/usr/bin/env bash
# Oxford Spires K=3 for the v5 cells, all 10 scenes, with base/a3/a4 on the same
# scene set so the comparison is apples-to-apples (the earlier oxford_k3 run was
# --debug: bodleian-library-02 only).  run.py skips scenes already complete, so
# the three baselines only fill in their missing 9 scenes.
set -u
cd "$(dirname "$0")/../benchmark"
G=$1; CFG=$2
export GPU=$G THREADS=8
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
log "=== RUN  $CFG on gpu$G ==="
./bench.sh run "configs/$CFG" 2>&1 | grep -avE "it/s\]|it/s," | grep -aE "Combination|Scene |Successful|failed|Error|error|Traceback"
log "=== EVAL $CFG on gpu$G ==="
./bench.sh evaluate "configs/$CFG" 2>&1 | grep -aE "Combination|Evaluating scene|Total success|Total failed|Error|error|Traceback"
log "OXFORD_V5_DONE gpu$G"
