#!/usr/bin/env bash
# Wait for v6i to finish, then sweep its checkpoints through Oxford K=1 and print
# the comparison against every run already scored on that grid.
#
# ★ OXFORD K=1 IS THE DECISION METRIC, NOT THE PROBE.  Measured this session:
# v6d was first on the label probe and last on Oxford (+23.3%); v6c was third on
# the probe and the only checkpoint ever to beat base (-7.1%).  Within v6f the
# probe improved from -4.8% to -18.8% while Oxford ATE went 6.31 -> 9.46 m.
#
# ★ SWEEP, DO NOT PICK ONE.  Every run so far degrades monotonically with steps
# on Oxford, so the final checkpoint is systematically the worst one.  The curve
# is the result; a single number would hide it.
set -u
cd "$(dirname "$0")/.."
NAME=${NAME:-v6i}
STEPS_LIST=${STEPS_LIST:-"50 100 200 300 400 500 600 700 800 900 1000 1100 1200 1250"}
MAXWAIT=${MAXWAIT:-43200}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for ckpt_train/$NAME.pt"
t=0
while [ ! -f "ckpt_train/$NAME.pt" ] && [ "$t" -lt "$MAXWAIT" ]; do sleep 120; t=$((t+120)); done
[ -f "ckpt_train/$NAME.pt" ] || { log "TIMEOUT -- $NAME never finished; nothing evaluated"; exit 1; }
log "$NAME finished.  preparing checkpoints"

M=""
for s in $STEPS_LIST; do
  src=ckpt_train/$NAME.step$s.pt
  [ -f "$src" ] || { log "  skip s$s (no checkpoint)"; continue; }
  dst=bench_ckpt/sd_${NAME}_step$s.pt
  [ -f "$dst" ] || python3 experiments/_strip_optim.py "$src" "$dst" >/dev/null || { log "  STRIP FAILED s$s"; continue; }
  cfg=benchmark/configs/methods/sd_${NAME}s${s}_k1.yaml
  [ -f "$cfg" ] || sed "s|sd_v6g_step50\.pt|sd_${NAME}_step${s}.pt|" \
      benchmark/configs/methods/sd_v6gs50_k1.yaml > "$cfg"
  M="$M  - sd_${NAME}s${s}_k1
"
done
[ -n "$M" ] || { log "no checkpoint prepared -- stopping"; exit 1; }

cat > benchmark/configs/oxford_${NAME}_sweep.yaml <<YAML
# Oxford stride-12 at K=1 across the $NAME checkpoint sweep.
workspace: /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford
datasets:
  - oxford
methods:
  - base_k1
$M
evaluation:
  traj: {enable: true, vis: false}
  auc: {enable: true, vis: false, aggregation: both}
  depth: {enable: false}
  points: {enable: false}
YAML

log "=== RUN  (base_k1 and anything already complete is skipped) ==="
GPU=0 THREADS=8 ./benchmark/bench.sh run "configs/oxford_${NAME}_sweep.yaml" 2>&1 \
  | grep -aE "Combination \(|Scenes to process|Successful:|Total failed|already complete|out of memory|rror|Traceback"
log "=== EVALUATE ==="
GPU=0 ./benchmark/bench.sh evaluate "configs/oxford_${NAME}_sweep.yaml" 2>&1 \
  | grep -aE "Total success|Total failed|rror"

log "=== RESULT ==="
NAME=$NAME python3 - <<'PY'
import json, os, re
B = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford/oxford/eval"
t = json.load(open(B + "/traj.json")); b = t["base_k1"]
N = os.environ["NAME"]
def d(k): return (t[k]["ate"]/b["ate"]-1)*100
print(f"  base_k1  ATE {b['ate']:.3f} m   rpe_trans {b['rpe_trans']:.4f}\n")
mine = sorted((int(re.search(r's(\d+)_k1$', k).group(1)), k)
              for k in t if k.startswith(f"sd_{N}s") and k.endswith("_k1"))
print(f"  {'step':>6}{'ATE':>9}{'vs base':>10}{'rpe_trans':>11}")
for s, k in mine:
    print(f"  {s:>6}{t[k]['ate']:>9.3f}{d(k):>9.1f}%{t[k]['rpe_trans']:>11.4f}")
if mine:
    bs, bk = min(mine, key=lambda x: t[x[1]]["ate"])
    print(f"\n  BEST {N}: step {bs}  ATE {t[bk]['ate']:.3f}  ({d(bk):+.1f}%)")
print("\n  기존 기준선:")
for k, l in (("sd_v6cs300_k1","v6c s300 (MCD만 L48, 구 weight)"),
             ("sd_v6as300_k1","v6a s300 (전량 L240, 구 weight)"),
             ("sd_v6as600_k1","v6a s600"),
             ("sd_v6hs600_k1","v6h s600 (전량 L48, 신 weight)"),
             ("sd_v6gs600_k1","v6g s600 (전량 L48, 구 weight)"),
             ("sd_v6fs600_k1","v6f s600 (얕은 horizon)")):
    if k in t: print(f"    {l:<38}{t[k]['ate']:>8.3f}{d(k):>9.1f}%")
PY
log "AUTO_EVAL_DONE $NAME"
