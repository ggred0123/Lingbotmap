#!/usr/bin/env bash
# Keep this node's half of the L=48 pipeline alive without supervision.
#
# WHAT IT OWNS.  Re-bake shards 2 and 3 (shards 0-1 belong to kaist-bispl-ym and
# are never touched here), then the v6f training run once every non-MCD corpus
# has its L=48 bank.  Both stages are idempotent -- build_banks_generic.sh skips
# any scene whose index.json exists, and launch_v6.sh resumes from the highest
# ckpt_train/v6f.step*.pt -- so a restart costs at most one scene or 50 steps.
#
# WHY A WATCHDOG AND NOT A LOG TAIL.  Every loss on this project so far was a
# SIGKILL or SIGTERM, which writes nothing a grep can match and leaves the log
# simply stopping.  Liveness has to be polled, not read.
#
#   experiments/watchdog.sh            # foreground, prints events to stdout
#   TICK=120 MAXRESTART=5 ...
set -u
cd "$(dirname "$0")/.."
TICK=${TICK:-120}
MAXRESTART=${MAXRESTART:-5}
NEED=${NEED:-331}
say(){ echo "[wd $(date '+%m-%d %H:%M:%S')] $*"; }

banks(){ ls -d labels/{slowtv,dl3dv,scannet,replica,dynamicreplica,unrealstereo4k,paralleldomain4d}_*_L48/index.json 2>/dev/null | wc -l; }
alive(){ [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

declare -A RC=( [shard2]=0 [shard3]=0 [v6f]=0 )
LASTBANK=-1; LASTFAIL=0; MILE=0

start_shard(){ local S=$1 G=$2
  setsid nohup env SHARD=$S NSHARD=4 GPU=$G experiments/rebake_all_L48.sh \
    >> experiments/logs/rebake_shard$S.log 2>&1 < /dev/null &
  echo $! > experiments/logs/wd_shard$S.pid
  say "START shard$S -> gpu$G pid=$(cat experiments/logs/wd_shard$S.pid)"
}

start_v6f(){
  # ★ NEVER LAUNCH ONTO A BROKEN BANK.  index.json is written last, so an
  # interrupted bake is normally invisible -- but a stale index over a rebuilt
  # directory is not, and v6f died two minutes in on exactly that
  # (scannet_scene0005_00_L48, 17 of 23 runs missing).  A 1250-step run is ~10 h;
  # a 3-second check in front of it is free.
  if ! python3 experiments/check_banks.py --suffix _L48 > experiments/logs/bankcheck.log 2>&1; then
    say "BANKS-BROKEN -- refusing to launch v6f:"
    grep -v "^checked" experiments/logs/bankcheck.log | head -5 | sed 's/^/         /'
    return 1
  fi
  CUDA_VISIBLE_DEVICES=0,1 NAME=v6f STEPS=1250 NPROC=2 THREADS=16 \
    BANK_SUFFIX=_L48 MCD_ONLY=0 \
    HORIZONS="96 192 384 768 320 960 1920 3840" \
    P_IDENTITY=0.35 FIXED_HORIZON=1 \
    experiments/launch_v6.sh >> experiments/logs/v6f_launch.log 2>&1
  # ★ launch_v6.sh prints "pid=NNN  log: ...", so the pid must be cut out of the
  # line, not taken as everything after the '='.  `cut -d= -f2` stored
  # "413279  log: experiments/logs/train_v6f.log", every kill -0 on it failed,
  # and the watchdog relaunched a second 1250-step run two minutes after the
  # first -- both on the same two cards, which OOM'd rank 1.
  local p; p=$(grep -a '^pid=' experiments/logs/v6f_launch.log | tail -1 | sed 's/^pid=\([0-9][0-9]*\).*/\1/')
  case "$p" in
    ''|*[!0-9]*) say "PIDPARSE failed on v6f launch line -- not tracking, will not relaunch"; return 1 ;;
  esac
  echo "$p" > experiments/logs/wd_v6f.pid
  say "LAUNCH v6f pid=$p  (resume: $(ls -1 ckpt_train/v6f.step*.pt 2>/dev/null | wc -l) ckpts on disk)"
}

say "watchdog up -- owns shard2, shard3, v6f.  tick ${TICK}s, max ${MAXRESTART} restarts each"
while true; do
  # ---------------------------------------------------------------- re-bake
  for pair in "2 0" "3 1"; do
    set -- $pair; S=$1; G=$2
    # ★ SHARD_DONE IS NOT THE SAME AS "ALL ITS BANKS EXIST".  build_banks_generic.sh
    # walks its list once; a scene that failed (73 did, to the pre-02:17 frame-count
    # bug) is simply skipped and the shard still reports done.  So a finished shard
    # is re-run while banks are still missing -- the skip check makes that cheap,
    # and it converges because a re-run only attempts what has no index.json.
    if grep -aq "SHARD_DONE $S" experiments/logs/rebake_shard$S.log 2>/dev/null; then
      [ "$(banks)" -ge "$NEED" ] && continue
      if [ "${RC[shard$S]}" -ge "$MAXRESTART" ]; then continue; fi
      if ! alive experiments/logs/wd_shard$S.pid; then
        say "SWEEP shard$S (reported done but banks $(banks)/$NEED -- re-running for the gaps)"
        RC[shard$S]=$(( RC[shard$S] + 1 ))
        start_shard "$S" "$G"
      fi
      continue
    fi
    if ! alive experiments/logs/wd_shard$S.pid; then
      if [ "${RC[shard$S]}" -ge "$MAXRESTART" ]; then
        say "GIVEUP shard$S after ${RC[shard$S]} restarts -- needs a human"; continue
      fi
      [ "${RC[shard$S]}" -gt 0 ] || [ -f experiments/logs/wd_shard$S.pid ] \
        && say "RESTART shard$S (process gone, banks $(banks)/$NEED)"
      RC[shard$S]=$(( RC[shard$S] + 1 ))
      start_shard "$S" "$G"
    fi
  done

  # new per-scene failures
  f=$(grep -ach "FAIL " experiments/logs/rebake_shard2.log experiments/logs/rebake_shard3.log 2>/dev/null | paste -sd+ | bc)
  f=${f:-0}
  if [ "$f" -gt "$LASTFAIL" ]; then
    say "FAIL-SCENE +$(( f - LASTFAIL )) (total $f)"
    grep -ah "FAIL " experiments/logs/rebake_shard2.log experiments/logs/rebake_shard3.log 2>/dev/null | tail -2 | sed 's/^/         /'
    LASTFAIL=$f
  fi

  n=$(banks)
  if [ "$n" -ge $(( MILE + 50 )) ]; then MILE=$(( n / 50 * 50 )); say "BANKS $n/$NEED"; fi

  # ---------------------------------------------------------------- v6f
  if [ "$n" -ge "$NEED" ]; then
    if [ -f ckpt_train/v6f.pt ]; then
      say "DONE v6f -- ckpt_train/v6f.pt present.  watchdog exiting."; exit 0
    fi
    # Belt and braces: the pid file is a hint, a live trainer is the truth.  A
    # second launch costs a full run and an OOM, so check the process table too.
    if pgrep -f 'wandb_name v6f' >/dev/null 2>&1; then
      if ! alive experiments/logs/wd_v6f.pid; then
        say "ADOPT v6f pid=$(pgrep -f 'torchrun.*nproc_per_node' | head -1) (running but untracked)"
        pgrep -f 'torchrun.*nproc_per_node' | head -1 > experiments/logs/wd_v6f.pid
      fi
    elif ! alive experiments/logs/wd_v6f.pid; then
      if [ "${RC[v6f]}" -ge "$MAXRESTART" ]; then
        say "GIVEUP v6f after ${RC[v6f]} restarts -- needs a human"; sleep "$TICK"; continue
      fi
      [ "${RC[v6f]}" -gt 0 ] && say "RESTART v6f (process gone)"
      RC[v6f]=$(( RC[v6f] + 1 ))
      start_v6f
    fi
  fi
  sleep "$TICK"
done
