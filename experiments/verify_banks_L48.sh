#!/usr/bin/env bash
# Is every L=48 bank complete and readable?  Exits non-zero if not.
#
# ★ index.json IS WRITTEN LAST BUT LISTS EVERY RUN, so its presence does not mean
# the run files are there.  A bank interrupted mid-build still opens, and then
# LabelBank._load raises on a missing .npz several minutes into training, after
# every scene has been mapped.  Check the files against the index instead.
#
# ★ THE EXPECTED SET COMES FROM THE L=240 BANKS, not from a checked-in list: the
# re-bake exists to reproduce that corpus scene-for-scene at a different teacher
# depth, so "what should exist" is exactly "what exists untagged".
#
#   experiments/verify_banks_L48.sh                    # every corpus
#   experiments/verify_banks_L48.sh dl3dv scannet      # some of them
set -u
cd "$(dirname "$0")/.."
CORPORA=${*:-"slowtv dl3dv scannet replica dynamicreplica unrealstereo4k paralleldomain4d"}
python3 - $CORPORA <<'PY'
import json, os, sys

want = sys.argv[1:]
prefixes = ["slowtv_", "dl3dv_", "scannet_", "replica_", "dynamicreplica_",
            "unrealstereo4k_", "paralleldomain4d_"]
ok = missing = broken = 0
for c in want:
    pre = c + "_"
    for d in sorted(os.listdir("labels")):
        if not d.startswith(pre) or d.endswith("_L48"):
            continue
        # 'replica_' is a strict prefix of nothing, but 'dynamicreplica_' would be
        # swallowed by a naive 'replica_' match if the order ever changed.
        if any(d.startswith(p2) and len(p2) > len(pre) for p2 in prefixes):
            continue
        if not os.path.exists(f"labels/{d}/index.json"):
            continue                       # not part of the L=240 corpus
        t = f"labels/{d}_L48"
        if not os.path.exists(f"{t}/index.json"):
            print(f"  MISSING  {t}"); missing += 1; continue
        runs = json.load(open(f"{t}/index.json"))["runs"]
        gone = [r["file"] for r in runs
                if not os.path.exists(f"{t}/{r['file']}")
                or os.path.getsize(f"{t}/{r['file']}") == 0]
        bad_L = sorted({r["L"] for r in runs if r["L"] > 48})
        if gone:
            print(f"  BROKEN   {t} -- {len(gone)}/{len(runs)} run files missing "
                  f"(first: {gone[0]})"); broken += 1
        elif bad_L:
            print(f"  WRONG L  {t} -- runs with L>48: {bad_L}"); broken += 1
        else:
            ok += 1
print(f"[verify] {' '.join(want)}: {ok} complete, {missing} not built, {broken} broken")
sys.exit(0 if (missing == 0 and broken == 0) else 1)
PY
