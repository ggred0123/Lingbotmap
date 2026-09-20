"""Verify every bank's index.json against the run files actually on disk.

WHY THIS EXISTS.  build_banks_generic.sh decides a scene is done by the presence
of index.json alone, and label_bank writes that file LAST -- so an interrupted
bake normally leaves no index and is retried.  But a directory that is removed
and rebuilt while a stale index survives (labels/scannet_scene0005_00_L48 on
2026-08-25) passes the skip check while missing 17 of its 23 runs, and the
trainer only finds out when a stream walks onto the missing frame -- which for
v6f was two minutes into a 1250-step run.

    python experiments/check_banks.py                # every bank
    python experiments/check_banks.py --suffix _L48  # one bake
    python experiments/check_banks.py --fix          # delete the broken ones
"""
import argparse
import json
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default=None, help="only banks whose name ends with this")
    ap.add_argument("--fix", action="store_true", help="rm -rf every broken bank so a re-bake rebuilds it")
    args = ap.parse_args()

    names = sorted(d for d in os.listdir("labels")
                   if os.path.exists(f"labels/{d}/index.json")
                   and (args.suffix is None or d.endswith(args.suffix)))
    broken = []
    for d in names:
        p = f"labels/{d}"
        try:
            runs = json.load(open(p + "/index.json")).get("runs") or []
        except Exception as e:
            broken.append((d, f"index unreadable: {e}", 0))
            continue
        miss = [r["file"] for r in runs
                if isinstance(r, dict) and r.get("file")
                and not os.path.exists(os.path.join(p, r["file"]))]
        if miss:
            broken.append((d, f"{len(miss)}/{len(runs)} runs missing", len(miss)))

    print(f"checked {len(names)} banks -- ok {len(names) - len(broken)}, broken {len(broken)}")
    for d, why, _ in broken:
        print(f"  {d}  {why}")
        if args.fix:
            shutil.rmtree(f"labels/{d}")
            print(f"    removed -- re-run the bake to rebuild")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
