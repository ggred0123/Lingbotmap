#!/usr/bin/env python3
"""Is each training sequence actually a video, in the order training reads it?

``label_bank.image_names`` is ``sorted(os.listdir(...))`` -- plain lexicographic
order over the ORIGINAL directory.  For a zero-padded single-camera dump that is
the video order.  For anything else it is not, and nothing downstream checks:
the bank is baked in the same wrong order, so the labels agree with the inputs
and every consistency check passes while neither is a video.

For each scene this reports the fraction of ADJACENT PAIRS in training order
that are genuinely one time step apart in the same camera.  A real video scores
1.00.  Anything less is the fraction of the rollout that is asking the model to
estimate ego-motion between frames that have no ego-motion between them.

    python3 experiments/corpus_order_audit.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lingbot_map.train.label_bank import image_names            # noqa: E402

WEIGHTS = {"mcd": 10, "slowtv": 10, "dl3dv": 10, "scannet": 10, "replica": 5,
           "dynamicreplica": 10, "unrealstereo4k": 15, "paralleldomain4d": 30}


def parse(ds: str, n: str):
    """(camera, time) for one filename, or None if the layout is unknown."""
    stem = os.path.splitext(n)[0]
    if ds == "paralleldomain4d":            # <18-digit ts>_<view>, 19 views
        m = re.match(r"^(\d+)_(.+)$", stem)
        return (m.group(2), int(m.group(1))) if m else None
    if ds == "unrealstereo4k":              # <5-digit>_cam<k>
        m = re.match(r"^(\d+)_(cam\d+)$", stem)
        return (m.group(2), int(m.group(1))) if m else None
    if ds == "dynamicreplica":              # <scene>_<left|right>-<4-digit>
        m = re.match(r"^(.*)_(left|right)-(\d+)$", stem)
        return (m.group(2), int(m.group(3))) if m else None
    if ds == "scannet":                     # <int>
        return ("cam", int(stem)) if stem.isdigit() else None
    m = re.search(r"(\d+)", stem)           # everything else: first integer
    return ("cam", int(m.group(1))) if m else None


def audit(ds: str, d: str) -> dict | None:
    try:
        names = image_names(d)
    except Exception:
        return None
    if not names:
        return None
    keys = [parse(ds, n) for n in names]
    if any(k is None for k in keys):
        return {"ds": ds, "n": len(names), "unparsed": True}
    cams = Counter(k[0] for k in keys)
    good = sum(1 for a, b in zip(keys, keys[1:])
               if a[0] == b[0] and b[1] > a[1])
    # * THE METRIC TRAINING ACTUALLY FEELS is per bank run, not per scene: runs
    # are L=240 from t0=80, so a defect that is rare over the whole directory can
    # still land inside most runs.  dynamicreplica is the case in point -- one
    # left/right seam in 600 frames is 0.2% of pairs, but every run that spans it
    # is corrupted.
    L, t0, runs_bad, runs = 240, 80, 0, 0
    for a0 in range(t0, len(keys) - L + 1, L):
        w = keys[a0:a0 + L]
        g = sum(1 for x, y in zip(w, w[1:]) if x[0] == y[0] and y[1] > x[1])
        runs += 1
        if g < len(w) - 1:
            runs_bad += 1
    # views sharing one timestamp -- the paralleldomain4d / unrealstereo4k shape
    per_t = Counter(k[1] for k in keys)
    return {"ds": ds, "n": len(names), "cameras": len(cams),
            "views_per_time": max(per_t.values()),
            "contiguous_frac": good / max(len(names) - 1, 1),
            "runs": runs, "runs_bad": runs_bad, "unparsed": False}


def main() -> int:
    meta = json.load(open(os.path.join(ROOT, "experiments/results/train_s0off.json")))["meta"]
    seen, rows = {}, []
    for spec in meta["scene"]:
        p = spec.split(":")
        ds = p[-1]
        seen.setdefault(ds, []).append(p[1])
    print(f"{'dataset':<18}{'weight':>7}{'scenes':>8}{'frames':>9}{'cams':>6}"
          f"{'views/t':>9}{'in-order':>10}{'bad runs':>10}  verdict")
    tot_bad = 0
    for ds, dirs in sorted(seen.items(), key=lambda kv: -WEIGHTS.get(kv[0], 0)):
        a = audit(ds, dirs[0])
        if not a:
            print(f"{ds:<18}{'?':>7}{len(dirs):>8}  (unreadable)")
            continue
        if a["unparsed"]:
            print(f"{ds:<18}{WEIGHTS.get(ds,0):>7}{len(dirs):>8}{a['n']:>9}"
                  f"{'?':>6}{'?':>9}{'?':>10}  name layout not recognised")
            continue
        ok = a["contiguous_frac"] > 0.99 and a["runs_bad"] == 0
        tot_bad += 0 if ok else WEIGHTS.get(ds, 0)
        rows.append(dict(a, weight=WEIGHTS.get(ds, 0), scenes=len(dirs), ok=ok))
        print(f"{ds:<18}{WEIGHTS.get(ds,0):>7}{len(dirs):>8}{a['n']:>9}"
              f"{a['cameras']:>6}{a['views_per_time']:>9}"
              f"{a['contiguous_frac']:>9.2%}"
              f"{a['runs_bad']:>4}/{a['runs']:<4}  {'video' if ok else '*** NOT A VIDEO ***'}")
    tot = sum(WEIGHTS.values())
    print(f"\nsampling weight on sequences that are not videos: {tot_bad}/{tot} "
          f"= {tot_bad / tot:.0%}")
    print("in-order = fraction of adjacent pairs in TRAINING order that are one "
          "time step apart in the same camera.  A real video is 100%.")
    json.dump({"rows": rows, "bad_weight": tot_bad, "total_weight": tot},
              open(os.path.join(ROOT, "experiments/results/corpus_order.json"), "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
