#!/usr/bin/env python3
"""Does the stitched bank actually hold ONE gauge over the whole scene?

experiments/stitch_bank.py chains overlapping teacher runs into a single track
in run 0's gauge, and its own report says every seam fitted (126/126 for
kth_day_10, median residual 1.4 mm).  That is a SELF-consistency check: it asks
whether neighbouring runs agree on the overlap, not whether the chained result
still agrees with the world 6,000 frames later.  126 small seam errors compound.

The distinction matters because the whole plan for a gauge-sensitive loss
(eval-metric-audit.html section 10 candidate 1, anchor-relative normalisation)
rests on the stitched track being a trustworthy single-gauge reference.  If it
is not, that plan inherits the problem it was meant to fix.

MCD has GT, so this is directly measurable.  Along the track, fit a
WINDOW-LOCAL Sim(3) to GT every so often and read the scale.  A consistent gauge
gives a flat scale; a drifting one gives a trend.  The per-run bank -- the labels
training actually uses -- is measured on the same windows for comparison.

    .venv-bench/bin/python experiments/stitched_gauge_audit.py --scene kth_day_10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import torch                                                        # noqa: E402
from lingbot_map.train.label_bank import LabelBank                  # noqa: E402
from lingbot_map.train.trainer import make_gt_score                 # noqa: E402


def spearman(x, y):
    rk = lambda v: [sorted(range(len(v)), key=lambda i: v[i]).index(i)   # noqa: E731
                    for i in range(len(v))]
    rx, ry = rk(x), rk(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--stride", type=int, default=240)
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    gt = make_gt_score(frames, os.path.join(ROOT, a.calib), a.sensor)
    stitched = LabelBank(os.path.join(ROOT, f"labels/{a.scene}_long"), cache_runs=1)
    perrun = LabelBank(os.path.join(ROOT, f"labels/{a.scene}"), cache_runs=2)

    sr = stitched.runs[0]
    STITCH_PE = np.load(os.path.join(ROOT, f"labels/{a.scene}_long", sr["file"]))["pose_enc"]
    t0, L = sr["t0"], sr["L"]
    print(f"stitched track: frames [{t0}, {t0 + L}) as one run, {L} frames")
    print(f"per-run bank  : {len(perrun.runs)} runs of L={perrun.runs[0]['L']}")

    starts = list(range(t0, t0 + L - a.S, a.stride))
    rows = []
    print(f"\n{'frame':>7} | {'stitched scale':>15} {'ATE(m)':>8} | "
          f"{'per-run scale':>14} {'ATE(m)':>8}")
    for t in starts:
        off = t - t0
        # ★ the stitched bank is pose-only, so LabelBank.get (which always wants
        # a "depth" array) cannot read it -- take pose_enc straight from the npz.
        gs = gt(torch.from_numpy(STITCH_PE[off:off + a.S].copy()).float(), t, a.S)
        row = {"t": t, "stitched": gs}
        # the per-run bank window that owns this frame, if any
        pr = None
        for rid, r in enumerate(perrun.runs):
            if r["t0"] <= t and t + a.S <= r["t0"] + r["L"]:
                pl = perrun.get(rid, t - r["t0"], a.S, device=torch.device("cpu"))
                pr = gt(pl["pose_enc"], t, a.S)
                row["perrun"] = pr
                break
        rows.append(row)
        print(f"{t:7d} | {gs['gt_scale']:15.4f} {gs['gt_ate']:8.4f} | "
              + (f"{pr['gt_scale']:14.4f} {pr['gt_ate']:8.4f}" if pr else f"{'-':>23}"))

    def summarise(key):
        v = [r[key]["gt_scale"] for r in rows if key in r]
        ts = [r["t"] for r in rows if key in r]
        if len(v) < 3:
            return None
        lv = [math.log(x) for x in v]
        rat = [v[i + 1] / v[i] for i in range(len(v) - 1)]
        return {"n": len(v), "median": st.median(v), "min": min(v), "max": max(v),
                "spread": max(v) / min(v), "sigma_log": st.pstdev(lv),
                "trend_r": spearman(ts, lv),
                "adjacent_abs_log_median": st.median([abs(math.log(x)) for x in rat]),
                "ate_median": st.median([r[key]["gt_ate"] for r in rows if key in r])}

    out = {"scene": a.scene, "S": a.S, "stride": a.stride,
           "stitched": summarise("stitched"), "perrun": summarise("perrun"),
           "rows": rows}
    print()
    for k in ("stitched", "perrun"):
        s = out[k]
        if not s:
            continue
        print(f"{k:9s} n={s['n']:3d}  scale {s['min']:.2f}-{s['max']:.2f} "
              f"(spread {s['spread']:.2f}x, sigma_log {s['sigma_log']:.3f})  "
              f"trend r={s['trend_r']:+.3f}  adjacent |log| median "
              f"{s['adjacent_abs_log_median']:.3f}  local ATE {s['ate_median']:.4f} m")
    print("\nA single consistent gauge would show a flat scale: spread near 1, "
          "sigma_log near 0, no trend.")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        print(f"[audit] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
