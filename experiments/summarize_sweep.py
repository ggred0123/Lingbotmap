"""Summarize phase0_density_sweep.py outputs.

Answers the Phase 0 question: does the sparse-keyframe run measurably drift
from the dense (interval=1) run, and does that drift GROW along the sequence?

Prints, per sequence:
  - a density x drift table (final + per-quartile), normalized by path length
  - the growth ratio (last quartile / second quartile) -- >1 means accumulating,
    ~1 means a constant offset rather than drift
"""

import glob
import json
import os
import sys

import numpy as np


def quartiles(x, start):
    """Mean of x over each quartile of the post-anchor region."""
    x = np.asarray(x)[start:]
    n = len(x)
    if n < 8:
        return [float("nan")] * 4
    b = [x[: n // 4], x[n // 4: n // 2], x[n // 2: 3 * n // 4], x[3 * n // 4:]]
    return [float(np.nanmean(v)) for v in b]


def main(paths):
    for p in sorted(paths):
        with open(p) as f:
            r = json.load(f)
        m = r["meta"]
        name = os.path.basename(m["image_folder"])
        S = m["n_frames"]
        anchor = m["num_scale_frames"]
        L = m["total_path_length_anchor_units"]
        ref = m["ref_interval"]

        print(f"\n{'='*78}")
        print(f"{name}   frames={S}   anchor={anchor}   "
              f"path_len={L:.3f} (anchor units)   ref=interval{ref}")
        print(f"{'='*78}")
        print(f"{'int':>4} {'#kf':>5} | {'trans final':>12} {'%path':>7} | "
              f"{'rot final':>10} | {'RPE mean':>10} | {'depth rel':>10} | {'growth':>7}")
        print("-" * 78)

        for it_s, d in sorted(r["runs"].items(), key=lambda kv: int(kv[0])):
            it = int(it_s)
            n_kf = anchor + max(0, (S - anchor + it - 1) // it)
            tr = np.asarray(d["trans_drift"])
            ro = np.asarray(d["rot_drift_deg"])
            rp = np.asarray(d["rpe_trans_drift"])
            dp = np.asarray(d["depth_rel_median"])

            qt = quartiles(tr, anchor)
            growth = qt[3] / qt[1] if qt[1] > 1e-9 else float("nan")

            print(f"{it:>4} {n_kf:>5} | {tr[-1]:>12.5f} {100*tr[-1]/max(L,1e-9):>6.2f}% | "
                  f"{ro[-1]:>9.3f}° | {np.nanmean(rp):>10.6f} | "
                  f"{np.nanmean(dp):>10.5f} | {growth:>7.2f}x")

        print("\n  translation drift by quartile (anchor units, post-anchor region):")
        for it_s, d in sorted(r["runs"].items(), key=lambda kv: int(kv[0])):
            qt = quartiles(d["trans_drift"], anchor)
            print(f"    interval {int(it_s):>3}: " +
                  "  ".join(f"Q{i+1}={v:.5f}" for i, v in enumerate(qt)))

        print("\n  local RPE drift by quartile (isolates local geometry, not accumulation):")
        for it_s, d in sorted(r["runs"].items(), key=lambda kv: int(kv[0])):
            qt = quartiles(d["rpe_trans_drift"], anchor)
            print(f"    interval {int(it_s):>3}: " +
                  "  ".join(f"Q{i+1}={v:.6f}" for i, v in enumerate(qt)))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "results", "*_sweep.json"))
    main(args)
