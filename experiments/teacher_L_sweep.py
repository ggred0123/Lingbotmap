"""Label error vs the teacher's run length L, on the frames all banks share.

teacher_depth_probe.py compares two banks; this compares a ladder of them, which
is what answers "is 96 already the flat part, or does shorter still buy?".  Each
run is aligned to GT on its own (see that script's header for why global Sim(3)
is the wrong instrument here), and only frames present in EVERY bank are scored,
so the columns describe the same ground.

    experiments/teacher_L_sweep.py <gt-spec> <stride> <bank> [<bank> ...]
      gt-spec  "mcd" or the path to an Oxford Spires scene directory
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from teacher_depth_probe import per_frame_error, gt_positions           # noqa: E402


def main():
    gt_spec, stride = sys.argv[1], int(sys.argv[2])
    roots = sys.argv[3:]
    gp = gt_positions(gt_spec)[::stride]

    tables, metas = [], []
    for r in roots:
        t, idx = per_frame_error(r, gp)
        tables.append(t)
        metas.append(idx)

    shared = sorted(set.intersection(*[set(t) for t in tables]))
    if not shared:
        raise SystemExit("no frames common to every bank")

    print(f"{'L':>6}{'runs':>7}{'max depth':>11}{'median err':>13}{'p90 err':>10}{'vs L=max':>10}")
    ref = None
    for r, t, idx in zip(roots, tables, metas):
        L = idx["runs"][0]["L"]
        e = np.array([t[f][0] for f in shared])
        d = np.array([t[f][1] for f in shared])
        med = float(np.median(e))
        if ref is None:
            ref = med
        print(f"{L:>6}{len(idx['runs']):>7}{int(d.max()):>11}{med:>13.3f}"
              f"{float(np.percentile(e, 90)):>10.3f}{med / ref:>10.2f}")
    print(f"\n{len(shared)} frames scored, {shared[0]}..{shared[-1]}")


if __name__ == "__main__":
    main()
