"""Is a shorter-L bank actually a BETTER label, or just a shorter measurement?

teacher_L_sweep.py and teacher_depth_probe.py both fit one Sim(3) PER RUN and
report the residual.  A run's length is L, so for a trajectory drifting at a
constant rate the residual scales with L and the "L=48 is 6x closer to GT"
headline is the ratio 240/48 wearing a different hat.  Measured: the L=96 / L=240
residual ratio is 0.37-0.53 at EVERY teacher depth, against a length ratio of
96/240 = 0.40.

Two instruments here that L cannot inflate:

  (A) FIXED-WINDOW RESIDUAL.  Align over W frames, not over the run.  W is the
      same for every bank, so the alignment window stops being a function of L.
      Binned by teacher depth, this answers "at the same depth, is the label
      locally more accurate?"

  (B) TURNING ANGLE.  The angle between consecutive displacement vectors is
      invariant to the arbitrary rotation, translation AND scale of the teacher's
      frame, so it needs no alignment at all.  Compares the local shape of the
      path directly against GT.

    python experiments/label_quality_fixed_window.py labels/kth_day_06 labels/kth_day_06_L96 labels/kth_day_06_L48
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from teacher_depth_probe import gt_positions                           # noqa: E402
from mcd_eval import umeyama                                           # noqa: E402

W = 48                      # == trainer --S, and the shortest run we compare


def windows(root, gp):
    """-> list of (mid_depth, fixed-window residual, turn-angle error deg)"""
    idx = json.load(open(os.path.join(root, "index.json")))
    out = []
    for r in idx["runs"]:
        z = np.load(os.path.join(root, r["file"]))
        pe = z["pose_enc"].astype(np.float64)
        t0, sf, B = r["t0"], r["scale_frames"], r["burn_in"]
        fs = np.arange(t0, t0 + pe.shape[0])
        ok = fs < len(gp)
        fs, p = fs[ok], pe[ok, :3]
        if len(fs) < W:
            continue
        g = gp[fs]
        for j0 in range(0, len(fs) - W + 1):
            q, h = p[j0:j0 + W], g[j0:j0 + W]
            s, R, t = umeyama(q, h)                       # W frames, always
            res = float(np.median(np.linalg.norm((s * (R @ q.T)).T + t - h, axis=1)))
            # turning angle: needs no alignment at all
            dq, dh = np.diff(q, axis=0), np.diff(h, axis=0)
            nq = np.linalg.norm(dq, axis=1); nh = np.linalg.norm(dh, axis=1)
            m = (nq > 1e-9) & (nh > 1e-9)
            if m.sum() < 4:
                continue
            cq = np.clip(np.einsum('ij,ij->i', dq[:-1], dq[1:]) /
                         (nq[:-1] * nq[1:] + 1e-12), -1, 1)[m[:-1] & m[1:]]
            ch = np.clip(np.einsum('ij,ij->i', dh[:-1], dh[1:]) /
                         (nh[:-1] * nh[1:] + 1e-12), -1, 1)[m[:-1] & m[1:]]
            if len(cq) < 4:
                continue
            turn = float(np.median(np.abs(np.degrees(np.arccos(cq) - np.arccos(ch)))))
            out.append((sf + B + j0 + W // 2, res, turn))
    return out


def main():
    roots = sys.argv[1:]
    gp = gt_positions("mcd")
    print(f"[gt] {len(gp)} poses   fixed window W={W}\n")
    data = {}
    for r in roots:
        L = json.load(open(os.path.join(r, "index.json")))["runs"][0]["L"]
        data[r] = (L, windows(r, gp))
        print(f"  {os.path.basename(r):<22} L={L:>3}  windows={len(data[r][1]):>5}")

    edges = [(80, 130), (130, 180), (180, 240), (240, 320)]
    print(f"\n{'depth bin':<12}" + "".join(f"{os.path.basename(r)[-4:] or 'L240':>16}" for r in roots))
    print(f"{'':12}" + "".join(f"{'resid  turn°':>16}" for _ in roots))
    for lo, hi in edges:
        row = f"{f'{lo}-{hi}':<12}"
        for r in roots:
            v = [(a, b) for d, a, b in data[r][1] if lo <= d < hi]
            row += f"{(f'{np.median([x[0] for x in v]):.3f} {np.median([x[1] for x in v]):5.2f}' if v else '--'):>16}"
        print(row)


if __name__ == "__main__":
    main()
