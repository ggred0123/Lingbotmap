"""Chain overlapping short teacher runs into one long track, and ask whether it
beats a single long run.

WHY.  v4 §3.2 fixes the teacher's cache budget at 8 + B + L <= 320, so with
B=72 one run can supervise at most L=240 frames (~38 m).  Depth beyond that is
unreachable by lengthening L -- the teacher itself degrades (L=600 measured 12x
worse).  v4 §3.2 therefore says long rollouts must be covered by SEVERAL short
runs with a bridge per run, and phase1-plan.md:497 records that `--stride < L`
was implemented to supply the material while the bridge itself never was.

This builds the bridge offline: consecutive runs share `L - stride` frames, so
the relative Sim(3) between them is a detached Umeyama fit on those frames
(v4 §3.3 explicitly allows Umeyama for label preprocessing).  Chaining puts every
run in run 0's gauge and yields ONE continuous track made of shallow labels.

    python experiments/stitch_runs.py labels/kth_day_06_L96s48 [labels/kth_day_06]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from teacher_depth_probe import gt_positions                           # noqa: E402
from mcd_eval import umeyama                                           # noqa: E402


def load_runs(root):
    idx = json.load(open(os.path.join(root, "index.json")))
    out = []
    for r in idx["runs"]:
        z = np.load(os.path.join(root, r["file"]))
        p = z["pose_enc"].astype(np.float64)[:, :3]
        fs = np.arange(r["t0"], r["t0"] + p.shape[0])
        out.append((fs, p, r))
    return out, idx


def stitch(runs):
    """-> {frame: global position}, [seam residuals]"""
    glob, seams = {}, []
    for i, (fs, p, _) in enumerate(runs):
        if not glob:
            glob.update({int(f): p[j] for j, f in enumerate(fs)})
            continue
        shared = [j for j, f in enumerate(fs) if int(f) in glob]
        if len(shared) < 8:
            seams.append(np.nan)          # cannot bridge -- gap in coverage
            continue
        q = p[shared]
        h = np.array([glob[int(fs[j])] for j in shared])
        s, R, t = umeyama(q, h)
        pg = (s * (R @ p.T)).T + t
        seams.append(float(np.median(np.linalg.norm(pg[shared] - h, axis=1))))
        for j, f in enumerate(fs):
            if int(f) not in glob:        # keep the earlier (shallower) estimate
                glob[int(f)] = pg[j]
    return glob, seams


def win_err(pos_by_frame, gp, W):
    """median residual of a W-frame window, aligned on that window alone"""
    fs = np.array(sorted(k for k in pos_by_frame if k < len(gp)))
    if len(fs) < W:
        return None
    P = np.array([pos_by_frame[f] for f in fs])
    G = gp[fs]
    out = []
    for a in range(0, len(fs) - W + 1, max(1, W // 4)):
        q, h = P[a:a + W], G[a:a + W]
        if not np.all(np.diff(fs[a:a + W]) == 1):
            continue
        s, R, t = umeyama(q, h)
        out.append(np.median(np.linalg.norm((s * (R @ q.T)).T + t - h, axis=1)))
    return (float(np.median(out)), len(out)) if out else None


def main():
    gp = gt_positions("mcd")
    stitched_root = sys.argv[1]
    runs, idx = load_runs(stitched_root)
    L = idx["runs"][0]["L"]
    stride = int(np.median(np.diff([r["t0"] for r in idx["runs"]])))
    glob, seams = stitch(runs)
    sv = [s for s in seams if not np.isnan(s)]
    print(f"[stitch] {os.path.basename(stitched_root)}  L={L} stride={stride} "
          f"overlap={L-stride}  runs={len(runs)}  frames={len(glob)}")
    print(f"[seam]   {len(sv)}/{len(seams)} bridged, median fit residual "
          f"{np.median(sv):.4f} m, worst {max(sv):.4f} m")

    refs = [(stitched_root + " (stitched)", glob)]
    for extra in sys.argv[2:]:
        r2, i2 = load_runs(extra)
        # a non-overlapping bank cannot be stitched: score each run on its own
        refs.append((extra + f" (per-run, L={i2['runs'][0]['L']})",
                     {int(f): p[j] for fs, p, _ in r2 for j, f in enumerate(fs)}))

    print(f"\n{'track':<40}" + "".join(f"{'W='+str(w):>12}" for w in (48, 96, 240, 480, 960)))
    for name, pos in refs:
        row = f"{os.path.basename(name):<40}"
        for W in (48, 96, 240, 480, 960):
            r = win_err(pos, gp, W)
            row += f"{(f'{r[0]:.3f}' if r else '--'):>12}"
        print(row)
    print("\n  주의: per-run 행은 run 마다 gauge 가 달라 W > L 이면 의미가 없다 (참고용).")


if __name__ == "__main__":
    main()
