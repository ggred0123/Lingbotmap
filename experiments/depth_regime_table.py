"""docs/add_loss.md 9-1's depth-regime table, straight from traj.txt.

Oxford `bodleian-library-02` is 320 frames, so `auto_keyframe_threshold=320`
makes K=1 and the frame index IS the cache depth.  Each window gets its OWN
Sim(3) (Umeyama, with scale) -- a global alignment would let a good half of the
route pay for the bad half, which is exactly the effect being measured.

    python experiments/depth_regime_table.py --methods base_h64 sd_a3s250_h64 ...
"""
import argparse
import os

import numpy as np

WS = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford/oxford/bodleian-library-02"
BINS = [(80, 180), (180, 280), (220, 320)]


def load_traj(path):
    """BSS Trajectory Format v2 -> {frame_idx: (3,) camera centre}."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            if len(v) < 13:
                continue
            idx = int(float(v[0]))
            m = np.array([float(x) for x in v[1:13]], dtype=np.float64).reshape(3, 4)
            out[idx] = m[:, 3]          # C2W translation = camera centre
    return out


def umeyama_sim3(X, Y):
    """Least-squares similarity aligning X (3,N) onto Y (3,N)."""
    mx, my = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    Xc, Yc = X - mx, Y - my
    S = Yc @ Xc.T / X.shape[1]
    U, D, Vt = np.linalg.svd(S)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var = (Xc ** 2).sum() / X.shape[1]
    c = float(np.trace(np.diag(D) @ W) / var) if var > 0 else 1.0
    return c * R @ X + (my - c * R @ mx)


def window_ates(est, gt, win):
    """(centre_depth, local ATE) for every full window, own Sim(3) each."""
    frames = sorted(set(est) & set(gt))
    out = []
    for i in range(len(frames) - win + 1):
        idx = frames[i:i + win]
        X = np.stack([est[f] for f in idx], 1)
        Y = np.stack([gt[f] for f in idx], 1)
        if not (np.isfinite(X).all() and np.isfinite(Y).all()):
            continue
        r = np.linalg.norm(umeyama_sim3(X, Y) - Y, axis=0)
        out.append((idx[len(idx) // 2], float(np.sqrt((r ** 2).mean()))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--windows", nargs="+", type=int, default=[24, 32, 48])
    ap.add_argument("--ref", default="base_h64", help="denominator for the ratio column")
    args = ap.parse_args()

    gt = load_traj(os.path.join(WS, "gt", "traj.txt"))
    trajs = {}
    for m in args.methods:
        p = os.path.join(WS, m, "traj.txt")
        if not os.path.exists(p):
            print(f"  (missing {m})")
            continue
        trajs[m] = load_traj(p)
    if not trajs:
        raise SystemExit("no trajectories found")

    for win in args.windows:
        print(f"\n=== window {win} -- median per-window Sim(3) local ATE ===")
        cells = {m: window_ates(t, gt, win) for m, t in trajs.items()}
        head = f"{'depth':>12} " + "".join(f"{m:>17}" for m in trajs)
        print(head)
        for lo, hi in BINS:
            row = f"{f'[{lo},{hi})':>12} "
            ref = None
            for m in trajs:
                v = [a for d, a in cells[m] if lo <= d < hi]
                med = float(np.median(v)) if v else float("nan")
                if m == args.ref:
                    ref = med
                cell = f"{med:.2f}"
                if ref and m != args.ref and np.isfinite(med):
                    cell += f" ({med / ref:.2f}x)"
                row += f"{cell:>17}"
            print(row)


if __name__ == "__main__":
    main()
