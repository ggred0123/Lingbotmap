"""Does the frozen teacher's label quality decay with its own rollout depth?

The banks are the student's only target and the teacher that writes them is the
base model, walking anchor(8) + burn_in(72) + L frames per run.  With L=240 a
run's last labels come from a teacher 320 frames deep; with L=96 it restarts
every 96, so no label is ever written past depth 176.  The SAME FRAME therefore
carries two labels written at different teacher depths, and MCD kth_day_06 has
GT to score both against.

    experiments/teacher_depth_probe.py labels/kth_day_06 labels/kth_day_06_L96

★ THE ALIGNMENT IS PER RUN, NOT GLOBAL.  A single Sim(3) over 2900 frames
measures how far the whole trajectory has bent, which is ~57 m here and swamps
the thing being asked about.  Each run is aligned to GT on its own, so what is
left is the label's LOCAL error -- which is what the student actually copies.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama          # noqa: E402

FRAMES = "data/mcd/kth_day_06/frames_10hz"
CALIB = "data/mcd/calib/hhs_calib.yaml"
SENSOR = "d455b_color"


def gt_positions(spec):
    """Camera-centre positions for the scene the banks were built from.

    ``spec`` is either "mcd" (kth_day_06 via the calib extrinsic) or the path to
    an Oxford Spires scene, whose poses_c2w.txt is 16 floats per line, row-major
    4x4 camera-to-world -- the translation column is already the camera centre,
    so no extrinsic is applied.  ``--stride`` on the bank subsamples the frames,
    and the same stride has to be applied here or frame i means two things.
    """
    if spec == "mcd":
        meta = np.load(os.path.join(FRAMES, "meta.npz"), allow_pickle=True)
        T, _, _ = load_extrinsic(CALIB, SENSOR)
        gp, _ = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
        return gp
    M = np.loadtxt(os.path.join(spec, "poses_c2w.txt")).reshape(-1, 4, 4)
    return M[:, :3, 3]


def per_frame_error(root, gp):
    """-> {frame: (local error vs GT in m, teacher depth when written)}"""
    idx = json.load(open(os.path.join(root, "index.json")))
    out = {}
    for r in idx["runs"]:
        z = np.load(os.path.join(root, r["file"]))
        pe = z["pose_enc"].astype(np.float64)
        t0, sf, B = r["t0"], r["scale_frames"], r["burn_in"]
        fs = np.arange(t0, t0 + pe.shape[0])
        ok = fs < len(gp)
        fs, p = fs[ok], pe[ok, :3]
        if len(fs) < 8:
            continue
        g = gp[fs]
        s, R, t = umeyama(p, g)                      # this run only
        err = np.linalg.norm((s * (R @ p.T)).T + t - g, axis=1)
        for j, f in enumerate(fs):
            out[int(f)] = (float(err[j]), sf + B + j)
    return out, idx


def main():
    a_root, b_root = sys.argv[1], sys.argv[2]
    gt_spec = sys.argv[3] if len(sys.argv) > 3 else "mcd"
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    gp = gt_positions(gt_spec)[::stride]
    print(f"[gt] {gt_spec}  stride={stride}  {len(gp)} poses")

    A, ia = per_frame_error(a_root, gp)
    B, ib = per_frame_error(b_root, gp)
    print(f"[A] {a_root}  L={ia['runs'][0]['L']}  runs={len(ia['runs'])}")
    print(f"[B] {b_root}  L={ib['runs'][0]['L']}  runs={len(ib['runs'])}")

    shared = sorted(set(A) & set(B))
    dA = np.array([A[f][1] for f in shared])
    dB = np.array([B[f][1] for f in shared])
    eA = np.array([A[f][0] for f in shared])
    eB = np.array([B[f][0] for f in shared])
    print(f"[shared] {len(shared)} frames.  Where the two disagree on depth, "
          f"B is the shallower teacher in {(dB < dA).sum()} of them "
          f"(median depth drop {np.median(dA[dB < dA] - dB[dB < dA]):.0f} frames)\n")

    print(f"{'A depth':>12}{'n':>7}{'A err':>10}{'B err':>10}{'B depth':>10}{'B/A':>8}")
    edges = [80, 128, 176, 224, 272, 320]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dA >= lo) & (dA < hi)
        if m.sum() < 20:
            print(f"{f'{lo}-{hi}':>12}{m.sum():>7}   (too few)")
            continue
        print(f"{f'{lo}-{hi}':>12}{m.sum():>7}{np.median(eA[m]):>10.3f}"
              f"{np.median(eB[m]):>10.3f}{np.median(dB[m]):>10.0f}"
              f"{np.median(eB[m]) / max(np.median(eA[m]), 1e-9):>8.2f}")

    deep = dA >= 176
    if deep.sum() >= 20:
        print(f"\nWhere A is past L=96's reach (depth >= 176, n={deep.sum()}):")
        print(f"  A (deep teacher)    median {np.median(eA[deep]):.3f} m")
        print(f"  B (shallow teacher) median {np.median(eB[deep]):.3f} m")
        d = (np.median(eA[deep]) - np.median(eB[deep])) / max(np.median(eA[deep]), 1e-9)
        print(f"  -> the shallow teacher is {d * 100:+.1f}% "
              f"{'better' if d > 0 else 'worse'} on the same frames")


if __name__ == "__main__":
    main()
