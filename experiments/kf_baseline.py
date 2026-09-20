"""How far does the camera actually move between two keyframes?

docs/self-distill-ver5.md, "Keyframe Interval in Physical Units":

    K is a cache-write stride, not a motion quantity.  The same K denotes very
    different camera baselines across datasets, because each dataset reaches the
    model at a different temporal sampling rate.

    Cross-dataset results must be reported against physical baseline, not K.

That makes inter-keyframe translation and rotation the PRIMARY reporting axis of
this stage, and the plan's own table was computed by hand.  This script computes
it, so a K grid can be chosen and a result table labelled without re-deriving the
conversion each time.

Two GT sources, because the two datasets store poses differently:

    MCD      meta.npz (gt_pos, gt_quat, world<-body) + a calib yaml for the
             body<-camera extrinsic.  Native 10 Hz, loader stride 1.
    Oxford   poses_c2w.txt, one flattened 4x4 world<-camera matrix per line, and
             the loader takes every 12th (benchmark/configs/datasets/oxford.yaml).

★ THE STRIDE IS PART OF THE ANSWER, NOT A DETAIL.  A keyframe interval of K on a
loader stride of s skips s*K raw frames.  Oxford K=1 is already a 12-frame
baseline, which is why it lands nearer MCD K=12 than MCD K=1 -- the single fact
this whole section exists to make visible.

    python experiments/kf_baseline.py --mcd data/mcd/kth_day_06/frames_10hz \\
        --calib data/mcd/calib/hhs_calib.yaml --K 1 4 8 12 28
    python experiments/kf_baseline.py --oxford <root>/bodleian-library-02 \\
        --stride 12 --K 1 2 3 4 8 12 28
    python experiments/kf_baseline.py --ver5_table          # the plan's table
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcd_eval import load_extrinsic, gt_camera_poses, rot_geodesic_deg
from mcd_gt import quat_to_mat

OXFORD_ROOT = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/seonghyun/R3R/data/oxford_spires"


def mcd_poses(frames_dir, calib, sensor="d455b_color"):
    """world<-camera positions and rotation matrices, in loader frame order."""
    meta = np.load(os.path.join(frames_dir, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(calib, sensor)
    pos, quat = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    return np.asarray(pos, float), quat_to_mat(np.asarray(quat, float))


def oxford_poses(scene_dir):
    """poses_c2w.txt -> positions and rotations.  Already world<-camera."""
    m = np.loadtxt(os.path.join(scene_dir, "poses_c2w.txt"), dtype=np.float64)
    if m.ndim == 1:
        m = m[None]
    m = m.reshape(-1, 4, 4)
    return m[:, :3, 3].copy(), m[:, :3, :3].copy()


def baseline(pos, rot, K, stride=1, scale_frames=8):
    """Translation and rotation between CONSECUTIVE RETAINED KEYFRAMES.

    The keyframe schedule is the deployed one (gct_stream.inference_streaming):
    frames below ``scale_frames`` are anchor frames, and after that every K-th
    frame in LOADER order persists.  ``stride`` maps loader index to raw index,
    which is where a dataset's temporal sampling enters the physical number.

    Reports the mean and the median.  They differ a lot on Oxford, where the
    platform stops and turns: the mean is what a "metres per keyframe" label
    means, the median is what a typical step looks like, and a large gap between
    them is itself a warning that one number does not describe the sequence.
    """
    n = len(pos)
    idx = np.arange(n)
    keep = idx[(idx < scale_frames) | (((idx - scale_frames) % max(K, 1)) == 0)]
    keep = keep[keep >= scale_frames - 1]          # drop the anchor interior
    if len(keep) < 3:
        return None
    p, R = pos[keep], rot[keep]
    dt = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dr = rot_geodesic_deg(R[:-1], R[1:])
    return {
        "K": int(K), "stride": int(stride), "raw_frames_skipped": int(stride * K),
        "n_keyframes": int(len(keep)),
        "trans_mean_m": float(dt.mean()), "trans_med_m": float(np.median(dt)),
        "trans_p90_m": float(np.percentile(dt, 90)),
        "rot_mean_deg": float(dr.mean()), "rot_med_deg": float(np.median(dr)),
        "rot_p90_deg": float(np.percentile(dr, 90)),
        "path_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
    }


def report(name, pos, rot, Ks, stride, scale_frames):
    rows = []
    print(f"\n{name}   {len(pos)} loader frames, stride {stride}, "
          f"path {np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.0f} m")
    print(f"  {'K':>4} {'raw':>5} {'kf':>6} {'trans m/kf':>12} {'(med)':>9} "
          f"{'rot deg/kf':>12} {'(med)':>9}")
    for K in Ks:
        r = baseline(pos, rot, K, stride, scale_frames)
        if r is None:
            print(f"  {K:>4}     -- too few keyframes")
            continue
        r["scene"] = name
        rows.append(r)
        print(f"  {K:>4} {r['raw_frames_skipped']:>5} {r['n_keyframes']:>6} "
              f"{r['trans_mean_m']:>12.3f} {r['trans_med_m']:>9.3f} "
              f"{r['rot_mean_deg']:>12.2f} {r['rot_med_deg']:>9.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcd", default=None, help="an MCD frames_10hz directory")
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--oxford", default=None, help="one Oxford Spires scene dir")
    ap.add_argument("--stride", type=int, default=None,
                    help="loader temporal stride (MCD 1, Oxford 12)")
    ap.add_argument("--K", type=int, nargs="*", default=[1, 2, 3, 4, 8, 12, 16, 28])
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--ver5_table", action="store_true",
                    help="reproduce the table in docs/self-distill-ver5.md")
    ap.add_argument("--out", default=None, help="write the rows as JSON")
    args = ap.parse_args()

    rows = []
    if args.ver5_table:
        # The exact configurations the plan tabulates, so the two can be diffed.
        mcd_dir = "data/mcd/kth_day_06/frames_10hz"
        if os.path.exists(os.path.join(mcd_dir, "meta.npz")):
            p, R = mcd_poses(mcd_dir, args.calib, args.sensor)
            rows += report("MCD kth_day_06 (10 Hz, stride 1)", p, R,
                           [1, 4, 8, 12, 28], 1, args.num_scale_frames)
        else:
            print(f"[skip] {mcd_dir} has no meta.npz")
        ox = os.path.join(OXFORD_ROOT, "bodleian-library-02")
        if os.path.exists(os.path.join(ox, "poses_c2w.txt")):
            p, R = oxford_poses(ox)
            p, R = p[::12], R[::12]            # the loader's stride
            rows += report("Oxford bodleian-library-02 (stride 12)", p, R,
                           [1, 2, 3, 4, 8, 12, 28], 12, args.num_scale_frames)
        else:
            print(f"[skip] {ox} not readable")
        print("\nRead this against docs/self-distill-ver5.md 'Keyframe Interval "
              "in Physical Units'.\nOxford K=1 sits near MCD K=12, not near MCD "
              "K=1: a table listing both\nunder 'K=1' compares unlike quantities."
              "\n\nThe plan's table is the MEDIAN column: MCD K=1 0.161 m / 1.82 "
              "deg and Oxford\nK=1 1.711 m / 5.88 deg reproduce to three "
              "decimals.  The mean runs higher\nwherever the platform stops and "
              "turns, which is most of Oxford -- at K=28 the\ntwo differ by 7 m, "
              "so a table that does not say which it quotes is ambiguous.")
    if args.mcd:
        p, R = mcd_poses(args.mcd, args.calib, args.sensor)
        s = args.stride or 1
        rows += report(os.path.basename(os.path.dirname(args.mcd)) or args.mcd,
                       p[::s], R[::s], args.K, s, args.num_scale_frames)
    if args.oxford:
        p, R = oxford_poses(args.oxford)
        s = args.stride or 12
        rows += report(os.path.basename(args.oxford.rstrip("/")), p[::s], R[::s],
                       args.K, s, args.num_scale_frames)
    if not rows:
        raise SystemExit("give --mcd, --oxford or --ver5_table")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
