"""Phase 0, decisive run: LS / LD / FD scored against ground truth on MCD.

The example-data version of this test (fresh_vs_fork.py) could only measure that
the long-dense fork teacher and the fresh-anchor teacher DISAGREE -- with no GT
there was no way to say which one is right, which is precisely what the teacher
design hinges on.  With MCD ground truth each condition is scored independently,
so the disagreement finally gets a sign.

  LS  long-sparse : full history from frame 0 at interval K   (what deploys)
  LD  long-dense  : fork LS's KV state at t0, continue denser (ver3 §4 teacher)
  FD  fresh-dense : new run anchored at t0-B, dense           (ver1 teacher)

Each is Sim(3)-aligned to GT over the eval window on its own, so the differing
gauges of LS/LD (shared anchor) and FD (own anchor) stop mattering.

Teacher density is budget-limited, not free: ver3 §4.2 requires the teacher's
total cache to stay under the 320-view training limit, so K_t is chosen as the
densest interval that fits.  The achieved ratio K/K_t is reported per t0 because
it shrinks as t0 grows.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from phase0_density_sweep import build_model
from fork_rig import snapshot_state, restore_state, run_anchor, step_frames
from mcd_eval import load_extrinsic, gt_camera_poses, score

KF_LIMIT = 320       # streaming curriculum upper bound (paper §4.2)


def n_keyframes(n_frames, interval, sf):
    return sf + max(0, (n_frames - sf + interval - 1) // interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True, help="dir from mcd_extract.py")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--K", type=int, default=28, help="student interval (deployment auto)")
    ap.add_argument("--L", type=int, default=200, help="eval window, frames")
    ap.add_argument("--B", type=int, default=72, help="fresh-run burn-in, frames")
    ap.add_argument("--n_t0", type=int, default=6)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames

    meta = np.load(os.path.join(args.frames, "meta.npz"), allow_pickle=True)
    names, gt_pos, gt_quat = meta["names"], meta["gt_pos"], meta["gt_quat"]
    S = len(names)
    T, timeshift, _ = load_extrinsic(args.calib, args.sensor)
    gcp, gcq = gt_camera_poses(gt_pos, gt_quat, T)      # convention verified in mcd_eval
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gcp, axis=0), axis=1))])
    print(f"[data] {S} frames  {dist[-1]:.0f} m  lever {np.linalg.norm(T[:3,3])*100:.1f} cm")

    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    print(f"[data] tensor {tuple(images.shape)}  ({images.numel()*4/1e9:.1f} GB)")

    t0_list = [int(round(v)) for v in
               np.linspace(args.B + sf, S - args.L, args.n_t0)]
    print(f"[plan] K={args.K} L={args.L} B={args.B}  t0={t0_list}")

    model = build_model(args.ckpt, dev, args.image_size, args.patch_size, 1024,
                        args.kv_cache_sliding_window, sf)

    res = {"meta": {**vars(args), "S": S, "total_m": float(dist[-1]),
                    "t0_list": t0_list}, "per_t0": {}}

    for t0 in t0_list:
        e0, e1 = t0, t0 + args.L
        gp, gq = gcp[e0:e1], gcq[e0:e1]

        # teacher density is capped by the remaining keyframe budget
        cache_t0 = n_keyframes(t0, args.K, sf)
        budget = max(1, KF_LIMIT - cache_t0)
        Kt = max(1, int(np.ceil(args.L / budget)))

        run_anchor(model, images, sf, dtype, dev)
        step_frames(model, images, sf, t0, sf, args.K, dtype, dev, sf)
        snap = snapshot_state(model)
        ls_p, _ = step_frames(model, images, e0, e1, sf, args.K, dtype, dev, sf)

        restore_state(model, snap)
        ld_p, _ = step_frames(model, images, e0, e1, sf, Kt, dtype, dev, sf)
        del snap

        a0 = t0 - args.B
        run_anchor(model, images[:, a0:], sf, dtype, dev)
        fd_all, _ = step_frames(model, images, a0 + sf, e1, sf, 1, dtype, dev, a0 + sf)
        fd_p = fd_all[-args.L:]
        del fd_all
        torch.cuda.empty_cache()

        out = {}
        for tag, pe in (("LS", ls_p), ("LD", ld_p), ("FD", fd_p)):
            p = pe[:, :3].double().numpy()
            q = pe[:, 3:7].double().numpy()
            out[tag] = score(p, q, gp, gq)
        out.update(cache_t0=cache_t0, Kt=Kt, ratio=args.K / Kt,
                   dist_m=float(dist[e0]), fd_kf=n_keyframes(args.B + args.L, 1, sf))
        res["per_t0"][str(t0)] = out

        print(f"\n  t0={t0:>5}  {dist[e0]:>7.0f} m from anchor | LS cache {cache_t0:>3} kf "
              f"| teacher K_t={Kt} (ratio {args.K/Kt:.1f}:1, +{n_keyframes(args.L,Kt,0)} kf)")
        print(f"    {'cond':<4} {'ATE rmse':>9} {'ATE med':>9} {'RPE rot°':>9} {'RPE tr m':>9} {'scale':>7}")
        for tag in ("LS", "LD", "FD"):
            m = out[tag]
            print(f"    {tag:<4} {m['ate_rmse']:>9.3f} {m['ate_median']:>9.3f} "
                  f"{m['rpe_rot_deg']:>9.3f} {m['rpe_trans_m']:>9.3f} {m['scale']:>7.3f}")

    print("\n=== verdict: is the fork teacher (LD) or the fresh teacher (FD) closer to GT? ===")
    print(f"  {'t0':>6} {'dist_m':>7} {'ratio':>6} | "
          f"{'LS rot':>7} {'LD rot':>7} {'FD rot':>7} | {'LS ATE':>7} {'LD ATE':>7} {'FD ATE':>7}")
    for t0 in t0_list:
        o = res["per_t0"][str(t0)]
        print(f"  {t0:>6} {o['dist_m']:>7.0f} {o['ratio']:>5.1f}x | " +
              " ".join(f"{o[c]['rpe_rot_deg']:>7.3f}" for c in ("LS", "LD", "FD")) + " | " +
              " ".join(f"{o[c]['ate_rmse']:>7.3f}" for c in ("LS", "LD", "FD")))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
