"""Dump the full route so the drift can be drawn as a map rather than a curve.

Saves three things in world coordinates:
  gt          the survey-grade camera trajectory over all 1,401 m
  ls_global   the deployed run, aligned to GT with ONE Sim(3) over the whole route
              -- this is the view where the accumulated distortion is visible as
              a mis-shaped loop, not as a number
  fd_segments fresh runs, one per window, each aligned to GT within its own window
              (a per-window gauge has no single world placement, so segments are
              the only honest way to put fresh on the same map)
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from phase0_density_sweep import build_model
from fork_rig import run_anchor, step_frames
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama

KF_LIMIT = 320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--B", type=int, default=72)
    ap.add_argument("--stride", type=int, default=400, help="fresh-segment spacing")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames
    meta = np.load(os.path.join(args.frames, "meta.npz"), allow_pickle=True)
    names, gt_pos, gt_quat = meta["names"], meta["gt_pos"], meta["gt_quat"]
    T, _, _ = load_extrinsic(args.calib, args.sensor)
    gp_all, _ = gt_camera_poses(gt_pos, gt_quat, T)
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gp_all, axis=0), axis=1))])
    S = len(names)

    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size, 1024,
                        args.kv_cache_sliding_window, sf)
    print(f"[data] {S} frames  {dist[-1]:.0f} m")

    print("[LS] streaming whole route ...")
    run_anchor(model, images, sf, dtype, dev)
    pose, _ = step_frames(model, images, sf, S, sf, args.K, dtype, dev, sf)
    pose = torch.cat([torch.zeros(sf, 9), pose], 0)
    lp = pose[:, :3].double().numpy()
    torch.cuda.empty_cache()

    s, R, t = umeyama(lp[sf:], gp_all[sf:])
    ls_global = (s * (R @ lp.T)).T + t
    err = np.linalg.norm(ls_global - gp_all, axis=1)
    print(f"[LS] one global Sim(3): scale {s:.3f}  ATE rmse {np.sqrt((err**2).mean()):.2f} m  "
          f"max {err.max():.2f} m")

    starts = list(range(args.B + sf, S - args.L, args.stride))
    print(f"[FD] {len(starts)} fresh segments (cache {sf+args.B+args.L} kf)")
    segs, segd = [], []
    for n, e0 in enumerate(starts):
        e1 = e0 + args.L
        a0 = e0 - args.B
        run_anchor(model, images[:, a0:], sf, dtype, dev)
        fa, _ = step_frames(model, images, a0 + sf, e1, sf, 1, dtype, dev, a0 + sf)
        fp = fa[-args.L:, :3].double().numpy()
        del fa
        torch.cuda.empty_cache()
        gs_, gR_, gt_ = umeyama(fp, gp_all[e0:e1])
        segs.append((gs_ * (gR_ @ fp.T)).T + gt_)
        segd.append([e0, e1])
        if n % 5 == 0:
            print(f"  [{n+1}/{len(starts)}] {dist[e0]:.0f} m")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, gt=gp_all, ls_global=ls_global, ls_err=err,
                        dist=dist, fd_segments=np.array(segs),
                        fd_ranges=np.array(segd), global_scale=s,
                        K=args.K, L=args.L, B=args.B)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
