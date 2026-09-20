"""ATE vs distance, densely, for the deployed run and the fresh baseline.

Panel A of the verification figure had to use rotation error because
mcd_single_run.py aligns one global Sim(3) over the whole trajectory, which
spreads the residual everywhere and hides the trend (ver4 §2.4).  That was a
limitation of what got saved, not of ATE.

Here the deployed run is streamed once with its poses kept, then scored with a
**per-window** Sim(3) at many points along the route -- the same local-ATE
definition used in the three-way test, just sampled densely.  A fresh run is
launched in each window as the in-distribution baseline, so the two curves are
directly comparable in metres.

Fresh runs respect the teacher budget from ver4 §3.1b: anchor + burn-in + window
must stay under the 320-keyframe training limit.
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
from fork_rig import run_anchor, step_frames
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama, rot_geodesic_deg
from mcd_gt import quat_to_mat

KF_LIMIT = 320


def local_scores(pp, pq, gp, gq, k=5):
    s, R, t = umeyama(pp, gp)
    ate = np.linalg.norm((s * (R @ pp.T)).T + t - gp, axis=1)
    Rp, Rg = quat_to_mat(pq), quat_to_mat(gq)
    dRp = np.einsum("nij,njk->nik", Rp[:-k].transpose(0, 2, 1), Rp[k:])
    dRg = np.einsum("nij,njk->nik", Rg[:-k].transpose(0, 2, 1), Rg[k:])
    return (float(np.sqrt((ate ** 2).mean())), float(np.median(ate)),
            float(np.mean(rot_geodesic_deg(dRp, dRg))), float(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--L", type=int, default=200, help="scoring window")
    ap.add_argument("--B", type=int, default=72, help="fresh burn-in")
    ap.add_argument("--stride", type=int, default=200, help="window spacing")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert args.num_scale_frames + args.B + args.L <= KF_LIMIT, (
        f"fresh teacher would cache {args.num_scale_frames + args.B + args.L} kf "
        f"> {KF_LIMIT} (ver4 §3.1b)")

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames
    meta = np.load(os.path.join(args.frames, "meta.npz"), allow_pickle=True)
    names, gt_pos, gt_quat = meta["names"], meta["gt_pos"], meta["gt_quat"]
    T, _, _ = load_extrinsic(args.calib, args.sensor)
    gp_all, gq_all = gt_camera_poses(gt_pos, gt_quat, T)
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gp_all, axis=0), axis=1))])
    S = len(names)

    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size, 1024,
                        args.kv_cache_sliding_window, sf)
    print(f"[data] {S} frames  {dist[-1]:.0f} m")

    # ── deployed run, once, poses kept ───────────────────────────────────────
    print("[LS] streaming the whole sequence at K=28 ...")
    run_anchor(model, images, sf, dtype, dev)
    ls_pose, _ = step_frames(model, images, sf, S, sf, args.K, dtype, dev, sf)
    ls_pose = torch.cat([torch.zeros(sf, 9), ls_pose], 0)   # pad anchor slots
    ls_p = ls_pose[:, :3].double().numpy()
    ls_q = ls_pose[:, 3:7].double().numpy()
    torch.cuda.empty_cache()

    starts = list(range(args.B + sf, S - args.L, args.stride))
    print(f"[FD] {len(starts)} fresh runs (B={args.B}, L={args.L}, "
          f"cache {sf + args.B + args.L} kf)")

    rows = []
    for n, t0 in enumerate(starts):
        e0, e1 = t0, t0 + args.L
        gp, gq = gp_all[e0:e1], gq_all[e0:e1]

        a0 = e0 - args.B
        run_anchor(model, images[:, a0:], sf, dtype, dev)
        fd_all, _ = step_frames(model, images, a0 + sf, e1, sf, 1, dtype, dev, a0 + sf)
        fd = fd_all[-args.L:]
        del fd_all
        torch.cuda.empty_cache()

        ls_a, ls_m, ls_r, ls_s = local_scores(ls_p[e0:e1], ls_q[e0:e1], gp, gq)
        fd_a, fd_m, fd_r, fd_s = local_scores(fd[:, :3].double().numpy(),
                                              fd[:, 3:7].double().numpy(), gp, gq)
        rows.append(dict(t0=t0, dist=float(dist[e0]),
                         ls_ate=ls_a, ls_ate_med=ls_m, ls_rot=ls_r, ls_scale=ls_s,
                         fd_ate=fd_a, fd_ate_med=fd_m, fd_rot=fd_r, fd_scale=fd_s))
        print(f"  [{n+1:>2}/{len(starts)}] {dist[e0]:>7.0f} m | "
              f"LS ate {ls_a:>7.3f} rot {ls_r:>6.3f}° | "
              f"FD ate {fd_a:>6.3f} rot {fd_r:>6.3f}° | ratio {ls_a/max(fd_a,1e-9):>6.1f}x")

    d = np.array([r["dist"] for r in rows])
    la = np.array([r["ls_ate"] for r in rows])
    fa = np.array([r["fd_ate"] for r in rows])
    print(f"\n[summary] over {d[-1]:.0f} m")
    print(f"  LS ATE  {la[0]:.3f} -> {la[-1]:.3f} m   (max {la.max():.3f}, "
          f"corr with distance {np.corrcoef(d, la)[0,1]:+.3f})")
    print(f"  FD ATE  {fa[0]:.3f} -> {fa[-1]:.3f} m   (max {fa.max():.3f}, "
          f"corr with distance {np.corrcoef(d, fa)[0,1]:+.3f})")
    print(f"  LS/FD ratio: median {np.median(la/fa):.1f}x, max {np.max(la/fa):.1f}x")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"meta": {**vars(args), "S": S, "total_m": float(dist[-1])},
               "rows": rows}, open(args.out, "w"))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
