"""Dump per-frame predictions for LS / LD / FD over one window, for plotting.

The scoring scripts only keep summary numbers.  This re-runs the same three
conditions on a chosen window and saves the raw trajectories, per-frame errors
and a few depth maps, so the collapse can be looked at rather than inferred from
a table.

Window default targets the worst local-rotation region found by mcd_single_run
(frames ~7400-8000, i.e. 1150-1280 m from the anchor).
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
from fork_rig import snapshot_state, restore_state, run_anchor, step_frames
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama, rot_geodesic_deg
from mcd_gt import quat_to_mat

KF_LIMIT = 320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--t0", type=int, default=7400)
    ap.add_argument("--L", type=int, default=600)
    ap.add_argument("--B", type=int, default=72)
    ap.add_argument("--depth_at", type=int, nargs="+", default=[100, 300, 500])
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
    gp_all, gq_all = gt_camera_poses(gt_pos, gt_quat, T)
    dist_all = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gp_all, axis=0), axis=1))])

    e0, e1 = args.t0, args.t0 + args.L
    print(f"[window] frames [{e0},{e1})  {dist_all[e0]:.0f}->{dist_all[e1-1]:.0f} m")

    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size, 1024,
                        args.kv_cache_sliding_window, sf)

    cache_t0 = sf + max(0, (args.t0 - sf + args.K - 1) // args.K)
    Kt = max(1, int(np.ceil(args.L / max(1, KF_LIMIT - cache_t0))))
    print(f"[plan] K={args.K}  LS cache at t0 = {cache_t0} kf  ->  teacher K_t={Kt}")

    out = {}
    run_anchor(model, images, sf, dtype, dev)
    step_frames(model, images, sf, e0, sf, args.K, dtype, dev, sf)
    snap = snapshot_state(model)
    p, d = step_frames(model, images, e0, e1, sf, args.K, dtype, dev, sf)
    out["LS"] = (p, d)

    restore_state(model, snap)
    p, d = step_frames(model, images, e0, e1, sf, Kt, dtype, dev, sf)
    out["LD"] = (p, d)
    del snap

    a0 = e0 - args.B
    run_anchor(model, images[:, a0:], sf, dtype, dev)
    p, d = step_frames(model, images, a0 + sf, e1, sf, 1, dtype, dev, a0 + sf)
    out["FD"] = (p[-args.L:], d[-args.L:])
    torch.cuda.empty_cache()

    gp, gq = gp_all[e0:e1], gq_all[e0:e1]
    save = {"gt_pos": gp, "gt_quat": gq, "dist": dist_all[e0:e1],
            "t0": args.t0, "L": args.L, "K": args.K, "Kt": Kt,
            "frame_names": names[e0:e1]}

    Rg = quat_to_mat(gq)
    k = 5
    dRg = np.einsum("nij,njk->nik", Rg[:-k].transpose(0, 2, 1), Rg[k:])

    print(f"\n{'cond':<5} {'ATE rmse':>9} {'ATE med':>9} {'RPE rot°':>9} {'scale':>8}")
    for tag, (pe, dep) in out.items():
        pp = pe[:, :3].double().numpy()
        pq = pe[:, 3:7].double().numpy()
        s, R, t = umeyama(pp, gp)
        aligned = (s * (R @ pp.T)).T + t
        ate = np.linalg.norm(aligned - gp, axis=1)
        Rp = quat_to_mat(pq)
        dRp = np.einsum("nij,njk->nik", Rp[:-k].transpose(0, 2, 1), Rp[k:])
        rot = rot_geodesic_deg(dRp, dRg)
        save[f"{tag}_traj"] = aligned
        save[f"{tag}_ate"] = ate
        save[f"{tag}_rot"] = rot
        save[f"{tag}_scale"] = s
        for j in args.depth_at:
            if j < dep.shape[0]:
                save[f"{tag}_depth_{j}"] = dep[j, ..., 0].numpy().astype(np.float32)
        print(f"{tag:<5} {np.sqrt((ate**2).mean()):>9.3f} {np.median(ate):>9.3f} "
              f"{rot.mean():>9.3f} {s:>8.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **save)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
