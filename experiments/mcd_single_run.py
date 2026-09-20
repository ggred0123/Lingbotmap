"""The direct measurement: one streaming pass, one global alignment, error vs distance.

The three-way test aligns Sim(3) per evaluation window, which absorbs exactly the
accumulated drift that makes long runs fail -- it measures local quality, not the
trajectory falling apart.  This does the plain thing instead: stream the whole
sequence once the way deployment would, align to GT **once**, and report error as
a function of distance travelled.

Also tracks local scale, fitted on a sliding window, because a streaming model
whose anchor-normalised unit drifts will show up here and nowhere else.
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
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama, rot_geodesic_deg
from mcd_gt import quat_to_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--K", type=int, default=28, help="keyframe interval (deployment auto)")
    ap.add_argument("--limit", type=int, default=0, help="cap frames (0 = all)")
    ap.add_argument("--win", type=int, default=200, help="sliding window for local metrics")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    # 3D-RoPE frame-axis table size (camera_head.py:231).  It must exceed the number
    # of KEYFRAMES, which is len(seq)/K -- fine at K=28 (318 of 8894) but blown at
    # K=1 (8894), where it used to fail with a rope dim mismatch 8 minutes in.
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = np.load(os.path.join(args.frames, "meta.npz"), allow_pickle=True)
    names, gt_pos, gt_quat = meta["names"], meta["gt_pos"], meta["gt_quat"]
    if args.limit:
        names, gt_pos, gt_quat = names[:args.limit], gt_pos[:args.limit], gt_quat[:args.limit]
    T, _, _ = load_extrinsic(args.calib, args.sensor)
    gp, gq = gt_camera_poses(gt_pos, gt_quat, T)
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))])
    S = len(names)
    print(f"[data] {S} frames  {dist[-1]:.0f} m  K={args.K} "
          f"(cache -> {args.num_scale_frames + (S - args.num_scale_frames + args.K - 1)//args.K} kf)")

    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size)
    model = build_model(args.ckpt, torch.device("cuda"), args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window,
                        args.num_scale_frames)

    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model.inference_streaming(images, num_scale_frames=args.num_scale_frames,
                                         keyframe_interval=args.K,
                                         output_device=torch.device("cpu"))
    pe = pred["pose_enc"][0].double().numpy()
    del pred
    torch.cuda.empty_cache()
    pp, pq = pe[:, :3], pe[:, 3:7]

    # ── one global Sim(3), fitted over the whole run ─────────────────────────
    s, R, t = umeyama(pp, gp)
    aligned = (s * (R @ pp.T)).T + t
    ate = np.linalg.norm(aligned - gp, axis=1)
    print(f"\n[global] Sim(3) over all {S} frames: scale={s:.3f}  "
          f"ATE rmse={np.sqrt((ate**2).mean()):.3f} m  median={np.median(ate):.3f} m  "
          f"max={ate.max():.3f} m")

    # ── the curve: error vs distance travelled ───────────────────────────────
    print(f"\n{'dist(m)':>8} {'frame':>7} | {'ATE(m)':>9} {'%path':>7} | "
          f"{'localRPErot°':>13} {'local scale':>11} {'scale/初':>9}")
    rows, s0 = [], None
    nb = 12
    edges = np.linspace(0, S, nb + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 20:
            continue
        seg_ate = ate[a:b]
        # local Sim(3) on this segment -> local scale, i.e. drift of the model's unit
        ls, lR, lt = umeyama(pp[a:b], gp[a:b])
        if s0 is None:
            s0 = ls
        Rp, Rg = quat_to_mat(pq[a:b]), quat_to_mat(gq[a:b])
        k = 5
        dRp = np.einsum("nij,njk->nik", Rp[:-k].transpose(0, 2, 1), Rp[k:])
        dRg = np.einsum("nij,njk->nik", Rg[:-k].transpose(0, 2, 1), Rg[k:])
        rot = float(np.mean(rot_geodesic_deg(dRp, dRg)))
        row = dict(frame=int(a), dist=float(dist[a]),
                   ate_rmse=float(np.sqrt((seg_ate ** 2).mean())),
                   ate_pct=float(np.sqrt((seg_ate ** 2).mean()) / max(dist[b - 1], 1e-9) * 100),
                   rpe_rot=rot, local_scale=float(ls), scale_ratio=float(ls / s0))
        rows.append(row)
        print(f"{row['dist']:>8.0f} {a:>7} | {row['ate_rmse']:>9.3f} {row['ate_pct']:>6.2f}% | "
              f"{rot:>13.3f} {ls:>11.3f} {ls/s0:>9.2f}x")

    first, last = rows[0], rows[-1]
    print(f"\n[verdict] ATE {first['ate_rmse']:.3f} -> {last['ate_rmse']:.3f} m "
          f"({last['ate_rmse']/max(first['ate_rmse'],1e-9):.1f}x over {last['dist']:.0f} m)")
    print(f"          local RPE rot {first['rpe_rot']:.3f} -> {last['rpe_rot']:.3f}° "
          f"({last['rpe_rot']/max(first['rpe_rot'],1e-9):.1f}x)")
    print(f"          local scale {first['local_scale']:.3f} -> {last['local_scale']:.3f} "
          f"({last['scale_ratio']:.2f}x drift)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"meta": {**vars(args), "S": S, "total_m": float(dist[-1]),
                        "global_scale": float(s),
                        "global_ate_rmse": float(np.sqrt((ate ** 2).mean()))},
               "rows": rows, "ate": ate.tolist(), "dist": dist.tolist()},
              open(args.out, "w"))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
