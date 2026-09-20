"""VO (windowed) mode on the same route: the baseline this project has to beat.

VO resets the KV cache every window and stitches the pieces with a Sim(3) fitted
on the overlap.  It is already shipped, it is free, and it is what the paper
recommends past Direct mode's range -- so "distil a fresh teacher into Direct
mode" only earns its keep if it buys something VO does not.

Scored the same three ways as everything else, so the numbers are comparable:
  - one global Sim(3)          -> the ATE number
  - Sim(3) on the opening      -> drift, the quantity the route map draws
  - per-window Sim(3)          -> local quality, matching the LS/LD/FD tables
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# demo.py uses this class for --mode windowed. The _v2 variant additionally
# builds a point_head that the released checkpoint has no weights for (62
# missing keys), which would leave part of the baseline randomly initialised.
from lingbot_map.models.gct_stream_window import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama, rot_geodesic_deg
from mcd_gt import quat_to_mat


def local_scores(pp, pq, gp, gq, k=5):
    s, R, t = umeyama(pp, gp)
    ate = np.linalg.norm((s * (R @ pp.T)).T + t - gp, axis=1)
    Rp, Rg = quat_to_mat(pq), quat_to_mat(gq)
    dRp = np.einsum("nij,njk->nik", Rp[:-k].transpose(0, 2, 1), Rp[k:])
    dRg = np.einsum("nij,njk->nik", Rg[:-k].transpose(0, 2, 1), Rg[k:])
    return float(np.sqrt((ate ** 2).mean())), float(np.mean(rot_geodesic_deg(dRp, dRg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--window_size", type=int, default=128, help="keyframes per window")
    ap.add_argument("--keyframe_interval", type=int, default=8)
    ap.add_argument("--overlap_keyframes", type=int, default=8)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--score_win", type=int, default=200)
    ap.add_argument("--score_stride", type=int, default=200)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = np.load(os.path.join(args.frames, "meta.npz"), allow_pickle=True)
    names, gt_pos, gt_quat = meta["names"], meta["gt_pos"], meta["gt_quat"]
    T, _, _ = load_extrinsic(args.calib, args.sensor)
    gp, gq = gt_camera_poses(gt_pos, gt_quat, T)
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))])
    S = len(names)

    per_win = (args.num_scale_frames
               + (args.window_size - args.num_scale_frames) * args.keyframe_interval)
    print(f"[data] {S} frames  {dist[-1]:.0f} m")
    print(f"[VO] window_size={args.window_size} kf, keyframe_interval={args.keyframe_interval} "
          f"-> {per_win} actual frames/window, ~{S/max(per_win,1):.1f} windows")

    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size)

    model = GCTStream(img_size=args.image_size, patch_size=args.patch_size,
                      enable_3d_rope=True, max_frame_num=1024,
                      kv_cache_sliding_window=args.kv_cache_sliding_window,
                      kv_cache_scale_frames=args.num_scale_frames,
                      kv_cache_cross_frame_special=True,
                      kv_cache_include_scale_frames=True,
                      use_sdpa=True, camera_num_iterations=4)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ck.get("model", ck), strict=False)
    print(f"  missing={len(miss)} unexpected={len(unexp)}")
    model = model.to("cuda").eval()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model.inference_windowed(
            images, window_size=args.window_size,
            overlap_keyframes=args.overlap_keyframes,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=torch.device("cpu"))
    pe = pred["pose_enc"][0].double().numpy()
    del pred
    torch.cuda.empty_cache()
    n = min(len(pe), S)
    pp, pq, gp, gq, dist = pe[:n, :3], pe[:n, 3:7], gp[:n], gq[:n], dist[:n]
    print(f"[VO] stitched {n} poses")

    s, R, t = umeyama(pp, gp)
    glob = np.linalg.norm((s * (R @ pp.T)).T + t - gp, axis=1)
    n0 = 200
    s0, R0, t0 = umeyama(pp[:n0], gp[:n0])
    drift = np.linalg.norm((s0 * (R0 @ pp.T)).T + t0 - gp, axis=1)

    print(f"\n[VO] whole-route Sim(3): ATE rmse {np.sqrt((glob**2).mean()):.3f} m  "
          f"median {np.median(glob):.3f}  max {glob.max():.3f}  scale {s:.3f}")
    print(f"[VO] start-anchored drift: max {drift.max():.1f} m at "
          f"{dist[int(np.argmax(drift))]:.0f} m  |  " + "  ".join(
              f"{dist[i]:.0f}m={drift[i]:.1f}" for i in
              [int(n * f) for f in (0.1, 0.25, 0.5, 0.75, 0.99)]))

    rows = []
    print(f"\n{'dist(m)':>8} {'VO ate':>8} {'VO rot°':>8}")
    for e0 in range(args.num_scale_frames, n - args.score_win, args.score_stride):
        e1 = e0 + args.score_win
        a, r = local_scores(pp[e0:e1], pq[e0:e1], gp[e0:e1], gq[e0:e1])
        rows.append(dict(t0=e0, dist=float(dist[e0]), vo_ate=a, vo_rot=r))
        if len(rows) % 6 == 1:
            print(f"{dist[e0]:>8.0f} {a:>8.3f} {r:>8.3f}")
    la = np.array([r["vo_ate"] for r in rows])
    dd = np.array([r["dist"] for r in rows])
    print(f"\n[VO] local ATE {la[0]:.3f} -> {la[-1]:.3f} m  median {np.median(la):.3f}  "
          f"corr with distance {np.corrcoef(dd, la)[0,1]:+.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"meta": {**vars(args), "S": n, "total_m": float(dist[-1]),
                        "global_ate_rmse": float(np.sqrt((glob ** 2).mean())),
                        "global_ate_median": float(np.median(glob)),
                        "drift_max": float(drift.max()),
                        "frames_per_window": per_win},
               "rows": rows, "drift": drift.tolist(), "dist": dist.tolist()},
              open(args.out, "w"))
    np.savez_compressed(args.out.replace(".json", "_traj.npz"),
                        vo_pos=pp, vo_quat=pq, gt=gp, dist=dist)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
