"""Is the retained context worth anything?  Measured at revisits.

The case for training over resetting rests on one premise: the streaming state
carries information a fresh run does not have.  So far this data says the
opposite -- FD, with no memory at all, beat LS by 8-25x everywhere.  Before
building on "keep the context", the premise needs a test.

Revisits are where retained context should pay.  GT says when the walk comes back
to a place it saw more than a minute earlier; if the trajectory memory works, the
model's own estimate should come back to the same coordinates too.

Three measurements:
  1. loop consistency  -- GT says these two frames are <thresh apart; how far
                          apart does each method put them?  This is the direct
                          test, and a fresh run cannot even be asked (each window
                          carries its own gauge).
  2. second-visit gain -- is local error lower on the revisit than on the first
                          pass through the same place?
  3. memory ablation   -- rerun the deployed config with cross-frame special
                          tokens switched off, i.e. with trajectory memory
                          disabled.  If nothing gets worse, the memory is not
                          contributing; if things get better, it is actively
                          harmful.
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
from lingbot_map.models.gct_stream import GCTStream
from fork_rig import run_anchor, step_frames
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama, rot_geodesic_deg
from mcd_gt import quat_to_mat


def build(ckpt, image_size, patch_size, sw, sf, cross_frame_special):
    m = GCTStream(img_size=image_size, patch_size=patch_size, enable_3d_rope=True,
                  max_frame_num=1024, kv_cache_sliding_window=sw,
                  kv_cache_scale_frames=sf,
                  kv_cache_cross_frame_special=cross_frame_special,
                  kv_cache_include_scale_frames=True,
                  use_sdpa=True, camera_num_iterations=4)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model", ck), strict=False)
    return m.to("cuda").eval()


def stream_all(model, images, S, sf, K, dtype, dev):
    model.clean_kv_cache()
    run_anchor(model, images, sf, dtype, dev)
    pose, _ = step_frames(model, images, sf, S, sf, K, dtype, dev, sf)
    pose = torch.cat([torch.zeros(sf, 9), pose], 0)
    torch.cuda.empty_cache()
    return pose[:, :3].double().numpy(), pose[:, 3:7].double().numpy()


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
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--radius", type=float, default=5.0, help="revisit distance (m)")
    ap.add_argument("--min_gap_s", type=float, default=60.0)
    ap.add_argument("--win", type=int, default=200)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames
    meta = np.load(os.path.join(args.frames, "meta.npz"), allow_pickle=True)
    names, gt_pos, gt_quat, stamps = (meta["names"], meta["gt_pos"],
                                      meta["gt_quat"], meta["stamps"])
    T, _, _ = load_extrinsic(args.calib, args.sensor)
    gp, gq = gt_camera_poses(gt_pos, gt_quat, T)
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))])
    S = len(names)

    # ── revisit pairs from GT ────────────────────────────────────────────────
    sub = np.arange(0, S, 5)
    P, Ts = gp[sub], stamps[sub]
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    G = np.abs(Ts[:, None] - Ts[None])
    ii, jj = np.where((D < args.radius) & (G > args.min_gap_s))
    keep = ii < jj
    ii, jj = sub[ii[keep]], sub[jj[keep]]
    # thin to well-separated pairs
    sel, last = [], -10 ** 9
    for a, b in sorted(zip(ii, jj)):
        if a - last > 100:
            sel.append((int(a), int(b))); last = a
    print(f"[revisit] {len(sel)} well-separated pairs "
          f"(<{args.radius} m apart in GT, >{args.min_gap_s:.0f}s later)")
    for a, b in sel[:12]:
        print(f"    frame {a:>5} ({dist[a]:>6.0f} m)  <->  {b:>5} ({dist[b]:>6.0f} m)   "
              f"GT gap {np.linalg.norm(gp[a]-gp[b]):.2f} m, "
              f"{(stamps[b]-stamps[a]):.0f}s later")

    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)

    res = {"pairs": [[int(a), int(b)] for a, b in sel], "variants": {}}
    for tag, cfs in (("memory_on", True), ("memory_off", False)):
        print(f"\n=== {tag} (cross_frame_special={cfs}) ===")
        model = build(args.ckpt, args.image_size, args.patch_size,
                      args.kv_cache_sliding_window, sf, cfs)
        pp, pq = stream_all(model, images, S, sf, args.K, dtype, dev)
        del model
        torch.cuda.empty_cache()

        # 1. loop consistency, in the run's own gauge, converted to metres via
        #    a single global Sim(3) scale
        s, R, t = umeyama(pp[sf:], gp[sf:])
        al = (s * (R @ pp.T)).T + t
        loop = [float(np.linalg.norm(al[a] - al[b])) for a, b in sel]
        gtgap = [float(np.linalg.norm(gp[a] - gp[b])) for a, b in sel]

        # 2. local quality at first vs second visit
        first, second = [], []
        for a, b in sel:
            for idx, acc in ((a, first), (b, second)):
                e0 = max(sf, min(idx - args.win // 2, S - args.win))
                acc.append(local_scores(pp[e0:e0 + args.win], pq[e0:e0 + args.win],
                                        gp[e0:e0 + args.win], gq[e0:e0 + args.win])[0])
        glob = np.linalg.norm(al - gp, axis=1)
        allw = [local_scores(pp[e:e + args.win], pq[e:e + args.win],
                             gp[e:e + args.win], gq[e:e + args.win])[0]
                for e in range(sf, S - args.win, args.win)]

        res["variants"][tag] = dict(loop=loop, gt_gap=gtgap, first=first, second=second,
                                    global_ate_rmse=float(np.sqrt((glob ** 2).mean())),
                                    local_ate_median=float(np.median(allw)),
                                    local_ate=allw, scale=float(s))
        print(f"  global ATE rmse      {np.sqrt((glob**2).mean()):.2f} m")
        print(f"  local ATE median     {np.median(allw):.3f} m  ({len(allw)} windows)")
        print(f"  loop consistency     GT gap {np.mean(gtgap):.2f} m  ->  "
              f"model puts them {np.mean(loop):.1f} m apart (median {np.median(loop):.1f})")
        print(f"  first visit  local ATE  {np.mean(first):.3f} m")
        print(f"  second visit local ATE  {np.mean(second):.3f} m  "
              f"({'better' if np.mean(second) < np.mean(first) else 'worse'})")

    a_, b_ = res["variants"]["memory_on"], res["variants"]["memory_off"]
    print("\n=== verdict ===")
    print(f"  local ATE median   memory_on {a_['local_ate_median']:.3f}  "
          f"memory_off {b_['local_ate_median']:.3f}  -> "
          f"{'memory helps' if a_['local_ate_median'] < b_['local_ate_median'] else 'memory does NOT help'}")
    print(f"  global ATE rmse    memory_on {a_['global_ate_rmse']:.2f}  "
          f"memory_off {b_['global_ate_rmse']:.2f}")
    print(f"  loop consistency   memory_on {np.mean(a_['loop']):.1f} m  "
          f"memory_off {np.mean(b_['loop']):.1f} m  (GT {np.mean(a_['gt_gap']):.1f} m)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
