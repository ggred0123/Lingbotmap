"""Phase 0 / experiment 0-1 (GT-free variant, docs/self-distill-ver3.md §8.1).

Sweep ``keyframe_interval`` over the SAME frame span and measure how far the
sparse-keyframe runs drift from the dense (interval=1) run.

Why absolute comparison is valid here
-------------------------------------
``inference_streaming`` processes the first ``num_scale_frames`` (=8) anchor
frames as one bidirectional block *before* keyframe logic kicks in
(``i >= scale_frames`` loop).  So every run in the sweep shares the identical
anchor context, hence the identical gauge (origin, orientation, scale).  No
Sim(3) alignment is needed -- pose_enc can be diffed directly.  This is the
same property the fork design exploits (ver3 §4.1).

Outputs a JSON with per-frame drift curves.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images


def build_model(ckpt_path, device, image_size, patch_size, max_frame_num,
                kv_cache_sliding_window, num_scale_frames):
    model = GCTStream(
        img_size=image_size,
        patch_size=patch_size,
        enable_3d_rope=True,
        max_frame_num=max_frame_num,
        kv_cache_sliding_window=kv_cache_sliding_window,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,              # flashinfer unavailable; also the differentiable path
        camera_num_iterations=4,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  e.g. missing: {missing[:5]}")
    if unexpected:
        print(f"  e.g. unexpected: {unexpected[:5]}")
    return model.to(device).eval()


def quat_geodesic_deg(q_a, q_b):
    """Angle between two unit quaternions [N,4] (XYZW, scalar-last), in degrees.

    Uses the relative-quaternion atan2 form rather than ``arccos(dot)``: arccos
    is ill-conditioned near dot=1 and produces a ~0.01 deg noise floor even when
    comparing a run against itself, which would swamp the small-drift regime we
    care about here.
    """
    q_a = q_a / q_a.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    q_b = q_b / q_b.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # q_rel = q_b^{-1} (x) q_a, with q_b^{-1} = conj(q_b) for unit quaternions.
    ax, ay, az, aw = q_a.unbind(-1)
    bx, by, bz, bw = q_b.unbind(-1)
    bx, by, bz = -bx, -by, -bz
    rw = bw * aw - bx * ax - by * ay - bz * az
    rx = bw * ax + bx * aw + by * az - bz * ay
    ry = bw * ay - bx * az + by * aw + bz * ax
    rz = bw * az + bx * ay - by * ax + bz * aw
    vec = torch.stack([rx, ry, rz], dim=-1).norm(dim=-1)
    return torch.rad2deg(2.0 * torch.atan2(vec, rw.abs()))  # abs -> double cover


def run_one(model, images, interval, num_scale_frames, dtype):
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = model.inference_streaming(
            images,
            num_scale_frames=num_scale_frames,
            keyframe_interval=interval,
            output_device=torch.device("cpu"),
        )
    out = {
        "pose_enc": pred["pose_enc"][0].float().clone(),      # [S,9]
        "depth": pred["depth"][0].float().clone(),            # [S,H,W,1]
        "depth_conf": pred["depth_conf"][0].float().clone(),  # [S,H,W]
    }
    del pred
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=1024)
    ap.add_argument("--first_k", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    paths = sorted(
        sum([glob.glob(os.path.join(args.image_folder, f"*{e}"))
             for e in (".png", ".jpg", ".jpeg", ".JPG", ".PNG")], [])
    )
    if args.stride > 1:
        paths = paths[::args.stride]
    if args.first_k:
        paths = paths[:args.first_k]
    print(f"[data] {len(paths)} frames from {args.image_folder}")

    images = load_and_preprocess_images(
        paths, mode="crop", image_size=args.image_size, patch_size=args.patch_size
    )
    print(f"[data] tensor {tuple(images.shape)}")

    print("[model] building...")
    model = build_model(args.ckpt, device, args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window,
                        args.num_scale_frames)

    runs = {}
    for it in args.intervals:
        n_kf = args.num_scale_frames + max(0, (len(paths) - args.num_scale_frames + it - 1) // it)
        print(f"[run] keyframe_interval={it}  (~{n_kf} keyframes cached)")
        runs[it] = run_one(model, images, it, args.num_scale_frames, dtype)

    ref_it = min(args.intervals)
    ref = runs[ref_it]
    S = ref["pose_enc"].shape[0]

    # Trajectory scale for normalizing translation drift (anchor-normalized units).
    c_ref = ref["pose_enc"][:, :3]
    path_len = (c_ref[1:] - c_ref[:-1]).norm(dim=-1).cumsum(0)
    path_len = torch.cat([torch.zeros(1), path_len])
    total_len = float(path_len[-1])
    print(f"[ref] interval={ref_it}  total path length = {total_len:.4f} (anchor units)")

    result = {
        "meta": {
            "image_folder": args.image_folder,
            "n_frames": S,
            "intervals": args.intervals,
            "ref_interval": ref_it,
            "num_scale_frames": args.num_scale_frames,
            "kv_cache_sliding_window": args.kv_cache_sliding_window,
            "total_path_length_anchor_units": total_len,
        },
        "path_len": path_len.tolist(),
        "runs": {},
    }

    for it in args.intervals:
        r = runs[it]
        c = r["pose_enc"][:, :3]
        q = r["pose_enc"][:, 3:7]

        trans_drift = (c - c_ref).norm(dim=-1)                       # absolute, anchor units
        rot_drift = quat_geodesic_deg(q, ref["pose_enc"][:, 3:7])    # degrees

        # Local (RPE-style) drift: relative pose between consecutive frames.
        d_c = c[1:] - c[:-1]
        d_c_ref = c_ref[1:] - c_ref[:-1]
        rpe_trans = (d_c - d_c_ref).norm(dim=-1)

        # Depth: median relative deviation per frame, masked by reference confidence.
        d = r["depth"][..., 0]
        d0 = ref["depth"][..., 0]
        conf = ref["depth_conf"]
        valid = (d0 > 1e-6) & (conf > 1.5)
        rel = torch.where(valid, (d - d0).abs() / d0.clamp(min=1e-6),
                          torch.full_like(d0, float("nan")))
        depth_relmed = torch.nanmedian(rel.reshape(S, -1), dim=1).values

        result["runs"][str(it)] = {
            "trans_drift": trans_drift.tolist(),
            "rot_drift_deg": rot_drift.tolist(),
            "rpe_trans_drift": rpe_trans.tolist(),
            "depth_rel_median": depth_relmed.tolist(),
        }

        def q4(x):
            x = x[~torch.isnan(x)]
            if x.numel() == 0:
                return [float("nan")] * 4
            return [float(x[: max(1, len(x) // 4)].mean()),
                    float(x[len(x) // 4: len(x) // 2].mean()),
                    float(x[len(x) // 2: 3 * len(x) // 4].mean()),
                    float(x[3 * len(x) // 4:].mean())]

        print(f"\n  --- interval={it} vs ref={ref_it} ---")
        print(f"  trans drift  final={float(trans_drift[-1]):.5f}  "
              f"mean={float(trans_drift.mean()):.5f}  "
              f"(% of path len: {100*float(trans_drift[-1])/max(total_len,1e-9):.2f}%)")
        print(f"    quartile means: {['%.5f' % v for v in q4(trans_drift)]}")
        print(f"  rot drift    final={float(rot_drift[-1]):.4f} deg  "
              f"mean={float(rot_drift.mean()):.4f} deg")
        print(f"    quartile means: {['%.4f' % v for v in q4(rot_drift)]}")
        print(f"  RPE trans    mean={float(rpe_trans.mean()):.6f}")
        print(f"    quartile means: {['%.6f' % v for v in q4(rpe_trans)]}")
        dm = depth_relmed[~torch.isnan(depth_relmed)]
        print(f"  depth relerr mean={float(dm.mean()):.5f}")
        print(f"    quartile means: {['%.5f' % v for v in q4(depth_relmed)]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
