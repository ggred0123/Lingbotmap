"""Phase 0 / experiments 0-3 and 0-4 (docs/self-distill-ver3.md §8).

Implements the KV-cache fork that ver3 §4 is built on, and uses it to answer:

  0-3  Is a dense-keyframe branch forked from the student's own state actually
       BETTER than the student continuing sparse?  (If not, ver3 §4 is dead.)

  0-4  How big is the theta-mismatch loss floor when the teacher branch runs on
       different weights (the EMA case, ver3 s4.4 option B)?  Measured by
       running the teacher at K_t == K (zero density gap) so the only residual
       is the weight difference.

Fork mechanics (SDPA backend only -- FlashInfer's paged cache is not clonable
or differentiable):
  aggregator.kv_cache : dict of k_<i>/v_<i>/k_<i>_special/v_<i>_special tensors
  aggregator.total_frames_processed : int (3D RoPE temporal index)
  camera_head.kv_cache : list[dict] (one per refinement iteration)
  camera_head.frame_idx : int
"""

import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from phase0_density_sweep import build_model, quat_geodesic_deg


# ─────────────────────────────────────────────────────────────────────────────
# Fork / restore
# ─────────────────────────────────────────────────────────────────────────────

def _clone_cache_dict(d):
    out = {}
    for k, v in d.items():
        out[k] = v.clone() if torch.is_tensor(v) else copy.copy(v)
    return out


def snapshot_state(model):
    """Deep-copy the full streaming state so a branch can be rolled back."""
    agg = model.aggregator
    snap = {
        "agg_cache": _clone_cache_dict(agg.kv_cache) if isinstance(agg.kv_cache, dict) else None,
        "agg_frames": agg.total_frames_processed,
        "cam_cache": ([_clone_cache_dict(d) for d in model.camera_head.kv_cache]
                      if getattr(model.camera_head, "kv_cache", None) is not None else None),
        "cam_frame_idx": getattr(model.camera_head, "frame_idx", 0),
        "agg_pos3d": agg._cached_pos3d.clone() if torch.is_tensor(getattr(agg, "_cached_pos3d", None)) else None,
    }
    return snap


def restore_state(model, snap):
    agg = model.aggregator
    if snap["agg_cache"] is not None:
        agg.kv_cache = _clone_cache_dict(snap["agg_cache"])
    agg.total_frames_processed = snap["agg_frames"]
    agg._cached_pos3d = snap["agg_pos3d"].clone() if torch.is_tensor(snap["agg_pos3d"]) else None
    if snap["cam_cache"] is not None:
        model.camera_head.kv_cache = [_clone_cache_dict(d) for d in snap["cam_cache"]]
    model.camera_head.frame_idx = snap["cam_frame_idx"]


def snapshot_bytes(snap):
    tot = 0
    for d in ([snap["agg_cache"]] if snap["agg_cache"] else []) + (snap["cam_cache"] or []):
        for v in d.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
    return tot


# ─────────────────────────────────────────────────────────────────────────────
# Streaming primitives
# ─────────────────────────────────────────────────────────────────────────────

def run_anchor(model, images, scale_frames, dtype, dev):
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = model.forward(
            images[:, :scale_frames].to(dev),
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )
    return {k: v.float().cpu() for k, v in out.items() if torch.is_tensor(v)}


def step_frames(model, images, lo, hi, scale_frames, interval, dtype, dev,
                kf_phase_origin):
    """Stream frames [lo, hi) with the given keyframe interval.

    ``kf_phase_origin`` anchors the keyframe phase so a forked branch keeps the
    same keyframe grid as the run it forked from when interval is unchanged.
    """
    poses, depths = [], []
    for i in range(lo, hi):
        is_kf = (interval <= 1) or ((i - kf_phase_origin) % interval == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(
                images[:, i:i + 1].to(dev),
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )
        if not is_kf:
            model._set_skip_append(False)
        poses.append(out["pose_enc"].float().cpu())
        depths.append(out["depth"].float().cpu())
        del out
    return torch.cat(poses, dim=1)[0], torch.cat(depths, dim=1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def branch_diff(a_pose, a_depth, b_pose, b_depth):
    """Difference between two branches over the same frames."""
    ca, cb = a_pose[:, :3], b_pose[:, :3]
    trans = (ca - cb).norm(dim=-1)
    rot = quat_geodesic_deg(a_pose[:, 3:7], b_pose[:, 3:7])
    da, db = a_depth[..., 0], b_depth[..., 0]
    rel = (da - db).abs() / db.clamp(min=1e-6)
    S = da.shape[0]
    depth = torch.nanmedian(rel.reshape(S, -1), dim=1).values
    return {
        "trans_mean": float(trans.mean()), "trans_final": float(trans[-1]),
        "rot_mean": float(rot.mean()), "rot_final": float(rot[-1]),
        "depth_mean": float(depth.mean()),
        "trans_curve": trans.tolist(), "rot_curve": rot.tolist(),
        "depth_curve": depth.tolist(),
    }


def perturb_weights(model, eps, seed=0):
    """Multiplicative Gaussian jitter -- a stand-in for EMA weight lag."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    backup = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            backup[n] = p.detach().clone()
            noise = torch.randn(p.shape, generator=g).to(p.device, p.dtype)
            p.mul_(1.0 + eps * noise)
    return backup


def restore_weights(model, backup):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(backup[n])


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--K", type=int, default=8, help="student keyframe interval")
    ap.add_argument("--Kt", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="teacher branch intervals (include K for the 0-4 floor test)")
    ap.add_argument("--t0", type=int, default=120, help="fork point (frame index)")
    ap.add_argument("--L", type=int, default=120, help="branch length in frames")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 1e-4, 1e-3],
                    help="weight-jitter levels for the theta-mismatch floor (0-4)")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev, dtype = torch.device("cuda"), torch.bfloat16
    paths = sorted(sum([glob.glob(os.path.join(args.image_folder, f"*{e}"))
                        for e in (".png", ".jpg", ".jpeg")], []))
    need = args.t0 + args.L
    assert len(paths) >= need, f"need >= {need} frames, have {len(paths)}"
    paths = paths[:need]
    images = load_and_preprocess_images(paths, mode="crop",
                                        image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    print(f"[data] {images.shape[1]} frames  fork@{args.t0}  branch_len={args.L}")

    model = build_model(args.ckpt, dev, args.image_size, args.patch_size, 1024,
                        args.kv_cache_sliding_window, args.num_scale_frames)

    sf = args.num_scale_frames
    result = {"meta": vars(args), "branches": {}}

    # ── prefix: anchor + sparse stream up to t0 ──────────────────────────────
    print(f"[prefix] anchor({sf}) + stream to t0={args.t0} at interval K={args.K}")
    run_anchor(model, images, sf, dtype, dev)
    step_frames(model, images, sf, args.t0, sf, args.K, dtype, dev, sf)
    snap = snapshot_state(model)
    n_kf_prefix = sf + (args.t0 - sf + args.K - 1) // args.K
    print(f"[prefix] done. cached keyframes ~{n_kf_prefix}, "
          f"snapshot {snapshot_bytes(snap)/1e9:.2f} GB")
    result["meta"]["n_kf_prefix"] = n_kf_prefix
    result["meta"]["snapshot_gb"] = snapshot_bytes(snap) / 1e9

    # ── reference branch: student continues sparse at K ──────────────────────
    restore_state(model, snap)
    s_pose, s_depth = step_frames(model, images, args.t0, args.t0 + args.L, sf,
                                  args.K, dtype, dev, sf)
    print(f"[student] interval K={args.K} over {args.L} frames -> done")

    # ── fork determinism check: restore + rerun identical branch ─────────────
    restore_state(model, snap)
    s2_pose, s2_depth = step_frames(model, images, args.t0, args.t0 + args.L, sf,
                                    args.K, dtype, dev, sf)
    det = branch_diff(s2_pose, s2_depth, s_pose, s_depth)
    print(f"\n[fork determinism] restore+rerun same interval, same weights")
    print(f"  trans={det['trans_mean']:.3e}  rot={det['rot_mean']:.3e} deg  "
          f"depth={det['depth_mean']:.3e}   <- must be ~0 for a correct fork")
    result["determinism"] = det

    # ── 0-3: teacher branches at varying density, live weights ───────────────
    print(f"\n[0-3] density signal (live weights)")
    for kt in args.Kt:
        restore_state(model, snap)
        t_pose, t_depth = step_frames(model, images, args.t0, args.t0 + args.L, sf,
                                      kt, dtype, dev, sf)
        d = branch_diff(t_pose, t_depth, s_pose, s_depth)
        n_kf = n_kf_prefix + (args.L + kt - 1) // kt
        print(f"  K_t={kt:>3} (ratio {args.K/kt:>5.1f}:1, cache {n_kf:>3} kf) | "
              f"trans {d['trans_mean']:.5f} | rot {d['rot_mean']:.4f} deg | "
              f"depth {d['depth_mean']:.5f}")
        result["branches"][f"live_Kt{kt}"] = d
        result["branches"][f"live_Kt{kt}"]["n_kf_total"] = n_kf

    # ── 0-4: theta-mismatch floor at zero density gap (K_t == K) ─────────────
    print(f"\n[0-4] theta-mismatch floor (K_t = K = {args.K}, so density gap = 0)")
    for eps in args.eps:
        if eps == 0.0:
            continue
        backup = perturb_weights(model, eps)
        restore_state(model, snap)     # prefix cache was written by unperturbed theta
        p_pose, p_depth = step_frames(model, images, args.t0, args.t0 + args.L, sf,
                                      args.K, dtype, dev, sf)
        restore_weights(model, backup)
        d = branch_diff(p_pose, p_depth, s_pose, s_depth)
        print(f"  eps={eps:<8g} | trans {d['trans_mean']:.5f} | "
              f"rot {d['rot_mean']:.4f} deg | depth {d['depth_mean']:.5f}")
        result["branches"][f"floor_eps{eps}"] = d

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
