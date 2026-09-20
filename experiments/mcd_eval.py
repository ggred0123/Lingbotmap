"""Run LingBot-Map on an extracted MCD window and score it against ground truth.

Resolves two conventions empirically rather than trusting a reading of the docs:

  1. the direction of ``hhs_calib.yaml``'s ``body.<sensor>.T`` (T_body_cam vs its
     inverse) -- decided by which one yields the lower Sim(3)-aligned ATE;
  2. that the resulting GT camera trajectory is consistent with the model's,
     cross-checked with a metric that needs no extrinsic at all.

Relative-rotation error is that cross-check: an unknown camera<-body rotation
R_BC conjugates every relative rotation (dR_cam = R_BC^T dR_body R_BC) and
geodesic angle is invariant under conjugation, so rotation RPE is exact even if
the extrinsic is wrong or missing.  Translation ATE is not, which is why the
convention has to be pinned down before any translation number is quoted.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from mcd_gt import quat_to_mat, mat_to_quat
from phase0_density_sweep import build_model


def load_extrinsic(calib_yaml, sensor="d455b_color"):
    with open(calib_yaml) as f:
        c = yaml.safe_load(f)
    e = c["body"][sensor]
    return np.array(e["T"], dtype=np.float64), e.get("timeshift_cam_imu", 0.0), e


def gt_camera_poses(gt_pos, gt_quat, T, invert=False):
    """world<-body (+ body<-cam extrinsic) -> world<-cam poses."""
    n = len(gt_pos)
    Twb = np.tile(np.eye(4), (n, 1, 1))
    Twb[:, :3, :3] = quat_to_mat(gt_quat)
    Twb[:, :3, 3] = gt_pos
    Tbc = np.linalg.inv(T) if invert else T
    Twc = Twb @ Tbc[None]
    return Twc[:, :3, 3], mat_to_quat(Twc[:, :3, :3])


def umeyama(src, dst):
    """Sim(3) mapping src -> dst (scale, R, t), closed form."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = np.trace(np.diag(sig) @ W) / max((S ** 2).sum() / len(src), 1e-12)
    return s, R, mu_d - s * R @ mu_s


def rot_geodesic_deg(A, B):
    M = np.einsum("nij,njk->nik", A.transpose(0, 2, 1), B)
    cos = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
    ax = np.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0],
                   M[:, 1, 0] - M[:, 0, 1]], -1)
    return np.degrees(np.arctan2(np.linalg.norm(ax, axis=-1) / 2, cos))


def score(pred_pos, pred_quat, gt_pos, gt_quat, rpe_stride=5):
    """Sim(3)-aligned ATE plus extrinsic-invariant relative-rotation RPE."""
    s, R, t = umeyama(pred_pos, gt_pos)
    aligned = (s * (R @ pred_pos.T)).T + t
    ate = np.linalg.norm(aligned - gt_pos, axis=1)

    Rp, Rg = quat_to_mat(pred_quat), quat_to_mat(gt_quat)
    k = rpe_stride
    dRp = np.einsum("nij,njk->nik", Rp[:-k].transpose(0, 2, 1), Rp[k:])
    dRg = np.einsum("nij,njk->nik", Rg[:-k].transpose(0, 2, 1), Rg[k:])
    rpe_rot = rot_geodesic_deg(dRp, dRg)

    dp = (s * (R @ (pred_pos[k:] - pred_pos[:-k]).T)).T
    dg = gt_pos[k:] - gt_pos[:-k]
    rpe_trans = np.linalg.norm(dp - dg, axis=1)

    return {"ate_rmse": float(np.sqrt((ate ** 2).mean())),
            "ate_median": float(np.median(ate)),
            "scale": float(s),
            "rpe_rot_deg": float(np.mean(rpe_rot)),
            "rpe_trans_m": float(np.mean(rpe_trans)),
            "gt_path_m": float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window", required=True, help="dir from mcd_extract.py")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    args = ap.parse_args()

    meta = np.load(os.path.join(args.window, "meta.npz"), allow_pickle=True)
    names, gt_pos, gt_quat = meta["names"], meta["gt_pos"], meta["gt_quat"]
    T, timeshift, ent = load_extrinsic(args.calib, args.sensor)
    print(f"[calib] {args.sensor}  lever arm {np.linalg.norm(T[:3,3])*100:.1f} cm  "
          f"timeshift {timeshift*1000:.2f} ms  intrinsics {ent['intrinsics']}")

    paths = [os.path.join(args.window, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size)
    print(f"[data] {images.shape}")

    model = build_model(args.ckpt, torch.device("cuda"), args.image_size, args.patch_size,
                        1024, args.kv_cache_sliding_window, args.num_scale_frames)

    for it in args.intervals:
        model.clean_kv_cache()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model.inference_streaming(images, num_scale_frames=args.num_scale_frames,
                                             keyframe_interval=it,
                                             output_device=torch.device("cpu"))
        pe = pred["pose_enc"][0].float().numpy()
        del pred
        torch.cuda.empty_cache()
        pp, pq = pe[:, :3].astype(np.float64), pe[:, 3:7].astype(np.float64)

        print(f"\n=== keyframe_interval={it} ===")
        for inv in (False, True):
            gp, gq = gt_camera_poses(gt_pos, gt_quat, T, invert=inv)
            m = score(pp, pq, gp, gq)
            tag = "T^-1 (cam<-body)" if inv else "T     (body<-cam)"
            print(f"  {tag}: ATE rmse {m['ate_rmse']:.3f} m  median {m['ate_median']:.3f} m | "
                  f"RPE rot {m['rpe_rot_deg']:.3f}°  trans {m['rpe_trans_m']:.3f} m | "
                  f"scale {m['scale']:.4f} | gt path {m['gt_path_m']:.1f} m")


if __name__ == "__main__":
    main()
