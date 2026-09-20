"""Phase 0 / decisive test: does ver1's fresh-anchor teacher carry information
that ver3's KV-fork teacher does not?

Three conditions evaluated on the SAME frame window [t0, t0+L):

  LS  long-sparse   : full history from frame 0 at interval K   (what deploys)
  LD  long-dense    : fork the LS state at t0, continue at interval 1
  FD  fresh-dense   : brand-new run, anchor at t0-B, stream dense through t0+L
                      (= ver1's teacher, with burn-in B)

LS vs LD  -> density component            (already known to be large)
LS vs FD  -> the whole "long vs fresh" gap
LD vs FD  -> the residual after removing density.  THIS is the number that
             decides whether the fresh teacher is redundant.

FD has a different anchor, hence a different gauge, so all metrics here are
gauge-invariant by construction -- no Sim(3)/Umeyama fitting anywhere:
  * relative rotation between consecutive frames   (invariant to global R,t,s)
  * translation direction in the local camera frame (invariant to global R,t,s)
  * per-step translation magnitude ratio, reported as log-std (a constant
    ratio is just the gauge scale; only its variation is a real disagreement)
  * depth normalized by its own per-frame median

t0 is swept so we can see whether each gap GROWS with elapsed length.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.rotation import quat_to_mat
from phase0_density_sweep import build_model
from fork_rig import snapshot_state, restore_state, run_anchor, step_frames


# ─────────────────────────────────────────────────────────────────────────────
# Gauge-invariant local metrics
# ─────────────────────────────────────────────────────────────────────────────

def _rot_geodesic_deg(M):
    """Stable rotation angle of a batch of rotation matrices [N,3,3]."""
    cos = (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) / 2.0
    ax = torch.stack([M[:, 2, 1] - M[:, 1, 2],
                      M[:, 0, 2] - M[:, 2, 0],
                      M[:, 1, 0] - M[:, 0, 1]], dim=-1)
    sin = ax.norm(dim=-1) / 2.0
    return torch.rad2deg(torch.atan2(sin, cos))


def local_frames(pose):
    """pose [S,9] (c2w center + xyzw quat) -> per-step relative rotation,
    local-frame translation direction, and step magnitude."""
    c = pose[:, :3]
    R = quat_to_mat(pose[:, 3:7])                 # [S,3,3]
    dR = torch.bmm(R[:-1].transpose(1, 2), R[1:])  # [S-1,3,3]
    dc = c[1:] - c[:-1]
    dc_local = torch.bmm(R[:-1].transpose(1, 2), dc.unsqueeze(-1)).squeeze(-1)
    mag = dc_local.norm(dim=-1)
    dir_ = dc_local / mag.clamp(min=1e-9).unsqueeze(-1)
    return dR, dir_, mag


def gauge_invariant_gap(pose_a, depth_a, pose_b, depth_b, motion_frac=0.1):
    """All-invariant disagreement between two runs over the same frames."""
    dRa, dira, maga = local_frames(pose_a)
    dRb, dirb, magb = local_frames(pose_b)

    # relative rotation disagreement
    rot = _rot_geodesic_deg(torch.bmm(dRa.transpose(1, 2), dRb))

    # translation direction disagreement (skip near-stationary steps: direction
    # is undefined there and would inject pure noise)
    ref = magb.median()
    moving = (maga > motion_frac * ref) & (magb > motion_frac * ref)
    dot = (dira * dirb).sum(-1).clamp(-1, 1)
    cross = torch.cross(dira, dirb, dim=-1).norm(dim=-1)
    ang = torch.rad2deg(torch.atan2(cross, dot))
    dir_deg = ang[moving]

    # per-step scale ratio -> only its VARIATION is a real disagreement
    lr = torch.log(maga.clamp(min=1e-9) / magb.clamp(min=1e-9))[moving]

    # depth, each frame normalized by its own median (scale-invariant)
    da, db = depth_a[..., 0], depth_b[..., 0]
    S = da.shape[0]
    ma = da.reshape(S, -1).median(dim=1).values.view(S, 1, 1).clamp(min=1e-9)
    mb = db.reshape(S, -1).median(dim=1).values.view(S, 1, 1).clamp(min=1e-9)
    rel = ((da / ma) - (db / mb)).abs() / (db / mb).clamp(min=1e-6)
    depth = rel.reshape(S, -1).median(dim=1).values

    return {
        "rel_rot_deg": float(rot.mean()),
        "dir_deg": float(dir_deg.mean()) if dir_deg.numel() else float("nan"),
        "scale_logstd": float(lr.std()) if lr.numel() > 1 else float("nan"),
        "depth_si": float(depth.mean()),
        "n_moving": int(moving.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--L", type=int, default=80, help="eval window length")
    ap.add_argument("--B", type=int, default=72, help="fresh-run burn-in (anchor + window fill)")
    ap.add_argument("--n_t0", type=int, default=5)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev, dtype = torch.device("cuda"), torch.bfloat16
    sf = args.num_scale_frames

    paths = sorted(sum([glob.glob(os.path.join(args.image_folder, f"*{e}"))
                        for e in (".png", ".jpg", ".jpeg")], []))
    S = len(paths)
    t0_lo, t0_hi = args.B + sf, S - args.L
    assert t0_hi > t0_lo, f"sequence too short: S={S}, need > {t0_lo + args.L}"
    t0_list = [int(round(v)) for v in np.linspace(t0_lo, t0_hi, args.n_t0)]
    name = os.path.basename(args.image_folder.rstrip("/"))
    print(f"[{name}] S={S}  K={args.K}  L={args.L}  B={args.B}  t0={t0_list}")

    images = load_and_preprocess_images(paths, mode="crop",
                                        image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size, 1024,
                        args.kv_cache_sliding_window, sf)

    result = {"meta": {**vars(args), "scene": name, "S": S, "t0_list": t0_list},
              "per_t0": {}}

    for t0 in t0_list:
        e0, e1 = t0, t0 + args.L

        # ── LS: long history at interval K, continue sparse ──────────────────
        run_anchor(model, images, sf, dtype, dev)
        step_frames(model, images, sf, t0, sf, args.K, dtype, dev, sf)
        snap = snapshot_state(model)
        ls_p, ls_d = step_frames(model, images, e0, e1, sf, args.K, dtype, dev, sf)

        # ── LD: fork the same state, continue dense ──────────────────────────
        restore_state(model, snap)
        ld_p, ld_d = step_frames(model, images, e0, e1, sf, 1, dtype, dev, sf)
        del snap

        # ── FD: fresh run anchored at t0-B, dense throughout ─────────────────
        a0 = t0 - args.B
        run_anchor(model, images[:, a0:], sf, dtype, dev)
        fd_all_p, fd_all_d = step_frames(model, images, a0 + sf, e1, sf, 1,
                                         dtype, dev, a0 + sf)
        fd_p, fd_d = fd_all_p[-args.L:], fd_all_d[-args.L:]
        del fd_all_p, fd_all_d
        torch.cuda.empty_cache()

        g = {
            "LS_vs_LD": gauge_invariant_gap(ls_p, ls_d, ld_p, ld_d),
            "LS_vs_FD": gauge_invariant_gap(ls_p, ls_d, fd_p, fd_d),
            "LD_vs_FD": gauge_invariant_gap(ld_p, ld_d, fd_p, fd_d),
        }
        n_kf = sf + (t0 - sf + args.K - 1) // args.K
        g["n_kf_at_t0"] = n_kf
        result["per_t0"][str(t0)] = g

        print(f"\n  t0={t0:>4} (LS cache {n_kf} kf, eval [{e0},{e1}))")
        print(f"    {'pair':<10} {'rel_rot°':>9} {'dir°':>8} {'scale_sd':>9} {'depth_si':>9}")
        for k in ("LS_vs_LD", "LS_vs_FD", "LD_vs_FD"):
            v = g[k]
            print(f"    {k:<10} {v['rel_rot_deg']:>9.4f} {v['dir_deg']:>8.3f} "
                  f"{v['scale_logstd']:>9.4f} {v['depth_si']:>9.5f}")

    # ── trend: does each gap grow with elapsed length? ───────────────────────
    print(f"\n[{name}] trend across t0  (first -> last, ratio)")
    for pair in ("LS_vs_LD", "LS_vs_FD", "LD_vs_FD"):
        for metric in ("rel_rot_deg", "dir_deg", "depth_si"):
            vals = [result["per_t0"][str(t)][pair][metric] for t in t0_list]
            r = vals[-1] / vals[0] if vals[0] > 1e-12 else float("nan")
            print(f"  {pair:<10} {metric:<12} " +
                  " ".join(f"{v:8.4f}" for v in vals) + f"   ratio={r:5.2f}x")
    result["t0_list"] = t0_list

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
