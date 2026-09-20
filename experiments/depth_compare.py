"""How much do the depth predictions actually differ between LS / LD / FD?

The trajectory panels show pose collapsing, but the depth maps look similar by eye.
Depth is the other half of the loss, so "how similar" needs a number rather than an
impression.

There is no depth ground truth here (that needs the LiDAR bag projected into the
camera), so this measures pairwise disagreement against FD, which the pose results
established as the in-distribution reference.  Each map is median-normalised first:
the runs live in different gauges, so raw depth is not comparable and only the
shape of the depth field is.

Metrics, all scale-invariant after median normalisation:
  AbsRel   mean |d_a - d_b| / d_b
  delta<t  fraction of pixels with max(d_a/d_b, d_b/d_a) < t   (higher = closer)
  RMSElog  sqrt(mean (log d_a - log d_b)^2)
  grad     mean |∇d_a - ∇d_b| / mean |∇d_b|   -- structure, not offset
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

KF_LIMIT = 320


def metrics(a, b, eps=1e-6):
    """a, b: [H,W] depth. Median-normalise each, then compare."""
    a = a / max(float(np.median(a)), eps)
    b = b / max(float(np.median(b)), eps)
    m = (a > eps) & (b > eps)
    a, b = a[m], b[m]
    ratio = np.maximum(a / b, b / a)
    ga = np.gradient(np.log(np.clip(a, eps, None)))
    gb = np.gradient(np.log(np.clip(b, eps, None)))
    return dict(
        absrel=float(np.mean(np.abs(a - b) / b)),
        d125=float(np.mean(ratio < 1.25)),
        d110=float(np.mean(ratio < 1.10)),
        rmselog=float(np.sqrt(np.mean((np.log(a) - np.log(b)) ** 2))),
    )


def grad_metric(a, b, eps=1e-6):
    a = a / max(float(np.median(a)), eps)
    b = b / max(float(np.median(b)), eps)
    gay, gax = np.gradient(a)
    gby, gbx = np.gradient(b)
    num = np.mean(np.hypot(gay - gby, gax - gbx))
    den = np.mean(np.hypot(gby, gbx)) + eps
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--t0", type=int, default=7400)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--B", type=int, default=72)
    ap.add_argument("--t0_ref", type=int, default=80,
                    help="a near-anchor window as the uncorrupted control")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames
    meta = np.load(os.path.join(args.frames, "meta.npz"), allow_pickle=True)
    names = meta["names"]
    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size, 1024,
                        args.kv_cache_sliding_window, sf)

    save = {}
    for label, t0 in (("far", args.t0), ("near", args.t0_ref)):
        e0, e1 = t0, t0 + args.L
        cache = sf + max(0, (t0 - sf + args.K - 1) // args.K)
        Kt = max(1, int(np.ceil(args.L / max(1, KF_LIMIT - cache))))

        run_anchor(model, images, sf, dtype, dev)
        step_frames(model, images, sf, e0, sf, args.K, dtype, dev, sf)
        snap = snapshot_state(model)
        _, d_ls = step_frames(model, images, e0, e1, sf, args.K, dtype, dev, sf)
        restore_state(model, snap)
        _, d_ld = step_frames(model, images, e0, e1, sf, Kt, dtype, dev, sf)
        del snap
        a0 = e0 - args.B
        run_anchor(model, images[:, a0:], sf, dtype, dev)
        _, d_fd_all = step_frames(model, images, a0 + sf, e1, sf, 1, dtype, dev, a0 + sf)
        d_fd = d_fd_all[-args.L:]
        del d_fd_all
        torch.cuda.empty_cache()

        D = {"LS": d_ls[..., 0].numpy(), "LD": d_ld[..., 0].numpy(),
             "FD": d_fd[..., 0].numpy()}
        print(f"\n=== {label}: t0={t0}  (LS cache {cache} kf, teacher K_t={Kt}) ===")
        print(f"  {'pair':<10} {'AbsRel':>8} {'δ<1.10':>8} {'δ<1.25':>8} {'RMSElog':>8} {'grad':>7}")
        for pair in (("LS", "FD"), ("LD", "FD"), ("LS", "LD")):
            per = [metrics(D[pair[0]][i], D[pair[1]][i]) for i in range(args.L)]
            g = float(np.mean([grad_metric(D[pair[0]][i], D[pair[1]][i])
                               for i in range(0, args.L, 4)]))
            agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            key = f"{label}_{pair[0]}_vs_{pair[1]}"
            save[key] = np.array([agg["absrel"], agg["d110"], agg["d125"],
                                  agg["rmselog"], g])
            save[key + "_absrel_per_frame"] = np.array([p["absrel"] for p in per])
            print(f"  {pair[0]}-{pair[1]:<7} {agg['absrel']:>8.4f} {agg['d110']:>8.4f} "
                  f"{agg['d125']:>8.4f} {agg['rmselog']:>8.4f} {g:>7.4f}")

        # keep one frame's maps for the figure
        j = args.L // 2
        for tag in D:
            save[f"{label}_{tag}_depth"] = D[tag][j].astype(np.float32)
        save[f"{label}_frame_idx"] = np.array([e0 + j])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **save)
    print(f"\n[saved] {args.out}")
    print("\nRead: 'near' is the uncorrupted control (10 m from anchor, all three agree).")
    print("      'far' minus 'near' is how much of the depth disagreement is caused by")
    print("      accumulated corruption rather than by keyframe density alone.")


if __name__ == "__main__":
    main()
