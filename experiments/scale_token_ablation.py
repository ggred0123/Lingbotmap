"""Does the anchor's scale token actually set the run's unit -- and for what?

The aggregator carries a dedicated ``scale_token`` (stream.py:184) and it is the
ONLY token whose distinguished variant spans the whole anchor block:

    camera_token    slice_expand_and_flatten(..)                       -> frame 0 only
    register_token  slice_expand_and_flatten(..)                       -> frame 0 only
    scale_token     slice_expand_and_flatten(.., first_num_frame=sf)   -> all 8 anchor frames

Both variants are alive in the released checkpoint (cos = -0.15), it survives
eviction forever as part of the special-6 block, and NEITHER head reads it
directly -- it acts only through attention.  So "the anchor's scale token sets
the unit" is a plausible reading, not a verified one.  This ablates it.

Monocular scale is not observable, so the anchor cannot CALIBRATE anything; the
question is only whether it FIXES THE UNIT, and whether that unit is shared:

    A  anchor marker removed   scale_token[:,0] = scale_token[:,1]
    B  scale token silenced    scale_token[:,:] = 0

The decisive statistic is dimensionless motion per unit scene depth,

    u = median ||dt|| / median D

which is INVARIANT under a pure change of unit.  So:

    depth and pose move together, u unchanged  ->  one shared run unit
    only depth moves, u changes                ->  the token is depth-specific
    nothing moves                              ->  the token is vestigial

Run at NEAR and FAR: if the ablation bites at t0=80 but not at t0=5248, the
anchor has lost its grip on the unit over 5248 frames -- which would be direct
evidence for scale-token-path corruption as the contamination mechanism (a
suspect v4 §13 does not list).

Nothing here needs a teacher, labels, or GT: every number is a ratio against
this script's own baseline on the same frames.

Usage:
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 python experiments/scale_token_ablation.py \\
        --ckpt ../ckpt/lingbot-map.pt --frames data/kth_day_06/frames_10hz --t0 80 5248
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.train.losses import quat_to_R
from phase0_density_sweep import build_model

CONDS = ["baseline", "A_no_anchor_marker", "B_token_silenced",
         "C_control_register", "D_control_random"]


def apply_cond(model, cond, orig, orig_reg=None, rand=None):
    """C and D are NEGATIVE CONTROLS.  Any single-token edit perturbs the rollout;
    without them a small shift under B cannot be told apart from generic
    sensitivity to touching a special token at all."""
    tok = model.aggregator.scale_token
    reg = model.aggregator.register_token
    with torch.no_grad():
        tok.data.copy_(orig)
        reg.data.copy_(orig_reg)
        if cond == "A_no_anchor_marker":
            tok.data[:, 0] = tok.data[:, 1]       # anchor loses its distinct marker
        elif cond == "B_token_silenced":
            tok.data.zero_()                       # no scale token content at all
        elif cond == "C_control_register":
            reg.data.zero_()                       # a different special token, silenced
        elif cond == "D_control_random":
            tok.data.copy_(rand)                   # same norm, learned value destroyed


@torch.no_grad()
def rollout_and_measure(model, images, t0, S, sf, K, dtype, dev):
    """Fresh anchor -> deployed rollout to t0 -> collect the window [t0, t0+S)."""
    model.clean_kv_cache()
    with torch.amp.autocast("cuda", dtype=dtype):
        model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                      num_frame_per_block=sf, causal_inference=True)
    poses, depths = [], []
    for i in range(sf, t0 + S):
        is_kf = (K <= 1) or ((i - sf) % K == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        if i >= t0:
            poses.append(out["pose_enc"].detach().float().cpu()[0])
            d = out["depth"].detach().float().cpu()[0]
            depths.append((d[..., 0] if d.dim() == 4 else d).reshape(1, -1).median())
        del out
    p = torch.cat(poses, 0)
    med_d = torch.stack(depths)                                   # [S] per-frame median depth
    R = quat_to_R(p[:, 3:7])
    dt = torch.einsum("nij,nj->ni", R[:-1].transpose(1, 2), p[1:, :3] - p[:-1, :3])
    mag = dt.norm(dim=-1)                                         # [S-1] step magnitudes
    return {
        "depth_geo_mean": float(med_d.log().mean().exp()),
        "depth_log_std": float(med_d.log().std()),                # within-window scale drift
        "mag_median": float(mag.median()),
        "u": float(mag.median() / med_d[:-1].median()),           # motion per unit depth
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--t0", type=int, nargs="+", default=[80, 5248])
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--out", default="experiments/results/scale_token_ablation.json")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    args = ap.parse_args()

    dev, dtype, sf, S = torch.device("cuda"), torch.bfloat16, args.num_scale_frames, args.S
    need = max(args.t0) + S
    names = sorted(os.listdir(args.frames))[:need]
    images = load_and_preprocess_images([os.path.join(args.frames, n) for n in names],
                                        mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window, sf)
    model.eval()
    orig = model.aggregator.scale_token.data.clone()
    orig_reg = model.aggregator.register_token.data.clone()
    g = torch.Generator(device="cpu").manual_seed(0)
    rand = torch.randn(orig.shape, generator=g).to(orig.device, orig.dtype)
    rand *= (orig.flatten(2).norm(dim=-1, keepdim=True).unsqueeze(-1)
             / rand.flatten(2).norm(dim=-1, keepdim=True).unsqueeze(-1))
    v0, v1 = orig[0, 0], orig[0, 1]
    print(f"[token] scale_token {tuple(orig.shape)}  ||anchor variant|| {v0.norm():.4e}  "
          f"||rest|| {v1.norm():.4e}  cos {torch.nn.functional.cosine_similarity(v0.flatten(), v1.flatten(), dim=0):+.4f}")

    res = {"meta": vars(args), "runs": []}
    for t0 in sorted(args.t0):
        print(f"\n{'=' * 78}\n  t0={t0}  S={S}  K={args.K}\n{'=' * 78}")
        base = None
        for cond in CONDS:
            apply_cond(model, cond, orig, orig_reg, rand)
            t = time.time()
            m = rollout_and_measure(model, images, t0, S, sf, args.K, dtype, dev)
            m.update({"t0": t0, "cond": cond, "sec": time.time() - t})
            if cond == "baseline":
                base = m
            m["depth_ratio"] = m["depth_geo_mean"] / base["depth_geo_mean"]
            m["mag_ratio"] = m["mag_median"] / base["mag_median"]
            m["u_ratio"] = m["u"] / base["u"]
            res["runs"].append(m)
            print(f"  {cond:<20} depth {m['depth_geo_mean']:>9.4f} ({m['depth_ratio']:>6.3f}x)   "
                  f"|dt| {m['mag_median']:>9.5f} ({m['mag_ratio']:>6.3f}x)   "
                  f"u {m['u']:>8.5f} ({m['u_ratio']:>6.3f}x)   "
                  f"drift(log std) {m['depth_log_std']:.4f}   {m['sec']:.0f}s")
        apply_cond(model, "baseline", orig, orig_reg, rand)

    print(f"\n{'=' * 78}\n  판정\n{'=' * 78}")
    for t0 in sorted(args.t0):
        rows = [r for r in res["runs"] if r["t0"] == t0]
        for r in rows[1:]:
            moved = max(abs(r["depth_ratio"] - 1), abs(r["mag_ratio"] - 1))
            if moved < 0.02:
                verdict = "영향 없음 -> 이 토큰은 단위를 정하지 않는다"
            elif abs(r["u_ratio"] - 1) < 0.05:
                verdict = "depth·pose가 함께 이동, u 불변 -> 공유된 run 단위"
            else:
                verdict = "u가 이동 -> 두 모달리티에 다르게 작용 (공유 단위 아님)"
            print(f"  t0={t0:<5} {r['cond']:<20} {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
