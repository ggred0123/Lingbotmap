"""Can the current loss terms EXPRESS the GT-ATE direction at all?

Every question so far has been "is this term pointing at GT" -- one cosine per
term.  That cannot distinguish two very different situations:

    the needed direction is not in the span   ->  no lambda, no gamma, no
                                                  re-weighting will ever produce
                                                  it; a new axis is required
    the needed direction IS in the span but
    the current weights miss it               ->  it is a tuning problem

So project the GT gradient onto the span of the term gradients and report the
fraction of it that survives:

    C_ab = <g_a, g_b>,   b_a = <g_a, g_ATE>,   r^2 = b' C^+ b / |g_ATE|^2

r is exactly the CEILING on cos(update, GT) over all real-weighted combinations
of the terms, so it is directly comparable to the cosine the trained objective
achieves at its own weights, and the gap between them is the headroom.

★ THE NON-NEGATIVE SOLUTION IS THE ONE THAT MATTERS.  A loss weight cannot be
negative.  If the unconstrained projection is large but the NNLS fit is small,
the span contains the direction only as a DIFFERENCE of terms -- meaning the
objective would have to subtract one of its own terms to point at GT, which is a
structural conflict rather than a tuning error.  Both are reported.

★ AND THE GRAM MATRIX IS THE WHOLE COST.  No gradient has to be stored twice and
nothing has to be re-run per weight: k backwards give C and b, and every weighted
combination's norm and cosine follows in closed form.

★ THE BACKWARDS ARE bf16-AUTOCAST, so C carries ~1e-3 relative noise and a
near-collinear pair can let ``pinv`` explain that noise.  The rcond sweep and the
correlation-matrix eigenvalues are printed for exactly that reason; read r^2 as
stable only where it stops moving with rcond.

    python experiments/span_probe.py --ckpt ckpt_train/v7f.step50.pt \\
        --frames data/mcd/kth_day_10/frames_10hz --bank labels/kth_day_10 \\
        --alt_bank labels/kth_day_10_long --t0 512 --K 1 \\
        --gt_calib data/mcd/calib/hhs_calib.yaml
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.train import long_loss as LL                          # noqa: E402
from lingbot_map.train.label_bank import LabelBank                     # noqa: E402
from lingbot_map.train.losses import (                                 # noqa: E402
    PRESETS, depth_si_loss, motion_depth_loss, rel_pose_loss)
from grad_probe import detach_caches                                   # noqa: E402
from long_grad_probe import gt_window_loss, roll                       # noqa: E402
from gap_weight_probe import gdot, gnorm, gt_rot_losses                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--alt_bank", default="")
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--preset", default="A1PC")
    ap.add_argument("--long_deltas", type=int, nargs="+", default=[48, 96, 192])
    ap.add_argument("--long_alt_deltas", type=int, nargs="*", default=[48, 96, 192, 319])
    ap.add_argument("--long_lam_rot", type=float, default=15.0)
    ap.add_argument("--long_lam_dir", type=float, default=1.9)
    ap.add_argument("--long_lam_scale", type=float, default=1.0)
    ap.add_argument("--lam_long", type=float, default=0.05)
    ap.add_argument("--long_tau", type=float, default=0.5)
    ap.add_argument("--gt_calib", default="")
    ap.add_argument("--gt_sensor", default="d455b_color")
    ap.add_argument("--gt_rot_gaps", type=int, nargs="*", default=[5, 24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--grad_device", default="cuda")
    ap.add_argument("--out", default="experiments/results/span_probe.json")
    a = ap.parse_args()

    if not a.gt_calib:
        raise SystemExit("--gt_calib is required: without GT there is no target "
                         "to project onto the span")
    dev, dtype, sf, S = torch.device("cuda"), torch.bfloat16, a.num_scale_frames, a.S
    pc = PRESETS[a.preset]
    pairs, min_gap = pc["pairs"], pc.get("min_gap", 1)

    bank = LabelBank(a.bank)
    rid, off = next(((i, a.t0 - r["t0"]) for i, r in enumerate(bank.runs)
                     if r["t0"] <= a.t0 and a.t0 + S <= r["t0"] + r["L"]))
    alt_bank = alt_rid = alt_off = None
    if a.alt_bank:
        alt_bank = LabelBank(a.alt_bank)
        hit = next(((i, a.t0 - r["t0"]) for i, r in enumerate(alt_bank.runs)
                    if r["t0"] <= a.t0 and a.t0 + S <= r["t0"] + r["L"]), None)
        if hit is None:
            raise SystemExit("alt bank has no run covering this window")
        alt_rid, alt_off = hit
    print(f"[span] run {rid} off {off}   alt off {alt_off}", flush=True)

    cache = os.path.join(a.frames, f"_cache_{a.image_size}_{a.patch_size}.npy")
    if os.path.exists(cache):
        mm = np.load(cache, mmap_mode="r")
        images = torch.from_numpy(np.ascontiguousarray(mm[:a.t0 + S])).unsqueeze(0)
    else:
        from lingbot_map.train.label_bank import image_names
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        names = image_names(a.frames)[:a.t0 + S]
        images = load_and_preprocess_images(
            [os.path.join(a.frames, n) for n in names], mode="crop",
            image_size=a.image_size, patch_size=a.patch_size).unsqueeze(0)

    from phase0_density_sweep import build_model
    model = build_model(a.ckpt, dev, a.image_size, a.patch_size, a.max_frame_num,
                        a.kv_cache_sliding_window, sf)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    params = [p for _, p in model.named_parameters() if p.requires_grad]

    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                      num_frame_per_block=sf, causal_inference=True)
    detach_caches(model)
    hist = {}
    roll(model, images, sf, a.t0, sf, a.K, dtype, dev, hist)

    lab = bank.get(rid, off, S, device=dev)
    with model.masked_window(window_start=a.t0, keyframe_interval=a.K):
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, a.t0:a.t0 + S].to(dev),
                                num_frame_for_scale=sf, num_frame_per_block=S,
                                causal_inference=False)
    stu = out["pose_enc"][0].float()
    dep = out["depth"][0].float()
    tea, tdep = lab["pose_enc"].detach(), lab["depth"].detach()
    conf = lab.get("depth_conf")
    conf = conf.detach() if conf is not None else None

    # ── the axes.  Each is ONE term at unit weight ──────────────────────────
    L_rot, L_dir, L_mag = rel_pose_loss(stu, tea, mag_mode=pc["mag_mode"] if "mag_mode"
                                        in pc else "closed_form_scale",
                                        pairs=pairs, min_gap=min_gap)
    L_dep = depth_si_loss(dep, tdep, conf, mode=pc.get("depth_mode", "median"))
    L_mot = motion_depth_loss(stu, tea, dep, tdep, pairs=pairs, min_gap=min_gap)
    axes = {"rot": L_rot, "dir": L_dir, "mag": L_mag, "depth": L_dep, "motion": L_mot}

    def lcrit(ladder, terms, prefix):
        return LL.LongPoseLoss(ladder=tuple(ladder), lam_rot=a.long_lam_rot,
                               lam_dir=a.long_lam_dir, lam_scale=a.long_lam_scale,
                               tau=a.long_tau, terms=terms, prefix=prefix)

    tw = bank.poses(rid, off, off + S, device=dev)
    lp = LL.build_pairs(tuple(a.long_deltas), hist,
                        lambda lo, hi: bank.poses(rid, lo, hi, device=dev),
                        bank.runs[rid]["t0"], off, S, dev)
    for t in ("rot", "dir"):
        if lp:
            axes[f"long_{t}"] = lcrit(a.long_deltas, (t,), f"lo_{t}")(stu, tw, lp)[0]
    if alt_bank is not None:
        atw = alt_bank.poses(alt_rid, alt_off, alt_off + S, device=dev)
        ap_ = LL.build_pairs(tuple(a.long_alt_deltas), hist,
                             lambda lo, hi: alt_bank.poses(alt_rid, lo, hi, device=dev),
                             alt_bank.runs[alt_rid]["t0"], alt_off, S, dev, tea_win=atw)
        if ap_:
            axes["long_scale"] = lcrit(a.long_alt_deltas, ("scale",), "lo_s")(
                stu, atw, ap_)[0]
    print(f"[span] axes: {list(axes)}", flush=True)

    # ── the targets ─────────────────────────────────────────────────────────
    L_gt, n_gt = gt_window_loss(a.frames, a.gt_calib, a.gt_sensor, a.t0, S, stu, dev)
    if L_gt is None:
        raise SystemExit("window has too little GT")
    targets = {"ATE": L_gt}
    for k, (v, _) in gt_rot_losses(stu, a.frames, a.gt_calib, a.gt_sensor,
                                   a.t0, S, a.gt_rot_gaps, dev).items():
        targets[f"rot@{k}"] = v

    names = list(axes) + [f"TARGET_{k}" for k in targets]
    losses = list(axes.values()) + list(targets.values())
    gdev = torch.device(a.grad_device)
    G = []
    for i, L in enumerate(losses):
        gi = torch.autograd.grad(L, params, retain_graph=(i < len(losses) - 1),
                                 allow_unused=True)
        G.append(tuple(None if x is None else x.detach().to(gdev, torch.float32)
                       for x in gi))
        del gi
    print(f"[span] {len(G)} gradients", flush=True)

    k = len(axes)
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            C[i, j] = C[j, i] = gdot(G[i], G[j])
    rec = {"scene": os.path.basename(a.bank.rstrip("/")), "t0": a.t0, "K": a.K,
           "offset": off, "axes": list(axes), "gram": C.tolist(),
           "loss": {n: float(l.detach()) for n, l in zip(names, losses)},
           "targets": {}}

    d = np.sqrt(np.clip(np.diag(C), 1e-30, None))
    corr = C / np.outer(d, d)
    ev = np.linalg.eigvalsh(corr)
    rec["corr"] = corr.tolist()
    rec["corr_eigs"] = ev.tolist()

    print(f"\n  term |g| and pairwise cosine")
    print(f"  {'':<12}" + "".join(f"{n:>11}" for n in axes))
    for i, n in enumerate(axes):
        print(f"  {n:<12}" + "".join(f"{corr[i, j]:>+11.3f}" for j in range(k)))
    print(f"  {'|g|':<12}" + "".join(f"{d[i]:>11.3e}" for i in range(k)))
    print(f"\n  correlation eigenvalues: "
          + " ".join(f"{e:.2e}" for e in ev)
          + f"   (cond {ev[-1] / max(ev[0], 1e-30):.1e})")

    # current objective weights, so the achieved cosine is comparable to r
    cur = np.array([pc["lam_rot"], pc["lam_dir"], pc["lam_mag"], pc["lam_depth"],
                    pc["lam_motion"]]
                   + [a.lam_long] * (1 if "long_rot" in axes else 0)
                   + [a.lam_long] * (1 if "long_dir" in axes else 0)
                   # the alt criterion already carries lam_scale internally, so
                   # the trainer's weight on this axis is lam_long alone
                   + [a.lam_long] * (1 if "long_scale" in axes else 0))

    from scipy.optimize import nnls
    for tname in targets:
        ti = names.index(f"TARGET_{tname}")
        gt_n2 = gdot(G[ti], G[ti])
        b = np.array([gdot(G[i], G[ti]) for i in range(k)])
        row = {"|g_target|": math.sqrt(gt_n2),
               "cos_per_axis": {n: b[i] / math.sqrt(max(1e-30, C[i, i] * gt_n2))
                                for i, n in enumerate(axes)}}

        r2 = {}
        for rc in (1e-2, 1e-4, 1e-6, 1e-8):
            r2[rc] = float(b @ np.linalg.pinv(C, rcond=rc) @ b / gt_n2)
        row["r2_by_rcond"] = r2

        # NNLS on a factorisation of the Gram: A'A = C and A'c = b, so
        # ||Aw - c||^2 = w'Cw - 2b'w + const -- the same objective as ||Gw - g||^2
        w_, V = np.linalg.eigh(C)
        keep = w_ > w_.max() * 1e-10
        A = (np.sqrt(w_[keep])[:, None] * V[:, keep].T)
        c = (V[:, keep] / np.sqrt(w_[keep])).T @ b
        wnn, _ = nnls(A, c)
        resid = float(wnn @ C @ wnn - 2 * b @ wnn + gt_n2)
        row["r2_nnls"] = 1 - resid / gt_n2
        row["w_nnls"] = {n: float(wnn[i]) for i, n in enumerate(axes)}

        cw = cur[:k]
        row["cos_current"] = float(cw @ b / math.sqrt(max(1e-30, (cw @ C @ cw) * gt_n2)))
        rec["targets"][tname] = row

        print(f"\n  === target {tname} ===   |g| {row['|g_target|']:.3e}")
        print(f"  per-axis cos: " + "  ".join(
            f"{n} {row['cos_per_axis'][n]:+.3f}" for n in axes))
        print(f"  r^2 by rcond: " + "  ".join(
            f"{rc:.0e} {r2[rc]:.4f}" for rc in sorted(r2, reverse=True)))
        print(f"  r (ceiling on cos over all real weights) = "
              f"{math.sqrt(max(0, r2[1e-4])):.4f}")
        print(f"  cos at the CURRENT weights                = {row['cos_current']:+.4f}")
        print(f"  r^2 with w >= 0 (NNLS)                    = {row['r2_nnls']:.4f}"
              f"   -> cos {math.sqrt(max(0, row['r2_nnls'])):.4f}")
        print(f"  NNLS weights: " + "  ".join(
            f"{n} {row['w_nnls'][n]:.3g}" for n in axes))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(f"[write] {a.out}", flush=True)


if __name__ == "__main__":
    main()
