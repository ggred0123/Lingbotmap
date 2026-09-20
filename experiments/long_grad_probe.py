"""Does L_long take a gradient share comparable to local A1PC, and does it pull
the same way?  docs/long_supervision_design.md section 10, plan gate G7.

The optimiser follows GRADIENT, not loss value, and this run clips at |g|=1 every
step -- so any overall scale is renormalised away and only the RATIO between the
local and long terms, and the ANGLE between them, decide what the update does.
Setting lam_long from the loss scalar would be setting it from the one number
that does not survive clipping.

    ||dL_local/dtheta||   ||dL_long/dtheta||   share   cos(local, long)
    per-delta norms and per-delta cosines
    per parameter group, so "which module does the long term actually move" is
    answerable rather than assumed

★ ONE WINDOW, REAL STATE.  The prefix is rolled exactly as the trainer rolls it
(no grad, detached each step) and the past poses are captured on the way, so the
long anchors are the same stale-by-construction poses training would use -- not
a fresh re-decode that would flatter the term.

    python experiments/long_grad_probe.py --ckpt ... --frames data/mcd/kth_day_10/frames_10hz \
        --bank labels/kth_day_10 --t0 272 --K 1

★ THE TWO-SOURCE SPLIT HAS TO BE PROBED THE WAY IT IS TRAINED.  v7e/v7f take
rot+dir from the in-run bank and scale from the stitched ``_long`` track, and
Delta=319 exists ONLY in the stitched one.  Probing with the single-bank default
measures ``--long_terms rot,dir,scale`` off the in-run bank -- which is v7d, the
configuration the GT audit says is wrong for scale, and which has no 319 rung at
all.  Pass ``--alt_bank`` to reproduce the cell that is actually training:

    python experiments/long_grad_probe.py --ckpt ckpt_train/v7f.step50.pt \\
        --frames data/mcd/kth_day_10/frames_10hz \\
        --bank labels/kth_day_10 --long_terms rot,dir \\
        --alt_bank labels/kth_day_10_long --long_alt_terms scale \\
        --long_alt_deltas 48 96 192 319 --lam_long 0.05 --t0 272 --K 1

★ AND IT HAS TO BE A DRIFTED CHECKPOINT.  design section 10: a share measured at
frozen theta_0 sets lam about 40x too high, because local A1PC is already at its
own optimum there while the long relations start large.  Probe a step-30..50
checkpoint against the local-only control, never the released weights.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.train import long_loss as LL                          # noqa: E402
from lingbot_map.train.label_bank import LabelBank                     # noqa: E402
from lingbot_map.train.losses import PRESETS, SelfDistillLoss          # noqa: E402
from grad_probe import detach_caches                                   # noqa: E402
from phase0_density_sweep import build_model                           # noqa: E402
from term_grad_probe import gdot, gnorm                                # noqa: E402

def gt_window_loss(frames_dir, calib, sensor, t0, S, stu_pose, dev):
    """ATE of the supervised window against GT, differentiable through the poses.

    ★ THE Sim(3) ALIGNMENT IS DETACHED ON PURPOSE.  ATE is defined after a global
    alignment, so the alignment is part of the METRIC, not of the thing being
    optimised; letting gradient flow into s, R, t would let the window lower the
    loss by moving the frame it is measured in.  Fitting it on detached poses and
    applying it as a constant gives exactly ``d ATE / d pose``.

    ★ THE TWO INDEXINGS DO NOT AGREE.  The bank walks the frames directory;
    meta.npz lists only the frames that have GT.  Rows without GT are dropped
    rather than silently paired against the wrong frame -- same rule as
    seam_audit.gt_by_disk_index.

    Returns ``(loss, n_used)`` or ``(None, 0)`` when the window has too little GT.
    """
    import numpy as np
    from lingbot_map.train.label_bank import image_names
    from mcd_eval import gt_camera_poses, load_extrinsic, umeyama

    meta = np.load(os.path.join(frames_dir, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(calib, sensor)
    gp, _ = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    row_of = {str(n): i for i, n in enumerate(meta["names"])}
    rows = [row_of.get(n, -1) for n in image_names(frames_dir)[t0:t0 + S]]
    keep = [k for k, r in enumerate(rows) if r >= 0]
    if len(keep) < 8:
        return None, len(keep)

    gt = torch.as_tensor(gp[[rows[k] for k in keep]], dtype=torch.float32, device=dev)
    p = stu_pose[keep, :3]
    s, R, t = umeyama(p.detach().double().cpu().numpy(), gt.double().cpu().numpy())
    R_t = torch.as_tensor(R, dtype=torch.float32, device=dev)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=dev)
    aligned = float(s) * (p @ R_t.T) + t_t
    return (aligned - gt).norm(dim=-1).mean(), len(keep)


PARAM_GROUPS = [("encoder", "aggregator.patch_embed"),
                ("frame_blocks", "aggregator.frame_blocks"),
                ("global_blocks", "aggregator.global_blocks"),
                ("camera_head", "camera_head."),
                ("depth_head", "depth_head.")]


def roll(model, images, lo, hi, sf, K, dtype, dev, hist):
    for i in range(lo, hi):
        is_kf = (K <= 1) or ((i - sf) % K == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        hist[i] = out["pose_enc"][0, 0].detach().float().cpu()
        del out
        detach_caches(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--t0", type=int, required=True,
                    help="window start; must satisfy off >= max(delta) for the "
                         "whole ladder to fire")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--preset", default="A1PC")
    ap.add_argument("--long_deltas", type=int, nargs="+", default=[48, 96, 192])
    ap.add_argument("--long_terms", default="rot,dir,scale",
                    help="terms served by --bank.  v7e/v7f use 'rot,dir'.")
    ap.add_argument("--alt_bank", default="",
                    help="stitched track (labels/<scene>_long).  Serves "
                         "--long_alt_terms at --long_alt_deltas; this is the only "
                         "source that has a Delta=319 rung.")
    ap.add_argument("--long_alt_deltas", type=int, nargs="*", default=[48, 96, 192, 319])
    ap.add_argument("--long_alt_terms", default="scale")
    ap.add_argument("--lam_long", type=float, default=1.0,
                    help="the lam the run uses; the share is reported both at "
                         "lam=1 and at this value.")
    ap.add_argument("--target_share", type=float, default=0.25,
                    help="design section 10's target long share; the lam that "
                         "reaches it is solved for and printed.")
    ap.add_argument("--per_delta", action="store_true",
                    help="also take a gradient per rung.  Off by default: each "
                         "one is another full-size gradient copy, and C.md's "
                         "decision table is per TERM, not per Delta.")
    ap.add_argument("--gt_calib", default="",
                    help="MCD calib yaml.  Given, the probe also takes the "
                         "gradient of the window's GT ATE, which is the only "
                         "direction here that is not a teacher's opinion.")
    ap.add_argument("--gt_sensor", default="d455b_color")
    ap.add_argument("--lam_scale_sweep", type=float, nargs="*", default=[],
                    help="extra --long_lam_scale values to report.  ★ NO EXTRA "
                         "ROLLOUT OR BACKWARD IS NEEDED: with terms disjoint the "
                         "alt criterion is exactly linear in lam_scale, so "
                         "g_alt(s) = (s/s0)*g_alt(s0) and the norm of the sum "
                         "follows from |g_long|, |g_alt| and their inner product.")
    ap.add_argument("--long_lam_rot", type=float, default=15.0)
    ap.add_argument("--long_lam_dir", type=float, default=1.9)
    ap.add_argument("--long_lam_scale", type=float, default=1.0)
    ap.add_argument("--long_tau", type=float, default=0.5)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--out", default="experiments/results/long_grad_probe.json")
    a = ap.parse_args()

    dev, dtype, sf, S = torch.device("cuda"), torch.bfloat16, a.num_scale_frames, a.S
    bank = LabelBank(a.bank)
    hit = next(((rid, a.t0 - r["t0"]) for rid, r in enumerate(bank.runs)
                if r["t0"] <= a.t0 and a.t0 + S <= r["t0"] + r["L"]), None)
    if hit is None:
        raise SystemExit(f"no single bank run covers [{a.t0}, {a.t0 + S})")
    rid, off = hit
    print(f"[probe] run {rid} offset {off}  (needs off >= delta for each rung)")

    # ★ THE ALT BANK IS LOCATED SEPARATELY, AND ITS OFFSET IS NOT ``off``.  The
    # stitched track is one long run per scene, so the same absolute t lands at a
    # different offset there -- and Delta=319 needs aoff >= 319 - S, which ``off``
    # (capped at L-S = 192 on the in-run bank) can never reach.
    alt_bank = alt_rid = alt_off = None
    if a.alt_bank:
        alt_bank = LabelBank(a.alt_bank)
        ahit = next(((r_i, a.t0 - r["t0"]) for r_i, r in enumerate(alt_bank.runs)
                     if r["t0"] <= a.t0 and a.t0 + S <= r["t0"] + r["L"]), None)
        if ahit is None:
            raise SystemExit(f"alt bank has no single run covering "
                             f"[{a.t0}, {a.t0 + S}) -- pick another --t0")
        alt_rid, alt_off = ahit
        print(f"[probe] alt run {alt_rid} offset {alt_off}  "
              f"(Delta=319 needs aoff >= {319 - S})")

    cache = os.path.join(a.frames, f"_cache_{a.image_size}_{a.patch_size}.npy")
    if os.path.exists(cache):
        import numpy as np
        mm = np.load(cache, mmap_mode="r")
        images = torch.from_numpy(np.ascontiguousarray(mm[:a.t0 + S])).unsqueeze(0)
    else:
        from lingbot_map.train.label_bank import image_names
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        names = image_names(a.frames)[:a.t0 + S]
        images = load_and_preprocess_images(
            [os.path.join(a.frames, n) for n in names], mode="crop",
            image_size=a.image_size, patch_size=a.patch_size).unsqueeze(0)

    model = build_model(a.ckpt, dev, a.image_size, a.patch_size, a.max_frame_num,
                        a.kv_cache_sliding_window, sf)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    names_p = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in model.named_parameters() if p.requires_grad]

    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                      num_frame_per_block=sf, causal_inference=True)
    detach_caches(model)
    hist = {}
    roll(model, images, sf, a.t0, sf, a.K, dtype, dev, hist)
    print(f"[probe] rolled to {a.t0}, {len(hist)} past poses held")

    lab = bank.get(rid, off, S, device=dev)
    with model.masked_window(window_start=a.t0, keyframe_interval=a.K):
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, a.t0:a.t0 + S].to(dev),
                                num_frame_for_scale=sf, num_frame_per_block=S,
                                causal_inference=False)
    stu = out["pose_enc"][0].float()

    crit = SelfDistillLoss.from_preset(a.preset)
    L_local, lp = crit(stu, lab["pose_enc"], out["depth"][0].float(),
                       lab["depth"], lab.get("depth_conf"))

    terms = tuple(t.strip() for t in a.long_terms.split(",") if t.strip())
    aterms = tuple(t.strip() for t in a.long_alt_terms.split(",") if t.strip())

    def crit(ladder, tms, prefix="long"):
        return LL.LongPoseLoss(ladder=tuple(ladder), lam_rot=a.long_lam_rot,
                               lam_dir=a.long_lam_dir, lam_scale=a.long_lam_scale,
                               tau=a.long_tau, terms=tms, prefix=prefix)

    long_crit = crit(a.long_deltas, terms)
    tea_win = bank.poses(rid, off, off + S, device=dev)
    pairs = LL.build_pairs(long_crit.ladder, hist,
                           lambda lo, hi: bank.poses(rid, lo, hi, device=dev),
                           bank.runs[rid]["t0"], off, S, dev)
    if not pairs:
        raise SystemExit(f"no long pair at offset {off} for ladder "
                         f"{list(long_crit.ladder)} -- pick --t0 with off >= delta")
    print(f"[probe] in-run deltas firing: {sorted(pairs)}  terms={terms}")

    # per-delta, so lam_delta can be set from these rather than from the scalar
    per = {}
    for d in sorted(pairs):
        per[d] = crit((d,), terms)(stu, tea_win, {d: pairs[d]})[0]
    L_long, gp_ = long_crit(stu, tea_win, pairs)

    # ── the stitched source, scored in ITS OWN gauge ─────────────────────────
    # ★ ``tea_win`` MUST COME FROM THE ALT BANK for the alt pairs.  Both endpoints
    # of a pair have to sit in one teacher's gauge; mixing the in-run window with
    # a stitched anchor would make the "relation" the difference between two
    # teachers.  trainer.long_term passes tea_win for exactly this reason.
    L_alt, ap_, aper = None, {}, {}
    if alt_bank is not None and aterms:
        alt_crit = crit(a.long_alt_deltas, aterms, prefix="longalt")
        atea = alt_bank.poses(alt_rid, alt_off, alt_off + S, device=dev)
        apairs = LL.build_pairs(alt_crit.ladder, hist,
                                lambda lo, hi: alt_bank.poses(alt_rid, lo, hi, device=dev),
                                alt_bank.runs[alt_rid]["t0"], alt_off, S, dev,
                                tea_win=atea)
        if not apairs:
            raise SystemExit(f"no alt pair at aoff {alt_off} for ladder "
                             f"{list(alt_crit.ladder)}")
        print(f"[probe] alt deltas firing: {sorted(apairs)}  terms={aterms}")
        for d in sorted(apairs):
            aper[d] = crit((d,), aterms, "longalt")(stu, atea, {d: apairs[d]})[0]
        L_alt, ap_ = alt_crit(stu, atea, apairs)

    # what the trainer actually scales by lam_long is the SUM of the two sources
    L_long_total = L_long if L_alt is None else L_long + L_alt

    # ── one criterion per TERM, because the aggregate hides the sign ─────────
    # ★ rot AND dir SHARE A SOURCE BUT NOT A FAILURE MODE.  The GT audit puts the
    # stitched track at 3.96-13.57 deg of rotation error against 1.12 in-run,
    # while its scale bias is the one that is near zero -- so "does the long term
    # fight the local one" has a different answer per term, and a single
    # cos(local, long) cannot say which.
    by_term = {}
    for tname in ("rot", "dir"):
        if tname in terms:
            by_term[tname] = crit(a.long_deltas, (tname,), f"only_{tname}")(
                stu, tea_win, pairs)[0]
    if L_alt is not None and "scale" in aterms:
        by_term["scale"] = L_alt          # the alt criterion is scale-only already

    # ── GT: the only direction here that no teacher chose ───────────────────
    L_gt, n_gt = (None, 0)
    if a.gt_calib:
        L_gt, n_gt = gt_window_loss(a.frames, a.gt_calib, a.gt_sensor,
                                    a.t0, S, stu, dev)
        print(f"[probe] GT window ATE on {n_gt}/{S} frames"
              + ("" if L_gt is None else f"  L_gt={float(L_gt.detach()):.4f}"))

    # ★ ONLY THE INDEPENDENT TERMS GET A BACKWARD.  Every autograd.grad here
    # materialises a full copy of the parameter gradient -- 4.6 GB at this model
    # size -- and retain_graph keeps the activations alive for all of them, so the
    # list length is the memory budget.  L_long is exactly L_term_rot +
    # L_term_dir (same lam_sum, disjoint terms) and L_long_total adds the alt
    # criterion, so those two gradients are VECTOR SUMS of ones already taken,
    # not new backwards.  Taking them anyway is what OOMs beside a running bench.
    named = [("L_local", L_local)] + \
            [(f"L_term_{k}", v) for k, v in by_term.items()] + \
            ([("L_gt", L_gt)] if L_gt is not None else []) + \
            ([(f"L_long_d{d}", v) for d, v in per.items()] +
             [(f"L_longalt_d{d}", v) for d, v in aper.items()] if a.per_delta else [])
    grads = []
    for i, (_, L) in enumerate(named):
        gi = torch.autograd.grad(L, params, retain_graph=(i < len(named) - 1),
                                 allow_unused=True)
        # off the GPU immediately; gnorm/gdot are device-agnostic
        grads.append(tuple(None if x is None else x.detach().to("cpu", torch.float32)
                           for x in gi))
        del gi
        torch.cuda.empty_cache()

    g = dict(zip([n for n, _ in named], grads))

    def gadd(*keys):
        out = None
        for k in keys:
            v = g.get(k)
            if v is None:
                continue
            out = v if out is None else tuple(
                (y if x is None else x if y is None else x + y) for x, y in zip(out, v))
        return out

    g["L_long"] = gadd(*[f"L_term_{k}" for k in ("rot", "dir") if k in by_term])
    if "scale" in by_term:
        g["L_longalt"] = g["L_term_scale"]
    g["L_long_total"] = gadd(*[f"L_term_{k}" for k in by_term])
    rec = {"t0": a.t0, "run": rid, "offset": off, "K": a.K,
           "preset": a.preset, "ladder": list(long_crit.ladder),
           "terms": list(terms), "lam_long": a.lam_long,
           "alt_bank": a.alt_bank or None,
           "alt_ladder": list(a.long_alt_deltas) if L_alt is not None else None,
           "alt_terms": list(aterms) if L_alt is not None else None,
           "alt_offset": alt_off,
           "loss": {**{k: float(v.detach()) for k, v in named},
                    "L_long": float(L_long.detach()),
                    "L_long_total": float(L_long_total.detach()),
                    **({"L_longalt": float(L_alt.detach())} if L_alt is not None else {})},
           "local_parts": lp, "long_parts": {**gp_, **ap_},
           "grad_norm": {k: gnorm(v) for k, v in g.items() if v is not None}}

    gl, gt = rec["grad_norm"]["L_local"], rec["grad_norm"]["L_long_total"]
    # share at lam=1 -- what the old single-source probe reported
    rec["share_long"] = gt / max(1e-12, gl + gt)
    # ★ THE NUMBER THE lam DECISION ACTUALLY NEEDS.  The optimiser sees
    # lam_long * dL_long, so the share at the CONFIGURED lam is what tells you
    # whether the term is switched on, and it is not the share at lam=1.
    rec["share_long_at_lam"] = (a.lam_long * gt) / max(1e-12, gl + a.lam_long * gt)
    # invert it: which lam puts the long term at --target_share
    tgt = min(max(a.target_share, 1e-6), 1 - 1e-6)
    rec["lam_for_target_share"] = (tgt * gl) / max(1e-12, (1 - tgt) * gt)
    rec["target_share"] = tgt

    rec["cos"] = {"local|long": gdot(g["L_local"], g["L_long_total"]) /
                  max(1e-12, gnorm(g["L_local"]) * gnorm(g["L_long_total"]))}
    if a.per_delta:
        for d in per:
            rec["cos"][f"local|d{d}"] = gdot(g["L_local"], g[f"L_long_d{d}"]) / max(
                1e-12, gnorm(g["L_local"]) * gnorm(g[f"L_long_d{d}"]))
        for d in aper:
            rec["cos"][f"local|alt_d{d}"] = gdot(g["L_local"], g[f"L_longalt_d{d}"]) / max(
                1e-12, gnorm(g["L_local"]) * gnorm(g[f"L_longalt_d{d}"]))

    # ── the per-term table C.md asks for ────────────────────────────────────
    def cs(x, y):
        return gdot(g[x], g[y]) / max(1e-12, gnorm(g[x]) * gnorm(g[y]))
    rec["term_grad_norm"] = {k: gnorm(g[f"L_term_{k}"]) for k in by_term}
    for k in by_term:
        rec["cos"][f"local|{k}"] = cs("L_local", f"L_term_{k}")
    if L_gt is not None:
        rec["n_gt"] = n_gt
        rec["grad_norm"]["L_gt"] = gnorm(g["L_gt"])
        # ★ THESE TWO DECIDE THE CELL IN C.md's TABLE.  A long term that fights
        # local is only a problem if it also fights GT; one that agrees with
        # local while both fight GT is the worst case and looks fine locally.
        rec["cos"]["local|gt"] = cs("L_local", "L_gt")
        rec["cos"]["long|gt"] = cs("L_long_total", "L_gt")
        for k in by_term:
            rec["cos"][f"{k}|gt"] = cs(f"L_term_{k}", "L_gt")
        if a.per_delta:
            for d in aper:
                rec["cos"][f"alt_d{d}|gt"] = cs(f"L_longalt_d{d}", "L_gt")
    ds = sorted(per)
    if a.per_delta:
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                ka, kb = f"L_long_d{ds[i]}", f"L_long_d{ds[j]}"
                rec["cos"][f"d{ds[i]}|d{ds[j]}"] = gdot(g[ka], g[kb]) / max(
                    1e-12, gnorm(g[ka]) * gnorm(g[kb]))

    rec["by_group"] = {}
    for gname, pref in PARAM_GROUPS:
        # ★ gnorm's ``keep`` is a per-parameter BOOLEAN MASK, not an index list.  Passing indices makes it read keep[i] for every i and walk off the end.
        keep = [n.startswith(pref) for n in names_p]
        rec["by_group"][gname] = {k: gnorm(v, keep) for k, v in g.items()}

    print(f"\n  loss   local {float(L_local):.4f}   long {float(L_long_total):.4f}"
          + (f"  (in-run {float(L_long):.4f} + alt {float(L_alt):.4f})"
             if L_alt is not None else ""))
    print(f"  |g|    local {gl:.4e}   long {gt:.4e}")
    print(f"  long share   at lam=1  {rec['share_long']:.1%}"
          f"   at lam={a.lam_long:g}  {rec['share_long_at_lam']:.1%}")
    print(f"  cos(local, long) = {rec['cos']['local|long']:+.4f}")
    if by_term:
        print(f"\n  {'term':<8}{'|g|':>12}{'cos(local)':>12}"
              + (f"{'cos(GT)':>10}" if L_gt is not None else ""))
        for k in ("rot", "dir", "scale"):
            if k not in by_term:
                continue
            print(f"  {k:<8}{rec['term_grad_norm'][k]:>12.3e}"
                  f"{rec['cos'][f'local|{k}']:>+12.4f}"
                  + (f"{rec['cos'][f'{k}|gt']:>+10.4f}" if L_gt is not None else ""))
    if L_gt is not None:
        print(f"\n  |g_GT| {rec['grad_norm']['L_gt']:.3e}   "
              f"cos(local, GT) {rec['cos']['local|gt']:+.4f}   "
              f"cos(long, GT) {rec['cos']['long|gt']:+.4f}")
    if a.per_delta:
        for d in ds:
            print(f"    in-run d{d:<4} |g| {rec['grad_norm'][f'L_long_d{d}']:.4e}  "
                  f"cos(local) {rec['cos'][f'local|d{d}']:+.4f}  "
                  f"loss {float(per[d].detach()):.4f}")
        for d in sorted(aper):
            print(f"    alt    d{d:<4} |g| {rec['grad_norm'][f'L_longalt_d{d}']:.4e}  "
                  f"cos(local) {rec['cos'][f'local|alt_d{d}']:+.4f}  "
                  f"loss {float(aper[d].detach()):.4f}")
    print(f"\n  {'group':<15}{'local':>13}{'long':>13}{'long share':>12}")
    for gname, _ in PARAM_GROUPS:
        b = rec["by_group"][gname]
        tot = b["L_local"] + b["L_long_total"]
        print(f"  {gname:<15}{b['L_local']:>13.3e}{b['L_long_total']:>13.3e}"
              f"{(b['L_long_total'] / tot if tot else 0):>11.1%}")

    # ── lam_scale sweep, derived rather than re-measured ─────────────────────
    # ★ ONLY VALID BECAUSE --long_terms AND --long_alt_terms ARE DISJOINT, which
    # trainer.py refuses to run without.  lam_scale then multiplies the alt
    # criterion and nothing else, so its gradient is exactly linear in it and the
    # combined norm is a quadratic in lam_scale -- no second backward.
    if a.lam_scale_sweep and L_alt is not None:
        ga = rec["grad_norm"]["L_longalt"]
        dot = gdot(g["L_long"], g["L_longalt"])
        s0 = a.long_lam_scale
        rows = []
        for s in a.lam_scale_sweep:
            k = s / s0                                   # g_alt(s) = k * g_alt(s0)
            gt_s = (gnorm(g["L_long"]) ** 2 + 2 * k * dot + (k * ga) ** 2) ** 0.5
            alt_share = (k * ga) / max(1e-12, gnorm(g["L_long"]) + k * ga)
            rows.append({
                "lam_scale": s,
                "g_alt": k * ga,
                "g_long_total": gt_s,
                "alt_share_within_long": alt_share,
                "lam_long_for_target": (tgt * gl) / max(1e-12, (1 - tgt) * gt_s),
            })
        rec["lam_scale_sweep"] = rows
        print(f"\n  lam_scale sweep  (alt share within the long term, and the "
              f"lam_long that then gives {tgt:.0%} overall)")
        print(f"  {'lam_scale':>10}{'|g_alt|':>12}{'|g_long_tot|':>14}"
              f"{'alt share':>11}{'lam_long':>10}")
        for r in rows:
            print(f"  {r['lam_scale']:>10g}{r['g_alt']:>12.3e}"
                  f"{r['g_long_total']:>14.3e}"
                  f"{r['alt_share_within_long']:>10.2%}"
                  f"{r['lam_long_for_target']:>10.4f}")

    # what lam_long would equalise the two shares at this window
    rec["lam_long_for_parity"] = gl / max(1e-12, gt)
    print(f"\n  lam_long for |g| parity (50%) at this window: "
          f"{rec['lam_long_for_parity']:.4f}")
    print(f"  lam_long for the design's {tgt:.0%} share:            "
          f"{rec['lam_for_target_share']:.4f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()
