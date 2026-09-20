"""Per-term GRADIENT probe -- does the gate-6 diagnosis survive the right measurement?

Gate 6 concluded that ``L_rot`` is starved because it holds ~2% of the total
loss VALUE.  But the optimiser does not follow loss share, it follows GRADIENT
share, and the two are not the same number: a term can be small and steep.

``L_rot`` is the textbook case of exactly that.  It is ``acos(cos)``, and

    d acos / d cos  =  1 / sin(theta)

which BLOWS UP as the rotation error shrinks -- 36.9x at 1.55 deg, 18.3x at
3.13 deg (the two gate-6 probe values).  ``L_dir = 1 - cos`` has derivative
exactly 1 on its own cosine.  So at lam=1 the two terms differ by ~37x in local
sensitivity in the direction OPPOSITE to their loss shares.  Whether that
survives the chain to the parameters is an empirical question, and nobody has
asked it: ``trainer.grad_norms`` measures the TOTAL loss's gradient split by
PARAMETER GROUP, and ``loss_probe`` measures loss VALUES.  There is no
``autograd.grad`` anywhere in this repository.

This script asks it.  One masked window, four terms differentiated separately
against the same forward graph:

    ||dL_i/dtheta||        raw pull of each term
    lam_i * ||dL_i/dtheta||  what the optimiser actually sees
    grad share vs loss share   <- the direct test of the gate-6 diagnosis
    cos(g_i, g_j)          are two terms fighting?  (probe1's rotation got 86%
                           worse while every other term improved)
    lam* = c / ||g_i||     the lam set that equalises gradient contribution

Run it at NEAR and FAR: the whole project is about what changes under
contamination, and a term's gradient can behave differently there than its value
does (cf. §1.3, where the READ-path signal is 21x weaker at FAR).

Usage:
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 python experiments/term_grad_probe.py \\
        --ckpt ../ckpt/lingbot-map.pt --frames data/kth_day_06/frames_10hz \\
        --probes labels/kth_near:80 labels/kth_far:5248
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.train.label_bank import LabelBank
from lingbot_map.train.losses import (
    PRESETS, rel_pose_loss, depth_si_loss, motion_depth_loss)
from phase0_density_sweep import build_model
from grad_probe import detach_caches
from lingbot_map.train import abs_loss as AL

TERMS = ["L_rot", "L_dir", "L_mag", "L_motion", "L_depth"]
#: docs/gtabs-plan.md §2-2: the run-gauge terms, appended when --abs_mode is set
ABS_TERMS = {"scale": ["trans_scale", "depth_scale"],
             "paper": ["trans_scale", "depth_scale", "abs_pos", "abs_rot", "rel_trans"]}

GROUPS = [("encoder", "aggregator.patch_embed"),
          ("frame_blocks", "aggregator.frame_blocks"),
          ("global_blocks", "aggregator.global_blocks"),
          ("camera_head", "camera_head."),
          ("depth_head", "depth_head.")]


# ─────────────────────────────────────────────────────────────────────────────
# gradient algebra.  Grads are kept as parallel lists with `None` for unused
# parameters (L_rot never touches depth_head), so an unused term costs nothing.
# ─────────────────────────────────────────────────────────────────────────────

def gnorm(g, keep=None) -> float:
    it = enumerate(g)
    return math.sqrt(sum(float(x.float().pow(2).sum())
                         for i, x in it if x is not None and (keep is None or keep[i])))


def gdot(a, b) -> float:
    return sum(float((x.float() * y.float()).sum())
               for x, y in zip(a, b) if x is not None and y is not None)


def term_grads(losses, params):
    """dL_i/dtheta for every term, all against ONE forward graph."""
    grads = []
    for i, L in enumerate(losses):
        g = torch.autograd.grad(L, params, retain_graph=(i < len(losses) - 1),
                                allow_unused=True)
        grads.append([None if x is None else x.detach() for x in g])
    return grads


def roll_prefix(model, images, lo, hi, sf, K, dtype, dev, fresh, hist=None):
    """Level 0 deployed rollout over [lo, hi): no grad, detached every step.
    ``hist`` (absolute frame -> [9] pose) is filled the way RolloutPool._roll
    fills RolloutStream.hist, so the run-gauge fit sees the same stale poses
    training would."""
    if fresh:
        model.clean_kv_cache()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                          num_frame_per_block=sf, causal_inference=True)
        detach_caches(model)
        lo = sf
    for i in range(lo, hi):
        is_kf = (K <= 1) or ((i - sf) % K == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        if hist is not None:
            hist[i] = out["pose_enc"][0, 0].detach().float().cpu()
        del out
        detach_caches(model)


def probe(model, images, bank, t0, S, sf, K, lam, dtype, dev, names, params,
          opt=None, abs_cfg=None, hist=None):
    """``abs_cfg`` = {"mode": scale|paper, "fit": prefix|hist, "lam": {term: lam}}
    adds the run-gauge terms, differentiated against the same graph."""
    hit = next(((rid, t0 - r["t0"]) for rid, r in enumerate(bank.runs)
                if r["t0"] <= t0 and t0 + S <= r["t0"] + r["L"]), None)
    if hit is None:
        raise SystemExit(f"no single bank run covers [{t0}, {t0 + S})")
    lab = bank.get(*hit, S, device=dev)

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with model.masked_window(window_start=t0, keyframe_interval=K):
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, t0:t0 + S].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=S, causal_inference=False)
    fwd = time.time() - t

    stu_pose, stu_depth = out["pose_enc"][0].float(), out["depth"][0].float()
    tea_pose, tea_depth = lab["pose_enc"].detach(), lab["depth"].detach()
    conf = lab.get("depth_conf")
    conf = None if conf is None else conf.detach()

    # ★ The preset's pairs/min_gap/estimator choices MUST reach the terms, or an
    # "A1P" measurement silently re-measures A1: rel_pose_loss defaults to
    # pairs="consecutive" and would ignore the very change being evaluated.
    o = opt or {}
    _pairs = o.get("pairs", "consecutive")
    _gap = o.get("min_gap", 1)
    L_rot, L_dir, L_mag = rel_pose_loss(stu_pose, tea_pose,
                                        mag_mode=o.get("mag_mode", "closed_form_scale"),
                                        pairs=_pairs, min_gap=_gap)
    L_mot = motion_depth_loss(stu_pose, tea_pose, stu_depth, tea_depth,
                              pairs=_pairs, min_gap=_gap)
    L_dep = depth_si_loss(stu_depth, tea_depth, conf,
                          mode=o.get("depth_mode", "median"))
    losses = [L_rot, L_dir, L_mag, L_mot, L_dep]
    terms = list(TERMS)
    lam = list(lam)
    fit_parts = {}
    if abs_cfg:
        rid, off = hit
        r = bank.runs[rid]
        u = AL.median_step(bank.poses(rid, 0, r["L"])[:, :3])
        fit = AL.fit_run_gauge(hist or {}, r["t0"], off, S, stu_pose.detach(),
                               lambda lo, hi: bank.poses(rid, lo, hi), mode=abs_cfg["fit"], u=u,
                               fallback_s=lambda: AL.depth_scale_fallback(stu_depth.detach(), tea_depth, conf))
        fit_parts = {**fit.parts(), "abs_u": u, "abs_offset": off}
        for k in ABS_TERMS[abs_cfg["mode"]]:
            crit = AL.RunGaugeLoss(**{f"lam_{k}": 1.0})
            L, p = crit(stu_pose, tea_pose, stu_depth, tea_depth, conf, fit, u)
            losses.append(L)
            terms.append(k)
            lam.append(float(abs_cfg["lam"].get(k, 1.0)))
            fit_parts[f"abs_{k}_value"] = float(L.detach())
    vals = [float(x) for x in losses]

    t = time.time()
    grads = term_grads(losses, params)
    bwd = time.time() - t

    norms = [gnorm(g) for g in grads]
    # what the optimiser actually receives, and how much of it survives summing
    eff = [l * n for l, n in zip(lam, norms)]
    summed = [None if all(g[i] is None for g in grads) else
              sum((l * g[i] for l, g in zip(lam, grads) if g[i] is not None),
                  torch.zeros_like(params[i]))
              for i in range(len(params))]
    total_norm = gnorm(summed)

    rec = {
        "t0": t0, "S": S, "K": K, "lam": list(lam), "terms": terms,
        "run": {"rid": hit[0], "offset": hit[1],
                "eff_burn_in": bank.runs[hit[0]]["burn_in"] + hit[1]},
        "loss": {k: v for k, v in zip(terms, vals)},
        "L_rot_deg": vals[0] * 57.29578,
        "total_loss": sum(l * v for l, v in zip(lam, vals)),
        "grad_norm": {k: n for k, n in zip(terms, norms)},
        "grad_norm_weighted": {k: e for k, e in zip(terms, eff)},
        "loss_share": {k: (l * v) / max(sum(a * b for a, b in zip(lam, vals)), 1e-30)
                       for k, l, v in zip(terms, lam, vals)},
        "grad_share": {k: e / max(sum(eff), 1e-30) for k, e in zip(terms, eff)},
        "sensitivity_1_over_sin_rot": 1.0 / math.sin(max(vals[0], 1e-9)),
        "grad_total_summed": total_norm,
        "cancellation": total_norm / max(sum(eff), 1e-30),      # 1 = aligned, <1 = fighting
        "cos": {}, "grad_norm_by_group": {},
        "fwd_s": fwd, "bwd_s": bwd,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "abs": fit_parts,
    }
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            d = gdot(grads[i], grads[j])
            rec["cos"][f"{terms[i]}|{terms[j]}"] = (
                d / (norms[i] * norms[j]) if norms[i] * norms[j] > 0 else 0.0)
    for label, pre in GROUPS:
        keep = [n.startswith(pre) for n in names]
        if not any(keep):
            continue
        rec["grad_norm_by_group"][label] = {k: gnorm(g, keep)
                                            for k, g in zip(terms, grads)}
    # lam that equalises gradient contribution, normalised to L_dir = 1
    ref = norms[terms.index("L_dir")]
    rec["lam_for_equal_grad"] = {k: (ref / n if n > 0 else None)
                                 for k, n in zip(terms, norms)}
    # ★ THE NUMBER THE PLAN SETS lam FROM (§2-4): the weighted share of the
    # OLD terms together, so "new terms at ~50%" is one line of arithmetic.
    old_eff = sum(e for k, e in zip(terms, eff) if k in TERMS)
    rec["old_terms_weighted_norm"] = old_eff
    rec["new_terms_weighted_norm"] = sum(eff) - old_eff
    rec["new_share"] = (sum(eff) - old_eff) / max(sum(eff), 1e-30)

    del out, grads, summed, losses
    torch.cuda.empty_cache()
    return rec


def report(rec):
    terms = rec.get("terms", TERMS)
    print(f"\n{'=' * 78}\n  t0={rec['t0']}  S={rec['S']}  K={rec['K']}  "
          f"(bank run {rec['run']['rid']} offset {rec['run']['offset']}, "
          f"effective burn-in {rec['run']['eff_burn_in']} kf)\n{'=' * 78}")
    if rec.get("abs"):
        a = rec["abs"]
        print(f"  run gauge: s={a['fit_s']:.4f} mode={a['fit_mode']} n={a['fit_n']:.0f} "
              f"path={a['fit_path']:.0f}u rot(world/cam/resid)="
              f"{a['fit_rot_world_deg']:.2f}/{a['fit_rot_cam_deg']:.2f}/{a['fit_rot_resid_deg']:.2f}deg")
    print(f"  {'term':<12} {'value':>10} {'loss share':>11} "
          f"{'||dL/dth||':>12} {'x lam':>12} {'grad share':>11}")
    for k in terms:
        print(f"  {k:<12} {rec['loss'][k]:>10.4f} {rec['loss_share'][k] * 100:>10.1f}% "
              f"{rec['grad_norm'][k]:>12.4e} {rec['grad_norm_weighted'][k]:>12.4e} "
              f"{rec['grad_share'][k] * 100:>10.1f}%")
    print(f"  (L_rot = {rec['L_rot_deg']:.3f} deg, acos amplification "
          f"1/sin = {rec['sensitivity_1_over_sin_rot']:.1f}x)")
    if "new_share" in rec and rec.get("abs"):
        print(f"  new terms' weighted share at these lam: {rec['new_share'] * 100:.1f}%")
    print(f"\n  summed ||sum lam_i g_i|| = {rec['grad_total_summed']:.4e}   "
          f"vs sum of parts {sum(rec['grad_norm_weighted'].values()):.4e}   "
          f"-> {rec['cancellation'] * 100:.0f}% survives")
    print("  pairwise cosine:")
    for k, v in rec["cos"].items():
        flag = "  <-- OPPOSED" if v < -0.05 else ""
        print(f"      {k:<22} {v:+.4f}{flag}")
    print("  lam for equal gradient contribution (L_dir = 1):")
    print("      " + "  ".join(f"{k} {v:.3g}" if v else f"{k} -"
                               for k, v in rec["lam_for_equal_grad"].items()))
    print("  grad norm by parameter group:")
    print(f"      {'group':<15}" + "".join(f"{k:>13}" for k in terms))
    for g, d in rec["grad_norm_by_group"].items():
        print(f"      {g:<15}" + "".join(f"{d[k]:>13.3e}" for k in terms))
    print(f"  fwd {rec['fwd_s']:.1f}s   4x bwd {rec['bwd_s']:.1f}s   "
          f"peak {rec['peak_gb']:.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--probes", nargs="+", required=True,
                    help="bank:t0 pairs, e.g. labels/kth_near:80 labels/kth_far:5248")
    ap.add_argument("--out", default="experiments/results/term_grad_probe.json")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--preset", default="A1", choices=sorted(PRESETS),
                    help="lambda set the measured gradients are weighted with")
    ap.add_argument("--lam", type=float, nargs=5, default=None,
                    help="rot dir mag motion depth -- overrides --preset")
    ap.add_argument("--freeze_encoder", type=int, default=1,
                    help="match the trainer: the encoder does not update, so its "
                         "gradient is not what the optimiser follows")
    ap.add_argument("--abs_mode", default="off", choices=["off", "scale", "paper"],
                    help="add the run-gauge terms (docs/gtabs-plan.md §2-2), fitted "
                         "from the rolled prefix's pose history like the trainer")
    ap.add_argument("--abs_fit", default="prefix", choices=["prefix", "hist"])
    ap.add_argument("--lam_abs", default="",
                    help="weights for the new terms as trans_scale:1,depth_scale:0.5,... "
                         "(default 1 each; the raw norms are what --abs_mode reports)")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    args = ap.parse_args()

    if args.lam is None:
        _p = PRESETS[args.preset]
        args.lam = [_p["lam_rot"], _p["lam_dir"], _p["lam_mag"],
                    _p["lam_motion"], _p["lam_depth"]]
    dev, dtype, sf, S = torch.device("cuda"), torch.bfloat16, args.num_scale_frames, args.S
    jobs = []
    for p in args.probes:
        b, _, t = p.rpartition(":")
        jobs.append((b, int(t)))
    jobs.sort(key=lambda x: x[1])          # ascending, so one rollout serves all
    need = max(t for _, t in jobs) + S

    from lingbot_map.train.label_bank import image_names
    fnames = image_names(args.frames)[:need]
    if len(fnames) < need:
        raise SystemExit(f"need {need} frames, {args.frames} has {len(fnames)}")
    t = time.time()
    images = load_and_preprocess_images(
        [os.path.join(args.frames, n) for n in fnames], mode="crop",
        image_size=args.image_size, patch_size=args.patch_size).unsqueeze(0)
    print(f"[data] {len(fnames)} frames {tuple(images.shape)} in {time.time() - t:.0f}s")

    model = build_model(args.ckpt, dev, args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window, sf)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    if args.freeze_encoder:
        for n, p in model.named_parameters():
            if n.startswith("aggregator.patch_embed"):
                p.requires_grad_(False)
    names, params = zip(*[(n, p) for n, p in model.named_parameters()
                          if p.requires_grad])
    names, params = list(names), list(params)
    print(f"[model] differentiating against {sum(p.numel() for p in params) / 1e6:.0f} M "
          f"parameters (encoder {'frozen' if args.freeze_encoder else 'live'})")
    print(f"[loss] preset={args.preset}  lam = rot {args.lam[0]} dir {args.lam[1]} "
          f"mag {args.lam[2]} motion {args.lam[3]} depth {args.lam[4]}")

    abs_cfg = None
    if args.abs_mode != "off":
        lam_abs = {}
        for tok in filter(None, args.lam_abs.split(",")):
            k, _, v = tok.partition(":")
            lam_abs[k.strip()] = float(v)
        abs_cfg = {"mode": args.abs_mode, "fit": args.abs_fit, "lam": lam_abs}
        print(f"[abs] mode={args.abs_mode} fit={args.abs_fit} lam={lam_abs or 'raw (1.0)'}")
    res = {"meta": {**vars(args), "jobs": jobs}, "probes": []}
    at = 0
    hist = {} if abs_cfg else None
    for bank_path, t0 in jobs:
        bank = LabelBank(bank_path)
        t = time.time()
        roll_prefix(model, images, at, t0, sf, args.K, dtype, dev, fresh=(at == 0), hist=hist)
        print(f"\n[prefix] rolled {at} -> {t0} at K={args.K} in {time.time() - t:.0f}s")
        at = t0
        rec = probe(model, images, bank, t0, S, sf, args.K, args.lam, dtype, dev,
                    names, params, opt=PRESETS[args.preset], abs_cfg=abs_cfg, hist=hist)
        rec["bank"] = bank_path
        report(rec)
        res["probes"].append(rec)
        # the masked window is read-only (T2-e), so the stream may keep walking

    if len(res["probes"]) >= 2:
        a, b = res["probes"][0], res["probes"][-1]
        print(f"\n{'=' * 78}\n  NEAR (t0={a['t0']}) vs FAR (t0={b['t0']})\n{'=' * 78}")
        print(f"  {'term':<10} {'loss FAR/NEAR':>14} {'grad FAR/NEAR':>15} "
              f"{'grad share NEAR':>17} {'grad share FAR':>16}")
        for k in a.get("terms", TERMS):
            print(f"  {k:<12} {b['loss'][k] / max(a['loss'][k], 1e-12):>14.2f} "
                  f"{b['grad_norm'][k] / max(a['grad_norm'][k], 1e-12):>15.2f} "
                  f"{a['grad_share'][k] * 100:>16.1f}% {b['grad_share'][k] * 100:>15.1f}%")
        res["far_over_near"] = {
            "loss": {k: b["loss"][k] / max(a["loss"][k], 1e-12) for k in a.get("terms", TERMS)},
            "grad": {k: b["grad_norm"][k] / max(a["grad_norm"][k], 1e-12) for k in a.get("terms", TERMS)}}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
