#!/usr/bin/env python3
"""What is the BEST alignment with GT any reweighting of A1PC's terms can reach?

gap_weight_probe.py measured cos(g_local, g_GT) for the shipped weights and for a
gap-reweighted rotation term.  Both answers were near zero: median +0.051 over
twelve windows, 87 degrees off the direction that reduces GT trajectory error.
That says the objective is BLIND, not hostile -- it never came out negative.

The open question is whether that is fixable by reweighting.  A1PC is a linear
combination, so it can be answered exactly rather than by sweeping:

    g(lam) = sum_i lam_i g_i           g_i = dL_i/dtheta for each of the five terms
    cos(g(lam), g_GT) = (lam.b) / (sqrt(lam.G.lam) * |g_GT|)
        G_ij = <g_i, g_j>              b_i = <g_i, g_GT>

That is a Rayleigh quotient: the maximum over lam is sqrt(b.G^-1.b)/|g_GT|, at
lam* proportional to G^-1 b.  So one backward per term gives both the CEILING and
the weights that reach it.  If the ceiling is still small, no choice of lam_rot /
lam_dir / lam_mag / lam_depth / lam_motion can fix the objective and the missing
information is not in this span -- which is the case for a new term (lam_long).

★ depth_head is structurally orthogonal to g_GT: the window ATE is a function of
the poses alone, so no gradient reaches the depth head from it.  Since depth_head
is the LARGEST block of |g_local| (15.5 of it, against camera_head's 3.0), a
cosine over all parameters is diluted by construction.  Both are reported, and
the pose-subspace number is the honest one.

    .venv-bench/bin/python experiments/term_align_probe.py \
        --ckpt ckpt/lingbot-map.pt --scene kth_day_10 --t0 512 --K 1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.train.label_bank import LabelBank                    # noqa: E402
from lingbot_map.train.losses import (                                # noqa: E402
    PRESETS, rel_pose_loss, depth_si_loss, motion_depth_loss)
from grad_probe import detach_caches                                  # noqa: E402
from long_grad_probe import PARAM_GROUPS, gt_window_loss, roll        # noqa: E402
from lingbot_map.train.trainer import image_names                     # noqa: E402
from lingbot_map.utils.load_fn import load_and_preprocess_images      # noqa: E402
from lingbot_map.train import abs_loss as AL                          # noqa: E402

TERMS = ["L_rot", "L_dir", "L_mag", "L_depth_si", "L_motion_depth"]
#: docs/gtabs-plan.md §5-3-4: the run-gauge terms ride along when --abs_mode is
#: set.  Their gradient is taken against the same graph and scored against TWO
#: GT directions: the window-local Sim(3) ATE (L_gt, the +0.09 number) and the
#: run-gauge ATE (abs_pos itself, fitted from the rolled history), which is the
#: quantity L_abs-pos measures -- so its own cosine there must be ~1, and the
#: scale terms' must sit strictly between A1PC's and that.
ABS_TERMS = {"off": [], "scale": ["trans_scale", "depth_scale"],
             "paper": ["trans_scale", "depth_scale", "abs_pos", "abs_rot", "rel_trans"]}


def keep_mask(names, drop_prefixes):
    return [not any(p in n for p in drop_prefixes) for n in names]


def gdot(a, b, keep=None):
    s = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if x is None or y is None or (keep is not None and not keep[i]):
            continue
        s += float((x * y).sum())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt")
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--frames", default="")
    ap.add_argument("--bank", default="")
    ap.add_argument("--t0", type=int, default=512)
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--preset", default="A1PC")
    ap.add_argument("--gt_calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--gt_sensor", default="d455b_color")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--grad_device", default="cuda")
    ap.add_argument("--abs_mode", default="off", choices=["off", "scale", "paper"])
    ap.add_argument("--abs_fit", default="prefix", choices=["prefix", "hist"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.frames = a.frames or f"data/mcd/{a.scene}/frames_10hz"
    a.bank = a.bank or f"labels/{a.scene}"
    S, sf, dev, dtype = a.S, a.num_scale_frames, torch.device("cuda"), torch.bfloat16

    bank = LabelBank(a.bank)
    # same selection gap_weight_probe.py uses: the run whose span covers t0
    hit = next(((rid, a.t0 - r["t0"]) for rid, r in enumerate(bank.runs)
                if r["t0"] <= a.t0 and a.t0 + S <= r["t0"] + r["L"]), None)
    if hit is None:
        raise SystemExit(f"no run in {a.bank} covers [{a.t0}, {a.t0+S})")
    rid, off = hit
    print(f"[probe] run {rid} offset {off}")
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
    pnames = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in model.named_parameters() if p.requires_grad]

    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                      num_frame_per_block=sf, causal_inference=True)
    detach_caches(model)
    hist = {}
    t = time.time(); roll(model, images, sf, a.t0, sf, a.K, dtype, dev, hist)
    print(f"[probe] rolled to {a.t0} in {time.time()-t:.1f}s", flush=True)

    lab = bank.get(rid, off, S, device=dev)
    with model.masked_window(window_start=a.t0, keyframe_interval=a.K):
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, a.t0:a.t0+S].to(dev),
                                num_frame_for_scale=sf, num_frame_per_block=S,
                                causal_inference=False)
    stu = out["pose_enc"][0].float()
    sdep = out["depth"][0].float()
    tea = lab["pose_enc"].detach(); tdep = lab["depth"].detach()
    conf = lab.get("depth_conf")
    conf = conf.detach() if conf is not None else None
    P = PRESETS[a.preset]

    # the five terms exactly as SelfDistillLoss.forward builds them (losses.py:765-776)
    L_rot, L_dir, L_mag = rel_pose_loss(stu, tea, mag_mode=P.get("mag_mode", "closed_form_scale"),
                                        pairs=P.get("pairs", "all"),
                                        min_gap=P.get("min_gap", 1),
                                        mag_trunc=P.get("mag_trunc", 1.0))
    L_dep = depth_si_loss(sdep, tdep, conf)
    L_mot = motion_depth_loss(stu, tea, sdep, tdep,
                              pairs=P.get("pairs", "all"), min_gap=P.get("min_gap", 1))
    L_gt, n_gt = gt_window_loss(a.frames, a.gt_calib, a.gt_sensor, a.t0, S, stu, dev)
    print(f"[probe] GT window on {n_gt}/{S} frames, L_gt={float(L_gt.detach()):.4f}", flush=True)

    named = list(zip(TERMS, [L_rot, L_dir, L_mag, L_dep, L_mot]))
    terms = list(TERMS)
    abs_info = {}
    if a.abs_mode != "off":
        r = bank.runs[rid]
        u = AL.median_step(bank.poses(rid, 0, r["L"])[:, :3])
        fit = AL.fit_run_gauge(hist, r["t0"], off, S, stu.detach(),
                               lambda lo, hi: bank.poses(rid, lo, hi), mode=a.abs_fit, u=u,
                               fallback_s=lambda: AL.depth_scale_fallback(sdep.detach(), tdep, conf))
        abs_info = {**fit.parts(), "abs_u": u}
        print(f"[abs] fit s={float(fit.s):.4f} mode={fit.mode} n={fit.n} path={fit.path:.0f}u "
              f"rot(world/cam/resid) {fit.rot_world_deg:.2f}/{fit.rot_cam_deg:.2f}/{fit.rot_resid_deg:.2f}deg")
        for k in ABS_TERMS[a.abs_mode]:
            crit = AL.RunGaugeLoss(**{f"lam_{k}": 1.0})
            Lk, pk = crit(stu, tea, sdep, tdep, conf, fit, u)
            named.append((k, Lk)); terms.append(k)
            abs_info[f"{k}_value"] = float(Lk.detach())
        if "abs_pos" not in terms:
            # the run-gauge GT direction, even when the term is not trained
            crit = AL.RunGaugeLoss(lam_abs_pos=1.0)
            Lk, _ = crit(stu, tea, sdep, tdep, conf, fit, u)
            named.append(("L_gt_run", Lk))
    named.append(("L_gt", L_gt))
    gdv = torch.device(a.grad_device)
    G = {}
    for i, (nm, L) in enumerate(named):
        t = time.time()
        gi = torch.autograd.grad(L, params, retain_graph=(i < len(named)-1), allow_unused=True)
        G[nm] = tuple(None if x is None else x.detach().to(gdv, torch.float32) for x in gi)
        print(f"[probe] backward {nm:<16} {time.time()-t:6.1f}s", flush=True)

    rec = {"scene": a.scene, "t0": a.t0, "run": rid, "offset": off, "K": a.K, "preset": a.preset, "n_gt": n_gt,
           "lam_shipped": {k: P[k] for k in ("lam_rot","lam_dir","lam_mag","lam_depth","lam_motion")},
           "abs_mode": a.abs_mode, "abs_fit": a.abs_fit, "abs": abs_info, "terms": terms,
           "views": {}}
    gt_run_key = "abs_pos" if "abs_pos" in terms else ("L_gt_run" if "L_gt_run" in G else None)
    for view, drop in (("all_params", []), ("pose_subspace", ["depth_head"])):
        keep = keep_mask(pnames, drop)
        import numpy as np
        n = len(TERMS)
        Gm = np.zeros((n, n)); b = np.zeros(n)
        for i in range(n):
            for j in range(i, n):
                v = gdot(G[TERMS[i]], G[TERMS[j]], keep); Gm[i, j] = Gm[j, i] = v
            b[i] = gdot(G[TERMS[i]], G["L_gt"], keep)
        gt_n = gdot(G["L_gt"], G["L_gt"], keep) ** 0.5
        lam0 = np.array([P["lam_rot"], P["lam_dir"], P["lam_mag"], P["lam_depth"], P["lam_motion"]])
        cos0 = float(lam0 @ b / ((lam0 @ Gm @ lam0) ** 0.5 * gt_n))
        lam_s = np.linalg.solve(Gm + 1e-12 * np.trace(Gm) / n * np.eye(n), b)
        cos_max = float((b @ lam_s) ** 0.5 / gt_n) if b @ lam_s > 0 else float("nan")
        per = {TERMS[i]: (float(b[i] / (Gm[i, i] ** 0.5 * gt_n)) if Gm[i, i] > 0 else float("nan"))
               for i in range(n)}
        # every term (old and new) against both GT directions
        def cosine(x, y):
            nx, ny = gdot(G[x], G[x], keep) ** 0.5, gdot(G[y], G[y], keep) ** 0.5
            return float(gdot(G[x], G[y], keep) / (nx * ny)) if nx * ny > 0 else float("nan")
        cos_win = {k: cosine(k, "L_gt") for k in terms}
        cos_run = {k: cosine(k, gt_run_key) for k in terms} if gt_run_key else {}
        cos_a1pc_run = None
        if gt_run_key:
            g_a1 = [None if all(G[k][i] is None for k in TERMS) else
                    sum(P[f] * G[k][i] for k, f in zip(TERMS, ("lam_rot", "lam_dir", "lam_mag", "lam_depth", "lam_motion")) if G[k][i] is not None)
                    for i in range(len(G["L_gt"]))]
            G["_A1PC"] = tuple(g_a1)
            cos_a1pc_run = cosine("_A1PC", gt_run_key)
        rec["views"][view] = {"cos_shipped": cos0, "cos_max": cos_max,
                              "lam_star": dict(zip(TERMS, (lam_s / abs(lam_s).max()).tolist())),
                              "cos_per_term": per,
                              "term_grad_norm": {k: float(gdot(G[k], G[k], keep) ** 0.5) for k in terms},
                              "cos_window_gt": cos_win, "cos_run_gt": cos_run,
                              "cos_a1pc_run_gt": cos_a1pc_run}
        print(f"\n=== {view} ===")
        print(f"  cos(g_A1PC, g_GT)        {cos0:+.4f}")
        print(f"  cos ceiling over all lam {cos_max:+.4f}   <- best ANY reweighting can do")
        if gt_run_key:
            print(f"  cos(g_A1PC, g_GT_run)    {cos_a1pc_run:+.4f}   <- run-gauge ATE direction")
        print(f"  {'term':>16} {'|g_i|':>12} {'cos(g_i,g_GT)':>15} {'cos(g_i,g_GTrun)':>17} {'lam shipped':>12} {'lam*':>9}")
        for i, k in enumerate(terms):
            extra = (f"{lam0[i]:>12.2f} {lam_s[i]/abs(lam_s).max():>+9.3f}" if i < n else f"{'':>12} {'':>9}")
            cr = cos_run.get(k, float('nan'))
            print(f"  {k:>16} {rec['views'][view]['term_grad_norm'][k]:>12.3f} {cos_win[k]:>+15.4f} "
                  f"{cr:>+17.4f} {extra}")
    out = a.out or (f"experiments/results/termalign_{a.scene}_t{a.t0}_K{a.K}.json" if a.abs_mode == "off"
                    else f"experiments/results/termalign_abs_{a.abs_mode}_{a.abs_fit}_{a.scene}_t{a.t0}_K{a.K}.json")
    json.dump(rec, open(out, "w"), indent=1)
    print(f"\n[probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
