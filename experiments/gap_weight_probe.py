"""Does tilting the LOCAL all-pairs rotation loss toward long gaps point more at GT?

The proposal: A1PC averages its pair set uniformly,

    alpha_p = 1 / |P|,

and since ``pairs="all"`` keeps both directions of every ordered pair, gap g has
2*(S-g) members -- so the short gaps hold the larger share of the total weight by
construction (46.8% of it sits at gap >= 16 for S=48).  Re-weighting

    alpha_p = gap_p^gamma / sum_q gap_q^gamma

moves that mass onto the long baselines.  This probe measures whether doing so
moves the UPDATE toward GT, using the same criterion docs2/C.md fixed for the
long-term question: cos(g, g_GT) on windows where GT exists.

★ THIS IS A DIFFERENT ``rot`` FROM THE ONE IN docs/metric-design-analysis.html
sec 07.  That one is ``LongPoseLoss``'s rot: student-vs-teacher relative rotation
across a Delta=48..319 anchor OUTSIDE the window, and it came out orthogonal to
GT in 12/12 windows.  This one is A1PC's within-window rot over gap 1..47.  They
share a name and nothing else, and a finding about one says nothing about the
other -- which is exactly why this needs its own measurement.

★ THE WEIGHT IS THE GRADIENT SHARE, ALMOST EXACTLY.  ``L_rot`` is a mean of
``acos``, and acos is an L1 in the angle (long_supervision_result.md item 2:
|dL/dtheta| is 4.33 at 0.55 deg and 4.33 at 55 deg).  So a pair's residual sets
its LOSS contribution but not its GRADIENT magnitude -- alpha_p does.  Tilting
alpha is therefore a direct re-allocation of the rotation update, not a soft
preference, and a band that is 20% of the weight is ~20% of the rot gradient
whether or not its residual is large.

★ NO EXTRA ROLLOUT PER gamma, AND NO APPROXIMATION.  Each gamma is one more
backward against the SAME retained graph, so the 500-frame prefix roll -- which
is the whole cost of this probe -- is paid once.  And the full-objective effect
needs no backward at all: only the rot term changes, so

    g_local(gamma) = g_local + lam_rot * (g_rot(gamma) - g_rot(0))

exactly, with lam_rot=15.0 from the preset.  The band split is reported too and
is checked against g_rot(0) as a sum, which catches an indexing error in either.

★ AND THE TARGET IS AUDITED, NOT ASSUMED.  Up-weighting long gaps is only worth
doing if the long-gap TEACHER labels are worth imitating.  On GT windows the
probe also reports, per gap and with no backward, the teacher's and the student's
relative-rotation error against GT.  If teacher error grows with gap faster than
student error does, gamma>0 buys a larger share of a noisier target -- the same
failure the depth-vs-Delta confound was (long_supervision_result.md defect A).

    python experiments/gap_weight_probe.py --ckpt ckpt_train/v7f.step50.pt \
        --frames data/mcd/kth_day_10/frames_10hz --bank labels/kth_day_10 \
        --t0 512 --K 1 --gt_calib data/mcd/calib/hhs_calib.yaml
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

from lingbot_map.train.label_bank import LabelBank                    # noqa: E402
from lingbot_map.train.losses import (                                # noqa: E402
    PRESETS, SelfDistillLoss, _pair_index, _relative)
from grad_probe import detach_caches                                  # noqa: E402
from long_grad_probe import PARAM_GROUPS, gt_window_loss, roll         # noqa: E402


# ★ THE REPORTING TAIL IS THE COST OF THIS PROBE, NOT THE BACKWARDS.  Measured
# on this model (1.15B parameters over 470 tensors, 4.6 GB per gradient): one
# full reduction written as ``term_grad_probe.gnorm`` does -- ``x.pow(2).sum()``,
# which allocates a full-size temporary -- takes 31 s on the host, and the group
# table plus the gamma sweep need on the order of a hundred of them.  That is a
# 45-minute tail on top of a 3-minute rollout, and it is pure host memory
# traffic.  Two changes remove it:
#
#   ``dot`` on a flat view instead of ``(x*y).sum()``   31 s -> 15 s (no temporary)
#   the gradients stay on the GPU                       15 s -> milliseconds
#
# ``--grad_device cpu`` keeps the old placement for a card that cannot hold them;
# at 9 gradients this probe wants ~41 GB of the 183 GB on a B200, next to the
# ~53 GB the retained graph is already using.
def gdot(a, b, keep=None) -> float:
    """<a, b> over the parameter list, optionally restricted to a boolean mask.

    ★ ``keep`` IS A PER-PARAMETER BOOLEAN MASK, not a list of indices -- the same
    convention ``term_grad_probe.gnorm`` uses, and the same trap (an index list
    silently reads keep[i] for every i and walks off the end).

    ★ ONE SYNC PER PASS, NOT ONE PER TENSOR.  ``float()`` on a CUDA scalar is a
    device sync; doing it inside the loop would pay 470 of them per reduction and
    give most of the GPU win straight back.  The accumulator stays a tensor.
    """
    acc = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x is None or y is None or (keep is not None and not keep[i]):
            continue
        v = x.reshape(-1).dot(y.reshape(-1))
        acc = v if acc is None else acc + v
    return 0.0 if acc is None else float(acc)


def gnorm(g, keep=None) -> float:
    return math.sqrt(max(0.0, gdot(g, g, keep)))


_NORM_MEMO = {}


def gnorm_c(g, keep=None, tag=None) -> float:
    """gnorm with a memo -- the same |g| is asked for once per cosine otherwise."""
    k = (id(g) if tag is None else tag, None if keep is None else id(keep))
    if k not in _NORM_MEMO:
        _NORM_MEMO[k] = gnorm(g, keep)
    return _NORM_MEMO[k]


def gcos(a, b, keep=None) -> float:
    return gdot(a, b, keep) / max(1e-12, gnorm_c(a, keep) * gnorm_c(b, keep))


def gaxpy(a, k, b):
    """a + k*b, elementwise over the parameter list, tolerating None entries."""
    return tuple(None if (x is None and y is None) else
                 (k * y if x is None else x if y is None else x + k * y)
                 for x, y in zip(a, b))


def rot_weighted(stu_pose, tea_pose, gamma=0.0, band=None,
                 pairs="all", min_gap=1):
    """A1PC's L_rot with pair weights ``gap^gamma``, optionally banded.

    ``gamma=0`` with no band reproduces ``rel_pose_loss``'s L_rot bit for bit --
    it is the same pair set, the same residual and an unweighted mean -- which is
    what makes ``g_rot(0)`` usable as the baseline the gamma sweep is measured
    against and as the term subtracted out of ``g_local``.

    Returns ``(L, weight_sum_fraction, mean_gap)``.
    """
    N = stu_pose.shape[0]
    tea_pose = tea_pose.detach()          # same rule as SelfDistillLoss.forward
    i, j = _pair_index(N, pairs, min_gap, stu_pose.device)
    gap = (j - i).abs().to(stu_pose.dtype)

    dRs, _ = _relative(stu_pose, i, j)
    dRt, _ = _relative(tea_pose, i, j)
    dR = torch.einsum("nij,njk->nik", dRs.transpose(1, 2), dRt)
    cos = ((dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]) - 1) / 2
    theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))

    w = gap ** gamma if gamma else torch.ones_like(gap)
    if band is not None:
        w = w * ((gap >= band[0]) & (gap <= band[1])).to(w.dtype)
    tot = w.sum().clamp(min=1e-12)
    return (w * theta).sum() / tot, float(tot / gap.numel()), float((w * gap).sum() / tot)


def gt_rot_losses(stu_pose, frames_dir, calib, sensor, t0, S, gaps, dev):
    """GT relative-rotation error of the window, at fixed baselines.

    ★ THE ATE CRITERION HAS NO ROTATION CHANNEL.  ``gt_window_loss`` reads
    ``stu_pose[keep, :3]`` -- translations only -- so in POSE space its gradient
    is exactly zero on every rotation component.  Scoring a rotation term against
    it asks "does fixing relative rotations also fix the window's positions",
    which within 48 frames and after a Sim(3) fit is a weak instrument and is not
    the question a rotation re-weighting is trying to answer.

    ★ AND THERE IS NO NEUTRAL ROTATION CRITERION.  Any single GT rotation loss
    embeds a baseline: an error measured at gap 5 rewards short-gap supervision
    and one at gap 47 rewards long-gap supervision, for free and by construction.
    So several are taken and reported side by side, and the comparison to make is
    ACROSS gamma at a FIXED criterion -- never across criteria.

    Gauge-invariant by construction (relative rotations only), which matches the
    A1PC family; ``keep`` drops rows without GT and gaps are counted in disk
    index, so a dropped frame removes its pairs rather than shortening the rest.
    """
    import numpy as np
    from lingbot_map.train.label_bank import image_names
    from lingbot_map.train.losses import quat_to_R
    from mcd_eval import gt_camera_poses, load_extrinsic
    from mcd_gt import quat_to_mat

    meta = np.load(os.path.join(frames_dir, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(calib, sensor)
    _, gq = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    row_of = {str(n): i for i, n in enumerate(meta["names"])}
    rows = np.array([row_of.get(n, -1) for n in image_names(frames_dir)[t0:t0 + S]])
    ok = rows >= 0

    R_gt_np = np.zeros((S, 3, 3))
    R_gt_np[ok] = quat_to_mat(gq[rows[ok]])
    R_gt = torch.as_tensor(R_gt_np, dtype=torch.float32, device=dev)
    R_st = quat_to_R(stu_pose[:, 3:7])

    out = {}
    for gp_ in gaps:
        i = np.arange(S - gp_)
        j = i + gp_
        m = ok[i] & ok[j]
        if m.sum() < 3:
            continue
        ii = torch.as_tensor(i[m], device=dev)
        jj = torch.as_tensor(j[m], device=dev)
        ds = torch.einsum("nij,njk->nik", R_st[ii].transpose(1, 2), R_st[jj])
        dg = torch.einsum("nij,njk->nik", R_gt[ii].transpose(1, 2), R_gt[jj])
        r = torch.einsum("nij,njk->nik", ds.transpose(1, 2), dg)
        c = ((r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]) - 1) / 2
        out[gp_] = (torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6)).mean(), int(m.sum()))
    return out


def gap_audit(stu_pose, tea_pose, frames_dir, calib, sensor, t0, S, bands):
    """Per-gap teacher-vs-GT and student-vs-GT relative rotation error, degrees.

    ★ SAME ROW-DROPPING RULE AS ``gt_window_loss``.  meta.npz lists only the
    frames that have GT and the bank walks the whole directory, so frames without
    GT are dropped rather than paired against the wrong row.  Gaps are then
    counted in DISK index, not in kept-row index, so a dropped frame removes its
    pairs instead of shortening everybody else's baseline.
    """
    import numpy as np
    from lingbot_map.train.label_bank import image_names
    from mcd_eval import gt_camera_poses, load_extrinsic, rot_geodesic_deg
    from mcd_gt import quat_to_mat

    meta = np.load(os.path.join(frames_dir, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(calib, sensor)
    _, gq = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    row_of = {str(n): i for i, n in enumerate(meta["names"])}
    rows = np.array([row_of.get(n, -1) for n in image_names(frames_dir)[t0:t0 + S]])
    ok = rows >= 0
    if ok.sum() < 8:
        return None

    R_gt = np.full((S, 3, 3), np.nan)
    R_gt[ok] = quat_to_mat(gq[rows[ok]])
    R_st = quat_to_mat(stu_pose[:, 3:7].detach().double().cpu().numpy())
    R_te = quat_to_mat(tea_pose[:, 3:7].detach().double().cpu().numpy())

    def rel(R, i, j):
        return np.einsum("nij,njk->nik", R[i].transpose(0, 2, 1), R[j])

    per_gap = {}
    for g in range(1, S):
        i = np.arange(S - g)
        j = i + g
        m = ok[i] & ok[j]
        if m.sum() < 3:
            continue
        i, j = i[m], j[m]
        d_gt = rel(R_gt, i, j)
        per_gap[g] = {"n": int(m.sum()),
                      "teacher_deg": float(rot_geodesic_deg(rel(R_te, i, j), d_gt).mean()),
                      "student_deg": float(rot_geodesic_deg(rel(R_st, i, j), d_gt).mean()),
                      "gt_deg": float(rot_geodesic_deg(d_gt, np.tile(np.eye(3), (len(i), 1, 1))).mean())}

    out = {"per_gap": per_gap, "bands": {}}
    for lo, hi in bands:
        sel = [v for g, v in per_gap.items() if lo <= g <= hi]
        if sel:
            n = sum(v["n"] for v in sel)
            out["bands"][f"{lo}-{hi}"] = {
                "n": n,
                # weighted by pair count, so the band number is the mean over
                # PAIRS -- the same population the loss averages over
                "teacher_deg": sum(v["teacher_deg"] * v["n"] for v in sel) / n,
                "student_deg": sum(v["student_deg"] * v["n"] for v in sel) / n,
                "gt_deg": sum(v["gt_deg"] * v["n"] for v in sel) / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--preset", default="A1PC")
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0],
                    help="pair weight exponent; gamma=0 is the current loss and "
                         "must be present -- it is the baseline every other row "
                         "is differenced against.")
    ap.add_argument("--bands", default="1-15,16-31,32-47",
                    help="gap bands to take a separate gradient for.  The "
                         "proposal's split is 1-15 against 16-47.")
    ap.add_argument("--gt_calib", default="")
    ap.add_argument("--gt_sensor", default="d455b_color")
    ap.add_argument("--gt_rot_gaps", type=int, nargs="*", default=[5, 24, 47],
                    help="baselines for the GT relative-rotation criteria.  Read "
                         "ACROSS gamma within one gap, never across gaps -- each "
                         "gap is a different question, not a better answer.")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--grad_device", default="cuda",
                    help="where the taken gradients live.  'cuda' keeps the "
                         "reduction tail at milliseconds; 'cpu' is the fallback "
                         "for a card that cannot hold ~41 GB of them.")
    ap.add_argument("--out", default="experiments/results/gap_weight_probe.json")
    a = ap.parse_args()

    if 0.0 not in a.gamma:
        raise SystemExit("--gamma must include 0 (the current loss, and the "
                         "baseline the sweep is differenced against)")
    bands = [tuple(int(x) for x in b.split("-")) for b in a.bands.split(",") if b]
    dev, dtype, sf, S = torch.device("cuda"), torch.bfloat16, a.num_scale_frames, a.S

    # ★ THE PAIR SET MUST COME FROM THE PRESET, not from this script's defaults.
    # A1PC is pairs="all", min_gap=1; probing a different pair set would measure
    # a re-weighting of a loss nothing is training on.
    pcfg = PRESETS[a.preset]
    pairs, min_gap, lam_rot = pcfg["pairs"], pcfg.get("min_gap", 1), pcfg["lam_rot"]
    print(f"[probe] preset {a.preset}: pairs={pairs} min_gap={min_gap} lam_rot={lam_rot}")

    bank = LabelBank(a.bank)
    hit = next(((rid, a.t0 - r["t0"]) for rid, r in enumerate(bank.runs)
                if r["t0"] <= a.t0 and a.t0 + S <= r["t0"] + r["L"]), None)
    if hit is None:
        raise SystemExit(f"no single bank run covers [{a.t0}, {a.t0 + S})")
    rid, off = hit
    print(f"[probe] run {rid} offset {off}")

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

    from phase0_density_sweep import build_model
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
    t_roll = time.time()
    roll(model, images, sf, a.t0, sf, a.K, dtype, dev, {})
    print(f"[probe] rolled to {a.t0} in {time.time() - t_roll:.1f}s", flush=True)

    lab = bank.get(rid, off, S, device=dev)
    with model.masked_window(window_start=a.t0, keyframe_interval=a.K):
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, a.t0:a.t0 + S].to(dev),
                                num_frame_for_scale=sf, num_frame_per_block=S,
                                causal_inference=False)
    stu = out["pose_enc"][0].float()
    tea = lab["pose_enc"]

    crit = SelfDistillLoss.from_preset(a.preset)
    L_local, lp = crit(stu, tea, out["depth"][0].float(),
                       lab["depth"], lab.get("depth_conf"))

    gam, meta_g = {}, {}
    for gm in a.gamma:
        L, frac, mg = rot_weighted(stu, tea, gm, None, pairs, min_gap)
        gam[gm] = L
        meta_g[gm] = {"mean_gap": mg}
    bnd, meta_b = {}, {}
    for lo, hi in bands:
        L, frac, mg = rot_weighted(stu, tea, 0.0, (lo, hi), pairs, min_gap)
        bnd[(lo, hi)] = L
        meta_b[(lo, hi)] = {"pair_frac": frac, "mean_gap": mg,
                            "resid_deg": float(L.detach()) * 57.29578}

    # ★ gamma=0 IS THE PRESET'S OWN L_rot.  If this ever disagrees, the pair set
    # or the residual has drifted from ``rel_pose_loss`` and every difference
    # below is measured against the wrong baseline.
    d = abs(float(gam[0.0].detach()) - lp["L_rot_rad"])
    if d > 1e-5:
        raise SystemExit(f"gamma=0 does not reproduce the preset L_rot "
                         f"({float(gam[0.0].detach()):.6f} vs {lp['L_rot_rad']:.6f})")

    L_gt, n_gt = (None, 0)
    if a.gt_calib:
        L_gt, n_gt = gt_window_loss(a.frames, a.gt_calib, a.gt_sensor, a.t0, S, stu, dev)
        print(f"[probe] GT window ATE on {n_gt}/{S} frames"
              + ("" if L_gt is None else f"  L_gt={float(L_gt.detach()):.4f}"))

    gt_rot = {}
    if a.gt_calib and a.gt_rot_gaps:
        gt_rot = gt_rot_losses(stu, a.frames, a.gt_calib, a.gt_sensor,
                               a.t0, S, a.gt_rot_gaps, dev)
        print("[probe] GT rot criteria: "
              + "  ".join(f"gap{k} {float(v.detach()) * 57.29578:.3f}deg (n={n})"
                          for k, (v, n) in gt_rot.items()), flush=True)

    named = ([("L_local", L_local)]
             + [(f"rot_g{gm:g}", v) for gm, v in gam.items()]
             + [(f"rot_b{lo}_{hi}", v) for (lo, hi), v in bnd.items()]
             + ([("L_gt", L_gt)] if L_gt is not None else [])
             + [(f"L_gtrot{k}", v) for k, (v, _) in gt_rot.items()])
    gdev = torch.device(a.grad_device)
    grads = []
    for i, (nm, L) in enumerate(named):
        t_b = time.time()
        gi = torch.autograd.grad(L, params, retain_graph=(i < len(named) - 1),
                                 allow_unused=True)
        grads.append(tuple(None if x is None else x.detach().to(gdev, torch.float32)
                           for x in gi))
        del gi
        print(f"[probe] backward {nm:<14} {time.time() - t_b:6.1f}s", flush=True)
    g = dict(zip([n for n, _ in named], grads))
    print(f"[probe] {len(named)} gradients held on {gdev}", flush=True)

    # depth_head takes no GT gradient at all (the window ATE is a function of the
    # poses alone), so a cosine over ALL parameters is diluted by a block that is
    # structurally orthogonal to GT and is most of |g_local|.  Both are reported.
    keep_nd = [not n.startswith("depth_head.") for n in names_p]
    t_red = time.time()

    rec = {"scene": os.path.basename(a.bank.rstrip("/")),
           "t0": a.t0, "run": rid, "offset": off, "K": a.K, "preset": a.preset,
           "pairs": pairs, "min_gap": min_gap, "lam_rot": lam_rot,
           "gamma": list(a.gamma), "bands": [f"{lo}-{hi}" for lo, hi in bands],
           "local_parts": lp, "n_gt": n_gt,
           "gt_rot_gaps": list(gt_rot),
           "gt_rot_deg": {str(k): float(v.detach()) * 57.29578
                          for k, (v, _) in gt_rot.items()},
           "loss": {k: float(v.detach()) for k, v in named},
           "grad_norm": {k: gnorm(v) for k, v in g.items()},
           "gamma_rows": [], "band_rows": []}

    g0 = g["rot_g0"]
    for gm in a.gamma:
        gr = g[f"rot_g{gm:g}"]
        # exact: only the rot term changes, so the rest of A1PC cancels
        gl = g["L_local"] if gm == 0.0 else gaxpy(gaxpy(g["L_local"], lam_rot, gr),
                                                  -lam_rot, g0)
        n_rot, n_loc = gnorm_c(gr), gnorm_c(gl, tag=f"gl{gm:g}")
        row = {"gamma": gm, "mean_gap": meta_g[gm]["mean_gap"],
               "loss_rot_rad": float(gam[gm].detach()),
               "g_rot": n_rot, "g_local": n_loc,
               "cos_rot_vs_rot0": gcos(gr, g0),
               "rot_share_of_local": lam_rot * n_rot / max(1e-12, n_loc)}
        if L_gt is not None:
            gtg = g["L_gt"]
            row.update({
                "cos_rot_gt": gdot(gr, gtg) / max(1e-12, n_rot * gnorm_c(gtg)),
                "cos_rot_gt_nodepth": gdot(gr, gtg, keep_nd) / max(
                    1e-12, gnorm_c(gr, keep_nd) * gnorm_c(gtg, keep_nd)),
                "cos_local_gt": gdot(gl, gtg) / max(1e-12, n_loc * gnorm_c(gtg)),
                "cos_local_gt_nodepth": gdot(gl, gtg, keep_nd) / max(
                    1e-12, gnorm_c(gl, keep_nd, tag=f"glnd{gm:g}")
                    * gnorm_c(gtg, keep_nd))})
        for k in gt_rot:
            gk = g[f"L_gtrot{k}"]
            row[f"cos_rot_gtrot{k}"] = gdot(gr, gk, keep_nd) / max(
                1e-12, gnorm_c(gr, keep_nd) * gnorm_c(gk, keep_nd))
            row[f"cos_local_gtrot{k}"] = gdot(gl, gk, keep_nd) / max(
                1e-12, gnorm_c(gl, keep_nd, tag=f"glnd{gm:g}")
                * gnorm_c(gk, keep_nd))
        rec["gamma_rows"].append(row)
        del gl

    for lo, hi in bands:
        gb = g[f"rot_b{lo}_{hi}"]
        row = {"band": f"{lo}-{hi}", **meta_b[(lo, hi)],
               "g_rot": gnorm(gb), "cos_band_vs_rot0": gcos(gb, g0),
               "cos_band_vs_local": gcos(gb, g["L_local"])}
        if L_gt is not None:
            row.update({"cos_band_gt": gcos(gb, g["L_gt"]),
                        "cos_band_gt_nodepth": gcos(gb, g["L_gt"], keep_nd)})
        for k in gt_rot:
            row[f"cos_band_gtrot{k}"] = gcos(gb, g[f"L_gtrot{k}"], keep_nd)
        rec["band_rows"].append(row)

    # ★ THE BANDS MUST RE-SUM TO g_rot(0).  L_rot is a mean over the whole pair
    # set, so it is the pair-fraction-weighted mix of the band means; a mismatch
    # means a band is double-counting or missing pairs.
    mix = None
    for lo, hi in bands:
        mix = (tuple(None if x is None else meta_b[(lo, hi)]["pair_frac"] * x
                     for x in g[f"rot_b{lo}_{hi}"]) if mix is None
               else gaxpy(mix, meta_b[(lo, hi)]["pair_frac"], g[f"rot_b{lo}_{hi}"]))
    rec["band_closure_rel"] = gnorm(gaxpy(mix, -1.0, g0)) / max(1e-12, gnorm(g0))
    del mix

    if L_gt is not None:
        rec["cos_local_gt_base"] = gcos(g["L_local"], g["L_gt"])
        rec["cos_local_gt_base_nodepth"] = gcos(g["L_local"], g["L_gt"], keep_nd)
        rec["grad_norm"]["L_gt"] = gnorm(g["L_gt"])
        aud = gap_audit(stu, tea, a.frames, a.gt_calib, a.gt_sensor, a.t0, S, bands)
        rec["gap_audit"] = aud

    rec["by_group"] = {}
    for gname, pref in PARAM_GROUPS:
        keep = [n.startswith(pref) for n in names_p]
        rec["by_group"][gname] = {k: gnorm(v, keep) for k, v in g.items()}

    print(f"[probe] reductions done in {time.time() - t_red:.1f}s", flush=True)

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\n  local {float(L_local):.4f}   |g_local| {gnorm(g['L_local']):.4e}"
          f"   L_rot {lp['L_rot_deg']:.3f} deg")
    print(f"  band closure |sum_b f_b g_b - g_rot(0)| / |g_rot(0)| = "
          f"{rec['band_closure_rel']:.2e}")

    print(f"\n  {'band':>8}{'pairs':>8}{'mean gap':>10}{'resid deg':>11}"
          f"{'|g_rot|':>12}{'cos(rot0)':>11}{'cos(local)':>12}"
          + (f"{'cos(GT)':>10}{'cos(GT)-nd':>12}" if L_gt is not None else ""))
    for r in rec["band_rows"]:
        print(f"  {r['band']:>8}{r['pair_frac']:>8.1%}{r['mean_gap']:>10.1f}"
              f"{r['resid_deg']:>11.3f}{r['g_rot']:>12.3e}"
              f"{r['cos_band_vs_rot0']:>+11.4f}{r['cos_band_vs_local']:>+12.4f}"
              + (f"{r['cos_band_gt']:>+10.4f}{r['cos_band_gt_nodepth']:>+12.4f}"
                 if L_gt is not None else ""))

    print(f"\n  {'gamma':>6}{'mean gap':>10}{'|g_rot|':>12}{'cos(rot0)':>11}"
          f"{'rot/local':>11}"
          + (f"{'cos(rot,GT)':>13}{'cos(loc,GT)':>13}{'d cos':>9}"
             f"{'cos(loc,GT)nd':>15}{'d nd':>9}" if L_gt is not None else ""))
    base = rec["gamma_rows"][[r["gamma"] for r in rec["gamma_rows"]].index(0.0)]
    for r in rec["gamma_rows"]:
        line = (f"  {r['gamma']:>6g}{r['mean_gap']:>10.1f}{r['g_rot']:>12.3e}"
                f"{r['cos_rot_vs_rot0']:>+11.4f}{r['rot_share_of_local']:>11.1%}")
        if L_gt is not None:
            line += (f"{r['cos_rot_gt']:>+13.4f}{r['cos_local_gt']:>+13.4f}"
                     f"{r['cos_local_gt'] - base['cos_local_gt']:>+9.4f}"
                     f"{r['cos_local_gt_nodepth']:>+15.4f}"
                     f"{r['cos_local_gt_nodepth'] - base['cos_local_gt_nodepth']:>+9.4f}")
        print(line)

    if gt_rot:
        print(f"\n  against the GT ROTATION criteria (no-depth params)")
        print(f"  {'gamma':>6}" + "".join(f"{'rot|gap' + str(k):>13}" for k in gt_rot)
              + "".join(f"{'loc|gap' + str(k):>13}" for k in gt_rot))
        for r in rec["gamma_rows"]:
            print(f"  {r['gamma']:>6g}"
                  + "".join(f"{r[f'cos_rot_gtrot{k}']:>+13.4f}" for k in gt_rot)
                  + "".join(f"{r[f'cos_local_gtrot{k}']:>+13.4f}" for k in gt_rot))
        print(f"  {'band':>6}" + "".join(f"{'gap' + str(k):>13}" for k in gt_rot))
        for r in rec["band_rows"]:
            print(f"  {r['band']:>6}"
                  + "".join(f"{r[f'cos_band_gtrot{k}']:>+13.4f}" for k in gt_rot))

    if L_gt is not None and rec.get("gap_audit"):
        print(f"\n  target audit -- relative rotation error vs GT, degrees")
        print(f"  {'band':>8}{'pairs':>8}{'GT motion':>11}{'teacher':>10}"
              f"{'student':>10}{'headroom':>10}")
        for k, v in rec["gap_audit"]["bands"].items():
            print(f"  {k:>8}{v['n']:>8}{v['gt_deg']:>11.3f}{v['teacher_deg']:>10.3f}"
                  f"{v['student_deg']:>10.3f}{v['student_deg'] - v['teacher_deg']:>+10.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()
