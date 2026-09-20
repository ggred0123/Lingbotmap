#!/usr/bin/env python3
"""Counterexample re-check for the whole objective, loss AND gradient.
docs/gtabs-plan.md §5-2.  CPU, no checkpoint.

nullspace_demo / nullspace_variants showed that A1PC's VALUE cannot see a slow
common scale ramp.  Training follows the GRADIENT, so this asks the sharper
question for every term, old and new, on the same perturbations:

    dL          does the term move at all?
    |dL/dpose|, |dL/ddepth|   which output does it pull on?
    cos(-g, x_clean - x_pert)  does the pull point BACK toward the truth?
                (> 0: descent undoes the perturbation; ~0: blind; < 0: hostile)

Perturbations, on a synthetic 48-frame window and on a real GT run
(kth_day_10, GT bank), each at several magnitudes:

    pose-only scale ramp    per-step length ramps 1 -> r, depth untouched
    depth-only scale ramp   per-frame depth ramps 1 -> r, poses untouched
    joint ramp              both (the unit drifting -- objective-diagnosis §03)
    rotation drift          heading drifts d deg/frame, trajectory re-integrated
    pure gauge              x3, a rotation, a translation on pose AND depth:
                            every term must read exactly 0

Two families: ``window`` puts the ramp inside one 48-frame window with a
window-only fit (what results-ledger O1 measured: joint ramp 1.10 -> L_depth
exactly 0, L_motion tiny); ``run`` ramps over a whole 240-frame run and scores
the LAST window (offset 192) under the run gauge fitted on the prefix, which is
the configuration gtscale trains in.

A small descent experiment closes it: from the perturbed state, how many
optimiser steps on the pose/depth tensors does each objective need to undo 90%
of the perturbation -- or does it never get there.

    .venv-bench/bin/python experiments/nullspace_grad_probe.py
    -> experiments/results/nullspace_grad_probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lingbot_map.train.abs_loss import (                                # noqa: E402
    GaugeFit, RunGaugeLoss, fit_run_gauge, median_step)
from lingbot_map.train.losses import (                                  # noqa: E402
    PRESETS, depth_si_loss, motion_depth_loss, quat_to_R, rel_pose_loss)
from lingbot_map.utils.rotation import mat_to_quat, quat_to_mat         # noqa: E402

S = 48
A1PC = PRESETS["A1PC"]
OLD = ["L_rot", "L_dir", "L_mag", "L_motion", "L_depth"]
NEW = ["trans_scale", "depth_scale", "abs_pos", "abs_rot", "rel_trans"]
OLD_LAM = {"L_rot": A1PC["lam_rot"], "L_dir": A1PC["lam_dir"], "L_mag": A1PC["lam_mag"],
           "L_motion": A1PC["lam_motion"], "L_depth": A1PC["lam_depth"]}


# ── trajectories ─────────────────────────────────────────────────────────────

def synthetic_run(n=240, step=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(n, dtype=torch.float64)
    yaw = i * 0.015 + 0.3 * torch.sin(i * 0.05)
    d = torch.stack([torch.sin(yaw), 0.05 * torch.sin(i * 0.11), torch.cos(yaw)], -1) * step
    C = torch.cat([torch.zeros(1, 3, dtype=torch.float64), torch.cumsum(d, 0)[:-1]])
    pitch = 0.05 * torch.sin(i * 0.07)
    Ry = torch.stack([torch.stack([torch.cos(yaw), 0 * yaw, torch.sin(yaw)], -1),
                      torch.stack([0 * yaw, 1 + 0 * yaw, 0 * yaw], -1),
                      torch.stack([-torch.sin(yaw), 0 * yaw, torch.cos(yaw)], -1)], -2)
    Rx = torch.stack([torch.stack([1 + 0 * pitch, 0 * pitch, 0 * pitch], -1),
                      torch.stack([0 * pitch, torch.cos(pitch), -torch.sin(pitch)], -1),
                      torch.stack([0 * pitch, torch.sin(pitch), torch.cos(pitch)], -1)], -2)
    R = Ry @ Rx
    pose = torch.cat([C, mat_to_quat(R), torch.full((n, 2), 0.9, dtype=torch.float64)], -1)
    depth = 3.0 + 4.0 * torch.rand(n, 28, 37, generator=g, dtype=torch.float64)
    conf = torch.ones(n, 28, 37, dtype=torch.float64)
    return pose, depth, conf


def gt_run(scene="kth_day_10", rid=3, stride=14):
    """A GT-bank run: GT poses in the teacher gauge, the teacher's depth and
    confidence thinned by ``stride`` so the CPU probe stays cheap."""
    root = os.path.join(ROOT, "labels", scene + "_gt")
    idx = json.load(open(os.path.join(root, "index.json")))
    r = idx["runs"][rid]
    z = np.load(os.path.join(root, r["file"]))
    pose = torch.from_numpy(z["pose_enc"].astype(np.float64))
    depth = torch.from_numpy(z["depth"][:, ::stride, ::stride].astype(np.float64))
    conf = torch.from_numpy(z["depth_conf"][:, ::stride, ::stride].astype(np.float64))
    z.close()
    return pose, depth, conf


# ── perturbations: return (pose', depth') ─────────────────────────────────────

def _reintegrate(pose, factor, rot=None):
    C = pose[:, :3]
    steps = C[1:] - C[:-1]
    if rot is not None:
        steps = torch.einsum("nij,nj->ni", rot[:-1], steps)
    steps = steps * factor[:, None]
    C2 = torch.cat([C[:1], C[:1] + torch.cumsum(steps, 0)])
    R = quat_to_mat(pose[:, 3:7])
    if rot is not None:
        R = rot @ R
    return torch.cat([C2, mat_to_quat(R), pose[:, 7:]], -1)


def ramp_factor(n_steps, r, lo, hi):
    """per-step factor: 1 outside [lo, hi), log-linear 1 -> r inside."""
    f = torch.ones(n_steps, dtype=torch.float64)
    k = torch.arange(hi - lo, dtype=torch.float64) / max(hi - lo - 1, 1)
    f[lo:hi] = torch.exp(k * math.log(r))
    return f


def perturb(kind, mag, pose, depth, lo, hi):
    n = pose.shape[0]
    if kind in ("pose_ramp", "joint_ramp"):
        f = ramp_factor(n - 1, mag, lo, max(hi - 1, lo + 1))
        p2 = _reintegrate(pose, f)
    else:
        p2 = pose.clone()
    if kind in ("depth_ramp", "joint_ramp"):
        f = ramp_factor(n, mag, lo, hi)
        d2 = depth * f[:, None, None]
    else:
        d2 = depth.clone()
    if kind == "rot_drift":
        ang = torch.zeros(n, dtype=torch.float64)
        ang[lo:hi] = torch.arange(hi - lo, dtype=torch.float64) * math.radians(mag)
        ang[hi:] = ang[hi - 1]
        rot = torch.stack([torch.stack([torch.cos(ang), 0 * ang, torch.sin(ang)], -1),
                           torch.stack([0 * ang, 1 + 0 * ang, 0 * ang], -1),
                           torch.stack([-torch.sin(ang), 0 * ang, torch.cos(ang)], -1)], -2)
        p2 = _reintegrate(pose, torch.ones(n - 1, dtype=torch.float64), rot)
    if kind == "gauge":
        s, ax = 3.0, torch.tensor([0.3, 1.0, -0.2], dtype=torch.float64)
        ax = ax / ax.norm(); a = 0.9
        K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=torch.float64)
        Q = torch.eye(3, dtype=torch.float64) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)
        c = torch.tensor([4.0, -1.0, 7.0], dtype=torch.float64)
        C = (s * (Q @ pose[:, :3].T)).T + c
        R = Q[None] @ quat_to_mat(pose[:, 3:7])
        p2 = torch.cat([C, mat_to_quat(R), pose[:, 7:]], -1)
        d2 = depth * s
    return p2, d2


# ── terms ────────────────────────────────────────────────────────────────────

def old_terms(sp, tp, sd, td, conf):
    L_rot, L_dir, L_mag = rel_pose_loss(sp, tp, mag_mode=A1PC.get("mag_mode", "closed_form_scale"),
                                        pairs=A1PC["pairs"], min_gap=A1PC["min_gap"])
    L_mot = motion_depth_loss(sp, tp, sd, td, pairs=A1PC["pairs"], min_gap=A1PC["min_gap"])
    L_dep = depth_si_loss(sd, td, conf, mode="median")
    return {"L_rot": L_rot, "L_dir": L_dir, "L_mag": L_mag, "L_motion": L_mot, "L_depth": L_dep}


def new_terms(sp, tp, sd, td, conf, fit, u):
    out = {}
    for k in NEW:
        crit = RunGaugeLoss(**{f"lam_{k}": 1.0}, huber_delta=0.1)
        tot, _ = crit(sp, tp, sd, td, conf, fit, u)
        out[k] = tot
    return out


def measure(pose_c, depth_c, conf, pose_p, depth_p, off, fit_mode, u, run_t0=80):
    """all terms at the perturbed state, their gradients w.r.t. the window's
    pose and depth, and the cosine against the vector back to the clean state."""
    w = slice(off, off + S)
    sp = pose_p[w].clone().float().requires_grad_(True)
    sd = depth_p[w].clone().float().requires_grad_(True)
    tp, td, tc = pose_c[w].float(), depth_c[w].float(), conf[w].float()
    if fit_mode == "window":
        fit = fit_run_gauge({}, run_t0, 0, S, sp.detach(), lambda lo, hi: tp[lo:hi], mode="prefix", u=u)
    else:
        hist = {run_t0 + k: pose_p[k].float() for k in range(off)}
        fit = fit_run_gauge(hist, run_t0, off, S, sp.detach(), lambda lo, hi: pose_c[lo:hi].float(),
                            mode=fit_mode, u=u)
    terms = {**old_terms(sp, tp, sd, td, tc), **new_terms(sp, tp, sd, td, tc, fit, u)}
    dpose = (pose_c[w, :7] - pose_p[w, :7]).float().reshape(-1)
    ddep = (depth_c[w] - depth_p[w]).float().reshape(-1)
    rec = {"fit_mode": fit.mode, "fit_s": float(fit.s)}
    for k, L in terms.items():
        g = torch.autograd.grad(L, [sp, sd], retain_graph=True, allow_unused=True)
        gp = torch.zeros_like(sp) if g[0] is None else g[0]
        gd = torch.zeros_like(sd) if g[1] is None else g[1]
        gp7, gdf = gp[:, :7].reshape(-1), gd.reshape(-1)
        cos_p = float(-(gp7 @ dpose) / (gp7.norm() * dpose.norm() + 1e-30)) if dpose.norm() > 0 else float("nan")
        cos_d = float(-(gdf @ ddep) / (gdf.norm() * ddep.norm() + 1e-30)) if ddep.norm() > 0 else float("nan")
        rec[k] = {"value": float(L), "g_pose": float(gp7.norm()), "g_depth": float(gdf.norm()),
                  "cos_pose": cos_p, "cos_depth": cos_d}
    return rec


def clean_values(pose_c, depth_c, conf, off, u, run_t0=80):
    w = slice(off, off + S)
    sp, sd = pose_c[w].float(), depth_c[w].float()
    fit = GaugeFit.identity()
    with torch.no_grad():
        t = {**old_terms(sp, sp, sd, sd, conf[w].float()),
             **new_terms(sp, sp, sd, sd, conf[w].float(), fit, u)}
    return {k: float(v) for k, v in t.items()}


# ── descent ──────────────────────────────────────────────────────────────────

def descent(objective, pose_c, depth_c, conf, pose_p, depth_p, off, fit_mode, u,
            steps=300, lr=0.02, run_t0=80, target=0.1):
    """Adam on the window's pose+depth under ``objective`` (a dict term -> lam).
    Reports steps to bring the true error (rel. RMS of centre + depth deviation
    from the clean window) under ``target`` of its start, or None."""
    w = slice(off, off + S)
    sp = pose_p[w].clone().float().requires_grad_(True)
    sd = depth_p[w].clone().float().requires_grad_(True)
    tp, td, tc = pose_c[w].float(), depth_c[w].float(), conf[w].float()
    if fit_mode == "window":
        fit = fit_run_gauge({}, run_t0, 0, S, sp.detach(), lambda lo, hi: tp[lo:hi], mode="prefix", u=u)
    else:
        hist = {run_t0 + k: pose_p[k].float() for k in range(off)}
        fit = fit_run_gauge(hist, run_t0, off, S, sp.detach(), lambda lo, hi: pose_c[lo:hi].float(),
                            mode=fit_mode, u=u)
    opt = torch.optim.Adam([sp, sd], lr=lr)

    def err():
        with torch.no_grad():
            ep = (sp[:, :3] - tp[:, :3]).norm(dim=-1).pow(2).mean().sqrt() / u
            ed = ((sd.clamp(min=1e-3).log() - td.clamp(min=1e-3).log()).abs().mean())
            return float(ep), float(ed)
    e0 = err()
    hit = None
    curve = [e0]
    for it in range(steps):
        opt.zero_grad()
        t = {**old_terms(sp, tp, sd, td, tc), **new_terms(sp, tp, sd, td, tc, fit, u)}
        L = sum(lam * t[k] for k, lam in objective.items() if lam != 0.0)
        if not torch.is_tensor(L):
            break
        L.backward()
        opt.step()
        e = err()
        curve.append(e)
        if hit is None and (e[0] <= target * max(e0[0], 1e-9) or e0[0] == 0) \
                and (e[1] <= target * max(e0[1], 1e-9) or e0[1] == 0):
            hit = it + 1
            break
    return {"steps_to_10pct": hit, "err_start": e0, "err_end": curve[-1], "n_steps_run": len(curve) - 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--rid", type=int, default=3)
    ap.add_argument("--fit", default="prefix", choices=["prefix", "hist"])
    ap.add_argument("--descent_steps", type=int, default=300)
    ap.add_argument("--out", default="experiments/results/nullspace_grad_probe.json")
    a = ap.parse_args()
    torch.set_num_threads(4)

    PERT = [("pose_ramp", [1.05, 1.10, 1.20]), ("depth_ramp", [1.05, 1.10, 1.20]),
            ("joint_ramp", [1.05, 1.10, 1.20]), ("rot_drift", [0.005, 0.02, 0.05]),
            ("gauge", [0.0])]
    sources = {"synthetic": synthetic_run(), "gt": gt_run(a.scene, a.rid)}
    res = {"meta": vars(a), "rows": [], "descent": []}
    ALL = OLD + NEW

    for src, (pose_c, depth_c, conf) in sources.items():
        u = median_step(pose_c[:, :3])
        for family, off, lo, hi in (("window", 0, 0, S), ("run", 192, 0, 240)):
            fit_mode = "window" if family == "window" else a.fit
            base = clean_values(pose_c, depth_c, conf, off, u)
            print(f"\n=== {src} / {family}: ramp over [{lo},{hi}), window at offset {off}, "
                  f"fit {fit_mode} ===")
            print(f"  {'perturbation':<18}" + "".join(f"{k:>12}" for k in ALL))
            for kind, mags in PERT:
                for mag in mags:
                    pose_p, depth_p = perturb(kind, mag, pose_c, depth_c, lo, hi)
                    rec = measure(pose_c, depth_c, conf, pose_p, depth_p, off, fit_mode, u)
                    row = {"source": src, "family": family, "kind": kind, "mag": mag,
                           "fit_mode": rec["fit_mode"], "fit_s": rec["fit_s"],
                           "terms": {k: {**rec[k], "dL": rec[k]["value"] - base[k]} for k in ALL}}
                    res["rows"].append(row)
                    lab = f"{kind} {mag}"
                    print(f"  {lab:<18}" + "".join(f"{row['terms'][k]['dL']:>12.2e}" for k in ALL) + "   dL")
                    print(f"  {'':<18}" + "".join(
                        f"{row['terms'][k]['cos_pose']:>+12.2f}" if row["terms"][k]["cos_pose"] == row["terms"][k]["cos_pose"] else f"{'-':>12}"
                        for k in ALL) + "   cos(-g_pose, back)")
                    print(f"  {'':<18}" + "".join(
                        f"{row['terms'][k]['cos_depth']:>+12.2f}" if row["terms"][k]["cos_depth"] == row["terms"][k]["cos_depth"] else f"{'-':>12}"
                        for k in ALL) + "   cos(-g_depth, back)")

            # descent on the joint ramp 1.10, the case the plan names
            pose_p, depth_p = perturb("joint_ramp", 1.10, pose_c, depth_c, lo, hi)
            objs = {
                "A1PC": {**OLD_LAM},
                "gtscale": {**OLD_LAM, "trans_scale": 1.0, "depth_scale": 0.5},
                "gtpaper": {"L_rot": OLD_LAM["L_rot"], "trans_scale": 1.0, "depth_scale": 0.5,
                            "abs_pos": 1.0, "abs_rot": 1.0, "rel_trans": 1.0},
            }
            print(f"  descent from joint ramp 1.10 (Adam, {a.descent_steps} steps): steps to undo 90%")
            for name, obj in objs.items():
                d = descent(obj, pose_c, depth_c, conf, pose_p, depth_p, off, fit_mode, u,
                            steps=a.descent_steps)
                res["descent"].append({"source": src, "family": family, "objective": name, **d})
                print(f"    {name:<8} {str(d['steps_to_10pct']):>5}   err(pose steps, |dlog depth|) "
                      f"{d['err_start'][0]:.3f}/{d['err_start'][1]:.3f} -> {d['err_end'][0]:.3f}/{d['err_end'][1]:.3f}")

    # ── pass criteria ──
    # 1e-3: abs_pos is in units of u (~0.02 in teacher units on MCD), so float32
    # on a x3 / 47-unit gauge leaves ~1e-4 of it; the smallest signal in the
    # table (joint ramp 1.05) is three orders above that
    ok_gauge = all(abs(r["terms"][k]["dL"]) < 1e-3 for r in res["rows"] if r["kind"] == "gauge" for k in ALL)
    joint = [r for r in res["rows"] if r["kind"] == "joint_ramp"]
    ok_new_sees = all(r["terms"]["trans_scale"]["dL"] > 1e-5 and r["terms"]["depth_scale"]["dL"] > 1e-5 for r in joint)
    ok_new_dir = all(r["terms"]["trans_scale"]["cos_pose"] > 0 and r["terms"]["depth_scale"]["cos_depth"] > 0 for r in joint)
    old_blind = all(abs(r["terms"]["L_depth"]["dL"]) < 1e-6 for r in joint)
    res["pass"] = {"gauge_all_zero": ok_gauge, "new_terms_see_joint_ramp": ok_new_sees,
                   "new_terms_point_back": ok_new_dir, "L_depth_blind_to_joint_ramp": old_blind}
    print("\n=== pass criteria ===")
    for k, v in res["pass"].items():
        print(f"  {k:<32} {'PASS' if v else 'FAIL'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"[probe] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
