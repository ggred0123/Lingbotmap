"""L_abs unit tests -- docs/gtabs-plan.md §5-1.  CPU, synthetic, seconds.

  ZERO       student == reference -> every term 0 (L_abs-rot at the Huber floor)
  GAUGE      a global Sim(3) on the student is absorbed by the fit -> 0; a constant
             camera-side orientation bias on the student is absorbed by the
             two-sided Procrustes -> every term 0; a straight run (centre fit
             undetermined about the travel axis) still aligns exactly
  RAMP       a joint scale ramp 1 -> 1.5 inside the run makes L_trans-scale and
             L_depth-scale grow monotonically
  ALIGN      prefix == hist at offset 0; at offset 192 prefix residual >= hist
  GUARD      straight / stationary runs trip the degeneracy guard, no NaN
  NEGATIVE   dropping the fitted s from L_trans-scale must BREAK the Sim(3)
             test -- otherwise the test is not testing the thing it claims
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lingbot_map.train.abs_loss import (                            # noqa: E402
    GaugeFit, RunGaugeLoss, fit_run_gauge, median_step, procrustes_rotation,
    procrustes_two_sided, rot_angle_deg, umeyama)
from lingbot_map.train.losses import quat_to_R                       # noqa: E402
from lingbot_map.utils.rotation import mat_to_quat, quat_to_mat      # noqa: E402

S, L = 48, 240
ROT_FLOOR = float(torch.acos(torch.tensor(1 - 1e-6)))


def traj(n, seed=0, step=0.15, yaw_rate=0.02, lateral=0.5, jitter=0.0, turn=True):
    """Camera centres + cam->world XYZW.  A gentle S-curve so the Procrustes fit
    is well posed; ``turn=False`` makes it a straight line (degenerate)."""
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(n, dtype=torch.float64)
    yaw = i * yaw_rate if turn else torch.zeros(n, dtype=torch.float64)
    d = torch.stack([torch.sin(yaw), torch.zeros(n, dtype=torch.float64), torch.cos(yaw)], -1) * step
    C = torch.cat([torch.zeros(1, 3, dtype=torch.float64), torch.cumsum(d, 0)[:-1]])
    if turn:
        C = C + torch.stack([torch.zeros(n, dtype=torch.float64),
                             torch.sin(i * 0.05) * lateral * 0.2,
                             torch.zeros(n, dtype=torch.float64)], -1)
    C = C + torch.randn(n, 3, generator=g, dtype=torch.float64) * jitter
    q = torch.stack([torch.zeros(n, dtype=torch.float64), torch.sin(yaw / 2),
                     torch.zeros(n, dtype=torch.float64), torch.cos(yaw / 2)], -1)
    return torch.cat([C, q, torch.full((n, 2), 0.9, dtype=torch.float64)], -1).float()


def rot_about(ax, ang):
    ax = torch.as_tensor(ax, dtype=torch.float64); ax = ax / ax.norm()
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def apply_sim3(pose, s=1.0, Q=None, c=None):
    """C -> s Q C + c,  R -> Q R  (world-side)."""
    Q = torch.eye(3, dtype=torch.float64) if Q is None else Q
    c = torch.zeros(3, dtype=torch.float64) if c is None else c
    C = (s * (Q @ pose[:, :3].double().T)).T + c
    R = Q[None] @ quat_to_mat(pose[:, 3:7].double())
    return torch.cat([C, mat_to_quat(R), pose[:, 7:].double()], -1).float()


def ramped(pose, ramp):
    """Per-step scale ramps log-linearly 1 -> ramp along the run (a slow common
    scale drift, the counterexample of objective-diagnosis §03)."""
    C = pose[:, :3].double()
    steps = C[1:] - C[:-1]
    n = steps.shape[0]
    g = torch.exp(torch.linspace(0, math.log(ramp), n, dtype=torch.float64))[:, None]
    C2 = torch.cat([C[:1], C[:1] + torch.cumsum(steps * g, 0)])
    return torch.cat([C2, pose[:, 3:].double()], -1).float(), g.squeeze(-1)


def depth_for(n, h=6, w=8, seed=0, base=4.0):
    g = torch.Generator().manual_seed(seed)
    return (base + torch.rand(n, h, w, 1, generator=g) * 3.0)


def hist_of(pose, t0=80):
    return {t0 + k: pose[k].clone() for k in range(pose.shape[0])}


def run_fit(stu, ref, off, mode, **kw):
    return fit_run_gauge(hist_of(stu), 80, off, S, stu[off:off + S],
                         lambda lo, hi: ref[lo:hi], mode=mode,
                         u=median_step(ref[:, :3].double()), **kw)


def crit_all(**kw):
    return RunGaugeLoss(lam_trans_scale=1.0, lam_depth_scale=1.0, lam_abs_pos=1.0,
                        lam_abs_rot=1.0, lam_rel_trans=1.0, **kw)


def terms(crit, stu, ref, dep_s, dep_r, off, fit):
    u = median_step(ref[:, :3].double())
    tot, p = crit(stu[off:off + S], ref[off:off + S], dep_s[off:off + S],
                  dep_r[off:off + S], None, fit, u)
    return float(tot), p


# ── 1. zero ─────────────────────────────────────────────────────────────────

def test_identical_is_zero():
    ref = traj(L, seed=1)
    dep = depth_for(L)
    crit = crit_all()
    for off in (0, 48, 192):
        for mode in ("prefix", "hist"):
            fit = run_fit(ref, ref, off, mode)
            tot, p = terms(crit, ref, ref, dep, dep, off, fit)
            assert abs(p["L_trans_scale"]) < 1e-6, p
            assert abs(p["L_depth_scale"]) < 1e-6, p
            assert abs(p["L_abs_pos"]) < 1e-5, p
            assert abs(p["L_rel_trans"]) < 1e-5, p
            assert p["L_abs_rot"] < 0.5 * ROT_FLOOR ** 2 / math.radians(1.0) * 1.01, p
            assert abs(float(fit.s) - 1) < 1e-6 and fit.mode in ("prefix", "hist", "span"), fit
    print("  [1] identical -> 0                    ok")


# ── 2. gauge ────────────────────────────────────────────────────────────────

def test_global_sim3_is_absorbed():
    ref = traj(L, seed=2)
    dep_r = depth_for(L)
    crit = crit_all()
    for seed, (s, ang) in enumerate([(3.7, 0.7), (0.31, 2.1), (1.0, 0.0)]):
        Q = rot_about([0.3, 1.0, -0.2], ang)
        c = torch.tensor([5.0, -2.0, 11.0], dtype=torch.float64) * seed
        stu = apply_sim3(ref, 1.0 / s, Q.T, -(Q.T @ c) / s)     # inverse gauge on the student
        dep_s = dep_r / s                                       # depth shares the unit
        for off in (0, 48, 192):
            for mode in ("prefix", "hist"):
                fit = run_fit(stu, ref, off, mode)
                assert abs(float(fit.s) - s) < 1e-4 * s, (float(fit.s), s, fit.mode)
                tot, p = terms(crit, stu, ref, dep_s, dep_r, off, fit)
                assert abs(p["L_trans_scale"]) < 1e-6, p
                assert abs(p["L_depth_scale"]) < 1e-5, p
                assert abs(p["L_abs_pos"]) < 1e-4, p
                assert abs(p["L_rel_trans"]) < 1e-4, p
                assert p["abs_rot_deg"] < 0.1, p
    print("  [2] global Sim(3) absorbed             ok")


def test_constant_orientation_bias_is_absorbed():
    """Two things the two-sided fit has to survive.  (i) The STUDENT's
    orientations carry a constant camera-side bias (an extrinsic-style error,
    the physical form of the bake's 14.4 deg) against a consistent reference:
    every term must still read zero.  (ii) A straight run: the centre-only
    rotation is undetermined about the travel axis, so the centres have to be
    aligned with the rotation the orientations pin down, or L_abs-pos reads a
    residual that is a fitting artifact rather than drift."""
    crit = crit_all()
    dep = depth_for(L)
    ref = traj(L, seed=3)
    Q = rot_about([1.0, 0.2, 0.1], math.radians(14.4))
    Rs = quat_to_mat(ref[:, 3:7].double()) @ Q[None]                 # camera-side bias
    stu = torch.cat([ref[:, :3], mat_to_quat(Rs).float(), ref[:, 7:]], -1)
    for off in (48, 192):
        fit = run_fit(stu, ref, off, "prefix")
        assert abs(float(fit.s) - 1) < 1e-6 and fit.rot_resid_deg < 0.1, fit
        tot, p = terms(crit, stu, ref, dep, dep, off, fit)
        assert p["abs_rot_deg"] < 0.1 and p["L_abs_pos"] < 1e-4 and p["L_rel_trans"] < 1e-4, p
        assert abs(p["L_trans_scale"]) < 1e-6 and abs(p["L_depth_scale"]) < 1e-6, p
    # (ii) straight line, student in a rotated gauge: the centre fit is
    # ill-conditioned (cond ~ 0) but the orientation fit is exact
    ref = traj(L, seed=4, turn=False)
    Qg = rot_about([0.2, 1.0, 0.3], 0.8)
    stu = apply_sim3(ref, 2.0, Qg.T)
    fit = run_fit(stu, ref, 192, "prefix")
    assert fit.cond < 0.05, fit.cond
    tot, p = terms(crit, stu, ref, dep / 2.0, dep, 192, fit)
    assert abs(float(fit.s) - 0.5) < 1e-6 and p["L_abs_pos"] < 1e-4 and p["abs_rot_deg"] < 0.1, (fit, p)
    print("  [3] constant orientation bias         ok  (camera-side bias; straight-run alignment)")


# ── 3. ramp ─────────────────────────────────────────────────────────────────

def test_joint_ramp_is_seen_and_monotone():
    ref = traj(L, seed=4)
    dep_r = depth_for(L)
    crit = RunGaugeLoss(lam_trans_scale=1.0, lam_depth_scale=1.0, huber_delta=1e-3)
    for mode in ("prefix", "hist"):
        prev_t = prev_d = -1.0
        for ramp in (1.0, 1.1, 1.25, 1.5):
            stu, g = ramped(ref, ramp)
            # depth scales with the same per-frame factor (student unit drifts)
            gf = torch.cat([g[:1], g]).float()
            dep_s = dep_r * gf[:, None, None, None]
            fit = run_fit(stu, ref, 192, mode)
            tot, p = terms(crit, stu, ref, dep_s, dep_r, 192, fit)
            assert p["L_trans_scale"] > prev_t - 1e-9 and p["L_depth_scale"] > prev_d - 1e-9, (mode, ramp, p)
            if ramp > 1.0:
                assert p["L_trans_scale"] > prev_t + 1e-4, (mode, ramp, p)
                assert p["L_depth_scale"] > prev_d + 1e-4, (mode, ramp, p)
            prev_t, prev_d = p["L_trans_scale"], p["L_depth_scale"]
    print("  [4] joint ramp -> monotone             ok")


# ── 4. prefix vs hist ───────────────────────────────────────────────────────

def test_prefix_vs_hist():
    ref = traj(L, seed=5)
    stu, _ = ramped(ref, 1.5)
    dep = depth_for(L)
    crit = RunGaugeLoss(lam_trans_scale=1.0, lam_abs_pos=1.0, huber_delta=1e-3)
    fp, fh = run_fit(stu, ref, 0, "prefix"), run_fit(stu, ref, 0, "hist")
    assert fp.mode == fh.mode == "span" and abs(float(fp.s) - float(fh.s)) < 1e-9
    # offset 48: both fit on [0, 48)
    fp, fh = run_fit(stu, ref, 48, "prefix"), run_fit(stu, ref, 48, "hist")
    assert fp.mode == "prefix" and fh.mode == "hist" and abs(float(fp.s) - float(fh.s)) < 1e-9
    # offset 192: prefix is nailed to the run start, hist absorbs part of the drift
    fp, fh = run_fit(stu, ref, 192, "prefix"), run_fit(stu, ref, 192, "hist")
    assert fp.n == 48 and fh.n == 192
    _, pp = terms(crit, stu, ref, dep, dep, 192, fp)
    _, ph = terms(crit, stu, ref, dep, dep, 192, fh)
    assert pp["L_trans_scale"] >= ph["L_trans_scale"] - 1e-9, (pp, ph)
    assert pp["L_abs_pos"] >= ph["L_abs_pos"] - 1e-9, (pp, ph)
    # a hole in the history degrades to the window-inclusive fit, not an error
    h = hist_of(stu); del h[80 + 10]
    f = fit_run_gauge(h, 80, 192, S, stu[192:240], lambda lo, hi: ref[lo:hi], mode="prefix")
    assert f.mode == "span" and f.n == 192 - 1 + 48, f
    print(f"  [5] prefix vs hist                     ok  off192 trans-scale prefix "
          f"{pp['L_trans_scale']:.4f} >= hist {ph['L_trans_scale']:.4f}")


# ── 5. guard ────────────────────────────────────────────────────────────────

def test_degenerate_runs_hit_the_guard():
    dep_r = depth_for(L)
    crit = crit_all()
    # stationary: reference path ~ 0
    ref = traj(L, seed=6, step=0.0, lateral=0.0)
    stu = ref.clone()
    fit = run_fit(stu, ref, 192, "prefix", fallback_s=lambda: torch.tensor(2.0))
    assert fit.mode == "fallback" and abs(float(fit.s) - 2.0) < 1e-6, fit
    tot, p = terms(crit, stu, ref, dep_r, dep_r, 192, fit)
    assert math.isfinite(tot) and p["abs_n_pairs"] == 0, p
    # straight line: the Procrustes/scale fit is still well posed (rotation is
    # pinned by the orientations, scale by the path), so no fallback
    ref = traj(L, seed=7, turn=False)
    fit = run_fit(ref, ref, 192, "hist")
    assert fit.mode == "hist" and abs(float(fit.s) - 1) < 1e-6, fit
    tot, p = terms(crit, ref, ref, dep_r, dep_r, 192, fit)
    assert math.isfinite(tot) and abs(p["L_trans_scale"]) < 1e-6
    # too few frames
    fit = fit_run_gauge(hist_of(ref), 80, 0, 8, ref[:8], lambda lo, hi: ref[lo:hi],
                        mode="hist", u=median_step(ref[:, :3].double()))
    assert fit.mode == "fallback", fit
    print("  [6] degeneracy guard                   ok")


# ── 6. negative control ─────────────────────────────────────────────────────

def test_negative_control_without_s_the_gauge_test_breaks():
    ref = traj(L, seed=8)
    s = 3.7
    stu = apply_sim3(ref, 1.0 / s)
    crit = RunGaugeLoss(lam_trans_scale=1.0, huber_delta=1e-3)
    fit = run_fit(stu, ref, 192, "prefix")
    dep = depth_for(L)
    _, p = terms(crit, stu, ref, dep, dep, 192, fit)
    assert abs(p["L_trans_scale"]) < 1e-6, p
    bad = GaugeFit(torch.ones(()), fit.R, fit.t, fit.mode)      # forget the scale
    _, q = terms(crit, stu, ref, dep, dep, 192, bad)
    assert q["L_trans_scale"] > 1.0, q                            # log(3.7) - delta/2
    print("  [7] negative control                   ok")


# ── 7. gradient plumbing ────────────────────────────────────────────────────

def test_gradient_reaches_pose_and_depth_but_not_the_fit():
    ref = traj(L, seed=9)
    stu, _ = ramped(ref, 1.3)
    dep_r = depth_for(L)
    win = stu[192:240].clone().requires_grad_(True)
    dep = (dep_r[192:240] * 1.2).clone().requires_grad_(True)
    fit = run_fit(stu, ref, 192, "prefix")
    crit = crit_all()
    tot, p = crit(win, ref[192:240], dep, dep_r[192:240], None, fit, median_step(ref[:, :3].double()))
    tot.backward()
    assert win.grad is not None and win.grad[:, :7].abs().sum() > 0
    assert win.grad[:, 7:].abs().sum() == 0                       # FoV untouched
    assert dep.grad is not None and dep.grad.abs().sum() > 0
    assert not fit.s.requires_grad and not fit.R.requires_grad
    print("  [8] gradient plumbing                  ok")


def test_procrustes_and_scale_closed_forms():
    g = torch.Generator().manual_seed(0)
    R_s = quat_to_mat(torch.nn.functional.normalize(torch.randn(30, 4, generator=g, dtype=torch.float64), dim=-1)).numpy()
    Q = rot_about([0.1, 0.9, 0.3], 1.2).numpy()
    R = procrustes_rotation(R_s, Q[None] @ R_s)
    assert rot_angle_deg(R.T @ Q) < 1e-6
    C = torch.randn(30, 3, generator=g, dtype=torch.float64).numpy()
    s, R2, t, _ = umeyama(C, 2.5 * (C @ Q.T) + 3.0)
    assert abs(s - 2.5) < 1e-9 and rot_angle_deg(R2.T @ Q) < 1e-6 and np.abs(t - 3.0).max() < 1e-9
    # two-sided: both constants at once, off the yaw axis so the split is identifiable
    Qc = rot_about([0.7, 0.1, 0.7], 0.4).numpy()
    R_L, R_C, resid = procrustes_two_sided(R_s, Q[None] @ R_s @ Qc[None])
    assert math.degrees(resid) < 1e-6 and rot_angle_deg(R_L.T @ Q) < 1e-4 and rot_angle_deg(R_C.T @ Qc) < 1e-4, \
        (resid, rot_angle_deg(R_L.T @ Q), rot_angle_deg(R_C.T @ Qc))
    print("  [9] closed forms                       ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("L_abs unit tests")
    test_identical_is_zero()
    test_global_sim3_is_absorbed()
    test_constant_orientation_bias_is_absorbed()
    test_joint_ramp_is_seen_and_monotone()
    test_prefix_vs_hist()
    test_degenerate_runs_hit_the_guard()
    test_negative_control_without_s_the_gauge_test_breaks()
    test_gradient_reaches_pose_and_depth_but_not_the_fit()
    test_procrustes_and_scale_closed_forms()
    print("all ok")
