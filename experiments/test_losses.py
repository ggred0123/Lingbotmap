"""Loss unit tests -- docs/add_loss.md §4 step 1 (baseline pinned) and step 2.

Two jobs:

  REGRESSION   the bare ``SelfDistillLoss()`` must still be the four-term loss
               that produced gate 6 / 6b, bit-for-bit.  ``L_motion-depth`` ships
               disabled at the module default so no existing caller moves.

  BEHAVIOUR    ``L_motion-depth`` exists to catch one specific thing: the pose
               scale drifting away from the depth scale WITHIN a run.  A term
               that merely correlates with contamination is not enough -- it has
               to be blind to the Sim(3) gauge and sensitive to exactly that
               mismatch.  Both halves are asserted here, on synthetic data with
               an analytically known answer, so a future edit that breaks the
               invariance shows up as a failed test rather than as a slightly
               worse training curve three days later.

The decisive case is ``test_motion_depth_catches_what_mag_misses``: scale the
student's translations and leave its depth alone.  That is not a gauge -- one
run has one unit -- yet ``L_mag`` is unchanged by construction because it fits
the scale away.  ``L_motion-depth`` must report exactly |log sigma|.

CPU only, no checkpoint, no data.  Runs in a second.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.train.alignment import (
    align, ratio_trunc, ratio_weights, robust_scale, subsample_for_align)
from lingbot_map.train.losses import (
    MIN_FRAMES_REL_POSE, PRESETS, SelfDistillLoss, closed_form_scale,
    motion_depth_loss, quat_to_R, rel_pose_loss, depth_si_loss,
)

TOL = 1e-5
S, H, W = 12, 8, 10


def synth(seed=0):
    """A short plausible trajectory: forward motion, small yaw, textured depth."""
    g = torch.Generator().manual_seed(seed)
    t = torch.zeros(S, 3)
    t[:, 2] = torch.arange(S, dtype=torch.float32) * 0.35
    t += torch.randn(S, 3, generator=g) * 0.01
    yaw = torch.arange(S, dtype=torch.float32) * 0.04
    q = torch.stack([torch.zeros(S), torch.sin(yaw / 2),
                     torch.zeros(S), torch.cos(yaw / 2)], dim=-1)   # XYZW
    pose = torch.cat([t, q, torch.full((S, 2), 0.9)], dim=-1)       # [S, 9]
    depth = 3.0 + torch.rand(S, H, W, generator=g) * 4.0
    return pose, depth


def apply_sim3(pose, depth, sigma=1.0, R_g=None, t_g=None, scale_depth=True):
    """c' = sigma*R_g*c + t_g,  R' = R_g*R,  D' = sigma*D."""
    pose = pose.clone()
    c, q = pose[:, :3], pose[:, 3:7]
    R = quat_to_R(q)
    if R_g is not None:
        c = torch.einsum("ij,nj->ni", R_g, c)
        R = torch.einsum("ij,njk->nik", R_g, R)
        # R -> quaternion (XYZW), via the trace formula; S is small so keep it plain
        w = torch.sqrt((1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(min=1e-12)) / 2
        q = torch.stack([(R[:, 2, 1] - R[:, 1, 2]) / (4 * w),
                         (R[:, 0, 2] - R[:, 2, 0]) / (4 * w),
                         (R[:, 1, 0] - R[:, 0, 1]) / (4 * w), w], dim=-1)
    c = c * sigma
    if t_g is not None:
        c = c + t_g
    pose[:, :3], pose[:, 3:7] = c, q
    return pose, (depth * sigma if scale_depth else depth.clone())


def parts_of(sp, tp, sd, td, **kw):
    crit = SelfDistillLoss(lam_motion=1.0, **kw)
    return crit(sp, tp, sd, td)[1]


# ─────────────────────────────────────────────────────────────────────────────

def test_bare_default_is_the_old_four_term_loss():
    """Module default must equal PRESETS['B0'] and give L_motion-depth zero weight."""
    crit = SelfDistillLoss()
    lam_rot, lam_dir, lam_mag, lam_depth, lam_motion = crit.lam
    assert (lam_rot, lam_dir, lam_mag, lam_depth) == (1.0, 1.0, 0.5, 1.0)
    assert lam_motion == 0.0, "the new term must ship disabled"
    assert PRESETS["B0"] == dict(lam_rot=1.0, lam_dir=1.0, lam_mag=0.5,
                                 lam_motion=0.0, lam_depth=1.0)

    sp, sd = synth(0)
    tp, td = synth(1)
    total, parts = crit(sp, tp, sd, td)
    L_rot, L_dir, L_mag = rel_pose_loss(sp, tp)
    L_dep = depth_si_loss(sd, td)
    expect = 1.0 * L_rot + 1.0 * L_dir + 0.5 * L_mag + 1.0 * L_dep
    assert torch.allclose(total, expect, atol=0, rtol=0), \
        f"B0 total drifted: {float(total)} vs {float(expect)}"
    assert parts["L_motion_depth"] == 0.0
    print("  [ok] bare default == B0, bit-for-bit, new term inert")


def test_identical_runs_are_at_the_floor():
    """student == teacher: every term is 0 -- except L_rot, which cannot reach it.

    ``rel_pose_loss`` clamps cos to 1-1e-6 before acos, so perfect agreement
    still scores acos(1-1e-6) = 0.081 deg.  fp32 round-trip through
    quat -> R -> trace lands a hair above even that.  The consequence is a DEAD
    ZONE: any rotation agreement finer than ~0.08 deg is both unmeasurable and
    unlearnable, because acos saturates and the gradient is exactly zero there.
    Measured on the real model at t0=80, 7 of 47 pairs already sit inside it --
    and the fraction grows as training improves rotation, so this floor is a
    live constraint on how far L_rot can drive the model, not a curiosity.
    """
    p, d = synth(0)
    parts = parts_of(p, p.clone(), d, d.clone())
    floor = math.acos(1 - 1e-6)
    assert floor <= parts["L_rot_rad"] < 1.1 * floor, \
        f"L_rot floor moved: {parts['L_rot_rad']} vs clamp floor {floor}"
    for k in ("L_dir", "L_mag", "L_depth_si", "L_motion_depth"):
        assert abs(parts[k]) < TOL, f"{k} = {parts[k]}, expected 0"
    print(f"  [ok] identical runs -> 0, except L_rot = {parts['L_rot_rad']:.4e} rad "
          f"({parts['L_rot_deg']:.4f} deg) -- the acos clamp dead zone")


def test_every_term_is_blind_to_a_full_sim3():
    """A real gauge: rotate, translate, and scale BOTH depth and pose together."""
    p, d = synth(0)
    ang = 0.7
    R_g = torch.tensor([[math.cos(ang), 0., math.sin(ang)],
                        [0., 1., 0.],
                        [-math.sin(ang), 0., math.cos(ang)]])
    base = parts_of(p, p.clone(), d, d.clone())
    sp, sd = apply_sim3(p, d, sigma=3.7, R_g=R_g, t_g=torch.tensor([5., -2., 11.]))
    moved = parts_of(sp, p.clone(), sd, d.clone())
    for k in ("L_rot_rad", "L_dir", "L_mag", "L_depth_si", "L_motion_depth"):
        assert abs(moved[k] - base[k]) < 1e-4, \
            f"{k} moved under a pure Sim(3): {base[k]} -> {moved[k]}"
    print("  [ok] all five terms invariant to (sigma=3.7, R, t) applied consistently")


def test_motion_depth_catches_what_mag_misses():
    """THE point of the term.  Scale translations only; leave depth alone.

    One run has one unit, so this is an ERROR, not a gauge.  L_mag fits the
    scale away and reports nothing; L_motion-depth must report exactly |log s|.
    """
    p, d = synth(0)
    for sigma in (1.5, 4.0, 13.52):          # 13.52 = the measured t0=5248 mismatch
        sp, sd = apply_sim3(p, d, sigma=sigma, scale_depth=False)
        m = parts_of(sp, p.clone(), sd, d.clone())
        assert abs(m["L_mag"]) < TOL, \
            f"L_mag should be blind here but read {m['L_mag']}"
        assert abs(m["L_dir"]) < TOL and abs(m["L_depth_si"]) < TOL
        assert abs(m["L_motion_depth"] - math.log(sigma)) < 1e-4, \
            f"expected |log {sigma}| = {math.log(sigma):.4f}, got {m['L_motion_depth']:.4f}"
        print(f"  [ok] pose scale x{sigma:<6}  L_mag {m['L_mag']:.2e} (blind)   "
              f"L_motion-depth {m['L_motion_depth']:.4f} = log {sigma}")


def test_motion_depth_is_symmetric_in_depth():
    """Scaling depth only must fire equally -- the mismatch has no preferred side."""
    p, d = synth(0)
    sigma = 4.0
    _, sd = apply_sim3(p, d, sigma=sigma, scale_depth=True)      # depth x sigma
    m = motion_depth_loss(p, p.clone(), sd, d.clone())
    assert abs(float(m) - math.log(sigma)) < 1e-4, float(m)
    print(f"  [ok] depth-only scale x{sigma} -> {float(m):.4f} = log {sigma}")


def test_min_frames_guard():
    p, d = synth(0)
    n = MIN_FRAMES_REL_POSE - 1
    for fn in (lambda: rel_pose_loss(p[:n], p[:n]),
               lambda: motion_depth_loss(p[:n], p[:n], d[:n], d[:n])):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError below {MIN_FRAMES_REL_POSE} frames")
    print(f"  [ok] both pose terms raise below {MIN_FRAMES_REL_POSE} frames")


def test_presets():
    assert PRESETS["A1"]["lam_mag"] == 0.0 and PRESETS["A1"]["lam_motion"] > 0
    assert PRESETS["B1"]["lam_rot"] == 30.0
    c = SelfDistillLoss.from_preset("A1", lam_depth=2.0)
    assert c.lam[3] == 2.0 and c.lam[4] == PRESETS["A1"]["lam_motion"]
    try:
        SelfDistillLoss.from_preset("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown preset should raise")
    print("  [ok] presets B0/B1/A1 + override + unknown-name guard")


def test_gradient_reaches_both_heads():
    """L_motion-depth must pull on pose AND depth -- that is what couples them."""
    p, d = synth(0)
    sp = p.clone().requires_grad_(True)
    sd = d.clone().requires_grad_(True)
    tp, td = synth(1)
    motion_depth_loss(sp, tp, sd, td).backward()
    assert sp.grad is not None and sp.grad[:, :3].abs().sum() > 0, "no gradient to translation"
    assert sd.grad is not None and sd.grad.abs().sum() > 0, "no gradient to depth"
    print(f"  [ok] grad reaches translation ({sp.grad[:, :3].abs().sum():.3e}) "
          f"and depth ({sd.grad.abs().sum():.3e})")


# ─────────────────────────────────────────────────────────────────────────────
# What was ported from Pi3: (a) the robust scale solver, (b) all-pairs relatives
# ─────────────────────────────────────────────────────────────────────────────

def pose_from(t, yaw):
    """[S,3] centres + [S] yaw about +Y -> the [S,9] encoding the losses take."""
    z = torch.zeros_like(yaw)
    q = torch.stack([z, torch.sin(yaw / 2), z, torch.cos(yaw / 2)], dim=-1)  # XYZW
    return torch.cat([t, q, torch.full((len(t), 2), 0.9)], dim=-1)


def test_align_actually_minimises_the_l1_objective():
    """The solver is exact, not iterative -- so check it against brute force.

    Every candidate optimum of ``sum_i w_i |a x_i - y_i|`` is some ratio
    ``y_k/x_k``; evaluating all of them is the definition we are claiming to
    solve in closed form.
    """
    g = torch.Generator().manual_seed(3)
    for trial in range(5):
        x = torch.rand(40, generator=g) + 0.2
        y = torch.rand(40, generator=g) + 0.2
        w = torch.rand(40, generator=g)
        a, loss, _ = align(x, y, w)
        obj = lambda v: (w * (v * x - y).abs()).sum()
        brute = min(float(obj(float(c))) for c in (y / x))
        assert float(loss) <= brute + 1e-5, f"{float(loss)} > brute {brute}"
        assert abs(float(obj(float(a))) - float(loss)) < 1e-5
    print("  [ok] align() attains the brute-force L1 minimum on 5 random problems")


def test_l1_scale_ignores_outliers_that_least_squares_obeys():
    """WHY the port matters.  Same data, same term, different estimator.

    90% of the samples say the scale is exactly 2.0.  Ten percent are garbage --
    the depth-discontinuity halos and sky pixels of a real frame.  Least squares
    is a mean-like statistic and moves; the L1 fit is a weighted median of the
    ratios and does not.
    """
    x = torch.linspace(1.0, 5.0, 100)
    y = 2.0 * x
    y[:10] = 400.0                                   # 10% gross outliers

    s_l2 = float(closed_form_scale(x, y))
    s_l1 = float(robust_scale(x, y))
    s_tr = float(robust_scale(x, y, trunc=torch.tensor(1.0)))

    assert abs(s_l1 - 2.0) < 1e-4, f"L1 fit moved to {s_l1}"
    assert abs(s_tr - 2.0) < 1e-4, f"truncated fit moved to {s_tr}"
    assert s_l2 > 3.0, f"L2 fit was supposed to be dragged, got {s_l2}"
    print(f"  [ok] 10% outliers:  L2 -> {s_l2:.3f}   L1 -> {s_l1:.3f}   "
          f"trunc-L1 -> {s_tr:.3f}   (truth 2.000)")


def test_truncated_branch_is_the_exact_argmin():
    """The trunc objective is NON-CONVEX -- walking the derivative is not enough,
    so the solver enumerates extrema and takes the best.  Check it really wins.

    Brute force here is a dense sweep UNION the three breakpoint families the
    solver claims are the only candidates; if that union ever beats the solver,
    either the enumeration or the claim is wrong.
    """
    g = torch.Generator().manual_seed(7)
    for trial in range(3):
        x = 0.3 + 3 * torch.rand(30, generator=g)
        y = 0.5 * x + 0.4 * torch.randn(30, generator=g)
        w = 0.2 + torch.rand(30, generator=g)
        for tr in (0.05, 0.2, 1.0):
            _, loss, _ = align(x, y, w, trunc=tr)
            obj = lambda v: float((w * (v * x - y).abs()).clamp(max=tr).sum())
            cand = torch.cat([y / x, (w * y - tr) / (w * x), (w * y + tr) / (w * x),
                              torch.linspace(-3, 3, 4001)])
            best = min(obj(float(c)) for c in cand)
            assert float(loss) <= best + 1e-5, \
                f"trunc={tr}: solver {float(loss)} lost to {best}"
    print("  [ok] truncated solver attains the brute-force minimum (3 trials x 3 caps)")


def test_align_is_the_weighted_median_of_ratios():
    """The identity the whole design rests on:

        sum_i w_i |a x_i - y_i|  =  sum_i (w_i x_i) |a - r_i|

    so with ``ratio_weights`` the answer must be EXACTLY the conf-weighted
    median of y/x.  If this ever drifts, the estimator has silently become
    something else and every claim about robustness stops following.
    """
    def wmedian(r, w):
        o = r.argsort(); r, w = r[o], w[o]
        return float(r[torch.searchsorted(w.cumsum(0), 0.5 * w.sum())])
    g = torch.Generator().manual_seed(11)
    for trial in range(3):
        x = 0.3 + 3 * torch.rand(5000, generator=g)
        y = x * (0.5 + 0.3 * torch.randn(5000, generator=g))
        c = 0.5 + 10 * torch.rand(5000, generator=g)
        a = float(robust_scale(x, y, c))
        assert abs(a - wmedian(y / x, c)) < 1e-4, f"{a} != {wmedian(y / x, c)}"
    print("  [ok] robust_scale == conf-weighted median(y/x), exactly")


def test_trunc_is_a_scale_free_relative_tolerance():
    """``trunc`` must mean the same thing regardless of gauge or conf scaling.

    ``align`` caps the WEIGHTED residual, which ``ratio_weights`` has already put
    in ratio units -- so a cap quoted in depth units (what a naive port passes)
    silently changes tolerance with the depth scale AND the confidence scale.
    ``ratio_trunc`` puts it back.
    """
    g = torch.Generator().manual_seed(5)
    t = 0.5 + 4 * torch.rand(4, 4000, generator=g)
    c = 0.5 + 10 * torch.rand(4, 4000, generator=g)
    REL = 0.1

    def tol(s_, t_, c_):
        cap = ratio_trunc(s_, t_, c_, rel=REL)
        r_ref = (t_.median(dim=-1, keepdim=True).values
                 / s_.median(dim=-1, keepdim=True).values)
        return float(((cap / c_) / r_ref).median())

    for name, s_, t_, c_ in [("baseline", 2 * t, t, c),
                             ("depth gauge x37", 2 * t * 37, t * 37, c),
                             ("conf x100", 2 * t, t, c * 100),
                             ("student x0.01", 0.02 * t, t, c)]:
        v = tol(s_, t_, c_)
        assert abs(v - REL) < 1e-4, f"{name}: tolerance drifted to {v}, expected {REL}"
    print("  [ok] trunc = 0.1 stays a 10% ratio tolerance under gauge / conf / scale changes")


def test_align_gradient_is_the_l1_subgradient():
    """The argmin index is chosen without grad; the VALUE is rebuilt as
    ``y[k]/x[k]`` so exactly one pair carries gradient.  That is the subgradient
    of an L1 objective, and it is what makes the fit differentiable at all."""
    g = torch.Generator().manual_seed(13)
    x = (0.3 + 3 * torch.rand(50, generator=g)).requires_grad_(True)
    y = 0.5 * x.detach() + 0.1 * torch.randn(50, generator=g)
    a, _, idx = align(x, y, torch.ones(50))
    a.backward()
    nz = (x.grad != 0).nonzero().flatten()
    assert len(nz) == 1 and int(nz[0]) == int(idx), \
        f"grad should touch only the selected pair {int(idx)}, got {nz.tolist()}"
    expect = float(-y[idx] / x[idx].detach() ** 2)
    assert abs(float(x.grad[idx]) - expect) < 1e-5
    print(f"  [ok] d a/d x_k = {float(x.grad[idx]):.5f} = -y_k/x_k^2 at the selected pair only")


def test_ratio_weighting_is_what_makes_the_l1_fit_robust():
    """The defect that a bare port of Pi3's ``align`` has, pinned so it cannot return.

    ``align`` minimises ``sum w_i |a x_i - y_i| = sum (w_i x_i) |a - r_i|``, so
    the vote on each ratio is ``w * x``, not ``w``.  Left alone, the samples with
    the largest x -- which for depth is the far field and for a corrupted pixel
    is the corruption itself -- buy their own influence, and the "robust" fit
    ends up WORSE than least squares.  ``ratio_weights`` divides the x back out.

    Numbers here match the real-bank measurement in docs/add_loss.md §7-1.
    """
    g = torch.Generator().manual_seed(0)
    n = 20000
    x = 0.5 + 4.0 * torch.rand(n, generator=g)          # "student depth"
    y = 0.5 * x                                          # truth: a = 0.5
    bad = torch.rand(n, generator=g) < 0.10              # 10% blown up x30
    x = torch.where(bad, x * 30.0, x)

    a_raw = float(robust_scale(x, y, ratio_weighted=False))
    a_fix = float(robust_scale(x, y))
    a_l2 = float(closed_form_scale(x, y))
    assert abs(a_fix - 0.5) < 0.02, f"ratio-weighted fit should hold 0.5, got {a_fix}"
    assert abs(a_raw - 0.5) > 0.1, \
        "the raw objective is supposed to FAIL here -- if it no longer does, " \
        "this test has stopped guarding anything"
    assert abs(a_l2 - 0.5) > 0.1
    # and the weights really are the reciprocal of the (floored) magnitude
    w = ratio_weights(x)
    assert torch.allclose(w * x.clamp_min(0.1 * x.median()), torch.ones_like(w), atol=1e-4)
    print(f"  [ok] 10% x30 outliers:  L2 {a_l2:.3f}   L1 raw {a_raw:.3f}   "
          f"L1 ratio-weighted {a_fix:.3f}   (truth 0.500)")


def test_l1_depth_mode_keeps_lam_depth_calibrated():
    """A different gauge estimator must not silently rescale the TERM.

    ``lam_depth=1.0`` was tuned against the median-normalised term, and A1L reuses
    that weight, so the l1 modes have to land on the same order of magnitude or
    the preset means something different from what §5 says it means.
    """
    p, d = synth(0)
    sigma = 2.0
    g = torch.Generator().manual_seed(1)
    sd = sigma * d * (1 + 0.05 * torch.randn(d.shape, generator=g))
    ref = float(depth_si_loss(sd, d, mode="median"))
    for mode in ("l1", "trunc_l1"):
        v = float(depth_si_loss(sd, d, mode=mode, align_res=32))
        assert 0.5 * ref < v < 2.0 * ref, \
            f"{mode} reads {v} against median {ref} -- lam_depth no longer transfers"
    print(f"  [ok] median {ref:.4f}; l1 and trunc_l1 within 2x -> lam_depth transfers")


def test_robust_depth_modes_are_still_sim3_blind():
    """A new estimator must not cost the gauge invariance the loss is built on."""
    p, d = synth(0)
    ang = 0.7
    R_g = torch.tensor([[math.cos(ang), 0., math.sin(ang)],
                        [0., 1., 0.],
                        [-math.sin(ang), 0., math.cos(ang)]])
    sp, sd = apply_sim3(p, d, sigma=3.7, R_g=R_g, t_g=torch.tensor([5., -2., 11.]))
    for mode in ("l1", "l1_linear", "trunc_l1", "trunc_l1_linear"):
        # align_res below H*W so the subsample path is exercised too
        base = depth_si_loss(d, d.clone(), mode=mode, align_res=32)
        moved = depth_si_loss(sd, d.clone(), mode=mode, align_res=32)
        assert float(base) < TOL, f"{mode}: identical depths gave {float(base)}"
        assert float(moved) < 1e-4, f"{mode}: a pure depth scale leaked in as {float(moved)}"
    print("  [ok] l1 / trunc_l1 (x linear, log) all blind to sigma=3.7, subsampled fit")


def test_all_pairs_is_sim3_blind_too():
    p, d = synth(0)
    ang = 0.7
    R_g = torch.tensor([[math.cos(ang), 0., math.sin(ang)],
                        [0., 1., 0.],
                        [-math.sin(ang), 0., math.cos(ang)]])
    sp, sd = apply_sim3(p, d, sigma=3.7, R_g=R_g, t_g=torch.tensor([5., -2., 11.]))
    kw = dict(pairs="all", lam_motion=1.0)
    base = SelfDistillLoss(**kw)(p, p.clone(), d, d.clone())[1]
    moved = SelfDistillLoss(**kw)(sp, p.clone(), sd, d.clone())[1]
    for k in ("L_rot_rad", "L_dir", "L_mag", "L_depth_si", "L_motion_depth"):
        assert abs(moved[k] - base[k]) < 1e-4, f"{k}: {base[k]} -> {moved[k]} under pairs='all'"
    print("  [ok] pairs='all' keeps every term Sim(3)-blind")


def test_all_pairs_sees_drift_that_consecutive_pairs_cannot():
    """THE point of the all-pairs port.  Two students, identical per-step error.

    A: every step is off by +e in yaw, always the same sign -- the error
       ACCUMULATES, and after 11 steps the far end is off by 11e.
    B: the yaw offset alternates +e/2, -e/2 -- the per-step error is the same
       size e, but it cancels and nothing accumulates.

    Consecutive-pair L_rot is a mean over per-step errors, so it reports the same
    number for both and is by construction incapable of preferring B.  That is
    the blind spot: drift is the accumulated quantity, and no consecutive pair
    ever contains it.  All-pairs L_rot contains every gap and must separate them.
    """
    _, d = synth(0)
    t = torch.zeros(S, 3)
    t[:, 2] = torch.arange(S, dtype=torch.float32) * 0.35
    n = torch.arange(S, dtype=torch.float32)
    e = 0.01                                          # rad/step, ~0.57 deg

    tea = pose_from(t, n * 0.04)
    stu_drift = pose_from(t, n * 0.04 + n * e)                    # accumulating
    stu_jitter = pose_from(t, n * 0.04 + ((-1.0) ** n) * (e / 2))  # cancelling

    con_drift = float(rel_pose_loss(stu_drift, tea)[0])
    con_jitter = float(rel_pose_loss(stu_jitter, tea)[0])
    all_drift = float(rel_pose_loss(stu_drift, tea, pairs="all")[0])
    all_jitter = float(rel_pose_loss(stu_jitter, tea, pairs="all")[0])

    assert abs(con_drift - e) < 1e-4 and abs(con_jitter - e) < 1e-4, \
        f"consecutive should read e={e} for both: {con_drift}, {con_jitter}"
    assert abs(con_drift - con_jitter) < 1e-5, \
        "consecutive pairs must be unable to tell these apart -- that is the premise"
    # mean gap over ordered pairs of S frames is (S+1)/3
    assert abs(all_drift - e * (S + 1) / 3) < 1e-3, f"all-pairs drift {all_drift}"
    assert all_drift > 5 * all_jitter, \
        f"all-pairs failed to separate them: drift {all_drift} vs jitter {all_jitter}"
    print(f"  [ok] consecutive: drift {con_drift:.5f} == jitter {con_jitter:.5f} (blind)")
    print(f"       all-pairs:   drift {all_drift:.5f} vs  jitter {all_jitter:.5f} "
          f"({all_drift / all_jitter:.1f}x separation)")


def test_min_gap_drops_the_short_noisy_pairs():
    p, _ = synth(0)
    from lingbot_map.train.losses import _pair_index
    i1, _ = _pair_index(S, "all", 1, p.device)
    i4, j4 = _pair_index(S, "all", 4, p.device)
    assert len(i1) == S * (S - 1), len(i1)
    assert ((i4 - j4).abs() >= 4).all()
    assert 0 < len(i4) < len(i1)
    print(f"  [ok] min_gap: {len(i1)} pairs at 1 -> {len(i4)} at 4, all with gap >= 4")


def test_motion_depth_all_pairs_still_reads_log_sigma():
    """The pose/depth mismatch is scale, not gap -- every pair must report it."""
    p, d = synth(0)
    for sigma in (1.5, 13.52):
        sp, sd = apply_sim3(p, d, sigma=sigma, scale_depth=False)
        m = motion_depth_loss(sp, p.clone(), sd, d.clone(), pairs="all")
        assert abs(float(m) - math.log(sigma)) < 1e-4, f"{sigma}: {float(m)}"
    print("  [ok] motion-depth under pairs='all' still reports exactly |log sigma|")


def test_ported_presets():
    for name in ("A1L", "A1P", "A1LP", "X1M"):
        assert name in PRESETS, name
        SelfDistillLoss.from_preset(name)          # must construct
    assert PRESETS["A1L"]["depth_mode"] == "l1" and "pairs" not in PRESETS["A1L"]
    assert PRESETS["A1P"]["pairs"] == "all" and "depth_mode" not in PRESETS["A1P"]
    # A1L / A1P each change exactly one thing relative to A1
    for name in ("A1L", "A1P"):
        for k, v in PRESETS["A1"].items():
            assert PRESETS[name][k] == v, f"{name} moved a weight: {k}"
    p, d = synth(0)
    tp, td = synth(1)
    for name in ("A1", "A1L", "A1P", "A1LP", "X1M"):
        total, _ = SelfDistillLoss.from_preset(name)(p, tp, d, td)
        assert torch.isfinite(total), f"{name} produced {total}"
    print("  [ok] A1L / A1P / A1LP / X1M construct, isolate one change each, run finite")


def test_gradient_survives_the_robust_fit():
    """An L1 fit is a subgradient through one selected sample -- but the RESIDUAL
    is dense, so the term as a whole must still pull on every frame."""
    p, d = synth(0)
    tp, td = synth(1)
    sp, sd = p.clone().requires_grad_(True), d.clone().requires_grad_(True)
    total, _ = SelfDistillLoss.from_preset("A1LP")(sp, tp, sd, td)
    total.backward()
    assert sp.grad is not None and sp.grad[:, :3].abs().sum() > 0
    assert sd.grad is not None and sd.grad.abs().sum() > 0
    touched = (sd.grad.reshape(S, -1).abs().sum(dim=1) > 0).sum()
    assert int(touched) == S, f"only {int(touched)}/{S} frames received depth gradient"
    print(f"  [ok] A1LP: grad reaches translation and all {S}/{S} depth frames")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"{'=' * 72}\n  losses.py -- {len(fns)} tests\n{'=' * 72}")
    for fn in fns:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'=' * 72}\n  ALL PASS\n{'=' * 72}")
