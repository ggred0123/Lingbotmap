"""L_long unit tests -- docs/long_supervision_plan.md Stage 3, gate G3.

Seven properties, all on synthetic data with an analytically known answer, so a
future edit that breaks one shows up as a failed test rather than as a slightly
worse training curve three days later.

  GAUGE      the three terms must be blind to an independent Sim(3) on the
             student and on the teacher, and to nothing else.  This is the whole
             reason the formulation is relative: the student rolls from its own
             anchor, the teacher from another, and the two differ by exactly one
             similarity transform.
  RAMP       L_scale must SEE a step-scale that drifts with time -- the failure
             a single Sim(3) alignment cannot remove and L_rot + L_dir cannot
             report.  Monotone in the ramp rate, zero at rate 0.
  INDEXING   delta >= S is a design constraint (the anchor must stay outside the
             supervised window); the coverage table in the design doc is a
             consequence of the 48-grid and is asserted here so a change to the
             stream stride cannot silently change the sample count.
  MASK       a near-zero teacher baseline makes both log-magnitude and direction
             ill-conditioned; those pairs must be dropped, not merely damped.

CPU only, no checkpoint, no data.  Runs in a second.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.train.long_loss import (                          # noqa: E402
    LongPoseLoss, angle_huber, build_pairs, log_huber, long_pair_index, rel_pair,
    v_local)
from lingbot_map.utils.rotation import mat_to_quat, quat_to_mat    # noqa: E402

TOL = 1e-5
S = 48

#: ★ L_rot HAS A FLOOR.  ``acos`` is clamped at ``1 - 1e-6`` (inherited from
#: losses.rel_pose_loss), so two IDENTICAL rotations score acos(1-1e-6) =
#: 1.41e-3 rad rather than 0.  Tests that expect "no rotation error" must expect
#: this value, not zero -- and so must anyone reading L_rot off a training log.
#: The reported ``long_rot_rad_d*`` is that raw angle; the term's CONTRIBUTION is
#: the angle-Huber of it, which below 1 deg is quadratic and therefore much
#: smaller still.
ROT_FLOOR = float(torch.acos(torch.tensor(1 - 1e-6)))
ROT_HUBER_DEG = 1.0


def traj(n, seed=0, step=0.35, ramp=0.0, yaw_rate=0.03, lateral=0.4, jitter=0.002):
    """Forward motion with small yaw.  ``ramp`` grows the step length with time.

    pose[:3] is the camera CENTRE in world coordinates, pose[3:7] the cam->world
    rotation as XYZW -- the convention experiments/mcd_eval.py validates against
    GT and the one losses._relative assumes.
    """
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(n, dtype=torch.float64)
    d = step * (1.0 + ramp * i)                 # per-step length
    z = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(d, 0)[:-1]])
    C = torch.stack([torch.sin(i * 0.02) * lateral, torch.zeros(n, dtype=torch.float64), z], -1)
    C = C + torch.randn(n, 3, generator=g, dtype=torch.float64) * jitter
    yaw = i * yaw_rate
    q = torch.stack([torch.zeros(n, dtype=torch.float64), torch.sin(yaw / 2),
                     torch.zeros(n, dtype=torch.float64), torch.cos(yaw / 2)], -1)
    return torch.cat([C, q, torch.full((n, 2), 0.9, dtype=torch.float64)], -1).float()


def apply_sim3(pose, seed=0, sigma=None):
    """C -> sigma Q C + c,  R -> Q R.  The gauge the two runs differ by."""
    g = torch.Generator().manual_seed(seed)
    ax = torch.randn(3, generator=g, dtype=torch.float64)
    ax = ax / ax.norm()
    ang = torch.tensor(0.7, dtype=torch.float64)
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]],
                     dtype=torch.float64)
    Q = torch.eye(3, dtype=torch.float64) + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)
    c = torch.randn(3, generator=g, dtype=torch.float64) * 5.0
    s = torch.rand(1, generator=g, dtype=torch.float64).item() * 2 + 0.3 if sigma is None else sigma
    C = (s * (Q @ pose[:, :3].double().T)).T + c
    R = Q[None] @ quat_to_mat(pose[:, 3:7].double())
    return torch.cat([C, mat_to_quat(R), pose[:, 7:].double()], -1).float()


def pairs_from(stu, tea, off, delta, S=S):
    """(stu_anchor, tea_anchor, k) for a window at run-offset ``off``.

    ``stu``/``tea`` are indexed by RUN OFFSET here, so window k maps to off + k.
    """
    k, a = long_pair_index(off, S, delta)
    return {delta: (stu[a], tea[a], torch.as_tensor(k))}


def win(p, off, S=S):
    return p[off:off + S]


# ── 1. gauge ────────────────────────────────────────────────────────────────

def test_sim3_invariance():
    n, off, d = 300, 192, 192
    st, te = traj(n, seed=1, ramp=0.002), traj(n, seed=2)
    crit = LongPoseLoss(ladder=(d,), tau=0.0, scale_gauge_norm=False)
    base, _ = crit(win(st, off), win(te, off), pairs_from(st, te, off, d))
    for seed in (3, 4, 5):
        st2, te2 = apply_sim3(st, seed), apply_sim3(te, seed + 10)
        got, _ = crit(win(st2, off), win(te2, off), pairs_from(st2, te2, off, d))
        assert abs(float(got) - float(base)) < 2e-3, (float(got), float(base))
    print("  [1] sim3 invariance                 ok")


def test_pure_global_scale_is_zero():
    """A student that is the teacher times sigma is a GAUGE, not an error."""
    n, off, d = 300, 192, 96
    te = traj(n, seed=7)
    st = apply_sim3(te, seed=8, sigma=3.7)
    crit = LongPoseLoss(ladder=(d,), tau=0.0, terms=("scale",), scale_gauge_norm=False)
    got, parts = crit(win(st, off), win(te, off), pairs_from(st, te, off, d))
    assert float(got) < 1e-4, (float(got), parts)
    print("  [2] pure global scale -> L_scale=0   ok")


# ── 2. the ramp L_rot + L_dir cannot see ────────────────────────────────────

def test_scale_ramp_monotone():
    n, off, d = 300, 192, 192
    te = traj(n, seed=11, ramp=0.0)
    crit = LongPoseLoss(ladder=(d,), tau=0.0, terms=("scale",), huber_delta=1e-3,
                        scale_gauge_norm=False)
    prev, vals = -1.0, []
    for ramp in (0.0, 0.001, 0.002, 0.004, 0.008):
        st = traj(n, seed=11, ramp=ramp)
        v = float(crit(win(st, off), win(te, off), pairs_from(st, te, off, d))[0])
        vals.append(round(v, 6))
        assert v > prev - 1e-9, vals
        prev = v
    assert vals[0] < 1e-6 and vals[-1] > 10 * max(vals[0], 1e-6), vals
    print(f"  [3] scale ramp monotone              ok  {vals}")


def test_rot_dir_blind_to_ramp():
    """The premise of section 3: the angular terms do NOT report a scale ramp.

    ★ THE CONSTRUCTION HAS TO BE STRAIGHT-LINE MOTION.  On a curved path a step
    ramp also bends the shape, so the directions genuinely differ and L_dir
    reports it -- which is real signal, not blindness.  The null space the design
    doc names is the case where rotation and direction are IDENTICAL and only the
    magnitude drifts, and that is straight motion with a growing step.  Yaw is
    kept (it is shared by both runs) so the test still exercises the R_i^T in
    rel_pair rather than degenerating to world coordinates.
    """
    n, off, d = 300, 192, 192
    kw = dict(lateral=0.0, jitter=0.0)
    te = traj(n, seed=11, ramp=0.0, **kw)
    st = traj(n, seed=11, ramp=0.008, **kw)
    ang = LongPoseLoss(ladder=(d,), tau=0.0, terms=("rot", "dir"))
    sca = LongPoseLoss(ladder=(d,), tau=0.0, terms=("scale",), huber_delta=1e-3,
                       scale_gauge_norm=False)
    a0 = float(ang(win(te, off), win(te, off), pairs_from(te, te, off, d))[0])
    a = float(ang(win(st, off), win(te, off), pairs_from(st, te, off, d))[0])
    s = float(sca(win(st, off), win(te, off), pairs_from(st, te, off, d))[0])
    floor = 15.0 * float(angle_huber(torch.tensor(ROT_FLOOR),
                                     ROT_HUBER_DEG / 57.29578))
    assert a0 < 5 * floor, (a0, floor)          # the floor and nothing else
    assert a - a0 < 1e-6 and s > 1e-2, (a0, a, s)
    print(f"  [4] rot+dir blind to ramp            ok  ang {a - a0:.2e} over floor, "
          f"scale {s:.2e}")


# ── 3. indexing ─────────────────────────────────────────────────────────────

def test_pair_index_coverage():
    """The design doc's coverage table, re-derived from the index function."""
    got = {}
    for d in (48, 96, 192):
        got[d] = [off for off in (0, 48, 96, 144, 192)
                  if len(long_pair_index(off, S, d)[0]) > 0]
        for off in got[d]:
            k, a = long_pair_index(off, S, d)
            assert len(k) == S and a[0] == off - d and a[-1] == off - d + S - 1
    assert got == {48: [48, 96, 144, 192], 96: [96, 144, 192], 192: [192]}, got
    print(f"  [5] pair index coverage              ok  {got}")


def test_delta_below_S_raises():
    for d in (1, 24, 47):
        try:
            long_pair_index(96, S, d)
        except ValueError:
            continue
        raise AssertionError(f"delta={d} < S={S} must raise")
    print("  [6] delta < S raises                 ok")


# ── 4. mask ─────────────────────────────────────────────────────────────────

def test_near_zero_baseline_masked():
    """A round trip that returns to its start must be dropped, not merely damped.

    ★ THE MASK IS ON THE LONG BASELINE, NOT ON THE LOCAL STEP.  Freezing a few
    frames does nothing to ||d_{i,i+Delta}||; what kills it is motion whose PERIOD
    is Delta, so the camera is back where it started Delta frames later.  Then
    both the log magnitude and the direction are decided by noise, which is
    exactly the pair the teacher-side tau exists to reject.
    """
    n, off, d = 300, 192, 96
    g = torch.Generator().manual_seed(31)
    i = torch.arange(n, dtype=torch.float64)
    te = traj(n, seed=13, lateral=0.0, jitter=0.0).double()
    te[:, 2] = 2.0 * torch.sin(2 * np.pi * i / d)            # period == Delta
    te[:, :3] += torch.randn(n, 3, generator=g, dtype=torch.float64) * 1e-4
    te = te.float()
    st = traj(n, seed=14)

    loose = LongPoseLoss(ladder=(d,), tau=0.0)
    tight = LongPoseLoss(ladder=(d,), tau=0.5)
    _, pl = loose(win(st, off), win(te, off), pairs_from(st, te, off, d))
    v, pt = tight(win(st, off), win(te, off), pairs_from(st, te, off, d))
    assert pl[f"long_n_d{d}"] == S and pt[f"long_n_d{d}"] == 0, (pl, pt)
    assert np.isfinite(float(v)) and float(v) == 0.0, float(v)

    # and the converse: ordinary forward motion must keep every pair
    ok = traj(n, seed=15)
    _, pk = tight(win(st, off), win(ok, off), pairs_from(st, ok, off, d))
    assert pk[f"long_maskrate_d{d}"] == 1.0, pk
    print(f"  [7] near-zero mask                   ok  round-trip {pl[f'long_n_d{d}']:.0f}"
          f" -> {pt[f'long_n_d{d}']:.0f}, forward keeps {pk[f'long_n_d{d}']:.0f}")


def test_identical_is_zero():
    """Student == teacher: dir and scale are exactly 0, rot sits at its floor."""
    n, off = 300, 192
    p = traj(n, seed=17, ramp=0.003)
    crit = LongPoseLoss(ladder=(48, 96, 192), tau=0.0)
    v, parts = crit(win(p, off), win(p, off),
                    {d: pairs_from(p, p, off, d)[d] for d in (48, 96, 192)})
    for d in (48, 96, 192):
        assert parts[f"long_dir_d{d}"] < 1e-7, parts
        assert parts[f"long_scale_d{d}"] < 1e-9, parts
        assert abs(parts[f"long_rot_rad_d{d}"] - ROT_FLOOR) < 1e-7, parts
    # weight_norm is ON by default, so the total is the MEAN over the rungs
    # that fired, not their sum -- one rung's floor, not three.
    exp = 15.0 * float(angle_huber(torch.tensor(ROT_FLOOR),
                                   ROT_HUBER_DEG / 57.29578))
    assert parts["long_lam_sum"] == 3.0, parts
    assert abs(float(v) - exp) < 1e-6, (float(v), exp, parts)
    assert parts["long_pairs"] == 3 * S, parts
    print(f"  [8] identical -> rot floor only      ok  ({ROT_FLOOR:.2e} rad, "
          f"huber -> {exp:.2e})")


# ── 5. wiring helper ────────────────────────────────────────────────────────

def test_build_pairs_skips_missing_history():
    n, off, t0 = 300, 192, 80
    p = traj(n, seed=19)
    hist = {t0 + i: p[i] for i in range(n)}
    tea = lambda lo, hi: p[lo:hi]                                  # noqa: E731
    full = build_pairs((48, 96, 192), hist, tea, t0, off, S, "cpu")
    assert set(full) == {48, 96, 192}, sorted(full)
    del hist[t0 + 0]                                               # anchor of d=192
    part = build_pairs((48, 96, 192), hist, tea, t0, off, S, "cpu")
    assert set(part) == {48, 96}, sorted(part)
    for d, entry in full.items():
        stu, te_, k = entry[0], entry[1], entry[2]
        assert len(entry) == 4 and entry[3] is None, entry[3]
        assert stu.shape == (S, 9) and te_.shape == (S, 9) and k.shape == (S,)

    # a rung served by another teacher carries that teacher's window poses, so
    # anchor and target stay in one gauge
    alt = build_pairs((96,), hist, tea, t0, off, S, "cpu", tea_win=p[off:off + S])
    assert alt[96][3] is not None and alt[96][3].shape == (S, 9)
    print("  [9] build_pairs history gating       ok")


def test_v_local_gradient_scaling():
    """The loss value is gauge-invariant; the raw gradient is not.  Pin the cure.

    ``r = log(||d_long|| / v_local)`` is dimensionless, so its VALUE does not
    move with the gauge.  Its gradient is dominated by ``-1/v_local``, the
    shortest length in the problem, so a slow scene gets orders of magnitude more
    gradient for the same relative error.

    ★ DETACHING v_local IS NOT THE CURE.  It shrinks the magnitude but leaves the
    1/sigma dependence, and it changes what the term supervises: with v_local
    frozen the loss reads "assume this window's local scale is right", and
    nothing constrains that reference -- A1PC is invariant to a joint
    (pose, depth) scaling.  The cure is to keep the gradient on both sides and
    multiply by a DETACHED v_local, which cancels the 1/sigma exactly.
    """
    p = traj(60, seed=23).requires_grad_(True)
    v = v_local(p[:S])
    v.backward()
    assert p.grad is not None and float(p.grad[:S, :3].abs().sum()) > 0

    off, d = 192, 96
    cfg = {"grad": dict(detach_vlocal=False, scale_gauge_norm=False),
           "detach": dict(detach_vlocal=True, scale_gauge_norm=False),
           "fixed": dict(detach_vlocal=False, scale_gauge_norm=True)}
    g = {k: [] for k in cfg}
    vals = []
    # ★ THE WHOLE TRAJECTORY MUST SCALE, not just the step: `lateral` and
    # `jitter` are absolute amplitudes, so leaving them fixed while shrinking the
    # step changes the SHAPE (at step 0.002 the jitter would be as large as the
    # motion) and the two gauges would no longer carry the same relative error.
    for step in (2.0, 0.002):
        kw = dict(lateral=0.4 * step / 0.35, jitter=0.002 * step / 0.35)
        te = traj(300, seed=5, step=step, **kw)
        st = traj(300, seed=5, step=step, ramp=0.002, **kw)
        for k, kw in cfg.items():
            w = win(st, off).clone().requires_grad_(True)
            c = LongPoseLoss(ladder=(d,), tau=0.0, terms=("scale",), **kw)
            val, _ = c(w, win(te, off), pairs_from(st, te, off, d))
            val.backward()
            g[k].append(float(w.grad.norm()))
            if k == "grad":
                vals.append(float(val.detach()))
    assert abs(vals[0] - vals[1]) < 5e-3, vals               # value: gauge-free
    sp = {k: max(v) / min(v) for k, v in g.items()}
    assert sp["grad"] > 100, sp                              # gradient: 1/sigma
    assert sp["fixed"] < 10, sp                              # the cure works
    assert sp["fixed"] < sp["detach"] / 5, sp                # and beats detaching
    assert LongPoseLoss(ladder=(d,)).detach_vlocal is False
    assert LongPoseLoss(ladder=(d,)).scale_gauge_norm is True
    print(f"  [10] v_local gauge scaling           ok  spread over 1000x gauge: "
          f"grad {sp['grad']:.0f}x  detach {sp['detach']:.0f}x  fixed {sp['fixed']:.1f}x")


def test_weight_norm_keeps_the_mix_constant():
    """A window where 1 rung fires and one where 3 do must weigh long the same.

    ★ THE DEFECT THIS EXISTS FOR.  A rung only fires where its anchor fits inside
    the run, so on the 48-grid the lambda-sum is 0/1/3/3/7 at offsets
    0/48/96/144/192.  Under a global clip that is not "7x the learning rate" --
    it is a different local:long direction mix at every window, so the objective
    is not the same one throughout and "long supervision on/off" is not what the
    experiment compared.
    """
    n = 400
    te, st = traj(n, seed=41), traj(n, seed=42, ramp=0.002)
    lam = {48: 1.0, 96: 2.0, 192: 4.0}
    got = {}
    for off, ds in ((48, (48,)), (96, (48, 96)), (192, (48, 96, 192))):
        pairs = {d: pairs_from(st, te, off, d)[d] for d in ds}
        for norm in (False, True):
            c = LongPoseLoss(ladder=(48, 96, 192), lam_delta=lam, tau=0.0,
                             weight_norm=norm)
            v, p = c(win(st, off), win(te, off), pairs)
            got[(off, norm)] = (float(v), p["long_lam_sum"])
    assert [got[(o, False)][1] for o in (48, 96, 192)] == [1.0, 3.0, 7.0]
    raw = [got[(o, False)][0] for o in (48, 96, 192)]
    nrm = [got[(o, True)][0] for o in (48, 96, 192)]
    assert max(raw) / min(raw) > 3, raw                      # the defect
    assert max(nrm) / min(nrm) < max(raw) / min(raw) / 2, (raw, nrm)
    assert LongPoseLoss(ladder=(48,)).weight_norm is True
    print(f"  [11] weight norm                     ok  lam-sum 1/3/7 -> "
          f"spread {max(raw)/min(raw):.1f}x becomes {max(nrm)/min(nrm):.1f}x")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("L_long unit tests")
    test_sim3_invariance()
    test_pure_global_scale_is_zero()
    test_scale_ramp_monotone()
    test_rot_dir_blind_to_ramp()
    test_pair_index_coverage()
    test_delta_below_S_raises()
    test_near_zero_baseline_masked()
    test_identical_is_zero()
    test_build_pairs_skips_missing_history()
    test_v_local_gradient_scaling()
    test_weight_norm_keeps_the_mix_constant()
    print("all ok")
