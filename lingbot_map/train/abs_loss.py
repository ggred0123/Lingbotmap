"""L_abs -- run-gauge supervision.  docs/gtabs-plan.md §2.

A1PC removes an independent Sim(3) per 48-frame window, so a scale that drifts
slowly along a run sits in its null space (docs/objective-diagnosis.html §03:
21 m of ATE, 5.7e-9 of loss).  gtsup put GT labels into the teacher's gauge per
run and still lost the scale/depth/ratio channels (rho 0.98) -- the loss never
looked at the channel.  This module makes it look, at RUN scale (<= 240 frames):

    fit     ONE Sim(3) (s, R, t) per window from the student's rollout history
            against the reference poses of the SAME bank run, detached.
    terms   gtscale   L_trans-scale   run-scale of every within-window pair length
                      L_depth-scale   the SAME s applied to the student's depth
            gtpaper   + L_abs-pos, L_abs-rot, L_rel-trans-l1  (paper Eq.1 in the
                      run gauge)

★ THE FIT IS DETACHED, THE WINDOW IS NOT.  Same construction as Pi3's
``S_opt_local`` and as ``long_grad_probe.gt_window_loss``: the gauge is part of
the measurement, not of the thing being optimised.  Only the window tensors
carry gradient, through the residuals.

★ TWO ALIGNMENTS, CHOSEN BY MEASUREMENT (experiments/gauge_precheck.py):
    prefix  fit once on the run's first 48 frames and hold it for every window
            of that run.  The gauge is nailed to the run start, so a deep
            window's residual is the whole drift accumulated since.
    hist    refit every window on the accumulated history [t0, t0+off).  The
            gauge absorbs part of the drift, the way ATE's alignment does.
Both degrade to a window-inclusive fit when the history is shorter than 48
frames (offset 0) or has a hole (a re-anchor inside the run).

★ SCALE FROM THE CENTRES, ROTATION FROM THE ORIENTATIONS.  MCD runs are nearly
straight lines -- the centre cross-covariance has sigma2/sigma1 of 0.01-0.08 on
a 48-frame prefix -- so a rotation fitted on centres alone is undetermined
about the direction of travel (that is the bake's "14.4 deg" on kth_day_10 run
0, and 9-24 deg between the two fits in the first smoke run).  The SCALE is not
affected: Umeyama's s is set by the singular values, which a rotation about
the principal axis leaves alone.  So:

    s           centre Umeyama (rotation-free by construction)
    (R_L, R_C)  two-sided Procrustes on the orientations, R_ref ~= R_L R_stu R_C:
                a world-side constant and a camera-side constant (an extrinsic-
                style bias) are both absorbed, per run, detached.  What is left
                is the orientation DRIFT inside the run.
    R, t        R = R_L, t = mu_ref - s R_L mu_stu: the centres are aligned with
                the rotation the orientations pin down, which the centres of a
                straight run cannot.

The gap between the centre-only rotation and R_L (fit_rot_world_deg), the
conditioning sigma2/sigma1 (fit_cond), R_C's angle and the orientation residual
spread are logged so gauge_precheck.py can decide per run whether L_abs-rot is
usable.  gtscale uses s alone and is untouched by any of the rotation choices.

[KR] run 단위 게이지(Sim(3))를 학생 히스토리로부터 detach 상태로 맞추고, 그
게이지 아래에서 창의 절대량(scale, 위치, 회전)을 GT 참조와 비교하는 항.
gtscale은 공통 scale 하나만 감독(L_trans-scale, L_depth-scale), gtpaper는
논문 Eq.1의 절대 항까지 포함한다. fit은 gradient가 없고 창 텐서만 잔차를
통해 gradient를 받는다.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn

from .alignment import robust_scale, subsample_for_align
from .losses import quat_to_R
from .long_loss import angle_huber, log_huber

#: the run's first this-many frames define the ``prefix`` gauge (= one window)
PREFIX_FRAMES = 48

FIT_MODES = ("prefix", "hist", "span", "fallback", "identity", "skipped")
FIT_MODE_ID = {m: i for i, m in enumerate(FIT_MODES)}

ABS_TERMS = ("trans_scale", "depth_scale", "abs_pos", "abs_rot", "rel_trans")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry -- pure tensor functions, float64, no grad
# ─────────────────────────────────────────────────────────────────────────────

# ★ NUMPY, NOT TORCH, FOR THE FIT.  These are 3x3 SVDs over at most 240
# rotations, and torch dispatches even those across its whole thread pool: with
# the 72 threads this container advertises (16 of quota) one fit took minutes of
# spinning where numpy takes 5-10 ms.  The loss terms below stay in torch --
# they carry gradient -- but nothing in the fit does.

def procrustes_rotation(R_src: np.ndarray, R_dst: np.ndarray) -> np.ndarray:
    """R maximising ``tr(R^T sum_i R_dst,i R_src,i^T)``, i.e. the world-side
    rotation with ``R R_src,i ~= R_dst,i`` in the least-squares sense.
    ``R_src`` / ``R_dst``: [n, 3, 3]."""
    M = np.einsum("nij,nkj->ik", R_dst, R_src)               # sum R_dst R_src^T
    U, _, Vh = np.linalg.svd(M)
    d = np.linalg.det(U @ Vh)
    return U @ np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0]) @ Vh


def _procrustes_right(R_left_src: np.ndarray, R_dst: np.ndarray) -> np.ndarray:
    """R_C maximising ``tr(R_C^T sum_i R_left_src,i^T R_dst,i)``, the camera-side
    constant with ``R_left_src,i R_C ~= R_dst,i``."""
    M = np.einsum("nji,njk->ik", R_left_src, R_dst)
    U, _, Vh = np.linalg.svd(M)
    d = np.linalg.det(U @ Vh)
    return U @ np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0]) @ Vh


def umeyama(c_src: np.ndarray, c_dst: np.ndarray):
    """Sim(3) ``c_dst ~= s R c_src + t`` in closed form (Umeyama 1991) -- the
    same estimator experiments/mcd_eval.py scores ATE with."""
    n = c_src.shape[0]
    mu_s, mu_d = c_src.mean(0), c_dst.mean(0)
    Sc, Dc = c_src - mu_s, c_dst - mu_d
    C = Dc.T @ Sc / n
    U, sig, Vh = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        W[2, 2] = -1
    R = U @ W @ Vh
    var_s = (Sc ** 2).sum() / n
    s = float((sig * np.diag(W)).sum() / max(var_s, 1e-12))
    t = mu_d - s * (R @ mu_s)
    return s, R, t, sig


def geodesic_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M = np.einsum("nji,njk->nik", A, B)
    cos = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.arccos(cos)


def procrustes_two_sided(R_src: np.ndarray, R_dst: np.ndarray,
                         R_L0: Optional[np.ndarray] = None, iters: int = 40):
    """(R_L, R_C) minimising ``sum_i || R_L R_src,i R_C - R_dst,i ||_F^2``.

    Alternating one-sided Procrustes.  The alternation converges only linearly
    and where it lands depends on which side moves first (a yaw-only track
    cannot tell a world-side from a camera-side constant), so both orders are
    run from the same seed and the lower residual wins.  ``R_L0`` seeds the
    world side with the centre fit's R: the orientation gauge starts at the
    centre gauge and moves only if the orientations ask it to.
    """
    I = np.eye(3)
    R_L0 = I if R_L0 is None else R_L0
    best = None
    for right_first in (True, False):
        R_L, R_C = R_L0.copy(), I.copy()
        for _ in range(max(1, iters)):
            if right_first:
                R_C = _procrustes_right(R_L[None] @ R_src, R_dst)
                R_L = procrustes_rotation(R_src @ R_C, R_dst)
            else:
                R_L = procrustes_rotation(R_src @ R_C, R_dst)
                R_C = _procrustes_right(R_L[None] @ R_src, R_dst)
        resid = float(geodesic_np(np.einsum("ij,njk,kl->nil", R_L, R_src, R_C), R_dst).mean())
        if best is None or resid < best[0]:
            best = (resid, R_L, R_C)
    return best[1], best[2], best[0]


def rot_angle_deg(R: np.ndarray) -> float:
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def quat_to_R_np(q: np.ndarray) -> np.ndarray:
    return quat_to_R(torch.as_tensor(q, dtype=torch.float64)).numpy()


def geodesic(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Angle between rotations, [n] radians.  Same clamp as losses.rel_pose_loss,
    so identical rotations sit at the 1.41e-3 rad floor rather than 0."""
    M = torch.einsum("nji,njk->nik", A, B)                  # A^T B
    cos = ((M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]) - 1) / 2
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))


def _np(x) -> np.ndarray:
    return x.detach().cpu().double().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)


def path_length(c) -> float:
    c = _np(c)
    return float(np.linalg.norm(c[1:] - c[:-1], axis=-1).sum()) if c.shape[0] > 1 else 0.0


def median_step(c, eps: float = 1e-6) -> float:
    """The run's unit ``u``: median consecutive step of the reference centres."""
    c = _np(c)
    if c.shape[0] < 2:
        return eps
    return max(float(np.median(np.linalg.norm(c[1:] - c[:-1], axis=-1))), eps)


# ─────────────────────────────────────────────────────────────────────────────
# The gauge fit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GaugeFit:
    """One gauge taking the STUDENT into the REFERENCE, all detached:
        centres        ``c_ref ~= s R c_stu + t``
        orientations   ``R_ref ~= R_L R_stu R_C``   (R_L == R when consistent)"""
    s: torch.Tensor                 # 0-dim
    R: torch.Tensor                 # [3, 3]  centre gauge
    t: torch.Tensor                 # [3]
    mode: str                       # one of FIT_MODES
    R_L: Optional[torch.Tensor] = None   # [3, 3]  world-side orientation constant
    R_C: Optional[torch.Tensor] = None   # [3, 3]  camera-side orientation constant
    n: int = 0                      # frames the fit used
    path: float = 0.0               # reference path over the fit range, in units of u
    lo: int = 0                     # fit range, run offsets [lo, hi)
    hi: int = 0
    rot_world_deg: float = 0.0      # angle between the centre-only rotation and R_L
    rot_cam_deg: float = 0.0        # angle of R_C
    rot_resid_deg: float = 0.0      # mean orientation residual after both constants
    cond: float = 1.0               # sigma2/sigma1 of the centre fit: ~0 = straight line

    def __post_init__(self):
        if self.R_L is None:
            self.R_L = self.R.clone()
        if self.R_C is None:
            self.R_C = torch.eye(3, dtype=self.R.dtype, device=self.R.device)

    @classmethod
    def identity(cls, device=None, dtype=torch.float32) -> "GaugeFit":
        return cls(s=torch.ones((), device=device, dtype=dtype),
                   R=torch.eye(3, device=device, dtype=dtype),
                   t=torch.zeros(3, device=device, dtype=dtype),
                   mode="identity", n=0, path=0.0)

    def to(self, device, dtype=torch.float32) -> "GaugeFit":
        cv = lambda x: x.to(device=device, dtype=dtype)
        return GaugeFit(cv(self.s), cv(self.R), cv(self.t), self.mode, cv(self.R_L), cv(self.R_C),
                        self.n, self.path, self.lo, self.hi,
                        self.rot_world_deg, self.rot_cam_deg, self.rot_resid_deg, self.cond)

    def parts(self, prefix: str = "fit") -> Dict[str, float]:
        return {f"{prefix}_s": float(self.s), f"{prefix}_log_s": float(self.s.clamp(min=1e-12).log()),
                f"{prefix}_mode": self.mode, f"{prefix}_mode_id": float(FIT_MODE_ID[self.mode]),
                f"{prefix}_n": float(self.n), f"{prefix}_path": float(self.path),
                f"{prefix}_lo": float(self.lo), f"{prefix}_hi": float(self.hi),
                f"{prefix}_rot_world_deg": self.rot_world_deg,
                f"{prefix}_rot_cam_deg": self.rot_cam_deg,
                f"{prefix}_rot_resid_deg": self.rot_resid_deg, f"{prefix}_cond": self.cond}


@torch.no_grad()
def depth_scale_fallback(stu_depth: torch.Tensor, tea_depth: torch.Tensor,
                         conf: Optional[torch.Tensor] = None,
                         align_res: int = 4096) -> torch.Tensor:
    """``s`` with ``s * D_stu ~= D_tea`` -- the conf-weighted median depth ratio
    over the window (the anchor point-cloud ratio the plan names, read off the
    only place both gauges' depth is available).  Used when the pose fit is
    degenerate: a straight or stationary stretch cannot fix a pose scale, but
    the two depth maps still share the run's unit."""
    s = stu_depth[..., 0] if stu_depth.dim() == 4 else stu_depth
    t = tea_depth[..., 0] if tea_depth.dim() == 4 else tea_depth
    S = s.shape[0]
    s = s.reshape(S, -1).clamp(min=1e-3).float()
    t = t.reshape(S, -1).clamp(min=1e-3).float()
    w = conf.reshape(S, -1).float() if conf is not None else None
    ss, tt, ww = subsample_for_align(s, t, w, n=align_res)
    a = robust_scale(ss.reshape(-1), tt.reshape(-1),
                     None if ww is None else ww.reshape(-1))
    return a.detach().double()


@torch.no_grad()
def fit_run_gauge(hist: Dict[int, torch.Tensor], run_t0: int, off: int, S: int,
                  win_pose: torch.Tensor,
                  ref_poses: Callable[[int, int], torch.Tensor],
                  mode: str = "prefix", u: Optional[float] = None,
                  min_path: float = 10.0, min_frames: int = 12,
                  fallback_s: Optional[Callable[[], torch.Tensor]] = None) -> GaugeFit:
    """Sim(3) from the student gauge to the reference gauge for ONE window.

    ``hist``       absolute frame -> [9] student pose (RolloutStream.hist),
                   filled by the rollout and cleared at every re-anchor.
    ``run_t0``     the bank run's first supervised frame; ``off`` the window's
                   offset inside it; ``win_pose`` [S, 9] the window's student
                   poses (used detached, and only when the history is short).
    ``ref_poses``  (lo, hi) -> [hi-lo, 9] reference poses at run offsets.
    ``mode``       'prefix' -- fit on run offsets [0, 48), fixed for the run;
                   'hist'   -- fit on [0, off), refit per window.
                   Both fall back to a window-inclusive fit ('span') when the
                   requested range is shorter than one window or has a hole.
    ``u``          the run's unit (median reference step); computed over the
                   fit range when None.
    ``fallback_s`` () -> scale to use when the fit is degenerate (see the
                   guard), called only then; None keeps s = 1.

    Guard: fewer than ``min_frames`` frames or a reference path shorter than
    ``min_path`` units (u = the run's median GT step) cannot determine a pose
    scale.  ★ 10u, not the plan's 20u: MCD runs often start from rest, and the
    48-frame prefix of kth_day_10 run 3 covers 19.9u -- a 20u guard would have
    thrown every window of such a run onto the fallback.  At 10u of extent the
    Umeyama scale is still conditioned to well under 1% against frame-level
    pose noise; fit_path is logged per window so the margin is visible.
    On a degenerate range s = ``fallback_s`` and (R, t) align the window's
    first frame -- the term still fires (DDP needs every rank to build the same
    graph) and the mode is logged so the share of degenerate windows is a number.
    """
    if mode not in ("prefix", "hist"):
        raise ValueError(f"unknown abs fit mode {mode!r}; expected 'prefix' or 'hist'")
    dev = win_pose.device
    wp = _np(win_pose)

    def gather(lo: int, hi: int):
        """student poses for run offsets [lo, hi) from hist (None if a hole)."""
        rows = []
        for k in range(lo, hi):
            p = hist.get(run_t0 + k)
            if p is None:
                return None
            rows.append(_np(p))
        return np.stack(rows) if rows else None

    # ── choose the range ────────────────────────────────────────────────────
    fit_mode, lo, hi, stu = "span", 0, off + S, None
    if mode == "prefix" and off >= PREFIX_FRAMES:
        stu = gather(0, PREFIX_FRAMES)
        if stu is not None:
            fit_mode, lo, hi = "prefix", 0, PREFIX_FRAMES
    elif mode == "hist" and off >= PREFIX_FRAMES:
        stu = gather(0, off)
        if stu is not None:
            fit_mode, lo, hi = "hist", 0, off
    if stu is None:
        # window-inclusive: every history frame that IS there, plus the window
        keep, rows = [], []
        for k in range(0, off):
            p = hist.get(run_t0 + k)
            if p is not None:
                keep.append(k); rows.append(_np(p))
        keep += list(range(off, off + S))
        rows += list(wp)
        stu = np.stack(rows)
        idx = np.asarray(keep)
        ref = _np(ref_poses(0, off + S))[idx]
        lo, hi = int(idx.min()), int(idx.max()) + 1
    else:
        ref = _np(ref_poses(lo, hi))

    c_s, c_r = stu[:, :3], ref[:, :3]
    R_s, R_r = quat_to_R_np(stu[:, 3:7]), quat_to_R_np(ref[:, 3:7])
    if u is None:
        u = median_step(ref_poses(0, hi)[:, :3])
    path = path_length(c_r) / max(float(u), 1e-6)
    n = int(stu.shape[0])

    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
    ok = n >= min_frames and path >= min_path
    if ok:
        s, R_c, _, sig = umeyama(c_s, c_r)
        ok = bool(np.isfinite(s)) and s > 1e-6
    if ok:
        R_L, R_C, resid = procrustes_two_sided(R_s, R_r, R_L0=R_c)
        # the centres follow the rotation the ORIENTATIONS determine (see the
        # module docstring: a straight run cannot pin R_c about its own axis)
        t = c_r.mean(0) - s * (R_L @ c_s.mean(0))
        fit = GaugeFit(s=T(s), R=T(R_L), t=T(t), mode=fit_mode, R_L=T(R_L), R_C=T(R_C),
                       n=n, path=path, lo=lo, hi=hi,
                       rot_world_deg=rot_angle_deg(R_L.T @ R_c),
                       rot_cam_deg=rot_angle_deg(R_C),
                       rot_resid_deg=float(np.degrees(resid)),
                       cond=float(sig[1] / max(sig[0], 1e-30)))
    else:
        # degenerate: scale from depth (or 1), (R, t) from the window's first
        # frame so the absolute terms still measure something finite
        s = float(fallback_s()) if fallback_s is not None else 1.0
        if not np.isfinite(s) or s <= 1e-6:
            s = 1.0
        r0 = _np(ref_poses(off, off + 1))[0]
        R = quat_to_R_np(r0[None, 3:7])[0] @ quat_to_R_np(wp[None, 0, 3:7])[0].T
        t = r0[:3] - s * (R @ wp[0, :3])
        fit = GaugeFit(s=T(s), R=T(R), t=T(t), mode="fallback", n=n, path=path, lo=lo, hi=hi)
    return fit.to(dev)


# ─────────────────────────────────────────────────────────────────────────────
# The terms
# ─────────────────────────────────────────────────────────────────────────────

def _pairs(N: int, device):
    g = torch.arange(N, device=device)
    gi, gj = torch.meshgrid(g, g, indexing="ij")
    keep = gi != gj
    return gi[keep], gj[keep]


class RunGaugeLoss(nn.Module):
    """Run-gauge terms on one supervised window.  docs/gtabs-plan.md §2-2.

    ``lam_*`` are set from the GRADIENT share (term_grad_probe.py), never from
    the loss scalar: clipping fires every step and renormalises any overall
    scale away.  A term at lam 0 is skipped entirely, as ``SelfDistillLoss``
    skips ``L_motion`` -- no 0 * x in the graph.

    Every term is in run units: ``u`` is the run's median GT step, so
    ``L_abs-pos`` and ``L_rel-trans`` read as "steps", and the log terms are in
    relative-error units (log-Huber with delta 0.1 = 10%, as L_long).
    """

    def __init__(self, lam_trans_scale: float = 0.0, lam_depth_scale: float = 0.0,
                 lam_abs_pos: float = 0.0, lam_abs_rot: float = 0.0,
                 lam_rel_trans: float = 0.0,
                 huber_delta: float = 0.1, rot_huber_deg: float = 1.0,
                 tau: float = 0.5, use_teacher_conf: bool = True, eps: float = 1e-8):
        super().__init__()
        self.lam = {"trans_scale": lam_trans_scale, "depth_scale": lam_depth_scale,
                    "abs_pos": lam_abs_pos, "abs_rot": lam_abs_rot,
                    "rel_trans": lam_rel_trans}
        self.huber_delta = huber_delta
        self.rot_huber = float(rot_huber_deg) / 57.29578
        #: pairs whose REFERENCE displacement is below tau * u are dropped from
        #: the log-magnitude and relative-translation terms: a stationary pair
        #: makes the log ratio noise (same rule as LongPoseLoss.tau).
        self.tau = tau
        self.use_teacher_conf = use_teacher_conf
        self.eps = eps

    @property
    def active(self):
        return tuple(k for k in ABS_TERMS if self.lam[k] != 0.0)

    def forward(self, stu_pose: torch.Tensor, ref_pose: torch.Tensor,
                stu_depth: Optional[torch.Tensor], ref_depth: Optional[torch.Tensor],
                ref_conf: Optional[torch.Tensor], fit: GaugeFit, u: float,
                rot_enabled: bool = True):
        """``stu_pose``/``ref_pose`` [S, 9]; depth [S, H, W(, 1)].  ``fit`` from
        ``fit_run_gauge`` (or ``GaugeFit.identity()`` on the identity branch).
        ``rot_enabled=False`` zeroes L_abs-rot for runs gauge_precheck excluded."""
        ref_pose = ref_pose.detach()
        S = stu_pose.shape[0]
        dev = stu_pose.device
        s, R, t = fit.s.to(dev), fit.R.to(dev), fit.t.to(dev)
        R_L, R_C = fit.R_L.to(dev), fit.R_C.to(dev)
        u = max(float(u), 1e-6)
        parts: Dict[str, float] = {}
        total = stu_pose.new_zeros(())

        c_s, c_r = stu_pose[:, :3], ref_pose[:, :3]
        i, j = _pairs(S, dev)
        d_s, d_r = c_s[j] - c_s[i], c_r[j] - c_r[i]
        n_s = d_s.norm(dim=-1).clamp(min=self.eps)
        n_r = d_r.norm(dim=-1)
        keep = n_r > self.tau * u
        n_keep = int(keep.sum())
        parts["abs_n_pairs"] = float(n_keep)
        parts["abs_maskrate"] = n_keep / max(1, keep.numel())

        need_R = any(self.lam[k] != 0.0 for k in ("abs_pos", "abs_rot", "rel_trans"))
        if need_R:
            R_s, R_r = quat_to_R(stu_pose[:, 3:7]), quat_to_R(ref_pose[:, 3:7])

        if self.lam["trans_scale"] != 0.0:
            r = (s * n_s).log() - n_r.clamp(min=self.eps).log()
            L = log_huber(r[keep], self.huber_delta).mean() if n_keep else stu_pose.new_zeros(())
            total = total + self.lam["trans_scale"] * L
            parts["L_trans_scale"] = float(L.detach())
            parts["abs_trans_resid"] = float(r[keep].abs().mean().detach()) if n_keep else 0.0
            parts["abs_trans_bias"] = float(r[keep].mean().detach()) if n_keep else 0.0

        if self.lam["depth_scale"] != 0.0 and stu_depth is not None and ref_depth is not None:
            ds = stu_depth[..., 0] if stu_depth.dim() == 4 else stu_depth
            dr = ref_depth[..., 0] if ref_depth.dim() == 4 else ref_depth
            ds = ds.reshape(S, -1).clamp(min=1e-3)
            dr = dr.reshape(S, -1).detach().clamp(min=1e-3)
            err = ((s * ds).log() - dr.log()).abs()
            if ref_conf is not None and self.use_teacher_conf:
                w = ref_conf.reshape(S, -1).detach()
                L = (err * w).sum() / w.sum().clamp(min=self.eps)
                bias = (((s * ds).log() - dr.log()).detach() * w).sum() / w.sum().clamp(min=self.eps)
            else:
                L = err.mean()
                bias = ((s * ds).log() - dr.log()).detach().mean()
            total = total + self.lam["depth_scale"] * L
            parts["L_depth_scale"] = float(L.detach())
            parts["abs_depth_bias"] = float(bias)

        if self.lam["abs_pos"] != 0.0:
            al = s * (c_s @ R.T) + t
            L = (al - c_r).norm(dim=-1).mean() / u
            total = total + self.lam["abs_pos"] * L
            parts["L_abs_pos"] = float(L.detach())

        if self.lam["abs_rot"] != 0.0:
            th = geodesic(torch.einsum("ij,njk,kl->nil", R_L, R_s, R_C), R_r)
            L = angle_huber(th, self.rot_huber).mean()
            if rot_enabled:
                total = total + self.lam["abs_rot"] * L
            parts["L_abs_rot"] = float(L.detach()) if rot_enabled else 0.0
            parts["abs_rot_deg"] = float(th.mean().detach()) * 57.29578
            parts["abs_rot_enabled"] = float(rot_enabled)

        if self.lam["rel_trans"] != 0.0:
            # reference: R_r,i^T d_r.  With R_r,i ~= R_L R_s,i R_C and d_r ~= s R_L d_s
            # that is s R_C^T R_s,i^T d_s: the world-side constant cancels and
            # the camera-side one is applied, so only the drift is left.
            rs = torch.einsum("ji,nkj,nk->ni", R_C, R_s[i], d_s) * s
            rr = torch.einsum("nji,nj->ni", R_r[i], d_r)            # R_r,i^T d_r
            L = ((rs - rr).norm(dim=-1)[keep].mean() / u) if n_keep else stu_pose.new_zeros(())
            total = total + self.lam["rel_trans"] * L
            parts["L_rel_trans"] = float(L.detach())

        parts["L_abs"] = float(total.detach())
        parts.update(fit.parts())
        parts["abs_u"] = u
        return total, parts


# ─────────────────────────────────────────────────────────────────────────────
# Trainer entry point
# ─────────────────────────────────────────────────────────────────────────────

def _locate(bank, t: int, S: int):
    for rid, r in enumerate(bank.runs):
        if r["t0"] <= t and t + S <= r["t0"] + r["L"]:
            return rid, t - r["t0"]
    raise KeyError(f"[{t}, {t + S}) is not inside a single run")


class RunUnitCache:
    """``u`` per (bank, run), computed once from the run's full reference track."""

    def __init__(self):
        self._u: Dict[Tuple[int, int], float] = {}

    def __call__(self, bank, rid: int) -> float:
        key = (id(bank), rid)
        if key not in self._u:
            L = bank.runs[rid]["L"]
            self._u[key] = median_step(bank.poses(rid, 0, L).double()[:, :3])
        return self._u[key]


def abs_term(crit: RunGaugeLoss, bank, hist: Optional[Dict[int, torch.Tensor]],
             t: int, S: int, stu_pose: torch.Tensor, stu_depth: Optional[torch.Tensor],
             lab: dict, dev, fit_mode: str = "prefix", identity: bool = False,
             min_path: float = 10.0, max_offset: int = 0,
             units: Optional[RunUnitCache] = None,
             rot_exclude: Optional[Dict[str, set]] = None, scene: str = "") -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """L_abs for one supervised window, on the same ``out`` the local loss
    scored -- no extra forward.  Returns ``(loss_or_None, parts)``; None only
    when ``max_offset`` excludes the window (logged as fit_mode 'skipped').

    ``identity=True`` is the identity branch: the student anchors where the
    teacher did, so the gauge is (1, I, 0) by construction and no history is
    needed (docs/gtabs-plan.md §2-3).
    """
    rid, off = _locate(bank, t, S)
    r = bank.runs[rid]
    parts = {"abs_offset": float(off)}
    if max_offset and off > max_offset:
        parts.update(GaugeFit.identity(dev).parts())
        parts["fit_mode"], parts["fit_mode_id"] = "skipped", float(FIT_MODE_ID["skipped"])
        parts["L_abs"] = 0.0
        return None, parts
    u = (units or RunUnitCache())(bank, rid)
    if identity:
        fit = GaugeFit.identity(dev)
    else:
        fb = (lambda: depth_scale_fallback(stu_depth.detach(), lab["depth"],
                                           lab.get("depth_conf"))) \
            if (stu_depth is not None and lab.get("depth") is not None) else None
        fit = fit_run_gauge(hist or {}, r["t0"], off, S, stu_pose.detach(),
                            lambda lo, hi: bank.poses(rid, lo, hi),
                            mode=fit_mode, u=u, min_path=min_path, fallback_s=fb)
    rot_ok = True
    if rot_exclude and scene in rot_exclude and rid in rot_exclude[scene]:
        rot_ok = False
    total, p = crit(stu_pose, lab["pose_enc"], stu_depth, lab.get("depth"),
                    lab.get("depth_conf"), fit, u, rot_enabled=rot_ok)
    parts.update(p)
    return total, parts
