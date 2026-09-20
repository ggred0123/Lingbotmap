"""L_long -- sparse long-relative pose supervision.  docs/long_supervision_design.md

The S=48 A1PC loss removes an independent Sim(3) per window, so a trajectory
deformation that is *locally* a single Sim(3) but changes slowly from window to
window sits in its quasi-null space.  This module supervises the relation
between the CURRENT window and a detached anchor Delta frames in the past, which
is exactly the relation the streaming gradient path can carry.

    L_long(Delta) = lam_rot * L_rot + lam_dir * L_dir + lam_scale * L_scale

★ DELTA >= S IS A DESIGN CONSTRAINT, NOT A TUNING CHOICE (design doc section 2).
With Delta < S the anchor ``i = t + k - Delta`` falls INSIDE the supervised
window, and two things break at once: the anchor picks up gradient, so "the past
is detached" -- the premise the whole credit-assignment argument rests on -- is
no longer true; and that pair is already supervised by A1PC's ``pairs="all"``
(gaps 1..S-1), so the term is double counted.  ``long_pair_index`` raises.

★ THE SCALE TERM IS DEPTH-FREE AND MUST STAY THAT WAY.  L_rot + L_dir cannot see
a translation scale that drifts slowly with time: the rotations agree, the
directions agree, only the magnitude ramps.  One global Sim(3) removes ONE
scale, not a time-varying one, so that ramp survives alignment and shows up as
ATE.  ``r = log(||d_long|| / v_local)`` is invariant to a global Sim(3) applied
to either trajectory (numerator and denominator scale together) yet reports the
ramp directly.

★ v_local KEEPS ITS GRADIENT.  See ``v_local``.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import quat_to_R

#: design doc section 5.  Every rung must be >= S; 48 is the S=48 window itself.
DEFAULT_LADDER = (48, 96, 192)

TERMS = ("rot", "dir", "scale")


# ─────────────────────────────────────────────────────────────────────────────
# Pair indexing -- pure, no model, no tensors
# ─────────────────────────────────────────────────────────────────────────────

def long_pair_index(off: int, S: int, delta: int) -> Tuple[np.ndarray, np.ndarray]:
    """Which (window index, anchor offset) pairs a window at run-offset ``off`` supports.

    ``k`` indexes the supervised window (0..S-1) and carries gradient; ``a`` is
    the anchor's offset INSIDE THE SAME BANK RUN and is detached.  Staying inside
    one run is what makes the teacher pair meaningful: every run carries its own
    Sim(3) (label_bank.LabelBank.windows), so an anchor from a neighbouring run
    would be expressed in a different gauge and the "relative" target would be
    measuring the seam.

    On the 48-grid the coverage is all-or-nothing -- a window either supports the
    full S pairs (off >= delta) or none -- because the stream advances by S and
    both off and delta are multiples of it.  With L=240 that gives 4/5 of windows
    for delta=48, 3/5 for 96 and 1/5 for 192.
    """
    if delta < S:
        raise ValueError(
            f"delta={delta} < S={S}: the anchor would fall inside the supervised "
            f"window, picking up gradient and duplicating A1PC's pairs='all' "
            f"supervision (which already covers gaps 1..{S - 1}).")
    k = np.arange(max(0, delta - off), S, dtype=np.int64)
    return k, off + k - delta


def build_pairs(ladder: Sequence[int], hist: Dict[int, torch.Tensor],
                tea_poses: Callable[[int, int], torch.Tensor],
                run_t0: int, off: int, S: int, device,
                tea_win: Optional[torch.Tensor] = None) -> Dict[int, tuple]:
    """Assemble ``{delta: (stu_anchor, tea_anchor, k)}`` for one supervised window.

    ``hist`` maps ABSOLUTE frame index -> [9] student pose, filled by the rollout
    and cleared at every re-anchor (see trainer.RolloutPool.reset).  A delta whose
    anchors are not all present is skipped rather than partially filled: early in
    a rollout the history simply has not reached back that far yet.

    ``tea_poses(lo, hi)`` returns the run's teacher poses for offsets [lo, hi).

    ``tea_win`` is that source's poses for the supervised window itself.  Pass it
    whenever the rungs being built come from a DIFFERENT teacher than the local
    loss's bank (the Delta=319 track, or a K_t=2 bank): both endpoints of a pair
    have to sit in one gauge, and the window half of the pair comes from here.
    Leave it None for rungs served by the same run the local loss is scoring.
    """
    pairs = {}
    for d in ladder:
        k, a = long_pair_index(off, S, d)
        if len(k) == 0:
            continue
        frames = [int(run_t0 + x) for x in a]
        if any(f not in hist for f in frames):
            continue
        stu = torch.stack([hist[f] for f in frames]).to(device=device, dtype=torch.float32)
        # ★ ``tea_poses`` is a RANGE fetch, so a non-contiguous anchor set would
        # line the teacher up against the wrong frames without raising -- the
        # loss would still be finite and still go down.  long_pair_index only
        # ever returns a contiguous run; assert it so a future variant (the
        # stitched Delta=319 track indexes absolute frames, not run offsets)
        # cannot quietly break the correspondence.
        assert len(a) == 1 or (np.diff(a) == 1).all(), "anchor offsets not contiguous"
        tea = tea_poses(int(a[0]), int(a[-1]) + 1).float()
        assert tea.shape[0] == len(a), (tea.shape, len(a))
        pairs[d] = (stu, tea, torch.as_tensor(k, device=device, dtype=torch.long),
                    tea_win)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def rel_pair(pi: torch.Tensor, pj: torch.Tensor):
    """Anchor i, target j -> (Ri^T Rj, Ri^T (tj - ti)), both expressed in camera i.

    Same construction as losses._relative, but over two SEPARATE tensors: the
    anchor comes from the rollout history and the target from the live window.

    ``pose[:3]`` is the camera CENTRE in world coordinates and ``pose[3:7]`` the
    cam->world rotation.  (utils/pose_enc.py's docstring says "camera from world";
    the codebase does not use it that way -- experiments/mcd_eval.py compares
    pose_enc[:, :3] directly against Twc[:, :3, 3] and gets a sane ATE, and
    losses._relative's gauge invariance only holds under this reading.)
    """
    Ri = quat_to_R(pi[:, 3:7])
    Rj = quat_to_R(pj[:, 3:7])
    RiT = Ri.transpose(1, 2)
    dR = torch.einsum("nij,njk->nik", RiT, Rj)
    d = torch.einsum("nij,nj->ni", RiT, pj[:, :3] - pi[:, :3])
    return dR, d


def v_local(win: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """RMS consecutive step of the supervised window.  [S, 9] -> scalar.

    ★ THE INDEX SET IS THE CURRENT WINDOW, NOT THE LONG SPAN.  Taking the RMS
    over [i, i+Delta] would put a linear step-scale ramp into the numerator AND
    the denominator, where it partly cancels -- the ramp is the thing this term
    exists to see.

    ★ IT CARRIES GRADIENT BY DEFAULT, AND THE 1/sigma PROBLEM IS FIXED ELSEWHERE.
    ``r = log(||d_long|| / v_local)`` is gauge-invariant in VALUE -- measured
    0.02657 / 0.02654 / 0.02616 as the gauge unit went 2.0 -> 0.35 -> 0.05 for
    the same relative error.  Its GRADIENT is not: the dominant path is
    ``-1/v_local``, the SHORTEST length in the problem, so ||dL/dpose|| ran
    7.4e-3 -> 3.8e0 over a 1000x gauge sweep (509x spread).

    ★ DETACHING IS THE WRONG CURE, and this file said otherwise for one run.
    The claim was that the window's absolute scale "already has an owner" in
    L_motion-depth.  It does not: A1PC is invariant to a JOINT (pose, depth)
    scaling -- measured, total 0.515680 at k = 0.25, 1, 2 and 10, bit for bit.
    L_motion-depth pins the pose/depth RATIO, not the scale.  So with v_local
    detached the term reads "assume this window's local scale is correct and fit
    the long displacement to it", where the reference it trusts is a free
    variable nothing constrains.  That is not scale-propagation consistency.

    The cure is ``scale_gauge_norm``: keep the gradient on both sides of the
    ratio and multiply the term by a DETACHED v_local, which cancels the 1/sigma
    exactly.  Measured spread over the same 1000x sweep: 509x (grad, no norm)
    -> 160x (detached) -> 2.5x (grad + norm).  The loss VALUE then scales with
    the gauge, so it is no longer comparable across scenes as a scalar -- read
    ``long_r_resid_d*`` for that -- but the gradient, which is the thing the
    optimiser follows and the thing clipping renormalises, finally is.

    The rotation cancels inside the norm, so this is a plain difference of camera
    centres -- no einsum needed.
    """
    return (win[1:, :3] - win[:-1, :3]).norm(dim=-1).pow(2).mean().add(eps ** 2).sqrt()


def log_huber(r: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber on a LOG ratio, so ``delta`` is in relative-error units (0.1 = 10%)."""
    a = r.abs()
    return torch.where(a <= delta, 0.5 * r * r / delta, a - 0.5 * delta)


def angle_huber(theta: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber on the geodesic ANGLE, quadratic below ``delta`` radians.

    ★ WHY THE PLAIN ANGLE CANNOT BE USED HERE.  ``acos`` is an L1 on the angle:
    its gradient with respect to the pose has CONSTANT magnitude no matter how
    small the residual is (measured: 4.33 at 0.55 deg and the same 4.33 at 55
    deg, then a hard 0 once the clamp saturates below 0.081 deg).  That is fine
    on the correction branch, where the long rotation residual is real.  It is
    wrong on the IDENTITY branch, whose whole purpose is that the student still
    agrees with theta_0 -- the residual there sits at the numerical floor, and an
    L1 term answers a zero error with a full-size pull.  Observed directly: the
    identity step gradient norm went 20.6 (v6i, no long term) -> 197.0 (v7d, same
    scene, same step, same teacher depth), and since clipping renormalises the
    update, that means identity steps were being steered almost entirely by a
    residual with no information in it.

    Quadratic below ``delta`` makes the pull proportional to the error again, so
    a converged pair stops pulling, while anything past ``delta`` keeps exactly
    the old radians-scaled behaviour -- which is what ``lam_rot`` was calibrated
    against.  Same construction as ``log_huber``, one axis over.
    """
    if delta <= 0:
        return theta
    return torch.where(theta <= delta, 0.5 * theta * theta / delta,
                       theta - 0.5 * delta)


# ─────────────────────────────────────────────────────────────────────────────

class LongPoseLoss(nn.Module):
    """Sum over the Delta ladder of (rot, dir, depth-free scale).

    ``lam_rot`` starts at A1PC's value because L_rot is in RADIANS in both
    losses, so the two are directly comparable; the ladder weights start flat and
    are then set from the GRADIENT share (design doc section 10), not from the
    loss scalar -- clipping fires every step and renormalises any overall scale
    away.
    """

    def __init__(self, ladder: Sequence[int] = DEFAULT_LADDER,
                 lam_delta: Optional[Dict[int, float]] = None,
                 lam_rot: float = 15.0, lam_dir: float = 1.9, lam_scale: float = 1.0,
                 tau: float = 0.5, huber_delta: float = 0.1,
                 rot_huber_deg: float = 1.0,
                 detach_vlocal: bool = False,
                 scale_gauge_norm: bool = True,
                 weight_norm: bool = True,
                 terms: Sequence[str] = TERMS, eps: float = 1e-8,
                 prefix: str = "long"):
        super().__init__()
        self.ladder = tuple(int(d) for d in ladder)
        self.lam_delta = dict(lam_delta or {})
        self.lam_rot, self.lam_dir, self.lam_scale = lam_rot, lam_dir, lam_scale
        self.tau, self.huber_delta = tau, huber_delta
        #: radians below which L_rot becomes quadratic; see ``angle_huber``.
        self.rot_huber = float(rot_huber_deg) / 57.29578
        self.detach_vlocal = detach_vlocal
        self.scale_gauge_norm = scale_gauge_norm
        #: ★ WITHOUT THIS THE OBJECTIVE CHANGES WITH WINDOW POSITION.  A rung
        #: only fires where its anchor fits inside the run, so on the 48-grid a
        #: window at offset 0/48/96/144/192 carries lambda-sums of 0/1/3/3/7.
        #: Under a global clip that is not "7x the learning rate" -- it is a
        #: different local:long direction mix at every offset, which is not a
        #: clean long-supervision on/off.  Dividing by the sum of the lambdas
        #: that actually fired keeps the mix constant.
        self.weight_norm = weight_norm
        bad = [t for t in terms if t not in TERMS]
        if bad:
            raise ValueError(f"unknown long term(s) {bad}; expected from {TERMS}")
        self.terms = tuple(terms)
        self.eps = eps
        #: part-key namespace.  Two instances run side by side when the terms are
        #: split across teacher sources -- rot/dir from the in-run bank, scale
        #: from the stitched track -- and their bookkeeping keys must not collide.
        self.prefix = prefix

    def forward(self, stu_win: torch.Tensor, tea_win: torch.Tensor,
                pairs: Dict[int, tuple]):
        """``stu_win`` / ``tea_win``: [S, 9].  ``pairs`` from ``build_pairs``.

        ★ BOTH ENDPOINTS OF A PAIR MUST COME FROM ONE TEACHER GAUGE.  For the
        in-run rungs the anchor and the window are the same bank run, so the
        shared ``tea_win`` is that gauge.  A rung served by a DIFFERENT source --
        the stitched Delta=319 track -- must bring its own window poses, or the
        anchor would be in the stitched gauge and the target in the L240 run's,
        and the "relative" target would be measuring the difference between two
        teachers.  ``pairs[d]`` may therefore be a 4-tuple whose last element is
        that rung's own ``tea_win``.
        """
        S = stu_win.shape[0]
        tea_win = tea_win.detach()
        v_s = v_local(stu_win, self.eps)
        if self.detach_vlocal:
            v_s = v_s.detach()
        v_t_shared = v_local(tea_win, self.eps).detach()

        total = stu_win.new_zeros(())
        parts: Dict[str, float] = {}
        n_tot = 0
        lam_sum = 0.0
        for d in self.ladder:
            if d not in pairs:
                parts[f"{self.prefix}_n_d{d}"] = 0.0
                continue
            entry = pairs[d]
            stu_a, tea_a, k = entry[0], entry[1], entry[2]
            tw = entry[3] if len(entry) > 3 and entry[3] is not None else None
            if d < S:
                raise ValueError(f"delta={d} < S={S}")
            stu_a = stu_a.detach()               # the past is a constant
            tea_a = tea_a.detach()
            if tw is None:
                tw, v_t = tea_win, v_t_shared
            else:
                tw = tw.detach()
                v_t = v_local(tw, self.eps).detach()

            dRs, ds = rel_pair(stu_a, stu_win[k])
            dRt, dt = rel_pair(tea_a, tw[k])

            nt = dt.norm(dim=-1)
            keep = (nt / v_t) > self.tau         # design doc section 4, teacher-side mask
            n = int(keep.sum())
            parts[f"{self.prefix}_n_d{d}"] = float(n)
            parts[f"{self.prefix}_maskrate_d{d}"] = float(n) / max(1, keep.numel())
            if n == 0:
                continue
            n_tot += n
            lam_d = float(self.lam_delta.get(d, 1.0))
            sub = stu_win.new_zeros(())

            if "rot" in self.terms:
                dR = torch.einsum("nij,njk->nik", dRs.transpose(1, 2), dRt)[keep]
                cos = ((dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]) - 1) / 2
                th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
                L_rot = angle_huber(th, self.rot_huber).mean()
                sub = sub + self.lam_rot * L_rot
                # report the raw angle too: the Huber value is not a readable
                # rotation error once it goes quadratic
                parts[f"{self.prefix}_rot_rad_d{d}"] = float(th.mean().detach())
                parts[f"{self.prefix}_rot_huber_d{d}"] = float(L_rot.detach())
            if "dir" in self.terms:
                L_dir = (1 - F.cosine_similarity(ds[keep], dt[keep], dim=-1)).mean()
                sub = sub + self.lam_dir * L_dir
                parts[f"{self.prefix}_dir_d{d}"] = float(L_dir.detach())
            if "scale" in self.terms:
                r_s = (ds.norm(dim=-1).clamp(min=self.eps) / v_s).log()
                r_t = (nt.clamp(min=self.eps) / v_t).log()
                L_sc = log_huber((r_s - r_t)[keep], self.huber_delta).mean()
                if self.scale_gauge_norm:
                    # ★ THE STUDENT'S v_local, NOT THE TEACHER'S.  The 1/sigma in
                    # dL/dpose comes from 1/v_s, in the STUDENT's gauge; the two
                    # runs are in different gauges, so cancelling with v_t would
                    # leave the ratio v_t/v_s behind.
                    L_sc = L_sc * v_s.detach()
                sub = sub + self.lam_scale * L_sc
                parts[f"{self.prefix}_scale_d{d}"] = float(L_sc.detach())
                # residual in relative-error units, for the ATE correlation the
                # design doc asks for as supporting evidence
                parts[f"{self.prefix}_r_resid_d{d}"] = float((r_s - r_t)[keep].abs().mean().detach())

            total = total + lam_d * sub
            lam_sum += lam_d
            parts[f"{self.prefix}_d{d}"] = float(sub.detach())

        if self.weight_norm and lam_sum > 0:
            total = total / lam_sum
        parts[f"{self.prefix}_lam_sum"] = lam_sum
        parts[f"L_{self.prefix}"] = float(total.detach())
        parts[f"{self.prefix}_pairs"] = float(n_tot)
        return total, parts
