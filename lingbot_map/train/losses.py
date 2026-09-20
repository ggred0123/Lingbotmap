"""T3 -- the self-distillation loss.  Two terms, both gauge-invariant, no labels.

    L = lam_rel * L_rel_pose(fresh teacher) + lam_depth * L_depth_si(fresh teacher)

docs/self-distill-ver4-fixed.md §3.3 and docs/phase1-plan.md §3-T3.  Promoted
from the drafts in ``experiments/grad_probe.py`` with the two fixes the plan
called for, plus one correction to the plan itself.

WHAT THE GAUGE ACTUALLY IS -- this decides the whole design.  The student rolls
from its own anchor, the teacher from a different one, so the two runs differ by
a **Sim(3)**: a rotation, a translation, and one global scale.  Every loss term
must be blind to those and to nothing else.

    relative rotation        invariant already
    relative translation dir invariant already (expressed in the previous camera)
    relative translation mag scales with sigma  -> remove ONE scale
    depth                    scales with sigma  -> remove ONE scale

★ Scale only -- NOT scale and shift.  §3-T3 says to replace the magnitude term
with StreamVGGT's ``closed_form_scale_and_shift``.  Half of that is right and
half is not: the closed-form fit is the right idea, but a **shift is not a gauge
freedom here**.  Sim(3) scales depth; it does not add a constant to it.
StreamVGGT fits a shift because the DUSt3R lineage targets *affine*-invariant
depth (MiDaS-style disparity, defined up to scale AND shift).  Our teacher and
student are the same network in two gauges, so the shift is not free -- fitting
it would silently absorb real error.  Both are implemented; scale-only is the
default and ``closed_form_scale_and_shift`` is kept so the ablation is one flag.

──────────────────────────────────────────────────────────────────────────────
[한국어 개요]
이 파일은 T3 -- self-distillation loss를 정의한다. 라벨(GT) 없이, 학생(student)과
교사(teacher)의 출력을 직접 비교하는 두 개의 항으로 구성된다:

    L = lam_rel * L_rel_pose(fresh teacher) + lam_depth * L_depth_si(fresh teacher)

핵심 아이디어(왜 "게이지(gauge) 불변"이어야 하는가):
학생은 "자기 자신의" anchor에서부터 롤아웃하고, 교사는 "다른" anchor에서
롤아웃했기 때문에, 두 결과는 Sim(3) -- 회전 + 이동 + 전역 스케일(sigma) 하나 --
만큼 차이가 난다. 즉 학생과 교사의 절대 좌표계가 다르므로, 절대 위치/절대
깊이를 직접 비교하면 안 되고, 이 Sim(3) 차이에 "무감각한(invariant)" 양만
비교해야 한다.

    상대 회전(relative rotation)        이미 invariant (회전은 게이지에 안 걸림)
    상대 이동 방향(translation dir)      이미 invariant (이전 카메라 좌표계 기준 표현)
    상대 이동 크기(translation mag)      sigma만큼 스케일됨 -> 스케일 1개만 제거하면 됨
    깊이(depth)                          sigma만큼 스케일됨 -> 스케일 1개만 제거하면 됨

★ 핵심 설계 포인트: "스케일만" 제거해야지, "스케일+이동(shift)"까지 제거하면
안 된다. Sim(3)은 깊이에 스케일만 곱하지, 상수를 더하지 않기 때문이다.
(DUSt3R 계열이 shift까지 맞추는 이유는 그쪽이 다루는 깊이가 affine-invariant한
disparity라서 그런 것이고, 여기서는 교사와 학생이 "같은 네트워크"이므로 shift는
자유도가 아니다 -- shift를 허용하면 진짜 오차를 조용히 흡수해버릴 수 있다.)
두 방식 모두 구현은 해뒀고(ablation을 위해), 기본값은 scale-only이다.
──────────────────────────────────────────────────────────────────────────────
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import (
    DEFAULT_ALIGN_RES, ratio_trunc, robust_scale, subsample_for_align)

# One relative pair needs 2 frames; the magnitude term additionally fits a scale
# out of those pairs, so with very few pairs it fits the noise instead of the
# gauge.  Measured: at n_sup=1 the draft returned nan (empty-tensor mean) and the
# value was still swinging between n_sup 8 and 16 (grad_probe.json).
# [KR] 상대 포즈 관련 loss(rel_pose_loss, motion_depth_loss)를 계산하는 데
# 필요한 최소 프레임 수. 프레임이 너무 적으면(예: 1개) 상대쌍이 거의 없어서
# magnitude 항의 스케일 피팅이 노이즈에 맞춰지거나 심하면 NaN이 난다.
MIN_FRAMES_REL_POSE = 5


def quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """[..., 4] XYZW (scalar-last) -> [..., 3, 3].
    [KR] XYZW 순서(스칼라가 마지막)의 쿼터니언을 3x3 회전행렬로 변환하는
    표준 공식. 먼저 정규화(norm=1)한 뒤 표준 쿼터니언->회전행렬 변환식을 적용."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Closed-form gauge removal
# ─────────────────────────────────────────────────────────────────────────────

def closed_form_scale(pred: torch.Tensor, target: torch.Tensor,
                      weights: Optional[torch.Tensor] = None,
                      eps: float = 1e-12) -> torch.Tensor:
    """s minimising sum_i w_i (s*pred_i - target_i)^2, in closed form.

    One normal equation:  s = sum(w * pred * target) / sum(w * pred^2).

    This is the estimator that replaces the draft's median normalisation.  Both
    remove exactly one scale; the difference is variance.  The median of M
    samples is a jumpy order statistic at the M we can afford (a supervised
    window of 48 frames gives 47 pairs, and the draft was evaluated at 7 and 15),
    whereas this is a least-squares fit using every sample.

    [KR] pred를 target에 맞추는 최소제곱(least-squares) 스케일 s를 닫힌형
    (closed-form)으로 구한다: sum_i w_i (s*pred_i - target_i)^2 을 s에 대해
    미분해서 0으로 놓으면 정규방정식(normal equation) 하나가 나온다:
        s = sum(w * pred * target) / sum(w * pred^2)
    (증명: 2*sum(w*pred*(s*pred-target))=0 => s*sum(w*pred^2)=sum(w*pred*target))
    median 정규화 대신 이걸 쓰는 이유: 둘 다 스케일 자유도 1개를 정확히
    제거하지만, median은 표본 수가 적을 때(윈도우 48프레임 -> 상대쌍 47개)
    값이 크게 튀는 반면, 이 최소제곱 추정량은 모든 샘플을 다 써서 더 안정적.
    """
    if weights is None:
        num = (pred * target).sum()
        den = (pred * pred).sum()
    else:
        num = (weights * pred * target).sum()
        den = (weights * pred * pred).sum()
    return num / den.clamp(min=eps)


def closed_form_scale_and_shift(pred: torch.Tensor, target: torch.Tensor,
                                weights: Optional[torch.Tensor] = None,
                                eps: float = 1e-12
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(s, t) minimising sum_i w_i (s*pred_i + t - target_i)^2.

    The 2x2 system
        [ sum w p^2   sum w p ] [s]   [ sum w p g ]
        [ sum w p     sum w   ] [t] = [ sum w g   ]

    Provided for the ablation ONLY.  A shift is not a Sim(3) freedom (see the
    module docstring): enabling it lets the loss explain away a genuine constant
    depth bias.  ``experiments/loss_probe.py`` measures how big the fitted shift
    actually is, which is the empirical form of that argument.

    [KR] pred를 target에 맞추는 스케일 s와 이동(shift) t를 동시에 최소제곱으로
    구한다: sum_i w_i (s*pred_i + t - target_i)^2 최소화.
    s, t 각각에 대해 편미분해서 0으로 놓으면 2x2 연립방정식이 나오고(docstring의
    행렬식 참고), 이를 크래머 공식(Cramer's rule)으로 풀면 아래처럼 된다:
        det   = sum(w p^2)*sum(w) - sum(w p)^2
        scale = (sum(w p g)*sum(w) - sum(w p)*sum(w g)) / det
        shift = (sum(w p^2)*sum(w g) - sum(w p)*sum(w p g)) / det
    (코드의 s_wpg, s_ww, s_wp, s_wg, s_wpp 표기와 정확히 대응된다.)
    ablation 전용으로만 쓰는 이유: shift는 Sim(3) 자유도가 아니므로, shift를
    허용하면 loss가 "진짜 존재하는" 깊이 편향(bias) 오차를 shift로 흡수해서
    숨겨버릴 수 있다.
    """
    w = torch.ones_like(pred) if weights is None else weights
    s_ww = w.sum()
    s_wp = (w * pred).sum()
    s_wpp = (w * pred * pred).sum()
    s_wg = (w * target).sum()
    s_wpg = (w * pred * target).sum()
    det = (s_wpp * s_ww - s_wp * s_wp)
    det = torch.where(det.abs() < eps, torch.full_like(det, eps), det)
    scale = (s_wpg * s_ww - s_wp * s_wg) / det
    shift = (s_wpp * s_wg - s_wp * s_wpg) / det
    return scale, shift


# ─────────────────────────────────────────────────────────────────────────────
# Pair sets
# ─────────────────────────────────────────────────────────────────────────────
#
# Which (i, j) pairs the relative-pose terms are computed over.  The original
# loss used consecutive pairs only, i.e. (i, i+1).  That is a complete
# description of the trajectory in the noiseless case -- compose the steps and
# you recover every long-range relation -- but it is the wrong thing to
# SUPERVISE, because the errors compose too.  A per-step rotation error of e
# reaches the far end of a 48-frame window as roughly e*sqrt(47), and no
# consecutive-pair loss ever sees that accumulated quantity; it only ever sees e.
#
# Pi3's ``CameraLoss`` (pi3/models/loss.py:204) instead builds the full N x N
# table of relative poses and supervises all of it, which puts direct gradient on
# long-baseline relations.  That is worth borrowing here specifically because
# docs/add_loss.md §6 makes "미학습 원거리 구간" -- performance at distances the
# model was not trained on -- an acceptance criterion, and long-range drift is
# exactly what consecutive pairs are blind to.  It needs no labels: the teacher
# supplies both endpoints, same as before.
#
# ``min_gap`` exists because the short-gap pairs are the noisy ones (a nearly
# stationary camera gives a near-zero relative translation, whose DIRECTION is
# then dominated by noise), so raising it trades sample count for signal.
# [KR] ── 쌍(pair) 선택 방식 ──
# "consecutive"(연속쌍만, (i,i+1))는 원래 방식인데, 오차가 없다면 이걸로도
# 궤적 전체를 복원할 수 있지만(스텝들을 이어붙이면 되니까), "지도학습
# 대상"으로는 부적절하다 -- 오차도 함께 누적되기 때문. 스텝당 회전 오차 e는
# 48프레임 창 끝에서는 대략 e*sqrt(47)까지 누적되는데, 연속쌍 loss는 이
# 누적된 양을 절대 직접 보지 못하고 항상 e만 본다.
# 그래서 Pi3의 CameraLoss처럼 "all"(모든 순서쌍) 옵션을 추가해서 장거리
# (long-baseline) 관계에도 직접 gradient가 걸리도록 한다. 라벨이 따로 필요
# 없다 -- 교사가 양쪽 끝점을 다 제공하므로 이전과 마찬가지.
# min_gap: 간격이 짧은 쌍은 노이즈가 심하다(카메라가 거의 안 움직이면 상대
# 이동이 거의 0이 되고, 그 방향은 노이즈에 지배됨) -- min_gap을 올리면 샘플
# 수를 줄이는 대신 신호 대 잡음비를 높인다.

def _pair_index(N: int, pairs: str, min_gap: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index tensors (i, j) selecting which relative poses to compare.

    ``consecutive`` -- (0,1), (1,2), ... , N-1 pairs.  The historical default.
    ``all``         -- every ordered pair with ``|i-j| >= min_gap``, N*(N-1) of
                       them at ``min_gap=1``.  Both directions are kept, as in
                       Pi3: the relative rotation error is symmetric but the
                       translation DIRECTION is not, since (i,j) is expressed in
                       camera i and (j,i) in camera j.

    [KR] 어떤 프레임 쌍 (i,j)들에 대해 상대 포즈를 비교할지 정하는 인덱스를
    만든다. "all" 모드는 (i,j)와 (j,i)를 "둘 다" 유지하는데, 상대 회전 오차는
    대칭이지만(순서 안 가림), 상대 이동 "방향"은 비대칭이기 때문(=(i,j)는
    카메라 i 기준으로 표현되고, (j,i)는 카메라 j 기준으로 표현되어 서로 다른
    양이므로 둘 다 필요).
    """
    if pairs == "consecutive":
        i = torch.arange(N - 1, device=device)
        return i, i + 1
    if pairs == "all":
        g = torch.arange(N, device=device)
        gi, gj = torch.meshgrid(g, g, indexing="ij")
        keep = (gi - gj).abs() >= max(1, int(min_gap))
        return gi[keep], gj[keep]
    raise ValueError(f"unknown pairs {pairs!r}; expected 'consecutive' or 'all'")


def _relative(pose: torch.Tensor, i: torch.Tensor, j: torch.Tensor):
    """Relative rotation ``Ri^T Rj`` and translation ``Ri^T (tj - ti)``.

    The translation is expressed in camera i, which is what makes it a
    within-window quantity: a global Sim(3) leaves the direction untouched and
    multiplies the magnitude by sigma alone.

    [KR] 두 프레임 i, j 사이의 "상대" 회전 Ri^T Rj와 "상대" 이동
    Ri^T(tj - ti)를 계산한다. 이동을 "카메라 i 좌표계 기준"으로 표현하는 게
    핵심: 이렇게 해야 전역 Sim(3)(회전+이동+스케일)이 걸려도, 회전과 방향은
    전혀 안 변하고 크기(magnitude)만 sigma배 되는 성질을 갖게 된다(=게이지
    불변 성질을 만족하는 양이 됨).
    """
    R = quat_to_R(pose[:, 3:7])
    t = pose[:, :3]
    Ri_T = R[i].transpose(1, 2)
    dR = torch.einsum("nij,njk->nik", Ri_T, R[j])
    d = torch.einsum("nij,nj->ni", Ri_T, t[j] - t[i])
    return dR, d


# ─────────────────────────────────────────────────────────────────────────────
# L_rel-pose
# ─────────────────────────────────────────────────────────────────────────────

def rel_pose_loss(stu_pose: torch.Tensor, tea_pose: torch.Tensor,
                  mag_mode: str = "closed_form_scale",
                  pairs: str = "consecutive", min_gap: int = 1,
                  mag_trunc: float = 1.0,
                  eps: float = 1e-8):
    """Relative rotation + relative translation direction and magnitude.

    ``stu_pose`` / ``tea_pose``: [N, 9] pose encodings -- [:3] translation,
    [3:7] quaternion XYZW, [7:9] FoV.  Every term is a within-window quantity,
    so a Sim(3) difference between the two runs cancels; no alignment, no bridge.

    ``pairs`` selects the relative pairs (see ``_pair_index``).  The default
    ``"consecutive"`` reproduces the gate-6 loss exactly; ``"all"`` adds the
    long-baseline relations that consecutive pairs cannot express.

    Returns (L_rot [rad], L_dir, L_mag).

    [KR] 세 개의 하위 loss를 계산한다:
    - L_rot: 학생과 교사의 "상대 회전"끼리 얼마나 다른지(각도, 라디안).
      dR = dRs^T @ dRt (두 상대회전의 잔차 회전) -> trace로 각도 계산
      (cos(theta) = (trace(R)-1)/2는 회전행렬에서 회전각을 구하는 표준 공식).
    - L_dir: 상대 이동 "방향"의 코사인 유사도 오차 (1 - cos_similarity).
      방향은 이미 게이지 불변이므로 스케일 처리 없이 바로 비교.
    - L_mag: 상대 이동 "크기"의 오차. 크기는 sigma(전역 스케일)만큼 다르므로
      _magnitude_loss()에서 스케일 하나를 제거한 뒤 비교한다.
    """
    N = stu_pose.shape[0]
    if N < MIN_FRAMES_REL_POSE:
        raise ValueError(
            f"rel_pose_loss needs >= {MIN_FRAMES_REL_POSE} frames, got {N}. "
            f"With {max(N - 1, 0)} relative pairs the magnitude term fits its "
            f"scale out of almost nothing; at N=1 it is nan (empty-tensor mean).")

    i, j = _pair_index(N, pairs, min_gap, stu_pose.device)
    dRs, ds = _relative(stu_pose, i, j)
    dRt, dt = _relative(tea_pose, i, j)

    # residual rotation between the student's and the teacher's relative rotation
    dR = torch.einsum("nij,njk->nik", dRs.transpose(1, 2), dRt)
    cos = ((dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]) - 1) / 2
    L_rot = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()

    L_dir = (1 - F.cosine_similarity(ds, dt, dim=-1)).mean()

    ns = ds.norm(dim=-1).clamp(min=eps)
    nt = dt.norm(dim=-1).clamp(min=eps)
    L_mag = _magnitude_loss(ns, nt, mode=mag_mode, trunc=mag_trunc, eps=eps)
    return L_rot, L_dir, L_mag


def _magnitude_loss(ns: torch.Tensor, nt: torch.Tensor, mode: str,
                    trunc: float = 1.0, eps: float = 1e-8):
    """Scale-invariant discrepancy between two sequences of step magnitudes.

    ``median`` is the draft (docs/phase1-plan.md §3-T3 item 1 replaces it):
    normalise each sequence by its own median, then L1.

    ``closed_form_scale`` fits one scale by least squares and takes the L1
    residual, normalised by the fitted teacher magnitude so the result is
    dimensionless.  The normaliser is DETACHED: it is a unit conversion, not
    something the student should be able to shrink to lower the loss.

    ``l1`` / ``trunc_l1`` are the same construction with the L2 fit swapped for
    the exact weighted-L1 (resp. truncated-L1) fit from ``alignment.py``.  The
    fitted scale is the one quantity in this term that the gauge argument does
    NOT protect: least squares lets a few extreme pairs choose it, and under
    ``pairs="all"`` the magnitudes span gap 1 to gap S-1, so the longest
    baselines effectively set the gauge for everybody.  The L1 fit is the
    weighted median of the ratios instead, which the long tail cannot move.
    ``trunc`` is in units of the mean student step, so the cap travels with the
    gauge rather than fixing an absolute length.

    [KR] 두 시퀀스(학생/교사의 스텝 크기들)를 "스케일 불변" 방식으로
    비교하는 여러 방법들:
    - "median": 각자 자기 중앙값(median)으로 정규화한 뒤 L1 차이. (초기 방식)
    - "closed_form_scale": nt(교사)에 스케일 s를 곱해서 ns(학생)에 최소제곱으로
      맞춘 뒤, 그 잔차를 "피팅된 교사 크기(s*nt.mean())"로 나눠서 무차원화한다.
      이 정규화 분모는 detach() 처리 -- 단위 환산일 뿐 학생이 이걸 줄여서
      loss를 낮추게 하면 안 되기 때문.
    - "l1"/"trunc_l1": 최소제곱(L2) 대신 강건한(robust) L1 피팅을 쓴다.
      "all" 페어 모드에서는 이동 크기가 gap 1부터 gap S-1까지 스케일이
      크게 벌어지는데, L2 피팅은 소수의 극단값(제일 긴 baseline)이 스케일을
      좌지우지해버린다. L1 피팅(비율들의 가중 중앙값)은 그런 긴 꼬리에
      흔들리지 않는다.
    - "log": 로그 비율의 평균을 빼서(스케일 자유도 제거) 비교.
    """
    if mode == "median":
        return ((ns / ns.median().clamp(min=eps))
                - (nt / nt.median().clamp(min=eps))).abs().mean()
    if mode == "closed_form_scale":
        # teacher is the reference; find s with s * nt ~ ns
        s = closed_form_scale(nt, ns)
        denom = (s * nt.mean()).detach().clamp(min=eps)
        return ((s * nt - ns).abs().mean()) / denom
    if mode in ("l1", "trunc_l1"):
        # ratio_weighted (the default) is what keeps this a median of the step
        # RATIOS.  Without it the vote is proportional to the teacher step
        # length, so under pairs="all" the gap-(S-1) pairs would set the gauge
        # for the gap-1 pairs -- the exact failure this mode exists to avoid.
        # ratio_trunc, not trunc*ns.mean(): align caps the WEIGHTED residual,
        # which ratio_weights has already put in ratio units.
        cap = ratio_trunc(nt, ns, rel=trunc) if mode == "trunc_l1" else None
        s = robust_scale(nt, ns, trunc=cap)
        denom = (s * nt.mean()).detach().clamp(min=eps)
        return ((s * nt - ns).abs().mean()) / denom
    if mode == "log":
        r = ns.log() - nt.log()
        return (r - r.mean()).abs().mean()
    raise ValueError(f"unknown mag_mode {mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# L_depth-SI
# ─────────────────────────────────────────────────────────────────────────────

# [KR] 깊이(depth) loss의 "모드"는 (게이지 추정 방법) x (잔차 계산 방식) 두
# 축의 조합이다. 처음 측정할 때 이 두 축이 뒤섞여 있어서 결과 해석이
# 어려웠기 때문에, 여기서는 두 축을 분리해서 이름을 붙였다.
# ``mode`` is (gauge estimator) x (residual).  Both axes matter and they were
# confounded in the first measurement, so they are named separately here.
#   norm  : "median" -- per-frame median, one scale, robust order statistic
#           "cfs"    -- closed-form least-squares scale
#           "cfss"   -- closed-form scale AND shift (ablation; see module docstring)
#           "l1"     -- exact weighted-L1 scale (alignment.py, from Pi3)
#           "trunc"  -- truncated-L1 scale: residuals past the cap stop counting
#   resid : "log"    -- |log s - log t|, i.e. relative error, equal weight per pixel
#           "lin"    -- |s - t| / mean(t), dominated by the far field
_DEPTH_MODES = {
    "median": ("median", "log"),                       # default
    "median_linear": ("median", "lin"),
    "closed_form_scale": ("cfs", "lin"),
    "closed_form_scale_log": ("cfs", "log"),
    "closed_form_scale_shift": ("cfss", "lin"),        # shift can go negative -> no log
    # Robust gauge estimators ported from Pi3 (pi3/utils/alignment.py).  The
    # per-frame scale is fitted over ~200k pixels of which the largest residuals
    # are sky, thin structure and depth-discontinuity halos -- i.e. least squares
    # hands the gauge to precisely the pixels the teacher is least sure about.
    # "l1" is the weighted median of the ratios and ignores them; "trunc" goes
    # further and caps each residual, so a badly wrong pixel stops contributing
    # at all rather than contributing a large constant.
    "l1": ("l1", "log"),
    "l1_linear": ("l1", "lin"),
    "trunc_l1": ("trunc", "log"),
    "trunc_l1_linear": ("trunc", "lin"),

    # ★ ONE SCALE FOR THE WHOLE WINDOW.  Every estimator above fits S scales --
    # one per frame -- but only ONE of them is a gauge freedom.  A run has
    # exactly one unit, fixed at the anchor block and shared by both heads; that
    # is the premise ``motion_depth_loss`` is built on and the same premise that
    # makes fitting a shift illegitimate (module docstring).  Fitting 48 scales
    # where 1 is free gives away 47 real degrees of freedom, and the loss goes
    # BLIND to what they hide: with D_i^s = sigma * k_i * D_i^t the per-frame
    # median cancels k_i exactly, so L_depth is 0 for any per-frame drift k_i.
    #
    # ``L_motion-depth`` catches that -- but only while the pose does NOT drift
    # the same way, since it measures pose-vs-depth MISMATCH by construction.
    # When k_i == m_i (the reconstruction "breathing" in scale over the window)
    # both terms read 0 and only ``L_mag`` would see it, which every A1 preset
    # switches off.  This mode closes that hole: the scale is global, so any
    # within-window drift lands in the residual.
    #
    # The estimator is the conf-weighted median of the ratios (``robust_scale``
    # on the pooled window), which is EXACTLY the L1-optimal offset for the log
    # residual -- min_a sum c_i |log a - log r_i| is solved by
    # a = weighted_median(r_i), and log is monotonic so the ratio-median and the
    # log-ratio-median coincide.  It also feeds ``conf`` into the FIT, which the
    # "median" default does not (it medians the raw depth, unweighted).
    "global": ("global", "log"),
    "global_linear": ("global", "lin"),
}

#: Public, ordered view of the depth modes -- for argparse ``choices`` and docs,
#: so a caller does not have to reach into the private table to enumerate them.
DEPTH_MODES = tuple(sorted(_DEPTH_MODES))


def depth_si_loss(stu_depth: torch.Tensor, tea_depth: torch.Tensor,
                  conf: Optional[torch.Tensor] = None,
                  mode: str = "median",
                  trunc: float = 0.1,
                  align_res: int = DEFAULT_ALIGN_RES,
                  eps: float = 1e-6):
    """Per-frame scale-invariant depth L1, optionally teacher-confidence weighted.

    ``stu_depth`` / ``tea_depth``: [S, H, W, 1] or [S, H, W].
    ``conf``: teacher Sigma^D, [S, H, W].  DETACHED -- v4 §3.3-2 weights by the
    teacher's confidence and deliberately does not train the student's own conf
    head on pseudo-labels.

    Default is per-frame median normalisation + log residual.  §3-T3's "replace
    the median" applies to the POSE magnitude term, not here: the pathology there
    is small-sample variance over 7-15 relative pairs, whereas this median is
    taken over ~200k pixels.  ``experiments/loss_probe.py`` measures both axes --
    swapping either one degrades the FAR/NEAR ordering this loss exists to
    produce, so the default is the measured choice, not the inherited one.

    ``trunc`` applies to the ``trunc_l1*`` modes only.  It is a RELATIVE
    tolerance on the depth ratio (see ``alignment.ratio_trunc``): a pixel whose
    student/teacher ratio misses the fitted scale by more than ``trunc`` times
    the typical ratio stops contributing.  Being a ratio, it is gauge-free.
    Those modes also subsample to ``align_res`` points before fitting, exactly as
    Pi3 does (``PointLoss.local_align_res = 4096``): the truncated solver
    enumerates every local extremum of a non-convex objective, which is sensible
    at a few thousand samples and wasteful at 200k.  The FIT is on the subsample;
    the residual is still evaluated on every pixel.

    ``global`` / ``global_linear`` fit ONE scale for the whole window instead of
    one per frame.  Every other mode quotients away S scales where only one is a
    Sim(3) freedom, which makes L_depth exactly 0 under per-frame scale drift;
    these two put that drift back into the residual.  Per-frame contribution
    imbalance -- the practical reason per-frame normalisation is attractive --
    is then a separate axis: the ``lin`` residual still divides by that frame's
    own mean, so the two concerns stop being conflated.  See ``_DEPTH_MODES``.

    [KR] 프레임마다(per-frame) 깊이맵의 스케일 불변 L1 오차를 계산한다.
    - "median" 정규화: 프레임별로 자기 깊이맵의 중앙값으로 나눠서 스케일을
      제거한다. 여기서는 median을 쓰는 게 문제 없다(§3-T3가 median을
      바꾸라고 한 건 상대포즈 magnitude 항 얘기 -- 거기는 표본이 7~15개뿐이라
      median이 불안정했지만, 여기 깊이맵의 median은 픽셀 ~20만 개 위에서
      계산하므로 훨씬 안정적).
    - "l1"/"trunc" 정규화: 최소제곱 대신 강건한(robust) 스케일 추정을 쓴다
      (Pi3에서 가져온 방식). 최소제곱으로 스케일을 피팅하면 하늘/얇은 구조물/
      깊이 불연속 경계 같이 "교사도 확신 없는" 픽셀들이 큰 잔차값으로
      스케일을 좌우해버리는데, robust 추정은 그런 극단값에 덜 흔들린다.
    - conf(교사의 신뢰도)가 있으면 가중치로 쓰되 항상 detach -- 학생 자신의
      confidence head를 pseudo-label로 학습시키지 않기 위함(v4 §3.3-2).
    - "global"/"global_linear": 프레임별이 아니라 "창 전체에 스케일 1개"만
      맞춘다. 위의 다른 모드들은 스케일을 S개(프레임 수만큼) 제거하는데,
      실제 게이지 자유도는 1개뿐이다 -- 그래서 창 안에서 깊이 스케일이
      프레임마다 출렁여도(D_i^s = sigma*k_i*D_i^t) 프레임별 median이 k_i를
      정확히 상쇄해버려 L_depth가 0이 된다. 이 모드는 그 드리프트를 잔차에
      남긴다. 프레임별 기여도 불균형(프레임별 정규화의 실질적 장점)은
      "lin" 잔차가 여전히 프레임별 평균으로 나누는 것으로 따로 처리되므로,
      두 관심사가 분리된다.
    """
    if mode not in _DEPTH_MODES:
        raise ValueError(f"unknown depth mode {mode!r}; expected one of "
                         f"{sorted(_DEPTH_MODES)}")
    norm, resid = _DEPTH_MODES[mode]

    s = stu_depth[..., 0] if stu_depth.dim() == 4 else stu_depth
    t = tea_depth[..., 0] if tea_depth.dim() == 4 else tea_depth
    S = s.shape[0]
    s = s.reshape(S, -1).clamp(min=1e-3)
    t = t.reshape(S, -1).clamp(min=1e-3)
    w = conf.reshape(S, -1).detach() if conf is not None else None

    if norm == "median":
        s_fit = s / s.median(dim=1, keepdim=True).values.clamp(min=eps)
        t_fit = t / t.median(dim=1, keepdim=True).values.clamp(min=eps)
    elif norm == "global":
        # ONE scale over the pooled window -- see the note in _DEPTH_MODES.
        #
        # Subsample PER FRAME first, then pool: ``subsample_for_align`` thins the
        # last dim, so each frame contributes exactly ``align_res`` samples and
        # no frame can dominate the fit by having more valid pixels.  S*4096
        # samples for a single scalar is already far past the precision a median
        # needs, and it keeps the sort off the full S*200k window.  The stride is
        # deterministic, so the same pixels vote at every step -- the fitted
        # scale never jitters for sampling reasons alone.
        ss, tt, ww = subsample_for_align(s, t, w, n=align_res)
        a = robust_scale(ss.reshape(-1), tt.reshape(-1),
                         None if ww is None else ww.reshape(-1))
        # ``a`` is a 0-dim tensor and broadcasts over [S, N]: every frame is
        # rescaled by the SAME factor, which is the whole point -- a per-frame
        # drift now survives into the residual instead of being fitted away.
        s_fit, t_fit = a * s, t
    elif norm in ("l1", "trunc"):
        # Batched over frames -- align() carries leading dims, so all S gauges
        # are solved in one call instead of the Python loop the L2 path needs.
        if norm == "trunc":
            ss, tt, ww = subsample_for_align(s, t, w, n=align_res)
            cap = ratio_trunc(ss, tt, ww, rel=trunc)
            a = robust_scale(ss, tt, ww, trunc=cap)
        else:
            a = robust_scale(s, t, w)
        # a is fitted on the subsample (trunc) or on every pixel (l1); either way
        # the residual below is evaluated on every pixel.
        s_fit, t_fit = a[:, None] * s, t
    else:
        fits = []
        for i in range(S):                      # per frame: its own gauge
            wi = None if w is None else w[i]
            if norm == "cfs":
                a = closed_form_scale(s[i], t[i], wi)
                fits.append(a * s[i])
            else:
                a, b = closed_form_scale_and_shift(s[i], t[i], wi)
                fits.append(a * s[i] + b)
        s_fit, t_fit = torch.stack(fits, dim=0), t

    if resid == "log":
        err = (s_fit.clamp(min=eps).log() - t_fit.clamp(min=eps).log()).abs()
    else:
        err = (s_fit - t_fit).abs() / t_fit.mean(dim=1, keepdim=True).detach().clamp(min=eps)

    if w is not None:
        return (err * w).sum() / w.sum().clamp(min=eps)
    return err.mean()


# ─────────────────────────────────────────────────────────────────────────────
# L_motion-depth
# ─────────────────────────────────────────────────────────────────────────────

def motion_depth_loss(stu_pose: torch.Tensor, tea_pose: torch.Tensor,
                      stu_depth: torch.Tensor, tea_depth: torch.Tensor,
                      pairs: str = "consecutive", min_gap: int = 1,
                      eps: float = 1e-8):
    """Motion per unit scene depth -- the term that ties the two heads together.

    ``L_mag`` fits one scale out of the step magnitudes and discards it, because
    under a Sim(3) gauge the absolute magnitude is meaningless.  That is right in
    isolation and wrong in context: a run has exactly ONE unit, fixed at the
    anchor block and shared by both heads (the pose head reads ``tokens[:,:,0]``
    and the depth head reads the patch tokens of the SAME aggregator, all of
    which attend to the same anchor).  So the pose scale is not free once the
    depth scale is chosen, and their ratio is a gauge-invariant error measure.

    Measured on the released checkpoint (S=48, K=28):

        t0=80    sigma_pose 1.22x  sigma_depth 1.06x  ->  mismatch  1.15x
        t0=5248  sigma_pose 22.2x  sigma_depth 1.65x  ->  mismatch 13.52x

    -- an internal inconsistency no Sim(3) can explain, and precisely the
    quantity ``L_mag`` quotients away by construction (its FAR/NEAR is 1.48x,
    the weakest of the four terms).

    Dividing each step by that frame's median depth cancels sigma per frame, so
    this stays gauge-free while keeping 47 samples per window instead of one
    scalar:

        u_i = log ||dt_i|| - log median(D_i)      L = mean_i |u_i^s - u_i^t|

    Measured FAR/NEAR 5.02x with a 1.57x swing over the window length: same raw
    material as ``L_mag``, 3.4x the discrimination, purely because the scale is
    pinned to depth instead of fitted away.  The |mean| variant (sigma_pose vs
    sigma_depth as one scalar) scores 18.6x but swings 4.49x over the window
    length -- the same small-sample pathology that killed the draft ``L_mag``,
    so the dense form is the one that ships.

    ``stu_depth`` / ``tea_depth``: [S, H, W, 1] or [S, H, W], aligned with the
    poses.  Pair (i, j) is expressed in camera i, so frame i's depth is the one
    that sets the reference -- which is why ``pairs="all"`` works here unchanged:
    every pair still has a well-defined reference frame, and a long-baseline pair
    measures accumulated pose-vs-depth scale mismatch rather than per-step
    mismatch.

    [KR] "포즈 헤드"와 "깊이 헤드"가 사실은 하나의 단위(unit)를 공유한다는
    사실을 이용하는 항. 한 롤아웃에는 오직 "하나의" 단위만 있고, 이건
    anchor 블록에서 고정되며 두 헤드가 같은 aggregator를 통해 이 anchor를
    함께 바라본다. 그래서 "포즈 스케일"과 "깊이 스케일"은 독립적이지 않고,
    둘의 "비율"은 게이지에 안 걸리는 불변량이 된다.
    - L_mag(상대이동 크기 항)는 스케일을 최소제곱으로 "피팅해서 없애버리는"
      방식인데, 그러다 보니 정작 "포즈 스케일과 깊이 스케일이 서로 안 맞는"
      내부 불일치(internal inconsistency) 신호까지 같이 없애버린다(측정상
      FAR/NEAR 구분력이 4항 중 제일 약함, 1.48x).
    - 이 항은 각 스텝의 이동량을 "그 프레임의 median 깊이"로 나눠서 로그를
      취한다: u_i = log||dt_i|| - log(median(D_i)). 이러면 프레임별 sigma가
      상쇄되면서도(게이지 불변 유지), 창 하나당 47개 샘플을 그대로 살릴 수
      있다(L_mag처럼 스칼라 하나로 뭉개지 않음). 측정상 FAR/NEAR 5.02x로
      L_mag보다 3.4배 더 좋은 구분력을 보였다.
    """
    N = stu_pose.shape[0]
    if N < MIN_FRAMES_REL_POSE:
        raise ValueError(
            f"motion_depth_loss needs >= {MIN_FRAMES_REL_POSE} frames, got {N}.")

    i, j = _pair_index(N, pairs, min_gap, stu_pose.device)
    _, ds = _relative(stu_pose, i, j)
    _, dt = _relative(tea_pose, i, j)

    s = stu_depth[..., 0] if stu_depth.dim() == 4 else stu_depth
    t = tea_depth[..., 0] if tea_depth.dim() == 4 else tea_depth
    S = s.shape[0]
    med_s = s.reshape(S, -1).clamp(min=1e-3).median(dim=1).values
    med_t = t.reshape(S, -1).clamp(min=1e-3).median(dim=1).values

    u_s = ds.norm(dim=-1).clamp(min=eps).log() - med_s[i].log()
    u_t = dt.norm(dim=-1).clamp(min=eps).log() - med_t[i].log()
    return (u_s - u_t).abs().mean()


# ─────────────────────────────────────────────────────────────────────────────

#: Reproducible configurations.  ``B0``/``B1`` exist so the pre-``L_motion-depth``
#: results stay one flag away; ``A1`` is docs/add_loss.md §5's first new cell.
# [KR] 재현 가능한 손실 설정(preset) 모음. 각 항의 가중치 lam_*와 옵션들을
# 미리 정해둔 조합. B0/B1은 v5 이전(gate 6/6b) 결과를 그대로 재현하기 위한
# 것이고, A1부터는 새로운 실험 셀(cell)들이다.
PRESETS = {
    # the four-term loss as shipped through gate 6
    "B0": dict(lam_rot=1.0, lam_dir=1.0, lam_mag=0.5, lam_motion=0.0, lam_depth=1.0),
    # gate 6b: lam_rot raised to degrees-equivalent, lam_mag suppressed
    "B1": dict(lam_rot=30.0, lam_dir=1.0, lam_mag=0.1, lam_motion=0.0, lam_depth=1.0),
    # A1: L_mag replaced by L_motion-depth, everything else as B1
    "A1": dict(lam_rot=30.0, lam_dir=1.0, lam_mag=0.0, lam_motion=0.5, lam_depth=1.0),

    # ---- the two things borrowed from Pi3, each isolated so §5 can attribute ----
    # A1-L: A1 with the depth gauge fitted by weighted L1 instead of the median.
    #       Changes the ESTIMATOR only; the term, the weights and the residual
    #       are identical to A1.
    "A1L": dict(lam_rot=30.0, lam_dir=1.0, lam_mag=0.0, lam_motion=0.5, lam_depth=1.0,
                depth_mode="l1"),
    # A1-P: A1 with every ordered relative pair instead of consecutive ones
    #       (Pi3 CameraLoss).  Changes the SUPPORT only; every weight is as A1.
    #       min_gap=1 keeps the consecutive pairs in the set, so this is a strict
    #       superset of A1's supervision.
    "A1P": dict(lam_rot=30.0, lam_dir=1.0, lam_mag=0.0, lam_motion=0.5, lam_depth=1.0,
                pairs="all", min_gap=1),
    # A1-LP: both.  These compose for a reason -- all-pairs is what makes the
    #        magnitudes span two orders of magnitude, which is what makes the L2
    #        fit indefensible, which is what the L1 fit is for.
    "A1LP": dict(lam_rot=30.0, lam_dir=1.0, lam_mag=0.0, lam_motion=0.5, lam_depth=1.0,
                 depth_mode="l1", pairs="all", min_gap=1),
    # A1-PC: A1P with lambda re-fitted to the all-pairs SUPPORT.
    #        A1P is a strict superset of A1's pairs, but not a neutral one: the
    #        long-baseline pairs it adds grow L_rot (0.0244 -> 0.0386 at FAR) and
    #        DILUTE L_dir / L_motion, whose per-pair residuals shrink faster than
    #        the pair count grows (0.948 -> 0.426, 2.849 -> 1.140).  Measured on
    #        global_blocks (experiments/results/tg_A1_consec.json vs tg_A1P_all.json),
    #        carrying A1's weights over to all-pairs moves L_rot's FAR gradient
    #        share 33.7% -> 62.9% -- it takes roughly double its intended share
    #        purely because the support changed.  These weights put the FAR shares
    #        back in FAR/NEAR-discrimination order (rot 7.52x > motion 5.02x >
    #        dir 3.15x > depth 2.36x): 32 / 28 / 28 / 12.
    #
    #        Only the RATIOS here are load-bearing.  Training clips at |g|=1 and
    #        the measured pre-clip norm is 7-47, so clipping fires every step and
    #        renormalises any overall scale away.
    "A1PC": dict(lam_rot=15.0, lam_dir=1.9, lam_mag=0.0, lam_motion=0.9, lam_depth=1.7,
                 pairs="all", min_gap=1),
    # X-mag: L_mag revived with a robust fit, to re-test docs/add_loss.md §3's
    #        "scale fitting removes the drift signal" against an estimator the
    #        outliers cannot steer.  Ablation only.
    "X1M": dict(lam_rot=30.0, lam_dir=1.0, lam_mag=0.5, lam_motion=0.5, lam_depth=1.0,
                mag_mode="l1", depth_mode="l1", pairs="all", min_gap=1),
}


class SelfDistillLoss(nn.Module):
    """The shared geometric distance ``D(student, teacher)`` of docs/add_loss.md §2.

    The trainer applies this SAME module to both branches::

        L_total = D(S_long, T_fresh_ema) + lam_fresh * D(S_fresh, T_frozen_bank)

    so the weights here describe one branch and are shared by construction --
    that is the point of factoring it out, since the two branches must measure
    the same geometry for their difference to mean anything.

    Teacher tensors are treated as constants.  ``L_rot`` is in RADIANS, so
    ``lam_rot=1`` gives it ~2% of the total despite the best FAR/NEAR
    discrimination (7.52x); ~57 converts it to degrees and gate 6b measured 30 to
    be enough (probe-1 rotation went from +86% to -65%).

    The bare defaults reproduce the pre-``L_motion-depth`` loss bit-for-bit
    (``PRESETS["B0"]``); pass a preset or explicit weights for anything else.

    [KR] 학생과 교사 사이의 "기하학적 거리" D(student, teacher)를 계산하는
    공유 모듈. trainer.py가 이 "같은" 모듈을 fresh 브랜치와 correction(long)
    브랜치 둘 다에 적용한다 -- 두 브랜치의 차이가 의미를 가지려면 반드시
    "같은 기하학적 잣대"로 재야 하기 때문에, 이렇게 하나로 빼둔 것이 핵심.
    forward()에서 4개 하위 항(L_rot, L_dir, L_mag, L_dep)과 옵션인 L_mot을
    각자의 가중치(lam)로 합산해서 최종 loss를 만든다.
    """

    def __init__(self, lam_rot: float = 1.0, lam_dir: float = 1.0,
                 lam_mag: float = 0.5, lam_depth: float = 1.0,
                 lam_motion: float = 0.0,
                 mag_mode: str = "closed_form_scale",
                 depth_mode: str = "median",
                 pairs: str = "consecutive", min_gap: int = 1,
                 mag_trunc: float = 1.0, depth_trunc: float = 0.1,
                 align_res: int = DEFAULT_ALIGN_RES,
                 use_teacher_conf: bool = True):
        super().__init__()
        # Refuse a bad mode HERE, not at the first forward.  ``depth_si_loss``
        # validates too, but by then the trainer has built the model and rolled
        # the pool in -- minutes of setup thrown away to report a typo.
        if depth_mode not in _DEPTH_MODES:
            raise ValueError(f"unknown depth_mode {depth_mode!r}; expected one "
                             f"of {list(DEPTH_MODES)}")
        self.lam = (lam_rot, lam_dir, lam_mag, lam_depth, lam_motion)
        self.mag_mode = mag_mode
        self.depth_mode = depth_mode
        # ``pairs`` is shared by L_rot / L_dir / L_mag / L_motion-depth on
        # purpose: they are four readouts of ONE pair set, and letting them
        # disagree would make their relative gradient shares (which §5 reports)
        # incomparable across the ablation.
        self.pairs = pairs
        self.min_gap = min_gap
        self.mag_trunc = mag_trunc
        self.depth_trunc = depth_trunc
        self.align_res = align_res
        self.use_teacher_conf = use_teacher_conf

    @classmethod
    def from_preset(cls, name: str, **overrides):
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})

    def forward(self, stu_pose, tea_pose, stu_depth, tea_depth, tea_conf=None):
        # [KR] 교사 텐서는 전부 detach -- 상수(constant)로 취급, gradient가
        # 교사 쪽으로는 절대 흐르지 않는다. lmo(motion 가중치)가 0이면
        # motion_depth_loss 자체를 호출하지 않아서(0*x 형태로 그래프에
        # 남기지 않음) 꺼진 항은 비용이 전혀 안 든다.
        tea_pose = tea_pose.detach()
        tea_depth = tea_depth.detach()
        conf = tea_conf.detach() if (tea_conf is not None and self.use_teacher_conf) else None
        lr, ld, lm, ldep, lmo = self.lam

        L_rot, L_dir, L_mag = rel_pose_loss(
            stu_pose, tea_pose, mag_mode=self.mag_mode,
            pairs=self.pairs, min_gap=self.min_gap, mag_trunc=self.mag_trunc)
        # Same rule as L_motion below: at lam 0 the term is not built at all.
        # --abs_mode paper drops the median-depth term (the paper's Eq.1 has
        # none), and a full-resolution per-frame median over 9.4M pixels is not
        # something to pay for a 0 * x.  Every shipped preset has lam_depth > 0,
        # so nothing that ran before this line is changed by it.
        L_dep = (depth_si_loss(stu_depth, tea_depth, conf, mode=self.depth_mode,
                               trunc=self.depth_trunc, align_res=self.align_res)
                 if ldep != 0.0 else stu_pose.new_zeros(()))
        # Skipped entirely at lam=0 so the disabled term costs nothing and, more
        # importantly, cannot contribute gradient through a 0 * x product.
        L_mot = (motion_depth_loss(stu_pose, tea_pose, stu_depth, tea_depth,
                                   pairs=self.pairs, min_gap=self.min_gap)
                 if lmo != 0.0 else stu_pose.new_zeros(()))

        total = lr * L_rot + ld * L_dir + lm * L_mag + ldep * L_dep + lmo * L_mot
        return total, {
            "loss": float(total.detach()),
            "L_rot_rad": float(L_rot.detach()),
            "L_rot_deg": float(L_rot.detach()) * 57.29578,
            "L_dir": float(L_dir.detach()),
            "L_mag": float(L_mag.detach()),
            "L_depth_si": float(L_dep.detach()),
            "L_motion_depth": float(L_mot.detach()),
        }
