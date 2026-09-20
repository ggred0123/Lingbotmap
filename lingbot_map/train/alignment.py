"""Robust closed-form scale fitting -- ported from Pi3's ``pi3/utils/alignment.py``.

WHY THIS EXISTS.  ``losses.py`` removes the Sim(3) scale gauge by fitting one
scale and quotienting it out.  Until now that fit was **least squares**
(``closed_form_scale``), which is the maximum-likelihood estimator for Gaussian
residuals and the worst possible one for the residuals we actually have.  Two
places where that bites:

  ``_magnitude_loss``   fits one scale over the relative-step magnitudes.  Under
                        ``pairs="all"`` those magnitudes span gap 1 to gap S-1,
                        i.e. two orders of magnitude, and an L2 fit is decided
                        almost entirely by the longest baselines.
  ``depth_si_loss``     fits one scale per frame over ~200k pixels.  Sky, thin
                        structure and depth-discontinuity halos are exactly the
                        pixels with the largest residual, so L2 hands them the
                        gauge.

docs/add_loss.md §3 lists ``L_mag`` as the default-off candidate because "scale
fitting removes valid drift signal".  That is a statement about the ESTIMATOR,
not about the term: an L2 fit absorbs whatever a handful of outliers ask it to.
This module supplies the robust replacements so that claim can be re-measured
rather than assumed.

WHAT IS PORTED.  ``align`` solves, exactly and in closed form,

    trunc is None:   min_a  sum_i w_i |a x_i - y_i|
    trunc is set:    min_a  sum_i min(trunc, w_i |a x_i - y_i|)

The first is the weighted median of the ratios ``y_i/x_i``, found by sorting the
ratios and walking the piecewise-constant derivative ``2*cumsum(w*x) - sum(w*x)``
to its sign change.  The second is non-convex -- truncation makes far-out
outliers cost a constant instead of growing -- so it enumerates every local
extremum (the candidate optima of a piecewise-linear objective are the breakpoints
``y/x``, ``(wy±trunc)/(wx)``) and takes the best.  Both return ``a = y[k]/x[k]``
for the selected index ``k``, so the result is differentiable through that one
pair, which is the correct subgradient of an L1 objective.

Pi3 uses these for GT-supervised point alignment (``align_points_scale`` inside
``PointLoss``).  Nothing about the solver is GT-specific -- it aligns two tensors
-- so it transfers to the unlabelled student/teacher setting unchanged.  Only the
scale-fitting core is taken; the affine (scale+shift) variants are deliberately
left behind, because a shift is not a Sim(3) freedom here (see the ``losses.py``
module docstring).

COST.  The L1 branch is one sort: O(n log n), fine on the full 200k-pixel frame.
The truncated branch materialises an ``(extrema, n)`` residual table and is only
sensible after subsampling -- Pi3 subsamples to 4096 points before calling it
(``PointLoss.local_align_res``), and ``depth_si_loss`` does the same.
"""

import math
from typing import Optional, Tuple, Union

import torch

#: Pi3's ``PointLoss.local_align_res``.  The truncated solver is quadratic-ish in
#: disguise; anything above a few thousand samples is wasted precision anyway.
DEFAULT_ALIGN_RES = 4096


def scatter_min(size: int, dim: int, index: torch.LongTensor,
                src: torch.Tensor) -> torch.return_types.min:
    """Segment-min: minimum of ``src`` grouped by ``index``, plus the argmin."""
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device
                         ).scatter_reduce(dim=dim, index=index, src=src,
                                          reduce='amin', include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1:])] = minimum_where[dim]
    return torch.return_types.min((minimum, indices))


def _pad_inf(x_: torch.Tensor):
    return torch.cat([torch.full_like(x_[..., :1], -torch.inf), x_,
                      torch.full_like(x_[..., :1], torch.inf)], dim=-1)


def _pad_cumsum(cumsum: torch.Tensor):
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)


def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    return a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)


def align(x: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None,
          trunc: Optional[Union[float, torch.Tensor]] = None,
          eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """``a`` minimising ``sum_i w_i |a x_i - y_i|`` (or its truncated form).

    Ported verbatim from Pi3 ``pi3/utils/alignment.py``; ``w`` made optional.

    ### Parameters
    - ``x``, ``y``: (..., n)
    - ``w``: (..., n), non-negative.  ``None`` means uniform.
    - ``trunc``: per-residual cap.  ``None`` -> plain weighted L1.  Scalar,
      ``(..., n)``, or -- unlike upstream Pi3 -- ``(..., 1)`` for one cap per
      batch element.

    ### Returns
    - ``a``: (...), differentiable through the selected sample
    - ``loss``: (...), objective at ``a``, detached
    - ``index``: (...), the ``k`` with ``a = y[k] / x[k]``
    """
    if w is None:
        w = torch.ones_like(x)

    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        # Derivative of the piecewise-linear objective, evaluated between
        # consecutive breakpoints.  w*x >= 0 so this is non-decreasing, which is
        # what makes the binary search below legal.
        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(derivatives, torch.zeros_like(derivatives[..., :1]),
                                    side='left').clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)

    else:
        # Reshape to (batch_size, n) for simplicity
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])

        # DEVIATION FROM Pi3.  Upstream passes ``trunc`` straight into
        # ``_compute_residual``, whose ``clamp_max_`` runs on a (extrema, n)
        # table -- so a per-BATCH cap (one per frame) raises a shape error there
        # and only a scalar or per-sample cap survives.  A per-frame cap is what
        # ``depth_si_loss`` actually wants, since the cap is a fraction of that
        # frame's median depth, so the batched form is gathered per split below.
        trunc_b = None
        if torch.is_tensor(trunc) and trunc.dim() > 0:
            trunc_b = trunc.reshape(batch_size, -1)

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)    # stacked for convenient gathering

        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
            # Prefix sums in the orders of A, B, C
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)

            # Left and right derivative at every candidate breakpoint
            j_A = torch.searchsorted(A, y_div_x, side='left').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='left').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='left').sub_(1)
            left_derivative = (2 * torch.gather(Q_A, dim=-1, index=j_A)
                               - torch.gather(Q_B, dim=-1, index=j_B)
                               - torch.gather(Q_C, dim=-1, index=j_C))
            j_A = torch.searchsorted(A, y_div_x, side='right').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='right').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='right').sub_(1)
            right_derivative = (2 * torch.gather(Q_A, dim=-1, index=j_A)
                                - torch.gather(Q_B, dim=-1, index=j_B)
                                - torch.gather(Q_C, dim=-1, index=j_C))

            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            # All-zero derivatives: take the first breakpoint as the extremum.
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)

            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]
            MAX_ELEMENTS = 4096 ** 2      # split so the residual table stays ~1G
            SPLIT_SIZE = max(1, MAX_ELEMENTS // x.shape[-1])
            extrema_value = torch.cat([
                _compute_residual(extrema_a_split[:, None], xyw[extrema_i_split, :, :],
                                  trunc if trunc_b is None else trunc_b[extrema_i_split])
                for extrema_a_split, extrema_i_split
                in zip(extrema_a.split(SPLIT_SIZE), where_extrema_batch.split(SPLIT_SIZE))
            ])

            minima, indices = scatter_min(size=batch_size, dim=0,
                                          index=where_extrema_batch, src=extrema_value)
            index = where_extrema_index[indices]

        a = (torch.gather(y, dim=-1, index=index[..., None])
             / torch.gather(x, dim=-1, index=index[..., None]).clamp_min(eps))
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index


def ratio_weights(pred: torch.Tensor, weights: Optional[torch.Tensor] = None,
                  floor: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """Weights that make ``align`` a plain (conf-weighted) median of the RATIOS.

    ★ THIS IS NOT OPTIONAL, AND LEAVING IT OUT INVERTS THE ESTIMATOR.  Rewrite
    the objective in terms of the per-sample ratio ``r_i = y_i / x_i``:

        sum_i w_i |a x_i - y_i|  =  sum_i (w_i x_i) |a - r_i|

    so ``align`` returns the median of ``r`` weighted by ``w * x`` -- NOT by
    ``w``.  With ``w = 1`` the weight is ``x`` itself, i.e. the very samples with
    the largest magnitude get the largest vote.  For depth those are the sky and
    the far field; for step magnitudes under ``pairs="all"`` those are the
    longest baselines.  Measured on the real teacher bank, an L1 fit left this
    way is WORSE than least squares under 10% gross outliers (recovered scale
    0.034 vs 0.043 against a truth of 1.000) because a corrupted pixel is
    corrupted precisely by being large, so it buys its own vote.

    Dividing by ``pred`` cancels the ``x``:  ``w_i = c_i / x_i``  =>  ratio
    weight ``c_i``.  Same measurement then recovers 0.993.

    ``floor`` guards the other end: ``x -> 0`` would hand one near pixel an
    unbounded vote, so ``x`` is clamped from below at ``floor`` times its own
    per-row median before inversion.  Pi3 does the same thing for the same
    reason (``PointLoss.forward``: ``weights.clamp_min(0.1 * weighted_mean(...))``
    then ``1 / weights``) -- it applies the clamp to the GT depth because that is
    its trusted reference, but the algebra above says the divisor has to be the
    tensor ``align`` receives as ``x``, so here it is ``pred``.
    """
    x = pred.abs()
    med = x.median(dim=-1, keepdim=True).values
    x = x.clamp_min((floor * med).clamp(min=eps))
    return (1.0 / x) if weights is None else (weights / x)


def ratio_trunc(pred: torch.Tensor, target: torch.Tensor,
                weights: Optional[torch.Tensor] = None,
                rel: float = 0.1, eps: float = 1e-12) -> torch.Tensor:
    """A truncation cap in the units ``align`` actually truncates in.

    ``align`` caps the WEIGHTED residual ``w_i |a x_i - y_i|``.  Once
    ``ratio_weights`` has set ``w_i = c_i / x_i`` that residual is ``c_i |a - r_i|``
    -- conf times a RATIO -- so a cap expressed in units of ``x`` or ``y`` (depth,
    step length) is simply the wrong dimension, and its effective tolerance then
    drifts with both the confidence scale and the depth scale of whatever frame
    it is applied to.  Measured on the real bank before this existed: a cap meant
    as "10% of median depth" acted as a 6.55% tolerance on the ratio, and would
    have moved with the data.

    Here ``rel`` is a RELATIVE tolerance on the ratio: samples whose ratio misses
    the fitted scale by more than ``rel`` times the typical ratio stop
    contributing.  ``median(target)/median(pred)`` is the typical ratio -- cheap,
    robust, and needed only to set a threshold, so it is detached.
    """
    with torch.no_grad():
        r_ref = (target.median(dim=-1, keepdim=True).values.abs()
                 / pred.median(dim=-1, keepdim=True).values.abs().clamp_min(eps))
        cap = rel * r_ref.clamp_min(eps)
        return cap if weights is None else cap * weights


def robust_scale(pred: torch.Tensor, target: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 trunc: Optional[Union[float, torch.Tensor]] = None,
                 ratio_weighted: bool = True) -> torch.Tensor:
    """``s`` minimising ``sum_i w_i |s*pred_i - target_i|`` -- the L1 twin of
    ``losses.closed_form_scale``.

    Same signature and same orientation as ``closed_form_scale`` so the two are
    interchangeable at the call sites; the only difference is the norm.
    Batched over leading dims: ``pred``/``target`` of shape ``(..., n)`` give
    ``(...)`` scales, which is how ``depth_si_loss`` fits all S frames at once
    instead of looping.

    ``ratio_weighted=True`` (the default) folds ``weights`` through
    ``ratio_weights`` first, so ``weights`` means what a caller expects it to
    mean -- a per-sample importance on the ratio -- rather than being silently
    multiplied by the sample magnitude.  Pass ``False`` for the raw upstream
    objective.
    """
    w = ratio_weights(pred, weights) if ratio_weighted else weights
    scale, _, _ = align(pred, target, w, trunc)
    return scale


def subsample_for_align(*tensors: torch.Tensor, n: int = DEFAULT_ALIGN_RES,
                        generator: Optional[torch.Generator] = None):
    """Uniformly thin ``(..., N)`` tensors along the last dim to at most ``n``.

    The truncated solver enumerates extrema, so it wants a few thousand samples,
    not 200k.  Pi3 does the same thing via ``PointLoss.prepare_ROE``; here the
    sampling is deterministic by default (a fixed stride) so a loss value is
    reproducible across runs.
    """
    N = tensors[0].shape[-1]
    if N <= n:
        return tensors if len(tensors) > 1 else tensors[0]
    if generator is None:
        idx = torch.arange(0, N, max(1, N // n), device=tensors[0].device)[:n]
    else:
        idx = torch.randperm(N, generator=generator, device=tensors[0].device)[:n]
    out = tuple(None if t is None else t.index_select(-1, idx) for t in tensors)
    return out if len(out) > 1 else out[0]
