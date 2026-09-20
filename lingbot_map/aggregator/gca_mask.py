"""GCA attention mask for the parallel (masked) training path.

The deployed aggregator reads cross-frame context out of a KV cache that is
mutated frame by frame: append -> evict -> attend.  That is a *sequential*
formulation and it cannot be trained efficiently (docs/phase1-plan.md §1.1,
§1.2 -- 10.07 GB per supervised frame).  The same read set can be expressed as
a mask over one parallel forward pass of the whole supervised window, which is
4.5x cheaper (§1.4).

This module builds that mask.  It is the arithmetic half of T1; the attention
plumbing lives in ``SDPAAttention`` / ``AggregatorStream``.

The rules it has to reproduce, and where they live in the cache path:

  1. anchor (scale) frames are never evicted and are visible to everyone
     ``attention.py:718-726`` keeps ``[:scale_frames]`` on every eviction
  2. a frame always sees itself
     keyframe: appended before attention; non-keyframe: ``cat(cache, current)``
  3. only keyframes persist -- a non-keyframe is never visible to any other
     frame (``_skip_append``, ``attention.py:642-664``)
  4. of the persisted keyframes only the most recent ``W`` keep their full
     tokens (``attention.py:718-726``)
  5. an evicted keyframe keeps its ``num_special`` special tokens forever
     (``attention.py:701-716``)

Rule 4 is history dependent: whether keyframe ``c`` still has full tokens when
frame ``w`` attends depends on how many keyframes had been appended *at that
moment*, not on the state at the end of the window.  ``plan_window`` computes
that count per query frame; everything else follows from it.

★ off-by-one -- eviction triggers on ``num_cached > W + sf`` (strict, see
``attention.py:694``), i.e. from the (W+sf+1)-th cached frame.  The plan is
written against the same strict inequality; see docs/phase1-plan.md §3-T2.
"""

from dataclasses import dataclass
from typing import Optional

import torch

# Per-frame role in the streaming schedule.
ROLE_SCALE = 0     # anchor frame -- permanent, bidirectional among themselves
ROLE_KEYFRAME = 1  # persisted in the KV cache, evictable
ROLE_NONKEY = 2    # attends but never persists (_skip_append)

# Per-(query frame, key frame) visibility.
VIS_NONE = 0
VIS_FULL = 1      # all P tokens of the key frame
VIS_SPECIAL = 2   # only its first `num_special` tokens (evicted keyframe)


@dataclass
class WindowPlan:
    """Everything about a supervised window that the mask and RoPE need.

    Frame counts, not token counts.  ``prefix_*`` describe the KV cache the
    window attends to; with no prefix they are all zero and the window is a
    standalone clip that starts at its own anchor.
    """
    S: int                       # window length in frames
    window_start: int            # absolute frame index of window[0]
    scale_frames: int            # sf -- anchor length of the sequence
    sliding_window: int          # W -- keyframes kept with full tokens
    roles: torch.Tensor          # [S] int64, ROLE_*
    kf_ordinal: torch.Tensor     # [S] int64, keyframe append-order index, -1 if not a keyframe
    n_kf_at_query: torch.Tensor  # [S] int64, keyframes in cache when frame w attends
    rope_frame_idx: torch.Tensor  # [S] int64, 3D-RoPE temporal index (keyframes only advance it)
    prefix_scale_frames: int     # anchor frames sitting in the prefix cache
    prefix_full_keyframes: int   # keyframes still holding full tokens in the prefix cache
    prefix_evicted: int          # keyframes reduced to special tokens before the window
    prefix_first_full_ordinal: int  # append-order index of the oldest full-token prefix keyframe

    @property
    def n_kf_before(self) -> int:
        """Keyframes appended before the window started."""
        return self.prefix_evicted + self.prefix_full_keyframes


def frame_roles(window_start: int, S: int, scale_frames: int, keyframe_interval: int,
                device=None) -> torch.Tensor:
    """Role of each window frame, from its ABSOLUTE index.

    Mirrors ``gct_stream.inference_streaming``:
        is_keyframe = (K <= 1) or ((i - scale_frames) % K == 0)
    and frames below ``scale_frames`` are anchor frames.
    """
    t = torch.arange(window_start, window_start + S, dtype=torch.long, device=device)
    K = max(int(keyframe_interval), 1)
    roles = torch.full((S,), ROLE_NONKEY, dtype=torch.long, device=device)
    is_scale = t < scale_frames
    is_kf = ((t - scale_frames) % K == 0) & ~is_scale
    roles[is_kf] = ROLE_KEYFRAME
    roles[is_scale] = ROLE_SCALE
    return roles


def plan_window(
    S: int,
    window_start: int,
    scale_frames: int,
    keyframe_interval: int,
    sliding_window: int,
    total_frames_processed: int,
    prefix_cached_frames: int = 0,
    prefix_evicted_frames: int = 0,
    roles: Optional[torch.Tensor] = None,
) -> WindowPlan:
    """Derive the window schedule from the live stream state.

    ``prefix_cached_frames`` / ``prefix_evicted_frames`` are read off the actual
    cache tensors (``k_0.shape[2]`` and ``k_0_special.shape[2]``) rather than
    recomputed, so a plan can never disagree with the state it is describing.

    ``total_frames_processed`` is the aggregator's own counter and becomes the
    RoPE origin.  It counts anchor frames and keyframes only -- non-keyframes do
    not advance it (``stream.py:526``), which is the RoPE convention the masked
    path has to match or it will be silently wrong (§3-T1 risk).
    """
    if roles is None:
        roles = frame_roles(window_start, S, scale_frames, keyframe_interval)
    roles = roles.to(torch.long)

    prefix_scale = min(scale_frames, prefix_cached_frames)
    prefix_full_kf = max(0, prefix_cached_frames - prefix_scale)
    N0 = prefix_evicted_frames + prefix_full_kf

    is_kf = roles == ROLE_KEYFRAME
    # keyframes strictly before each frame, within the window
    cnt_before = torch.cumsum(is_kf.to(torch.long), 0) - is_kf.to(torch.long)
    kf_ordinal = torch.where(is_kf, N0 + cnt_before, torch.full_like(cnt_before, -1))
    # keyframes in cache at the moment frame w attends: a keyframe appends itself first
    n_kf_at_query = N0 + cnt_before + is_kf.to(torch.long)

    # RoPE: anchor frames and keyframes advance the counter, non-keyframes do not
    advances = (roles != ROLE_NONKEY).to(torch.long)
    rope_frame_idx = total_frames_processed + torch.cumsum(advances, 0) - advances

    return WindowPlan(
        S=S,
        window_start=window_start,
        scale_frames=scale_frames,
        sliding_window=sliding_window,
        roles=roles,
        kf_ordinal=kf_ordinal,
        n_kf_at_query=n_kf_at_query,
        rope_frame_idx=rope_frame_idx,
        prefix_scale_frames=prefix_scale,
        prefix_full_keyframes=prefix_full_kf,
        prefix_evicted=prefix_evicted_frames,
        prefix_first_full_ordinal=prefix_evicted_frames,
    )


# The camera head's cache never evicts.  Its eviction guard is
# ``if kv_cache[k].shape[3] > 1`` (attention.py:307) and shape[3] is tokens per
# frame, which is 1 there (one camera token per frame) -- so the branch is dead
# and the cache grows monotonically.  Verified empirically: after 30 frames at
# K=4 the cache holds 14 frames (8 anchor + 6 keyframes) with no special block.
# Expressing that as W = infinity keeps ONE predicate for both stacks.
NO_EVICTION = 1 << 30

# SDPA backend dispatch is sensitive to the KV length; see build_gca_mask.
KV_ALIGN = 64


def plan_camera_window(
    S: int,
    window_start: int,
    scale_frames: int,
    keyframe_interval: int,
    prefix_cached_frames: int = 0,
) -> WindowPlan:
    """Window plan for the camera head: same GCA rules minus eviction.

    The camera head is a second causal stack (4 refinement iterations x 4
    ``CameraBlock``s) over one token per frame.  Relative to the aggregator it
    drops three rules and keeps two:

      dropped  sliding-window eviction   (guard is dead, see NO_EVICTION)
      dropped  evicted special tokens    (nothing is ever evicted)
      dropped  3D RoPE                   (``rope3d`` is None and the trunk blocks
                                          are built without ``rope``; the deployed
                                          config never sets enable_camera_3d_rope)
      kept     anchor frames bidirectional and permanent
      kept     only keyframes persist; non-keyframes read and vanish

    ``total_frames_processed`` is passed as 0 because the head has no positional
    encoding to index -- ``rope_frame_idx`` is unused here.  (Its own
    ``frame_idx`` counter is also not comparable: ``trunk_fn`` advances it on
    every frame including non-keyframes, unlike the aggregator's counter.  It
    feeds nothing but the absent RoPE.)
    """
    return plan_window(
        S=S,
        window_start=window_start,
        scale_frames=scale_frames,
        keyframe_interval=keyframe_interval,
        sliding_window=NO_EVICTION,
        total_frames_processed=0,
        prefix_cached_frames=prefix_cached_frames,
        prefix_evicted_frames=0,
    )


def frame_visibility(plan: WindowPlan) -> torch.Tensor:
    """[S, F] VIS_* code -- query window frame x key frame.

    Key frames are laid out in the same order the cache path concatenates them
    (``attention.py:669-674``), minus the evicted-special block which is
    unconditionally visible and handled by ``build_gca_mask``:

        [ prefix anchor | prefix full keyframes | window frames ]
    """
    S = plan.S
    n_pre_scale = plan.prefix_scale_frames
    n_pre_full = plan.prefix_full_keyframes
    F = n_pre_scale + n_pre_full + S
    dev = plan.roles.device

    key_kind = torch.empty(F, dtype=torch.long, device=dev)
    key_ord = torch.full((F,), -1, dtype=torch.long, device=dev)
    key_wpos = torch.full((F,), -1, dtype=torch.long, device=dev)  # -1 == prefix, not in window

    key_kind[:n_pre_scale] = ROLE_SCALE
    key_kind[n_pre_scale:n_pre_scale + n_pre_full] = ROLE_KEYFRAME
    key_ord[n_pre_scale:n_pre_scale + n_pre_full] = torch.arange(
        plan.prefix_first_full_ordinal,
        plan.prefix_first_full_ordinal + n_pre_full, device=dev)
    key_kind[n_pre_scale + n_pre_full:] = plan.roles
    key_ord[n_pre_scale + n_pre_full:] = plan.kf_ordinal
    key_wpos[n_pre_scale + n_pre_full:] = torch.arange(S, device=dev)

    q_pos = torch.arange(S, device=dev).unsqueeze(1)          # [S, 1]
    q_lo = (plan.n_kf_at_query - plan.sliding_window).unsqueeze(1)  # [S, 1]
    k_kind = key_kind.unsqueeze(0)                            # [1, F]
    k_ord = key_ord.unsqueeze(0)
    k_wpos = key_wpos.unsqueeze(0)

    is_own = k_wpos == q_pos
    # prefix frames are unconditionally in the past; window frames only if earlier
    is_past = (k_wpos < 0) | (k_wpos < q_pos)
    kf_visible = (k_kind == ROLE_KEYFRAME) & is_past
    kf_full = kf_visible & (k_ord >= q_lo)

    vis = torch.full((S, F), VIS_NONE, dtype=torch.uint8, device=dev)
    vis[kf_visible] = VIS_SPECIAL          # evicted -> special tokens survive
    vis[kf_full] = VIS_FULL                # still inside the sliding window
    vis[(k_kind == ROLE_SCALE).expand(S, F)] = VIS_FULL   # anchor never evicted
    vis[is_own.expand(S, F)] = VIS_FULL    # a frame always sees itself
    return vis


def build_gca_mask(
    plan: WindowPlan,
    tokens_per_frame: int,
    num_special: int,
    device,
    mask_dtype: str = "bool",
    value_dtype: torch.dtype = torch.bfloat16,
    special_tokens_per_evicted: Optional[int] = None,
    align: Optional[int] = None,   # None -> module-level KV_ALIGN, resolved late
) -> torch.Tensor:
    """[1, 1, S*P, KV] attention mask for the whole window, shared by all blocks.

    Key layout, matching the order ``SDPAAttention`` concatenates them in:

        [ evicted specials | prefix anchor + full keyframes | window ]
          E0 * n_spec           (sf + F0) * P                 S * P

    Built once per step and reused by all 24 global blocks -- rebuilding it per
    block is the single biggest avoidable cost (§2.3: 16.3 -> 44.6 GB when the
    build lands inside the measured region).

    ``mask_dtype``:
      "bool"  -- True == attend.  1 byte/element.
      "float" -- additive bias, 0 / finfo.min.  2 bytes/element, but SDPA does
                 not have to convert it on every call (§1.6).  Which one wins
                 under gradient checkpointing is open -- docs/phase1-plan.md
                 §6-Q2, to be measured after T1.
    """
    align = KV_ALIGN if align is None else align
    n_spec_ev = num_special if special_tokens_per_evicted is None else special_tokens_per_evicted
    P = tokens_per_frame
    # Imported lazily: lingbot_map.aggregator's __init__ pulls in stream -> block
    # -> attention, so a module-level import here would be circular.  This runs
    # once per step, not once per block.
    from lingbot_map.layers.attention import attention_dtype
    value_dtype = attention_dtype(value_dtype)

    vis = frame_visibility(plan).to(device)                    # [S, F]
    S, F = vis.shape

    # frame codes -> token mask for the [prefix full | window] blocks
    within = torch.arange(P, device=device).repeat(F)          # [F*P]
    is_special_tok = within < num_special
    m = (vis == VIS_FULL).repeat_interleave(P, dim=1)
    m |= (vis == VIS_SPECIAL).repeat_interleave(P, dim=1) & is_special_tok.unsqueeze(0)

    # evicted-special block: those frames were evicted before the window began,
    # so every query in the window sees exactly their surviving special tokens.
    if plan.prefix_evicted > 0:
        head = torch.ones((S, plan.prefix_evicted * n_spec_ev), dtype=torch.bool, device=device)
        m = torch.cat([head, m], dim=1)

    # ★ Pad the KV axis to a multiple of `align` with columns that attend to
    # nothing.  Numerically inert, but SDPA's backend dispatch is extremely
    # sensitive to KV alignment and the natural length is effectively random:
    # KV = n_evicted*6 + (n_cached + S)*P with P = 1042 = 2 x 521, so the 2-adic
    # alignment is decided by the parity of the cached frame count as the rollout
    # walks forward.  Measured at S=48, one attention call:
    #     KV 61478 (2-aligned)  408 ms      <- two keys short of...
    #     KV 61480 (8-aligned)   99 ms
    #     KV 61504 (32-aligned)  76 ms
    # i.e. a 5.4x swing that a trainer would experience as random throughput.
    # The Q axis does not show this (S=47 and S=48 measure the same), so only the
    # keys are padded -- padding queries would also risk fully-masked rows.
    if align > 1:
        pad = (-m.shape[1]) % align
        if pad:
            m = torch.cat([m, torch.zeros((m.shape[0], pad), dtype=torch.bool,
                                          device=m.device)], dim=1)

    m = m.repeat_interleave(P, dim=0).unsqueeze(0).unsqueeze(0)  # [1, 1, S*P, KV]
    if mask_dtype == "bool":
        return m
    if mask_dtype == "float":
        # ``~m`` would allocate a second full-size bool (6.3 GB at S=48) -- the
        # exact intermediate that flipped this comparison the first time it was
        # measured (docs/phase1-plan.md §6-Q2 note).  Negate in place instead; we
        # own ``m`` and are about to drop it.
        bias = torch.zeros(m.shape, dtype=value_dtype, device=device)
        bias.masked_fill_(m.logical_not_(), torch.finfo(value_dtype).min)
        del m
        return bias
    raise ValueError(f"mask_dtype must be 'bool' or 'float', got {mask_dtype!r}")


def describe(plan: WindowPlan) -> str:
    """One-line summary -- print it in any harness that builds a window.

    §3-T2 asks for the evicted-frame count to be visible: a window that evicts
    nothing passes the eviction tests without testing eviction.
    """
    roles = plan.roles
    n_scale = int((roles == ROLE_SCALE).sum())
    n_kf = int((roles == ROLE_KEYFRAME).sum())
    n_nk = int((roles == ROLE_NONKEY).sum())
    evicted_end = max(0, int(plan.n_kf_at_query[-1]) - plan.sliding_window)
    evicted_in_window = max(0, evicted_end - plan.prefix_evicted)
    return (f"window t0={plan.window_start} S={plan.S} "
            f"(scale {n_scale} / keyframe {n_kf} / non-keyframe {n_nk})  "
            f"prefix: {plan.prefix_scale_frames} anchor + {plan.prefix_full_keyframes} full "
            f"+ {plan.prefix_evicted} evicted (N0={plan.n_kf_before})  "
            f"W={plan.sliding_window}  evictions during window: {evicted_in_window}  "
            f"rope idx {int(plan.rope_frame_idx[0])}..{int(plan.rope_frame_idx[-1])}")
