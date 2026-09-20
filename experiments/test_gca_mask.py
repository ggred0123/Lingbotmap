"""Does the GCA mask predicate reproduce the cache path?  Bookkeeping only.

This is the cheap half of T2.  The expensive half (docs/phase1-plan.md §3-T2)
runs both paths through the real weights and compares tensors; it can only tell
you THAT they differ.  This one simulates the cache's append/evict schedule
exactly as ``attention.py`` performs it, records which key frames each query
frame could actually see, and diffs that against ``gca_mask``.  When it fails it
names the frame pair.

No GPU, no checkpoint, seconds to run.  Run it before spending a rollout.

    python experiments/test_gca_mask.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.aggregator import gca_mask
from lingbot_map.aggregator.gca_mask import (
    ROLE_SCALE, ROLE_KEYFRAME, ROLE_NONKEY, VIS_NONE, VIS_FULL, VIS_SPECIAL,
)


class CacheSim:
    """The streaming KV cache, in frame ids.

    Mirrors, in order:
      append          attention.py:642-649
      evict           attention.py:687-729  (strict `>`, see §3-T2 off-by-one)
      specials kept   attention.py:701-716
      skip_append     attention.py:657-664
    """

    def __init__(self, scale_frames, sliding_window, keyframe_interval):
        self.sf = scale_frames
        self.W = sliding_window
        self.K = max(keyframe_interval, 1)
        self.cache = []       # frame ids holding full tokens, oldest first
        self.specials = []    # frame ids reduced to their special tokens
        self.tfp = 0          # aggregator.total_frames_processed
        self.visible = {}     # frame id -> {key frame id: 'full' | 'special'}
        self.rope_idx = {}    # frame id -> 3D RoPE temporal index
        self.n_evicted_at = {}

    def _evict(self):
        if len(self.cache) > self.W + self.sf:
            end = len(self.cache) - self.W
            if end > self.sf:
                self.specials.extend(self.cache[self.sf:end])
                self.cache = self.cache[:self.sf] + self.cache[-self.W:]

    def _snapshot(self, t, extra_full=()):
        vis = {f: 'special' for f in self.specials}
        for f in self.cache:
            vis[f] = 'full'
        for f in extra_full:
            vis[f] = 'full'
        self.visible[t] = vis
        self.n_evicted_at[t] = len(self.specials)

    def run_anchor(self):
        """Phase 1: the sf anchor frames go in as one block, fully bidirectional."""
        self.cache = list(range(self.sf))
        self._evict()
        for t in range(self.sf):
            self.rope_idx[t] = t
            self._snapshot(t)
        self.tfp = self.sf

    def step(self, t):
        is_kf = (t - self.sf) % self.K == 0
        self.rope_idx[t] = self.tfp
        if is_kf:
            self.cache.append(t)
            self._evict()
            self._snapshot(t)
            self.tfp += 1
        else:
            # attends to [cache + own], stores nothing
            self._snapshot(t, extra_full=(t,))

    def run(self, n_frames):
        self.run_anchor()
        for t in range(self.sf, n_frames):
            self.step(t)
        return self


def plan_key_frame_ids(sim_at_t0, plan, window_start, S):
    """Frame ids of the mask's key columns, in mask order.

    [ evicted specials | prefix anchor | prefix full keyframes | window ]
    Block A is not part of ``frame_visibility``'s output, so it is returned
    separately.
    """
    head = list(sim_at_t0['specials'])
    body = list(sim_at_t0['cache']) + list(range(window_start, window_start + S))
    assert len(head) == plan.prefix_evicted, (len(head), plan.prefix_evicted)
    assert len(sim_at_t0['cache']) == plan.prefix_scale_frames + plan.prefix_full_keyframes
    return head, body


def prefix_state(sf, W, K, window_start):
    """Stream state after frames [0, window_start) -- the prefix the window attends to.

    window_start == 0 means the window contains its own anchor: nothing is cached
    and the frame counter is still 0.  The anchor is atomic (all sf frames enter
    as one block), so a window may start at 0 or at/after sf, never inside it.
    """
    assert window_start == 0 or window_start >= sf, \
        f"window_start={window_start} lands inside the anchor (sf={sf})"
    pre = CacheSim(sf, W, K)
    if window_start > 0:
        pre.run_anchor()
        for t in range(sf, window_start):
            pre.step(t)
    return pre


def check_case(name, sf, W, K, window_start, S, verbose=False):
    total = window_start + S
    sim = CacheSim(sf, W, K).run(total)

    pre = prefix_state(sf, W, K, window_start)
    prefix = {'cache': list(pre.cache), 'specials': list(pre.specials), 'tfp': pre.tfp}

    plan = gca_mask.plan_window(
        S=S, window_start=window_start, scale_frames=sf, keyframe_interval=K,
        sliding_window=W, total_frames_processed=prefix['tfp'],
        prefix_cached_frames=len(prefix['cache']),
        prefix_evicted_frames=len(prefix['specials']),
    )
    head_ids, body_ids = plan_key_frame_ids(prefix, plan, window_start, S)
    vis = gca_mask.frame_visibility(plan)          # [S, F] over body_ids

    fails = []

    # RoPE indices: the counter must land where the streaming rollout put it.
    for w in range(S):
        t = window_start + w
        got, want = int(plan.rope_frame_idx[w]), sim.rope_idx[t]
        if got != want:
            fails.append(f"rope idx frame {t}: mask {got} != stream {want}")

    for w in range(S):
        t = window_start + w
        want = sim.visible[t]                       # {key id: 'full'|'special'}
        got = {}
        for f in head_ids:
            got[f] = 'special'                      # block A is unconditional
        for j, f in enumerate(body_ids):
            code = int(vis[w, j])
            if code == VIS_FULL:
                got[f] = 'full'
            elif code == VIS_SPECIAL:
                # a frame can appear as both a head id and a body id only if the
                # layout is wrong; head/body are disjoint by construction
                got[f] = 'special'
        # every key the stream could see must be reachable in the mask, identically
        for f in sorted(set(want) | set(got)):
            a, b = want.get(f), got.get(f)
            if a != b:
                fails.append(f"query frame {t}: key frame {f} stream={a} mask={b}")

    n_ev_window = sim.n_evicted_at[window_start + S - 1] - len(prefix['specials'])
    status = "FAIL" if fails else "ok"
    print(f"  [{status:>4}] {name:<44} sf={sf} W={W} K={K} t0={window_start} S={S} "
          f"| prefix {len(prefix['cache'])}f+{len(prefix['specials'])}ev "
          f"| evictions during window: {n_ev_window}")
    if verbose or fails:
        print(f"         {gca_mask.describe(plan)}")
    for f in fails[:12]:
        print(f"         {f}")
    if len(fails) > 12:
        print(f"         ... and {len(fails) - 12} more")
    return len(fails), n_ev_window


def check_token_mask(sf, W, K, window_start, S, P, num_special):
    """The token-level mask must expand the frame codes and nothing else."""
    pre = prefix_state(sf, W, K, window_start)
    plan = gca_mask.plan_window(
        S=S, window_start=window_start, scale_frames=sf, keyframe_interval=K,
        sliding_window=W, total_frames_processed=pre.tfp,
        prefix_cached_frames=len(pre.cache), prefix_evicted_frames=len(pre.specials))
    vis = gca_mask.frame_visibility(plan)
    m = gca_mask.build_gca_mask(plan, tokens_per_frame=P, num_special=num_special,
                                device="cpu", mask_dtype="bool", align=1)[0, 0]
    n_head = plan.prefix_evicted * num_special
    exp_kv = n_head + (plan.prefix_scale_frames + plan.prefix_full_keyframes + S) * P
    bad = []
    if tuple(m.shape) != (S * P, exp_kv):
        bad.append(f"shape {tuple(m.shape)} != {(S * P, exp_kv)}")
    # the aligned build must be the same mask plus dead columns
    ma = gca_mask.build_gca_mask(plan, tokens_per_frame=P, num_special=num_special,
                                 device="cpu", mask_dtype="bool")[0, 0]
    if ma.shape[1] % gca_mask.KV_ALIGN:
        bad.append(f"aligned KV {ma.shape[1]} is not a multiple of {gca_mask.KV_ALIGN}")
    if not bool((ma[:, :exp_kv] == m).all()):
        bad.append("padding changed the real columns")
    if bool(ma[:, exp_kv:].any()):
        bad.append("padded columns are not all False")
    if not bool(m[:, :n_head].all()):
        bad.append("evicted-special block is not fully visible")
    for w in (0, S // 2, S - 1):
        for j in range(vis.shape[1]):
            code = int(vis[w, j])
            col = m[w * P, n_head + j * P: n_head + (j + 1) * P]
            n_true = int(col.sum())
            want = P if code == VIS_FULL else (num_special if code == VIS_SPECIAL else 0)
            if n_true != want:
                bad.append(f"q{w} key{j}: {n_true} visible tokens, expected {want} (code {code})")
    # rows within one query frame must be identical -- visibility is per frame
    if not bool((m[0] == m[P - 1]).all()):
        bad.append("tokens of the same query frame disagree")
    print(f"  [{'FAIL' if bad else '  ok':>4}] token expansion "
          f"(P={P}, {num_special} special) -> {tuple(m.shape)}, "
          f"KV-aligned {ma.shape[1]} (+{ma.shape[1] - exp_kv})")
    for b in bad:
        print(f"         {b}")
    return len(bad)


def check_camera_case(name, sf, K, window_start, S):
    """The camera head's trunk: same rules, no eviction, one token per frame.

    Simulated with W = infinity, which is what the dead eviction guard
    (attention.py:307, ``shape[3] > 1`` with 1 token per frame) amounts to.  If
    that guard ever starts firing this check fails loudly rather than the
    training step drifting.
    """
    W = gca_mask.NO_EVICTION
    sim = CacheSim(sf, W, K).run(window_start + S)
    pre = prefix_state(sf, W, K, window_start)

    plan = gca_mask.plan_camera_window(
        S=S, window_start=window_start, scale_frames=sf, keyframe_interval=K,
        prefix_cached_frames=len(pre.cache))
    vis = gca_mask.frame_visibility(plan)
    body_ids = list(pre.cache) + list(range(window_start, window_start + S))

    fails = []
    if plan.prefix_evicted != 0 or pre.specials:
        fails.append("camera plan produced an evicted block; it must never evict")
    if bool((vis == VIS_SPECIAL).any()):
        fails.append("camera plan produced SPECIAL-only visibility")

    for w in range(S):
        t = window_start + w
        want = sim.visible[t]
        got = {f: 'full' for j, f in enumerate(body_ids) if int(vis[w, j]) == VIS_FULL}
        for f in sorted(set(want) | set(got)):
            if want.get(f) != got.get(f):
                fails.append(f"query frame {t}: key {f} stream={want.get(f)} mask={got.get(f)}")

    # one token per frame -> the token mask is the frame mask
    m = gca_mask.build_gca_mask(plan, tokens_per_frame=1, num_special=1,
                                device="cpu", mask_dtype="bool", align=1)[0, 0]
    if tuple(m.shape) != (S, len(body_ids)):
        fails.append(f"token mask {tuple(m.shape)} != {(S, len(body_ids))}")
    elif not bool((m == (vis == VIS_FULL)).all()):
        fails.append("P=1 token mask does not equal the frame mask")

    print(f"  [{'FAIL' if fails else '  ok':>4}] {name:<44} sf={sf} K={K} t0={window_start} "
          f"S={S} | prefix {len(pre.cache)} frames | KV {len(body_ids)}")
    for f in fails[:8]:
        print(f"         {f}")
    return len(fails)


def main():
    print("GCA mask vs. simulated cache schedule\n")
    print("A. standalone windows (no prefix; window starts at its own anchor)")
    fails = 0
    ev = []
    for name, sf, W, K, t0, S in [
        ("T2-a  causal + anchor, no eviction",      8, 64, 1, 0, 16),
        ("T2-b  eviction (sw=8): 20 > 16",          8,  8, 1, 0, 20),
        ("T2-b' boundary: 16 > 16 is FALSE",        8,  8, 1, 0, 16),
        ("T2-b''boundary: 17 > 16 is true",         8,  8, 1, 0, 17),
        ("T2-c  deeper eviction",                   8,  8, 1, 0, 24),
        ("T2-d  non-keyframes K=4",                 8,  8, 4, 0, 20),
        ("      non-keyframes K=4, long",           8,  8, 4, 0, 48),
        ("      K=28 (deployed), W=64",             8, 64, 28, 0, 96),
        ("      W=1 degenerate",                    8,  1, 1, 0, 16),
        ("      sf=1",                              1,  4, 3, 0, 24),
    ]:
        f, n = check_case(name, sf, W, K, t0, S)
        fails += f
        ev.append((name, n))

    print("\nB. windows on top of a rolled prefix cache")
    for name, sf, W, K, t0, S in [
        ("prefix inside sliding window",            8, 64, 1, 40, 16),
        ("prefix past eviction",                    8,  8, 1, 40, 20),
        ("prefix past eviction, K=4",               8,  8, 4, 60, 20),
        ("deployed shape K=28 W=64",                8, 64, 28, 2000, 48),
        ("deployed shape, t0 not on a keyframe",    8, 64, 28, 2001, 48),
        ("K=4 window inside a K=4 rollout",         8, 16, 4, 300, 32),
    ]:
        f, n = check_case(name, sf, W, K, t0, S)
        fails += f
        ev.append((name, n))

    print("\nC. token expansion")
    fails += check_token_mask(8, 8, 4, 60, 20, P=17, num_special=6)
    fails += check_token_mask(8, 64, 28, 2000, 12, P=13, num_special=6)

    print("\nC2. camera head trunk (T2-e): no eviction, 1 token/frame")
    for name, sf, K, t0, S in [
        ("standalone, K=1",                  8,  1, 0, 16),
        ("standalone, K=4",                  8,  4, 0, 24),
        ("on prefix, K=1",                   8,  1, 40, 20),
        ("on prefix, K=4",                   8,  4, 60, 20),
        ("deployed K=28, far prefix",        8, 28, 5248, 48),
    ]:
        fails += check_camera_case(name, sf, K, t0, S)

    print("\nD. eviction coverage (a test that evicts nothing tests nothing)")
    silent = [n for n, c in ev if c == 0]
    print(f"  {len(ev) - len(silent)}/{len(ev)} cases actually evict during the window")
    if silent:
        print(f"  no eviction in: {', '.join(silent)}")

    print(f"\n{'FAILED' if fails else 'PASSED'} -- {fails} mismatches")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
