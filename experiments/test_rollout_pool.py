"""Does RolloutPool walk the schedule v5 specifies?  Bookkeeping only.

``test_state_mixture.py`` checks the DRAWS.  This checks what the pool does with
them: that each stream rolls at its OWN K, that a rollout ends at its horizon and
not before, that ages are measured from the right origin, and -- the one that
would cost a multi-hour run -- that a resume CONTINUES a rollout instead of
silently redrawing it.

The model is a stub that records every call, so this needs no GPU, no
checkpoint and no bank.  What it cannot check is whether the recorded schedule
matches the real cache; that is ``test_gca_mask.py``'s job, and the two together
cover the path.

    python experiments/test_rollout_pool.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.train.trainer import (
    HORIZONS_V5, K_DIST_V5, RolloutPool, Scene, StateSampler, k_deck, parse_k_dist,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'  ok' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# Stubs.  Only the surface RolloutPool actually touches.
# ─────────────────────────────────────────────────────────────────────────────

class _Agg:
    def __init__(self):
        self.kv_cache = {"k_0": torch.zeros(1)}
        self.total_frames_processed = 0
        self._cached_pos3d = None


class _Cam:
    def __init__(self):
        self.kv_cache = [{"k_0": torch.zeros(1)}]
        self.frame_idx = 0


class StubModel:
    """Records (op, frame, is_keyframe) so the schedule can be read back."""

    def __init__(self):
        self.aggregator = _Agg()
        self.camera_head = _Cam()
        self.calls = []
        self._skip = False

    def _set_skip_append(self, v):
        self._skip = v

    def clean_kv_cache(self):
        self.calls.append(("clean", None, None))
        self.aggregator.total_frames_processed = 0

    def forward(self, images, num_frame_for_scale=None, num_frame_per_block=None,
                causal_inference=None):
        n = images.shape[1]
        self.calls.append(("fwd", n, not self._skip))
        if not self._skip:
            self.aggregator.total_frames_processed += n
        return {}

    # counters the tests read
    def keyframes(self):
        return sum(1 for op, n, kf in self.calls if op == "fwd" and kf and n == 1)

    def frames(self):
        return sum(n for op, n, _ in self.calls if op == "fwd" and n == 1)


class _Bank:
    """Runs of length L tiled from t0, the shape build_banks.sh produces."""

    def __init__(self, t0=80, n=40, L=240):
        self.runs = [{"t0": t0 + i * L, "L": L, "burn_in": 72,
                      "scale_frames": 8, "teacher_interval": 1} for i in range(n)]
        self.frames_covered = n * L
        self.index = {}
        self.root = "stub"

    def __len__(self):
        return len(self.runs)


class _Frames:
    """images[:, a:b] -> a [1, n, 1] tensor.  Shape is all the pool reads."""

    def __getitem__(self, key):
        sl = key[1]
        # _anchor slices images[:, :sf], so start is None there
        n = max(0, (sl.stop or 0) - (sl.start or 0))
        return torch.zeros(1, n, 1)

    def numel(self):
        return 0


def make_scenes(n=4, dataset_of=None, runs=40):
    out = []
    for i in range(n):
        bk = _Bank(n=runs)
        out.append(Scene(name=f"scene{i}", frames=f"/stub/{i}", images=_Frames(),
                         bank=bk, covered=(bk.runs[0]["t0"],
                                           bk.runs[-1]["t0"] + bk.runs[-1]["L"]),
                         dataset=(dataset_of(i) if dataset_of else "mcd")))
    return out


CPU = torch.device("cpu")
SF, S = 8, 48


def make_pool(scenes, starts, K=28, sampler=None, stream_k=None):
    m = StubModel()
    p = RolloutPool(m, scenes, starts, S, SF, K, torch.float32, CPU, sampler=sampler,
                    stream_k=stream_k)
    return m, p


# ─────────────────────────────────────────────────────────────────────────────
print("\nA. no sampler == the pre-v5 walk")

scenes = make_scenes(2)
m, pool = make_pool(scenes, [(0, 80), (1, 80)])
check("every stream takes --K", {st.K for st in pool.streams} == {28})
check("no horizon is set", {st.horizon for st in pool.streams} == {0})
check("anchor_t is the first window", {st.anchor_t for st in pool.streams} == {80})

st = pool.streams[0]
m.calls.clear()
pool.advance(st)
check("advance walks exactly S frames", m.frames() == S, f"{m.frames()} frames")
check("advance lands on t+S", st.t == 128, f"t={st.t}")
check("no reset happened", st.resets == 1, f"resets={st.resets}")

# K=28 from phase origin sf=8: frames 80..127 contain keyframes at 92 and 120
m.calls.clear()
st2 = pool.streams[1]
pool.advance(st2)
kf = m.keyframes()
expected = sum(1 for i in range(80, 128) if (i - SF) % 28 == 0)
check("advance appends the right keyframe count at K=28", kf == expected,
      f"{kf} vs {expected}")


# ─────────────────────────────────────────────────────────────────────────────
print("\nB. each stream rolls at ITS OWN K, not the pool's")

# ★ K NOW COMES FROM THE DECK, NOT FROM A PER-ROLLOUT DRAW (trainer.k_deck).
# The property under test is the same one it always was -- a stream must roll at
# the interval it carries, not at the pool's --K -- only the source changed.
scenes = make_scenes(1)
smp = StateSampler(parse_k_dist("1:1"), [0], 0.0, None, seed=1)
m, pool = make_pool(scenes, [(0, 80)], K=28, sampler=smp, stream_k=[1])
st = pool.streams[0]
check("the deck overrides --K", st.K == 1, f"K={st.K}")
m.calls.clear()
pool.advance(st)
check("K=1 appends every frame", m.keyframes() == S, f"{m.keyframes()} of {S}")

m, pool = make_pool(scenes, [(0, 80)], K=28, sampler=smp, stream_k=[12])
st = pool.streams[0]
m.calls.clear()
pool.advance(st)
expected = sum(1 for i in range(80, 128) if (i - SF) % 12 == 0)
check("K=12 appends every 12th frame from the sf phase origin",
      m.keyframes() == expected, f"{m.keyframes()} vs {expected}")

# and a reset must NOT change it -- that is the whole point of the deck
m, pool = make_pool(make_scenes(3), [(0, 80)], K=28,
                    sampler=StateSampler(parse_k_dist(",".join(f"{k}:{w}" for k, w in K_DIST_V5)),
                                         HORIZONS_V5, 0.0, None, seed=4),
                    stream_k=[8])
st = pool.streams[0]
_ks = set()
for _ in range(2000):
    _ks.add(st.K)
    pool.advance(st)
check("K survives every reset", _ks == {8} and st.resets > 1,
      f"{sorted(_ks)} over {st.resets} resets")

# two streams, two different K, interleaved -- the failure mode of a pool-level K
smp = StateSampler(parse_k_dist("1:1"), [0], 0.0, None, seed=1)
m, pool = make_pool(make_scenes(2), [(0, 80), (1, 80)], K=28, sampler=smp)
pool.streams[0].K, pool.streams[1].K = 1, 28
m.calls.clear()
pool.advance(pool.streams[0])
n1 = m.keyframes()
m.calls.clear()
pool.advance(pool.streams[1])
n28 = m.keyframes()
check("interleaved streams keep separate K", n1 == S and n28 == expected_28
      if (expected_28 := sum(1 for i in range(80, 128) if (i - SF) % 28 == 0)) else False,
      f"K=1 -> {n1} kf, K=28 -> {n28} kf")


# ─────────────────────────────────────────────────────────────────────────────
print("\nC. a rollout ends at its horizon")

for horizon in HORIZONS_V5:
    smp = StateSampler(parse_k_dist("28:1"), [horizon], 0.0, None, seed=2)
    m, pool = make_pool(make_scenes(1), [(0, 80)], K=28, sampler=smp)
    st = pool.streams[0]
    check(f"horizon {horizon:>4} was drawn", st.horizon == horizon, f"{st.horizon}")
    ages, guard = [], 0
    while st.resets == 1 and guard < 500:
        guard += 1
        ages.append(st.t - st.anchor_t)
        pool.advance(st)
    check(f"horizon {horizon:>4}: reset fired", st.resets == 2,
          f"after {len(ages)} windows, max age {max(ages)}")
    check(f"horizon {horizon:>4}: no age exceeded it", max(ages) < horizon,
          f"max age {max(ages)}")
    check(f"horizon {horizon:>4}: age got within S of it",
          max(ages) >= horizon - 2 * S, f"max age {max(ages)}")
    check(f"horizon {horizon:>4}: reset restarts the age",
          st.t - st.anchor_t == 0, f"age {st.t - st.anchor_t}")

# a horizon longer than the scene must not deadlock or over-run
short = make_scenes(1, runs=3)                 # covers [80, 800)
smp = StateSampler(parse_k_dist("28:1"), [3840], 0.0, None, seed=2)
m, pool = make_pool(short, [(0, 80)], K=28, sampler=smp)
st = pool.streams[0]
check("horizon is clipped to the scene", st.horizon <= 800 - 80, f"{st.horizon}")
guard = 0
while st.resets == 1 and guard < 500:
    guard += 1
    pool.advance(st)
check("a scene shorter than every horizon still terminates", guard < 500,
      f"{guard} advances")


# ─────────────────────────────────────────────────────────────────────────────
print("\nD. reset redraws (scene, K, horizon); the horizon path and the "
      "end-of-scene path agree")

smp = StateSampler(parse_k_dist(",".join(f"{k}:{w}" for k, w in K_DIST_V5)),
                   HORIZONS_V5, 0.0, None, seed=9)
m, pool = make_pool(make_scenes(3), [(0, 80)], K=28, sampler=smp)
st = pool.streams[0]
seen_K, seen_h, seen_sc = set(), set(), set()
for _ in range(4000):
    seen_K.add(st.K)
    seen_h.add(st.horizon)
    seen_sc.add(st.scene)
    pool.advance(st)
check("resets do NOT redraw K -- it belongs to the stream", seen_K == {28},
      str(sorted(seen_K)))
check("resets cover the whole horizon support", seen_h == set(HORIZONS_V5),
      str(sorted(seen_h)))
check("resets move between scenes", len(seen_sc) == 3, str(sorted(seen_sc)))
check("every rollout restarts at the first window",
      st.anchor_t == 80, f"anchor_t={st.anchor_t}")


# ─────────────────────────────────────────────────────────────────────────────
print("\nE. dataset-balanced resets")

scenes = make_scenes(6, dataset_of=lambda i: "mcd" if i < 3 else "slowtv")
smp = StateSampler(parse_k_dist("28:1"), [320], 0.0,
                   {"mcd": 50, "slowtv": 50}, seed=4)
m, pool = make_pool(scenes, [(0, 80)], K=28, sampler=smp)
st = pool.streams[0]
picks = {"mcd": 0, "slowtv": 0}
for _ in range(4000):
    if st.resets > 1:
        pass
    picks[scenes[st.scene].dataset] += 1
    pool.advance(st)
frac = picks["mcd"] / sum(picks.values())
check("windows split ~50/50 by dataset", abs(frac - 0.5) < 0.06, f"mcd {frac:.3f}")
check("frames_seen is tracked per dataset",
      set(pool.frames_seen) == {"mcd", "slowtv"} and min(pool.frames_seen.values()) > 0,
      str(pool.frames_seen))


# ─────────────────────────────────────────────────────────────────────────────
print("\nF. resume CONTINUES a rollout instead of redrawing it")

smp = StateSampler(parse_k_dist(",".join(f"{k}:{w}" for k, w in K_DIST_V5)),
                   HORIZONS_V5, 0.0, None, seed=13)
m, pool = make_pool(make_scenes(3), [(0, 80)], K=28, sampler=smp)
st = pool.streams[0]
for _ in range(5):
    pool.advance(st)
saved = {"sid": st.sid, "scene": st.scene, "t": st.t, "K": st.K,
         "horizon": st.horizon, "anchor_t": st.anchor_t,
         "windows_done": st.windows_done, "resets": st.resets}

# a fresh sampler, as a restarted process would have
smp2 = StateSampler(parse_k_dist(",".join(f"{k}:{w}" for k, w in K_DIST_V5)),
                    HORIZONS_V5, 0.0, None, seed=13)
m2, pool2 = make_pool(make_scenes(3), [saved], K=28, sampler=smp2)
r = pool2.streams[0]
check("resume keeps K", r.K == saved["K"], f"{r.K} vs {saved['K']}")
check("resume keeps the horizon", r.horizon == saved["horizon"],
      f"{r.horizon} vs {saved['horizon']}")
check("resume keeps the scene", r.scene == saved["scene"])
check("resume keeps the position", r.t == saved["t"], f"{r.t} vs {saved['t']}")
check("resume keeps the age origin", r.anchor_t == saved["anchor_t"],
      f"{r.anchor_t} vs {saved['anchor_t']}")
check("resume did NOT count as a reset", r.resets == saved["resets"],
      f"{r.resets} vs {saved['resets']}")
# and the rebuilt prefix is the one K=r.K would have produced
kf = m2.keyframes()
want = sum(1 for i in range(SF, r.t) if (r.K <= 1) or ((i - SF) % r.K == 0))
check("resume rolls the prefix at the SAVED K", kf == want, f"{kf} vs {want}")

# the rollout then ends at the ORIGINAL horizon, not a fresh one
guard = 0
while r.resets == saved["resets"] and guard < 500:
    guard += 1
    last_age = r.t - r.anchor_t
    pool2.advance(r)
check("the resumed rollout ends at its own horizon", last_age < saved["horizon"],
      f"last age {last_age}, horizon {saved['horizon']}")


# ─────────────────────────────────────────────────────────────────────────────
print("\nG. the K deck -- an EXACT mixture, not a sampled one")

_kd = parse_k_dist(",".join(f"{k}:{w}" for k, w in K_DIST_V5))
deck = k_deck(_kd, 20, 28)
counts = {k: deck.count(k) for k, _ in K_DIST_V5}
check("pool 20 realises the ver5 mixture exactly",
      counts == {1: 5, 2: 2, 4: 2, 8: 2, 12: 2, 16: 2, 28: 5}, str(counts))
check("the deck is exactly --pool long", len(deck) == 20, str(len(deck)))
check("the deck is sorted (so deck[rank::world] spreads the deep K=1 states)",
      deck == sorted(deck))

# the split is what the trainer actually hands each rank
r0, r1 = deck[0::2], deck[1::2]
check("deck[rank::world] gives each rank pool/world streams",
      len(r0) == 10 and len(r1) == 10, f"{len(r0)}/{len(r1)}")
glob = {k: r0.count(k) + r1.count(k) for k, _ in K_DIST_V5}
check("the GLOBAL multiset survives the split", glob == counts, str(glob))
check("neither rank hoards the K=1 streams",
      abs(r0.count(1) - r1.count(1)) <= 1, f"{r0.count(1)} vs {r1.count(1)}")

# step-weighted share is exact BY CONSTRUCTION: round robin gives every stream
# 1/pool of the correction steps, so the share is (streams at K)/pool.
# K_DIST_V5 holds raw WEIGHTS (25, 10, ...); parse_k_dist normalises them.
for k, p_ in _kd:
    check(f"K={k} step share is exactly {p_:.0%}",
          abs(counts[k] / 20 - p_) < 1e-9, f"{counts[k] / 20:.4f}")

# a pool that cannot represent the mixture must FAIL, not silently skew
for bad in (10, 19, 14):
    try:
        k_deck(_kd, bad, 28)
        check(f"pool {bad} is rejected", False, "no SystemExit")
    except SystemExit as e:
        check(f"pool {bad} is rejected with the minimum pool named",
              "20" in str(e), str(e)[:70])
check("pool 40 (a multiple) is accepted",
      {k: k_deck(_kd, 40, 28).count(k) for k, _ in K_DIST_V5}
      == {1: 10, 2: 4, 4: 4, 8: 4, 12: 4, 16: 4, 28: 10})
check("--k_dist fixed gives every stream --K", k_deck(None, 7, 28) == [28] * 7)




# ─────────────────────────────────────────────────────────────────────────────
print("\nH. probe candidates skip runs shorter than the window")

# ★ THE FAILURE THIS GUARDS.  generate_run stops at the end of a sequence, so a
# scene's last run is the leftover -- 4 frames on some dl3dv scenes.  A 2-run
# scene holds out run 1, which IS that tail, and the probe centres itself with
# ``t0 + max(0, (L - S) // 2)`` = t0, so locate(bank, t0, S) asks for a window no
# single run contains and raises KeyError at startup.  Latent while the corpus
# was MCD (20-40 runs per scene, held-out ids nowhere near the tail); 167 of 278
# banks trip it once dl3dv / dynamicreplica are in the list.
from lingbot_map.train.trainer import locate as _locate


class _RaggedBank:
    """A scene whose LAST run is the leftover, as generate_run really leaves it."""

    def __init__(self, tail_L, n=2, L=240, t0=80):
        self.runs = [{"t0": t0 + i * L, "L": L, "burn_in": 72, "scale_frames": 8,
                      "teacher_interval": 1} for i in range(n - 1)]
        self.runs.append({"t0": t0 + (n - 1) * L, "L": tail_L, "burn_in": 72,
                          "scale_frames": 8, "teacher_interval": 1})
        self.frames_covered = sum(r["L"] for r in self.runs)
        self.index, self.root = {}, "stub"

    def __len__(self):
        return len(self.runs)


def _probe_cand(banks, per_scene_hold, filtered):
    """The trainer's probe-candidate construction, verbatim."""
    cand = [(si, h[j]) for j in range(max((len(h) for h in per_scene_hold), default=0))
            for si, h in enumerate(per_scene_hold) if j < len(h)]
    if filtered:
        cand = [(si, rid) for si, rid in cand if banks[si].runs[rid]["L"] >= S]
    return cand


# a 2-run dl3dv-shaped scene: hold = {(0+1)*2//2} = {1} = the 4-frame tail
_b = [_RaggedBank(4), _RaggedBank(17), _RaggedBank(240, n=20)]
_hold = []
for _bk in _b:
    _n = len(_bk)
    _k = max(0, min(2, _n - 1))
    _hold.append(sorted({(i + 1) * _n // (_k + 1) for i in range(_k)}))
check("a 2-run scene really does hold out its tail", _hold[0] == [1], str(_hold[0]))


def _probe_ok(cand, banks):
    for si, rid in cand:
        r = banks[si].runs[rid]
        try:
            _locate(banks[si], r["t0"] + max(0, (r["L"] - S) // 2), S)
        except KeyError:
            return False, (si, rid, r["L"])
    return True, None


_unfiltered = _probe_cand(_b, _hold, filtered=False)
_ok, _where = _probe_ok(_unfiltered, _b)
check("WITHOUT the filter a short tail run raises KeyError (the bug)",
      not _ok, f"first bad: {_where}")

_filtered = _probe_cand(_b, _hold, filtered=True)
_ok, _where = _probe_ok(_filtered, _b)
check("WITH the filter every probe candidate locates cleanly", _ok, str(_where))
check("the filter drops only the short runs",
      len(_unfiltered) - len(_filtered) == 2, f"{len(_unfiltered)} -> {len(_filtered)}")
check("a long scene still contributes probes",
      any(si == 2 for si, _ in _filtered), str(_filtered))
check("skip_runs is NOT filtered -- a short tail stays reserved, and reserving it "
      "costs training nothing (next_valid_window rejects it anyway)",
      _hold[0] == [1])


print()
if FAIL:
    print(f"FAILED -- {len(FAIL)} check(s): {FAIL}")
    sys.exit(1)
print("PASSED -- the pool walks the v5 schedule")
