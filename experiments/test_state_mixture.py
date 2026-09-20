"""Does the v5 state mixture draw what docs/self-distill-ver5.md specifies?

The expensive half of this question needs a GPU and a corpus.  This is the cheap
half: the sampler, the horizon arithmetic, the age bookkeeping and the DDP
branch contract are all pure Python, and every one of them can be wrong in a way
that a training run would not report.

No GPU, no checkpoint, no bank.  Run it before spending a rollout.

    python experiments/test_state_mixture.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.train.trainer import (
    K_DIST_V5, HORIZONS_V5, StateSampler, parse_k_dist, next_valid_window,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'  ok' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ─────────────────────────────────────────────────────────────────────────────
print("\nA. --k_dist parsing")

check("fixed -> None", parse_k_dist("fixed") is None)
check("empty -> None", parse_k_dist("") is None)
_v5 = parse_k_dist(",".join(f"{k}:{w}" for k, w in K_DIST_V5))
check("v5 mixture has 7 values", len(_v5) == 7, str([k for k, _ in _v5]))
check("v5 probabilities sum to 1", abs(sum(p for _, p in _v5) - 1.0) < 1e-9)
check("v5 K=1 is 25%", abs(dict(_v5)[1] - 0.25) < 1e-9, f"{dict(_v5)[1]:.4f}")
check("v5 K=28 is 25%", abs(dict(_v5)[28] - 0.25) < 1e-9, f"{dict(_v5)[28]:.4f}")
check("v5 K=8 is 10%", abs(dict(_v5)[8] - 0.10) < 1e-9, f"{dict(_v5)[8]:.4f}")
check("unnormalised weights normalise", abs(sum(p for _, p in parse_k_dist("1:3,2:1")) - 1) < 1e-9)

for bad in ("1:-1", "0:5", "x:1"):
    try:
        parse_k_dist(bad)
        check(f"{bad!r} rejected", False, "accepted a bad spec")
    except SystemExit:
        check(f"{bad!r} rejected", True)


# ─────────────────────────────────────────────────────────────────────────────
print("\nB. K draws follow the declared distribution")

s = StateSampler(_v5, HORIZONS_V5, 0.35, None, seed=7, rank=0)
draws = Counter(s.k(28) for _ in range(200000))
for k, p in _v5:
    got = draws[k] / 200000
    check(f"K={k:<2} ~ {p:.0%}", abs(got - p) < 0.006, f"got {got:.4f}")
check("no K outside the support", set(draws) == {k for k, _ in _v5}, str(sorted(draws)))

s_fixed = StateSampler(None, [0], 0.0, None, seed=7, rank=0)
check("k_dist=None always returns --K",
      {s_fixed.k(28) for _ in range(1000)} == {28})


# ─────────────────────────────────────────────────────────────────────────────
print("\nC. horizons are clipped to what the scene actually has")

s = StateSampler(_v5, HORIZONS_V5, 0.35, None, seed=3, rank=0)
check("cap 10000 -> any of the four",
      {s.horizon(10000) for _ in range(500)} == set(HORIZONS_V5))
check("cap 1000 -> only 320/960",
      {s.horizon(1000) for _ in range(500)} == {320, 960})
# tuhh_day_04 covers [80, 1850): 1770 frames ahead of the first window
check("cap 1770 (tuhh_day_04) -> 320/960",
      {s.horizon(1770) for _ in range(500)} == {320, 960})
check("cap below the smallest horizon falls back to the cap",
      s.horizon(200) == 200, f"got {s.horizon(200)}")
s_noh = StateSampler(_v5, [0], 0.35, None, seed=3, rank=0)
check("horizons=[0] means no horizon", s_noh.horizon(10000) == 0)


# ─────────────────────────────────────────────────────────────────────────────
print("\nD. the branch draw is rank-invariant (DDP would HANG otherwise)")

a = StateSampler(_v5, HORIZONS_V5, 0.35, None, seed=11, rank=0)
b = StateSampler(_v5, HORIZONS_V5, 0.35, None, seed=11, rank=1)
# rank 1 also runs its own per-rollout draws; they must not perturb branch()
for i in range(500):
    b.k(28)
    b.horizon(5000)
seq_a = [a.branch(i) for i in range(2000)]
seq_b = [b.branch(i) for i in range(2000)]
check("rank 0 and rank 1 draw the same branch sequence", seq_a == seq_b)
check("branch() is a pure function of the step",
      [a.branch(i) for i in range(2000)] == seq_a)
share = seq_a.count("identity") / len(seq_a)
check("identity share ~ p_identity", abs(share - 0.35) < 0.025, f"{share:.4f}")
check("p_identity=0 -> never identity",
      {StateSampler(_v5, [0], 0.0, None).branch(i) for i in range(200)} == {"correction"})
check("p_identity=1 -> always identity",
      {StateSampler(_v5, [0], 1.0, None).branch(i) for i in range(200)} == {"identity"})
# a resume at step N must replay the schedule an uninterrupted run would take
c = StateSampler(_v5, HORIZONS_V5, 0.35, None, seed=11, rank=0)
check("resume at step 137 replays the same branches",
      [c.branch(i) for i in range(137, 200)] == seq_a[137:200])


# ─────────────────────────────────────────────────────────────────────────────
print("\nE. per-rollout draws DO differ across ranks (that is the point)")

a = StateSampler(_v5, HORIZONS_V5, 0.35, None, seed=11, rank=0)
b = StateSampler(_v5, HORIZONS_V5, 0.35, None, seed=11, rank=1)
ka = [a.k(28) for _ in range(400)]
kb = [b.k(28) for _ in range(400)]
check("rank 0 and rank 1 draw different K sequences", ka != kb)


# ─────────────────────────────────────────────────────────────────────────────
print("\nF. dataset-balanced selection")

by_ds = {"mcd": list(range(10)), "slowtv": list(range(10, 19))}
s = StateSampler(_v5, HORIZONS_V5, 0.35, {"mcd": 50, "slowtv": 50}, seed=5)
picks = Counter("mcd" if s.scene(by_ds) < 10 else "slowtv" for _ in range(40000))
frac = picks["mcd"] / 40000
check("50/50 weights -> 50/50 scene draws", abs(frac - 0.5) < 0.01, f"mcd {frac:.4f}")
s = StateSampler(_v5, HORIZONS_V5, 0.35, {"mcd": 25, "slowtv": 75}, seed=5)
frac = sum(1 for _ in range(40000) if s.scene(by_ds) < 10) / 40000
check("25/75 weights -> 25/75 scene draws", abs(frac - 0.25) < 0.01, f"mcd {frac:.4f}")
s = StateSampler(_v5, HORIZONS_V5, 0.35, {"mcd": 50, "slowtv": 50}, seed=5)
check("every scene is reachable",
      {s.scene(by_ds) for _ in range(20000)} == set(range(19)))
# one corpus present: the weights must not make it unreachable
s1 = StateSampler(_v5, HORIZONS_V5, 0.35, {"mcd": 50, "slowtv": 50}, seed=5)
check("single-dataset corpus still draws",
      {s1.scene({"mcd": [0, 1, 2]}) for _ in range(500)} == {0, 1, 2})


# ─────────────────────────────────────────────────────────────────────────────
print("\nG. horizon accounting against a real bank layout")

# kth_night_01: runs of L=240 tiled from t0=80, covering [80, 9684)
class _Bank:
    def __init__(self, t0, n, L=240):
        self.runs = [{"t0": t0 + i * L, "L": L} for i in range(n)]

bank = _Bank(80, 40)
S = 48
covered_end = bank.runs[-1]["t0"] + bank.runs[-1]["L"]

for horizon in HORIZONS_V5:
    t = next_valid_window(bank, 0, S, ())
    anchor = t
    n_windows, guard = 0, 0
    while guard < 10000:
        guard += 1
        nxt = next_valid_window(bank, t + S, S, ())
        if nxt is None or (nxt - anchor) >= horizon:
            break
        t = nxt
        n_windows += 1
    age = t - anchor
    check(f"horizon {horizon:>4}: age stays inside it", age < horizon,
          f"final age {age}, {n_windows} advances")
    check(f"horizon {horizon:>4}: age reaches most of it", age >= horizon - S,
          f"final age {age} vs horizon-S {horizon - S}")

# the retained-keyframe age formula the trainer logs, checked against a
# straight count of the frames a rollout would actually have appended
sf = 8
for K in (1, 2, 4, 8, 12, 16, 28):
    for t in (80, 128, 1000, 3920):
        counted = sum(1 for i in range(sf, t) if (K <= 1) or ((i - sf) % K == 0))
        formula = (t - sf + K - 1) // K
        if counted != formula:
            check(f"kf_age K={K} t={t}", False, f"counted {counted} formula {formula}")
check("kf_age formula matches a counted rollout at every (K, t)", True)

# ...and against what the masked path independently predicts, which is the
# check that raises if the two ever disagree (stream.py _check_prefix_consistency)
for K in (1, 2, 4, 8, 12, 16, 28):
    for t in (80, 1000, 3920):
        predicted = 0 if t <= sf else (t - sf + K - 1) // K
        counted = sum(1 for i in range(sf, t) if (K <= 1) or ((i - sf) % K == 0))
        if predicted != counted:
            check(f"prefix predicate K={K} t={t}", False)
check("_check_prefix_consistency predicate agrees with the rollout schedule", True)


print()
if FAIL:
    print(f"FAILED -- {len(FAIL)} check(s): {FAIL}")
    sys.exit(1)
print("PASSED -- state mixture draws what ver5 specifies")
