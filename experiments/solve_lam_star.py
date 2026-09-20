#!/usr/bin/env python3
"""Turn the per-window lambda* solutions into one set of trainable weights.

term_align_probe.py solves  max_lam cos(sum_i lam_i g_i, g_GT)  per window and
gets lam* proportional to G^-1 b.  Two things have to happen before that can be
handed to the trainer:

  sign      lam* comes out negative for terms whose gradient is anti-aligned
            (L_rot is, at median cos -0.01).  A negative weight MAXIMISES that
            term -- the trainer would drive rotation error up.  Negatives are
            clamped to 0: "stop pulling that way", not "pull the other way".
  scale     cos is scale-free, so lam* is only a direction.  The trainer's step
            size is set by AdamW and --clip 1.0 anyway, so the family is fixed by
            normalising to the shipped total weight: sum(lam) is held at A1PC's
            15.0+1.9+0.0+1.7+0.9 = 19.5, which keeps the update magnitude in the
            regime every previous run was tuned in.

Aggregation is the MEDIAN over windows, not the mean: one window (kth_night_04)
sits at cos 0.75 while the rest are near 0.02-0.09, and a mean would let it
write the prescription by itself.
"""
from __future__ import annotations
import glob, json, statistics as st

TERMS = ["L_rot", "L_dir", "L_mag", "L_depth_si", "L_motion_depth"]
FLAG = {"L_rot": "lam_rot", "L_dir": "lam_dir", "L_mag": "lam_mag",
        "L_depth_si": "lam_depth", "L_motion_depth": "lam_motion"}
SHIPPED = {"lam_rot": 15.0, "lam_dir": 1.9, "lam_mag": 0.0,
           "lam_depth": 1.7, "lam_motion": 0.9}

def main() -> int:
    F = sorted(glob.glob("experiments/results/termalign_*.json"))
    if not F:
        raise SystemExit("no termalign_*.json yet")
    rows = []
    for f in F:
        d = json.load(open(f))
        v = d["views"]["pose_subspace"]
        rows.append((d["scene"], d["t0"], d["K"], v))
    print(f"windows: {len(rows)}")
    med_star = {t: st.median([v["lam_star"][t] for *_, v in rows]) for t in TERMS}
    med_cos  = {t: st.median([v["cos_per_term"][t] for *_, v in rows]) for t in TERMS}
    cos_now  = st.median([v["cos_shipped"] for *_, v in rows])
    cos_max  = st.median([v["cos_max"] for *_, v in rows])
    print(f"cos shipped {cos_now:+.4f}   ceiling {cos_max:+.4f}   "
          f"({cos_max/cos_now:.1f}x)" if cos_now else "")
    print(f"\n  {'term':>16} {'median cos':>12} {'median lam*':>12} {'clamped':>9}")
    clamped = {}
    for t in TERMS:
        c = max(0.0, med_star[t])
        clamped[t] = c
        print(f"  {t:>16} {med_cos[t]:>+12.4f} {med_star[t]:>+12.3f} {c:>9.3f}")
    s = sum(clamped.values())
    if s <= 0:
        raise SystemExit("every lambda* clamped to zero -- nothing to train")
    total = sum(SHIPPED.values())                       # 19.5
    train = {FLAG[t]: round(clamped[t] / s * total, 4) for t in TERMS}
    print(f"\n  normalised so sum(lam) = {total} (A1PC's own total)")
    print(f"  {'flag':>12} {'shipped':>9} {'new':>9}")
    for k in SHIPPED:
        print(f"  {k:>12} {SHIPPED[k]:>9.2f} {train[k]:>9.3f}")
    out = {"n_windows": len(rows),
           "cos_shipped_median": cos_now, "cos_ceiling_median": cos_max,
           "median_cos_per_term": med_cos, "median_lam_star": med_star,
           "train_lambdas": train,
           "windows": [{"scene": s_, "t0": t_, "K": k_,
                        "cos_shipped": v["cos_shipped"], "cos_max": v["cos_max"]}
                       for s_, t_, k_, v in rows]}
    json.dump(out, open("experiments/results/lam_star.json", "w"), indent=1)
    print("\n[solve] wrote experiments/results/lam_star.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
