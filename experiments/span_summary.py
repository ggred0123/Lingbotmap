"""Recompute the projection in the UNIT-NORM basis and aggregate.

★ pinv's rcond IS RELATIVE TO THE LARGEST SINGULAR VALUE OF WHAT YOU HAND IT.
The raw Gram has |g| spanning 0.16 to 40.6, so its singular values span ~6e4
before any collinearity, and an rcond that looks conservative silently deletes
real axes -- the probe's own console table showed r^2 BELOW cos^2 at the current
weights, which is impossible for a projection.  Rescaling each axis to unit norm
is a pure reparametrisation of its weight: the span, r^2 and the NNLS feasible
set are unchanged, and the matrix to invert becomes the correlation matrix, whose
condition number here is ~5.

    b_hat_i = <g_i, g_t> / (|g_i| |g_t|) = the per-axis cosine already stored
    r^2     = b_hat' corr^+ b_hat
    w_orig  = w_hat * |g_t| / |g_i|
"""
import glob, json, math, sys
import numpy as np
from scipy.optimize import nnls

recs = [json.load(open(f)) for f in sorted(glob.glob(
    sys.argv[1] if len(sys.argv) > 1 else "experiments/results/span_v7f_*.json"))]
AX = recs[0]["axes"]
LOCAL = [a for a in AX if not a.startswith("long_")]
ACTIVE = ["rot", "dir", "depth", "motion"]          # A1PC has lam_mag = 0


def solve(corr, bh, idx):
    """(r2_unconstrained, r2_nnls, w_hat) restricted to the axes in idx."""
    C = corr[np.ix_(idx, idx)]
    b = bh[idx]
    r2 = float(b @ np.linalg.pinv(C, rcond=1e-10) @ b)
    w_, V = np.linalg.eigh(C)
    keep = w_ > max(w_.max(), 1e-30) * 1e-10
    A = np.sqrt(w_[keep])[:, None] * V[:, keep].T
    c = (V[:, keep] / np.sqrt(w_[keep])).T @ b
    w, _ = nnls(A, c)
    r2n = 1 - float(w @ C @ w - 2 * b @ w + 1.0)
    return r2, r2n, w


def med(xs):
    return float(np.median([x for x in xs if x is not None and np.isfinite(x)]))


for tname in recs[0]["targets"]:
    print(f"\n{'='*96}\n  TARGET  {tname}\n{'='*96}")
    print(f"  {'window':<24}{'r (all)':>9}{'r NNLS':>9}{'r now':>9}"
          f"{'r loc':>8}{'r act':>8}{'r act+mag':>11}{'cond':>7}")
    rows = []
    for r in recs:
        t = r["targets"][tname]
        corr = np.array(r["corr"])
        bh = np.array([t["cos_per_axis"][a] for a in AX])
        i_all = list(range(len(AX)))
        i_loc = [AX.index(a) for a in LOCAL]
        i_act = [AX.index(a) for a in ACTIVE]
        i_am = [AX.index(a) for a in ACTIVE + ["mag"]]
        r2, r2n, w = solve(corr, bh, i_all)
        r2l, _, _ = solve(corr, bh, i_loc)
        r2a, _, _ = solve(corr, bh, i_act)
        r2am, _, _ = solve(corr, bh, i_am)
        cond = np.linalg.cond(corr)
        gi = np.sqrt(np.diag(np.array(r["gram"])))
        w_orig = {a: float(w[k] * t["|g_target|"] / gi[k]) for k, a in enumerate(AX)}
        rows.append(dict(scene=r["scene"], t0=r["t0"], K=r["K"], r2=r2, r2n=r2n,
                         cur=t["cos_current"], r2l=r2l, r2a=r2a, r2am=r2am,
                         w=w_orig, bh={a: bh[k] for k, a in enumerate(AX)}))
        print(f"  {r['scene']+' t'+str(r['t0'])+' K'+str(r['K']):<24}"
              f"{math.sqrt(max(0,r2)):>9.3f}{math.sqrt(max(0,r2n)):>9.3f}"
              f"{abs(t['cos_current']):>9.3f}{math.sqrt(max(0,r2l)):>8.3f}"
              f"{math.sqrt(max(0,r2a)):>8.3f}{math.sqrt(max(0,r2am)):>11.3f}{cond:>7.1f}")

    print(f"\n  medians:  r(all axes) {math.sqrt(med([x['r2'] for x in rows])):.3f}"
          f"   r(NNLS, w>=0) {math.sqrt(med([x['r2n'] for x in rows])):.3f}"
          f"   |cos| now {med([abs(x['cur']) for x in rows]):.3f}")
    print(f"            r(local only) {math.sqrt(med([x['r2l'] for x in rows])):.3f}"
          f"   r(A1PC active 4) {math.sqrt(med([x['r2a'] for x in rows])):.3f}"
          f"   r(active+mag) {math.sqrt(med([x['r2am'] for x in rows])):.3f}")

    print(f"\n  per-axis cosine with the target (median over windows)")
    for a in AX:
        v = [x["bh"][a] for x in rows]
        print(f"    {a:<12}{med(v):>+8.3f}   [{min(v):+.3f}, {max(v):+.3f}]"
              f"   sign+ {sum(1 for y in v if y>0)}/{len(v)}")

    print(f"\n  NNLS weights (median; 0 = the term is not used at all)")
    for a in AX:
        v = [x["w"][a] for x in rows]
        print(f"    {a:<12}{med(v):>10.4g}   used in {sum(1 for y in v if y>1e-9)}/{len(v)} windows")
