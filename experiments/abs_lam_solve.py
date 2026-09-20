#!/usr/bin/env python3
"""Turn term_grad_probe --abs_mode measurements into the L_abs weights.
docs/gtabs-plan.md §2-4: the new terms' GRADIENT share is set to ~50% of the
pre-clip update, never their loss share -- clipping fires every step.

Per probe window the probe records ``grad_norm[term]`` = ||dL_term/dtheta|| at
lam = 1.  With A1PC's shipped weights the old terms pull with

    W_old = sum_{i in old} lam_i ||g_i||

and the plan asks for the new terms to pull, together, as hard again:

    gtscale   lam_ts ||g_ts|| + lam_ds ||g_ds|| = W_old,   ts : ds = 2 : 1
              (plan §8: the depth target is the teacher's own depth, which
              drifts inside a run, so depth-scale starts at half the pull of
              the pose-scale term)
    gtpaper   lam_ts, lam_ds AS IN gtscale (plan §2-4: shared terms share lam),
              abs_pos / abs_rot / rel_trans each at depth-scale's pull, i.e.
              ts : ds : ap : ar : rt = 2 : 1 : 1 : 1 : 1.  The old side of
              gtpaper is L_rot alone (dir / motion / median-depth are dropped),
              so its new share comes out well above 50% and is reported, not
              hidden -- the shared-lam rule is the binding one.

Aggregation is the MEDIAN over windows (t0 x K x scene), as solve_lam_star.py
does: one deep window must not write the prescription by itself.

    python3 experiments/abs_lam_solve.py experiments/results/tg_abs_*.json
    -> experiments/results/abs_grad_share.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st

OLD = ["L_rot", "L_dir", "L_mag", "L_motion", "L_depth"]
A1PC = {"L_rot": 15.0, "L_dir": 1.9, "L_mag": 0.0, "L_motion": 0.9, "L_depth": 1.7}
PAPER_OLD = {"L_rot": 15.0, "L_dir": 0.0, "L_mag": 0.0, "L_motion": 0.0, "L_depth": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=sorted(glob.glob("experiments/results/tg_abs_*.json")))
    ap.add_argument("--ts_ds_ratio", type=float, default=2.0, help="trans-scale : depth-scale pull")
    ap.add_argument("--new_share", type=float, default=0.5, help="gtscale's new-term share of the pull")
    ap.add_argument("--out", default="experiments/results/abs_grad_share.json")
    a = ap.parse_args()
    if not a.files:
        raise SystemExit("no term_grad_probe --abs_mode outputs found (tg_abs_*.json)")

    rows = []
    for f in a.files:
        d = json.load(open(f))
        for p in d["probes"]:
            g = p["grad_norm"]
            if "trans_scale" not in g:
                continue
            rows.append({"file": f, "t0": p["t0"], "K": p["K"], "g": g,
                         "fit": p.get("abs", {}).get("fit_mode"), "s": p.get("abs", {}).get("fit_s")})
    if not rows:
        raise SystemExit("no window carries the abs terms -- run term_grad_probe.py --abs_mode scale|paper")
    print(f"windows: {len(rows)}   from {len(a.files)} probe files")

    # ── per-window solution ──
    r_ts = a.ts_ds_ratio / (1 + a.ts_ds_ratio)          # share of the new pull on trans-scale
    frac = a.new_share / (1 - a.new_share)              # new pull as a multiple of the old pull
    per = []
    for r in rows:
        g = r["g"]
        w_old = sum(A1PC[k] * g[k] for k in OLD)
        new_pull = frac * w_old
        lam_ts = r_ts * new_pull / max(g["trans_scale"], 1e-30)
        lam_ds = (1 - r_ts) * new_pull / max(g["depth_scale"], 1e-30)
        rec = {**r, "w_old": w_old, "lam_trans_scale": lam_ts, "lam_depth_scale": lam_ds}
        if "abs_pos" in g:
            unit = (1 - r_ts) * new_pull                # depth-scale's pull
            for k, flag in (("abs_pos", "lam_abs_pos"), ("abs_rot", "lam_abs_rot"), ("rel_trans", "lam_rel_trans")):
                rec[flag] = unit / max(g[k], 1e-30)
        per.append(rec)

    flags = [k for k in per[0] if k.startswith("lam_")]
    med = {k: st.median(r[k] for r in per) for k in flags}
    print(f"\n  {'window':<40} {'W_old':>8} " + " ".join(f"{k[4:]:>12}" for k in flags) + "   fit")
    for r in per:
        name = f"{r['file'].split('/')[-1][:24]} t{r['t0']} K{r['K']}"
        print(f"  {name:<40} {r['w_old']:>8.2f} " + " ".join(f"{r[k]:>12.4g}" for k in flags)
              + f"   {r['fit']} s={r['s']:.3f}" if r['s'] is not None else "")
    print(f"  {'MEDIAN':<40} {'':>8} " + " ".join(f"{med[k]:>12.4g}" for k in flags))

    # ── the shares the median lam actually produce, per window ──
    def shares(lam, old):
        out = []
        for r in per:
            g = r["g"]
            w_old = sum(old[k] * g[k] for k in OLD)
            w_new = sum(lam[k] * g[k[4:]] for k in lam if k[4:] in g)
            out.append(w_new / (w_old + w_new))
        return out
    sc_lam = {k: med[k] for k in ("lam_trans_scale", "lam_depth_scale")}
    sh_scale = shares(sc_lam, A1PC)
    res = {"n_windows": len(per), "per_window": per, "median": med,
           "gtscale": {"lam": sc_lam, "new_share_per_window": sh_scale,
                       "new_share_median": st.median(sh_scale)},
           "rule": {"new_share": a.new_share, "ts_ds_ratio": a.ts_ds_ratio,
                    "paper": "ts:ds:ap:ar:rt = 2:1:1:1:1 with ts, ds shared with gtscale"}}
    print(f"\n  gtscale  --lam_trans_scale {sc_lam['lam_trans_scale']:.4g} --lam_depth_scale {sc_lam['lam_depth_scale']:.4g}"
          f"   -> new-term share median {st.median(sh_scale):.1%} "
          f"(min {min(sh_scale):.1%} max {max(sh_scale):.1%}) against A1PC")
    if "lam_abs_pos" in med:
        pp_lam = {k: med[k] for k in flags}
        sh_paper = shares(pp_lam, PAPER_OLD)
        res["gtpaper"] = {"lam": pp_lam, "new_share_per_window": sh_paper,
                          "new_share_median": st.median(sh_paper)}
        print(f"  gtpaper  " + " ".join(f"--{k} {v:.4g}" for k, v in pp_lam.items())
              + f"   -> new-term share median {st.median(sh_paper):.1%} against L_rot alone")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n[lam] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
