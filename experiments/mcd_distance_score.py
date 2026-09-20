#!/usr/bin/env python3
"""Score the MCD distance ladder (mcd_distance_ladder.py) against GT and split
the Oxford question: distance regime or scene domain?

Per sequence the same decomposition as drift_decomposition.py -- ATE after one
global Sim(3), per-step rotation / direction / scale ratio, the windowed scale
against the global one, and the depth channel RELATIVE TO BASE on the same
frames -- then medians per (method, speed bin, campus) and paired dlog ATE
against base_k1's role here (the released checkpoint streamed the same way)
and against gtctrl at the same step.

    speed bins   near < 0.55 m/frame (the training regime and NTU's native
                 speed), mid 0.55-1.2, far >= 1.2 (Oxford is 1.6)
    campus       kth = same campus as the training scenes, ntu = a different one

    python3 experiments/mcd_distance_score.py
    -> experiments/results/mcd_distance.json, docs/gtabs-distance.md
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics as st
import sys
from datetime import datetime
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from bake_gt_bank import gt_rows_for                                    # noqa: E402
from drift_decomposition import geo_deg, rel_rot, umeyama, windows, W     # noqa: E402
from mcd_gt import quat_to_mat                                          # noqa: E402
from mcd_distance_ladder import TRAIN                                   # noqa: E402

DIST = os.path.join(ROOT, "experiments", "results", "mcd_dist")
CALIB, SENSOR = os.path.join(ROOT, "data/mcd/calib/hhs_calib.yaml"), "d455b_color"
BINS = [("near", 0.0, 0.55), ("mid", 0.55, 1.2), ("far", 1.2, 99.0)]
KEYS = ["ate", "s_glob", "rot1", "rot48", "absrot", "dir1", "scale_breath", "scale_drift",
        "scale_end", "scale_range", "wscale_std", "wscale_range",
        "depth_breath", "depth_drift", "depth_end", "depth_range",
        "ratio_breath", "ratio_drift", "ratio_end", "ratio_range"]
_gt_cache = {}


def gt_for(scene):
    if scene not in _gt_cache:
        fd = os.path.join(ROOT, "data", "mcd", scene, "frames_10hz")
        _gt_cache[scene] = gt_rows_for(fd, CALIB, SENSOR)
    return _gt_cache[scene]


def analyse(pe, dm, cg, Rg, pe_b, dm_b):
    """same body as drift_decomposition.analyse, on arrays.  ``pe`` [N, 9]
    pose_enc (c2w centre + XYZW quaternion), ``dm`` per-frame depth medians,
    ``cg``/``Rg`` GT centres and rotations, ``pe_b``/``dm_b`` the base model's
    on the same frames (depth is only meaningful relative to base)."""
    cp, Rp = pe[:, :3].astype(np.float64), quat_to_mat(pe[:, 3:7].astype(np.float64))
    n = min(len(cp), len(cg))
    cp, Rp, cg, Rg = cp[:n], Rp[:n], cg[:n], Rg[:n]
    s, Ra, ta = umeyama(cp, cg)
    al = (s * (Ra @ cp.T)).T + ta
    out = {"ate": float(np.sqrt(((al - cg) ** 2).sum(1).mean())), "s_glob": float(s)}
    out["rot1"] = float(geo_deg(rel_rot(Rp, 1), rel_rot(Rg, 1)).mean())
    k = min(W, n - 1)
    out["rot48"] = float(geo_deg(rel_rot(Rp, k), rel_rot(Rg, k)).mean())
    out["absrot"] = float(geo_deg(np.einsum("ij,njk->nik", Ra, Rp), Rg).mean())
    dp, dg = al[1:] - al[:-1], cg[1:] - cg[:-1]
    np_, ng = np.linalg.norm(dp, axis=1), np.linalg.norm(dg, axis=1)
    ok = ng > 1e-3
    cosd = np.clip((dp * dg).sum(1) / np.maximum(np_ * ng, 1e-12), -1, 1)
    out["dir1"] = float(np.degrees(np.arccos(cosd[ok])).mean())
    lrho = np.where(ok, np.log(np.maximum(np_, 1e-12)) - np.log(np.maximum(ng, 1e-12)), np.nan)
    wins = windows(n - 1)

    def win_stats(x):
        inside = [np.nanstd(x[a:b]) for a, b in wins]
        meds = [np.nanmedian(x[a:b]) for a, b in wins]
        return (float(np.nanmean(inside)), float(np.nanstd(meds)),
                float(meds[-1] - meds[0]), float(np.nanmax(meds) - np.nanmin(meds)))
    out["scale_breath"], out["scale_drift"], out["scale_end"], out["scale_range"] = win_stats(lrho)
    ls = []
    for a, b in windows(n):
        sw, _, _ = umeyama(cp[a:b], cg[a:b])
        ls.append(np.log(max(sw, 1e-12) / max(s, 1e-12)))
    out["wscale_std"], out["wscale_range"] = float(np.std(ls)), float(np.max(ls) - np.min(ls))
    if dm_b is not None:
        m = min(len(dm), len(dm_b), n)
        ldelta = np.log(dm[:m]) - np.log(dm_b[:m])
        out["depth_breath"], out["depth_drift"], out["depth_end"], out["depth_range"] = win_stats(ldelta)
        cb = pe_b[:m, :3].astype(np.float64)
        sb = np.linalg.norm(cb[1:] - cb[:-1], axis=1)
        sp_ = np.linalg.norm(cp[1:m] - cp[:m - 1], axis=1)
        okb = (sb > 1e-6) & (sp_ > 1e-6)
        lpi = np.where(okb, np.log(np.maximum(sp_, 1e-12)) - np.log(np.maximum(sb, 1e-12)), np.nan)
        lr = lpi[:m - 1] - 0.5 * (ldelta[:m - 1] + ldelta[1:m])
        out["ratio_breath"], out["ratio_drift"], out["ratio_end"], out["ratio_range"] = win_stats(lr)
    return out


def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return st.median(xs) if xs else None


def sign_p(wins, n):
    return (sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n) if n else None


def main():
    seqs = json.load(open(os.path.join(DIST, "sequences.json")))
    methods = sorted(d for d in os.listdir(DIST) if os.path.isdir(os.path.join(DIST, d)))
    if "base" not in methods:
        raise SystemExit("base results missing")

    def key(s):
        return f"{s['scene']}_s{s['stride']}_f{s['start']}"

    def load(m, s):
        p = os.path.join(DIST, m, key(s) + ".npz")
        return np.load(p) if os.path.exists(p) else None

    per = {}          # method -> seq key -> metrics
    for m in methods:
        per[m] = {}
        for s in seqs:
            z = load(m, s)
            if z is None:
                continue
            zb = load("base", s)
            rows, gp, gq = gt_for(s["scene"])
            idx = rows[np.asarray(s["frames"])]
            cg, Rg = gp[idx], quat_to_mat(gq[idx])
            try:
                per[m][key(s)] = analyse(z["pose_enc"], z["depth_med"], cg, Rg,
                                         zb["pose_enc"] if zb is not None else None,
                                         zb["depth_med"] if zb is not None else None)
            except Exception as e:                      # keep going, report
                per[m][key(s)] = {"error": repr(e)}
    meta = {key(s): s for s in seqs}

    def bin_of(s):
        v = s["m_per_frame"]
        return next(b for b, lo, hi in BINS if lo <= v < hi)

    def campus(s):
        # 'train' = one of the ten training scenes (any distance is then a
        # test of DISTANCE alone); 'kth' = held-out scene on the training
        # campus; 'ntu' = held-out scene on another campus
        return "train" if s["scene"] in TRAIN else s["scene"].split("_")[0]

    # ── aggregate ──
    table = {}
    for m in methods:
        arm, _, step = m.partition("_s")
        ctrl = f"gtctrl_s{step}" if arm != "gtctrl" and step else None
        for b, _, _ in BINS:
            for camp in ("train", "kth", "ntu", "holdout"):
                ks = [k for k, s in meta.items() if bin_of(s) == b and
                      (campus(s) == camp or (camp == "holdout" and campus(s) != "train"))
                      and k in per[m] and "error" not in per[m][k]]
                if not ks:
                    continue
                row = {k2: med([per[m][k][k2] for k in ks if k2 in per[m][k]]) for k2 in KEYS}
                row["n"] = len(ks)
                row["m_per_frame"] = med([meta[k]["m_per_frame"] for k in ks])
                row["path_m"] = med([meta[k]["path_m"] for k in ks])
                if m != "base":
                    common = [k for k in ks if k in per["base"] and "error" not in per["base"][k]]
                    dl = [math.log(per[m][k]["ate"]) - math.log(per["base"][k]["ate"]) for k in common]
                    row["vs_base"] = {"n": len(dl), "dlog_median": med(dl),
                                      "wins": sum(1 for x in dl if x < 0), "p": sign_p(sum(1 for x in dl if x < 0), len(dl))}
                if ctrl and ctrl in per:
                    common = [k for k in ks if k in per[ctrl] and "error" not in per[ctrl][k]]
                    dl = [math.log(per[m][k]["ate"]) - math.log(per[ctrl][k]["ate"]) for k in common]
                    cw = {}
                    for k2 in ("depth_drift", "wscale_std", "ratio_drift", "absrot"):
                        cw[k2] = sum(1 for k in common if per[m][k].get(k2, 0) < per[ctrl][k].get(k2, 0))
                    row["vs_gtctrl"] = {"n": len(dl), "dlog_median": med(dl),
                                        "wins": sum(1 for x in dl if x < 0), "channel_wins": cw}
                table.setdefault(m, {}).setdefault(b, {})[camp] = row
    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "bins": BINS, "table": table,
           "per_sequence": per, "sequences": meta}
    json.dump(res, open(os.path.join(ROOT, "experiments", "results", "mcd_distance.json"), "w"),
              indent=1, default=float)

    # ── report ──
    def f(x, d=3, sign=False):
        return "—" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")
    L = [f"# MCD 거리 사다리 — 거리 체제인가 장면 도메인인가 (자동 생성 {res['generated']})", "",
         "해석은 `docs/gtabs-plan.md` §13. `experiments/mcd_distance_ladder.py` + `mcd_distance_score.py`. 학습 10 scene + hold-out MCD 8 scene(kth 2 = 학습과 같은 캠퍼스, ntu 6 = 다른 캠퍼스), "
         "320 프레임 스트리밍 K=1(Oxford 프로토콜), GT 대비. 속도 구간: near < 0.55 m/frame(학습 체제 0.16, ntu 원속도 0.35~0.48), mid 0.55~1.2, far ≥ 1.2(Oxford 1.6). "
         "depth·ratio 채널은 같은 프레임의 base 대비.", ""]
    for camp in ("train", "holdout", "kth", "ntu"):
        L += [f"## scenes = {camp}" + {"train": " (학습에 쓴 10 scene — 거리만의 시험)", "holdout": " (hold-out 8 scene, 두 캠퍼스 합산)",
                                     "kth": " (hold-out, 학습과 같은 캠퍼스)", "ntu": " (hold-out, 다른 캠퍼스)"}[camp], "",
              "| method | bin | n | m/frame | path m | ATE | Δlog vs base | win | Δlog vs gtctrl | win | s_glob | wscale_std | scale_drift | depth_drift | depth_end | ratio_drift | rot1 | absrot | dir1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        order = ["base"] + [m for m in methods if m != "base"]
        for m in order:
            for b, _, _ in BINS:
                r = table.get(m, {}).get(b, {}).get(camp)
                if not r:
                    continue
                vb, vg = r.get("vs_base"), r.get("vs_gtctrl")
                wb = f"{vb['wins']}/{vb['n']}" if vb else "—"
                wg = f"{vg['wins']}/{vg['n']}" if vg else "—"
                L.append(f"| {m} | {b} | {r['n']} | {f(r['m_per_frame'],2)} | {f(r['path_m'],0)} | {f(r['ate'])} | "
                         f"{f(vb['dlog_median'],3,True) if vb else '—'} | {wb} | "
                         f"{f(vg['dlog_median'],3,True) if vg else '—'} | {wg} | "
                         f"{f(r['s_glob'])} | {f(r['wscale_std'])} | {f(r['scale_drift'])} | {f(r['depth_drift'])} | {f(r['depth_end'])} | "
                         f"{f(r['ratio_drift'])} | {f(r['rot1'])} | {f(r['absrot'],2)} | {f(r['dir1'],1)} |")
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(ROOT, "docs", "gtabs-distance.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
