#!/usr/bin/env python3
"""Scene-count ladder read-out -- docs/coverage-corpus-plan.md §2-§3.

Every rung is the same recipe on a corpus of N outdoor-walking scenes; the
hold-outs (Oxford 10 seq, MCD 8 hold-out scenes = 29 seq) are the same
sequences for every rung, so everything is paired per sequence.

    D(N)  = median over hold-out sequences of log ATE(rung) - log ATE(base)
    slope = least-squares b in D = a + b*log2(N), with a sequence-bootstrap CI
            (resample sequences, recompute every D(N), refit)

The plan pre-registers what each pattern may be read as (§2); this script only
computes.  Rungs are given as N=arm[:step]; the default set is the ladder.

    python3 experiments/coverage_ladder_score.py
    python3 experiments/coverage_ladder_score.py --rung 45=rung45:275 --rung 45=rung45_long:2000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from gtabs_report import scene_ate, sign_p, med            # noqa: E402

MCD = os.path.join(ROOT, "experiments", "results", "mcd_distance.json")
OUT_JSON = os.path.join(ROOT, "experiments", "results", "coverage_ladder.json")
OUT_MD = os.path.join(ROOT, "docs", "coverage-result.md")
DEFAULT = ["2=rung2:275", "5=rung5:275", "10=teasup:275", "18=rung18:275", "45=rung45:275"]


def mcd_per_seq(method):
    """seq key -> ATE for one method dir name (e.g. teasup_s275), with bins."""
    r = json.load(open(MCD))
    seqs = r["sequences"]
    meta = dict(seqs) if isinstance(seqs, dict) else {f"{s['scene']}_s{s['stride']}_f{s['start']}": s for s in seqs}
    per = r["per_sequence"].get(method, {})
    out = {}
    for k, v in per.items():
        if "ate" in v and k in meta and meta[k]["holdout"]:
            out[k] = v["ate"]
    return out, meta


def holdout_sets(meta):
    def b(s):
        v = s["m_per_frame"]
        return "near" if v < 0.55 else ("mid" if v < 1.2 else "far")
    sets = {"mcd_all": [k for k, s in meta.items() if s["holdout"]],
            "mcd_far": [k for k, s in meta.items() if s["holdout"] and b(s) == "far"],
            "mcd_kth": [k for k, s in meta.items() if s["holdout"] and s["scene"].startswith("kth")],
            "mcd_ntu": [k for k, s in meta.items() if s["holdout"] and s["scene"].startswith("ntu")]}
    return sets


def dlog(a, base, keys):
    return {k: math.log(a[k]) - math.log(base[k]) for k in keys if k in a and k in base}


def fit_slope(points):
    """points: [(log2N, D)] -> (a, b)"""
    n = len(points)
    if n < 2:
        return None, None
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        return my, None
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", action="append", default=None, help="N=arm[:step] (repeatable)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rungs = []
    for spec in (a.rung or DEFAULT):
        n, _, rest = spec.partition("=")
        arm, _, step = rest.partition(":")
        rungs.append((int(n), arm, int(step or 275)))

    ox_base = scene_ate("base_k1")
    mcd_base, meta = mcd_per_seq("base")
    sets = holdout_sets(meta)
    sets["oxford"] = sorted(ox_base)

    # per rung: per-sequence dlog on each hold-out set
    rows = []
    for n, arm, step in rungs:
        ox = scene_ate(f"sd_{arm}s{step}_k1")
        mc, _ = mcd_per_seq(f"{arm}_s{step}")
        entry = {"N": n, "arm": arm, "step": step, "present": {}, "sets": {}}
        for name, keys in sets.items():
            src, base = (ox, ox_base) if name == "oxford" else (mc, mcd_base)
            d = dlog(src, base, keys)
            entry["present"][name] = len(d)
            if d:
                wins = sum(1 for v in d.values() if v < 0)
                entry["sets"][name] = {"n": len(d), "D": med(d.values()), "wins": wins,
                                       "p_worse": sign_p(len(d) - wins, len(d)), "per_seq": d}
        rows.append(entry)

    # paired vs the N=10 rung (teasup) where both exist
    ref = next((r for r in rows if r["N"] == 10), None)
    if ref:
        for r in rows:
            if r is ref:
                continue
            for name in r["sets"]:
                if name in ref["sets"]:
                    common = set(r["sets"][name]["per_seq"]) & set(ref["sets"][name]["per_seq"])
                    diff = [r["sets"][name]["per_seq"][k] - ref["sets"][name]["per_seq"][k] for k in common]
                    if diff:
                        w = sum(1 for x in diff if x < 0)
                        r["sets"][name]["vs_N10"] = {"n": len(diff), "dD": med(diff), "wins": w, "p": sign_p(w, len(diff))}

    # slope per hold-out set with sequence bootstrap
    rng = random.Random(a.seed)
    slopes = {}
    for name in sets:
        have = [r for r in rows if name in r["sets"]]
        if len(have) < 2:
            continue
        # D per rung on the common sequences only, so the fit is paired
        common = sorted(set.intersection(*[set(r["sets"][name]["per_seq"]) for r in have]))
        if len(common) < 3:
            continue
        pts = [(math.log2(r["N"]), med([r["sets"][name]["per_seq"][k] for k in common])) for r in have]
        _, b = fit_slope(pts)
        bs = []
        for _ in range(a.boot):
            samp = [rng.choice(common) for _ in common]
            p2 = [(math.log2(r["N"]), med([r["sets"][name]["per_seq"][k] for k in samp])) for r in have]
            _, bb = fit_slope(p2)
            if bb is not None:
                bs.append(bb)
        bs.sort()
        ci = (bs[int(0.05 * len(bs))], bs[int(0.95 * len(bs)) - 1]) if bs else (None, None)
        slopes[name] = {"n_seq": len(common), "rungs": [r["N"] for r in have], "D": {r["N"]: y for r, (_, y) in zip(have, pts)},
                        "slope_per_doubling": b, "ci90": ci,
                        "reading": ("down" if ci[1] is not None and ci[1] < 0 else
                                    "up" if ci[0] is not None and ci[0] > 0 else "flat (CI spans 0)")}

    rep = {"rungs": [{k: v for k, v in r.items()} for r in rows], "slopes": slopes,
           "sets": {k: len(v) for k, v in sets.items()}}
    # strip per_seq from the json summary copy but keep it in a side field
    json.dump(rep, open(OUT_JSON, "w"), indent=1, default=float)

    # markdown
    L = ["# 커버리지 사다리 결과 (자동 생성 — experiments/coverage_ladder_score.py)", "",
         "D(N) = hold-out 시퀀스별 log ATE(rung) − log ATE(base)의 중앙값 (양수 = 손상). win = base보다 나은 시퀀스 수. "
         "vs N=10 = 같은 시퀀스에서 teasup(N=10) 대비 ΔD의 중앙값과 부호 검정. 해석 규칙은 docs/coverage-corpus-plan.md §2.", ""]
    names = ["oxford", "mcd_far", "mcd_all", "mcd_kth", "mcd_ntu"]
    L.append("| N | arm@step | " + " | ".join(f"{nm} D / win / vs N=10" for nm in names) + " |")
    L.append("|---|---|" + "---|" * len(names))
    for r in rows:
        cells = []
        for nm in names:
            s = r["sets"].get(nm)
            if not s:
                cells.append("—"); continue
            v = s.get("vs_N10")
            cells.append(f"{s['D']:+.3f} / {s['wins']}/{s['n']}" + (f" / {v['dD']:+.3f} ({v['wins']}/{v['n']})" if v else ""))
        L.append(f"| {r['N']} | {r['arm']}@{r['step']} | " + " | ".join(cells) + " |")
    L += ["", "## 기울기 (D = a + b·log₂N, 시퀀스 bootstrap 90 % CI)", "",
          "| hold-out | n seq | rungs | b per doubling | 90 % CI | 읽기 |", "|---|---|---|---|---|---|"]
    for nm, s in slopes.items():
        ci = s["ci90"]
        L.append(f"| {nm} | {s['n_seq']} | {s['rungs']} | {s['slope_per_doubling']:+.4f} | "
                 f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {s['reading']} |" if ci[0] is not None else
                 f"| {nm} | {s['n_seq']} | {s['rungs']} | — | — | — |")
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
