#!/usr/bin/env python3
"""Everything docs/gtabs-plan.md §4 and §7 ask for, read back from what exists.

  §4    exposure table per arm: ranks, streams, K deck, restarts, branch split,
        and for the treated arms the fit-mode histogram (fallback share)
  §7-2  Q-base  paired dlog ATE vs base_k1 per scene, wins, sign-test p
        Q-gt    paired dlog ATE vs gtctrl at the same step, wins, sign-test p
        in-domain GT probe (gt_ate every 75 steps)
        channels from drift_decomposition.json: wscale_std, depth_drift,
        depth_range, ratio_drift, ratio_range (+ rot1, absrot, dir1)
  §7-4  learning-side residuals: L_trans-scale / L_depth-scale of the
        correction windows by offset band x step segment, identity separately
  §7-3  the verdict row the numbers fall into

Nothing here is hand-edited: rerun after every scoring pass.

    python3 experiments/gtabs_report.py [--arms gtctrl gtscale gtpaper]
    -> experiments/results/gtabs_report.json, docs/gtabs-result.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
from datetime import datetime
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford/oxford"
RES = os.path.join(ROOT, "experiments", "results")
LOGS = os.path.join(ROOT, "experiments", "logs")
GRID = [25, 50, 75, 100, 150, 200, 250, 275]
CHANNELS = ["wscale_std", "depth_drift", "depth_range", "ratio_drift", "ratio_range"]
EXTRA = ["ate", "rot1", "rot48", "absrot", "dir1", "scale_drift"]
NOISE_ATE_M, NOISE_AUC = 0.118, 1.42          # plan §7-2 / objective-diagnosis §02


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return st.median(xs) if xs else None


def sign_p(wins, n):
    """two-sided exact sign test P(X >= wins) for X ~ Bin(n, 1/2), one-sided
    'better' tail -- the plan quotes 8/10 -> 0.055, 9/10 -> 0.011"""
    if n == 0:
        return None
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


# ── §4 exposure ──────────────────────────────────────────────────────────────

def exposure(arm):
    out = {"arm": arm}
    lp = os.path.join(LOGS, f"train_{arm}.log")
    if os.path.exists(lp):
        txt = open(lp, errors="ignore").read()
        out["ranks"] = 2 if "[ddp] rank" in txt else 1
        out["ddp_lines"] = re.findall(r"\[ddp\] rank \d/\d takes \[[^\]]*\]", txt)[:4]
        m = re.findall(r"\[pool\] (\d+) streams", txt)
        out["pool_streams"] = [int(x) for x in m]
        m = re.findall(r"\[v5\] K deck over (\d+) streams: ([^\n]*?) +-- rank", txt)
        out["k_deck"] = m[0][1].strip() if m else None
        out["resumes"] = re.findall(r"\[resume\] \S+ -- step (\d+)", txt)
        out["launches"] = txt.count("[pool] ")
        out["oom_kills"] = txt.count("exitcode  : -9")
        m = re.findall(r"\[abs\] ([^\n]*)", txt)
        out["abs_line"] = m[0] if m else None
    j = load(os.path.join(RES, f"train_{arm}.json"))
    if j:
        steps = merged_steps(arm)
        out["steps_in_json"] = len(j.get("steps", []))
        cor = [r for r in steps if r.get("branch") != "identity"]
        out["steps"] = len(steps)
        out["correction_steps"] = len(cor)
        out["identity_steps"] = len(steps) - len(cor)
        out["wall_s"] = j.get("wall_s")
        out["finished"] = bool(j.get("wall_s"))
        # wall_s covers the LAST launch only, so divide by the steps that
        # launch logged (the json's own), not by the merged total
        out["s_per_step"] = (j["wall_s"] / max(len(j.get("steps", [])), 1)) if j.get("wall_s") else None
        kc = {}
        for r in cor:
            kc[int(r["K"])] = kc.get(int(r["K"]), 0) + 1
        out["k_realized"] = dict(sorted(kc.items()))
        _modes = ("prefix", "hist", "span", "fallback", "identity", "skipped")
        fm = {}
        for r in cor:
            m = r.get("fit_mode") or (_modes[int(r["fit_mode_id"])] if "fit_mode_id" in r else None)
            if m:
                fm[m] = fm.get(m, 0) + 1
        out["fit_modes"] = fm
        out["fallback_share"] = (fm.get("fallback", 0) / max(sum(fm.values()), 1)) if fm else None
        fp = [r["fit_path"] for r in cor if "fit_path" in r and r.get("fit_mode") in ("prefix", "hist", "span")]
        out["fit_path_median"] = med(fp)
        out["fit_s_median"] = med([r["fit_s"] for r in cor if "fit_s" in r and r.get("fit_mode") != "identity"])
        out["probes"] = [{"step": p["step"], "gt_ate": p.get("gt_ate_mean"), "gt_rot": p.get("gt_rot_mean"),
                          "loss": p.get("mean")} for p in j.get("probes", [])]
        out["max_step_logged"] = max((r["step"] for r in steps), default=-1)
    ck = sorted(int(re.search(r"step(\d+)", f).group(1)) for f in os.listdir(os.path.join(ROOT, "ckpt_train"))
                if f.startswith(f"{arm}.step"))
    out["ckpts"] = ck
    return out


# ── §7-2 Oxford ──────────────────────────────────────────────────────────────

def scene_ate(method):
    out = {}
    if not os.path.isdir(WS):
        return out
    for sc in sorted(os.listdir(WS)):
        j = load(os.path.join(WS, sc, "eval", "traj.json"))
        if j and method in j and j[method].get("ate") is not None:
            out[sc] = j[method]["ate"]
    return out


def paired(a, b):
    common = sorted(set(a) & set(b))
    dl = [math.log(a[s]) - math.log(b[s]) for s in common]
    wins = sum(1 for x in dl if x < 0)
    return {"n": len(dl), "dlog_median": med(dl), "wins": wins, "p": sign_p(wins, len(dl)),
            "per_scene": {s: round(x, 4) for s, x in zip(common, dl)}}


def oxford_table(arms):
    auc = load(os.path.join(WS, "eval", "auc_macro.json")) or {}
    base = scene_ate("base_k1")
    tab = {"base_k1": {"ate_mean": (sum(base.values()) / len(base)) if base else None,
                       "ate_median": med(base.values()), "auc03": (auc.get("base_k1") or {}).get("AUC_03"),
                       "n": len(base)}}
    for arm in arms:
        tab[arm] = {}
        for s in GRID:
            m = f"sd_{arm}s{s}_k1"
            a = scene_ate(m)
            if not a:
                continue
            row = {"ate_mean": sum(a.values()) / len(a), "ate_median": med(a.values()), "n": len(a),
                   "auc03": (auc.get(m) or {}).get("AUC_03"),
                   "vs_base": paired(a, base)}
            if arm != "gtctrl":
                c = scene_ate(f"sd_gtctrls{s}_k1")
                if c:
                    row["vs_gtctrl"] = paired(a, c)
            tab[arm][str(s)] = row
    return tab


# ── channels ─────────────────────────────────────────────────────────────────

def channel_table(arms):
    d = load(os.path.join(RES, "drift_decomposition.json"))
    if not d:
        return {}
    per = d.get("per_scene", {})
    brows = {sc: r for sc, r in per.get("base_k1", {}).items() if "error" not in r}
    out = {"base": {k: med([r.get(k) for r in brows.values()]) for k in CHANNELS + EXTRA}}
    out["base"]["n"] = len(brows)
    for arm in arms:
        out[arm] = {}
        for s in GRID:
            m = f"sd_{arm}s{s}_k1"
            if m not in per:
                continue
            rows = {sc: r for sc, r in per[m].items() if "error" not in r}
            row = {k: med([r.get(k) for r in rows.values()]) for k in CHANNELS + EXTRA}
            row["n"] = len(rows)
            if arm != "gtctrl":
                c = per.get(f"sd_gtctrls{s}_k1", {})
                cmp = {}
                for k in CHANNELS + ["ate", "absrot", "rot1"]:
                    common = [sc for sc in rows if sc in c and k in rows[sc] and k in c[sc]]
                    if not common:
                        continue
                    wins = sum(1 for sc in common if rows[sc][k] < c[sc][k])
                    ratio = med([rows[sc][k] / c[sc][k] if c[sc][k] else None for sc in common])
                    cmp[k] = {"n": len(common), "wins": wins, "p": sign_p(wins, len(common)),
                              "ratio_median": ratio}
                row["vs_gtctrl"] = cmp
            out[arm][str(s)] = row
    return out


# ── §7-4 learning side ───────────────────────────────────────────────────────

def merged_steps(arm):
    """the json's own steps, plus wandb rows (gtabs_wandb_fill.py) for steps
    the json lost across a resume"""
    j = load(os.path.join(RES, f"train_{arm}.json"))
    steps = list(j.get("steps", [])) if j else []
    have = {r["step"] for r in steps}
    w = load(os.path.join(RES, f"train_{arm}_wandb.json")) or []
    for r in w:
        if r["step"] not in have:
            r = dict(r)
            ident = r.get("is_identity")
            if ident is None:
                ident = r.get("fit_mode_id") == 4.0          # FIT_MODES.index("identity")
            r["branch"] = "identity" if ident else "correction"
            steps.append(r)
    steps.sort(key=lambda r: r["step"])
    return steps


def learning_side(arm):
    steps = merged_steps(arm)
    if not steps:
        return None
    segs = [(0, 100), (100, 200), (200, 300)]
    bands = [0, 48, 96, 144, 192]
    out = {"correction": {}, "identity": {}, "n": len(steps)}
    for lo, hi in segs:
        key = f"{lo}-{hi}"
        for branch in ("correction", "identity"):
            rows = [r for r in steps if lo <= r["step"] < hi and (r.get("branch") != "identity") == (branch == "correction")
                    and "L_trans_scale" in r]
            if not rows:
                continue
            b = {}
            for off in bands:
                rr = [r for r in rows if int(r.get("abs_offset", -1)) == off]
                if rr:
                    b[str(off)] = {"n": len(rr),
                                   "trans_resid": med([r.get("abs_trans_resid") for r in rr]),
                                   "trans_bias": med([r.get("abs_trans_bias") for r in rr]),
                                   "L_trans_scale": med([r.get("L_trans_scale") for r in rr]),
                                   "L_depth_scale": med([r.get("L_depth_scale") for r in rr]),
                                   "depth_bias": med([r.get("abs_depth_bias") for r in rr]),
                                   "fit_s": med([r.get("fit_s") for r in rr]),
                                   "fallback": sum(1 for r in rr if r.get("fit_mode") == "fallback"
                                                   or r.get("fit_mode_id") == 3.0)}
            out[branch][key] = b
    return out


# ── §7-3 verdict ─────────────────────────────────────────────────────────────

def verdict(arm, ox, ch, step="275"):
    row = (ox.get(arm) or {}).get(step, {})
    chrow = (ch.get(arm) or {}).get(step, {})
    vg = row.get("vs_gtctrl")
    if not vg or not chrow.get("vs_gtctrl"):
        return {"status": "incomplete", "why": "missing gtctrl-paired scores at step " + step}
    cmp = chrow["vs_gtctrl"]
    scale_keys = ["wscale_std", "depth_drift", "ratio_drift"]
    scale_lower = all(k in cmp and (cmp[k]["wins"] >= 9 or (cmp[k]["ratio_median"] is not None and cmp[k]["ratio_median"] <= 0.5))
                      for k in scale_keys)
    scale_noise = all(k in cmp and 3 <= cmp[k]["wins"] <= 7 for k in scale_keys)
    ate_win = vg["wins"] >= 8
    ate_noise = 3 <= vg["wins"] <= 7 and abs(vg["dlog_median"] or 0) < 0.05
    if scale_lower and ate_win:
        v = ("효과 있음", "H1 기여 시사(선별 통과) -> §10 확정 실험")
    elif scale_lower and not ate_win:
        v = ("혼합", "scale 채널은 잡히고 손상은 다른 채널 -> gtpaper 우선, D")
    elif scale_noise and ate_noise:
        v = ("효과 없음", "run 규모 감독은 이 체제에서 기여 없음 -> §7-4로 못 배움/전이 안 됨 분리")
    else:
        v = ("미정", "판정표의 세 행 어디에도 깨끗이 들어가지 않음 -- 노이즈 바닥 2배 미만이면 미정")
    return {"status": v[0], "next": v[1], "scale_lower": scale_lower, "ate_win_ge8": ate_win,
            "ate_wins": vg["wins"], "ate_dlog": vg["dlog_median"],
            "channels": {k: cmp.get(k) for k in scale_keys}}


# ── markdown ─────────────────────────────────────────────────────────────────

def f3(x, sign=False):
    if x is None:
        return "—"
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def markdown(rep):
    L = [f"# gtabs — 결과 (자동 생성 {rep['generated']})", "",
         "`experiments/gtabs_report.py`가 벤치·분해·학습 json에서 읽어 만든다. 손으로 고치지 않는다.", "",
         "## 노출 표 (§4)", "",
         "| arm | ranks | streams | K deck | launches | resumes | OOM | steps (cor/id) | s/step | fit modes | fallback | fit_s med | finished |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, e in rep["exposure"].items():
        L.append(f"| {arm} | {e.get('ranks','—')} | {e.get('pool_streams','—')} | {e.get('k_deck','—')} | "
                 f"{e.get('launches','—')} | {e.get('resumes','—')} | {e.get('oom_kills','—')} | "
                 f"{e.get('steps','—')} ({e.get('correction_steps','—')}/{e.get('identity_steps','—')}) | "
                 f"{f3(e.get('s_per_step'))} | {e.get('fit_modes') or '—'} | "
                 f"{(f'{e['fallback_share']:.1%}' if e.get('fallback_share') is not None else '—')} | "
                 f"{f3(e.get('fit_s_median'))} | {e.get('finished','—')} |")
    L += ["", "## Oxford K=1 (§7-2)", "",
          f"base_k1: ATE mean {f3(rep['oxford']['base_k1']['ate_mean'])} m, median {f3(rep['oxford']['base_k1']['ate_median'])} m, "
          f"AUC_03 {f3(rep['oxford']['base_k1']['auc03'])}. 노이즈 바닥 |ΔATE| {NOISE_ATE_M} m, AUC_03 ±{NOISE_AUC}.", ""]
    for arm, rows in rep["oxford"].items():
        if arm == "base_k1" or not rows:
            continue
        L += [f"### {arm}", "",
              "| step | ATE mean | ATE med | AUC_03 | Δlog vs base (med) | win/base | p | Δlog vs gtctrl | win/gtctrl | p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s, r in rows.items():
            vb, vg = r["vs_base"], r.get("vs_gtctrl")
            L.append(f"| {s} | {f3(r['ate_mean'])} | {f3(r['ate_median'])} | {f3(r['auc03'])} | {f3(vb['dlog_median'], True)} | "
                     f"{vb['wins']}/{vb['n']} | {f3(vb['p'])} | "
                     + (f"{f3(vg['dlog_median'], True)} | {vg['wins']}/{vg['n']} | {f3(vg['p'])} |" if vg else "— | — | — |"))
        L.append("")
    L += ["## 채널 분해 (drift_decomposition, 10 scene 중앙값)", ""]
    if rep["channels"]:
        cols = CHANNELS + ["rot1", "absrot", "dir1"]
        L += ["| arm | step | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        b = rep["channels"].get("base", {})
        L.append("| base | 0 | " + " | ".join(f3(b.get(c)) for c in cols) + " |")
        for arm, rows in rep["channels"].items():
            if arm == "base":
                continue
            for s, r in rows.items():
                L.append(f"| {arm} | {s} | " + " | ".join(f3(r.get(c)) for c in cols) + " |")
        L.append("")
        for arm, rows in rep["channels"].items():
            if arm in ("base", "gtctrl"):
                continue
            for s, r in rows.items():
                if "vs_gtctrl" in r and r["vs_gtctrl"]:
                    L.append(f"{arm} @ {s} vs gtctrl (paired, scene별): " + ", ".join(
                        f"{k} win {v['wins']}/{v['n']} ratio {f3(v['ratio_median'])}" for k, v in r["vs_gtctrl"].items()))
        L.append("")
    L += ["## in-domain GT probe", ""]
    for arm, e in rep["exposure"].items():
        if e.get("probes"):
            L.append(f"- {arm}: " + "  ".join(f"@{p['step']} ate {f3(p['gt_ate'])} rot {f3(p['gt_rot'])}" for p in e["probes"] if p.get("gt_ate") is not None))
    L += ["", "## 학습 측 잔차 (§7-4) — correction 창, offset 밴드 × step 구간 중앙값", ""]
    for arm, ls in rep["learning"].items():
        if not ls:
            continue
        for branch in ("correction", "identity"):
            if not ls[branch]:
                continue
            L += [f"### {arm} / {branch}", "",
                  "| step seg | offset | n | trans resid | trans bias | L_depth-scale | depth bias | fit_s | fallback |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for seg, bands in ls[branch].items():
                for off, v in bands.items():
                    L.append(f"| {seg} | {off} | {v['n']} | {f3(v['trans_resid'])} | {f3(v['trans_bias'], True)} | "
                             f"{f3(v['L_depth_scale'])} | {f3(v['depth_bias'], True)} | {f3(v['fit_s'])} | {v['fallback']} |")
            L.append("")
    L += ["## 판정 (§7-3)", ""]
    for arm, v in rep["verdict"].items():
        L.append(f"- **{arm}**: {v.get('status')} — {v.get('next', v.get('why', ''))}"
                 + (f" (ATE win {v['ate_wins']}/10, Δlog {f3(v['ate_dlog'], True)}; scale channels lower: {v['scale_lower']})"
                    if "ate_wins" in v else ""))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["gtctrl", "gtscale", "gtpaper"])
    ap.add_argument("--md", default=os.path.join(ROOT, "docs", "gtabs-result.md"))
    a = ap.parse_args()
    arms = [x for x in a.arms if os.path.exists(os.path.join(LOGS, f"train_{x}.log"))]
    rep = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "exposure": {arm: exposure(arm) for arm in arms},
           "oxford": oxford_table(arms), "channels": channel_table(arms),
           "learning": {arm: learning_side(arm) for arm in arms}}
    rep["verdict"] = {arm: verdict(arm, rep["oxford"], rep["channels"]) for arm in arms if arm != "gtctrl"}
    json.dump(rep, open(os.path.join(RES, "gtabs_report.json"), "w"), indent=1, default=float)
    open(a.md, "w").write(markdown(rep))
    print(markdown(rep))
    print(f"[report] wrote {a.md} and experiments/results/gtabs_report.json")


if __name__ == "__main__":
    main()
