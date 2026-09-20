#!/usr/bin/env python3
"""Insert (or refresh) the gtabs section of docs/results-ledger.html from
experiments/results/gtabs_report.json.  Idempotent: the section sits between
two marker comments and is replaced in place.

    python3 experiments/gtabs_report.py && python3 experiments/gtabs_ledger_section.py
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "results-ledger.html")
REPORT = os.path.join(ROOT, "experiments", "results", "gtabs_report.json")
BEGIN, END = "<!-- gtabs:begin -->", "<!-- gtabs:end -->"


def f(x, d=3, sign=False):
    if x is None:
        return "&mdash;"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def main():
    rep = json.load(open(REPORT))
    ox, ch, ex, ver = rep["oxford"], rep["channels"], rep["exposure"], rep["verdict"]
    L = [BEGIN, '<h2><span class="num">06</span>gtabs — run 규모 공통 scale 감독 (2026-09-16)</h2>',
         f'<p class="small">docs/gtabs-plan.md의 1차 선별. 자동 생성 {rep["generated"]} · 원본 표는 <code>docs/gtabs-result.md</code>. '
         'gtctrl = A1PC 대조군(1 rank), gtscale = + L_trans-scale·L_depth-scale(run 게이지, prefix fit), gtpaper = 논문 Eq.1의 run-게이지 판. '
         'Oxford K=1 10 scene, base_k1 대비 paired Δlog ATE(중앙값)와 win 수, 채널은 drift_decomposition 중앙값.</p>']
    # exposure
    L.append('<table><thead><tr><th>arm</th><th>ranks</th><th>launches / OOM</th><th>steps (cor/id)</th><th>s/step</th><th>fit modes</th><th>fallback</th><th>완주</th></tr></thead><tbody>')
    for arm, e in ex.items():
        fb = e.get("fallback_share")
        L.append(f'<tr><td class="id">{arm}</td><td>{e.get("ranks","—")}</td><td>{e.get("launches","—")} / {e.get("oom_kills","—")}</td>'
                 f'<td>{e.get("steps","—")} ({e.get("correction_steps","—")}/{e.get("identity_steps","—")})</td><td>{f(e.get("s_per_step"),1)}</td>'
                 f'<td>{e.get("fit_modes") or "—"}</td><td>{(f"{fb:.1%}" if fb is not None else "—")}</td><td>{"예" if e.get("finished") else "아니오"}</td></tr>')
    L.append('</tbody></table>')
    # oxford
    L.append('<table><thead><tr><th>arm</th><th>step</th><th>ATE mean</th><th>AUC_03</th><th>Δlog vs base</th><th>win/base</th><th>Δlog vs gtctrl</th><th>win/gtctrl</th></tr></thead><tbody>')
    b = ox.get("base_k1", {})
    L.append(f'<tr><td class="id">base_k1</td><td>0</td><td>{f(b.get("ate_mean"))}</td><td>{f(b.get("auc03"),2)}</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td></tr>')
    for arm, rows in ox.items():
        if arm == "base_k1":
            continue
        for s, r in rows.items():
            vb, vg = r["vs_base"], r.get("vs_gtctrl")
            L.append(f'<tr><td class="id">{arm}</td><td>{s}</td><td>{f(r["ate_mean"])}</td><td>{f(r["auc03"],2)}</td>'
                     f'<td class="n {"bad" if (vb["dlog_median"] or 0) > 0 else "good"}">{f(vb["dlog_median"],3,True)}</td><td>{vb["wins"]}/{vb["n"]}</td>'
                     + (f'<td class="n {"bad" if (vg["dlog_median"] or 0) > 0 else "good"}">{f(vg["dlog_median"],3,True)}</td><td>{vg["wins"]}/{vg["n"]}</td>' if vg else '<td>&mdash;</td><td>&mdash;</td>')
                     + '</tr>')
    L.append('</tbody></table>')
    # channels
    cols = ["wscale_std", "depth_drift", "ratio_drift", "rot1", "absrot"]
    if ch:
        L.append('<table><thead><tr><th>arm</th><th>step</th>' + ''.join(f'<th>{c}</th>' for c in cols) + '<th>vs gtctrl (paired win)</th></tr></thead><tbody>')
        bb = ch.get("base", {})
        L.append('<tr><td class="id">base</td><td>0</td>' + ''.join(f'<td>{f(bb.get(c))}</td>' for c in cols) + '<td>&mdash;</td></tr>')
        for arm, rows in ch.items():
            if arm == "base":
                continue
            for s, r in rows.items():
                cmp = r.get("vs_gtctrl") or {}
                txt = ", ".join(f'{k} {v["wins"]}/{v["n"]}' for k, v in cmp.items() if k in ("wscale_std", "depth_drift", "ratio_drift", "ate"))
                L.append(f'<tr><td class="id">{arm}</td><td>{s}</td>' + ''.join(f'<td>{f(r.get(c))}</td>' for c in cols) + f'<td>{txt or "&mdash;"}</td></tr>')
        L.append('</tbody></table>')
    # verdict
    for arm, v in ver.items():
        L.append(f'<p><b>{arm}</b>: <span class="st sure">{v.get("status")}</span> — {v.get("next", v.get("why", ""))}'
                 + (f' (ATE win {v["ate_wins"]}/10, Δlog {f(v["ate_dlog"],3,True)}; scale 채널 하락 {v["scale_lower"]})' if "ate_wins" in v else "") + '</p>')
    L.append('<p><b>거리 사다리 (09-17, docs/gtabs-plan.md §13)</b>: 학습 10 scene을 Oxford와 같은 1.5 m/frame·473 m로 스트리밍하면 세 팔 모두 base와 같다(Δlog +0.01~+0.07). '
             'hold-out MCD scene(같은 캠퍼스·다른 캠퍼스 모두)은 mid·far에서 +0.1~+0.5로 나빠지고, Oxford의 +0.25~+0.29는 그 연장선. '
             '→ <span class="st refuted">D1 거리 체제 기각</span>, <span class="st sure">손상 = 학습 장면 10개에 대한 특화</span>. depth drift는 학습 scene에서 가장 큰데 ATE는 멀쩡 → 메커니즘이 아니라 지문. '
             '다음: 400 GB 컨테이너에서 c0on(214 scene + correction) 재실행·채점.</p>')
    L.append('<p><b>broad 팔 (09-17, plan §14)</b>: gtctrl 레시피를 214 scene 코퍼스(교사 라벨)로 275 step. Oxford Δlog +0.132 (2/10), MCD hold-out far +0.174 (3/15) — '
             '10 scene 팔(gtctrl +0.291/+0.120, teasup +0.185/+0.126)과 같은 모양·크기. 275 step에 correction이 실제로 굴린 scene은 25개(scene당 5창). '
             'identity-only c0off(같은 코퍼스, 1250 step)는 hold-out far +0.023 (6/15). → <span class="st refuted">장면 폭 가설 반증</span>; '
             '손상 재료는 <span class="st sure">correction 브랜치의 on-policy 캐시 학습</span>(굴러본 scene에만 무해). 남은 시험: broad 2500 step(≈11 h).</p>')
    L.append(END)
    frag = "\n".join(L) + "\n"
    html = open(LEDGER).read()
    if BEGIN in html and END in html:
        html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", frag, html, flags=re.S)
    else:
        anchor = '<h2><span class="num">05</span>출처</h2>'
        assert anchor in html
        html = html.replace(anchor, frag + anchor, 1)
    open(LEDGER, "w").write(html)
    print(f"[ledger] gtabs section written into {LEDGER}")


if __name__ == "__main__":
    main()
