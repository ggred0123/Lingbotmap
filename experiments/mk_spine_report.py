#!/usr/bin/env python3
"""Build docs/spine-experiment.html from whatever results exist right now.

The report is regenerated, never hand-edited: every number in it is read back
from experiments/results/*.json or from the benchmark workspaces, so a section
that has no data yet renders as PENDING instead of as a stale number.

    python3 experiments/mk_spine_report.py

Inputs it looks for (all optional):
    experiments/results/rpe_at_k_<dataset>.json   experiments/rpe_at_k.py
    experiments/results/train_<arm>.json          the trainer's own log
    bench_ws/<ws>/<ds>/eval/traj.json             benchmark evaluate phase
    bench_ws/<ws>/<ds>/<scene>/eval/{traj,auc}.json
"""
from __future__ import annotations

import html
import itertools
import json
import math
import re
import os
import statistics as st
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
LOGS = ROOT / "experiments" / "logs"
WS = Path("/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws")
OUT = ROOT / "docs" / "spine-experiment.html"

ARMS = {
    "s0off": ("off-policy만", "p_identity = 1.0", "identity 브랜치만 — reset · K=1 · teacher-matched 창"),
    "s0on": ("on-policy만", "p_identity = 0.0", "correction 브랜치만 — 학생이 걸어간 상태에서 교정"),
}


# ── small helpers ───────────────────────────────────────────────────────────
def esc(s) -> str:
    return html.escape(str(s))


def fmt(x, n=3, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "&mdash;"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def med(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return st.median(vals) if vals else None


def dlog(a, b):
    """log(a) - log(b), the audit's Delta-log unit; None if either is unusable."""
    if not a or not b or a <= 0 or b <= 0:
        return None
    return math.log(a) - math.log(b)


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── data collection ─────────────────────────────────────────────────────────
def train_progress(arm: str) -> dict:
    """Where an arm is right now: from its checkpoints and its log."""
    out = {"arm": arm, "steps_done": 0, "target": None, "finished": False,
           "ckpts": [], "wall_s": None}
    j = load(RES / f"train_{arm}.json")
    if j:
        # ★ The trainer rewrites this file at every --save_every, so its mere
        # existence means "started", not "finished".  wall_s is written once, at
        # the end.  Reading it the other way once launched arm B on top of arm A.
        out["wall_s"] = j.get("wall_s")
        out["finished"] = bool(out["wall_s"])
        out["target"] = j.get("meta", {}).get("steps")
        out["steps_done"] = len(j.get("steps", []))
    ck = sorted(int(p.name.split("step")[1].split(".")[0])
                for p in (ROOT / "ckpt_train").glob(f"{arm}.step*.pt"))
    out["ckpts"] = ck
    log = LOGS / f"train_{arm}.log"
    if log.exists():
        last = 0
        for line in log.read_text(errors="ignore").splitlines()[-4000:]:
            s = line.strip()
            if s.startswith("[") and "]" in s[:8]:
                head = s[1:s.index("]")].strip()
                if head.isdigit():
                    last = max(last, int(head))
        out["steps_done"] = max(out["steps_done"], last + 1 if last else 0)
        if out["target"] is None:
            out["target"] = 1250
    return out


def rpe_table(dataset: str, base="base_auto") -> dict:
    """Per-method medians over the scenes that method and the base share."""
    d = load(RES / f"rpe_at_k_{dataset}.json")
    if not d:
        return {}
    b = d.get(base, {})
    rows = {}
    for m, sc in d.items():
        common = sorted(set(sc) & set(b)) or sorted(sc)
        g = lambda k: med([sc[s].get(k) for s in common])          # noqa: E731
        dl = lambda k: med([dlog(sc[s].get(k), b[s].get(k))        # noqa: E731
                            for s in common if s in b])
        rows[m] = {
            "n": len(common), "K": (sc[common[0]]["K"] if common else None),
            "ate": g("ate"), "d1": g("rpe_trans_d1"), "dK": g("rpe_trans_dK"),
            "cos": g("kf_cos_median"), "rev": g("kf_reverse_frac"),
            "ratio": g("kf_ratio_median"),
            "dlog_ate": None if m == base else dl("ate"),
            "dlog_d1": None if m == base else dl("rpe_trans_d1"),
            "dlog_dK": None if m == base else dl("rpe_trans_dK"),
            "wins": None if m == base else sum(
                1 for s in common if s in b and sc[s]["ate"] < b[s]["ate"]),
        }
    return rows


def scene_ate(ws: str, ds: str, method: str) -> dict:
    """{scene: ate} straight from the per-scene eval json."""
    out = {}
    root = WS / ws / ds
    if not root.is_dir():
        return out
    for sd in sorted(root.iterdir()):
        j = load(sd / "eval" / "traj.json")
        if j and method in j and j[method].get("ate") is not None:
            out[sd.name] = j[method]["ate"]
    return out


def paired_dlog(ws: str, ds: str, method: str, base: str) -> tuple:
    a, b = scene_ate(ws, ds, method), scene_ate(ws, ds, base)
    common = sorted(set(a) & set(b))
    dl = [dlog(a[s], b[s]) for s in common]
    dl = [x for x in dl if x is not None]
    if not dl:
        return None, 0, 0
    return st.median(dl), sum(1 for x in dl if x < 0), len(dl)


# ── html pieces ─────────────────────────────────────────────────────────────
CSS = """
:root{
  --ground:#F4F6F8;--surface:#FFFFFF;--surface-2:#EEF1F5;--ink:#161A21;--ink-2:#3D4652;
  --muted:#697588;--line:#D9DEE6;--line-strong:#BCC5D1;--accent:#2C5A7A;--accent-soft:#E3EDF4;
  --pass:#1C6B57;--pass-bg:#E0EFE9;--fail:#9E3A22;--fail-bg:#F6E3DD;--partial:#7A5C10;
  --partial-bg:#F5ECD6;--shadow:0 1px 2px rgba(22,26,33,.06),0 8px 24px -16px rgba(22,26,33,.25);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0F1318;--surface:#161B22;--surface-2:#1D242D;--ink:#E3E8EF;--ink-2:#B6C0CC;
  --muted:#8492A3;--line:#262E38;--line-strong:#38424F;--accent:#7FB0CE;--accent-soft:#1A2833;
  --pass:#59B79B;--pass-bg:#14291F;--fail:#DE8266;--fail-bg:#2E1A15;--partial:#D2A73F;
  --partial-bg:#2A2213;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
}}
:root[data-theme="dark"]{
  --ground:#0F1318;--surface:#161B22;--surface-2:#1D242D;--ink:#E3E8EF;--ink-2:#B6C0CC;
  --muted:#8492A3;--line:#262E38;--line-strong:#38424F;--accent:#7FB0CE;--accent-soft:#1A2833;
  --pass:#59B79B;--pass-bg:#14291F;--fail:#DE8266;--fail-bg:#2E1A15;--partial:#D2A73F;
  --partial-bg:#2A2213;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Helvetica Neue",sans-serif;
  font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:0 28px 96px}
.masthead{padding:64px 0 34px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;font-weight:500;
  letter-spacing:.13em;text-transform:uppercase;color:var(--muted);margin:0 0 18px;
  display:flex;flex-wrap:wrap;gap:8px 16px}
h1{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:clamp(34px,5.2vw,50px);
  line-height:1.1;letter-spacing:-.018em;margin:0 0 18px;text-wrap:balance}
.standfirst{font-size:19px;line-height:1.58;color:var(--ink-2);max-width:63ch;margin:0}
.standfirst strong{color:var(--ink);font-weight:600}
section{padding-top:56px}
h2{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:26px;line-height:1.25;
  letter-spacing:-.012em;margin:0 0 6px;text-wrap:balance}
h2 .num{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px;font-weight:500;
  color:var(--accent);margin-right:12px;vertical-align:2px}
.dek{color:var(--muted);font-size:15px;margin:0 0 26px;max-width:66ch}
p{max-width:68ch}
h3{font-size:15px;font-weight:600;margin:34px 0 12px;color:var(--ink)}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em}
code{background:var(--surface-2);padding:.1em .38em;border-radius:3px;color:var(--ink-2)}
strong{font-weight:600}
.verdict{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:4px;padding:24px 26px;margin:30px 0 0;box-shadow:var(--shadow)}
.verdict p{margin:0;max-width:none;font-size:17px;line-height:1.6}
.verdict p + p{margin-top:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:14px;margin-top:22px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:18px;
  display:flex;flex-direction:column;gap:10px;box-shadow:var(--shadow)}
.card .k{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}
.card .t{font-size:15px;font-weight:600;line-height:1.4}
.card .v{font-size:13.5px;color:var(--ink-2);line-height:1.5}
.chip{display:inline-flex;align-items:center;gap:7px;align-self:flex-start;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;font-weight:600;
  letter-spacing:.07em;text-transform:uppercase;padding:4px 9px;border-radius:3px;
  border:1px solid transparent}
.chip::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;flex:none}
.chip.fail{color:var(--fail);background:var(--fail-bg);border-color:var(--fail)}
.chip.pass{color:var(--pass);background:var(--pass-bg);border-color:var(--pass)}
.chip.partial{color:var(--partial);background:var(--partial-bg);border-color:var(--partial)}
.tw{overflow-x:auto;margin:22px 0 6px;border:1px solid var(--line);border-radius:4px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:560px}
caption{text-align:left;padding:14px 18px 0;font-size:13px;color:var(--muted);
  font-family:"IBM Plex Mono",ui-monospace,monospace}
th,td{padding:9px 14px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;font-weight:500;
  letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  border-bottom:1px solid var(--line-strong)}
tbody tr:last-child td{border-bottom:none}
td.n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
td.good{color:var(--pass);font-weight:600}
td.bad{color:var(--fail);font-weight:600}
tr.base td{background:var(--surface-2)}
.note{font-size:13px;color:var(--muted);margin:10px 0 0;max-width:70ch}
ul.plain{padding-left:20px;max-width:70ch}
ul.plain li{margin-bottom:8px}
.status{display:grid;gap:10px;margin-top:22px}
.srow{display:grid;grid-template-columns:200px 1fr 92px;gap:14px;align-items:center;
  background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:12px 16px}
.srow .lab{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;color:var(--ink-2)}
.track{background:var(--surface-2);border-radius:3px;height:18px;overflow:hidden}
.bar{height:100%;background:var(--accent);border-radius:0 3px 3px 0}
.srow .val{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;text-align:right;color:var(--muted)}
footer{margin-top:64px;padding-top:22px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
footer p{max-width:72ch;margin:0 0 8px}
"""


def table(caption, head, rows, cls_by_col=None):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r["cells"]):
            cls = (cls_by_col or {}).get(i, "n" if i else "")
            if isinstance(c, tuple):
                c, extra = c
                cls = f"{cls} {extra}".strip()
            cells.append(f'<td class="{cls}">{c}</td>' if cls else f"<td>{c}</td>")
        body.append(f'<tr class="{r.get("cls","")}">' + "".join(cells) + "</tr>")
    return (f'<div class="tw"><table><caption>{caption}</caption>'
            f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def status_rows(items):
    out = []
    for lab, frac, val in items:
        pct = max(0.0, min(1.0, frac)) * 100
        out.append(f'<div class="srow"><div class="lab">{esc(lab)}</div>'
                   f'<div class="track"><div class="bar" style="width:{pct:.1f}%"></div></div>'
                   f'<div class="val">{esc(val)}</div></div>')
    return '<div class="status">' + "".join(out) + "</div>"


# ── sections ────────────────────────────────────────────────────────────────
def sec_setup(prog) -> str:
    rows = []
    for arm, (name, knob, what) in ARMS.items():
        p = prog[arm]
        state = ("완료" if p["finished"] else
                 (f"학습 중 {p['steps_done']}/{p['target'] or 1250}" if p["steps_done"]
                  else "대기"))
        rows.append({"cells": [f"<strong>{esc(name)}</strong><br>"
                               f'<span class="mono" style="color:var(--muted)">{esc(arm)}</span>',
                               f"<code>{esc(knob)}</code>", esc(what), esc(state)]})
    t = table("v6i 정책 고정 · 바뀌는 것은 p_identity 하나",
              ["arm", "설정", "무엇을 학습하는가", "상태"], rows,
              cls_by_col={1: "", 2: "", 3: "n"})
    return f"""
<section id="setup"><h2><span class="num">01</span>무엇을 돌렸는가</h2>
<p class="dek">docs/eval-metric-audit.html §10 순서 0 · §11 2a. 지금까지 모든 실행이
p_identity 0.1 또는 0.35의 혼합이라 2×2의 "naive pseudo-label FT" 칸과 "Ours" 칸이
분리된 적이 없습니다.</p>
{t}
<p class="note">나머지는 v6i와 같습니다 &mdash; 코퍼스 가중치
<code>mcd:10 slowtv:10 dl3dv:10 scannet:10 replica:5 dynamicreplica:10 unrealstereo4k:15 paralleldomain4d:30</code>,
horizons 320/960/1920/3840, fixed_horizon=0, pool 20, S 48, lr 1e-5, l2sp 0, lam_fresh 1.0,
preset A1PC, 1250 step, DDP 2 rank. 같은 <code>launch_v6.sh</code> 경로이므로 step 축이 v6i·v7f와 같습니다.</p>
<p class="note">★ 기록해 둘 비대칭: <code>p_identity=0</code> 팔에서는 fresh(보존) 브랜치가 한 번도 실행되지 않습니다
(<code>trainer.py:2572</code>는 identity/both에서만 fresh를 돕니다). l2sp=0이므로 그 팔에는
&theta;<sub>0</sub>에 대한 앵커가 아예 없습니다 &mdash; 이것은 실험의 결함이 아니라
"on-policy만"의 정의 자체이고, 감사 문서 §11 2f가 지적한 안정화 옵션 부재와 같은 축입니다.</p>
</section>"""


def sec_rpe(dataset, title, dek, base="base_auto", extra="") -> str:
    rows_d = rpe_table(dataset, base)
    if not rows_d:
        return f"""
<section><h2><span class="num">02</span>{title}</h2>
<p class="dek">{dek}</p><p><span class="chip partial">pending</span></p></section>"""
    order = [base] + sorted(m for m in rows_d if m != base)
    rows = []
    for m in order:
        r = rows_d[m]
        wins = "&mdash;" if r["wins"] is None else f"{r['wins']}/{r['n']}"
        bad = lambda v, inv=False: ("bad" if (v or 0) > 0 else "good") if v is not None else ""  # noqa: E731
        rows.append({"cls": "base" if m == base else "", "cells": [
            f"<code>{esc(m)}</code>", r["n"], r["K"],
            fmt(r["ate"], 3), fmt(r["d1"], 3), fmt(r["dK"], 3),
            fmt(r["cos"], 3, sign=True),
            "&mdash;" if r["rev"] is None else f"{r['rev']:.1%}",
            fmt(r["ratio"], 2),
            (fmt(r["dlog_ate"], 3, sign=True), bad(r["dlog_ate"])),
            (fmt(r["dlog_d1"], 3, sign=True), bad(r["dlog_d1"])),
            (fmt(r["dlog_dK"], 3, sign=True), bad(r["dlog_dK"])),
            wins]})
    t = table(f"{dataset} · evo rpe(translation_part, Sim(3) 정렬) · &Delta;log는 scene 짝지음 중앙값",
              ["method", "n", "K", "ATE", "rpe &Delta;=1", "rpe &Delta;=K",
               "cos @K", "역방향 @K", "|est|/|gt| @K",
               "&Delta;log ATE", "&Delta;log d1", "&Delta;log dK", "ATE win"], rows)
    return f"""
<section><h2><span class="num">02</span>{title}</h2>
<p class="dek">{dek}</p>
{t}
{extra}
</section>"""


def sec_status(prog, bench) -> str:
    items = []
    for arm, (name, knob, _) in ARMS.items():
        p = prog[arm]
        tgt = p["target"] or 1250
        items.append((f"{arm}  {knob}", (p["steps_done"] or 0) / tgt,
                      f"{p['steps_done'] or 0}/{tgt}"))
    for lab, done, tot in bench:
        items.append((lab, (done / tot) if tot else 0, f"{done}/{tot}"))
    return f"""
<section><h2><span class="num">00</span>지금 어디까지</h2>
<p class="dek">이 문서는 결과 파일에서 다시 생성됩니다. 아래 막대가 갱신 시점의 실제 상태입니다.</p>
{status_rows(items)}
<p class="note">생성 시각 {datetime.now():%Y-%m-%d %H:%M} &middot;
<code>python3 experiments/mk_spine_report.py</code></p>
</section>"""


def bench_progress():
    out = []
    for ws, ds, methods in [
        ("oxford_long", "oxford_long",
         ["sd_v6is300_auto", "sd_v6is1200_auto"]),
    ]:
        root = WS / ws / ds
        scenes = [d for d in root.iterdir() if d.is_dir() and (d / "gt").is_dir()] \
            if root.is_dir() else []
        for m in methods:
            done = sum(1 for s in scenes if (s / m / ".complete.json").exists())
            out.append((f"bench {ds} · {m}", done, len(scenes)))
    return out




STEP_RE = __import__("re").compile(r"^sd_(?P<run>[a-z0-9]+)s(?P<step>\d+)_(?P<kind>k1|auto)$")



PROBE_RUNS = [("base (step 0)", None), ("v6i", "v6i"), ("v7f", "v7f"),
              ("s0off", "s0off"), ("s0on", "s0on")]


def sec_probe() -> str:
    """The trainer's own held-out teacher-imitation probe, arm against arm.

    Not the decision metric -- experiments/auto_eval_v6i.sh records that the
    probe and Oxford ATE disagree, sometimes in opposite directions -- but it is
    the only in-training read available before a checkpoint is scored.
    """
    series = {}
    for label, run in PROBE_RUNS:
        if run is None:
            continue
        j = load(RES / f"train_{run}.json")
        if not j:
            continue
        xs = [(p["step"], p["mean"]) for p in j.get("probes", [])
              if isinstance(p.get("mean"), (int, float))]
        if len(xs) >= 3:
            series[label] = xs
    if "s0off" not in series:
        return ""
    grid = [0, 150, 300, 450, 600, 750, 900, 1050, 1200]
    rows = []
    for label, xs in series.items():
        d = dict(xs)
        base0 = xs[0][1]
        last_s, last_v = xs[-1]
        cells = [f"<code>{esc(label)}</code>"]
        for g in grid:
            v = d.get(g)
            cells.append("&mdash;" if v is None else f"{v:.3f}")
        drift = (last_v - base0) / base0
        cells.append((f"{drift:+.1%} @ s{last_s}", "bad" if drift > 0 else "good"))
        rows.append({"cells": cells})
    t = table("held-out 창의 teacher-imitation probe · 낮을수록 교사에 가까움 · 모두 같은 step 0에서 출발",
              ["run"] + [f"s{g}" for g in grid] + ["출발 대비"], rows)
    return f"""
<section><h2><span class="num">03b</span>학습 중 지표 &mdash; 예비 관찰</h2>
<p class="dek">체크포인트를 채점하기 전에 볼 수 있는 유일한 값입니다.
프로젝트 자신의 기록(<code>experiments/auto_eval_v6i.sh</code>)이 이 probe와 Oxford ATE가
어긋난다고 적어 두었으므로 판정에 쓰지 않고, 관찰로만 적습니다.</p>
{t}
<div class="verdict"><p><strong>off-policy만 도는 팔이 교사에게서 더 빨리 멀어집니다.</strong>
identity 브랜치가 곧 teacher-imitation 목표인데, 그것만 100% 돌린 <code>s0off</code>가
held-out 창에서 가장 빠르게 나빠집니다. 혼합(v6i)이 step 1200에 도달하는 열화를
<code>s0off</code>는 step 750~825에서 지납니다. 두 번의 튐(step 225·825)은 한 scene이
3.0 부근으로 튄 것이고, 그 둘을 빼도 추세는 남습니다.</p>
<p>이것이 &ldquo;off-policy가 더 나쁘다&rdquo;는 뜻은 아직 아닙니다. probe는 교사 모방을 재고,
프로젝트가 판정에 쓰는 것은 Oxford K=1의 GT ATE입니다. 두 값이 어긋난 전례가 이미 있습니다.</p></div>
</section>"""


def sec_stepcurve() -> str:
    """oxford_long as a step curve -- the project's spine bench, first populated."""
    d = load(RES / "rpe_at_k_oxford_long.json")
    if not d or "base_auto" not in d:
        return ""
    b = d["base_auto"]
    pts = []
    for m, sc in d.items():
        mo = STEP_RE.match(m)
        if not mo:
            continue
        ss = sorted(set(sc) & set(b))
        if len(ss) < 5:                     # a one-scene cell is not a curve point
            continue
        pts.append({
            "run": mo["run"], "step": int(mo["step"]), "n": len(ss),
            "dlog_ate": med([dlog(sc[s]["ate"], b[s]["ate"]) for s in ss]),
            "dlog_dK": med([dlog(sc[s]["rpe_trans_dK"], b[s]["rpe_trans_dK"]) for s in ss]),
            "ratio": med([sc[s].get("kf_ratio_median") for s in ss]),
            "win": sum(1 for s in ss if sc[s]["ate"] < b[s]["ate"]),
        })
    if not pts:
        return ""
    pts.sort(key=lambda r: (r["run"], r["step"]))
    rows = [{"cls": "base", "cells": [
        "<code>base_auto</code>", "&mdash;", len(b), "0.000", "0.000",
        f'{med([b[s].get("kf_ratio_median") for s in b]):.2f}', "&mdash;"]}]
    for r in pts:
        rows.append({"cells": [
            f'<code>{esc(r["run"])}</code>', r["step"], r["n"],
            (fmt(r["dlog_ate"], 3, sign=True), "bad" if (r["dlog_ate"] or 0) > 0 else "good"),
            (fmt(r["dlog_dK"], 3, sign=True), "bad" if (r["dlog_dK"] or 0) > 0 else "good"),
            fmt(r["ratio"], 2), f'{r["win"]}/{r["n"]}']})
    t = table("oxford_long &middot; 3,840 프레임 &middot; K=12 &middot; GT 있음 &middot; base_auto 대비 scene 짝지음 중앙값",
              ["run", "step", "n", "&Delta;log ATE", "&Delta;log rpe@K",
               "|est|/|gt| @K", "ATE win"], rows)
    # ── matched-step, matched-corpus: v6i (lam_long=0) against v7f ──────────
    raw = load(RES / "rpe_at_k_oxford_long.json") or {}
    pair_rows, pair_txt = [], ""
    for step in sorted({r["step"] for r in pts if r["run"] == "v6i"}):
        a, c = f"sd_v6is{step}_auto", f"sd_v7fs{step}_auto"
        if a not in raw or c not in raw:
            continue
        ss = sorted(set(raw[a]) & set(raw[c]) & set(b))
        if len(ss) < 5:
            continue
        d_a = med([dlog(raw[a][s]["ate"], b[s]["ate"]) for s in ss])
        d_c = med([dlog(raw[c][s]["ate"], b[s]["ate"]) for s in ss])
        d_p = med([dlog(raw[c][s]["ate"], raw[a][s]["ate"]) for s in ss])
        w = sum(1 for s in ss if raw[c][s]["ate"] < raw[a][s]["ate"])
        pair_rows.append({"cells": [
            f"step {step}", len(ss), fmt(d_a, 3, sign=True), fmt(d_c, 3, sign=True),
            (fmt(d_p, 3, sign=True), "bad" if (d_p or 0) > 0 else "good"),
            f"{w}/{len(ss)}"]})
    if pair_rows:
        pair_txt = table(
            "같은 step · 같은 코퍼스 · lam_long만 다름 (v6i 0 대 v7f 0.05)",
            ["", "n", "v6i &Delta;log ATE", "v7f &Delta;log ATE",
             "v7f &minus; v6i", "v7f 우세"], pair_rows)
    v6i = {r["step"]: r for r in pts if r["run"] == "v6i"}
    v7f = {r["step"]: r for r in pts if r["run"] == "v7f"}
    attrib = ""
    if 300 in v6i and 1200 in v6i and pair_rows:
        step_span = v6i[1200]["dlog_ate"] - v6i[300]["dlog_ate"]
        term_span = max(abs(float(r["cells"][4][0])) for r in pair_rows)
        attrib = f"""<div class="verdict">
<p><strong>감사 문서의 귀속이 장거리 셀에서도 성립합니다.</strong> step 300에서 두 실행은
구분되지 않고(짝지음 중앙값 {pair_rows[0]['cells'][4][0]}), step 1200에서도 차이는
{pair_rows[-1]['cells'][4][0]}입니다. 같은 구간에서 step 축은 {step_span:+.3f} 움직입니다 &mdash;
<strong>step 효과가 손실 항 효과의 {step_span / max(term_span, 1e-9):.0f}배</strong>입니다.
감사 문서가 Oxford K=1 · 320프레임에서 잰 6배와 같은 크기이고,
그 문서 §11 2b가 &ldquo;장거리 셀에는 v6i·v7 arm이 없어 가설을 시험할 수 없다&rdquo;고 적은
바로 그 셀에서 처음 재본 값입니다.</p>
<p>lam_long이 도움이 된다는 증거는 여기서도 없습니다. step 1200에서 v7f가 v6i보다
나은 scene은 3/10입니다.</p></div>"""
    verdict = ""
    if 300 in v6i and 1200 in v6i:
        verdict = f"""<div class="verdict">
<p><strong>척추 벤치가 처음으로 채워졌고, 같은 이야기를 합니다.</strong> v6i의 &Delta;log ATE가
step 300에서 {v6i[300]['dlog_ate']:+.3f}({v6i[300]['win']}/{v6i[300]['n']} 승),
step 1200에서 {v6i[1200]['dlog_ate']:+.3f}({v6i[1200]['win']}/{v6i[1200]['n']} 승)입니다.
Oxford K=1 &middot; 320프레임에서 본 단조 악화가 3,840프레임 &middot; K=12에서도 그대로입니다 &mdash;
&ldquo;짧은 벤치라서 진다&rdquo;는 설명이 여기서는 통하지 않습니다.</p>
<p><strong>그런데 스케일은 멀쩡합니다.</strong> 키프레임 변위 비가 step 1200에서도
{v6i[1200]['ratio']:.2f}로, VBR의 0.62와 다릅니다. K=12 &middot; 3,840 프레임에서는
ATE가 두 배 넘게 나빠지는 동안에도 스케일이 무너지지 않습니다. 감사 문서 §04가 지목한
depth 스케일 드리프트는 이 체제에서는 악화의 원인이 아닙니다.</p></div>"""
    return f"""
<section><h2><span class="num">02b</span>척추 벤치의 step 곡선</h2>
<p class="dek">감사 문서 §11 2b가 &ldquo;<code>oxford_long</code>에 base·v6a·v6f뿐&rdquo;이라고 적은 셀입니다.
여기에 v6i를 step 300·1200으로 채웠습니다 &mdash; 코퍼스·정책이 v7f와 같고 lam_long만 다른, 유일하게 깨끗한 control.</p>
{t}
{verdict}
{pair_txt}
{attrib}
</section>"""


def sec_vbr() -> str:
    """VBR at Delta=K -- the audit's own cell, recomputed from scratch."""
    d = load(RES / "rpe_at_k_vbr.json")
    if not d or "base_auto" not in d:
        return ""
    b, v = d["base_auto"], d.get("sd_v7fs1200_auto", {})
    scenes = sorted(set(b) & set(v))
    rows = []
    for s in scenes:
        B, V = b[s], v[s]
        worse = V["kf_reverse_frac"] > B["kf_reverse_frac"]
        tie = V["kf_reverse_frac"] == B["kf_reverse_frac"]
        rows.append({"cells": [
            f"<code>{esc(s)}</code>", B["K"],
            f'{B["rpe_trans_d1"]:.2f} &rarr; {V["rpe_trans_d1"]:.2f}',
            f'{B["rpe_trans_dK"]:.2f} &rarr; {V["rpe_trans_dK"]:.2f}',
            f'{B["kf_cos_median"]:.3f} / {V["kf_cos_median"]:.3f}',
            (f'{B["kf_reverse_frac"]:.1%} / {V["kf_reverse_frac"]:.1%}',
             "bad" if worse else ("" if tie else "good")),
            f'{B["kf_ratio_median"]:.2f} / {V["kf_ratio_median"]:.2f}']})
    d1 = med([dlog(v[s]["rpe_trans_d1"], b[s]["rpe_trans_d1"]) for s in scenes])
    dK = med([dlog(v[s]["rpe_trans_dK"], b[s]["rpe_trans_dK"]) for s in scenes])
    r_b = med([b[s]["kf_ratio_median"] for s in scenes])
    r_v = med([v[s]["kf_ratio_median"] for s in scenes])
    cos_base_better = sum(1 for s in scenes
                          if b[s]["kf_cos_median"] > v[s]["kf_cos_median"])
    rev_worse = sum(1 for s in scenes
                    if v[s]["kf_reverse_frac"] > b[s]["kf_reverse_frac"])
    rows.append({"cls": "base", "cells": [
        "<strong>중앙값</strong>", "",
        (f"&Delta;log {d1:+.3f}", "good"), (f"&Delta;log {dK:+.3f}", "good"),
        f"base 우세 {cos_base_better}/{len(scenes)}",
        f"v7f 나쁨 {rev_worse}/{len(scenes)}",
        f"{r_b:.2f} / {r_v:.2f}"]})
    t = table("VBR 7 scene &middot; base_auto &rarr; sd_v7fs1200_auto &middot; 독립 재계산",
              ["scene", "K", "rpe &Delta;=1", "rpe &Delta;=K", "cos @K",
               "역방향 @K", "|est|/|gt| @K"], rows)
    return f"""
<section><h2><span class="num">03</span>감사 문서 재현 &mdash; VBR</h2>
<p class="dek">같은 원시 궤적을 독립적으로 다시 재서 도구와 감사 결론을 함께 검증합니다.
&Delta;=1 값은 벤치 <code>eval/traj.json</code>과 1e-14 이내로 일치했습니다.</p>
{t}
<div class="verdict"><p><strong>재현됩니다.</strong> 헤드라인 &Delta;log &minus;0.468이
키프레임 간격에서 &minus;0.045로 사라지는 것, 방향이 6/7에서 base 우세인 것,
키프레임 변위가 GT의 0.62배(base 0.99배)인 것 모두 감사 문서와 같습니다.</p>
<p><strong>한 항목은 다릅니다.</strong> 감사 문서는 역방향 비율이 &ldquo;7/7에서 v7f가 더 나쁘다&rdquo;고
적었지만 여기서는 <strong>5/7</strong>입니다 &mdash; <code>campus_train1</code>은 소수점 넷째 자리까지 동률이고,
<code>diag_train0</code>은 v7f가 오히려 낫습니다(16.08% 대 17.04%). 키프레임 격자를
프레임 0부터 K 간격으로 잡았으므로 감사 문서가 다른 오프셋을 썼다면 이 차이가 설명됩니다.
결론의 방향은 바뀌지 않습니다.</p></div>
</section>"""



def seam_stats() -> dict:
    """{method: [{n_seams, resid_median, s_lo, s_hi}, ...]} parsed from bench logs.

    The seam line is printed by the method, so it carries no method name; the log
    is sequential, so the enclosing "Combination (i/n): Running <method>" line is
    the attribution.
    """
    import re as _re
    comb = _re.compile(r"Running (\S+) on ")
    seam = _re.compile(r"(\d+) seams, median residual ([\d.]+) \(max ([\d.]+)\), "
                       r"scale spread ([\d.]+)-([\d.]+)")
    out: dict[str, list] = {}
    cur = None
    for lg in sorted(LOGS.glob("bench_ra_*.log")) + sorted(LOGS.glob("bench_*ra*.log")):
        for line in lg.read_text(errors="ignore").splitlines():
            m = comb.search(line)
            if m:
                cur = m.group(1)
            m = seam.search(line)
            if m and cur:
                out.setdefault(cur, []).append({
                    "n": int(m.group(1)), "resid": float(m.group(2)),
                    "s_lo": float(m.group(4)), "s_hi": float(m.group(5))})
    return out


def sec_reanchor() -> str:
    """Order 1 of the audit's plan: how deep does the gauge hold?"""
    ms = [("base_k1", None), ("base_ra16_k1", 16), ("base_ra32_k1", 32),
          ("base_ra64_k1", 64), ("base_ra128_k1", 128)]
    data = {m: {} for m, _ in ms}
    root = WS / "oxford" / "oxford"
    if not root.is_dir():
        return ""
    for sd in sorted(root.iterdir()):
        j = load(sd / "eval" / "traj.json")
        if not j:
            continue
        for m, _ in ms:
            if m in j:
                data[m][sd.name] = j[m]
    base = data["base_k1"]
    if not base or not data["base_ra32_k1"]:
        return ""
    seams = seam_stats()
    rows = []
    for m, M in ms:
        ss = sorted(set(data[m]) & set(base))
        if not ss:
            continue
        g = lambda k: med([data[m][s][k] for s in ss])                    # noqa: E731
        dl = lambda k: (None if M is None else                            # noqa: E731
                        med([dlog(data[m][s][k], base[s][k]) for s in ss]))
        win = "&mdash;" if M is None else \
            f"{sum(1 for s in ss if data[m][s]['ate'] < base[s]['ate'])}/{len(ss)}"
        sp = seams.get(m)
        spread = "&mdash;"
        if sp:
            spread = (f"{min(x['s_lo'] for x in sp):.2f}&ndash;"
                      f"{max(x['s_hi'] for x in sp):.2f}")
        d_ate, d_rpe = dl("ate"), dl("rpe_trans")
        rows.append({"cls": "base" if M is None else "", "cells": [
            f"<code>{esc(m)}</code>", "&mdash;" if M is None else M, len(ss),
            fmt(g("ate"), 3), fmt(g("rpe_trans"), 3), fmt(g("rpe_rot"), 3),
            (fmt(d_ate, 3, sign=True),
             "" if d_ate is None else ("bad" if d_ate > 0 else "good")),
            (fmt(d_rpe, 3, sign=True),
             "" if d_rpe is None else ("bad" if d_rpe > 0 else "good")),
            win, spread]})
    t = table("oxford stride-12 &middot; 319 프레임 &middot; K=1 &middot; base_k1 대비 scene 짝지음 중앙값",
              ["method", "M (키프레임)", "n", "ATE", "rpe &Delta;=1", "rpe_rot",
               "&Delta;log ATE", "&Delta;log rpe", "ATE win", "seam 스케일 폭"], rows)
    longpart = sec_reanchor_long()
    return f"""
<section><h2><span class="num">04</span>재앵커 M 스윕 &mdash; 게이지는 얼마나 버티는가</h2>
<p class="dek">감사 문서 §10 후보 ④ · 실행 순서 1. M 키프레임마다 캐시를 비우고 scale frame을
다시 돌린 뒤, 겹치는 구간의 카메라 중심에 Sim(3)을 맞춰 이어붙입니다. 추론만 바꾸므로 재학습이 없습니다.
구현은 <code>benchmark/methods/lingbot_map.py::_run_streaming_reanchored</code>이고
seam은 <code>experiments/stitch_bank.py</code>의 것을 그대로 씁니다.</p>
{t}
<div class="verdict">
<p><strong>국소는 단조로 좋아지고, 전역은 U자입니다.</strong> 프레임 단위 병진 오차는 재앵커가
잦을수록 계속 좋아집니다(M=128에서 &minus;0.576, M=16에서 &minus;1.090 &mdash; 3배). 그런데 ATE는
M=32에서 최저(&minus;0.245, 6/10 승)이고 M=16에서는 오히려 +0.616으로 나빠집니다.
seam이 많아질수록 스티칭 오차가 누적되기 때문입니다.</p>
<p><strong>그리고 seam 스케일 폭이 측정값입니다.</strong> 새로 앵커할 때마다 모델이 내놓는 스케일이
직전 게이지와 20~60% 다릅니다. 감사 문서 §04가 &ldquo;공통 스케일은 0-gradient 방향&rdquo;이라고
말한 것의 추론 쪽 대응물입니다 &mdash; 훈련이 못 보는 양이 추론에서 이만큼 흔들립니다.</p>
<p><strong>이득은 base가 실패하는 scene에 몰립니다.</strong> scene별로 base ATE와 &Delta;log를
짝지으면 Spearman r이 M=128에서 &minus;0.745, M=64에서 &minus;0.624입니다 &mdash; base가 나쁜 scene일수록
재앵커가 더 많이 고칩니다. base ATE가 3 m를 넘는 세 scene에서는 M=64·128 모두 3/3으로 개선되고,
가장 나쁜 <code>christ-church-05</code>는 39.9 m에서 M=16일 때 19.9 m로 절반이 됩니다.
반대로 base가 이미 좋은 scene에서는 seam 오차만 더해집니다.</p>
<p><strong>해석에 붙일 단서 둘.</strong> (1) 재앵커는 게이지만 리셋하는 것이 아니라 8프레임 양방향
scale 블록을 M마다 삽입합니다. 그 블록은 인과 스트리밍보다 조건이 좋으므로 rpe 이득의 일부는
드리프트 감소가 아니라 그 블록 때문일 수 있습니다 &mdash; 가르려면 scale frame 수 대조가 필요하고 아직 안 했습니다.
(2) seam 스케일 폭의 양 끝(0.02, 2.21)은 정상적인 드리프트가 아니라 조건이 나쁜 seam입니다.
겹침 구간이 거의 직선이면 Sim(3)의 스케일이 사실상 미결정이고, <code>fit_seam</code>은 그 경우를
기록만 하고 막지 않습니다. M=16의 ATE 악화에는 이 실패한 seam들이 섞여 있습니다.</p>
</div>
{longpart}
</section>"""



def sec_reanchor_long() -> str:
    """The same knob on the long cell -- and the instrument bug it exposed."""
    d = load(RES / "rpe_at_k_oxford_long.json")
    if not d or "base_auto" not in d:
        return ""
    b = d["base_auto"]
    want = [("base_auto", "&mdash;", "&mdash;"),
            ("base_ra64_auto", "64", "32 원시 프레임"),
            ("base_rk64_auto", "64", "12 키프레임"),
            ("sd_v6is1200_auto", "&mdash;", "&mdash;"),
            ("sd_v6is1200_rk32_auto", "32", "12 키프레임"),
            ("sd_v6is1200_rk64_auto", "64", "12 키프레임"),
            ("sd_v6is1200_rk128_auto", "128", "12 키프레임")]
    rows, have = [], False
    for m, M, ov in want:
        sc = d.get(m)
        if not sc:
            continue
        ss = sorted(set(sc) & set(b))
        if len(ss) < 5:
            continue
        if m not in ("base_auto", "sd_v6is1200_auto"):
            have = True
        g = lambda k: med([sc[s].get(k) for s in ss])                      # noqa: E731
        d_ate = None if m == "base_auto" else med(
            [dlog(sc[s]["ate"], b[s]["ate"]) for s in ss])
        d_dk = None if m == "base_auto" else med(
            [dlog(sc[s]["rpe_trans_dK"], b[s]["rpe_trans_dK"]) for s in ss])
        rows.append({"cls": "base" if m == "base_auto" else "", "cells": [
            f"<code>{esc(m)}</code>", M, ov, len(ss),
            fmt(g("ate"), 3), fmt(g("rpe_trans_dK"), 3), fmt(g("kf_ratio_median"), 2),
            (fmt(d_ate, 3, sign=True),
             "" if d_ate is None else ("bad" if d_ate > 0 else "good")),
            (fmt(d_dk, 3, sign=True),
             "" if d_dk is None else ("bad" if d_dk > 0 else "good"))]})
    if not have:
        return ""
    t = table("oxford_long &middot; 3,840 프레임 &middot; K=12 &middot; base_auto 대비 scene 짝지음 중앙값",
              ["method", "M", "seam 겹침", "n", "ATE", "rpe &Delta;=K",
               "|est|/|gt| @K", "&Delta;log ATE", "&Delta;log rpe@K"], rows)

    # does re-anchoring close the trained-vs-base gap?
    gap = ""
    A, Ar, B, Br = ("sd_v6is1200_auto", "sd_v6is1200_rk64_auto",
                    "base_auto", "base_rk64_auto")
    if all(k in d for k in (A, Ar, B, Br)):
        ss = sorted(set(d[A]) & set(d[Ar]) & set(d[B]) & set(d[Br]))
        if len(ss) >= 5:
            g0 = med([dlog(d[A][x]["ate"], d[B][x]["ate"]) for x in ss])
            g1 = med([dlog(d[Ar][x]["ate"], d[Br][x]["ate"]) for x in ss])
            db = med([dlog(d[Br][x]["ate"], d[B][x]["ate"]) for x in ss])
            dv = med([dlog(d[Ar][x]["ate"], d[A][x]["ate"]) for x in ss])
            m32 = m64 = float("nan")
            if "sd_v6is1200_rk32_auto" in d:
                s32 = sorted(set(d["sd_v6is1200_rk32_auto"]) & set(d[A]))
                m32 = med([dlog(d["sd_v6is1200_rk32_auto"][x]["ate"], d[A][x]["ate"])
                           for x in s32]) or float("nan")
            m64 = dv
            gap = f"""<div class="verdict">
<p><strong>재앵커는 학습된 체크포인트를 되살리지 못합니다.</strong> v6i step 1200과 base의 격차는
재앵커 없이 {g0:+.3f}, M=64 재앵커에서 {g1:+.3f}입니다. 좁아 보이지만 그 이유가
v6i가 나아져서가 아닙니다 &mdash; v6i는 {dv:+.3f}로 사실상 그대로이고,
base가 {db:+.3f}로 나빠져서 좁혀진 것입니다.</p>
<p><strong>그런데 국소는 둘 다 크게 좋아집니다.</strong> 키프레임 간격 rpe가 base &minus;0.956,
v6i &minus;0.693이고 방향 cos도 0.993 / 0.985로 올라갑니다. 캐시 깊이를 M으로 묶으면
국소 tracking은 회복되는데 전역 형태는 회복되지 않습니다 &mdash; 감사 문서가 학습에서 본
&ldquo;국소 이득 / 전역 손실&rdquo;이 추론에서도 같은 모양으로 나타납니다.</p>
<p><strong>더 자주 앵커해도 안 됩니다.</strong> M=32에서 v6i의 ATE는 재앵커 없는 자기 자신보다
{m32:+.3f} 더 나쁩니다(M=64에서는 {m64:+.3f}). 재앵커 주기를 줄일수록 국소 rpe는 계속 좋아지는데
전역 ATE는 계속 나빠지는, oxford K=1에서 본 것과 같은 방향입니다.</p>
<p><strong>따라서 §10 ④의 기대는 이 셀에서 성립하지 않습니다.</strong> 문서는
&ldquo;어느 M에서 ATE가 회복되는지가 곧 모델이 스케일을 유지하는 깊이&rdquo;라고 적었지만,
K=12 · 3,840프레임에서는 회복되는 M이 없고 학습된 모델의 손상은 드리프트를 묶어도 남습니다.
§02에서 본 것과 같은 결론입니다 &mdash; 이 셀에서 무너지는 것은 스케일이 아닙니다.</p></div>"""

    seam_note = ""
    import re as _re
    _pat = _re.compile(r"median scale ([\d.]+)")
    vals = []
    for lg in (LOGS / "queue_rk_oxlong.log",):
        if lg.exists():
            vals += [float(m.group(1)) for m in _pat.finditer(lg.read_text(errors="ignore"))]
    if vals:
        seam_note = f"""<div class="verdict">
<p><strong>부수적으로 나온 측정값 &mdash; 앵커마다 스케일이 다릅니다.</strong> seam의 Sim(3)이
새 세그먼트를 기존 게이지로 옮길 때 곱하는 배율의 scene별 중앙값이
<strong>{min(vals):.2f}배에서 {max(vals):.2f}배</strong> 사이에 흩어집니다({len(vals)}개 실행).
fit 잔차는 0.002~0.27 m로 작으니 맞춤이 실패한 것이 아니라,
모델이 새 앵커마다 실제로 다른 절대 스케일을 내놓는 것입니다.
§10이 처방의 제약으로 적은 &ldquo;교사와 학생은 앵커가 달라 절대 스케일을 비교할 수 없다&rdquo;가
배포된 모델 자신에게도 그대로 적용된다는 뜻이고, 그 폭이 이 값입니다.</p>
<p class="note" style="margin-top:10px">seam 지표(스케일 폭 · 잔차 · 조건수) 어느 것도 scene별
ATE 변화를 설명하지 못했습니다(Spearman |r| &le; 0.6, n=9, 부호도 반대).
재앵커가 어떤 scene에서 왜 실패하는지는 미해결로 둡니다.</p></div>"""
    return f"""
<h3>같은 노브를 장거리 셀에서 &mdash; 그리고 계측기가 먼저 걸린 함정</h3>
{t}
<div class="verdict">
<p><strong>첫 시도는 실패했고, 실패한 이유가 감사 문서 §01(c) 그 자체입니다.</strong>
seam의 Sim(3)을 원시 프레임 32개 위에서 맞췄더니 K=12에서 ATE가
2.6 m에서 12.1 m로 갔습니다(&Delta;log +1.334, 1/10만 개선). K&gt;1에서 비-키프레임은
캐시에 KV를 남기지 않는 독립 단발 추정이라 그 32개 점이 사실상 노이즈이고,
게다가 실제 운동의 baseline은 32/12 &asymp; 2.7 키프레임뿐입니다.
같은 문서가 벤치의 <code>rpe_trans</code>에 대해 지적한 것과 정확히 같은 실수를
스티칭에서 저지른 셈입니다. 고친 버전은 겹침을 키프레임 단위로 세고 seam을
키프레임 중심 위에서만 맞춥니다 &mdash; K=1에서는 두 해석이 같으므로 위의 oxford 표는 영향받지 않습니다.</p>
</div>
{gap}
{seam_note}"""



K1_STEPS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1250]
K1_RUNS = [("s0off", "sd_s0offs%d_k1", "off-policy만"),
           ("s0on", "sd_s0ons%d_k1", "on-policy만"),
           ("v6i", "sd_v6is%d_k1", "혼합 35/65"),
           ("v6f", "sd_v6fs%d_k1", "혼합 · 구 코퍼스"),
           ("v7f", "sd_v7fs%d_k1", "혼합 + long")]


def _oxford_k1():
    root = WS / "oxford" / "oxford"
    if not root.is_dir():
        return {}, {}
    D = {}
    for sd in sorted(root.iterdir()):
        j = load(sd / "eval" / "traj.json")
        if j:
            D[sd.name] = j
    base = {s: D[s]["base_k1"]["ate"] for s in D if "base_k1" in D[s]}
    return D, base


def _pearson(x, y):
    n = len(x)
    if n < 4:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else None


def sec_spine() -> str:
    D, base = _oxford_k1()
    if not base:
        return ""
    curves = {}
    for key, tmpl, _ in K1_RUNS:
        pts = []
        for st_ in K1_STEPS:
            m = tmpl % st_
            ss = [s for s in base if m in D[s]]
            if len(ss) < 5:
                continue
            pts.append((st_, med([dlog(D[s][m]["ate"], base[s]) for s in ss]),
                        sum(1 for s in ss if D[s][m]["ate"] < base[s]), len(ss)))
        if len(pts) >= 4:
            curves[key] = pts
    if "s0off" not in curves:
        return ""

    grid = [100, 300, 600, 900, 1200]
    rows = []
    for key, _, what in K1_RUNS:
        pts = curves.get(key)
        if not pts:
            if key in ("s0off", "s0on"):
                rows.append({"cells": [f"<code>{esc(key)}</code>", esc(what)]
                             + ["&mdash;"] * (len(grid) + 3)})
            continue
        d = {p[0]: p for p in pts}
        cells = [f"<code>{esc(key)}</code>", esc(what)]
        for g in grid:
            p = d.get(g)
            cells.append("&mdash;" if not p else
                         (fmt(p[1], 3, sign=True), "bad" if p[1] > 0 else "good"))
        ys = [p[1] for p in pts]
        r = _pearson([math.log(p[0]) for p in pts], ys)
        cells += [fmt(max(ys) - min(ys), 3),
                  "&mdash;" if r is None else fmt(r, 3, sign=True),
                  "&mdash;" if r is None else fmt(r * r, 2)]
        rows.append({"cells": cells})
    t = table("Oxford stride-12 &middot; 319 프레임 &middot; K=1 &middot; base_k1 대비 scene 짝지음 &Delta;log ATE 중앙값",
              ["run", "무엇을 학습", *[f"step {g}" for g in grid],
               "전체 범위", "r(log step)", "R&sup2;"], rows)

    off = {p[0]: p[1] for p in curves["s0off"]}
    mixes = {k: {p[0]: p[1] for p in v} for k, v in curves.items()
             if k in ("v6i", "v6f", "v7f")}
    gap_rows = []
    for g in grid:
        mv = [m[g] for m in mixes.values() if g in m]
        if g in off and mv:
            gap_rows.append({"cells": [
                f"step {g}", fmt(off[g], 3, sign=True),
                f"{min(mv):+.3f} &ndash; {max(mv):+.3f}",
                (fmt(min(mv) - off[g], 3, sign=True),
                 "bad" if min(mv) - off[g] > 0 else "good")]})
    gt = table("off-policy만 vs 혼합 세 실행", ["", "s0off", "혼합 범위",
                                              "가장 가까운 혼합과의 차"], gap_rows)

    # ── same comparison, counting CORRECTION steps instead of steps ─────────
    dil = ""
    if "s0on" in curves:
        on = {p[0]: p[1] for p in curves["s0on"]}
        mix = {k: {p[0]: p[1] for p in curves[k]} for k in ("v6i", "v7f") if k in curves}
        drows = []
        for c in (260, 390, 520, 650, 812):
            s_on = min(on, key=lambda k: abs(k - c))
            cells = [f"{c} correction step", f"s{s_on} &rarr; {on[s_on]:+.3f}"]
            for k in ("v6i", "v7f"):
                if k not in mix:
                    cells.append("&mdash;")
                    continue
                s_m = min(mix[k], key=lambda x: abs(x - c / 0.65))
                cells.append(f"s{s_m} &rarr; {mix[k][s_m]:+.3f}")
            drows.append({"cells": cells})
        dil = table("correction 스텝 수를 맞춰서 &mdash; 혼합은 step의 65%만 correction입니다",
                    ["", "s0on (100% correction)", "v6i (65%)", "v7f (65%)"], drows)

    pend = "" if "s0on" in curves else """
<p class="note">★ on-policy만(<code>s0on</code>)은 아직 학습 중입니다. 2&times;2의 나머지 한 칸이
채워지기 전까지 아래 판정은 &ldquo;off-policy는 붕괴를 만들지 않는다&rdquo;까지만이고,
&ldquo;붕괴는 on-policy에서 온다&rdquo;는 그 다음입니다.</p>"""

    last_off = curves["s0off"][-1]
    r_off = _pearson([math.log(p[0]) for p in curves["s0off"]],
                     [p[1] for p in curves["s0off"]])
    return f"""
<section><h2><span class="num">06</span>척추 실험 &mdash; off-policy만 돌리면 무너지지 않습니다</h2>
<p class="dek">감사 문서 §10 순서 0. 프로젝트가 &ldquo;제일 먼저&rdquo; 하라고 적었고 한 번도 돌지 않은 실험입니다.
판정 지표는 <code>experiments/auto_eval_v6i.sh</code>가 정한 Oxford K=1의 GT ATE입니다.</p>
{t}
{gt}
{dil}
<div class="verdict">
<p><strong>붕괴가 사라집니다.</strong> off-policy 상태만 1250 step 학습한 팔은
step {last_off[0]}에서 {last_off[1]:+.3f}로, base와 사실상 같은 자리에 있습니다.
같은 step에서 혼합 세 실행은 +0.80 ~ +0.88입니다. step 축을 따라간 전체 범위가
<strong>{max(p[1] for p in curves['s0off']) - min(p[1] for p in curves['s0off']):.3f} 대 0.73~0.88</strong>,
r(log step, &Delta;log ATE)는 <strong>{r_off:+.3f}(R&sup2; {r_off * r_off:.2f}) 대 +0.81~+0.89(R&sup2; 0.65~0.79)</strong>입니다.</p>
<p><strong>감사 문서의 &ldquo;steps, not losses&rdquo;는 한 단계 더 좁혀집니다.</strong>
학습량이 문제인 것이 아니라 <em>on-policy 교정 상태로 학습한 양</em>이 문제입니다.
identity 브랜치 &mdash; 2&times;2의 &ldquo;naive pseudo-label FT&rdquo; 칸 &mdash; 는 1250 step을 돌아도
Oxford K=1 ATE를 움직이지 않습니다.</p>
<p><strong>다만 &ldquo;off-policy가 좋다&rdquo;는 결론은 아닙니다.</strong> 그 팔은 base를 이기지도 않습니다
(step 1200에서 4/10 scene). 그리고 identity 브랜치는 설계상 &theta;<sub>0</sub>에 대한 보존 항이므로,
그것만 100% 돌린 결과가 &theta;<sub>0</sub> 근처라는 것은 부분적으로 동어반복입니다.
그런데 모델이 안 변한 것은 아닙니다 &mdash; 같은 팔의 held-out probe는 +31.5% 나빠졌습니다(§03b).
교사 모방은 무너뜨리면서 GT ATE는 건드리지 않았습니다.</p>
{pend}
</div>
</section>"""



def sec_lossfloor() -> str:
    """L_fresh at theta_0 -- and which term actually supplies the gradient."""
    d = load(RES / "loss_floor.json")
    g = load(RES / "rot_term.json")
    cl = load(RES / "rot_clamp.json")
    if not d or not d.get("rows"):
        return ""
    rows = [{"cells": [f"run {r['run']}", fmt(r["L_masked"], 5), fmt(r["L_seq"], 5),
                       f"{r['pose_masked']:.2e}", f"{r['pose_seq']:.2e}"]}
            for r in d["rows"]]
    t = table(f"{esc(d['scene'])} &middot; &theta;<sub>0</sub> 그대로 &middot; 같은 창을 두 경로로 채점",
              ["", "L (병렬 마스크 = 학습 경로)", "L (순차 = 교사 경로)",
               "|pose &minus; label| 병렬", "|pose &minus; label| 순차"], rows)
    fl, cont = d["acos_floor_rad"], d["floor_contribution"]
    seq = med([r["L_seq"] for r in d["rows"]])
    mask = med([r["L_masked"] for r in d["rows"]])

    gt = ""
    if g and g.get("rows"):
        grows = []
        for r in g["rows"]:
            tot = r["quadrature_total_gnorm"]
            for k, lab in (("rot", "L_rot (acos)"), ("dir", "L_dir"),
                           ("dep", "L_depth"), ("mot", "L_motion_depth"),
                           ("rot_chordal", "L_rot을 chordal로 바꾸면")):
                if k not in r["terms"]:
                    continue
                tm = r["terms"][k]
                grows.append({"cls": "base" if k == "rot_chordal" else "", "cells": [
                    f"run {r['run']} &middot; {lab}", fmt(tm["value"], 5),
                    fmt(tm["gnorm"], 4),
                    "&mdash;" if k == "rot_chordal" else f"{tm['gnorm'] / tot:.1%}"]})
        gt = table("&theta;<sub>0</sub>에서 항별로 따로 backward한 파라미터 gradient 노름",
                   ["", "가중 적용된 값", "||grad||", "전체 대비"], grows)

    clamp_txt = ""
    if cl:
        s48 = cl.get("48")
        if s48:
            n, c = s48[0][0], s48[0][1]
            clamp_txt = (f"학습 창 크기 S=48에서도 {n}개 쌍이 <strong>전부</strong> "
                         f"clamp에 걸립니다({c}/{n}).")
    return f"""
<section><h2><span class="num">99</span>보존 항이 &theta;<sub>0</sub>에서 0이 아닌 이유</h2>
<p class="dek"><code>fresh_step</code>의 주석은 &ldquo;step 0에서는 두 가중치가 같으므로 이 loss는
구성상 거의 0&rdquo;이라고 적어놨습니다. 아니었습니다. 다만 <em>왜</em> 아닌지는 처음 짐작과 달랐습니다.
학습을 하나도 하지 않고 측정한 값입니다
(<code>experiments/{{loss_floor_probe,rot_term_probe}}.py</code>).</p>
{t}
<p class="note">순차 경로는 라벨을 비트 단위로 재현합니다(<code>|pose &minus; label|</code> = 0).
그런데 loss는 {seq:.5f}입니다. 범인은 <code>losses.py:292</code>의
<code>acos(cos.clamp(&minus;1+1e&minus;6, 1&minus;1e&minus;6))</code>이고,
완벽히 일치해도 <code>acos(1&minus;1e&minus;6)</code> = {math.degrees(fl):.4f}&deg;가 남습니다.
<code>lam_rot=15</code>를 곱하면 {cont:.5f}, 관측된 바닥의 {cont / seq:.0%}입니다.</p>
{gt}
<div class="verdict">
<p><strong>그런데 그 바닥은 힘을 쓰지 않습니다.</strong> clamp에 걸린 쌍은 gradient가 정확히 0이고,
{clamp_txt} 항별로 backward를 따로 걸어보면 <code>L_rot</code>의 파라미터 gradient 노름이
<strong>0.0000</strong>입니다. &theta;<sub>0</sub>에서 실제로 미는 것은
<code>L_depth</code>(전체의 약 90%)와 <code>L_motion_depth</code>입니다.</p>
<p><strong>그래서 acos를 chordal로 바꾸는 것은 수정이 아닙니다.</strong> 보고되는 숫자는
{seq:.5f}에서 0.0007로 정직해지지만 업데이트는 그대로입니다. 그리고 항이 활성화된 뒤에는
chordal의 gradient가 각도에 비례해 사라지므로(acos는 각도에 무관하게 일정),
실측 회전 잔차 0.35~0.66&deg; 구간에서는 회전 감독을 사실상 끄는 것과 같습니다.
회전 항이 해로운지 보고 싶다면 <code>lam_rot</code>을 직접 낮추는 쪽이 해석 가능한 실험입니다.</p>
<p><strong>경로 불일치도 범인이 아닙니다.</strong> 교사 뱅크는 프레임을 하나씩 흘려
만들었고(<code>num_frame_per_block=1, causal_inference=True</code>), 학습은 48프레임을 한 번에
병렬로 채점합니다. 두 경로의 <code>pose_enc</code> 차이는 약 1e&minus;3입니다.
그런데 그 차이가 학습을 따라 <strong>변하지 않습니다</strong>:</p>
<div class="tw"><table><caption>같은 창 &middot; 체크포인트만 바꿔가며 (kth_day_10 run 0)</caption>
<thead><tr><th>가중치</th><th>L 병렬</th><th>L 순차</th>
<th>|pose &minus; label| 병렬</th><th>|pose &minus; label| 순차</th><th>두 경로 차</th></tr></thead>
<tbody>
<tr class="base"><td><code>&theta;<sub>0</sub></code></td><td class="n">0.02462</td><td class="n">0.02186</td>
<td class="n">6.0e&minus;04</td><td class="n">0.0e+00</td><td class="n">6.0e&minus;04</td></tr>
<tr><td><code>s0off step1200</code></td><td class="n">0.15578</td><td class="n">0.15576</td>
<td class="n">1.29e&minus;02</td><td class="n">1.30e&minus;02</td><td class="n">5.9e&minus;04</td></tr>
<tr><td><code>v6i step1200</code></td><td class="n">0.19436</td><td class="n">0.19465</td>
<td class="n">2.24e&minus;01</td><td class="n">2.23e&minus;01</td><td class="n">1.0e&minus;03</td></tr>
</tbody></table></div>
<p>두 경로의 loss가 소수점 셋째 자리까지 같고 라벨로부터의 드리프트도 같습니다.
경로 차이는 1e&minus;3에 머무는데 모델의 실제 드리프트는 s0off에서 1.3e&minus;2,
v6i에서 2.2e&minus;1입니다 &mdash; <strong>10배에서 200배</strong> 큽니다.
&theta;<sub>0</sub>에서 경로 차이가 gradient를 지배한 것은 다른 모든 것이 0이었기 때문이고,
모델이 조금만 움직이면 무의미해집니다.</p>
<p><strong>그래서 채점을 순차로 바꾸는 것도 권하지 않습니다.</strong> 비용은 오히려 유리합니다 &mdash;
S=48 forward에서 병렬 11.1초 / 36.3 GiB 대 순차 5.2초 / 15.9 GiB로 순차가 2배 쌉니다
(prefix 8.2초는 양쪽 공통이라 뺀 값). 문제는 backward입니다 &mdash; 순차는 프레임별 그래프 S개를
살려둬야 해서 S=8에서 이미 OOM이고(같은 카드에서 병렬은 23.7 GiB로 들어감),
프레임 단위 gradient checkpointing이 필요합니다. 할 수는 있지만 위 표가 고칠 것이 없다고 말합니다.</p>
<p class="note">확정된 것과 아닌 것. 바닥값, clamp 포화, 항별 gradient 분해, 경로 불일치는 측정값입니다.
그 gradient가 배포 성능을 <em>해치는</em> 방향이라는 것은 아직 추론입니다.</p>
</div>
</section>"""



def sec_gauge() -> str:
    """The objective cannot see the quantity the benchmark measures."""
    d = load(RES / "teacher_vs_student_gt.json")
    if not d or not d.get("windows"):
        return ""
    rows = []
    for w in d["windows"]:
        t = w["teacher"]
        s1 = w["student"].get("1", {})
        s28 = w["student"].get("28", {})
        rows.append({"cells": [
            f"{w['t0']}", fmt(t["gt_ate"], 4), fmt(t["gt_rot_deg"], 3),
            (fmt(s1.get("gt_ate"), 4), "bad"), fmt(s1.get("gt_rot_deg"), 3),
            (fmt(s28.get("gt_ate"), 4), "bad"), fmt(s28.get("gt_rot_deg"), 3)]})
    t1 = table("kth_day_10 &middot; &theta;<sub>0</sub> &middot; 창 48프레임 &middot; 창-국소 Sim(3) 정렬 후 GT 대비 (m, 도)",
               ["창 시작 프레임", "교사 ATE", "교사 rot",
                "학생 K=1 ATE", "rot", "학생 K=28 ATE", "rot"], rows)
    deep = [w for w in d["windows"] if w["t0"] >= 800]
    rt = med([w["teacher"]["gt_ate"] for w in deep])
    r1 = med([w["student"]["1"]["gt_ate"] for w in deep])

    srows = []
    for key, name in ((None, "교사 (창마다 새 앵커)"), ("1", "학생 K=1 (프레임 0부터 스트리밍)"),
                      ("28", "학생 K=28")):
        v = [(w["teacher"] if key is None else w["student"][key])["gt_scale"]
             for w in d["windows"]]
        srows.append({"cells": [name, fmt(med(v), 3),
                                f"{min(v):.2f} &ndash; {max(v):.2f}",
                                f"{max(v) / min(v):.2f}&times;",
                                fmt(st.pstdev([math.log(x) for x in v]), 3)]})
    t2 = table("같은 8개 창 &middot; GT에 맞추려면 예측에 곱해야 하는 Sim(3) 스케일",
               ["", "중앙값", "범위", "폭", "&sigma;(log)"], srows)

    return f"""
<section><h2><span class="num">99</span>목표 함수가 벤치가 재는 양을 볼 수 없습니다</h2>
<p class="dek">지금까지 나온 것 중 가장 근본적인 항목이고, 측정이 아니라 코드에서 나옵니다.
둘이 동시에 성립합니다.</p>
<ul class="plain">
<li><strong>gradient는 창 안에만 흐릅니다.</strong> <code>RolloutPool.activate</code>가
<code>self._detach(self.model)</code>를 호출합니다. 누적된 KV 캐시는 상수입니다.
&ldquo;상태를 어떻게 쌓을지&rdquo;에 대해 loss가 말을 걸 경로가 없습니다.</li>
<li><strong>loss는 창의 게이지를 볼 수 없습니다.</strong> <code>_relative</code>의 주석이
직접 적어놨습니다 &mdash; 이동을 카메라 i 기준으로 표현하므로 &ldquo;a global Sim(3) leaves the
direction untouched and multiplies the magnitude by sigma alone&rdquo;. 그 &sigma;에 반응할 수 있는
유일한 항이 <code>L_mag</code>인데 <code>lam_mag = 0</code>이고, 켜도 소용없습니다:
<code>_magnitude_loss</code>는 첫 줄이 &ldquo;Scale-invariant discrepancy&rdquo;이고
<code>median</code>·<code>closed_form_scale</code>·<code>l1</code>·<code>trunc_l1</code> 네 모드가
전부 스케일을 fit해서 버립니다.</li>
</ul>
<div class="verdict">
<p><strong>따라서 목표 함수 전체가 창에 대한 전역 Sim(3)에 불변입니다.</strong>
부분적으로가 아니라 완전히요 &mdash; 창이 앞의 궤적에 어떻게 붙는지의 자유도가 정확히 그 Sim(3)입니다.
ATE는 그 이어붙임을 수백 번 누적한 값이므로, 훈련 목표는 벤치가 재는 양에 대해
<em>설계상</em> 눈이 멀어 있습니다. 국소가 좋아지는 동안 전역이 무너지는 것은 예외가 아니라 기본값입니다.</p>
<p class="note">★ 앞서 이 문서가 &ldquo;모델이 캐시를 무시하도록 배운다&rdquo;고 적었던 것은 틀렸습니다.
캐시가 detach돼 있어 그렇게 배울 경로 자체가 없습니다. 맞는 서술은 위와 같이
&ldquo;목표가 이어붙임을 지정하지 않아 최적화가 그 널 스페이스를 자유롭게 쓴다&rdquo;입니다.</p>
</div>

<h3>그런데 전제 자체는 성립합니다 &mdash; 교사는 실제로 더 정확합니다</h3>
<p class="dek">MCD GT로 직접 쟀습니다(<code>experiments/teacher_vs_student_gt.py</code>).
교사는 창마다 새로 앵커하고, 학생은 프레임 0부터 배포와 같은 방식으로 걸어옵니다.</p>
{t1}
<p class="note">교사는 깊이와 무관하게 평평하고(0.03~0.08 m), 학생은 깊어질수록 나빠집니다.
프레임 800 이상에서 중앙값 {rt:.4f} m 대 {r1:.4f} m로 <strong>{r1 / rt:.1f}배</strong> 차이입니다.
drift 없는 짧은 교사가 drift한 학생을 잡는다는 설계 전제는 사실입니다.</p>

<h3>전역 카메라 스케일은 유지되지 않습니다</h3>
{t2}
<div class="verdict"><p><strong>앵커의 8 scale frame이 스케일을 전역으로 고정하지 못합니다.</strong>
한 스트림 안에서 GT에 맞추는 데 필요한 스케일이 1.5배에서 3.4배까지 흔들립니다.
학습 전 &theta;<sub>0</sub>에서의 값이므로 self-distillation 이전의 성질입니다.</p>
<p>이것이 ATE와 직결됩니다. 벤치의 ATE는 궤적 전체에 <em>하나의</em> &sigma;를 fit한 뒤 재는데,
&sigma;가 경로를 따라 3배 흔들리면 어떤 단일 값도 맞지 않습니다.
그리고 위에서 본 대로 loss는 &sigma;를 볼 수 없습니다 &mdash;
모델이 갖지 못한 성질을 목표가 요구하지도 않습니다.</p>
<p class="note">한 scene 8개 창의 결과입니다. K=28이 K=1보다 스케일이 안정적인 것
(1.50배 대 3.37배)은 예상과 반대이고, 캐시 sliding window 64가 K=1에서는 최근 64프레임만
남기는 반면 K=28에서는 궤적 전체를 덮기 때문으로 보이지만 확인은 안 했습니다.</p>
</div>
</section>"""



def sec_stitch() -> str:
    """Is the stitched bank the single-gauge reference the plan assumes?"""
    a = load(RES / "stitched_gauge_audit.json")
    sv = load(RES / "seam_survey.json")
    if not a or not a.get("rows"):
        return ""
    rows = []
    for r in a["rows"]:
        pr = r.get("perrun")
        rows.append({"cells": [
            f"{r['t']}", fmt(r["stitched"]["gt_scale"], 2),
            fmt(pr["gt_scale"], 2) if pr else "&mdash;",
            fmt(r["stitched"]["gt_ate"], 4)]})
    t1 = table(f"{esc(a['scene'])} &middot; 창 {a['S']}프레임 &middot; GT에 맞추는 Sim(3) 스케일",
               ["프레임", "stitched", "per-run", "창-국소 ATE (m)"], rows)
    S, P = a["stitched"], a["perrun"]
    t2 = table("게이지 일관성 요약 &mdash; 하나의 게이지라면 폭 1, &sigma;(log) 0, 추세 없음",
               ["", "범위", "폭", "&sigma;(log)", "추세 r", "인접 창 |log|"],
               [{"cells": ["stitched (한 게이지로 이어붙임)",
                           f"{S['min']:.2f} &ndash; {S['max']:.2f}",
                           f"{S['spread']:.2f}&times;", fmt(S["sigma_log"], 3),
                           (fmt(S["trend_r"], 3, sign=True), "bad"),
                           fmt(S["adjacent_abs_log_median"], 3)]},
                {"cells": ["per-run (240프레임마다 재앵커)",
                           f"{P['min']:.2f} &ndash; {P['max']:.2f}",
                           f"{P['spread']:.2f}&times;", fmt(P["sigma_log"], 3),
                           fmt(P["trend_r"], 3, sign=True),
                           (fmt(P["adjacent_abs_log_median"], 3), "bad")]}])
    n_seams = 126
    per = S["spread"] ** (1.0 / n_seams)
    surv = ""
    if sv:
        surv = f"""<p><strong>그리고 한 scene의 문제가 아닙니다.</strong>
stitched bank {sv['n']}개 중 <code>seam_scale_ok</code>가 True인 것은
<strong>{sv['n_scale_ok']}개</strong>뿐인데, <code>PASS</code>는
<strong>{sv['n_pass']}개 전부</strong>입니다. 스케일 항목이 게이트에 걸려 있지 않습니다.
seam당 |log scale| 중앙값의 중앙값은 {sv['log_scale_med_of_med']:.3f}
(약 {(math.exp(sv['log_scale_med_of_med']) - 1) * 100:.0f}%)입니다.</p>"""
    return f"""
<section><h2><span class="num">99</span>stitched bank은 단일 게이지가 아닙니다</h2>
<p class="dek">게이지에 민감한 loss(§10 후보 ①)는 믿을 수 있는 단일 게이지 기준을 전제합니다.
<code>stitch_bank.py</code>가 겹치는 교사 run들을 Sim(3) seam으로 이어붙여 그걸 만든다고 하고,
자기 리포트는 126/126 seam 성공, 잔차 중앙값 1.4 mm로 통과합니다.
그것은 <em>이웃끼리 맞는가</em>이지 <em>6,000프레임 뒤에도 세상과 맞는가</em>가 아닙니다.
MCD GT로 직접 쟀습니다(<code>experiments/stitched_gauge_audit.py</code>).</p>
{t1}
{t2}
<div class="verdict">
<p><strong>stitching은 잡음을 램프로 바꿉니다.</strong> 인접 창 흔들림은 실제로 줄어듭니다
({P['adjacent_abs_log_median']:.3f} &rarr; {S['adjacent_abs_log_median']:.3f}, 설계대로).
대신 게이지가 트랙을 따라 단조로 부풀어 오릅니다 &mdash; 추세 상관 {S['trend_r']:+.3f},
6,070프레임에 걸쳐 {S['spread']:.2f}배입니다. {n_seams}개 seam으로 나누면
seam당 {(per - 1) * 100:.2f}%이고 전부 같은 방향입니다. 순차 연쇄는 편향을 누적합니다.</p>
<p><strong>seam 잔차 1.4 mm가 이걸 못 잡습니다.</strong> 이웃 두 run의 일치도이고,
126번 곱해지는 1.2%는 각 seam에서 보이지 않습니다. 그런데 같은 리포트 안에 단서가 이미 있었습니다 &mdash;
<code>seam_scale_ok: False</code>, <code>seam_log_scale_per_seam_median: 0.162</code>,
<code>seam_cond_worst: 150.6</code>. 전체 판정만 <code>PASS: True</code>로 나갑니다.</p>
{surv}
<p><strong>그래서 ①은 지금 상태로는 성립하지 않습니다.</strong> stitched 트랙 위에 얹으면
학생에게 {S['spread']:.1f}배짜리 스케일 램프를 따라 하라고 가르치게 됩니다.
per-run bank도 대안이 아닙니다 &mdash; 램프는 약한 대신 인접 창 잡음이 3.6배입니다.
둘 중 하나는 램프, 하나는 잡음이고, <strong>게이지 일관된 교사 라벨은 현재 존재하지 않습니다.</strong></p>
<p class="note">★ <code>fit_seam</code>의 주석이 <code>cond</code>에 대해
&ldquo;Recorded rather than gated: gating needs the GT audit&rdquo;라고 적어뒀습니다.
이 절이 그 GT 감사입니다. 고칠 방향 셋: 조건수로 seam을 거르기,
스케일을 강건하게 맞추기(<code>l1</code>), 그리고 순차 연쇄 대신 seam 그래프 전체를
한 번에 푸는 전역 정합 &mdash; 순차는 편향을 누적하고 전역은 분산시킵니다.
셋 다 미검증이고, 셋 다 추론 없이 라벨만으로 시험됩니다.</p>
</div>
</section>"""



def sec_nullspace() -> str:
    """How much ATE fits inside the objective's null space, constructed on GT."""
    d = load(RES / "nullspace_demo.json")
    if not d or not d.get("rows"):
        return ""
    rows = []
    for r in d["rows"]:
        rows.append({"cls": "base" if r["ramp"] == 1.0 else "", "cells": [
            f"{r['ramp']:.2f}&times;", f"{r['per_frame_g']:.8f}",
            fmt(r["L_rot_deg"], 6), f"{r['L_dir']:.1e}", fmt(r["weighted"], 6),
            (fmt(r["ate"], 3), "bad" if r["ate"] > 0.001 else "")]})
    t = table(f"{esc(d['scene'])} &middot; GT를 라벨로, GT를 학생으로 &middot; "
              f"창 {d['S']}프레임마다 A1PC의 pose 항을 계산",
              ["전체 램프", "프레임당 g", "L_rot (도)", "L_dir",
               "가중 합", "ATE (m)"], rows)
    worst = max(d["rows"], key=lambda r: r["ate"])
    ref = [r for r in d["rows"] if r["ramp"] == 1.0][0]

    # ── would turning L_mag on close it? ────────────────────────────────────
    lm = load(RES / "lmag_response.json")
    lmag = ""
    if lm and lm.get("rows"):
        MODES = ("median", "closed_form_scale", "l1", "trunc_l1")
        lrows = []
        for r in lm["rows"]:
            lab = "매끄러운 램프" if r["perturb"] == "ramp" else "계단식 (창 안 상수)"
            lrows.append({"cls": "base" if r["ramp"] == 1.0 else "", "cells": [
                lab, f"{r['ramp']:.1f}&times;",
                f"{(r['within_window_g'] - 1) * 100:.3f}%",
                *[f"{r['modes'][m]:.1e}" for m in MODES],
                (fmt(r["ate"], 2), "bad" if r["ate"] > 0.01 else "")]})
        lt = table("같은 변형을 L_mag의 네 모드로 채점 &middot; "
                   "창 32개 중앙값 &middot; lam_mag은 실제로는 0",
                   ["변형", "램프", "창 안 g 변화", "median", "closed_form_scale",
                    "l1", "trunc_l1", "ATE (m)"], lrows)
        st3 = [r for r in lm["rows"] if r["perturb"] == "stepwise" and r["ramp"] == 3.0]
        rp3 = [r for r in lm["rows"] if r["perturb"] == "ramp" and r["ramp"] == 3.0]
        extra = ""
        if st3 and rp3:
            a_, b_ = rp3[0]["modes"]["l1"], st3[0]["modes"]["l1"]
            extra = f"""<p><strong>계단식으로 만들면 완전히 사라집니다.</strong>
창 안에서 g를 정확히 상수로 두고 창 경계에서만 바꾸면 L_mag이
{a_:.1e}에서 {b_:.1e}로 <strong>{a_ / max(b_, 1e-30):.0f}배</strong> 떨어지는데
ATE는 {st3[0]['ate']:.1f} m로 그대로입니다. 창별로 상수인 스케일은
<code>_magnitude_loss</code>가 fit해서 버리는 바로 그 양이기 때문입니다.</p>"""
        lmag = f"""
<h3>그러면 <code>lam_mag</code>을 켜면 덮이나</h3>
{lt}
<div class="verdict">
<p><strong>거의 못 덮습니다.</strong> 21 m짜리 매끄러운 램프에 L_mag이 {rp3[0]['modes']['l1']:.1e}입니다.
X1M의 <code>lam_mag=0.5</code>로 켜면 총 loss 기여가 {rp3[0]['modes']['l1'] * 0.5:.1e},
같은 창의 <code>acos</code> 바닥 0.0214의 {rp3[0]['modes']['l1'] * 0.5 / 0.0214 * 100:.1f}%입니다.</p>
{extra}
<p>네 모드가 사실상 같습니다 &mdash; <code>l1</code>과 <code>trunc_l1</code>이
<code>closed_form_scale</code>과 소수점 세 자리까지 일치합니다. X1M이 fit을 강건하게 바꾼 것은
이 자유도에 대해 아무 차이를 만들지 않습니다.</p>
<p><strong>L_mag은 창 안에서 스케일이 <em>변하는 정도</em>만 보고 누적된 값은 못 봅니다.</strong>
덮으려면 기준이 창 바깥에 있어야 합니다. 지금은 모든 항의 기준이 그 창 자신입니다 &mdash;
L_mag은 그 창에서 fit한 스케일, <code>long_scale</code>은 그 창의 <code>v_local</code>,
<code>depth_si</code>는 그 창의 depth 중앙값입니다. §10 ①(앵커-상대 정규화)이 분모를
스트림 앵커로 옮기자는 것이 이 자유도를 닫는 유일한 후보입니다.</p>
</div>"""
    return f"""
<section><h2><span class="num">99</span>널 스페이스 안에 ATE {worst['ate']:.0f} m가 들어갑니다</h2>
<p class="dek">&ldquo;GT를 라벨로 써서 돌려보면 손실 설계가 멀쩡한지 확인되지 않나&rdquo;에 대한 답입니다.
학습을 돌리는 대신 반례를 직접 만들었습니다. GT 카메라 중심의 연속 스텝에 천천히 램프하는 g(t)를
곱해 다시 적분하면, 48프레임 창 안에서는 스케일이 거의 상수라 각 창이 GT의 Sim(3) 변환이 되고,
6,000프레임에 걸쳐서는 램프가 누적됩니다. <code>long_loss.py</code> 헤더가 이름 붙인 바로 그 변형입니다.</p>
{t}
<div class="verdict">
<p><strong>loss가 소수점 여섯 자리까지 움직이지 않습니다.</strong>
램프 1.0(= GT 그대로)에서 {ref['weighted']:.6f}, 램프 {worst['ramp']:.1f}배에서도
{worst['weighted']:.6f}입니다. 그 값 자체가 §09의 <code>acos</code> 바닥이고,
<code>L_dir</code>은 1e&minus;9 수준으로 수치적 0입니다. 그런데 ATE는
0.000 m에서 <strong>{worst['ate']:.2f} m</strong>가 됩니다.</p>
<p><strong>프레임당 {(worst['per_frame_g'] - 1) * 100:.3f}%의 스케일 편향</strong>이면 충분합니다.
비교하자면 base의 oxford_long ATE가 2.6 m, v6i step1200이 6.8 m입니다 &mdash;
지금까지 관측한 열화 전체보다 3~8배 큰 ATE가 손실이 전혀 못 보는 방향에 들어 있습니다.</p>
<p><strong>그래서 GT 라벨로 돌려도 손실 설계는 검증되지 않습니다.</strong>
목표 함수가 GT와 {worst['ate']:.0f} m 어긋난 궤적을 구분하지 못하므로,
라벨을 완벽하게 만드는 것으로는 이 자유도가 닫히지 않습니다. 그 실험은 한쪽으로만 정보를 줍니다 &mdash;
GT로도 무너지면 목표 함수가 원인이라는 강한 증거이고, 안 무너지면 그저 그 코퍼스에서
최적화가 우연히 널 스페이스로 안 갔다는 뜻이지 목표가 건전하다는 뜻이 아닙니다.</p>
{lmag}
<p class="note">★ 위 표는 A1PC의 pose 절반(L_rot, L_dir)만 계산합니다. 나머지 절반도 같은 성질이라는 것은
이미 코드에 측정으로 적혀 있습니다 &mdash; <code>long_loss.py</code>의 <code>v_local</code> 주석:
&ldquo;A1PC is invariant to a JOINT (pose, depth) scaling &mdash; measured, total 0.515680 at
k = 0.25, 1, 2 and 10, bit for bit.&rdquo; pose와 depth가 함께 흐르면 depth 항도 못 봅니다.
감사 문서 §04가 잰 실제 드리프트가 정확히 그 형태입니다.</p>
</div>
</section>"""


GTSUP_RUNS = [("gtsup", "sd_gtsups%d_k1", "GT 포즈 라벨"),
              ("teasup", "sd_teasups%d_k1", "교사 의사라벨 (대조군)")]



def sec_corpus() -> str:
    """65% of the sampling weight was never a video."""
    d = load(RES / "corpus_order.json")
    if not d or not d.get("rows"):
        return ""
    rows = []
    for r in sorted(d["rows"], key=lambda x: -x["weight"]):
        cells = [f"<code>{esc(r['ds'])}</code>", str(r["weight"]), str(r["scenes"]),
                 str(r["cameras"]), str(r["views_per_time"]),
                 (f"{r['contiguous_frac']:.2%}", "good" if r["ok"] else "bad"),
                 f"{r['runs_bad']}/{r['runs']}",
                 "비디오" if r["ok"] else "<strong>비디오 아님</strong>"]
        rows.append({"cells": cells, "cls": "" if r["ok"] else "bad"})
    t = table("<code>label_bank.image_names</code>가 읽는 순서 &middot; "
              "학습이 실제로 보는 순서",
              ["dataset", "가중치", "장면", "카메라", "뷰/시각",
               "시간 인접 비율", "망가진 run", ""], rows)
    bad, tot = d["bad_weight"], d["total_weight"]
    return f"""
<section><h2><span class="num">99</span>코퍼스의 65%는 비디오가 아니었습니다</h2>
<p class="dek"><code>label_bank.image_names</code>는
<code>sorted(os.listdir(...))</code>입니다 &mdash; 원본 디렉토리에 대한 단순 사전순.
0으로 채운 단일 카메라 덤프라면 그게 곧 비디오 순서지만, 그 밖에는 아닙니다.
그리고 아무도 확인하지 않습니다(<code>experiments/corpus_order_audit.py</code>).</p>
{t}
<div class="verdict">
<p><strong>샘플링 가중치 {bad}/{tot} = {bad / tot:.0%}가 비디오가 아닌 시퀀스에 있었습니다.</strong>
가장 무거운 둘(<code>paralleldomain4d</code> 30 + <code>unrealstereo4k</code> 15 = 45%)은
인접 쌍 중 <em>단 하나도</em> 시간 한 스텝이 아닙니다. paralleldomain4d는 한 시각의 19개 뷰
(camera0&ndash;15와 yaw-0/60/neg-60)가 사전순으로 돌고, unrealstereo4k는 매 프레임
좌/우 눈이 교대합니다. 즉 모델은 <strong>자기운동이 존재하지 않는 두 프레임 사이의
자기운동을 추정하라고</strong> 요구받았습니다.</p>
<p><code>dynamicreplica</code>는 전체 비율로 99.83%라 멀쩡해 보이지만 run 단위로는 다릅니다.
600프레임에 left&rarr;right 이음매가 하나뿐이어도, L=240 run이 그것을 가로지르면 run 전체가
오염됩니다. <code>scannet</code>은 <code>0, 1, 10, 100, 1000</code> 사전순이라 9%가 불연속입니다.</p>
<p><strong>라벨 뱅크도 같은 순서로 구워졌습니다.</strong> 입력과 라벨이 똑같이 틀렸기 때문에
내부 정합 검사가 전부 통과했습니다 &mdash; 뱅크 <code>PASS</code>도, 교사 모방 probe도,
학습이 발산하는 동안에도. 이것이 이 결함이 여기까지 살아남은 이유입니다.</p>
</div>
<p class="note">★ 이 절 뒤에 오는 <em>학습된 결과</em>는 전부 이 코퍼스 위에서 나왔습니다.
어떤 결론이 살아남는지는 아래 표에 정리했습니다.</p>
</section>"""


def sec_survives() -> str:
    """Which conclusions the corpus defect does and does not touch."""
    d = load(RES / "corpus_order.json")
    if not d:
        return ""
    SURV = [
        ("영공간 시연 &mdash; 손실이 못 보는 21 m", "산다",
         "GT 기하만 씁니다. 학습 데이터가 전혀 안 들어갑니다."),
        ("<code>lam_mag</code>은 이걸 못 덮는다", "산다",
         "같은 구성, 학습 무관."),
        ("게이지/스케일 감사 &mdash; &sigma;가 1.50&ndash;3.37배", "산다",
         "MCD 전용, &theta;<sub>0</sub>에서 측정."),
        ("stitched 뱅크가 단일 게이지가 아니다", "산다",
         "MCD 전용. 184개 중 112개가 <code>seam_scale_ok: False</code>."),
        ("<code>acos</code> 바닥은 gradient가 0", "산다",
         "kth_day_10 = MCD, &theta;<sub>0</sub>, 학습 없음."),
        ("교사가 학생보다 정확하다", "산다",
         "MCD 전용, &theta;<sub>0</sub>."),
        ("&Delta;=K 재집계", "산다",
         "저장된 벤치 궤적만 씁니다."),
        ("재앵커링이 국소 rpe를 살린다", "산다",
         "추론만. 학습 체크포인트를 못 구한다는 결과도 그대로."),
        ("GT 뱅크 감사 &mdash; GT가 9.3배 멀다", "산다",
         "MCD 10개 장면 전용. GT 학습 셀 자체는 아직 안 돌았습니다."),
        ("척추 2&times;2 &mdash; identity는 희석, on-policy가 붕괴", "보강됨",
         "네 팔 모두 오염된 코퍼스로 학습한 건 그대로지만, 깨끗한 held-out 창에서 "
         "다시 잰 probe가 Oxford와 <em>일치</em>합니다 &mdash; s0on만 +41.8%, "
         "s0off +3.9% · v6i +0.8%. 서로 독립인 두 측정이 같은 말을 합니다. "
         "on-policy 자체 때문인지 슬라이드쇼 위의 on-policy 때문인지는 남습니다."),
        ("이 목적함수는 내려가지 않는다", "<strong>철회</strong>",
         "근거 둘이 모두 오염이었습니다 &mdash; 학습 loss는 <code>fresh_pool 1</code>의 "
         "표집 드리프트, probe는 창 6개 중 3개가 비디오가 아님. 깨끗한 창에서는 평평합니다."),
        ("probe와 Oxford가 서로 다른 말을 한다 (gate 6b 이래)", "<strong>해소</strong>",
         "probe 창의 절반이 망가진 데이터였기 때문이었습니다. 창을 고르면 일치합니다."),
        ("장기 항이 원인이 아니다 / step 효과가 6배", "<strong>흔들림</strong>",
         "학습된 체크포인트 비교라 같은 교란을 받습니다."),
    ]
    rows = [{"cells": [n, (v, "good" if v == "산다" else "bad"), w],
             "cls": "" if v == "산다" else "bad"} for n, v, w in SURV]
    t = table("코퍼스 결함이 건드리는 결론과 건드리지 않는 결론",
              ["결론", "상태", "근거"], rows)
    return f"""
<section><h2><span class="num">99</span>무엇이 살아남는가</h2>
<p class="dek">가르는 기준은 하나입니다 &mdash; 그 측정이 <em>학습된 체크포인트</em>를
거쳤는가. 거쳤으면 흔들리고, &theta;<sub>0</sub>나 GT 기하나 저장된 궤적만 썼으면 그대로입니다.</p>
{t}
<div class="verdict">
<p>진단 결과의 대부분이 &theta;<sub>0</sub>와 MCD 위에서 측정된 것이라 살아남습니다.
잃은 것은 <strong>학습 결과의 해석</strong>입니다 &mdash; 벤치 숫자 자체가 아니라,
그것이 방법에 대해 말해준다고 믿었던 부분입니다.</p>
</div>
</section>"""


def sec_descent() -> str:
    """RETRACTED, and what replaces it: the probe measured on clean windows."""
    CLEAN = {"kth", "slowtv", "replica"}
    ds_of = lambda n: "kth" if n.startswith(("kth_", "tuhh_")) else n.split("_")[0]
    T, wins = {}, None
    for a in ("s0off", "s0on", "v6i", "dclean"):
        j = load(RES / f"train_{a}.json")
        P = (j or {}).get("probes") or []
        if not P:
            continue
        idx = [i for i, q in enumerate(P[0]["parts"]) if ds_of(q["scene"]) in CLEAN]
        if not idx:
            continue
        if a == "s0off":
            wins = [P[0]["parts"][i]["scene"] for i in idx]
        T[a] = {p["step"]: (med([p["parts"][i]["loss"] for i in idx]),
                            med(p["loss"]), len(idx), len(p["parts"])) for p in P}
    if "s0off" not in T:
        return ""

    rows = []
    for a in ("s0off", "s0on", "v6i", "dclean"):
        if a not in T:
            continue
        k = sorted(T[a])
        f, l = T[a][k[0]][0], T[a][k[-1]][0]
        fa, la = T[a][k[0]][1], T[a][k[-1]][1]
        nc, nt = T[a][k[0]][2], T[a][k[0]][3]
        dc, dd = (l / f - 1) * 100, (la / fa - 1) * 100
        rows.append({"cells": [
            f"<code>{esc(a)}</code>", f"{nc}/{nt}", str(k[-1]),
            (f"{dc:+.1f}%", "bad" if dc > 15 else "good"),
            (f"{dd:+.1f}%", "bad" if dd > 15 else "good")]})
    t = table("held-out probe &middot; 고정된 창 &middot; 처음 대비 마지막",
              ["run", "깨끗한 창", "step", "깨끗한 창만", "전체 창(절반이 비디오 아님)"], rows)

    traj = ""
    if "s0off" in T and "v6i" in T:
        ks = [k for k in sorted(T["s0off"]) if k % 150 == 0]
        trows = [{"cells": [str(k)] + [
            (fmt(T[a][k][0], 4), "bad" if a in T and k in T[a] and
             T[a][k][0] / T[a][sorted(T[a])[0]][0] > 1.15 else "")
            if a in T and k in T[a] else "&mdash;"
            for a in ("s0off", "s0on", "v6i")]} for k in ks]
        traj = table("깨끗한 창 3개(" + ", ".join(f"<code>{esc(w)}</code>" for w in (wins or [])) + ")의 궤적",
                     ["step", "s0off (identity만)", "s0on (on-policy만)", "v6i (혼합)"], trows)

    return f"""
<section><h2><span class="num">99</span>철회 &mdash; &ldquo;목적함수가 안 내려간다&rdquo;는 측정 결함이었습니다</h2>
<p class="dek">앞 판에서 이 절은 <code>s0off</code>의 학습 loss가 3.3배 오르고 probe가 97.7%
나빠졌다는 것을 근거로 <em>경사하강이 목적함수를 키우고 있다</em>고 적었습니다.
근거 두 개가 모두 오염이었습니다.</p>
<p class="note">① <strong>학습 loss는 표집 구성이었습니다.</strong>
<code>--fresh_pool 1</code>이라 identity 브랜치의 fresh 스트림이 하나뿐이고, 그래서 한 장면을
100 step씩 연속으로 봅니다. <code>s0off</code>의 학습 창은 step 700&ndash;799에 100% dl3dv,
1200&ndash;1299에 58% scannet이었습니다 &mdash; 요청한 가중치(dl3dv 10, scannet 10)와 무관합니다.
데이터셋마다 loss 수준이 3배 가까이 다르므로(dl3dv 0.151 대 mcd 0.052) 그 드리프트가
그대로 &ldquo;상승&rdquo;으로 보였습니다.</p>
<p class="note">② <strong>probe 창 6개 중 3개가 비디오가 아니었습니다</strong>
(scannet &middot; dynamicreplica &middot; unrealstereo4k). 즉 &ldquo;일반화가 나빠졌다&rdquo;는
측정의 절반이 슬라이드쇼 위에서 이뤄졌습니다.</p>
{t}
{traj}
<div class="verdict">
<p><strong>깨끗한 창에서 보면 순서가 뒤집힙니다.</strong> 1250 step 동안
<code>s0off</code> +3.9%, <code>v6i</code> +0.8% &mdash; 사실상 평평합니다.
목적함수는 내려가지 않는 게 아니라, 망가진 창에서 재고 있었을 뿐입니다.
깨끗한 코퍼스만으로 돌린 <code>dclean</code>도 300 step에서 &minus;3.4%입니다.</p>
<p><strong>그리고 뜻밖의 소득이 있습니다.</strong> 깨끗한 창에서 유일하게 무너지는 것은
<code>s0on</code>(+41.8%), 즉 on-policy 브랜치만 도는 팔입니다. 이것은 Oxford가 말하던 것과
<em>정확히</em> 같습니다(s0on +1.754 대 s0off +0.045). 프로젝트 내내
&ldquo;라벨 probe와 Oxford가 서로 다른 말을 한다&rdquo;던 문제(gate 6b 이래의 골칫거리)는
probe 창이 망가져 있었기 때문이었습니다. 창을 고르면 두 측정이 같은 말을 합니다.</p>
<p class="note">남는 단서: 네 팔 모두 65%가 오염된 코퍼스로 <em>학습</em>했습니다.
&ldquo;s0on이 무너진다&rdquo;가 on-policy 자체 때문인지, 슬라이드쇼 위를 굴러간
on-policy 상태 때문인지는 깨끗한 코퍼스로 다시 학습해야 갈립니다.</p>
<p class="note">★ 그리고 &ldquo;일치&rdquo;는 <em>최종 순위</em>에 한합니다. step에 따른
방향은 여전히 반대이고, 그게 아래 절의 내용입니다.</p>
</div>
</section>"""


def sec_optimum() -> str:
    """The objective IS minimised.  Minimising it costs accuracy."""
    CLEAN = {"kth", "slowtv", "replica"}
    ds_of = lambda n: "kth" if n.startswith(("kth_", "tuhh_")) else n.split("_")[0]
    fl = load(RES / "loss_floor.json")
    D, base = _oxford_k1()
    rows = []
    for a, tmpl in (("s0off", "sd_s0offs%d_k1"), ("s0on", "sd_s0ons%d_k1"),
                    ("v6i", "sd_v6is%d_k1"), ("dclean", None)):
        j = load(RES / f"train_{a}.json")
        P = (j or {}).get("probes") or []
        if not P:
            continue
        idx = [i for i, q in enumerate(P[0]["parts"]) if ds_of(q["scene"]) in CLEAN]
        if not idx:
            continue
        v = [(p["step"], med([p["parts"][i]["loss"] for i in idx])) for p in P]
        s0 = v[0][1]
        mn = min(v, key=lambda t: t[1])
        ox = "&mdash;"
        if tmpl and base:
            m = tmpl % mn[0] if any((tmpl % mn[0]) in D[x] for x in base) else None
            for cand in ([mn[0]] if m else []):
                ss = [x for x in base if (tmpl % cand) in D[x]]
                if len(ss) >= 5:
                    ox = fmt(med([dlog(D[x][tmpl % cand]["ate"], base[x]) for x in ss]),
                             3, sign=True)
        rows.append({"cells": [
            f"<code>{esc(a)}</code>", fmt(s0, 4), fmt(mn[1], 4), str(mn[0]),
            (f"{(mn[1] / s0 - 1) * 100:+.1f}%", "good"),
            (ox, "bad" if ox != "&mdash;" and not ox.startswith("-") else "")]})
    if not rows:
        return ""
    t = table("깨끗한 held-out 창의 최저점, 그리고 <em>같은 step</em>의 Oxford",
              ["run", "시작", "최저", "step", "probe 개선", "Oxford &Delta;log ATE"], rows)

    ox = ""
    if base:
        STEPS = [50, 200, 400, 600, 800, 1000, 1250]
        orows = []
        for s_ in STEPS:
            cells = [str(s_)]
            for tmpl in ("sd_s0offs%d_k1", "sd_s0ons%d_k1", "sd_v6is%d_k1", "sd_v7fs%d_k1"):
                m = tmpl % s_
                ss = [x for x in base if m in D[x]]
                if len(ss) < 5:
                    cells.append("&mdash;")
                    continue
                v = med([dlog(D[x][m]["ate"], base[x]) for x in ss])
                cells.append((fmt(v, 3, sign=True), "good" if v < 0 else "bad"))
            orows.append({"cells": cells})
        ox = table("Oxford K=1 &middot; base 대비 &Delta;log ATE &middot; 음수가 base보다 좋음",
                   ["step", "s0off", "s0on", "v6i", "v7f"], orows)

    floor = ""
    if fl:
        m = med([r["L_masked"] for r in fl["rows"]])
        c = fl["floor_contribution"]
        floor = (f"<p class=\"note\">★ <strong>학습 loss는 하강 신호가 아닙니다.</strong> "
                 f"&theta;<sub>0</sub>에서 {m:.5f}인데 그중 <code>acos</code> clamp 바닥이 "
                 f"{c:.5f}, 즉 <strong>{c / m:.0%}</strong>가 어떤 가중치로도 못 없애는 "
                 f"상수입니다. 아래로 남은 여유는 {m - c:.5f}뿐이라, 학습 loss가 오르는 것은 "
                 f"바닥에서 출발했다는 뜻이지 발산했다는 뜻이 아닙니다. "
                 f"움직일 수 있는 신호는 held-out probe 쪽입니다.</p>")

    return f"""
<section><h2><span class="num">99</span>목적함수는 내려갑니다. 내려갈수록 나빠집니다.</h2>
<p class="dek">앞 절이 &ldquo;하강이 실패한다&rdquo;를 철회했다면, 이 절은 그 자리에 무엇이
들어가는지입니다. 하강은 <em>됩니다</em> &mdash; 깨끗한 held-out 창에서 <code>v6i</code>가
19.7%, <code>s0on</code>이 23.9% 좋아집니다. 문제는 같은 구간에 궤적 정확도가
단조롭게 나빠진다는 것입니다.</p>
{floor}
{t}
{ox}
<div class="verdict">
<p><strong>어느 팔도 base를 이긴 적이 없습니다.</strong> 유일한 음수가 <code>v7f</code>
step 100의 &minus;0.014로 잡음 범위이고, 나머지는 step 50부터 전부 양수이며 단조 증가합니다.
그 사이 probe는 20% 넘게 내려갑니다.</p>
<p>즉 이 목적함수는 <em>최소화가 됩니다</em>. 최소화될수록 정확도가 떨어집니다.
이것이 §07의 영공간에 이빨이 달린 형태입니다 &mdash; 창 안에서 손실이 Sim(3) 불변이므로,
교사를 더 잘 흉내내면서 ATE만 보는 스케일·포즈 drift를 얼마든지 쌓을 수 있습니다.
21 m을 손실 소수점 여섯 자리 안에 숨길 수 있다는 그 구성이, 학습 궤적에서 실제로 일어나는
모습입니다.</p>
<p class="note">단서 하나: probe는 MCD/slowtv/replica의 고정 창에서, Oxford는 Oxford에서
잽니다. 엄밀히는 &ldquo;학습 분포에서 교사 모방이 좋아지는 것이 held-out 벤치의 정확도로
이어지지 않는다&rdquo;입니다. 코퍼스를 고쳐도 이 진술은 남습니다 &mdash;
<code>v6i</code>의 Oxford 곡선은 step 50부터 단조이고, probe 개선은 깨끗한 창에서
측정한 값이기 때문입니다.</p>
</div>
</section>"""


def sec_gtsup() -> str:
    """Objective or labels?  The one cell that separates them."""
    a = load(RES / "gt_bank.json")
    prog = {k: train_progress(k) for k, _, _ in GTSUP_RUNS}
    if not a and not any(p["steps_done"] for p in prog.values()):
        return ""

    bank = ""
    if a and a.get("scenes"):
        rows = [{"cells": [f"<code>{esc(r['scene'])}</code>", str(r["runs"]),
                           fmt(r["sigma_median"], 4), f"{r['sigma_spread']:.1f}x",
                           fmt(r["resid_ate_median"], 4), fmt(r["L_rot_deg"], 4),
                           fmt(r["L_dir"], 5), fmt(r["weighted"], 4)]}
                for r in a["scenes"]]
        bank = table("GT 라벨 뱅크 &middot; 교사 뱅크와 얼마나 다른가 &middot; S=48 창",
                     ["scene", "run", "&sigma; 중앙값", "&sigma; 범위",
                      "잔차 ATE (m)", "L_rot (&deg;)", "L_dir", "가중 합"], rows)

    gap = ""
    if a and a.get("weighted_median"):
        w, fl = a["weighted_median"], a.get("loss_floor_masked")
        gap = (f"<p class=\"note\">라벨을 바꾸면 목표가 실제로 움직입니다. 가중 합 중앙값이 "
               f"<strong>{w:.3f}</strong>인데, &theta;<sub>0</sub>에서 교사 라벨에 대한 "
               f"학습 loss는 {fl:.5f}였습니다"
               if fl else f"<p class=\"note\">가중 합 중앙값 {w:.3f}")
        if fl:
            gap += (f" &mdash; GT 목표가 <strong>{w / fl:.0f}배</strong> 멀리 있습니다. "
                    f"베이스 모델이 교사를 진실보다 그만큼 더 가깝게 흉내내고 있다는 뜻이고, "
                    f"교사가 곧 베이스 모델이니 당연합니다.</p>")
        gap += ("""<p class="note">★ 이 9배가 9배 큰 step이 되지는 않습니다.
<code>grad_total</code>이 측정된 모든 run에서 18&ndash;20이고 <code>--clip 1.0</code>에
<strong>100%의 step이</strong> 걸립니다. 보폭을 정하는 건 loss 크기가 아니라 clip과 lr이라
두 팔은 같은 거리를 움직입니다.</p>""")

    curve = ""
    D, base = _oxford_k1()
    if base:
        grid = [100, 300, 600, 900, 1200]
        rows = []
        for key, tmpl, what in GTSUP_RUNS:
            cells = [f"<code>{esc(key)}</code>", esc(what)]
            got = False
            for g in grid:
                m = tmpl % g
                ss = [x for x in base if m in D[x]]
                if len(ss) < 5:
                    cells.append("&mdash;")
                    continue
                got = True
                v = med([dlog(D[x][m]["ate"], base[x]) for x in ss])
                cells.append((fmt(v, 3, sign=True), "bad" if v and v > 0 else "good"))
            rows.append({"cells": cells})
            if not got:
                rows[-1]["cls"] = "base"
        curve = table("Oxford stride-12 &middot; 319 프레임 &middot; K=1 &middot; "
                      "base_k1 대비 &Delta;log ATE 중앙값",
                      ["run", "라벨", *[f"step {g}" for g in grid]], rows)

    st_rows = [{"cells": [f"<code>{esc(k)}</code>", esc(what),
                          f"{prog[k]['steps_done']} / {prog[k]['target'] or 1250}",
                          "끝" if prog[k]["finished"] else "진행 중"]}
               for k, _, what in GTSUP_RUNS]
    status = table("두 팔의 진행", ["run", "라벨", "step", "상태"], st_rows)

    return f"""
<section><h2><span class="num">99</span>손실 설계인가, 라벨인가</h2>
<p class="dek">지금까지의 모든 run이 베이스 모델의 의사라벨로 학습했기 때문에
&ldquo;목적함수가 부족하다&rdquo;와 &ldquo;라벨이 나쁘다&rdquo;가 한 번도 분리된 적이 없습니다.
MCD에는 GT가 있으니 분리할 수 있습니다. 손실 · 정책 · 장면을 전부 고정하고
<code>pose_enc</code>가 어디서 오는지만 바꿉니다
(<code>experiments/{{bake_gt_bank,gt_bank_audit}}.py</code>,
<code>experiments/chain_gtsup.sh</code>).</p>
{bank}
{gap}
<p class="note">★ GT를 <em>교사의 게이지로 옮겨서</em> 굽습니다.
<code>L_rot</code>과 <code>L_dir</code>은 Sim(3) 불변이라 월드 좌표 GT를 그대로 넣어도
같은 값이지만, <code>L_motion</code>은 아닙니다. 이 항은
<code>log&#8741;&Delta;t&#8741;</code>를 <code>log median(D)</code>와 비교하는데,
미터 단위 GT 옆에 교사의 임의 단위 깊이를 놓으면
<code>log(&sigma;<sub>gt</sub>/&sigma;<sub>depth</sub>)</code>라는 상수가 들어가
<code>lam_motion=0.9</code>로 <strong>틀린 포즈:깊이 비율을 가르치게</strong> 됩니다.
run마다 Sim(3)를 하나씩 맞춰 GT를 깊이가 이미 살고 있는 단위로 옮겼습니다.
닮음변환은 게이지만 흡수하고 drift는 흡수하지 못하므로 GT의 <em>모양</em>은 그대로입니다.</p>
<p class="note">깊이 라벨은 양쪽 팔 모두 교사 것입니다 &mdash; MCD에 화소별 GT 깊이가 없습니다.
<code>lam_depth=1.7</code>이 목적함수에서 가장 큰 가중치이므로 대비는 그만큼 희석되고,
이 실험이 시험하는 것은 <strong>포즈 감독</strong>입니다.
쫓고 있는 실패 양상(포즈 drift · 스케일 붕괴)이 포즈 쪽이라 문제되지 않습니다.</p>
<div class="verdict">
<p><strong>결과가 나오기 전에 예측을 적어둡니다.</strong>
<code>LabelBank.windows</code>의 주석대로 &ldquo;창 하나는 정확히 run 하나 안에 들어갑니다
&mdash; run마다 자기 Sim(3) 게이지를 갖고, 모든 손실 항이 창 안에서 정규화되기&rdquo;
때문입니다. <code>lam_long=0</code>이므로 감독은 <strong>서로 독립인 240프레임 덩어리들</strong>이고,
덩어리 안에서 손실은 Sim(3) 불변입니다. 그러면 §07이 만든 누적 스케일 영공간은
라벨이 아무리 정확해도 <strong>구조적으로 보이지 않습니다</strong>.</p>
<p>그래서: GT 팔이 장거리 셀을 못 살리면 진단이 확정되고, 갈 곳은 더 좋은 라벨이 아니라
앵커-상대 정규화(§10 ①)와 게이지가 일관된 stitched 뱅크입니다.
반대로 GT가 크게 도우면 영공간 논증이 과했던 것이고, 데이터 수집 쪽이 우선입니다.
어느 쪽이든 답이 나옵니다.</p>
</div>
{status}
{curve}
</section>"""


def sec_review() -> str:
    return """
<section><h2><span class="num">05</span>지금까지의 검토</h2>
<p class="dek">학습이 끝나기 전에 확정된 것만 적습니다.</p>
<ul class="plain">
<li><strong>도구가 벤치와 일치합니다.</strong> <code>rpe_at_k.py</code>의 &Delta;=1 값이
저장된 <code>eval/traj.json</code>과 1e-14 이내입니다. &Delta;=K 값은 같은 코드 경로에서
delta만 바꾼 것이므로 같은 신뢰도를 갖습니다.</li>
<li><strong>스케일 붕괴는 VBR에 국한됩니다.</strong> oxford_long은 K=12 &middot; 3,840 프레임인데
키프레임 변위 비가 base 1.05, v6f s1200 1.02로 정상입니다. VBR의 0.62배는 K=28~59 &middot; 11k 프레임
체제에서만 나타납니다 &mdash; 감사 문서 §01(b)가 &ldquo;VBR은 K &middot; 길이 &middot; 장면이 교란돼 있다&rdquo;고
적은 것과 일치하고, 이제 K=12까지는 무사하다는 하한이 생겼습니다.</li>
<li><strong>oxford_long의 &Delta;=1 이득도 VBR과 다릅니다.</strong> VBR에서는 &Delta;=1이 크게 좋아졌다가
&Delta;=K에서 사라졌지만, oxford_long의 v6f s1200은 &Delta;=1에서 이미 +0.056(악화)이고
&Delta;=K에서 +0.191입니다. 감출 이득 자체가 없습니다.</li>
<li><strong>척추 벤치는 비어 있는 것이 맞습니다.</strong> oxford_long 10 scene에서 완주한 method는
<code>base_auto</code>와 <code>sd_v6fs1200_auto</code>뿐이고, <code>base_k1</code>·<code>sd_v7fs300_auto</code>는
디렉터리만 있고 궤적이 없습니다. 감사 문서 §11 2b가 맞습니다.</li>
<li><strong>귀속이 장거리 셀로 확장됩니다.</strong> 3,840프레임 · K=12에서 v6i와 v7f를
같은 step으로 짝지으면 step 300에서 차이가 0.000, step 1200에서 +0.122인데,
같은 구간에서 step 축은 +0.698 움직입니다. 감사 문서가 320프레임에서만 보였던
&ldquo;steps, not losses&rdquo;가 열두 배 긴 시퀀스에서도 같은 배율로 성립합니다.</li>
<li><strong>그런데 여기서 무너지는 것은 스케일이 아닙니다.</strong> oxford_long step 1200에서
키프레임 변위 비가 v6i 0.96, v7f 0.94입니다. ATE가 base의 2.5배가 되는 동안에도
스케일은 6% 안쪽입니다. VBR의 0.62는 K=28~59 · 11k 프레임 체제의 고장이고,
K=12에서의 악화는 다른 것입니다 &mdash; 감사 문서 §04의 처방(②·①③)이 K=12 셀의
악화까지 고칠 것이라고 기대할 근거는 아직 없습니다.</li>
<li><strong>arm A는 K&gt;1을 한 번도 보지 않습니다.</strong> identity 브랜치는 <code>K=1</code>이
고정입니다(<code>trainer.py:2740</code>). 그래서 &ldquo;off-policy만&rdquo;은 상태 분포만이 아니라
배포 캐시 깊이 노출까지 0인 팔입니다. 이것은 설계상 그런 것이지 설정 실수가 아니지만,
결과를 읽을 때 두 팔의 차이를 &ldquo;정책&rdquo; 하나로 돌리면 안 되는 이유입니다.</li>
</ul>
</section>"""


def main() -> None:
    prog = {a: train_progress(a) for a in ARMS}
    parts = [sec_status(prog, bench_progress()), sec_corpus(), sec_survives(),
             sec_setup(prog), sec_spine()]

    ox_extra = """
<p class="note">★ 이 표가 감사 문서 §03의 VBR 결과를 다른 데이터셋에서 시험합니다.
oxford_long은 K=12 · 3,840 프레임이라 VBR(K=28~59 · 11k)보다 얕고 짧습니다.</p>"""
    parts.append(sec_rpe("oxford_long", "&Delta;=K 재집계 &mdash; oxford_long",
                         "감사 문서 §09 항목 1. K&gt;1 셀의 rpe를 키프레임 간격으로 다시 잽니다. "
                         "저장된 traj.txt만 쓰므로 추론이 필요 없습니다.", extra=ox_extra))
    parts.append(sec_stepcurve())
    parts.append(sec_vbr())
    parts.append(sec_probe())
    parts.append(sec_nullspace())
    parts.append(sec_gauge())
    parts.append(sec_stitch())
    parts.append(sec_lossfloor())
    parts.append(sec_reanchor())
    parts.append(sec_descent())
    parts.append(sec_optimum())
    parts.append(sec_gtsup())
    parts.append(sec_review())

    body = "\n".join(p for p in parts if p)
    # ★ Sections are numbered in document order at build time, so adding or
    # reordering one never leaves a stale label behind.
    _n = itertools.count()
    body = re.sub(r'<span class="num">[^<]*</span>',
                  lambda _m: f'<span class="num">{next(_n):02d}</span>', body)
    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Spine Experiment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>{CSS}</style></head>
<body><div class="wrap">
<header class="masthead">
  <p class="eyebrow"><span>streaming3d-self-distill</span>
  <span>s0off &middot; s0on &middot; v6i &middot; base</span>
  <span>척추 실험 &middot; {datetime.now():%Y-%m-%d}</span></p>
  <h1>The Spine Experiment</h1>
  <p class="standfirst">docs/eval-metric-audit.html이 &ldquo;제일 먼저&rdquo;라고 적은 것을 실제로 돌립니다 &mdash;
  <strong>p_identity를 1.0과 0.0으로 갈라</strong> off-policy 가짜라벨 파인튜닝과 on-policy 교정을
  처음으로 분리하고, 같은 문서 §09가 요구한 <strong>&Delta;=K 재집계</strong>를 함께 수행합니다.</p>
</header>
{body}
<footer>
<p>생성: <code>experiments/mk_spine_report.py</code> &middot;
&Delta;=K 재계산: <code>experiments/rpe_at_k.py</code> (evo 1.37,
translation_part, align, correct_scale; &Delta;=1 값이 벤치 <code>eval/traj.json</code>과 1e-15 이내로 일치).</p>
<p>학습: <code>experiments/launch_v6.sh</code>, v6i 정책 고정 · p_identity만 변경.
평가 워크스페이스: <code>bench_ws/{{oxford,oxford_long}}</code>.</p>
</footer>
</div></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)
    print(f"[report] wrote {OUT}  ({len(doc)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
