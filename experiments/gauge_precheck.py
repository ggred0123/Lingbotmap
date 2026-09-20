#!/usr/bin/env python3
"""Alignment pre-check for L_abs -- docs/gtabs-plan.md §5-3 (1) and (2).  CPU.

(1) ORIENTATION AUDIT, every run of every training scene.  The GT bank was
    baked with a centre-only Umeyama (bake_gt_bank.py), so its orientations sit a
    CONSTANT rotation away from the teacher's own -- 14.4 deg on kth_day_10 run 0.
    L_abs-rot can only be trusted where that constant is really constant: the
    two-sided Procrustes fit (abs_loss.procrustes_two_sided) takes the constant
    out, and what is left is the SPREAD.  Runs with spread > --rot_spread_max
    are written to ``abs_rot_exclude`` for the trainer's --abs_rot_exclude.
    The centre residual (ATE after Sim(3)) is recomputed alongside the bank's
    own ``gt_resid_ate`` as a consistency check.

(2) PREFIX vs HIST.  Two ways to fit the run gauge (abs_loss.fit_run_gauge):
    nailed to the run's first 48 frames, or refit on the whole history.  Both
    are run on (a) a synthetic scale ramp over a real GT run and (b) the saved
    Oxford K=1 trajectories of base / v6i s1200 / s0on s1250 against GT, and the
    residual is tabulated per window offset.  Selection rule (plan §5-3-2): the
    residual should grow with offset, be largest at the deepest window, and not
    blow up at the shallow ones.  The verdict goes to ``abs_fit_default``.

    .venv-bench/bin/python experiments/gauge_precheck.py
    -> experiments/results/gauge_precheck.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lingbot_map.train.abs_loss import (                                # noqa: E402
    RunGaugeLoss, fit_run_gauge, geodesic_np, median_step, procrustes_rotation,
    procrustes_two_sided, quat_to_R_np, rot_angle_deg, umeyama)
from mcd_gt import mat_to_quat                                          # noqa: E402

SCENES = ["kth_day_10", "kth_night_01", "kth_night_04", "kth_night_05",
          "tuhh_day_02", "tuhh_day_03", "tuhh_day_04", "tuhh_night_07",
          "tuhh_night_08", "tuhh_night_09"]
OXFORD_WS = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford/oxford"
OXFORD_METHODS = ["base_k1", "sd_v6is1200_k1", "sd_s0ons1250_k1"]
S = 48
OFFSETS = [0, 48, 96, 144, 192]


def bank_poses(root: str):
    idx = json.load(open(os.path.join(root, "index.json")))
    out = []
    for r in idx["runs"]:
        z = np.load(os.path.join(root, r["file"]))
        out.append((r, z["pose_enc"].astype(np.float64)))
        z.close()
    return idx, out


# ── (1) orientation audit ────────────────────────────────────────────────────

def audit_scene(scene: str, spread_max: float):
    _, tea = bank_poses(os.path.join(ROOT, "labels", scene))
    idx, gt = bank_poses(os.path.join(ROOT, "labels", scene + "_gt"))
    rows = []
    for (r, pt), (rg, pg) in zip(tea, gt):
        assert r["t0"] == rg["t0"] and r["L"] == rg["L"]
        s, R, t, sig = umeyama(pg[:, :3], pt[:, :3])     # GT-bank -> teacher (should be ~identity)
        al = (s * (R @ pg[:, :3].T)).T + t
        ate = float(np.sqrt(((al - pt[:, :3]) ** 2).sum(1).mean()))
        Rt, Rg = quat_to_R_np(pt[:, 3:7]), quat_to_R_np(pg[:, 3:7])
        # one-sided (world) constant and its spread -- the plan's numbers
        Rw = procrustes_rotation(Rg, Rt)
        left = np.degrees(geodesic_np(Rw[None] @ Rg, Rt))
        # two-sided constants -- what the loss actually removes
        RL, RC, resid = procrustes_two_sided(Rg, Rt, R_L0=R)
        two = np.degrees(geodesic_np(np.einsum("ij,njk,kl->nil", RL, Rg, RC), Rt))
        raw = np.degrees(geodesic_np(Rg, Rt))
        rows.append({
            "rid": len(rows), "t0": r["t0"], "L": r["L"],
            "ate": ate, "bank_gt_resid_ate": rg.get("gt_resid_ate"), "sigma": s,
            "rot_raw_med_deg": float(np.median(raw)),
            "rot_const_world_deg": rot_angle_deg(Rw),
            "rot_spread_left_deg": float(left.mean()),
            "rot_world_deg": rot_angle_deg(RL), "rot_cam_deg": rot_angle_deg(RC),
            "rot_spread_two_deg": float(two.mean()),
            "rot_spread_two_max_deg": float(two.max()),
            "cond": float(sig[1] / max(sig[0], 1e-30)),
            "exclude_abs_rot": bool(two.mean() > spread_max),
        })
    return rows


# ── (2) prefix vs hist ───────────────────────────────────────────────────────

def pose_enc(c, q):
    pe = np.zeros((len(c), 9)); pe[:, :3] = c; pe[:, 3:7] = q; pe[:, 7:] = 1.0
    return torch.from_numpy(pe).float()


def ramped(pose, ramp):
    C = pose[:, :3].double().numpy()
    steps = C[1:] - C[:-1]
    g = np.exp(np.linspace(0, math.log(ramp), len(steps)))[:, None]
    C2 = np.concatenate([C[:1], C[:1] + np.cumsum(steps * g, 0)])
    return torch.cat([torch.from_numpy(C2), pose[:, 3:].double()], -1).float()


def offset_curve(stu, ref, run_t0=0, offsets=OFFSETS, huber=0.1):
    """residuals of the scale term and the absolute-position term per window
    offset, under both fits.  ``stu``/``ref`` [L, 9] indexed by run offset."""
    hist = {run_t0 + k: stu[k] for k in range(stu.shape[0])}
    u = median_step(ref[:, :3])
    crit = RunGaugeLoss(lam_trans_scale=1.0, lam_abs_pos=1.0, huber_delta=huber)
    out = {}
    for mode in ("prefix", "hist"):
        cur = []
        for off in offsets:
            if off + S > stu.shape[0]:
                continue
            fit = fit_run_gauge(hist, run_t0, off, S, stu[off:off + S],
                                lambda lo, hi: ref[lo:hi], mode=mode, u=u)
            _, p = crit(stu[off:off + S], ref[off:off + S], None, None, None, fit, u)
            cur.append({"off": off, "mode": fit.mode, "s": float(fit.s), "n": fit.n,
                        "trans_scale": p["L_trans_scale"], "trans_resid": p["abs_trans_resid"],
                        "abs_pos": p["L_abs_pos"]})
        out[mode] = cur
    return out


def load_traj(path):
    a = np.loadtxt(path)
    M = a[:, 1:].reshape(-1, 3, 4)
    return M[:, :, :3], M[:, :, 3]


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r) if r == r else 0.0


def judge(curves, key="trans_resid"):
    """plan §5-3-2: monotone in offset, largest at the deepest offset, no blow-up
    at the shallow ones.  Returns per-mode scores and the pick."""
    verdict = {}
    for mode, rows in curves.items():
        rows = [r for r in rows if r["off"] > 0]            # offset 0 is a window fit for both
        if len(rows) < 2:
            verdict[mode] = {"score": -1}
            continue
        xs, ys = [r["off"] for r in rows], [r[key] for r in rows]
        rho = spearman(xs, ys)
        deepest = ys[-1] >= max(ys) - 1e-12
        blowup = ys[0] > 1.5 * ys[-1]                      # shallow already above deep
        verdict[mode] = {"rho": rho, "deepest_is_max": bool(deepest), "shallow_blowup": bool(blowup),
                         "score": rho + (1.0 if deepest else 0.0) - (1.0 if blowup else 0.0),
                         "resid": dict(zip(map(str, xs), ys))}
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--rot_spread_max", type=float, default=3.0)
    ap.add_argument("--ramp_scene", default="kth_day_10")
    ap.add_argument("--ramp_runs", type=int, nargs="*", default=[1, 3, 5])
    ap.add_argument("--ramps", type=float, nargs="*", default=[1.1, 1.3, 1.5])
    ap.add_argument("--out", default="experiments/results/gauge_precheck.json")
    a = ap.parse_args()
    torch.set_num_threads(4)

    res = {"rot_spread_max": a.rot_spread_max, "audit": {}, "abs_rot_exclude": {},
           "ramp": {}, "oxford": {}, "verdict": {}}

    # ── (1) ──
    print(f"=== (1) orientation audit: GT bank vs teacher, {len(a.scenes)} scenes ===")
    print(f"{'scene':<14}{'runs':>5}{'ATE med':>9}{'raw rot':>9}{'const(L)':>10}{'spread(L)':>10}"
          f"{'spread(2)':>10}{'max(2)':>8}{'excl':>6}")
    for sc in a.scenes:
        rows = audit_scene(sc, a.rot_spread_max)
        res["audit"][sc] = rows
        ex = [r["rid"] for r in rows if r["exclude_abs_rot"]]
        if ex:
            res["abs_rot_exclude"][sc] = ex
        med = lambda k: st.median(r[k] for r in rows)
        print(f"{sc:<14}{len(rows):>5}{med('ate'):>9.4f}{med('rot_raw_med_deg'):>9.2f}"
              f"{med('rot_const_world_deg'):>10.2f}{med('rot_spread_left_deg'):>10.2f}"
              f"{med('rot_spread_two_deg'):>10.2f}{max(r['rot_spread_two_deg'] for r in rows):>8.2f}"
              f"{len(ex):>6}", flush=True)
    n_all = sum(len(v) for v in res["audit"].values())
    n_ex = sum(len(v) for v in res["abs_rot_exclude"].values())
    print(f"  -> {n_ex}/{n_all} runs excluded from L_abs-rot (two-sided spread > {a.rot_spread_max} deg)")
    # the bank's own residual must agree with the recomputation
    bad = [(sc, r["rid"]) for sc, rows in res["audit"].items() for r in rows
           if r["bank_gt_resid_ate"] is not None and abs(r["ate"] - r["bank_gt_resid_ate"]) > 1e-3]
    print(f"  ATE recomputation matches index.json on {n_all - len(bad)}/{n_all} runs")

    # ── (2a) synthetic ramp ──
    print(f"\n=== (2a) prefix vs hist on a synthetic ramp, {a.ramp_scene} GT runs {a.ramp_runs} ===")
    _, gt = bank_poses(os.path.join(ROOT, "labels", a.ramp_scene + "_gt"))
    for rid in a.ramp_runs:
        r, pg = gt[rid]
        ref = torch.from_numpy(pg).float()
        for ramp in a.ramps:
            stu = ramped(ref, ramp)
            cur = offset_curve(stu, ref, run_t0=r["t0"])
            res["ramp"][f"run{rid}_ramp{ramp}"] = cur
            for mode in ("prefix", "hist"):
                print(f"  run {rid} ramp {ramp}  {mode:<6} trans_resid " +
                      " ".join(f"{x['off']:>3}:{x['trans_resid']:.4f}" for x in cur[mode]) +
                      "   abs_pos " + " ".join(f"{x['abs_pos']:.2f}" for x in cur[mode]))

    # ── (2b) Oxford trajectories ──
    print(f"\n=== (2b) prefix vs hist on saved Oxford K=1 trajectories vs GT ===")
    scenes_ox = sorted(d for d in os.listdir(OXFORD_WS)
                       if os.path.isdir(os.path.join(OXFORD_WS, d)) and d != "eval")
    agg = {m: {"prefix": {}, "hist": {}} for m in OXFORD_METHODS}
    for m in OXFORD_METHODS:
        for sc in scenes_ox:
            p = os.path.join(OXFORD_WS, sc, m, "traj.txt")
            g = os.path.join(OXFORD_WS, sc, "gt", "traj.txt")
            if not (os.path.exists(p) and os.path.exists(g)):
                continue
            Rp, cp = load_traj(p)
            Rg, cg = load_traj(g)
            n = min(len(cp), len(cg))
            stu = pose_enc(cp[:n], mat_to_quat(Rp[:n]))
            ref = pose_enc(cg[:n], mat_to_quat(Rg[:n]))
            offs = [o for o in range(0, n - S + 1, S)]
            cur = offset_curve(stu, ref, run_t0=0, offsets=offs)
            res["oxford"][f"{m}/{sc}"] = cur
            for mode in ("prefix", "hist"):
                for x in cur[mode]:
                    agg[m][mode].setdefault(x["off"], []).append(x)
        for mode in ("prefix", "hist"):
            offs = sorted(agg[m][mode])
            print(f"  {m:<18} {mode:<6} trans_resid " +
                  " ".join(f"{o:>3}:{st.median(x['trans_resid'] for x in agg[m][mode][o]):.4f}" for o in offs))
            print(f"  {'':<18} {'':<6} abs_pos     " +
                  " ".join(f"{o:>3}:{st.median(x['abs_pos'] for x in agg[m][mode][o]):>6.2f}" for o in offs))
            print(f"  {'':<18} {'':<6} fallback    " +
                  " ".join(f"{o:>3}:{sum(x['mode'] == 'fallback' for x in agg[m][mode][o])}/{len(agg[m][mode][o])}" for o in offs))

    # ── verdict ──
    # scored on the drifted Oxford arms (the trained checkpoints), where the
    # residual is real drift rather than a synthetic ramp both fits see
    scores = {"prefix": [], "hist": []}
    detail = {}
    for m in OXFORD_METHODS:
        curves = {mode: [{"off": o, "trans_resid": st.median(x["trans_resid"] for x in agg[m][mode][o]),
                          "abs_pos": st.median(x["abs_pos"] for x in agg[m][mode][o])}
                         for o in sorted(agg[m][mode])] for mode in ("prefix", "hist")}
        v = judge(curves, "trans_resid")
        detail[m] = v
        for mode in scores:
            scores[mode].append(v[mode]["score"])
    for k, cur in res["ramp"].items():
        v = judge(cur, "trans_resid")
        detail[k] = v
        for mode in scores:
            scores[mode].append(v[mode]["score"])
    tot = {mode: float(sum(v)) for mode, v in scores.items()}
    # ★ A TIE GOES TO PREFIX.  The rule (monotone, deepest-largest, no shallow
    # blow-up) scores the two within a few percent of each other on every
    # source; what separates them is not in the rule but in what the residual
    # MEANS.  The prefix residual at a deep window is the drift accumulated
    # since the run start -- exactly the quantity H1 names -- and it is linear
    # in the ramp (rho 1.00 on all nine synthetic curves), whereas the hist fit
    # absorbs part of the drift into the gauge (ATE's behaviour) and reads
    # sub-linear.  Only a clear margin for hist overrides that.
    margin = (tot["hist"] - tot["prefix"]) / max(abs(tot["prefix"]), 1e-9)
    pick = "hist" if margin > 0.10 else "prefix"
    res["verdict"] = {"scores": tot, "detail": detail, "abs_fit_default": pick,
                      "hist_margin": margin, "rule": "hist only if it wins by > 10%"}
    print(f"\n=== verdict: prefix {tot['prefix']:.2f}  hist {tot['hist']:.2f}  "
          f"(hist margin {margin:+.1%}) -> --abs_fit {pick} ===")
    for k, v in detail.items():
        print(f"  {k:<22} " + "  ".join(
            f"{mode}: rho {v[mode].get('rho', float('nan')):+.2f} deepest {int(v[mode].get('deepest_is_max', 0))} "
            f"blowup {int(v[mode].get('shallow_blowup', 0))}" for mode in ("prefix", "hist")))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"\n[precheck] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
