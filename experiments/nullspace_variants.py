#!/usr/bin/env python3
"""Which change to the objective, if any, lets it SEE the 21 m ramp?

nullspace_demo.py established that A1PC cannot distinguish GT from a slowly
ramping rescale of it: the weighted loss moves 5.7e-09 while ATE moves 21 m.
Its own header says the point is to test loss designs here, statically, instead
of spending a GT training run to confirm what the counterexample already shows.
gtsup spent that run anyway and confirmed it (AUC_03 -45%).  So: before the next
training cell, put the candidate fixes through the same counterexample.

Nothing here edits lingbot_map/ -- the other node's teasup run imports the same
files off the shared filesystem and its watchdog may restart it at any moment.
Each variant is computed from rel_pose_loss's own outputs, or reimplements only
the one term it changes.

    .venv-bench/bin/python experiments/nullspace_variants.py --scene kth_day_10
"""
from __future__ import annotations
import argparse, json, math, os, statistics as st, sys
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
from lingbot_map.train.losses import PRESETS, rel_pose_loss          # noqa: E402
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama        # noqa: E402


def pose_enc_from(cent, quat):
    pe = np.zeros((len(cent), 9), dtype=np.float64)
    pe[:, :3] = cent; pe[:, 3:7] = quat; pe[:, 7:] = 1.0
    return torch.from_numpy(pe).float()


def ate_after_global_sim3(pred, gtp):
    s, R, t = umeyama(pred, gtp)
    al = (s * (R @ pred.T)).T + t
    return float(np.sqrt(((al - gtp) ** 2).sum(1).mean()))


def ramped(gp, ramp):
    """The counterexample: per-step scale ramps log-linearly, mean unchanged."""
    steps = np.diff(gp, axis=0)
    u = np.linspace(-0.5, 0.5, len(steps))[:, None]
    g = np.exp(u * math.log(ramp))
    return np.concatenate([gp[:1], gp[:1] + np.cumsum(steps * g, axis=0)])


def window_terms(pe_s, pe_t, N, S, P, stride=None):
    """rel_pose_loss over disjoint (or strided) windows -> median of each term."""
    stride = stride or S
    lr, ld, lm = [], [], []
    for t in range(0, max(N - S, 1), stride):
        Lr, Ld, Lm = rel_pose_loss(pe_s[t:t + S], pe_t[t:t + S],
                                   mag_mode=P.get("mag_mode", "closed_form_scale"),
                                   pairs=P.get("pairs", "all"),
                                   min_gap=P.get("min_gap", 1),
                                   mag_trunc=P.get("mag_trunc", 1.0))
        lr.append(float(Lr)); ld.append(float(Ld)); lm.append(float(Lm))
    return st.median(lr), st.median(ld), st.median(lm)


def anchor_rel_scale(cent, gtp, S, delta):
    """The cross-window quantity A1PC never forms.

    For anchors a and a+delta, compare the ratio of the LONG displacement to a
    LOCAL step, student vs teacher:  r = log(||d_long|| / v_local).  Inside one
    window the ramp is a similarity and cancels; across delta frames it does not,
    which is precisely what a single global Sim(3) cannot absorb either -- the
    definition of ATE.  This is the shape --lam_long supervises.
    """
    res = []
    for a in range(0, len(cent) - delta - S, S):
        d_s = np.linalg.norm(cent[a + delta] - cent[a])
        d_t = np.linalg.norm(gtp[a + delta] - gtp[a])
        v_s = np.linalg.norm(np.diff(cent[a:a + S], axis=0), axis=1).mean()
        v_t = np.linalg.norm(np.diff(gtp[a:a + S], axis=0), axis=1).mean()
        if min(d_s, d_t, v_s, v_t) <= 1e-9:
            continue
        res.append(abs(math.log((d_s / v_s) / (d_t / v_t))))
    return st.median(res) if res else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--ramp", type=float, nargs="*", default=[1.0, 3.0])
    ap.add_argument("--huber_deg", type=float, default=1.0)
    ap.add_argument("--out", default="experiments/results/nullspace_variants.json")
    a = ap.parse_args()

    frames = os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    meta = np.load(os.path.join(frames, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(os.path.join(ROOT, a.calib), a.sensor)
    gp, gq = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    N = len(gp)
    P = PRESETS["A1PC"]
    lam_rot, lam_dir, lam_mag = P["lam_rot"], P["lam_dir"], P["lam_mag"]
    print(f"{a.scene}: {N} GT poses   A1PC lam_rot={lam_rot} lam_dir={lam_dir} lam_mag={lam_mag}")

    rows = {}
    for ramp in a.ramp:
        cent = ramped(gp, ramp)
        pe_s, pe_t = pose_enc_from(cent, gq), pose_enc_from(gp, gq)
        r = {"ate": ate_after_global_sim3(cent, gp)}
        # baseline: A1PC on 48-frame windows, exactly as the trainer supervises
        lr, ld, lm = window_terms(pe_s, pe_t, N, a.S, P)
        r["A1PC"] = lam_rot * lr + lam_dir * ld
        r["L_rot_deg"], r["L_dir"], r["L_mag"] = math.degrees(lr), ld, lm
        # (1) angle-Huber on the rotation term
        d = math.radians(a.huber_deg)
        r["huber_rot"] = lam_rot * (0.5 * lr * lr / d if lr < d else lr - 0.5 * d) + lam_dir * ld
        # (2) turn the magnitude term on (A1PC ships it at lam_mag = 0)
        r["A1PC_lam_mag15"] = r["A1PC"] + 15.0 * lm
        # (3) widen the supervision window
        for S2 in (240, 960):
            if N - S2 > S2:
                l2, d2, m2 = window_terms(pe_s, pe_t, N, S2, P)
                r[f"win{S2}"] = lam_rot * l2 + lam_dir * d2
                r[f"win{S2}_L_mag"] = m2
        # (4) the cross-window anchor-relative scale residual (what lam_long forms)
        for delta in (240, 960, 1920):
            if N - delta - a.S > 0:
                r[f"long_r_d{delta}"] = anchor_rel_scale(cent, gp, a.S, delta)
        rows[ramp] = r

    lo, hi = min(a.ramp), max(a.ramp)
    print(f"\n=== can the objective SEE the ramp?  ramp {lo} (GT) vs {hi} ===")
    print(f"  ATE: {rows[lo]['ate']:.3f} m -> {rows[hi]['ate']:.3f} m\n")
    print(f"  {'variant':>22} {'ramp '+str(lo):>14} {'ramp '+str(hi):>14} {'rel. change':>14}  verdict")
    keys = [k for k in rows[lo] if k not in ("ate",)]
    for k in keys:
        v0, v1 = rows[lo][k], rows[hi][k]
        if not (isinstance(v0, float) and math.isfinite(v0)) or abs(v0) < 1e-12:
            rel = float("nan")
        else:
            rel = (v1 - v0) / abs(v0)
        mark = "BLIND" if abs(rel) < 1e-3 else ("weak" if abs(rel) < 0.05 else "SEES IT")
        print(f"  {k:>22} {v0:>14.9f} {v1:>14.9f} {rel:>+13.2%}  {mark}")
    json.dump({"scene": a.scene, "S": a.S, "rows": {str(k): v for k, v in rows.items()}},
              open(a.out, "w"), indent=1, default=float)
    print(f"\n[variants] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
