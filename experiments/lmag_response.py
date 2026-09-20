#!/usr/bin/env python3
"""Would turning L_mag on close the null space?

experiments/nullspace_demo.py showed a scale ramp worth 21 m of ATE that the
pose half of A1PC cannot see at all.  ``lam_mag = 0`` in every run, so the
obvious question is whether switching it on would catch it.

``_magnitude_loss`` normalises each window by a scale fitted inside that window,
in all four modes, so it can only ever respond to how the scale VARIES ACROSS a
window -- never to how much scale has accumulated since the anchor.  Two
perturbations separate those:

  ramp        g(t) ramps log-linearly over the whole track.  Inside a 48-frame
              window g still changes a little, so L_mag sees something small.
  stepwise    g is exactly constant within each window and jumps between
              windows, accumulating the same way.  L_mag should see nothing at
              all, because a constant per-window scale is what it fits away.

Both are scored with every mag_mode the codebase offers.

    .venv-bench/bin/python experiments/lmag_response.py
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

from lingbot_map.train.losses import rel_pose_loss                   # noqa: E402
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama        # noqa: E402

MODES = ("median", "closed_form_scale", "l1", "trunc_l1")


def pose_enc(c, q):
    a = np.zeros((len(c), 9))
    a[:, :3] = c
    a[:, 3:7] = q
    a[:, 7:] = 1.0
    return torch.from_numpy(a).float()


def ate(pred, gtp):
    s, R, t = umeyama(pred, gtp)
    al = (s * (R @ pred.T)).T + t
    return float(np.sqrt(((al - gtp) ** 2).sum(1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--every", type=int, default=4, help="score every Nth window")
    ap.add_argument("--ramp", type=float, nargs="*", default=[1.0, 1.5, 3.0])
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    meta = np.load(os.path.join(frames, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(os.path.join(ROOT, a.calib), a.sensor)
    gp, gq = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    steps = np.diff(gp, axis=0)
    S, n = a.S, len(steps)
    tea = pose_enc(gp, gq)
    starts = list(range(0, len(gp) - S, S * a.every))
    print(f"{a.scene}: {len(gp)} poses, scoring {len(starts)} windows of {S}",
          flush=True)

    def g_ramp(ramp):
        u = np.linspace(-0.5, 0.5, n)[:, None]
        return np.exp(u * math.log(ramp))

    def g_step(ramp):
        """Constant inside each window, jumping at window boundaries."""
        w = np.arange(n) // S
        u = (w / max(w.max(), 1) - 0.5)[:, None]
        return np.exp(u * math.log(ramp))

    rec = {"scene": a.scene, "S": S, "rows": []}
    hdr = (f"{'perturb':>9} {'ramp':>6} {'win g var':>11} | "
           + " ".join(f"{m:>17}" for m in MODES) + f" | {'ATE (m)':>8}")
    print("\n" + hdr, flush=True)
    for name, gf in (("ramp", g_ramp), ("stepwise", g_step)):
        for ramp in a.ramp:
            g = gf(ramp)
            cent = np.concatenate([gp[:1], gp[:1] + np.cumsum(steps * g, axis=0)])
            stu = pose_enc(cent, gq)
            row = {"perturb": name, "ramp": ramp, "modes": {}}
            cells = []
            for mode in MODES:
                v = []
                for t in starts:
                    _, _, Lm = rel_pose_loss(stu[t:t + S], tea[t:t + S],
                                             mag_mode=mode, pairs="all",
                                             min_gap=1, mag_trunc=1.0)
                    v.append(float(Lm))
                m = st.median(v)
                row["modes"][mode] = m
                cells.append(f"{m:17.3e}")
            # how much g moves across one window, for context
            gv = float(g[S - 1, 0] / g[0, 0]) if name == "ramp" else 1.0
            row["within_window_g"] = gv
            row["ate"] = ate(cent, gp)
            rec["rows"].append(row)
            print(f"{name:>9} {ramp:6.2f} {(gv - 1) * 100:10.4f}% | "
                  + " ".join(cells) + f" | {row['ate']:8.2f}", flush=True)

    print("\nL_mag is a median over windows.  ramp 1.00 is unperturbed GT: "
          "that row is the floor.", flush=True)
    base = {r["perturb"]: r for r in rec["rows"] if r["ramp"] == 1.0}
    print(f"\n{'perturb':>9} {'ramp':>6} | " + " ".join(f"{m:>17}" for m in MODES)
          + "   (ramp 1.0 대비 배수)", flush=True)
    for r in rec["rows"]:
        if r["ramp"] == 1.0:
            continue
        b = base[r["perturb"]]["modes"]
        print(f"{r['perturb']:>9} {r['ramp']:6.2f} | "
              + " ".join(f"{r['modes'][m] / max(b[m], 1e-30):17.2f}" for m in MODES),
              flush=True)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1, default=float)
        print(f"[lmag] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
