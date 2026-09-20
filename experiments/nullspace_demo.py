#!/usr/bin/env python3
"""How much ATE fits inside the objective's null space?

Before spending a training run on "supervise with GT and see if the loss design
is fine", construct the counterexample directly.  If a trajectory exists that the
A1PC loss cannot tell apart from GT yet whose ATE is large, then GT labels cannot
fix the objective, and the training run would only confirm it expensively.

The construction is the one ``long_loss.py`` names in its header: a deformation
that is locally a single Sim(3) but changes slowly from window to window.  Take
the GT camera centres, scale each consecutive step by a slowly ramping g(t), and
re-integrate.  Inside any 48-frame window the scale is nearly constant, so the
window is nearly a similarity transform of GT and every term of the loss -- which
is built to be invariant to exactly that -- barely moves.  Over 6,000 frames the
ramp accumulates and no single global Sim(3) can absorb it, which is what ATE
measures.

    .venv-bench/bin/python experiments/nullspace_demo.py --scene kth_day_10
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

from lingbot_map.train.losses import PRESETS, rel_pose_loss           # noqa: E402
from mcd_eval import load_extrinsic, gt_camera_poses, umeyama         # noqa: E402
from lingbot_map.train.trainer import image_names                     # noqa: E402


def pose_enc_from(cent: np.ndarray, quat: np.ndarray) -> torch.Tensor:
    pe = np.zeros((len(cent), 9), dtype=np.float64)
    pe[:, :3] = cent
    pe[:, 3:7] = quat
    pe[:, 7:] = 1.0
    return torch.from_numpy(pe).float()


def ate_after_global_sim3(pred: np.ndarray, gtp: np.ndarray) -> float:
    s, R, t = umeyama(pred, gtp)
    al = (s * (R @ pred.T)).T + t
    return float(np.sqrt(((al - gtp) ** 2).sum(1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--ramp", type=float, nargs="*", default=[1.0, 1.5, 2.0, 3.0],
                    help="total scale factor from start to end of the track")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    meta = np.load(os.path.join(frames, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(os.path.join(ROOT, a.calib), a.sensor)
    gp, gq = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    N = len(gp)
    print(f"{a.scene}: {N} GT poses")

    P = PRESETS["A1PC"]
    lam_rot, lam_dir = P["lam_rot"], P["lam_dir"]

    steps = np.diff(gp, axis=0)                       # [N-1, 3]
    rec = {"scene": a.scene, "S": a.S, "n": N, "rows": []}
    print(f"\n{'total ramp':>11} {'per-frame g':>13} | {'L_rot(deg)':>11} {'L_dir':>9} "
          f"{'weighted L':>11} | {'ATE (m)':>9}")
    for ramp in a.ramp:
        # g(t) ramps log-linearly from 1/sqrt(ramp) to sqrt(ramp) so the mean
        # scale is unchanged and only the RAMP is introduced.
        u = np.linspace(-0.5, 0.5, len(steps))[:, None]
        g = np.exp(u * math.log(ramp))
        cent = np.concatenate([gp[:1], gp[:1] + np.cumsum(steps * g, axis=0)])

        pe_s = pose_enc_from(cent, gq)
        pe_t = pose_enc_from(gp, gq)
        lr, ld, lm = [], [], []
        for t in range(0, N - a.S, a.S):
            Lr, Ld, Lm = rel_pose_loss(pe_s[t:t + a.S], pe_t[t:t + a.S],
                                       mag_mode=P.get("mag_mode", "closed_form_scale"),
                                       pairs=P.get("pairs", "all"),
                                       min_gap=P.get("min_gap", 1),
                                       mag_trunc=P.get("mag_trunc", 1.0))
            lr.append(float(Lr)); ld.append(float(Ld)); lm.append(float(Lm))
        w = lam_rot * st.median(lr) + lam_dir * st.median(ld)
        ate = ate_after_global_sim3(cent, gp)
        per = ramp ** (1.0 / max(len(steps), 1))
        rec["rows"].append({"ramp": ramp, "per_frame_g": per,
                            "L_rot_deg": math.degrees(st.median(lr)),
                            "L_dir": st.median(ld), "L_mag": st.median(lm),
                            "weighted": w, "ate": ate})
        print(f"{ramp:11.2f} {per:13.8f} | {math.degrees(st.median(lr)):11.6f} "
              f"{st.median(ld):9.2e} {w:11.6f} | {ate:9.4f}")

    print("\nL_rot / L_dir are the pose half of A1PC; lam_mag = 0 so L_mag is shown "
          "but unused.\nramp 1.00 is the unperturbed GT and is the reference row.")
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1, default=float)
        print(f"[demo] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
