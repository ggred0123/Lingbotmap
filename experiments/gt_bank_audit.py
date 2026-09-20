#!/usr/bin/env python3
"""How far apart are the GT bank and the teacher bank, in the loss's own units?

The GT arm is only worth its GPU hours if swapping the labels actually changes
what the objective asks for.  Two numbers settle that, both scored on the S=48
windows training really sees:

  L_rot / L_dir     GT label vs teacher label, weighted by A1PC's own lambdas
  vs the loss floor experiments/loss_floor.py measured at theta_0 (~0.022)

If the weighted gap is comparable to that floor the arms will barely differ and
the run proves nothing; if it dwarfs it, the GT arm is a genuinely different
target.  The per-run sigma (baked into each _gt index.json) is reported beside
it, because its spread is the gauge problem section 08 found, measured here on
the very banks the training reads.

    .venv-bench/bin/python experiments/gt_bank_audit.py
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

# * pairs="all" on S=48 is 1,128 tiny tensor ops per window over ~1,200
# windows.  torch's default pool spawns a thread per core for each of them
# and spends the run in contention -- measured 18 minutes of CPU in 69
# seconds of wall.  Four threads finishes; all of them does not.
torch.set_num_threads(4)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lingbot_map.train.losses import PRESETS, rel_pose_loss              # noqa: E402

SCENES = ["kth_day_10", "kth_night_01", "kth_night_04", "kth_night_05",
          "tuhh_day_02", "tuhh_day_03", "tuhh_day_04",
          "tuhh_night_07", "tuhh_night_08", "tuhh_night_09"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--suffix", default="_gt")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--preset", default="A1PC")
    ap.add_argument("--out", default="experiments/results/gt_bank.json")
    a = ap.parse_args()

    P = PRESETS[a.preset]
    lam_r, lam_d = P["lam_rot"], P["lam_dir"]
    rec = {"preset": a.preset, "S": a.S, "lam_rot": lam_r, "lam_dir": lam_d,
           "scenes": []}

    print(f"{'scene':<16}{'runs':>5}{'sigma med':>11}{'spread':>8}"
          f"{'resid m':>9}{'L_rot deg':>11}{'L_dir':>9}{'weighted':>10}")
    for s in a.scenes:
        gd = os.path.join(ROOT, "labels", s + a.suffix)
        td = os.path.join(ROOT, "labels", s)
        if not os.path.exists(os.path.join(gd, "index.json")):
            print(f"{s:<16}{'(not baked)':>50}")
            continue
        idx = json.load(open(os.path.join(gd, "index.json")))
        lr, ld, sig, res = [], [], [], []
        for r in idx["runs"]:
            sig.append(r["gt_sigma"]); res.append(r["gt_resid_ate"])
            A = torch.from_numpy(np.load(os.path.join(gd, r["file"]))["pose_enc"]).float()
            B = torch.from_numpy(np.load(os.path.join(td, r["file"]))["pose_enc"]).float()
            for w in range(0, r["L"] - a.S + 1, a.S):
                Lr, Ld, _ = rel_pose_loss(A[w:w + a.S], B[w:w + a.S],
                                          mag_mode=P.get("mag_mode", "closed_form_scale"),
                                          pairs=P.get("pairs", "all"),
                                          min_gap=P.get("min_gap", 1),
                                          mag_trunc=P.get("mag_trunc", 1.0))
                lr.append(float(Lr)); ld.append(float(Ld))
        w8 = lam_r * st.median(lr) + lam_d * st.median(ld)
        row = {"scene": s, "runs": len(idx["runs"]), "windows": len(lr),
               "sigma_median": st.median(sig), "sigma_spread": max(sig) / min(sig),
               "resid_ate_median": st.median(res),
               "L_rot_deg": math.degrees(st.median(lr)), "L_dir": st.median(ld),
               "weighted": w8}
        rec["scenes"].append(row)
        print(f"{s:<16}{row['runs']:5d}{row['sigma_median']:11.4f}"
              f"{row['sigma_spread']:7.1f}x{row['resid_ate_median']:9.4f}"
              f"{row['L_rot_deg']:11.4f}{row['L_dir']:9.5f}{w8:10.4f}", flush=True)

    if rec["scenes"]:
        rec["weighted_median"] = st.median([r["weighted"] for r in rec["scenes"]])
        rec["resid_median"] = st.median([r["resid_ate_median"] for r in rec["scenes"]])
        fl = None
        try:
            f = json.load(open(os.path.join(ROOT, "experiments/results/loss_floor.json")))
            fl = st.median([r["L_masked"] for r in f["rows"]])
        except Exception:
            pass
        rec["loss_floor_masked"] = fl
        print(f"\nweighted GT-vs-teacher gap, median over scenes: "
              f"{rec['weighted_median']:.4f}")
        if fl:
            print(f"training loss against the teacher at theta_0: {fl:.5f}  "
                  f"-> the GT target is {rec['weighted_median'] / fl:.1f}x further away")
        print("sigma is the per-run Sim(3) scale between GT and the teacher: its "
              "spread IS the gauge problem, measured on the training banks.")
        json.dump(rec, open(os.path.join(ROOT, a.out), "w"), indent=1)
        print(f"[audit] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
