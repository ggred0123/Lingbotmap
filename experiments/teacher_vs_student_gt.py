#!/usr/bin/env python3
"""Is the training label actually better than what it is correcting?

The method's premise: a student that has accumulated drift over a long rollout is
pulled back by a teacher that re-anchored recently and therefore has not drifted.
That premise has never been checked against ground truth -- the probe in the
trainer measures teacher IMITATION, and GT is metric-only and off by default.

MCD carries GT, so it can be checked directly.  For every bank run of one scene:

  teacher   the label the correction branch trains against: theta_0, fresh
            anchor, 72 frames of burn-in, K=1                (label_bank)
  student   theta_0 streamed from frame 0 at the deployed keyframe interval,
            i.e. the on-policy state the label is attached to

Both are scored on the SAME frames against the same GT, with the same
window-local Sim(3) that the trainer's own GT probe uses.  If the teacher is not
the more accurate of the two, the correction branch is pulling the student
toward something worse, and the premise fails.

    .venv-bench/bin/python experiments/teacher_vs_student_gt.py --scene kth_day_10 --K 1 28
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lingbot_map.train.label_bank import LabelBank                     # noqa: E402
from lingbot_map.train.trainer import make_gt_score                    # noqa: E402
from phase0_density_sweep import build_model                           # noqa: E402
from loss_floor_probe import load_images                               # noqa: E402


def stream_and_collect(model, images, upto, sf, K, wanted, dtype, dev):
    """One deployment-shaped pass from frame 0, keeping the wanted windows.

    ``wanted`` maps a window start to its length; the returned dict maps the same
    starts to [L, 9] pose_enc.  This is the ON-POLICY state: the cache at frame t
    is whatever this walk put there, not a fresh anchor near t.
    """
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                            num_frame_per_block=sf, causal_inference=True)
    acc = {t: [] for t in wanted}
    keep = {}
    for t, L in wanted.items():
        for f in range(t, t + L):
            keep.setdefault(f, []).append(t)
    for f in range(sf):
        if f in keep:
            for t in keep[f]:
                acc[t].append(out["pose_enc"][:, f:f + 1].float().cpu())
    del out
    for i in range(sf, upto):
        is_kf = (K <= 1) or ((i - sf) % K == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            o = model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                              num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        if i in keep:
            for t in keep[i]:
                acc[t].append(o["pose_enc"].float().cpu())
        del o
    return {t: torch.cat(v, dim=1)[0] for t, v in acc.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, nargs="*", default=[1, 28])
    ap.add_argument("--max-runs", type=int, default=8)
    ap.add_argument("--ckpt", default="/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/"
                                      "youngmin/ckpt/lingbot-map.pt")
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    bank = LabelBank(os.path.join(ROOT, "labels", a.scene), cache_runs=2)
    gt = make_gt_score(frames, os.path.join(ROOT, a.calib), a.sensor)
    dev, dtype = torch.device("cuda"), torch.bfloat16

    runs = [(rid, r) for rid, r in enumerate(bank.runs) if r["L"] >= a.S][:a.max_runs]
    wanted = {r["t0"]: a.S for _, r in runs}
    upto = max(wanted) + a.S
    print(f"scene {a.scene}: {len(runs)} windows, deepest starts at frame {max(wanted)}")

    imgs = load_images(frames, upto)
    model = build_model(a.ckpt, dev, 518, 14, 20000, 64, 8)
    model.eval()

    rec = {"scene": a.scene, "S": a.S, "windows": []}
    tea = {}
    for rid, r in runs:
        lab = bank.get(rid, 0, a.S, device=torch.device("cpu"))
        tea[r["t0"]] = gt(lab["pose_enc"], r["t0"], a.S)

    stu = {}
    for K in a.K:
        print(f"  streaming from frame 0 at K={K} up to {upto} ...", flush=True)
        stu[K] = stream_and_collect(model, imgs, upto, 8, K, wanted, dtype, dev)

    print(f"\n{'window':>8} {'depth':>7} | {'teacher ATE':>12} {'rot':>7} "
          + " ".join(f"| K={K}: {'ATE':>9} {'rot':>6}" for K in a.K))
    for rid, r in runs:
        t0 = r["t0"]
        tg = tea[t0]
        line = f"{t0:8d} {t0:7d} | {tg['gt_ate']:12.4f} {tg['gt_rot_deg']:7.3f}"
        row = {"t0": t0, "teacher": tg, "student": {}}
        for K in a.K:
            pe = stu[K].get(t0)
            if pe is None:
                line += f" |{'-':>19}"
                continue
            sg = gt(pe, t0, a.S)
            row["student"][str(K)] = sg
            line += f" | {sg['gt_ate']:12.4f} {sg['gt_rot_deg']:6.3f}"
        rec["windows"].append(row)
        print(line)

    print("\nteacher = the label the correction branch trains against (fresh anchor, K=1)")
    print("K=n     = theta_0 streamed from frame 0 at keyframe interval n (the on-policy state)")
    print("ATE     = window-local, Sim(3)-aligned, metres.  Lower is better.")
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1, default=float)
        print(f"[probe] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
