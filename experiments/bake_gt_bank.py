#!/usr/bin/env python3
"""Bake a GT-pose label bank so the objective can be trained against truth.

The spine experiment left one question open: the runs degrade, but is that the
OBJECTIVE or the LABELS?  Every run so far trained against base-model pseudo
labels, so the two are confounded.  MCD carries GT, so the confound can be cut
directly -- supervise the SAME loss, the SAME policy, the SAME scenes, and
change only where ``pose_enc`` comes from.

What this writes, per bank run:

  pose_enc[:, :3]   GT camera centres
  pose_enc[:, 3:7]  GT camera orientation (XYZW, as ``quat_to_R`` reads it)
  pose_enc[:, 7:9]  the teacher's FoV, untouched
  depth, depth_conf the teacher's, byte-for-byte

★ WHY THE GT IS PUT INTO THE TEACHER'S GAUGE FIRST.  ``L_rot`` and ``L_dir`` are
Sim(3)-invariant, so raw world-frame GT would score identically.  ``L_motion``
is not: it compares ``log||dt||`` against ``log median(D)``, and its whole point
is that pose and depth share ONE unit fixed at the anchor.  Metric GT beside a
teacher depth in the teacher's arbitrary unit puts a constant
``log(sigma_gt / sigma_depth)`` into that term, i.e. it would teach the student a
wrong pose:depth ratio -- with lam_motion = 0.9 that is not a rounding error.
Fitting one Sim(3) per run against the teacher moves GT into the unit the depth
already lives in.  A similarity cannot absorb drift, only the gauge, so GT's
SHAPE survives intact; that shape is the entire point of the run.

The residual printed per run is GT-vs-teacher after that fit: it is the
teacher's own error, and experiments/teacher_vs_student_gt.py measured it at
0.03-0.08 m, so anything near that confirms the conventions line up.

    .venv-bench/bin/python experiments/bake_gt_bank.py --scene kth_day_10
    .venv-bench/bin/python experiments/bake_gt_bank.py --scene kth_day_10 --check
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from mcd_eval import (load_extrinsic, gt_camera_poses, umeyama,      # noqa: E402
                      rot_geodesic_deg)
from mcd_gt import quat_to_mat, mat_to_quat                          # noqa: E402
from lingbot_map.train.trainer import image_names                    # noqa: E402


def gt_rows_for(frames_dir: str, calib: str, sensor: str):
    """frame index on disk -> row in meta.npz, the mapping make_gt_score uses."""
    meta = np.load(os.path.join(frames_dir, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(calib, sensor)
    gp, gq = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    row_of = {str(n): i for i, n in enumerate(meta["names"])}
    disk = image_names(frames_dir)
    return np.array([row_of.get(n, -1) for n in disk], dtype=np.int64), gp, gq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--suffix", default="_gt")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--check", action="store_true",
                    help="report the fit per run and write nothing")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src = os.path.join(ROOT, "labels", a.scene)
    dst = os.path.join(ROOT, "labels", a.scene + a.suffix)
    frames = a.frames or os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    idx = json.load(open(os.path.join(src, "index.json")))

    gt_row, gp, gq = gt_rows_for(frames, os.path.join(ROOT, a.calib), a.sensor)
    Rgt = quat_to_mat(gq)
    print(f"{a.scene}: {len(idx['runs'])} runs, {len(gt_row)} frames on disk, "
          f"{len(gp)} GT rows", flush=True)

    if not a.check:
        if os.path.exists(dst) and not a.force:
            print(f"  {dst} exists -- pass --force to overwrite")
            return 1
        os.makedirs(dst, exist_ok=True)

    print(f"\n{'run':>4} {'t0':>7} {'L':>5} {'sigma':>10} {'resid ATE':>10} "
          f"{'resid rot':>10}")
    out_runs, sig = [], []
    t_start = time.time()
    for rid, r in enumerate(idx["runs"]):
        t0, L = r["t0"], r["L"]
        rows = gt_row[t0:t0 + L]
        if len(rows) < L or rows[0] < 0 or (np.diff(rows) != 1).any():
            raise RuntimeError(f"run {rid} window [{t0},{t0+L}) is not a "
                               f"contiguous GT range -- refusing to bake")
        z = np.load(os.path.join(src, r["file"]))
        pe = z["pose_enc"].astype(np.float64)
        g0 = int(rows[0])
        c_gt, R_g = gp[g0:g0 + L], Rgt[g0:g0 + L]

        # GT -> the teacher's gauge.  Scale is the only part the loss can feel;
        # R and t come along so every diagnostic stays comparable.
        s, R, t = umeyama(c_gt, pe[:, :3])
        c_new = (s * (R @ c_gt.T)).T + t
        q_new = mat_to_quat(np.einsum("ij,njk->nik", R, R_g))

        resid = float(np.sqrt(((c_new - pe[:, :3]) ** 2).sum(1).mean()))
        rot = float(np.median(rot_geodesic_deg(quat_to_mat(pe[:, 3:7].astype(np.float64)),
                                               np.einsum("ij,njk->nik", R, R_g))))
        sig.append(s)
        print(f"{rid:4d} {t0:7d} {L:5d} {s:10.4f} {resid:10.4f} {rot:10.4f}",
              flush=True)

        if not a.check:
            new = pe.copy()
            new[:, :3], new[:, 3:7] = c_new, q_new
            np.savez(os.path.join(dst, r["file"]),
                     pose_enc=new.astype(np.float32),
                     depth=z["depth"], depth_conf=z["depth_conf"])
        out_runs.append(dict(r, gt_sigma=s, gt_resid_ate=resid,
                             gt_resid_rot_deg=rot, gt_row0=g0))
        z.close()

    if a.check:
        # ★ THE NUMBER THAT DECIDES WHETHER THE RUN IS WORTH 11 HOURS.  If GT and
        # the teacher are nearly the same label in the loss's own units, the GT
        # arm just reproduces the teacher arm and proves nothing.  Scored on
        # S=48 windows, which is what training actually sees.
        import torch
        from lingbot_map.train.losses import rel_pose_loss, PRESETS
        P = PRESETS["A1PC"]
        lr, ld = [], []
        for rid, r in enumerate(idx["runs"]):
            z = np.load(os.path.join(src, r["file"]))
            pe = z["pose_enc"].astype(np.float64)
            g0 = out_runs[rid]["gt_row0"]
            s_, R_, t_ = umeyama(gp[g0:g0 + r["L"]], pe[:, :3])
            new = pe.copy()
            new[:, :3] = (s_ * (R_ @ gp[g0:g0 + r["L"]].T)).T + t_
            new[:, 3:7] = mat_to_quat(np.einsum("ij,njk->nik", R_, Rgt[g0:g0 + r["L"]]))
            A = torch.from_numpy(new).float()
            B = torch.from_numpy(pe).float()
            for w in range(0, r["L"] - 48 + 1, 48):
                Lr, Ld, _ = rel_pose_loss(A[w:w + 48], B[w:w + 48],
                                          mag_mode=P.get("mag_mode", "closed_form_scale"),
                                          pairs=P.get("pairs", "all"),
                                          min_gap=P.get("min_gap", 1),
                                          mag_trunc=P.get("mag_trunc", 1.0))
                lr.append(float(Lr)); ld.append(float(Ld))
            z.close()
        import math, statistics as stt
        print(f"\nGT label vs teacher label, {len(lr)} windows of 48:")
        print(f"  L_rot median {math.degrees(stt.median(lr)):8.4f} deg   "
              f"(x lam_rot {P['lam_rot']} -> {P['lam_rot'] * stt.median(lr):.5f})")
        print(f"  L_dir median {stt.median(ld):8.5f}       "
              f"(x lam_dir {P['lam_dir']} -> {P['lam_dir'] * stt.median(ld):.5f})")
        print(f"  weighted total {P['lam_rot'] * stt.median(lr) + P['lam_dir'] * stt.median(ld):.5f}")

    print(f"\nsigma over runs: median {np.median(sig):.4f}  "
          f"min {min(sig):.4f}  max {max(sig):.4f}  spread {max(sig)/min(sig):.2f}x")
    print("residual is GT-vs-teacher AFTER the per-run Sim(3): it is the "
          "teacher's own error, and should land near 0.03-0.08 m.")
    if not a.check:
        json.dump(dict(idx, runs=out_runs, gt_baked=dict(
            source=os.path.relpath(src, ROOT), calib=a.calib, sensor=a.sensor,
            sigma_median=float(np.median(sig)))),
            open(os.path.join(dst, "index.json"), "w"), indent=1)
        print(f"\n[bake] wrote {dst}  ({time.time() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
