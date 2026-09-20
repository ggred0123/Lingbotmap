#!/usr/bin/env python
"""RPE measured at the keyframe interval instead of at Delta = 1.

WHY THIS EXISTS.  docs/eval-metric-audit.html section 01(c): when the deployed
keyframe interval K is greater than one, non-keyframes leave no KV in the cache
(gct_stream.py:427-431).  Neighbouring non-keyframes are therefore independent
one-shot estimates against the same frozen cache, and rpe(delta=1) measures the
amplitude of that noise rather than tracking.  The audit re-measured VBR at
delta=K and the reported v7f gain of -0.468 collapsed to -0.045.

Section 09 asks for the same re-aggregation on every K>1 cell, using the stored
traj.txt so no inference is needed.  This script is that re-aggregation, and it
also emits the keyframe-level diagnostics the audit's VBR table carries:
direction agreement, reverse fraction and displacement ratio between consecutive
KEYFRAMES after the same Sim(3) alignment ATE uses.

    .venv-bench/bin/python experiments/rpe_at_k.py \
        --ws  /path/bench_ws/oxford_long --dataset oxford_long \
        --out experiments/results/rpe_at_k_oxford_long.json

Delta = 1 is recomputed as well and must reproduce the benchmark's own
eval/traj.json to three decimals; --check reports the residual per cell.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe


# ── trajectory io ───────────────────────────────────────────────────────────
def read_bss_traj(path: Path):
    """(frame_idx, (N,4,4) c2w) from a BSS Trajectory Format v2 file."""
    idx, mats = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            idx.append(int(float(parts[0])))
            m = np.eye(4)
            m[:3, :4] = np.asarray(parts[1:13], dtype=float).reshape(3, 4)
            mats.append(m)
    if not mats:
        raise ValueError(f"no poses in {path}")
    return np.asarray(idx, dtype=int), np.stack(mats)


def orthogonalize(pose: np.ndarray) -> np.ndarray:
    out = pose.copy()
    R = out[:3, :3]
    U, _, Vh = np.linalg.svd(R)
    Ro = U @ Vh
    if np.linalg.det(Ro) < 0:
        U[:, -1] *= -1
        Ro = U @ Vh
    out[:3, :3] = Ro
    return out


def to_evo(poses: np.ndarray, stamps: np.ndarray) -> PoseTrajectory3D:
    return PoseTrajectory3D(poses_se3=[orthogonalize(p) for p in poses],
                            timestamps=stamps)


# ── keyframe interval ───────────────────────────────────────────────────────
def method_k(method_dir: Path, n_frames: int, cfg_root: Path) -> int:
    """Deployed keyframe interval for this method, from its yaml."""
    cfg = cfg_root / f"{method_dir.name}.yaml"
    ki, thr = "auto", 320
    if cfg.exists():
        with open(cfg) as fh:
            y = yaml.safe_load(fh) or {}
        ki = y.get("_keyframe_interval", "auto")
        thr = int(y.get("_auto_keyframe_threshold", 320))
    if isinstance(ki, int):
        return max(1, ki)
    if isinstance(ki, str) and ki.isdigit():
        return max(1, int(ki))
    return 1 if n_frames <= thr else math.ceil(n_frames / thr)


# ── metrics ─────────────────────────────────────────────────────────────────
def rpe_at(traj_ref, traj_est, delta: int, relation) -> float:
    r = main_rpe.rpe(traj_ref, traj_est, est_name="traj",
                     pose_relation=relation, align=True, correct_scale=True,
                     delta=delta, delta_unit=Unit.frames, rel_delta_tol=0.01,
                     all_pairs=True)
    return float(r.stats["rmse"])


def keyframe_geometry(traj_ref, traj_est, K: int) -> dict:
    """Direction / reverse rate / displacement ratio between consecutive keyframes.

    The estimate is put in the GT frame with the SAME Sim(3) alignment ATE uses,
    so a ratio far from 1.0 is a scale error the global fit could not absorb --
    it is the drift of step size ALONG the stream, not a constant gauge offset.
    """
    est = copy.deepcopy(traj_est)
    est.align(traj_ref, correct_scale=True)
    pe = np.asarray(est.positions_xyz)
    pg = np.asarray(traj_ref.positions_xyz)
    kf = np.arange(0, len(pg), max(1, K))
    if len(kf) < 3:
        return {}
    de, dg = np.diff(pe[kf], axis=0), np.diff(pg[kf], axis=0)
    ne, ng = np.linalg.norm(de, axis=1), np.linalg.norm(dg, axis=1)
    ok = (ne > 1e-9) & (ng > 1e-9)
    if ok.sum() < 3:
        return {}
    cos = np.sum(de[ok] * dg[ok], axis=1) / (ne[ok] * ng[ok])
    return {
        "kf_steps": int(ok.sum()),
        "kf_cos_median": float(np.median(cos)),
        "kf_reverse_frac": float((cos < 0).mean()),
        "kf_ratio_median": float(np.median(ne[ok] / ng[ok])),
    }


def evaluate_pair(gt_dir: Path, m_dir: Path, cfg_root: Path) -> dict:
    gi, gp = read_bss_traj(gt_dir / "traj.txt")
    ei, ep = read_bss_traj(m_dir / "traj.txt")
    # Predictions are dense here; index-match defensively all the same.
    common = np.intersect1d(gi, ei)
    if len(common) < 8:
        raise ValueError("fewer than 8 shared frames")
    gsel = {v: k for k, v in enumerate(gi)}
    esel = {v: k for k, v in enumerate(ei)}
    gp = gp[[gsel[c] for c in common]]
    ep = ep[[esel[c] for c in common]]
    good = (np.isfinite(gp.reshape(len(gp), -1)).all(1)
            & np.isfinite(ep.reshape(len(ep), -1)).all(1))
    gp, ep, stamps = gp[good], ep[good], common[good].astype(float)

    ref, est = to_evo(gp, stamps), to_evo(ep, stamps)
    K = method_k(m_dir, len(stamps), cfg_root)

    aligned = copy.deepcopy(est)
    aligned.align(ref, correct_scale=True)
    ate = float(main_ape.ape(ref, aligned, est_name="traj",
                             pose_relation=PoseRelation.translation_part,
                             align=False, correct_scale=False).stats["rmse"])

    out = {"frames": int(len(stamps)), "K": int(K), "ate": ate,
           "rpe_trans_d1": rpe_at(ref, est, 1, PoseRelation.translation_part),
           "rpe_rot_d1": rpe_at(ref, est, 1, PoseRelation.rotation_angle_deg)}
    if K > 1:
        out["rpe_trans_dK"] = rpe_at(ref, est, K, PoseRelation.translation_part)
        out["rpe_rot_dK"] = rpe_at(ref, est, K, PoseRelation.rotation_angle_deg)
    else:                       # K == 1: the two are the same measurement
        out["rpe_trans_dK"] = out["rpe_trans_d1"]
        out["rpe_rot_dK"] = out["rpe_rot_d1"]
    out.update(keyframe_geometry(ref, est, K))
    return out


# ── driver ──────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True, help="bench workspace root")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="default: every method directory present")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--cfg-root", default=None,
                    help="benchmark/configs/methods (default: relative to this file)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true",
                    help="compare delta=1 against the workspace eval/traj.json")
    a = ap.parse_args()

    root = Path(a.ws) / a.dataset
    cfg_root = Path(a.cfg_root) if a.cfg_root else \
        Path(__file__).resolve().parents[1] / "benchmark" / "configs" / "methods"
    scenes = sorted(d.name for d in root.iterdir()
                    if d.is_dir() and d.name != "eval" and (d / "gt").is_dir())
    if a.scenes:
        scenes = [s for s in scenes if s in a.scenes]

    res: dict[str, dict] = {}
    for s in scenes:
        sd = root / s
        methods = a.methods or sorted(
            d.name for d in sd.iterdir()
            if d.is_dir() and d.name not in ("gt", "eval") and (d / "traj.txt").exists())
        for m in methods:
            md = sd / m
            if not (md / "traj.txt").exists():
                continue
            try:
                r = evaluate_pair(sd / "gt", md, cfg_root)
            except Exception as exc:                      # noqa: BLE001
                print(f"  !! {s}/{m}: {exc}", file=sys.stderr)
                continue
            res.setdefault(m, {})[s] = r
            print(f"  {s:34s} {m:26s} K={r['K']:>3} "
                  f"ate {r['ate']:8.3f}  d1 {r['rpe_trans_d1']:7.3f}  "
                  f"dK {r['rpe_trans_dK']:7.3f}"
                  + (f"  cos {r['kf_cos_median']:+.3f} rev {r['kf_reverse_frac']:.0%} "
                     f"|e|/|g| {r['kf_ratio_median']:.2f}" if "kf_cos_median" in r else ""),
                  flush=True)

            if a.check:
                ev = sd / "eval" / "traj.json"
                if ev.exists():
                    ref = json.load(open(ev)).get(m)
                    if ref:
                        d_ate = abs(ref["ate"] - r["ate"])
                        d_rpe = abs(ref["rpe_trans"] - r["rpe_trans_d1"])
                        flag = "OK " if (d_ate < 1e-3 and d_rpe < 1e-3) else "DIFF"
                        print(f"      [{flag}] bench ate {ref['ate']:.4f} rpe {ref['rpe_trans']:.4f}"
                              f"  |d| {d_ate:.2e} / {d_rpe:.2e}", flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"[rpe_at_k] wrote {a.out}  ({sum(len(v) for v in res.values())} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
