"""Is the window scale being quotiented away where it should be supervised?

THE ARCHITECTURE FIXES THE SCALE AT THE ANCHOR.  ``num_frame_for_scale`` is not
an estimator -- it is the mask rule that keeps the first 8 frames globally
visible (attention.py:246-256).  Every later frame attends to them, and the pose
head and the depth head both read the SAME aggregator tokens.  So a run has ONE
scale reference, shared by both modalities, and the only quantity a loss may
legitimately quotient out is the single ratio sigma_student / sigma_teacher.

The current loss quotients out 49:

    L_mag       closed_form_scale over the window      ->  1 scale removed
    L_depth-SI  s.median(dim=1), per frame             -> 48 scales removed

That grants a freedom the network does not have.  Two consequences are
measurable without any training, from the cached rollouts:

  DRIFT    per-frame depth scale r_i = log(median D_stu,i / median D_tea,i) must
           be CONSTANT across the window.  Its variation is pure error and the
           loss cannot see any of it.

  COUPLE   sigma_pose (what L_mag fits away) must EQUAL exp(mean r_i).  Their
           ratio is gauge-free -- sigma cancels -- so it is an error measure with
           no free parameters, and the loss cannot see it either.

  MOTION   the dense version of COUPLE: u_i = ||dt_i|| / median(D_i) is motion
           per unit scene depth, dimensionless and gauge-free per frame.  47
           samples per window instead of one scalar.

Scored the way T3 scored its estimators (docs/phase1-plan.md §3-T3): FAR/NEAR
discrimination, and stability against the window length.  A term that cannot
separate a contaminated window from a fresh one cannot supervise, whatever its
theoretical appeal.

CPU only -- it reuses the student rollouts loss_probe already cached.

Usage:
    python experiments/scale_probe.py --near labels/kth_near:80 --far labels/kth_far:5248
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.train.label_bank import LabelBank
from lingbot_map.train.losses import (
    quat_to_R, closed_form_scale, rel_pose_loss, depth_si_loss,
)


def _depth(x):
    return x[..., 0] if x.dim() == 4 else x


def window_terms(sp, tp, sd, td, n=None):
    """The three candidate terms plus the four current ones, on one window."""
    if n is not None:
        sp, tp, sd, td = sp[:n], tp[:n], sd[:n], td[:n]
    S = sp.shape[0]

    s = sd.reshape(S, -1).clamp(min=1e-3)
    t = td.reshape(S, -1).clamp(min=1e-3)
    r = s.median(1).values.log() - t.median(1).values.log()      # log(stu/tea) per frame

    Rs, Rt = quat_to_R(sp[:, 3:7]), quat_to_R(tp[:, 3:7])
    ds = torch.einsum("nij,nj->ni", Rs[:-1].transpose(1, 2), sp[1:, :3] - sp[:-1, :3])
    dt = torch.einsum("nij,nj->ni", Rt[:-1].transpose(1, 2), tp[1:, :3] - tp[:-1, :3])
    ns, nt = ds.norm(dim=-1).clamp(min=1e-8), dt.norm(dim=-1).clamp(min=1e-8)

    # sigma_pose: the scale L_mag fits and throws away
    sigma_pose = closed_form_scale(nt, ns)

    # motion per unit scene depth -- gauge-free per FRAME, so 47 samples not 1
    u_s = ns.log() - s.median(1).values[:-1].log()
    u_t = nt.log() - t.median(1).values[:-1].log()

    out = {
        "L_scale_drift": float((r - r.mean()).abs().mean()),
        "L_scale_couple": float((sigma_pose.log() - r.mean()).abs()),
        "L_motion_depth": float((u_s - u_t).abs().mean()),
        "sigma_pose": float(sigma_pose),
        "sigma_depth": float(r.mean().exp()),
        "r_std": float(r.std()),
    }
    if S >= 5:
        L_rot, L_dir, L_mag = rel_pose_loss(sp, tp)
        out.update({"L_rot": float(L_rot), "L_dir": float(L_dir),
                    "L_mag": float(L_mag)})
    out["L_depth_si"] = float(depth_si_loss(sd, td))
    return out


def load(spec, S):
    bank_path, _, t0 = spec.rpartition(":")
    t0 = int(t0)
    cache = f"experiments/results/_stu_t{t0}_S{S}_K28.pt"
    if not os.path.exists(cache):
        raise SystemExit(f"missing cached rollout {cache} -- run loss_probe.py first")
    stu = torch.load(cache, map_location="cpu")
    bank = LabelBank(bank_path)
    hit = next(((rid, t0 - r["t0"]) for rid, r in enumerate(bank.runs)
                if r["t0"] <= t0 and t0 + S <= r["t0"] + r["L"]), None)
    if hit is None:
        raise SystemExit(f"no single bank run covers [{t0}, {t0 + S})")
    lab = bank.get(*hit, S)
    return (stu["pose_enc"].float(), lab["pose_enc"].float(),
            _depth(stu["depth"]).float(), _depth(lab["depth"]).float(), t0)


NEW = ["L_scale_drift", "L_scale_couple", "L_motion_depth"]
CUR = ["L_rot", "L_dir", "L_mag", "L_depth_si"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", default="labels/kth_near:80")
    ap.add_argument("--far", default="labels/kth_far:5248")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--sweep", type=int, nargs="+",
                    default=[5, 8, 12, 16, 24, 32, 40, 48])
    ap.add_argument("--out", default="experiments/results/scale_probe.json")
    args = ap.parse_args()

    near = load(args.near, args.S)
    far = load(args.far, args.S)
    a, b = window_terms(*near[:4]), window_terms(*far[:4])

    print(f"\n{'=' * 78}\n  NEAR t0={near[4]}   vs   FAR t0={far[4]}   S={args.S}\n{'=' * 78}")
    print(f"  {'term':<18} {'NEAR':>10} {'FAR':>10} {'FAR/NEAR':>10}   status")
    rows = []
    for k in NEW + CUR:
        ratio = b[k] / max(a[k], 1e-12)
        tag = "NEW -- not in the loss" if k in NEW else "current"
        print(f"  {k:<18} {a[k]:>10.4f} {b[k]:>10.4f} {ratio:>10.2f}x   {tag}")
        rows.append({"term": k, "near": a[k], "far": b[k], "ratio": ratio,
                     "in_loss": k in CUR})
    print(f"\n  sigma_pose   NEAR {a['sigma_pose']:.4f}  FAR {b['sigma_pose']:.4f}")
    print(f"  sigma_depth  NEAR {a['sigma_depth']:.4f}  FAR {b['sigma_depth']:.4f}")
    print(f"  -> pose/depth mismatch  NEAR {a['sigma_pose'] / a['sigma_depth']:.2f}x   "
          f"FAR {b['sigma_pose'] / b['sigma_depth']:.2f}x   "
          f"(1.00x if the student were internally consistent)")

    # stability against window length -- the failure that killed the L_mag draft
    print(f"\n  window-length stability (NEAR), relative to the S={args.S} value:")
    print(f"      {'n_sup':>6}" + "".join(f"{k:>17}" for k in NEW))
    sweep = []
    for n in args.sweep:
        if n > args.S:
            continue
        w = window_terms(*near[:4], n=n)
        sweep.append({"n_sup": n, **{k: w[k] for k in NEW}})
        print(f"      {n:>6}" + "".join(f"{w[k]:>17.4f}" for k in NEW))
    for k in NEW:
        vals = [row[k] for row in sweep if row[k] == row[k]]
        swing = max(vals) / max(min(vals), 1e-12)
        print(f"      {k:<18} swing over n_sup = {swing:.2f}x")

    res = {"meta": vars(args), "near": {"t0": near[4], **a},
           "far": {"t0": far[4], **b}, "rows": rows, "sweep": sweep}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
