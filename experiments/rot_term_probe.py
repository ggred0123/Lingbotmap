#!/usr/bin/env python3
"""Does L_rot's acos actually matter, or only look alarming?

The floor is real: at theta_0 the preservation loss is 0.0219 on a window whose
pose_enc matches the label bit for bit, and acos(1-1e-6) * lam_rot accounts for
0.0212 of it (experiments/loss_floor_probe.py).  Whether that MATTERS is a
different question, and the alarming-sounding "d(acos)/d(cos) = 707" is not the
right frame: L_rot is a mean over 1128 pairs, so the per-pair factor after
averaging is ~0.6, and in terms of the rotation ANGLE the gradient of acos is
exactly 1 -- bounded, not singular.

The real difference between the two candidate forms is whether the gradient
VANISHES when the student is already right:

    L = acos(cos) = theta        dL/dtheta = 1          (never settles)
    L = 1 - cos(theta)           dL/dtheta = sin theta  (vanishes at 0)

So this probe measures, at theta_0, on a real window:

  * how much of the total parameter gradient each loss term is responsible for,
  * the distribution of per-pair rotation residuals, and how many are clamped,
  * what the same window would give under the chordal form.

    .venv-bench/bin/python experiments/rot_term_probe.py --scene kth_day_10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lingbot_map.train.label_bank import LabelBank                    # noqa: E402
from lingbot_map.train.losses import (PRESETS, _pair_index, _relative,  # noqa: E402
                                      depth_si_loss, motion_depth_loss,
                                      rel_pose_loss)
from phase0_density_sweep import build_model                          # noqa: E402
from loss_floor_probe import build_prefix, load_images                # noqa: E402


def gnorm(model) -> float:
    n = 0.0
    for p in model.parameters():
        if p.grad is not None:
            n += float((p.grad.float() ** 2).sum())
    return math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--runs", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--preset", default="A1PC")
    ap.add_argument("--ckpt", default="/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/"
                                      "youngmin/ckpt/lingbot-map.pt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    bank = LabelBank(os.path.join(ROOT, "labels", a.scene), cache_runs=2)
    P = PRESETS[a.preset]
    dev, dtype = torch.device("cuda"), torch.bfloat16
    model = build_model(a.ckpt, dev, 518, 14, 20000, 64, 8)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    rec = {"scene": a.scene, "preset": a.preset, "lam": dict(P), "rows": []}
    for rid in a.runs:
        r = bank.runs[rid]
        sf_b, B, Kt = r["scale_frames"], r["burn_in"], r["teacher_interval"]
        a0 = r["t0"] - B * Kt - sf_b
        ws = r["t0"] - a0
        if a0 < 0 or r["L"] < a.S:
            continue
        sub = load_images(frames, r["t0"] + a.S)[:, a0:r["t0"] + a.S]
        lab = bank.get(rid, 0, a.S, device=dev)

        build_prefix(model, sub, sf_b, ws, Kt, dtype, dev)
        with model.masked_window(window_start=ws, keyframe_interval=Kt):
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model(sub[:, ws:ws + a.S].to(dev), num_frame_for_scale=sf_b,
                            num_frame_per_block=a.S, causal_inference=False)
        sp = out["pose_enc"][0].float()
        sd_ = out["depth"][0].float()
        tp, td = lab["pose_enc"].detach(), lab["depth"].detach()
        conf = lab.get("depth_conf")
        conf = conf.detach() if conf is not None else None

        # ── per-pair rotation residual, as an angle ─────────────────────────
        with torch.no_grad():
            N = sp.shape[0]
            i, j = _pair_index(N, P.get("pairs", "all"), P.get("min_gap", 1), sp.device)
            dRs, _ = _relative(sp, i, j)
            dRt, _ = _relative(tp, i, j)
            dR = torch.einsum("nij,njk->nik", dRs.transpose(1, 2), dRt)
            cos = ((dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]) - 1) / 2
            clamped = int((cos >= 1 - 1e-6).sum())
            th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            q = [float(torch.quantile(th, x)) for x in (0.5, 0.9, 0.99)]

        terms = {}
        for name in ("rot", "dir", "dep", "mot", "rot_chordal"):
            model.zero_grad(set_to_none=True)
            L_rot, L_dir, _ = rel_pose_loss(sp, tp, mag_mode=P.get("mag_mode", "closed_form_scale"),
                                            pairs=P.get("pairs", "all"),
                                            min_gap=P.get("min_gap", 1),
                                            mag_trunc=P.get("mag_trunc", 1.0))
            if name == "rot":
                L = P["lam_rot"] * L_rot
            elif name == "dir":
                L = P["lam_dir"] * L_dir
            elif name == "dep":
                L = P["lam_depth"] * depth_si_loss(sd_, td, conf,
                                                   mode=P.get("depth_mode", "median"),
                                                   trunc=P.get("depth_trunc", 0.1),
                                                   align_res=P.get("align_res", 518))
            elif name == "mot":
                L = P["lam_motion"] * motion_depth_loss(
                    sp, tp, sd_, td, pairs=P.get("pairs", "all"),
                    min_gap=P.get("min_gap", 1))
            else:                       # chordal: 1 - cos, smooth and vanishing
                dRs2, _ = _relative(sp, i, j)
                dRt2, _ = _relative(tp, i, j)
                dR2 = torch.einsum("nij,njk->nik", dRs2.transpose(1, 2), dRt2)
                c2 = ((dR2[:, 0, 0] + dR2[:, 1, 1] + dR2[:, 2, 2]) - 1) / 2
                L = P["lam_rot"] * (1 - c2).mean()
            L.backward(retain_graph=True)
            terms[name] = {"value": float(L.detach()), "gnorm": gnorm(model)}
        model.zero_grad(set_to_none=True)
        del out

        tot_g = math.sqrt(sum(terms[k]["gnorm"] ** 2
                              for k in ("rot", "dir", "dep", "mot")))
        row = {"run": rid, "pairs": int(len(cos)), "clamped": clamped,
               "theta_med_deg": math.degrees(q[0]), "theta_p90_deg": math.degrees(q[1]),
               "theta_p99_deg": math.degrees(q[2]), "terms": terms,
               "quadrature_total_gnorm": tot_g}
        rec["rows"].append(row)

        print(f"\nrun {rid}: {len(cos)} pairs, {clamped} clamped at cos>=1-1e-6 "
              f"({clamped / len(cos):.1%})")
        print(f"  per-pair rotation residual: median {math.degrees(q[0]):.4f}deg  "
              f"p90 {math.degrees(q[1]):.4f}  p99 {math.degrees(q[2]):.4f}")
        print(f"  {'term':>12} {'weighted value':>15} {'||grad||':>12} {'share':>8}")
        for k in ("rot", "dir", "dep", "mot"):
            print(f"  {k:>12} {terms[k]['value']:15.5f} {terms[k]['gnorm']:12.4f} "
                  f"{terms[k]['gnorm'] / tot_g:8.1%}")
        print(f"  {'rot_chordal':>12} {terms['rot_chordal']['value']:15.5f} "
              f"{terms['rot_chordal']['gnorm']:12.4f}  <- same weight, smooth form")

    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"\n[probe] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
