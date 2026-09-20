#!/usr/bin/env python3
"""Is L_fresh really ~0 at theta_0, and if not, where does the floor come from?

``fresh_step``'s docstring says "at step 0 the two are the same weights and this
is ~0 by construction".  The measured identity-branch loss at step 0 of the
p_identity=1.0 run is 0.025-0.030, with a gradient norm of 4.0, and it rises to
~0.20 by step 1200 at MATCHED window depth.  A preservation term that is not
zero at the point it is meant to preserve is pulling the model somewhere from
the very first step.

There is a candidate reason in the code.  The bank was produced one frame at a
time -- ``label_bank.stream_collect`` runs ``num_frame_per_block=1,
causal_inference=True``, which is exactly the deployment path.  Both training
branches instead score the window in ONE parallel forward,
``num_frame_per_block=S, causal_inference=False`` inside ``masked_window``.  The
GCA mask is meant to make the parallel path equivalent to the sequential one, so
the question is whether it actually is.

This probe answers that without training anything: build the teacher's own
prefix at theta_0, then score the same window twice against the same labels --
once the way training does, once the way the teacher did.

    .venv-bench/bin/python experiments/loss_floor_probe.py --scene kth_day_10
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lingbot_map.train.label_bank import LabelBank                  # noqa: E402
from lingbot_map.train.losses import SelfDistillLoss                # noqa: E402
from phase0_density_sweep import build_model                        # noqa: E402


def load_images(frames_dir, upto, image_size=518, patch_size=14):
    cache = os.path.join(frames_dir, f"_cache_{image_size}_{patch_size}.npy")
    if os.path.exists(cache):
        import numpy as np
        arr = np.load(cache, mmap_mode="r")
        return torch.from_numpy(np.ascontiguousarray(arr[:upto])).unsqueeze(0)
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.train.trainer import image_names
    names = image_names(frames_dir)[:upto]
    return load_and_preprocess_images(
        [os.path.join(frames_dir, n) for n in names], mode="crop",
        image_size=image_size, patch_size=patch_size).unsqueeze(0)


def build_prefix(model, sub, sf_b, ws, Kt, dtype, dev):
    """Anchor + burn-in exactly as fresh_step (and the teacher) did."""
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.forward(sub[:, :sf_b].to(dev), num_frame_for_scale=sf_b,
                      num_frame_per_block=sf_b, causal_inference=True)
    for i in range(sf_b, ws):
        is_kf = (Kt <= 1) or ((i - sf_b) % Kt == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(sub[:, i:i + 1].to(dev), num_frame_for_scale=sf_b,
                          num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--bank", default=None)
    ap.add_argument("--runs", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--ckpt", default="/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/"
                                      "youngmin/ckpt/lingbot-map.pt")
    ap.add_argument("--preset", default="A1PC")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = a.frames or os.path.join(ROOT, "data", a.scene, "frames_10hz")
    if not os.path.isdir(frames):
        frames = os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    bank_dir = a.bank or os.path.join(ROOT, "labels", a.scene)
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    bank = LabelBank(bank_dir, cache_runs=2)
    crit = SelfDistillLoss.from_preset(a.preset)
    model = build_model(a.ckpt, dev, 518, 14, 20000, 64, 8)
    model.eval()

    rec = {"scene": a.scene, "preset": a.preset, "S": a.S, "rows": []}
    print(f"\nscene {a.scene}  preset {a.preset}  S={a.S}")
    print(f"{'run':>4} {'off':>4} | {'L_masked':>10} {'L_seq':>10} | "
          f"{'pose_masked':>12} {'pose_seq':>12} | {'seq vs masked':>13}")
    for rid in a.runs:
        if rid >= len(bank.runs):
            continue
        r = bank.runs[rid]
        sf_b, B, Kt = r["scale_frames"], r["burn_in"], r["teacher_interval"]
        a0 = r["t0"] - B * Kt - sf_b
        ws = r["t0"] - a0
        if a0 < 0 or r["L"] < a.S:
            continue
        imgs = load_images(frames, r["t0"] + a.S)
        sub = imgs[:, a0:r["t0"] + a.S]
        lab = bank.get(rid, 0, a.S, device=dev)

        # ── path A: what training scores, one parallel masked window ─────────
        build_prefix(model, sub, sf_b, ws, Kt, dtype, dev)
        with torch.no_grad():
            with model.masked_window(window_start=ws, keyframe_interval=Kt):
                with torch.amp.autocast("cuda", dtype=dtype):
                    out_m = model(sub[:, ws:ws + a.S].to(dev),
                                  num_frame_for_scale=sf_b,
                                  num_frame_per_block=a.S, causal_inference=False)
            l_m, _ = crit(out_m["pose_enc"][0].float(), lab["pose_enc"],
                          out_m["depth"][0].float(), lab["depth"],
                          lab.get("depth_conf"))
            pe_m = out_m["pose_enc"][0].float().cpu()
        del out_m

        # ── path B: what the teacher did, one frame at a time ────────────────
        build_prefix(model, sub, sf_b, ws, Kt, dtype, dev)
        pe, dep = [], []
        with torch.no_grad():
            for i in range(ws, ws + a.S):
                is_kf = (Kt <= 1) or ((i - sf_b) % Kt == 0)
                if not is_kf:
                    model._set_skip_append(True)
                with torch.amp.autocast("cuda", dtype=dtype):
                    o = model(sub[:, i:i + 1].to(dev), num_frame_for_scale=sf_b,
                              num_frame_per_block=1, causal_inference=True)
                if not is_kf:
                    model._set_skip_append(False)
                pe.append(o["pose_enc"].float())
                dep.append(o["depth"].float())
                del o
            pe_s = torch.cat(pe, dim=1)
            dep_s = torch.cat(dep, dim=1)
            l_s, _ = crit(pe_s[0], lab["pose_enc"], dep_s[0], lab["depth"],
                          lab.get("depth_conf"))

        lp = lab["pose_enc"].float().cpu()
        d_m = (pe_m - lp).abs().max().item()
        d_s = (pe_s[0].cpu() - lp).abs().max().item()
        d_ms = (pe_m - pe_s[0].cpu()).abs().max().item()
        rec["rows"].append({"run": rid, "L_masked": float(l_m), "L_seq": float(l_s),
                            "pose_masked": d_m, "pose_seq": d_s, "masked_vs_seq": d_ms})
        print(f"{rid:4d} {0:4d} | {float(l_m):10.5f} {float(l_s):10.5f} | "
              f"{d_m:12.2e} {d_s:12.2e} | {d_ms:13.2e}")
    print("\nL_masked = the loss training actually minimises.")
    print("L_seq    = the same weights and window scored the way the bank was made.")
    print("pose_*   = max |pose_enc - label| for each path.")
    if a.out:
        import json, math as _m
        floor = _m.acos(1 - 1e-6)
        rec["acos_floor_rad"] = floor
        rec["acos_floor_deg"] = _m.degrees(floor)
        rec["lam_rot"] = 15.0
        rec["floor_contribution"] = 15.0 * floor
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"[probe] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
