#!/usr/bin/env python3
"""What would it cost to score the training window the way the teacher did?

experiments/loss_floor_probe.py showed the two paths disagree at theta_0: the
bank was made one frame at a time (``num_frame_per_block=1,
causal_inference=True``) and both training branches score S frames in one
parallel masked forward.  Removing that mismatch means scoring sequentially --
so this measures what that costs, in time and in peak memory, for the same
window.

Three variants, because they are not the same thing:

  parallel     one masked forward over S frames                (what training does)
  seq-detach   S single-frame forwards, cache detached between (deployment shape,
               but no gradient through the student's own KV)
  seq-full     S single-frame forwards, cache kept in the graph (the honest
               sequential objective, and the expensive one)

    .venv-bench/bin/python experiments/seq_scoring_cost.py --S 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lingbot_map.train.label_bank import LabelBank                    # noqa: E402
from lingbot_map.train.losses import SelfDistillLoss                  # noqa: E402
from phase0_density_sweep import build_model                          # noqa: E402
from loss_floor_probe import build_prefix, load_images                # noqa: E402


def timed(fn, warmup=0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 2**30, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kth_day_10")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--preset", default="A1PC")
    ap.add_argument("--grad", type=int, default=1)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--ckpt", default="/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/"
                                      "youngmin/ckpt/lingbot-map.pt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = os.path.join(ROOT, "data", "mcd", a.scene, "frames_10hz")
    bank = LabelBank(os.path.join(ROOT, "labels", a.scene), cache_runs=2)
    crit = SelfDistillLoss.from_preset(a.preset)
    dev, dtype = torch.device("cuda"), torch.bfloat16
    model = build_model(a.ckpt, dev, 518, 14, 20000, 64, 8)
    model.train()
    for p in model.parameters():
        p.requires_grad_(bool(a.grad))

    r = bank.runs[a.run]
    sf_b, B, Kt = r["scale_frames"], r["burn_in"], r["teacher_interval"]
    a0 = r["t0"] - B * Kt - sf_b
    ws = r["t0"] - a0
    sub = load_images(frames, r["t0"] + a.S)[:, a0:r["t0"] + a.S]
    lab = bank.get(a.run, 0, a.S, device=dev)
    S = a.S
    from grad_probe import detach_caches

    def score(total):
        if a.grad and total is not None:
            total.backward()
        model.zero_grad(set_to_none=True)

    def parallel():
        build_prefix(model, sub, sf_b, ws, Kt, dtype, dev)
        with model.masked_window(window_start=ws, keyframe_interval=Kt):
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model(sub[:, ws:ws + S].to(dev), num_frame_for_scale=sf_b,
                            num_frame_per_block=S, causal_inference=False)
        t, _ = crit(out["pose_enc"][0].float(), lab["pose_enc"],
                    out["depth"][0].float(), lab["depth"], lab.get("depth_conf"))
        score(t)
        return float(t.detach())

    def sequential(detach: bool):
        build_prefix(model, sub, sf_b, ws, Kt, dtype, dev)
        pe, dp = [], []
        for i in range(ws, ws + S):
            with torch.amp.autocast("cuda", dtype=dtype):
                o = model(sub[:, i:i + 1].to(dev), num_frame_for_scale=sf_b,
                          num_frame_per_block=1, causal_inference=True)
            pe.append(o["pose_enc"].float())
            dp.append(o["depth"].float())
            if detach:
                detach_caches(model)
        t, _ = crit(torch.cat(pe, 1)[0], lab["pose_enc"],
                    torch.cat(dp, 1)[0], lab["depth"], lab.get("depth_conf"))
        score(t)
        return float(t.detach())

    rec = {"scene": a.scene, "run": a.run, "S": S, "grad": bool(a.grad), "variants": {}}
    plan = [("parallel", parallel), ("seq-detach", lambda: sequential(True))]
    if not a.skip_full:
        plan.append(("seq-full", lambda: sequential(False)))

    print(f"\nscene {a.scene} run {a.run}  S={S}  grad={bool(a.grad)}")
    print(f"{'variant':>12} {'loss':>10} {'time (s)':>10} {'peak GiB':>10}")
    base_t = None
    for name, fn in plan:
        try:
            dt, mem, val = timed(fn)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{name:>12} {'-':>10} {'OOM':>10} {'-':>10}")
            rec["variants"][name] = {"oom": True}
            continue
        base_t = base_t or dt
        rec["variants"][name] = {"loss": val, "sec": dt, "peak_gib": mem,
                                 "x_parallel": dt / base_t}
        print(f"{name:>12} {val:10.5f} {dt:10.3f} {mem:10.2f}"
              + (f"   {dt / base_t:.1f}x" if name != "parallel" else ""))
        torch.cuda.empty_cache()

    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"[cost] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
