#!/usr/bin/env python3
"""Is the step curve already a learning-rate sweep?

Every run clips ``grad_total`` (18-20) to ``--clip 1.0`` on 100% of steps, so the
update is a normalised direction scaled by lr, not by the loss.  If that is the
whole story then cumulative movement is roughly lr x steps, and lowering lr 10x
just relabels the x axis: the damage a checkpoint carries should depend on HOW
FAR it moved from theta_0, not on which (lr, step) pair got it there.

That is testable without another training run.  Measure ||theta - theta_0|| for
every saved checkpoint and plot the benchmark against it.  If the arms fall on
one curve, lr/10 is predicted to land at 1/10 the distance -- a point the
existing step curve has already measured -- and buys nothing new.  If they do
not, the step size matters on its own and the run is worth doing.

    .venv-bench/bin/python experiments/weight_distance.py
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

BASE = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt"


def flat_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "ema", "weights"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    return {k: v for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["s0off", "s0on", "v6i"])
    ap.add_argument("--steps", type=int, nargs="*",
                    default=[50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1250])
    ap.add_argument("--out", default="experiments/results/weight_distance.json")
    a = ap.parse_args()

    print("loading base ...", flush=True)
    b = flat_state(BASE)
    bn = math.sqrt(sum(float(v.double().pow(2).sum()) for v in b.values()))
    print(f"||theta_0|| = {bn:.4f}  ({len(b)} float tensors)", flush=True)

    rec = {"base_norm": bn, "arms": {}}
    print(f"\n{'run':>8}{'step':>7}{'||d||':>12}{'rel':>10}")
    for arm in a.arms:
        rec["arms"][arm] = []
        for s in a.steps:
            p = os.path.join(ROOT, "ckpt_train", f"{arm}.step{s}.pt")
            if not os.path.exists(p):
                continue
            try:
                w = flat_state(p)
            except Exception as e:
                print(f"{arm:>8}{s:>7}  load failed: {e}")
                continue
            d = 0.0
            for k, v in b.items():
                u = w.get(k)
                if u is None or u.shape != v.shape:
                    continue
                d += float((u.double() - v.double()).pow(2).sum())
            d = math.sqrt(d)
            rec["arms"][arm].append({"step": s, "dist": d, "rel": d / bn})
            print(f"{arm:>8}{s:>7}{d:12.4f}{d / bn:10.5f}", flush=True)
            del w
    json.dump(rec, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"\n[dist] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
