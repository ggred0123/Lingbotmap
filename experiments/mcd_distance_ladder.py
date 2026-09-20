#!/usr/bin/env python3
"""Is the Oxford damage a DISTANCE regime or a SCENE domain?  results-ledger D1/D4.

Every trained arm improves its in-domain GT probe and degrades Oxford K=1.  The
two explanations left after the gtabs cell are (a) the training runs never
reach the distance where the damage grows (MCD 240 frames = 38 m, Oxford 240
frames = 380 m) and (b) outdoor driving is simply a different scene domain.
They are separable inside MCD: the same held-out MCD scenes (GT, never trained
on) are streamed at frame strides 1 / 4 / 8, i.e. ~0.19 / 0.76 / 1.5 m per
frame, the last matching Oxford's 1.6 m/frame, for 320 frames each -- the
Oxford K=1 protocol at three distances.  The NTU scenes are a different campus
from the KTH/TUHH training scenes, so a scene-domain axis rides along too.

This script only STREAMS: one checkpoint, every sequence, poses and per-frame
depth medians to an npz per sequence.  mcd_distance_score.py does the scoring.

    CUDA_VISIBLE_DEVICES=0 python3 experiments/mcd_distance_ladder.py \\
        --ckpts base=/.../ckpt/lingbot-map.pt gtctrl_s275=bench_ckpt/sd_gtctrl_step275.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lingbot_map.train.label_bank import image_names                    # noqa: E402
from lingbot_map.utils.load_fn import load_and_preprocess_images        # noqa: E402
from bake_gt_bank import gt_rows_for                                    # noqa: E402
from phase0_density_sweep import build_model                            # noqa: E402

OUT = os.path.join(ROOT, "experiments", "results", "mcd_dist")
HOLDOUT = ["kth_day_06", "kth_day_09", "ntu_day_01", "ntu_day_02", "ntu_day_10",
           "ntu_night_04", "ntu_night_08", "ntu_night_13"]
TRAIN = ["kth_day_10", "kth_night_01", "kth_night_04", "kth_night_05", "tuhh_day_02",
         "tuhh_day_03", "tuhh_day_04", "tuhh_night_07", "tuhh_night_08", "tuhh_night_09"]


def plan_sequences(scenes, targets, n_frames, per_scene, calib, sensor):
    """(scene, stride, start) triples whose every frame has a GT row.

    ★ THE KNOB IS METRES PER FRAME, NOT THE STRIDE.  The NTU scenes move at
    ~0.55 m/frame at 10 Hz and the KTH ones at ~0.16, so one stride would put
    the two campuses at different distances.  Each scene gets the stride that
    lands nearest each target speed (0.16 = the training regime, 1.6 = Oxford),
    and the realised m/frame is stored with the sequence.
    """
    seqs = []
    for sc in scenes:
        fd = os.path.join(ROOT, "data", "mcd", sc, "frames_10hz")
        if not os.path.exists(os.path.join(fd, "meta.npz")):
            continue
        rows, gp, _ = gt_rows_for(fd, calib, sensor)
        n = len(rows)
        valid = np.where(rows >= 0)[0]
        lo, hi = int(valid[0]), int(valid[-1]) + 1
        speed = float(np.median(np.linalg.norm(np.diff(gp, axis=0), axis=1)))
        strides = sorted({max(1, int(round(t / speed))) for t in targets})
        for st in strides:
            span = n_frames * st
            if span > hi - lo:
                continue
            # the far target is the question; the near/mid ones are controls
            # and one window each is enough for them
            k = min(per_scene if st == strides[-1] else 1, (hi - lo) // span)
            starts = ([lo + int(round(i * (hi - lo - span) / (k - 1))) for i in range(k)]
                      if k > 1 else [lo])
            for s0 in starts:
                idx = np.arange(s0, s0 + span, st)
                if (rows[idx] < 0).any():
                    continue
                path = float(np.linalg.norm(np.diff(gp[rows[idx]], axis=0), axis=1).sum())
                seqs.append({"scene": sc, "stride": st, "start": int(s0), "frames": idx.tolist(),
                             "holdout": sc in HOLDOUT, "scene_speed": speed,
                             "m_per_frame": path / (len(idx) - 1), "path_m": path,
                             "target": min(targets, key=lambda t: abs(t - path / (len(idx) - 1)))})
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path")
    ap.add_argument("--scenes", nargs="*", default=HOLDOUT)
    ap.add_argument("--targets", type=float, nargs="*", default=[0.16, 0.8, 1.6],
                    help="metres per frame to aim for; 0.16 = MCD 10 Hz (training), 1.6 = Oxford")
    ap.add_argument("--n_frames", type=int, default=320)
    ap.add_argument("--per_scene", type=int, default=2)
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    a = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)

    seqs = plan_sequences(a.scenes, a.targets, a.n_frames, a.per_scene,
                          os.path.join(ROOT, a.calib), a.sensor)
    # the sequence list ACCUMULATES across invocations (hold-out scenes first,
    # the training-scene control later), keyed by (scene, stride, start)
    sp = os.path.join(OUT, "sequences.json")
    have = {(x["scene"], x["stride"], x["start"]): x for x in (json.load(open(sp)) if os.path.exists(sp) else [])}
    for x in seqs:
        have[(x["scene"], x["stride"], x["start"])] = x
    json.dump(list(have.values()), open(sp, "w"), indent=1)
    print(f"[plan] {len(seqs)} sequences: " + ", ".join(
        f"target {t}: {sum(1 for s in seqs if s['target'] == t)}" for t in a.targets), flush=True)
    for s in seqs:
        print(f"    {s['scene']:<14} stride {s['stride']:>2} @ {s['start']:>5}  {s['m_per_frame']:.2f} m/frame  "
              f"{s['path_m']:.0f} m  -> target {s['target']}", flush=True)

    for spec in a.ckpts:
        name, _, path = spec.partition("=")
        odir = os.path.join(OUT, name)
        os.makedirs(odir, exist_ok=True)
        todo = [s for s in seqs if not os.path.exists(
            os.path.join(odir, f"{s['scene']}_s{s['stride']}_f{s['start']}.npz"))]
        if not todo:
            print(f"[{name}] all {len(seqs)} sequences present, skipping", flush=True)
            continue
        t = time.time()
        model = build_model(path, dev, a.image_size, a.patch_size, 1024,
                            a.kv_cache_sliding_window, a.num_scale_frames)
        model.eval()
        print(f"[{name}] loaded {path} in {time.time() - t:.0f}s; {len(todo)} sequences to run", flush=True)
        for s in todo:
            fd = os.path.join(ROOT, "data", "mcd", s["scene"], "frames_10hz")
            names = image_names(fd)
            paths = [os.path.join(fd, names[i]) for i in s["frames"]]
            t = time.time()
            images = load_and_preprocess_images(paths, mode="crop", image_size=a.image_size,
                                                patch_size=a.patch_size)
            model.clean_kv_cache()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model.inference_streaming(images, num_scale_frames=a.num_scale_frames,
                                                 keyframe_interval=1, output_device=torch.device("cpu"))
            pe = pred["pose_enc"][0].float().numpy()
            d = pred["depth"][0].float()
            d = d[..., 0] if d.dim() == 4 else d
            n = d.shape[0]
            dflat = d.reshape(n, -1)
            depth_med = dflat.median(dim=1).values.numpy()
            # a thinned depth map (every 8th pixel each way) for scale/ratio work
            depth_thin = d[:, ::8, ::8].numpy().astype(np.float16)
            np.savez_compressed(os.path.join(odir, f"{s['scene']}_s{s['stride']}_f{s['start']}.npz"),
                                pose_enc=pe, depth_med=depth_med, depth_thin=depth_thin,
                                frames=np.asarray(s["frames"]), stride=s["stride"])
            del pred, d, dflat
            torch.cuda.empty_cache()
            print(f"[{name}] {s['scene']} stride {s['stride']} @ {s['start']}: {n} frames in "
                  f"{time.time() - t:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("[ladder] DONE", flush=True)


if __name__ == "__main__":
    main()
