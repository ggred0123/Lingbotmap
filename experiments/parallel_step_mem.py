"""Model-level peak memory of a parallel S-frame training step.

The attention microbenchmarks measured one component in isolation.  This
measures the whole model: patch_embed (DINOv2, 304M) + frame_blocks (302M) +
24 global blocks (302M) + camera_head (216M, 4 refinement iters) + depth_head
(DPT), forward and backward, as a function of the parallel window length S and
of gradient checkpointing.

IMPORTANT -- what this is NOT.  It runs the aggregator's *batch* path
(``causal_inference=False``), which is the only parallel path that exists in
the released code.  That path applies NO attention mask: the aggregator's 24
global blocks are SDPABlock/FlashInferBlock, both cache-only, and the masked
GCA formulation lives solely in CameraBlock (block.py:313-460), i.e. the camera
head.  So these numbers are the activation cost of a parallel S-frame step with
full attention; the GCA mask tensor cost, measured separately, adds on top.

Usage:
    python experiments/parallel_step_mem.py --ckpt ... --frames ... \
        --sweep 2 4 8 16 24 32 48 64
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from fork_rig import run_anchor
from grad_probe import detach_caches, snapshot_state_cpu, restore_state_cpu


def build(ckpt, dev, image_size, patch_size, max_frame_num, sw, sf, ckpt_grad):
    m = GCTStream(
        img_size=image_size, patch_size=patch_size, enable_3d_rope=True,
        max_frame_num=max_frame_num, kv_cache_sliding_window=sw,
        kv_cache_scale_frames=sf, kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True, use_sdpa=True,
        camera_num_iterations=4, use_gradient_checkpoint=ckpt_grad,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    miss, unexp = m.load_state_dict(sd, strict=False)
    del sd
    return m.to(dev), len(miss), len(unexp)


def build_prefix(model, images, t0, sf, K, dtype, dev):
    """Roll the student to t0 the way it is deployed, detaching every step.

    This is the Level-0 boundary: the prefix cache becomes a constant that the
    supervised window attends to.  Without it, a window measurement is a
    standalone clip and understates the real step -- the supervised window's
    KV is prefix + window, not window alone.
    """
    model.clean_kv_cache()
    run_anchor(model, images[:, :sf], sf, dtype, dev)
    detach_caches(model)
    for i in range(sf, t0):
        is_kf = (K <= 1) or ((i - sf) % K == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        del out
        detach_caches(model)


def cache_tokens(model, tpf):
    """Prefix KV size in tokens, split into full-token frames and special tokens."""
    agg = model.aggregator
    if not isinstance(getattr(agg, "kv_cache", None), dict):
        return 0, 0
    full = spec = 0
    k0 = agg.kv_cache.get("k_0")
    if torch.is_tensor(k0):
        full = k0.shape[2] * k0.shape[3]
    ks = agg.kv_cache.get("k_0_special")
    if torch.is_tensor(ks):
        spec = ks.shape[2] * ks.shape[3]
    return full, spec


def one_step(model, images, S, sf, dtype, dev, opt=None, t0=0, snap=None):
    """Parallel forward over an S-frame window + backward on a dummy loss.

    t0 == 0  -> standalone window, cache cleared (no prefix).
    t0  > 0  -> window attends to the prefix cache restored from `snap`.
                The window is pushed as ONE block, so Q = S*P and KV = prefix +
                S*P, which is the tensor shape the masked training step will
                have.  Attention inside the window is bidirectional here rather
                than frame-causal -- the mask only changes which entries are
                read, not the workspace, so this is the right memory proxy.
    """
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    if t0 == 0:
        model.clean_kv_cache()
    else:
        restore_state_cpu(model, snap, dev)
        detach_caches(model)
    torch.cuda.reset_peak_memory_stats()

    lo = t0 if t0 else 0
    t_start = time.time()
    with torch.amp.autocast("cuda", dtype=dtype):
        out = model.forward(
            images[:, lo:lo + S].to(dev),
            num_frame_for_scale=sf,
            num_frame_per_block=S,           # whole window in one block = parallel
            causal_inference=bool(t0),       # t0>0: cached path; else batch path
        )
    fwd_gb, fwd_s = torch.cuda.max_memory_allocated() / 1e9, time.time() - t_start

    # Stand-in for the §3.3 pair: something that touches both heads so the
    # backward graph covers pose and depth, at the right magnitude.
    loss = out["pose_enc"].float().pow(2).mean() + out["depth"].float().pow(2).mean()
    t1 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_s = time.time() - t1

    rep = {"S": S, "peak_gb_fwd": fwd_gb, "peak_gb_total": torch.cuda.max_memory_allocated() / 1e9,
           "fwd_s": fwd_s, "bwd_s": bwd_s, "loss": float(loss)}
    if opt is not None:                  # include optimizer state in the peak
        t2 = time.time()
        opt.step()
        torch.cuda.synchronize()
        rep["opt_s"] = time.time() - t2
        rep["peak_gb_with_opt"] = torch.cuda.max_memory_allocated() / 1e9
    del out, loss
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="experiments/results/parallel_step_mem.json")
    ap.add_argument("--sweep", type=int, nargs="+", default=[2, 4, 8, 16, 24, 32, 48, 64])
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--prefix_t0", type=int, default=0,
                    help="roll the student to this frame first, so the window attends "
                         "to a real deployed prefix cache (0 = standalone window)")
    ap.add_argument("--K", type=int, default=28, help="student keyframe interval for the prefix")
    ap.add_argument("--ckpt_modes", type=int, nargs="+", default=[0, 1],
                    help="0=checkpointing off, 1=on")
    ap.add_argument("--with_optimizer", action="store_true",
                    help="also build AdamW and run a step, so states count toward the peak")
    args = ap.parse_args()

    dev, dtype = torch.device("cuda"), torch.bfloat16
    sf = args.num_scale_frames
    need = args.prefix_t0 + max(args.sweep) if args.prefix_t0 else max(args.sweep)
    names = sorted(os.listdir(args.frames))[:need]
    paths = [os.path.join(args.frames, n) for n in names]
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    H, W = images.shape[-2:]
    tpf = (H // args.patch_size) * (W // args.patch_size) + 6
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[cfg] {H}x{W}, {tpf} tok/frame, card {total:.0f} GB, "
          f"optimizer={'yes' if args.with_optimizer else 'no'}")

    res = {"meta": {**vars(args), "H": H, "W": W, "tokens_per_frame": tpf,
                    "gpu": torch.cuda.get_device_name(0), "total_gb": total},
           "runs": {}}

    for ckpt_grad in [bool(m) for m in args.ckpt_modes]:
        tag = "checkpoint_ON" if ckpt_grad else "checkpoint_OFF"
        if args.prefix_t0:
            tag += f"_prefix{args.prefix_t0}"
        print(f"\n=== {tag} ===")
        model, miss, unexp = build(args.ckpt, dev, args.image_size, args.patch_size,
                                   args.max_frame_num, args.kv_cache_sliding_window,
                                   sf, ckpt_grad)
        model.train()                     # checkpointing is gated on self.training
        for p in model.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5) if args.with_optimizer else None
        print(f"  loaded (missing={miss}, unexpected={unexp}), "
              f"params {sum(p.numel() for p in model.parameters())/1e6:.0f} M")

        snap = None
        if args.prefix_t0:
            t_r = time.time()
            build_prefix(model, images, args.prefix_t0, sf, args.K, dtype, dev)
            full, spec = cache_tokens(model, tpf)
            snap = snapshot_state_cpu(model)
            print(f"  prefix rolled to t0={args.prefix_t0} in {time.time()-t_r:.0f}s: "
                  f"{full} full-token + {spec} special = {full+spec} KV tokens "
                  f"({(full+spec)/tpf:.1f} frame-equivalents)")
            res["meta"]["prefix_kv_tokens"] = full + spec
            res["meta"]["prefix_full_tokens"] = full
            res["meta"]["prefix_special_tokens"] = spec
        hdr = f"  {'S':>4} {'peak fwd':>10} {'peak total':>12}"
        if opt is not None:
            hdr += f" {'w/ opt':>9}"
        print(hdr + f" {'fwd s':>8} {'bwd s':>8}")

        rows = []
        for S in args.sweep:
            try:
                r = one_step(model, images, S, sf, dtype, dev, opt,
                             t0=args.prefix_t0, snap=snap)
                rows.append(r)
                line = f"  {S:>4} {r['peak_gb_fwd']:>10.1f} {r['peak_gb_total']:>12.1f}"
                if opt is not None:
                    line += f" {r['peak_gb_with_opt']:>9.1f}"
                print(line + f" {r['fwd_s']:>8.2f} {r['bwd_s']:>8.2f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                rows.append({"S": S, "oom": True})
                print(f"  {S:>4} {'OOM':>10}", flush=True)
                torch.cuda.empty_cache()
                break
        res["runs"][tag] = rows
        del model, opt, snap
        torch.cuda.empty_cache()

    # slope + max feasible S
    print("\n=== summary ===")
    for tag, rows in res["runs"].items():
        ok = [r for r in rows if not r.get("oom")]
        if len(ok) >= 2:
            a, b = ok[-2], ok[-1]
            slope = (b["peak_gb_total"] - a["peak_gb_total"]) / (b["S"] - a["S"])
            budget = res["meta"]["total_gb"] - 19          # AdamW states for 1.16B
            smax = (budget - b["peak_gb_total"]) / slope + b["S"]
            print(f"  {tag:<16} slope {slope:>6.2f} GB/frame   "
                  f"largest measured S={ok[-1]['S']} at {ok[-1]['peak_gb_total']:.1f} GB   "
                  f"-> S_max with optimizer ~= {smax:.0f}")
        res["runs"][tag] = rows

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
