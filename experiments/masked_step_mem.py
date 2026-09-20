"""Peak memory of the T1 masked step vs. the full-attention proxy it replaces.

docs/phase1-plan.md §1.4 / §5 budget the supervised step from
``parallel_step_mem.py``, which had to use the aggregator's *cache* path with a
cleaned cache -- the only parallel path that existed before T1.  That proxy does
two things the real masked step does not:

  * it WRITES the window's K/V into the cache and then ``.clone()``s it back out
    (attention.py:645-656), so 24 layers x 2 x (S x P x C) stay in the graph
  * it attends with no mask at all

So §5's 132.8 GB at S=48 is an upper bound of unknown tightness.  Now that the
masked path exists, measure it directly.

Every mode runs a warm-up step BEFORE the measured one so AdamW state is already
resident: measuring the first step charges optimizer-state allocation to
whichever mode ran first, which is how the §6-Q2 note says these numbers get
flipped.

    python experiments/masked_step_mem.py --ckpt ... --frames ... --sweep 16 32 48 64
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from parallel_step_mem import build, build_prefix, cache_tokens
from grad_probe import detach_caches, snapshot_state_cpu, restore_state_cpu


def step(model, images, S, sf, dtype, dev, opt, mode, K, t0=0, snap=None):
    model.zero_grad(set_to_none=True)
    if snap is None:
        model.clean_kv_cache()
    else:
        restore_state_cpu(model, snap, dev)
        detach_caches(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()

    if mode == "full_attn_cache":
        # what §1.4 measured: cache path, no mask, window written into the cache
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, t0:t0 + S].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=S, causal_inference=bool(t0))
    else:
        with model.masked_window(window_start=t0, keyframe_interval=K,
                                 mask_dtype=mode.split("_")[-1]):
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model.forward(images[:, t0:t0 + S].to(dev), num_frame_for_scale=sf,
                                    num_frame_per_block=S, causal_inference=False)
    fwd_gb, fwd_s = torch.cuda.max_memory_allocated() / 1e9, time.time() - t

    loss = out["pose_enc"].float().pow(2).mean() + out["depth"].float().pow(2).mean()
    t1 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_s = time.time() - t1
    t2 = time.time()
    opt.step()
    torch.cuda.synchronize()
    rep = {"S": S, "mode": mode, "t0": t0, "peak_gb_fwd": fwd_gb,
           "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
           "fwd_s": fwd_s, "bwd_s": bwd_s, "opt_s": time.time() - t2,
           "loss": float(loss.detach())}
    del out, loss
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="experiments/results/masked_step_mem.json")
    ap.add_argument("--sweep", type=int, nargs="+", default=[16, 32, 48, 64])
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--prefix_t0", type=int, default=0,
                    help="roll the student to this frame first so the window attends to a "
                         "real deployed prefix cache -- the shape T5 actually runs")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--no_ckpt", action="store_true",
                    help="gradient checkpointing OFF -- the masked path checkpoints all 24 "
                         "global blocks, which the cache path never did (§2)")
    ap.add_argument("--modes", nargs="+",
                    default=["masked_bool", "masked_float", "full_attn_cache"])
    args = ap.parse_args()

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames
    need = args.prefix_t0 + max(args.sweep) if args.prefix_t0 else max(args.sweep)
    names = sorted(os.listdir(args.frames))[:need]
    images = load_and_preprocess_images([os.path.join(args.frames, n) for n in names],
                                        mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    H, W = images.shape[-2:]
    tpf = (H // args.patch_size) * (W // args.patch_size) + 6
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[cfg] {H}x{W}, {tpf} tok/frame, card {total:.0f} GB, "
          f"ckpt {'OFF' if args.no_ckpt else 'ON'}, AdamW included")

    res = {"meta": {**vars(args), "H": H, "W": W, "tokens_per_frame": tpf,
                    "gpu": torch.cuda.get_device_name(0), "total_gb": total}, "runs": {}}

    for mode in args.modes:
        print(f"\n=== {mode} ===")
        model, _, _ = build(args.ckpt, dev, args.image_size, args.patch_size,
                            args.max_frame_num, args.kv_cache_sliding_window, sf,
                            not args.no_ckpt)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0)   # lr=0: warm-up must not
        snap = None                                           # move the weights
        if args.prefix_t0:
            t_r = time.time()
            build_prefix(model, images, args.prefix_t0, sf, args.K, dtype, dev)
            full, spec = cache_tokens(model, tpf)
            snap = snapshot_state_cpu(model)
            print(f"  prefix -> t0={args.prefix_t0} in {time.time()-t_r:.0f}s: "
                  f"{full}+{spec} = {full+spec} KV tokens ({(full+spec)/tpf:.1f} frame-eq)")
            res["meta"]["prefix_kv_tokens"] = full + spec
        try:
            step(model, images, min(args.sweep), sf, dtype, dev, opt, mode, args.K,
                 t0=args.prefix_t0, snap=snap)
        except torch.cuda.OutOfMemoryError:
            print("  warm-up OOM")
        rows = []
        print(f"  {'S':>4} {'peak fwd':>10} {'peak GB':>9} {'fwd s':>7} {'bwd s':>7}")
        for S in args.sweep:
            try:
                r = step(model, images, S, sf, dtype, dev, opt, mode, args.K,
                         t0=args.prefix_t0, snap=snap)
                rows.append(r)
                print(f"  {S:>4} {r['peak_gb_fwd']:>10.1f} {r['peak_gb']:>9.1f} "
                      f"{r['fwd_s']:>7.2f} {r['bwd_s']:>7.2f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                rows.append({"S": S, "mode": mode, "oom": True})
                print(f"  {S:>4} {'OOM':>10}", flush=True)
                torch.cuda.empty_cache()
                break
        res["runs"][mode] = rows
        del model, opt, snap
        torch.cuda.empty_cache()

    ck = "OFF" if args.no_ckpt else "ON"
    print(f"\n=== summary (peak GB, checkpointing {ck}, AdamW resident) ===")
    hdr = f"  {'S':>4}" + "".join(f" {m:>18}" for m in args.modes)
    print(hdr)
    for S in args.sweep:
        line = f"  {S:>4}"
        for m in args.modes:
            r = next((x for x in res["runs"].get(m, []) if x["S"] == S), None)
            if r is None:
                cell = "-"
            elif r.get("oom"):
                cell = "OOM"
            else:
                cell = f"{r['peak_gb']:.1f}"
            line += f" {cell:>18}"
        print(line)
    for m in args.modes:
        ok = [r for r in res["runs"].get(m, []) if not r.get("oom")]
        if len(ok) >= 2:
            a, b = ok[-2], ok[-1]
            slope = (b["peak_gb"] - a["peak_gb"]) / (b["S"] - a["S"])
            print(f"  {m:<18} slope {slope:.2f} GB/frame")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
