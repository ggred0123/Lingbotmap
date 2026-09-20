"""Phase 1 gate: can this model be trained at all through the streaming cache?

docs/self-distill-ver4-fixed.md §3.7 proposes a "gradient ladder" whose Level 0
is "roll the cache forward with no grad, re-run the supervised frames with
grad".  The doc treats that as free, citing v3 §9.2's claim that "the KV cache
is detached under truncated BPTT".  The released code does no such thing:

    attention.py:645   kv_cache[f"k_{i}"] = torch.cat((kv_cache[f"k_{i}"], k_new), dim=2)
    attention.py:657   k_cached = kv_cache[f"k_{i}"].clone()

There is no detach anywhere on that path, so with grad enabled every frame
retains a fresh copy of the whole cache in the autograd graph -- O(N^2) memory
in rollout length.  Level 0 is therefore a code change, not a config.

This script answers the three questions that gate Phase 1, in order:

  T1  How bad is the no-detach blowup, really?          -> memory curve vs frames
  T2  Does a detached-cache backward run at all?        -> Level 0 mechanics
  T3  Does gradient actually reach the *read* policy?   -> Level 0 premise

T3 is the one that matters.  Level 0 only buys anything if "how the current
frame reads a contaminated cache" is learnable, so we make the cache tensors
autograd leaves and check whether they receive gradient -- and compare a
contaminated state (long prefix) against a fresh one (short prefix).

The loss is the real §3.3 pair (gauge-invariant relative pose + scale-invariant
depth) against a fresh-dense teacher, not a dummy, so the reported numbers are
the actual starting loss and the actual gradient scale.

Usage:
    python experiments/grad_probe.py \
        --ckpt /path/to/lingbot-map.pt \
        --frames data/kth_day_06/frames_10hz \
        --t0 5248 --n_sup 4 --out experiments/results/grad_probe.json
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
from fork_rig import run_anchor, step_frames, snapshot_state, restore_state
from phase0_density_sweep import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Cache plumbing -- the Level 0 boundary the released code is missing
# ─────────────────────────────────────────────────────────────────────────────

def cache_dicts(model):
    """Every live streaming-state dict: aggregator's, plus one per camera iter."""
    out = []
    agg = getattr(model, "aggregator", None)
    if isinstance(getattr(agg, "kv_cache", None), dict):
        out.append(agg.kv_cache)
    ch = getattr(model, "camera_head", None)
    if ch is not None and getattr(ch, "kv_cache", None) is not None:
        out.extend(ch.kv_cache)
    return out


def detach_caches(model, requires_grad=False):
    """Cut the autograd graph at the cache -- this is Level 0.

    With requires_grad=True the cache tensors become autograd *leaves*, so after
    backward their .grad is exactly the read-path signal (T3).

    Returns (n_tensors, n_bytes, leaves) where ``leaves`` holds direct
    references to the leaf tensors.  Holding them matters: on a keyframe the
    attention code does ``kv_cache[k] = torch.cat((kv_cache[k], k_new))``, which
    REPLACES the dict entry with a non-leaf.  Reading .grad off the dict after
    the backward would then silently report zero on exactly the frames that
    write to the cache.
    """
    n_t, n_b, leaves = 0, 0, []
    agg_cache = getattr(getattr(model, "aggregator", None), "kv_cache", None)
    for d in cache_dicts(model):
        stream = "agg" if d is agg_cache else "cam"
        for k, v in list(d.items()):
            if torch.is_tensor(v):
                t = v.detach()
                if requires_grad and t.is_floating_point():
                    t.requires_grad_(True)
                    leaves.append((stream, k, t))
                d[k] = t
                n_t += 1
                n_b += t.numel() * t.element_size()
    return n_t, n_b, leaves


def snapshot_state_cpu(model):
    """Host-resident copy of the whole streaming state.

    fork_rig.snapshot_state clones on-device, which is fine for inference but
    not here: at a far t0 the cache is ~11 GB and the supervised step already
    peaks near the card limit, so an on-device clone OOMs by itself.
    """
    agg = model.aggregator
    ch = model.camera_head
    return {
        "agg": {k: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v)
                for k, v in agg.kv_cache.items()} if isinstance(agg.kv_cache, dict) else None,
        "agg_frames": agg.total_frames_processed,
        "pos3d": (agg._cached_pos3d.detach().to("cpu", copy=True)
                  if torch.is_tensor(getattr(agg, "_cached_pos3d", None)) else None),
        "cam": ([{k: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v)
                  for k, v in d.items()} for d in ch.kv_cache]
                if getattr(ch, "kv_cache", None) is not None else None),
        "cam_frame_idx": getattr(ch, "frame_idx", 0),
    }


def restore_state_cpu(model, snap, dev):
    agg = model.aggregator
    ch = model.camera_head
    if snap["agg"] is not None:
        agg.kv_cache = {k: (v.to(dev, copy=True) if torch.is_tensor(v) else v)
                        for k, v in snap["agg"].items()}
    agg.total_frames_processed = snap["agg_frames"]
    agg._cached_pos3d = snap["pos3d"].to(dev, copy=True) if snap["pos3d"] is not None else None
    if snap["cam"] is not None:
        ch.kv_cache = [{k: (v.to(dev, copy=True) if torch.is_tensor(v) else v)
                        for k, v in d.items()} for d in snap["cam"]]
    ch.frame_idx = snap["cam_frame_idx"]


def cache_grad_stats(leaves):
    """L2 norm of gradient landing on the cache leaves, split by stream."""
    acc = {"agg": [0.0, 0], "cam": [0.0, 0]}
    none_n = 0
    for stream, _k, t in leaves:
        if t.grad is None:
            none_n += 1
            continue
        acc[stream][0] += float(t.grad.detach().float().pow(2).sum())
        acc[stream][1] += 1
    return {
        "agg_cache_grad_norm": acc["agg"][0] ** 0.5, "agg_cache_tensors_with_grad": acc["agg"][1],
        "cam_cache_grad_norm": acc["cam"][0] ** 0.5, "cam_cache_tensors_with_grad": acc["cam"][1],
        "cache_leaves_total": len(leaves), "cache_leaves_without_grad": none_n,
    }


PARAM_GROUPS = [
    ("encoder(DINOv2 patch_embed)", "aggregator.patch_embed"),
    ("aggregator.frame_blocks", "aggregator.frame_blocks"),
    ("aggregator.global_blocks", "aggregator.global_blocks"),   # <- the cached, cross-frame path
    ("camera_head", "camera_head."),
    ("depth_head", "depth_head."),
]


def param_grad_report(model):
    rows = []
    for label, prefix in PARAM_GROUPS:
        sq, n_p, n_g = 0.0, 0, 0
        for name, p in model.named_parameters():
            if not name.startswith(prefix):
                continue
            n_p += p.numel()
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
                n_g += p.numel()
        rows.append({
            "group": label, "params_M": n_p / 1e6,
            "params_with_grad_M": n_g / 1e6, "grad_norm": sq ** 0.5,
        })
    return rows


def mem_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# §3.3 loss -- two terms, both gauge-invariant, both unlabeled
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_R(q):
    """[N,4] XYZW (scalar-last) -> [N,3,3]."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def rel_pose_loss(stu_pose, tea_pose):
    """Relative rotation + relative translation (direction & normalized magnitude).

    Every term is a within-window quantity, so a Sim(3) gauge difference between
    the student's run and the teacher's run cancels -- no alignment, no bridge.
    Translation magnitude is normalized by the window's own median step, which
    is exactly the blind spot §3.3-3 admits to.
    """
    Rs, Rt = quat_to_R(stu_pose[:, 3:7]), quat_to_R(tea_pose[:, 3:7])
    ts, tt = stu_pose[:, :3], tea_pose[:, :3]

    # relative rotation between consecutive frames
    dRs = torch.einsum("nij,njk->nik", Rs[:-1].transpose(1, 2), Rs[1:])
    dRt = torch.einsum("nij,njk->nik", Rt[:-1].transpose(1, 2), Rt[1:])
    dR = torch.einsum("nij,njk->nik", dRs.transpose(1, 2), dRt)
    cos = ((dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]) - 1) / 2
    L_rot = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()

    # relative translation expressed in the previous camera frame
    ds = torch.einsum("nij,nj->ni", Rs[:-1].transpose(1, 2), ts[1:] - ts[:-1])
    dt = torch.einsum("nij,nj->ni", Rt[:-1].transpose(1, 2), tt[1:] - tt[:-1])
    ns, nt = ds.norm(dim=-1).clamp(min=1e-8), dt.norm(dim=-1).clamp(min=1e-8)
    L_dir = (1 - torch.nn.functional.cosine_similarity(ds, dt, dim=-1)).mean()
    # NOTE: with <4 pairs the median IS the sample, so this term is identically
    # zero.  It only carries signal once the window holds enough steps -- one
    # more reason the supervised window cannot be tiny.
    L_mag = ((ns / ns.median().clamp(min=1e-8)) - (nt / nt.median().clamp(min=1e-8))).abs().mean()
    return L_rot, L_dir, L_mag


def depth_si_loss(stu_depth, tea_depth, conf=None):
    """Per-frame median-normalized log-depth L1, optionally teacher-conf weighted."""
    s = stu_depth[..., 0].flatten(1).clamp(min=1e-3)
    t = tea_depth[..., 0].flatten(1).clamp(min=1e-3)
    s = s / s.median(dim=1, keepdim=True).values.clamp(min=1e-6)
    t = t / t.median(dim=1, keepdim=True).values.clamp(min=1e-6)
    err = (s.log() - t.log()).abs()
    if conf is not None:
        w = conf.flatten(1).detach()
        return (err * w).sum() / w.sum().clamp(min=1e-6)
    return err.mean()


# ─────────────────────────────────────────────────────────────────────────────
# T1: how fast does the un-detached cache blow up?
# ─────────────────────────────────────────────────────────────────────────────

def t1_no_detach_blowup(model, images, sf, dev, dtype, cap_gb, max_frames):
    print("\n" + "=" * 78)
    print("T1  no-detach blowup -- grad on, cache left connected (current code)")
    print("=" * 78)
    model.clean_kv_cache()
    reset_mem()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                      num_frame_per_block=sf, causal_inference=True)
    detach_caches(model)

    curve, oom_at = [], None
    print(f"  {'frames':>7} {'peak GB':>9} {'cache GB':>9}")
    for i in range(max_frames):
        try:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model.forward(images[:, sf + i:sf + i + 1].to(dev),
                                    num_frame_for_scale=sf, num_frame_per_block=1,
                                    causal_inference=True)
            del out
        except torch.cuda.OutOfMemoryError:
            oom_at = i + 1
            print(f"  OOM at frame {oom_at}")
            break
        cache_gb = sum(v.numel() * v.element_size()
                       for d in cache_dicts(model) for v in d.values()
                       if torch.is_tensor(v)) / 1e9
        curve.append({"frames": i + 1, "peak_gb": mem_gb(), "cache_gb": cache_gb})
        if (i + 1) % 5 == 0 or i < 3:
            print(f"  {i+1:>7} {mem_gb():>9.1f} {cache_gb:>9.2f}")
        if mem_gb() > cap_gb:
            print(f"  stopped at {i+1} frames (peak {mem_gb():.1f} GB > cap {cap_gb} GB)")
            break

    model.clean_kv_cache()
    torch.cuda.synchronize()
    reset_mem()
    n = curve[-1]["frames"] if curve else 0
    print(f"\n  -> {n} frames of *rollout* consumed {curve[-1]['peak_gb']:.1f} GB "
          f"if it survived; the doc's target rollout is hundreds of frames.")
    return {"curve": curve, "oom_at_frame": oom_at,
            "frames_survived": n, "cap_gb": cap_gb}


# ─────────────────────────────────────────────────────────────────────────────
# T2/T3: Level 0 -- detached rollout, supervised frames with grad
# ─────────────────────────────────────────────────────────────────────────────

def rollout_detached(model, images, lo, hi, sf, interval, dtype, dev, phase_origin):
    """Advance the student state with no grad, detaching at every step."""
    for i in range(lo, hi):
        is_kf = (interval <= 1) or ((i - phase_origin) % interval == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        del out
        detach_caches(model)


def level0_step(model, images, t0, n_sup, sf, K, dtype, dev, teacher, label,
                lam=(1.0, 1.0, 0.5, 1.0)):
    """One Level-0 training step at t0: forward n_sup frames with grad, backward."""
    l_rot, l_dir, l_mag, l_dep = lam
    model.zero_grad(set_to_none=True)
    n_t, n_b, leaves = detach_caches(model, requires_grad=True)
    reset_mem()
    t_start = time.time()

    poses, depths, dconfs, n_kf = [], [], [], 0
    for i in range(t0, t0 + n_sup):
        is_kf = (K <= 1) or ((i - sf) % K == 0)
        n_kf += int(is_kf)
        if not is_kf:
            model._set_skip_append(True)
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        poses.append(out["pose_enc"][0].float())
        depths.append(out["depth"][0].float())
        if "depth_conf" in out:
            dconfs.append(out["depth_conf"][0].float())
    fwd_gb, fwd_s = mem_gb(), time.time() - t_start

    stu_pose = torch.cat(poses, dim=0)
    stu_depth = torch.cat(depths, dim=0)
    tea_pose = teacher["pose"].to(dev)
    tea_depth = teacher["depth"].to(dev)
    tea_conf = teacher["conf"].to(dev) if teacher.get("conf") is not None else None

    L_rot, L_dir, L_mag = rel_pose_loss(stu_pose, tea_pose)
    L_dep = depth_si_loss(stu_depth, tea_depth, tea_conf)
    loss = l_rot * L_rot + l_dir * L_dir + l_mag * L_mag + l_dep * L_dep

    t_bwd = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_s = time.time() - t_bwd

    rep = {
        "label": label, "t0": t0, "n_sup": n_sup, "n_keyframes_supervised": n_kf,
        "loss_total": float(loss), "L_rot_rad": float(L_rot), "L_rot_deg": float(L_rot) * 57.29578,
        "L_dir": float(L_dir), "L_mag": float(L_mag), "L_depth_si": float(L_dep),
        "cache_tensors": n_t, "cache_bytes_gb": n_b / 1e9,
        "peak_gb_fwd": fwd_gb, "peak_gb_total": mem_gb(),
        "fwd_s": fwd_s, "bwd_s": bwd_s,
        "params": param_grad_report(model),
    }
    rep.update(cache_grad_stats(leaves))
    return rep


def print_step(rep):
    print(f"\n  --- {rep['label']}  (t0={rep['t0']}, {rep['n_sup']} supervised frames) ---")
    print(f"  loss {rep['loss_total']:.4f}  = rot {rep['L_rot_deg']:.3f}deg"
          f" + dir {rep['L_dir']:.4f} + mag {rep['L_mag']:.4f}"
          f" + depth-SI {rep['L_depth_si']:.4f}")
    print(f"  memory: peak {rep['peak_gb_total']:.1f} GB "
          f"(cache leaves {rep['cache_bytes_gb']:.2f} GB in {rep['cache_tensors']} tensors)")
    print(f"  time:   fwd {rep['fwd_s']:.2f}s  bwd {rep['bwd_s']:.2f}s")
    print(f"  supervised keyframes: {rep['n_keyframes_supervised']}/{rep['n_sup']}"
          f"   (cache leaves {rep['cache_leaves_total']}, "
          f"{rep['cache_leaves_without_grad']} received no grad)")
    print(f"  READ-PATH gradient on cache leaves:"
          f"  aggregator {rep['agg_cache_grad_norm']:.4e} ({rep['agg_cache_tensors_with_grad']} tensors)"
          f"  camera {rep['cam_cache_grad_norm']:.4e} ({rep['cam_cache_tensors_with_grad']})")
    print(f"  {'param group':<30} {'params M':>9} {'w/ grad M':>10} {'grad norm':>12}")
    for r in rep["params"]:
        print(f"  {r['group']:<30} {r['params_M']:>9.1f} {r['params_with_grad_M']:>10.1f} "
              f"{r['grad_norm']:>12.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="experiments/results/grad_probe.json")
    ap.add_argument("--t0", type=int, default=5248, help="contaminated probe point")
    ap.add_argument("--t0_near", type=int, default=80, help="fresh control point")
    ap.add_argument("--n_sup", type=int, default=4, help="supervised frames with grad")
    ap.add_argument("--K", type=int, default=28, help="student keyframe interval")
    ap.add_argument("--B", type=int, default=72, help="teacher burn-in frames")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--t1_cap_gb", type=float, default=120.0)
    ap.add_argument("--t1_max_frames", type=int, default=80)
    ap.add_argument("--skip_t1", action="store_true")
    ap.add_argument("--sweep", type=int, nargs="*", default=[1, 2, 4, 8, 16],
                    help="T4: supervised-window lengths to measure peak memory at")
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    sf = args.num_scale_frames

    names = sorted(os.listdir(args.frames))
    need = max(args.t0 + args.n_sup, args.t0_near + args.n_sup) + 4
    paths = [os.path.join(args.frames, n) for n in names[:need]]
    print(f"[probe] loading {len(paths)} frames")
    images = load_and_preprocess_images(paths, mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    H, W = images.shape[-2:]
    tpf = (H // args.patch_size) * (W // args.patch_size) + 6
    print(f"[probe] {H}x{W}, {tpf} tokens/frame")
    if args.n_sup < 5:
        print(f"[warn] n_sup={args.n_sup} gives {args.n_sup - 1} relative pairs; "
              f"the median-normalized magnitude term degenerates to 0 below ~4 pairs.")

    print("[probe] building model")
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window, sf)

    res = {"meta": {**vars(args), "H": H, "W": W, "tokens_per_frame": tpf,
                    "gpu": torch.cuda.get_device_name(0),
                    "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9}}

    # ── T1 ────────────────────────────────────────────────────────────────────
    if not args.skip_t1:
        res["T1_no_detach"] = t1_no_detach_blowup(
            model, images, sf, dev, dtype, args.t1_cap_gb, args.t1_max_frames)

    # ── T2/T3 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("T2/T3  Level 0 -- detached rollout + supervised backward")
    print("=" * 78)

    # model must be in train() for dropout/checkpoint paths to be exercised
    # honestly, but keep eval() first so the T1 numbers match the measured runs.
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    steps = []
    for label, t0 in (("FAR (contaminated)", args.t0), ("NEAR (fresh control)", args.t0_near)):
        # teacher: fresh dense run over [t0-B-sf, t0+n_sup)
        a0 = t0 - args.B - sf
        print(f"\n[teacher] {label}: fresh anchor at {a0}, dense to {t0 + args.n_sup}")
        run_anchor(model, images[:, a0:a0 + sf], sf, dtype, dev)
        with torch.no_grad():
            t_pose, t_depth = step_frames(model, images, a0 + sf, t0 + args.n_sup,
                                          sf, 1, dtype, dev, a0 + sf)
        teacher = {"pose": t_pose[-args.n_sup:], "depth": t_depth[-args.n_sup:], "conf": None}

        # student: real deployed rollout from frame 0 at K, detached every step
        print(f"[student] {label}: LS rollout 0 -> {t0} at K={args.K} (no grad, detached)")
        run_anchor(model, images[:, :sf], sf, dtype, dev)
        t_roll = time.time()
        rollout_detached(model, images, sf, t0, sf, args.K, dtype, dev, sf)
        print(f"[student] rollout done in {time.time() - t_roll:.0f}s")

        # T4: memory slope in the supervised-window length, at this cache depth.
        # §3.7 claims Level 0 is "메모리 O(1프레임)".  It is not: every supervised
        # frame attends over the WHOLE cache and SDPA saves k/v for backward, so
        # the cost is O(n_sup x cache_length).
        #
        # The snapshot has to live on the HOST.  fork_rig.snapshot_state clones
        # on-device, and at the far point the cache is ~11 GB on top of a step
        # that already peaks at ~183 GB -- that alone OOMs the card.
        snap = snapshot_state_cpu(model)
        sweep = sorted(set(list(args.sweep) + [args.n_sup]))
        print(f"  [T4] memory vs supervised-window length at {label}: {sweep}")
        sw, rep = [], None
        for n in sweep:
            restore_state_cpu(model, snap, dev)
            model.zero_grad(set_to_none=True)
            reset_mem()
            try:
                r = level0_step(model, images, t0, n, sf, args.K, dtype, dev,
                                {"pose": teacher["pose"][:n], "depth": teacher["depth"][:n],
                                 "conf": None},
                                label if n == args.n_sup else f"{label} n_sup={n}")
                sw.append({"n_sup": n, "peak_gb": r["peak_gb_total"], "oom": False,
                           "fwd_s": r["fwd_s"], "bwd_s": r["bwd_s"],
                           "loss_total": r["loss_total"],
                           "agg_cache_grad_norm": r["agg_cache_grad_norm"]})
                print(f"       n_sup={n:>3}  peak {r['peak_gb_total']:>6.1f} GB  "
                      f"fwd {r['fwd_s']:.2f}s  bwd {r['bwd_s']:.2f}s  "
                      f"loss {r['loss_total']:.4f}")
                if n == args.n_sup:
                    rep = r
            except torch.cuda.OutOfMemoryError:
                sw.append({"n_sup": n, "peak_gb": None, "oom": True})
                print(f"       n_sup={n:>3}  OOM")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if rep is None:      # the headline n_sup OOM'd; fall back to the largest that ran
            ok = [s for s in sw if not s["oom"]]
            print(f"  [warn] n_sup={args.n_sup} OOM'd; "
                  f"reporting largest surviving window n_sup={ok[-1]['n_sup'] if ok else None}")
            restore_state_cpu(model, snap, dev)
            rep = level0_step(model, images, t0, ok[-1]["n_sup"], sf, args.K, dtype, dev,
                              {"pose": teacher["pose"][:ok[-1]["n_sup"]],
                               "depth": teacher["depth"][:ok[-1]["n_sup"]], "conf": None}, label)
        rep["sweep_n_sup"] = sw
        print_step(rep)
        steps.append(rep)

    res["T2_T3_level0"] = steps

    # ── verdict ───────────────────────────────────────────────────────────────
    far, near = steps[0], steps[1]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    gb = far["peak_gb_total"]
    print(f"  T2  Level 0 backward RUNS.  peak {gb:.1f} GB for {args.n_sup} supervised "
          f"frames on top of a {args.t0}-frame rollout.")
    print(f"      -> with 1.16B params, optimizer states add ~19 GB; "
          f"headroom on {res['meta']['total_gb']:.0f} GB is "
          f"{res['meta']['total_gb'] - gb - 19:.0f} GB.")
    gnorm = far["agg_cache_grad_norm"]
    if gnorm > 0:
        print(f"  T3  Read path is LIVE: gradient reaches the cache leaves "
              f"({gnorm:.3e}).  Level 0's premise holds.")
    else:
        print(f"  T3  Read path is DEAD: no gradient on the cache leaves. "
              f"Level 0 cannot learn a read policy -- Level 1 is mandatory.")
    gl = {r["group"]: r["grad_norm"] for r in far["params"]}
    print(f"      global_blocks (cached cross-frame path) grad norm "
          f"{gl.get('aggregator.global_blocks', 0):.3e}  vs  "
          f"encoder {gl.get('encoder(DINOv2 patch_embed)', 0):.3e}")
    print(f"  contamination check: loss FAR {far['loss_total']:.4f} "
          f"vs NEAR {near['loss_total']:.4f} "
          f"(ratio {far['loss_total'] / max(near['loss_total'], 1e-9):.2f}x)")
    print(f"      -> the gap the distillation is supposed to close, in loss units.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
