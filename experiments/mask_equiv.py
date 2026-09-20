"""T1 acceptance: does the masked parallel path equal the cache path?

docs/phase1-plan.md §3-T2.  Runs the same frames through both aggregator paths
and diffs the outputs, turning on one GCA rule at a time:

    a  K=1, S=16, sw=64   blockwise causal + anchor          (no eviction)
    b  K=1, S=20, sw=8    sliding window + eviction          (4 frames evicted)
    c  K=1, S=24, sw=8    evicted special-token memory       (8 frames evicted)
    d  K=4, S=20, sw=8    non-keyframes never persist
    e  camera head        its own 4x4 causal trunk (pose_enc)

Lowering ``kv_cache_sliding_window`` to 8 is what makes b-d affordable: with the
deployed 64 the first eviction needs S >= 73.  The rule is the same rule.

★ What is compared.  ``aggregated_tokens`` (all 4 exported blocks),
``depth``/``depth_conf``, and ``pose_enc``.

``pose_enc`` is the camera head's output and only became comparable with T2-e:
the head is a second, independent causal stack (4 refinement iterations x 4
CameraBlocks over one token per frame) with its own KV cache, and it now has its
own masked path.  Its rules are the aggregator's minus eviction and minus
special tokens -- see ``gca_mask.plan_camera_window``.  A pose diff therefore
tests the head; a token/depth diff tests the aggregator.  Both run in one step,
so a regression in either shows up here.

Usage:
    python experiments/mask_equiv.py --ckpt .../lingbot-map.pt \
        --frames data/kth_day_06/frames_10hz
    python experiments/mask_equiv.py ... --prefix_t0 200 --K 28   # with a real prefix
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
from lingbot_map.aggregator import gca_mask
from phase0_density_sweep import build_model
from grad_probe import detach_caches, snapshot_state_cpu, restore_state_cpu


class TokenTap:
    """Capture the aggregator's own output, before either head touches it."""

    def __init__(self, model):
        self.out = None
        self.h = model.aggregator.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.out = [t.detach().float().cpu() for t in out[0]]

    def close(self):
        self.h.remove()


def set_sliding_window(model, W):
    """Retune eviction without rebuilding 1.16B parameters.

    ``kv_cache_sliding_window`` is a plain attribute on every SDPAAttention plus
    the aggregator; nothing derived from it is precomputed.
    """
    model.aggregator.kv_cache_sliding_window = W
    for blk in model.aggregator.global_blocks:
        blk.attn.kv_cache_sliding_window = W


def run_sequential(model, images, lo, hi, sf, K, dtype, dev, tap):
    """Reference: the deployed path, one frame at a time through the KV cache."""
    tokens, depth, dconf, pose = [], [], [], []

    def take(out):
        tokens.append(tap.out)
        depth.append(out["depth"].detach().float().cpu())
        dconf.append(out["depth_conf"].detach().float().cpu())
        pose.append(out["pose_enc"].detach().float().cpu())

    if lo == 0:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=sf, causal_inference=True)
        take(out)
        del out
        start = sf
    else:
        start = lo

    for i in range(start, hi):
        is_kf = (K <= 1) or ((i - sf) % K == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        take(out)
        del out

    n_blocks = len(tokens[0])
    cat_tokens = [torch.cat([t[b] for t in tokens], dim=1) for b in range(n_blocks)]
    return (cat_tokens, torch.cat(depth, dim=1), torch.cat(dconf, dim=1),
            torch.cat(pose, dim=1))


def run_masked(model, images, lo, S, sf, K, dtype, dev, tap, mask_dtype):
    """Candidate: one parallel forward over the window, GCA rules as a mask."""
    with model.masked_window(window_start=lo, keyframe_interval=K, mask_dtype=mask_dtype):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, lo:lo + S].to(dev), num_frame_for_scale=sf,
                                num_frame_per_block=S, causal_inference=False)
        plan = model.aggregator.last_window_plan
    return (tap.out, out["depth"].detach().float().cpu(),
            out["depth_conf"].detach().float().cpu(),
            out["pose_enc"].detach().float().cpu(), plan)


def diff(a, b):
    d = (a - b).abs()
    rel = d / b.abs().clamp(min=1e-6)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "median_rel": float(rel.median()), "max_rel": float(rel.max())}


def stage(model, images, name, S, K, W, sf, dtype, dev, args, prefix_snap=None, t0=0):
    print(f"\n{'=' * 78}\n{name}   S={S} K={K} sliding_window={W} t0={t0}\n{'=' * 78}")
    set_sliding_window(model, W)
    tap = TokenTap(model)
    try:
        if prefix_snap is None:
            model.clean_kv_cache()
        else:
            restore_state_cpu(model, prefix_snap, dev)
        t = time.time()
        ref_tok, ref_depth, ref_conf, ref_pose = run_sequential(
            model, images, t0, t0 + S, sf, K, dtype, dev, tap)
        t_seq = time.time() - t

        if prefix_snap is None:
            model.clean_kv_cache()
        else:
            restore_state_cpu(model, prefix_snap, dev)
        t = time.time()
        got_tok, got_depth, got_conf, got_pose, plan = run_masked(
            model, images, t0, S, sf, K, dtype, dev, tap, args.mask_dtype)
        t_par = time.time() - t
    finally:
        tap.close()

    print(f"  {gca_mask.describe(plan)}")
    n_ev = max(0, int(plan.n_kf_at_query[-1]) - plan.sliding_window) - plan.prefix_evicted
    if n_ev <= 0 and "evict" in name.lower():
        print("  [WARN] this stage claims to test eviction but evicted nothing")

    rep = {"stage": name, "S": S, "K": K, "sliding_window": W, "t0": t0,
           "evictions_in_window": int(n_ev), "seq_s": t_seq, "parallel_s": t_par,
           "speedup": t_seq / max(t_par, 1e-9)}
    print(f"  sequential {t_seq:.2f}s   parallel {t_par:.2f}s   "
          f"({t_seq / max(t_par, 1e-9):.1f}x)")

    worst, worst_rel = 0.0, 0.0
    for b, (a_, b_) in enumerate(zip(ref_tok, got_tok)):
        d = diff(b_, a_)
        rep[f"tokens_block{b}"] = d
        worst = max(worst, d["max_abs"])
        worst_rel = max(worst_rel, d["median_rel"])
        print(f"  tokens[block {b}]  max abs {d['max_abs']:.3e}  "
              f"mean {d['mean_abs']:.3e}  median rel {d['median_rel']:.3e}")
    d = diff(got_depth, ref_depth)
    rep["depth"] = d
    print(f"  depth             max abs {d['max_abs']:.3e}  "
          f"median rel {d['median_rel']:.3e}")
    d = diff(got_conf, ref_conf)
    rep["depth_conf"] = d
    print(f"  depth_conf        max abs {d['max_abs']:.3e}  "
          f"median rel {d['median_rel']:.3e}")
    d = diff(got_pose, ref_pose)
    rep["pose_enc"] = d
    cam_plan = getattr(model.camera_head, "last_window_plan", None)
    print(f"  pose_enc  (T2-e)  max abs {d['max_abs']:.3e}  "
          f"mean {d['mean_abs']:.3e}"
          + (f"   [camera prefix {cam_plan.prefix_scale_frames}+"
             f"{cam_plan.prefix_full_keyframes} frames]" if cam_plan else ""))

    # Tokens are gated too: they are the aggregator's ACTUAL output, and both
    # depth and pose are downstream of them.  Gating only the derived quantities
    # lets a token-level regression through whenever the heads happen to smooth
    # it out.  Gate on median relative error -- token magnitudes vary by orders
    # of magnitude across blocks, so max abs is not a scale-free criterion.
    ok_tok = worst_rel < args.tol_tokens_rel
    ok_depth = rep["depth"]["median_rel"] < args.tol_depth_rel
    ok_pose = rep["pose_enc"]["max_abs"] < args.tol_pose_abs
    rep["tokens_max_abs"] = worst
    rep["tokens_median_rel"] = worst_rel
    rep["pass_tokens"] = bool(ok_tok)
    rep["pass_depth"] = bool(ok_depth)
    rep["pass_pose"] = bool(ok_pose)
    rep["pass"] = bool(ok_tok and ok_depth and ok_pose)
    rep["margins"] = {"tokens": args.tol_tokens_rel / max(worst_rel, 1e-30),
                      "depth": args.tol_depth_rel / max(rep["depth"]["median_rel"], 1e-30),
                      "pose": args.tol_pose_abs / max(rep["pose_enc"]["max_abs"], 1e-30)}
    print(f"  -> {'PASS' if rep['pass'] else 'FAIL'}  "
          f"tokens rel {worst_rel:.3e} {'ok' if ok_tok else 'FAIL'}   "
          f"depth rel {rep['depth']['median_rel']:.3e} {'ok' if ok_depth else 'FAIL'}   "
          f"pose abs {rep['pose_enc']['max_abs']:.3e} {'ok' if ok_pose else 'FAIL'}"
          f"   (tightest margin {min(rep['margins'].values()):.0f}x)")
    return rep


def state_fingerprint(model):
    """Every scalar the streaming state is made of, as plain Python/tensors.

    A training step must leave all of it untouched, or successive steps from one
    restored snapshot silently drift.  Before T2-e the camera head failed this:
    ``trunk_fn`` keys its cache path off ``self.kv_cache is not None`` rather than
    off ``causal_inference``, so a parallel step appended the whole window to the
    head's cache and advanced ``frame_idx``.
    """
    agg, ch = model.aggregator, model.camera_head
    fp = {"agg.total_frames_processed": agg.total_frames_processed,
          "cam.frame_idx": getattr(ch, "frame_idx", None)}
    if isinstance(agg.kv_cache, dict):
        for k, v in agg.kv_cache.items():
            fp[f"agg.{k}"] = tuple(v.shape) if torch.is_tensor(v) else v
    if getattr(ch, "kv_cache", None) is not None:
        for i, d in enumerate(ch.kv_cache):
            for k, v in d.items():
                fp[f"cam{i}.{k}"] = tuple(v.shape) if torch.is_tensor(v) else v
    return fp


def state_checksums(model):
    """Content hashes, so a same-shape overwrite is caught too."""
    agg, ch = model.aggregator, model.camera_head
    out = {}
    if isinstance(agg.kv_cache, dict):
        for k, v in agg.kv_cache.items():
            if torch.is_tensor(v):
                out[f"agg.{k}"] = float(v.detach().float().sum())
    if getattr(ch, "kv_cache", None) is not None:
        for i, d in enumerate(ch.kv_cache):
            for k, v in d.items():
                if torch.is_tensor(v):
                    out[f"cam{i}.{k}"] = float(v.detach().float().sum())
    return out


def _one_masked_step(model, images, S, sf, K, dtype, dev, t0, snap, mask_dtype, camera):
    restore_state_cpu(model, snap, dev)
    before = (state_fingerprint(model), state_checksums(model))
    with model.masked_window(window_start=t0, keyframe_interval=K,
                             mask_dtype=mask_dtype, camera_head=camera):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(images[:, t0:t0 + S].to(dev), num_frame_for_scale=sf,
                          num_frame_per_block=S, causal_inference=False)
    after = (state_fingerprint(model), state_checksums(model))

    bad = []
    for k in sorted(set(before[0]) | set(after[0])):
        if before[0].get(k) != after[0].get(k):
            bad.append(f"{k}: {before[0].get(k)} -> {after[0].get(k)}")
    for k in sorted(set(before[1]) | set(after[1])):
        if before[1].get(k) != after[1].get(k):
            bad.append(f"{k}: content changed")
    return before[0], bad


def check_state_invariance(model, images, S, sf, K, dtype, dev, t0, snap, mask_dtype):
    """A masked step must be a pure read of the stream state.

    Run twice.  ``camera_head=True`` is the real path and must change nothing.
    ``camera_head=False`` leaves the head on its old path and is the NEGATIVE
    CONTROL: it must change something, or the check has no teeth and would pass
    even if the masked path silently reverted.
    """
    print(f"\n{'=' * 78}\nstate invariance: does a masked step move the stream?"
          f"\n{'=' * 78}")
    fp, bad = _one_masked_step(model, images, S, sf, K, dtype, dev, t0, snap,
                               mask_dtype, camera=True)
    print(f"  {len(fp)} state entries + cache tensor checksums")
    for b in bad[:10]:
        print(f"  [CHANGED] {b}")
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more")
    print(f"  masked (camera_head=True):  {'PASS - untouched' if not bad else 'FAIL'}")

    _, ctrl = _one_masked_step(model, images, S, sf, K, dtype, dev, t0, snap,
                               mask_dtype, camera=False)
    cam_ctrl = [c for c in ctrl if c.startswith("cam")]
    print(f"  control (camera_head=False): {len(ctrl)} entries move "
          f"({len(cam_ctrl)} in the camera head) -- e.g. "
          f"{cam_ctrl[0] if cam_ctrl else (ctrl[0] if ctrl else 'NOTHING')}")
    has_teeth = len(cam_ctrl) > 0
    if not has_teeth:
        print("  [WARN] the control changed nothing in the camera head; this check "
              "cannot distinguish a working fix from a no-op")
    ok = (not bad) and has_teeth
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    restore_state_cpu(model, snap, dev)
    return {"entries": len(fp), "changed": bad, "control_changed": ctrl,
            "control_has_teeth": has_teeth, "pass": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="experiments/results/mask_equiv.json")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--mask_dtype", default="bool", choices=["bool", "float"])
    ap.add_argument("--tol_tokens_rel", type=float, default=None)
    ap.add_argument("--tol_depth_rel", type=float, default=None)
    ap.add_argument("--tol_pose_abs", type=float, default=None)
    ap.add_argument("--prefix_t0", type=int, default=0,
                    help="also run stage (f): a window on top of a real rolled prefix")
    ap.add_argument("--K", type=int, default=28, help="keyframe interval for the prefix roll")
    ap.add_argument("--prefix_S", type=int, default=16)
    ap.add_argument("--prefix_W", type=int, default=64,
                    help="sliding window for the prefix roll. Lower it (8) to reach a "
                         "prefix that has ALREADY evicted without rolling 1800 frames.")
    ap.add_argument("--bf16", action="store_true",
                    help="run in bf16 instead of fp32. bf16 measures deployment "
                         "numerics but CANNOT decide the gate: a bf16 gap says "
                         "nothing about whether the rules are right (§3-T2). Use it "
                         "to characterise, not to accept.")
    ap.add_argument("--fp32", action="store_true",
                    help="(accepted for compatibility; fp32 is now the default)")
    ap.add_argument("--train_step", action="store_true",
                    help="also run one backward through the masked path with "
                         "gradient checkpointing on")
    args = ap.parse_args()

    dev = torch.device("cuda")
    # fp32 is the DEFAULT.  The verdict in §3-T2 rests on the fp32 numbers -- a
    # bf16 gap is dominated by accumulation-order noise (the cache path physically
    # reorders keys after an eviction) and cannot separate "the rules are right"
    # from "the rules are wrong by a little".  Running the gate in bf16 by default
    # would mean the acceptance criterion was never actually evaluated.
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    # fp32 tolerances are the plan's stated bar: "fp32에서 1e-5 아래로 안 내려가면
    # 규칙이 틀린 것이다".  bf16 tolerances only bound deployment-numerics drift.
    tol = ({"tokens_rel": 5e-2, "depth_rel": 1e-3, "pose_abs": 1e-3} if args.bf16
           else {"tokens_rel": 1e-5, "depth_rel": 1e-5, "pose_abs": 1e-5})
    for k, v in tol.items():
        if getattr(args, f"tol_{k}") is None:
            setattr(args, f"tol_{k}", v)
    sf = args.num_scale_frames

    need = (args.prefix_t0 + args.prefix_S) if args.prefix_t0 else 0
    need = max(need, 48) + 4
    names = sorted(os.listdir(args.frames))[:need]
    images = load_and_preprocess_images([os.path.join(args.frames, n) for n in names],
                                        mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    print(f"[cfg] {images.shape[1]} frames {tuple(images.shape[-2:])}, dtype={dtype}")
    print(f"[gate] tokens median rel < {args.tol_tokens_rel:.0e}   "
          f"depth median rel < {args.tol_depth_rel:.0e}   "
          f"pose max abs < {args.tol_pose_abs:.0e}"
          + ("   [bf16: characterisation only, not an acceptance test]" if args.bf16 else ""))

    model = build_model(args.ckpt, dev, args.image_size, args.patch_size,
                        args.max_frame_num, 64, sf)
    model.eval()

    res = {"meta": {**vars(args), "gpu": torch.cuda.get_device_name(0),
                    "dtype": str(dtype)}, "stages": []}

    for name, S, K, W in [
        ("a  blockwise causal + anchor", 16, 1, 64),
        ("b  sliding window + eviction", 20, 1, 8),
        ("c  evicted special-token memory", 24, 1, 8),
        ("d  non-keyframes never persist (K=4)", 20, 4, 8),
        # (d) alone evicts nothing -- 3 keyframes never exceed W=8.  This one
        # makes the two rules interact: non-keyframes AND eviction of keyframes
        # that non-keyframes had already read.
        ("d' non-keyframes + eviction (K=4)", 48, 4, 8),
    ]:
        res["stages"].append(stage(model, images, name, S, K, W, sf, dtype, dev, args))
        torch.cuda.empty_cache()

    if args.prefix_t0:
        print(f"\n{'=' * 78}\nf  window on a real rolled prefix "
              f"(t0={args.prefix_t0}, K={args.K})\n{'=' * 78}")
        set_sliding_window(model, args.prefix_W)
        model.clean_kv_cache()
        t = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                          num_frame_per_block=sf, causal_inference=True)
        detach_caches(model)
        for i in range(sf, args.prefix_t0):
            is_kf = (args.K <= 1) or ((i - sf) % args.K == 0)
            if not is_kf:
                model._set_skip_append(True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                              num_frame_per_block=1, causal_inference=True)
            if not is_kf:
                model._set_skip_append(False)
        print(f"  rolled to {args.prefix_t0} in {time.time() - t:.0f}s")
        snap = snapshot_state_cpu(model)
        cached, evicted, _ = model.aggregator._prefix_layout()
        print(f"  prefix cache: {cached} full-token frames + {evicted} evicted "
              f"(evicted-special block is {'LIVE' if evicted else 'EMPTY'})")
        res["stages"].append(stage(model, images, "f  window on rolled prefix",
                                   args.prefix_S, args.K, args.prefix_W, sf, dtype, dev,
                                   args, prefix_snap=snap, t0=args.prefix_t0))
        res["state_invariance"] = check_state_invariance(
            model, images, args.prefix_S, sf, args.K, dtype, dev,
            args.prefix_t0, snap, args.mask_dtype)

    if args.train_step:
        print(f"\n{'=' * 78}\ntrain step: masked path, checkpointing ON, backward\n{'=' * 78}")
        set_sliding_window(model, 8)
        model.clean_kv_cache()
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        S = 24
        t = time.time()
        with model.masked_window(window_start=0, keyframe_interval=1,
                                 mask_dtype=args.mask_dtype):
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model.forward(images[:, :S].to(dev), num_frame_for_scale=sf,
                                    num_frame_per_block=S, causal_inference=False)
        loss = out["depth"].float().pow(2).mean() + out["pose_enc"].float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        gb = torch.cuda.max_memory_allocated() / 1e9
        gnorm = {}
        for label, pre in [("patch_embed", "aggregator.patch_embed"),
                           ("frame_blocks", "aggregator.frame_blocks"),
                           ("global_blocks", "aggregator.global_blocks"),
                           ("camera_head", "camera_head."),
                           ("depth_head", "depth_head.")]:
            sq = sum(float(p.grad.float().pow(2).sum())
                     for n, p in model.named_parameters()
                     if n.startswith(pre) and p.grad is not None)
            gnorm[label] = sq ** 0.5
        print(f"  S={S}  peak {gb:.1f} GB  {time.time() - t:.2f}s  loss {float(loss):.4f}")
        for k, v in gnorm.items():
            print(f"    grad norm {k:<14} {v:.4e}")
        bad = [k for k, v in gnorm.items() if v == 0.0]
        res["train_step"] = {"S": S, "peak_gb": gb, "loss": float(loss),
                             "grad_norm": gnorm, "groups_without_grad": bad}
        if bad:
            print(f"  [FAIL] no gradient reached: {bad}")
        else:
            print("  -> gradient reaches every group, including global_blocks "
                  "and the camera trunk")
        model.eval()

    passed = all(s["pass"] for s in res["stages"])
    if "state_invariance" in res:
        passed = passed and res["state_invariance"]["pass"]
    print(f"\n{'=' * 78}")
    for s in res["stages"]:
        print(f"  {'PASS' if s['pass'] else 'FAIL'}  {s['stage']:<40} "
              f"tokens rel {s['tokens_median_rel']:.3e}  "
              f"depth rel {s['depth']['median_rel']:.3e}  "
              f"pose abs {s['pose_enc']['max_abs']:.3e}  "
              f"evict {s['evictions_in_window']:>3}  "
              f"margin {min(s['margins'].values()):.0f}x")
    print(f"{'ALL STAGES PASS' if passed else 'FAILURES'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[saved] {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
