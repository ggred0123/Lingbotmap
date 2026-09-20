"""One real supervised step: T1 masked window + T4 bank labels + T3 loss.

This is the integration test for gate 5, and the seed of T5.  Everything that
came before is exercised together:

    T1/T2-e   the whole window forward through the masked parallel path
    Level 0   prefix rolled with no grad and detached (grad_probe.detach_caches)
    T4        teacher labels read from the offline bank
    T3        the §3.3 loss, backward

and it checks the two properties a trainer depends on:

  * gradient reaches every parameter group, including the cached cross-frame path
  * the stream state is unchanged afterwards, so the next step can start from the
    same restored snapshot

Usage:
    python experiments/train_step_check.py --ckpt ... --frames ... \\
        --bank labels/kth_near --t0 80 --S 48
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
from lingbot_map.train.label_bank import LabelBank
from lingbot_map.train.losses import SelfDistillLoss
from phase0_density_sweep import build_model
from grad_probe import detach_caches
from mask_equiv import state_fingerprint, state_checksums

GROUPS = [("encoder(patch_embed)", "aggregator.patch_embed"),
          ("aggregator.frame_blocks", "aggregator.frame_blocks"),
          ("aggregator.global_blocks", "aggregator.global_blocks"),
          ("camera_head", "camera_head."),
          ("depth_head", "depth_head.")]


def roll_prefix(model, images, t0, sf, K, dtype, dev):
    """Level 0: deployed rollout to t0, no grad, detached every step."""
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                      num_frame_per_block=sf, causal_inference=True)
    detach_caches(model)
    for i in range(sf, t0):
        is_kf = (K <= 1) or ((i - sf) % K == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(images[:, i:i + 1].to(dev), num_frame_for_scale=sf,
                          num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        detach_caches(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", default="experiments/results/train_step_check.json")
    ap.add_argument("--t0", type=int, default=80)
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--steps", type=int, default=2,
                    help="repeat the step from the same snapshot; the second one "
                         "proves the state survived the first")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.0,
                    help="0 keeps the weights fixed so repeated steps are comparable")
    args = ap.parse_args()

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames
    S, t0 = args.S, args.t0

    bank = LabelBank(args.bank)
    hit = next(((rid, t0 - r["t0"]) for rid, r in enumerate(bank.runs)
                if r["t0"] <= t0 and t0 + S <= r["t0"] + r["L"]), None)
    if hit is None:
        raise SystemExit(f"no single bank run covers [{t0}, {t0 + S})")
    lab = bank.get(*hit, S, device=dev)
    print(f"[labels] bank run {hit[0]} offset {hit[1]}  pose{list(lab['pose_enc'].shape)} "
          f"depth{list(lab['depth'].shape)} conf={'depth_conf' in lab}")

    names = sorted(os.listdir(args.frames))[:t0 + S]
    images = load_and_preprocess_images([os.path.join(args.frames, n) for n in names],
                                        mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window, sf)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = SelfDistillLoss()
    print(f"[loss] mag_mode={crit.mag_mode}  depth_mode={crit.depth_mode}  "
          f"lam={crit.lam}")

    t = time.time()
    roll_prefix(model, images, t0, sf, args.K, dtype, dev)
    print(f"[prefix] rolled 0 -> {t0} at K={args.K} in {time.time() - t:.0f}s "
          f"(detached every step)")
    before = (state_fingerprint(model), state_checksums(model))

    res = {"meta": {**vars(args)}, "steps": []}
    for step in range(args.steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        with model.masked_window(window_start=t0, keyframe_interval=args.K):
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model.forward(images[:, t0:t0 + S].to(dev),
                                    num_frame_for_scale=sf, num_frame_per_block=S,
                                    causal_inference=False)
        fwd = time.time() - t

        total, parts = crit(out["pose_enc"][0].float(), lab["pose_enc"],
                            out["depth"][0].float(), lab["depth"],
                            lab.get("depth_conf"))
        t1 = time.time()
        total.backward()
        torch.cuda.synchronize()
        bwd = time.time() - t1
        opt.step()

        gn = {}
        for label, pre in GROUPS:
            sq = sum(float(p.grad.float().pow(2).sum())
                     for n, p in model.named_parameters()
                     if n.startswith(pre) and p.grad is not None)
            gn[label] = sq ** 0.5
        peak = torch.cuda.max_memory_allocated() / 1e9
        rec = {"step": step, **parts, "peak_gb": peak, "fwd_s": fwd, "bwd_s": bwd,
               "grad_norm": gn, "groups_without_grad": [k for k, v in gn.items() if v == 0.0]}
        res["steps"].append(rec)
        print(f"\n[step {step}] loss {parts['loss']:.4f} = "
              f"rot {parts['L_rot_deg']:.3f}deg + dir {parts['L_dir']:.4f} + "
              f"mag {parts['L_mag']:.4f} + depth {parts['L_depth_si']:.4f}")
        print(f"          peak {peak:.1f} GB   fwd {fwd:.2f}s   bwd {bwd:.2f}s")
        for k, v in gn.items():
            print(f"          grad {k:<26} {v:.4e}")
        del out, total

    after = (state_fingerprint(model), state_checksums(model))
    changed = [k for k in set(before[0]) | set(after[0]) if before[0].get(k) != after[0].get(k)]
    changed += [f"{k} (content)" for k in set(before[1]) | set(after[1])
                if before[1].get(k) != after[1].get(k)]
    res["state_unchanged"] = not changed
    res["state_changed"] = changed

    a, b = res["steps"][0], res["steps"][-1]
    same_loss = abs(a["loss"] - b["loss"]) < 1e-6 if args.lr == 0 else None
    print(f"\n{'=' * 78}")
    print(f"  stream state after {args.steps} steps: "
          f"{'UNCHANGED' if not changed else f'{len(changed)} entries moved'}")
    if args.lr == 0:
        print(f"  loss step0 {a['loss']:.6f} vs step{args.steps-1} {b['loss']:.6f}  "
              f"-> {'identical (state truly reusable)' if same_loss else 'DRIFTED'}")
    bad = res["steps"][0]["groups_without_grad"]
    print(f"  gradient: {'reaches every group' if not bad else f'MISSING for {bad}'}")
    ok = (not changed) and (not bad) and (same_loss is not False)
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[saved] {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
