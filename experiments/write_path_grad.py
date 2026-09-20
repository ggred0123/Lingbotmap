"""Is the cache WRITE path already being trained inside the masked window?

v4 §3.7 called it a ladder: Level 0 trains only how a frame READS a frozen
cache; Level 1 -- training how a frame WRITES into the cache, so a supervised
frame's loss reaches an earlier frame's contribution -- was a further step
needing its own plumbing (a keyframe-chain TBPTT).

The masked parallel path may have collapsed that distinction for free.  In
``_forward_masked_parallel`` the keys are ``[prefix specials | prefix cache | k]``
and ``k`` is the window's own K/V, still attached to the graph.  The mask makes
a later window frame read an earlier window frame's K/V.  If that is true, the
window is full BPTT over the keyframe chain and the only detach boundary left is
the prefix.

That is a claim about gradient flow, so measure it rather than read it off the
source.  Four probes, run on the real weights:

  A  write path      loss on the LAST window frame -> gradient on EARLIER frames' pixels?
  B  causality       loss on the FIRST window frame -> gradient on LATER frames must be 0
  C  read path       gradient on the prefix cache leaves (Level 0's premise, §1.3)
  D  keyframe chain  with K=8, only window frame 0 persists: gradient must reach
                     frame 0 and must NOT reach the non-keyframes in between

D is the decisive one.  A alone could be explained by the anchor frames, which
everyone attends to.  D says the live path is exactly the set of frames that
persist -- i.e. the cache write.

Usage:
    python experiments/write_path_grad.py --ckpt ... --frames data/kth_day_06/frames_10hz
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.aggregator import gca_mask
from phase0_density_sweep import build_model
from grad_probe import detach_caches


def roll_prefix(model, images, t0, sf, K, dtype, dev):
    """Deployed rollout to t0, detached every step -- the Level-0 boundary."""
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


def per_frame_grad(model, images, t0, S, sf, K, dtype, dev, loss_frame,
                   cache_leaves=False):
    """Backward from ONE window frame's loss; return per-frame pixel grad norms."""
    win = images[:, t0:t0 + S].to(dev).clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    leaves = []
    if cache_leaves:
        _, _, leaves = detach_caches(model, requires_grad=True)

    with model.masked_window(window_start=t0, keyframe_interval=K):
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(win, num_frame_for_scale=sf,
                                num_frame_per_block=S, causal_inference=False)
        plan = model.aggregator.last_window_plan

    loss = (out["pose_enc"][:, loss_frame].float().pow(2).sum()
            + out["depth"][:, loss_frame].float().pow(2).mean())
    loss.backward()

    g = win.grad.detach()[0]                      # [S, 3, H, W]
    norms = [float(g[j].float().norm()) for j in range(S)]
    cache_norm = 0.0
    for _stream, _k, t in leaves:
        if t.grad is not None:
            cache_norm += float(t.grad.detach().float().pow(2).sum())
    return norms, cache_norm ** 0.5, plan


def role_str(plan):
    names = {gca_mask.ROLE_SCALE: "anchor", gca_mask.ROLE_KEYFRAME: "KEY",
             gca_mask.ROLE_NONKEY: "non-kf"}
    return [names[int(r)] for r in plan.roles]


def show(title, norms, roles, loss_frame, note=""):
    print(f"\n  {title}")
    print(f"    {'frame':>5} {'role':>7} {'grad norm':>12}   {note}")
    for j, (n, r) in enumerate(zip(norms, roles)):
        tag = "  <- loss here" if j == loss_frame else ""
        z = "  ZERO" if n == 0.0 else ""
        print(f"    {j:>5} {r:>7} {n:>12.4e}{z}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="experiments/results/write_path_grad.json")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--t0", type=int, default=24)
    args = ap.parse_args()

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames
    S, t0 = args.S, args.t0
    assert (t0 - sf) % S == 0, "t0 must sit on a K=S keyframe for probe D"

    names = sorted(os.listdir(args.frames))[:t0 + S]
    images = load_and_preprocess_images([os.path.join(args.frames, n) for n in names],
                                        mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window, sf)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    res = {"meta": {**vars(args)}, "probes": {}}

    print("=" * 78)
    print(f"write-path gradient inside the masked window   t0={t0} S={S} sf={sf}")
    print("=" * 78)

    # ---- A + C : every window frame is a keyframe -------------------------
    roll_prefix(model, images, t0, sf, 1, dtype, dev)
    norms, cache_g, plan = per_frame_grad(model, images, t0, S, sf, 1, dtype, dev,
                                          loss_frame=S - 1, cache_leaves=True)
    roles = role_str(plan)
    show("A  K=1, loss on the LAST frame -- does it reach earlier frames?",
         norms, roles, S - 1)
    earlier = norms[:-1]
    a_ok = all(n > 0 for n in earlier)
    print(f"    -> earlier frames {'ALL nonzero' if a_ok else 'SOME ZERO'}; "
          f"weakest {min(earlier):.3e}, own-frame {norms[-1]:.3e} "
          f"(ratio {min(earlier) / norms[-1]:.3f})")
    print(f"    C  read path: prefix cache leaf grad norm {cache_g:.4e} "
          f"({'live' if cache_g > 0 else 'DEAD'})")
    res["probes"]["A_write_path_K1"] = {"per_frame": norms, "roles": roles,
                                        "pass": bool(a_ok)}
    res["probes"]["C_read_path"] = {"cache_grad_norm": cache_g,
                                    "pass": bool(cache_g > 0)}

    # ---- B : causality ----------------------------------------------------
    roll_prefix(model, images, t0, sf, 1, dtype, dev)
    norms_b, _, plan_b = per_frame_grad(model, images, t0, S, sf, 1, dtype, dev,
                                        loss_frame=0)
    show("B  K=1, loss on the FIRST frame -- later frames must be untouched",
         norms_b, role_str(plan_b), 0)
    later = norms_b[1:]
    b_ok = all(n == 0.0 for n in later)
    print(f"    -> later frames {'ALL zero (causal)' if b_ok else 'LEAKED'}; "
          f"max {max(later):.3e}")
    res["probes"]["B_causality"] = {"per_frame": norms_b, "pass": bool(b_ok)}

    # ---- D : the write path IS the keyframe chain -------------------------
    roll_prefix(model, images, t0, sf, S, dtype, dev)
    norms_d, _, plan_d = per_frame_grad(model, images, t0, S, sf, S, dtype, dev,
                                        loss_frame=S - 1)
    roles_d = role_str(plan_d)
    show(f"D  K={S}: only window frame 0 persists -- gradient must follow the chain",
         norms_d, roles_d, S - 1)
    kf_live = norms_d[0] > 0
    nonkf_dead = all(n == 0.0 for j, n in enumerate(norms_d)
                     if roles_d[j] == "non-kf" and j != S - 1)
    print(f"    -> keyframe (frame 0) {'LIVE' if kf_live else 'DEAD'} "
          f"{norms_d[0]:.3e};  intervening non-keyframes "
          f"{'all zero' if nonkf_dead else 'LEAKED'}")
    res["probes"]["D_keyframe_chain"] = {"per_frame": norms_d, "roles": roles_d,
                                         "pass": bool(kf_live and nonkf_dead)}

    print("\n" + "=" * 78)
    ok = all(p["pass"] for p in res["probes"].values())
    for k, p in res["probes"].items():
        print(f"  {'PASS' if p['pass'] else 'FAIL'}  {k}")
    print("VERDICT: the masked window trains the cache WRITE path (v4 §3.7 'Level 1'). "
          if ok else "VERDICT: inconclusive -- see failures above.")
    if ok:
        print("  The detach boundary is the prefix and nothing else; inside the window")
        print("  the keyframe chain is full BPTT over S frames.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[saved] {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
