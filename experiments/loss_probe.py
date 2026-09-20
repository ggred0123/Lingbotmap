"""T3 acceptance: is the new magnitude term actually better than the draft?

docs/phase1-plan.md §3-T3 replaces ``L_mag``'s in-window median normalisation
because it "요동한다" -- it swings with the sample count.  That is a claim about
an estimator, so measure the estimator.

The draft's own numbers (experiments/results/grad_probe.json) say something
sharper than "it swings":

    L_mag   NEAR (fresh) 0.7538   >   FAR (contaminated) 0.4101

The term is LARGER where the student is healthier.  A loss component that is
anti-correlated with the error it is supposed to measure is not noisy, it is
measuring the wrong thing -- and at NEAR it is 60% of the total loss.

So the acceptance test is not "is it smoother", it is:

  1  ORDERING     FAR > NEAR.  A term that fails this cannot supervise.
  2  STABILITY    flat across the supervised-window length n_sup.
  3  n=1          no nan.

Also settles the scale-vs-shift question the plan left ambiguous (§3-T3 says to
use StreamVGGT's ``closed_form_scale_and_shift``): it reports how large the
fitted depth SHIFT is relative to the depth itself.  If our gauge is really
Sim(3), that shift has nothing legitimate to absorb.

Teacher labels come from the T4 bank, so this doubles as the T4 end-to-end read
test.

Usage:
    python experiments/loss_probe.py --ckpt ... --frames ... \\
        --bank_near labels/kth_near --bank_far labels/kth_far
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
from lingbot_map.train.label_bank import LabelBank, stream_collect
from lingbot_map.train.losses import (
    rel_pose_loss, depth_si_loss, closed_form_scale_and_shift,
)
from phase0_density_sweep import build_model

# The l1/trunc_l1 entries are the Pi3 port (docs/add_loss.md §7-1).  They are on
# this list because the L2 rows already measured here are the case AGAINST the
# estimator they replace -- depth FAR/NEAR falls 2.36 (median) -> 2.02
# (closed_form_scale_log) -> 1.02 (closed_form_scale), i.e. fitting the gauge by
# least squares costs most of the discrimination the term exists to provide.
# Whether the robust fit gets it back is the open number in §7-1.
MAG_MODES = ["median", "closed_form_scale", "log", "l1", "trunc_l1"]
DEPTH_MODES = ["median", "median_linear", "closed_form_scale_log",
               "closed_form_scale", "closed_form_scale_shift",
               "l1", "l1_linear", "trunc_l1", "trunc_l1_linear"]


@torch.no_grad()
def student_window(model, images, t0, S, sf, K, dtype, dev):
    """Deployed rollout from frame 0 at interval K, then collect [t0, t0+S)."""
    model.clean_kv_cache()
    with torch.amp.autocast("cuda", dtype=dtype):
        model.forward(images[:, :sf].to(dev), num_frame_for_scale=sf,
                      num_frame_per_block=sf, causal_inference=True)
    return stream_collect(model, images, sf, t0 + S, sf, K, dtype, dev, sf,
                          keys=("pose_enc", "depth"), collect_from=t0)


def sweep_mag(stu_pose, tea_pose, sweep):
    """L_mag under each estimator, as a function of the window length."""
    rows = []
    for n in sweep:
        if n > stu_pose.shape[0]:
            continue
        row = {"n_sup": n}
        for m in MAG_MODES:
            try:
                _, _, L = rel_pose_loss(stu_pose[:n], tea_pose[:n], mag_mode=m)
                row[m] = float(L)
            except ValueError:
                row[m] = None          # below MIN_FRAMES_REL_POSE
        rows.append(row)
    return rows


def depth_shift_magnitude(stu_depth, tea_depth, conf=None):
    """How much does a fitted SHIFT actually move the depth?

    Reported relative to the frame's mean depth.  Under a pure Sim(3) gauge the
    honest answer is ~0; anything large means the shift is absorbing error that
    the loss should have been charging the student for.
    """
    s = stu_depth[..., 0] if stu_depth.dim() == 4 else stu_depth
    t = tea_depth[..., 0] if tea_depth.dim() == 4 else tea_depth
    S = s.shape[0]
    s = s.reshape(S, -1).clamp(min=1e-3)
    t = t.reshape(S, -1).clamp(min=1e-3)
    w = conf.reshape(S, -1) if conf is not None else None
    rel = []
    for i in range(S):
        a, b = closed_form_scale_and_shift(s[i], t[i], None if w is None else w[i])
        rel.append(float(b.abs() / t[i].mean().clamp(min=1e-6)))
    return rel


def probe(model, images, bank, label, t0, S, sf, K, dtype, dev, sweep):
    print(f"\n{'=' * 78}\n{label}   t0={t0} S={S} K={K}\n{'=' * 78}")

    # locate a bank window covering [t0, t0+S) -- inside ONE run (gauge)
    hit = None
    for rid, r in enumerate(bank.runs):
        if r["t0"] <= t0 and t0 + S <= r["t0"] + r["L"]:
            hit = (rid, t0 - r["t0"])
            break
    if hit is None:
        raise SystemExit(f"no single bank run covers [{t0}, {t0 + S}); "
                         f"runs: {[(r['t0'], r['L']) for r in bank.runs]}")
    rid, off = hit
    lab = bank.get(rid, off, S)
    r = bank.runs[rid]
    eff_B = r["burn_in"] + off
    print(f"  teacher: bank run {rid} (t0={r['t0']} L={r['L']}), offset {off} "
          f"-> effective burn-in {eff_B} kf, cache {r['scale_frames'] + eff_B + S} "
          f"/ 320 budget")

    cache = os.path.join("experiments/results",
                         f"_stu_t{t0}_S{S}_K{K}.pt")
    if os.path.exists(cache):
        stu = torch.load(cache)
        print(f"  student: reusing cached rollout {cache}")
    else:
        t = time.time()
        stu = student_window(model, images, t0, S, sf, K, dtype, dev)
        os.makedirs("experiments/results", exist_ok=True)
        torch.save(stu, cache)
        print(f"  student: rolled 0 -> {t0 + S} at K={K} in {time.time() - t:.0f}s")

    stu_pose, tea_pose = stu["pose_enc"], lab["pose_enc"]
    stu_depth, tea_depth = stu["depth"], lab["depth"]
    conf = lab.get("depth_conf")

    # Conditioning diagnostic.  If the ordering failure is caused by near-zero
    # step magnitudes rather than by contamination, no change of estimator fixes
    # it -- the term would simply not be measurable at that operating point.
    from lingbot_map.train.losses import quat_to_R
    Rs, Rt = quat_to_R(stu_pose[:, 3:7]), quat_to_R(tea_pose[:, 3:7])
    ds = torch.einsum("nij,nj->ni", Rs[:-1].transpose(1, 2),
                      stu_pose[1:, :3] - stu_pose[:-1, :3])
    dt = torch.einsum("nij,nj->ni", Rt[:-1].transpose(1, 2),
                      tea_pose[1:, :3] - tea_pose[:-1, :3])
    ns, nt_ = ds.norm(dim=-1), dt.norm(dim=-1)
    stats = {}
    for nm, v in (("student", ns), ("teacher", nt_)):
        stats[nm] = {"min": float(v.min()), "median": float(v.median()),
                     "mean": float(v.mean()), "max": float(v.max()),
                     "min_over_median": float(v.min() / v.median().clamp(min=1e-12))}
    print(f"\n  step magnitudes (per-frame translation, previous-camera frame)")
    for nm in ("student", "teacher"):
        d_ = stats[nm]
        print(f"    {nm:<8} min {d_['min']:.4e}  median {d_['median']:.4e}  "
              f"max {d_['max']:.4e}   min/median {d_['min_over_median']:.3f}")

    mag = sweep_mag(stu_pose, tea_pose, sweep)
    print(f"\n  L_mag vs supervised-window length")
    print(f"    {'n_sup':>6}" + "".join(f" {m:>20}" for m in MAG_MODES))
    for row in mag:
        line = f"    {row['n_sup']:>6}"
        for m in MAG_MODES:
            line += f" {('nan/guard' if row[m] is None else f'{row[m]:.4f}'):>20}"
        print(line)

    dep = {}
    for m in DEPTH_MODES:
        dep[m] = float(depth_si_loss(stu_depth, tea_depth, conf, mode=m))
    shift = depth_shift_magnitude(stu_depth, tea_depth, conf)
    print(f"\n  L_depth  (gauge estimator x residual)")
    for m in DEPTH_MODES:
        print(f"    {m:<26} {dep[m]:.4f}")
    print(f"  fitted depth SHIFT / mean depth: median {sorted(shift)[len(shift)//2]:.4f}  "
          f"max {max(shift):.4f}   (Sim(3) has no shift DOF -- large = absorbing error)")

    L_rot, L_dir, _ = rel_pose_loss(stu_pose, tea_pose)
    print(f"  L_rot {float(L_rot) * 57.29578:.4f} deg   L_dir {float(L_dir):.4f}")

    return {"label": label, "t0": t0, "S": S, "K": K, "run_id": rid,
            "offset": off, "effective_burn_in": eff_B, "step_stats": stats,
            "mag_sweep": mag, "depth": dep, "shift_rel": shift,
            "L_rot_deg": float(L_rot) * 57.29578, "L_dir": float(L_dir)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--bank_near", required=True)
    ap.add_argument("--bank_far", required=True)
    ap.add_argument("--out", default="experiments/results/loss_probe.json")
    ap.add_argument("--t0_near", type=int, default=80)
    ap.add_argument("--t0_far", type=int, default=5248)
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=28)
    ap.add_argument("--sweep", type=int, nargs="+",
                    default=[1, 2, 4, 5, 8, 16, 24, 32, 48])
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    args = ap.parse_args()

    dev, dtype, sf = torch.device("cuda"), torch.bfloat16, args.num_scale_frames
    need = max(args.t0_near, args.t0_far) + args.S
    names = sorted(os.listdir(args.frames))[:need]
    images = load_and_preprocess_images([os.path.join(args.frames, n) for n in names],
                                        mode="crop", image_size=args.image_size,
                                        patch_size=args.patch_size).unsqueeze(0)
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window, sf)

    res = {"meta": {**vars(args)}, "points": []}
    for label, bank_dir, t0 in (("NEAR (fresh)", args.bank_near, args.t0_near),
                                ("FAR (contaminated)", args.bank_far, args.t0_far)):
        bank = LabelBank(bank_dir)
        res["points"].append(probe(model, images, bank, label, t0, args.S, sf,
                                   args.K, dtype, dev, args.sweep))

    near, far = res["points"][0], res["points"][1]
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(f"  {'estimator':<22} {'NEAR':>10} {'FAR':>10} {'FAR/NEAR':>10} "
          f"{'spread over n_sup':>19}")
    verdict = {}
    for m in MAG_MODES:
        n_v = {r["n_sup"]: r[m] for r in near["mag_sweep"] if r[m] is not None}
        f_v = {r["n_sup"]: r[m] for r in far["mag_sweep"] if r[m] is not None}
        full = max(n_v)
        vals = [v for v in f_v.values()]
        spread = (max(vals) - min(vals)) / max(sum(vals) / len(vals), 1e-9)
        ratio = f_v[full] / max(n_v[full], 1e-9)
        ok = ratio > 1.0
        verdict[m] = {"near": n_v[full], "far": f_v[full], "ratio": ratio,
                      "spread": spread, "ordering_ok": bool(ok)}
        print(f"  {m:<22} {n_v[full]:>10.4f} {f_v[full]:>10.4f} {ratio:>10.2f} "
              f"{spread:>18.1%}  {'ok' if ok else 'ORDERING FAILS'}")
    res["verdict"] = verdict
    print(f"\n  {'depth estimator':<26} {'NEAR':>10} {'FAR':>10} {'FAR/NEAR':>10}")
    dv = {}
    for m in DEPTH_MODES:
        ratio = far["depth"][m] / max(near["depth"][m], 1e-12)
        dv[m] = {"near": near["depth"][m], "far": far["depth"][m], "ratio": ratio,
                 "ordering_ok": bool(ratio > 1.0)}
        print(f"  {m:<26} {near['depth'][m]:>10.4f} {far['depth'][m]:>10.4f} "
              f"{ratio:>10.2f}  {'ok' if ratio > 1.0 else 'ORDERING FAILS'}")
    res["depth_verdict"] = dv
    print(f"\n  depth shift / mean depth: NEAR median "
          f"{sorted(near['shift_rel'])[len(near['shift_rel'])//2]:.4f}, FAR median "
          f"{sorted(far['shift_rel'])[len(far['shift_rel'])//2]:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
