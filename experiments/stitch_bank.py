"""Chain an overlapping pose-only teacher bank into ONE trajectory in run 0's gauge.

docs/long_supervision_design.md section 6, docs/long_supervision_plan.md Stage 1.

    L=96 / stride=48  ->  adjacent runs share exactly 48 frames
                      ->  robust Sim(3) on the shared camera centres
                      ->  compose into run 0's gauge
                      ->  each run's FIRST 48 frames own their raw frames
                      ->  one non-overlapping pose track, Delta=319 reachable

★ THIS IS NOT experiments/stitch_runs.py.  That one reads pose_enc[:, :3] only,
prints window errors and writes nothing.  L_long needs rotations too, so the
seam has to transport the full pose, and the result has to land on disk.

★ POSE CONVENTION -- the one thing here that fails silently.
``pose_enc[:3]`` is the camera CENTRE in world coordinates and ``pose_enc[3:7]``
the cam->world rotation as XYZW.  utils/pose_enc.py's docstring says the
extrinsic is "camera from world"; the codebase does not use it that way --
experiments/mcd_eval.py compares pose_enc[:, :3] straight against
Twc[:, :3, 3] and gets a sane ATE, and losses._relative's gauge invariance only
holds under this reading.  Fitting a Sim(3) to the EXTRINSIC translation would
be fitting a per-frame, rotation-dependent map: not a similarity transform at
all, and the fit residual would be large rather than the ~4 mm observed.

    C_f  <-  s R_seam C_f + t_seam
    R_f  <-  R_seam R_f            <- transporting the centres alone is a bug

Usage:
    python experiments/stitch_bank.py labels/kth_day_10_L96s48 \
        --out labels/kth_day_10_long
    python experiments/stitch_bank.py labels/... --check      # validate only
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.utils.rotation import mat_to_quat, quat_to_mat      # noqa: E402
from mcd_eval import umeyama                                          # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

#: a seam needs at least this many shared frames to fit a Sim(3) worth having
MIN_OVERLAP = 8


def load_bank(root, drop_short_tail=True):
    """-> (runs, index).  runs[i] = dict(t0, L, frames [L], pose [L, 9]).

    ★ THE TAIL RUNS CANNOT BE BRIDGED, SO THEY ARE DROPPED.  ``plan_runs`` fills
    the end of a sequence with whatever is left, so the last runs come out short
    -- L=96, 96, ..., 54, 6 on a 950-frame scene.  A run of 6 frames shares only
    6 with its predecessor, below the 8 a Sim(3) fit needs, so the chain breaks
    there and every scene failed on exactly one seam.

    Dropping them costs the last ~50 frames of the sequence (0.5% of a 9600-frame
    scene, 5% of a 950-frame one), and those frames are nearly unusable as long
    anchors anyway: no supervised window sits far enough past them.

    ★ ONLY FROM THE TAIL.  A gap in the MIDDLE is a real discontinuity, not a
    tiling artifact, and silently deleting past it would hide a break the long
    pairs must not cross.  Those still fail the gate, correctly.
    """
    idx = json.load(open(os.path.join(root, "index.json")))
    runs = []
    for r in idx["runs"]:
        z = np.load(os.path.join(root, r["file"]))
        p = z["pose_enc"].astype(np.float64)
        runs.append({"t0": int(r["t0"]), "L": int(p.shape[0]),
                     "frames": np.arange(r["t0"], r["t0"] + p.shape[0]),
                     "pose": p})
    runs.sort(key=lambda r: r["t0"])
    dropped = 0
    while drop_short_tail and len(runs) > 1:
        prev, last = runs[-2], runs[-1]
        if len(np.intersect1d(prev["frames"], last["frames"])) >= MIN_OVERLAP:
            break
        runs.pop()
        dropped += 1
    idx = {**idx, "_dropped_tail_runs": dropped}
    return runs, idx


def quat_of(R):
    return mat_to_quat(torch.as_tensor(R)).numpy()


def mat_of(q):
    return quat_to_mat(torch.as_tensor(q)).numpy()


def apply_sim3(pose, s, R, t):
    """C -> s R C + t,  R_wc -> R R_wc.  Returns a new [N, 9]."""
    out = pose.copy()
    out[:, :3] = (s * (R @ pose[:, :3].T)).T + t
    out[:, 3:7] = quat_of(R[None] @ mat_of(pose[:, 3:7]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Seam
# ─────────────────────────────────────────────────────────────────────────────

def fit_seam(src, dst, trim=0.2, iters=2):
    """Robust Sim(3) mapping ``src`` centres onto ``dst`` centres.

    Plain least squares hands the seam to whichever overlap frame the teacher
    was least sure about, and a seam is composed 60 times -- so one bad fit does
    not stay local.  Trim the worst ``trim`` fraction and refit.

    ``cond`` is the ratio of the largest to the smallest singular value of the
    centred source points.  An overlap that is nearly a straight line has almost
    no lateral extent, so the rotation about the direction of travel is
    unconstrained and the fit can be numerically fine while being geometrically
    meaningless.  Recorded rather than gated: gating needs the GT audit.
    """
    keep = np.ones(len(src), bool)
    s = R = t = None
    for _ in range(max(1, iters)):
        s, R, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        if trim <= 0:
            break
        thr = np.quantile(res[keep], 1 - trim)
        nk = res <= max(thr, 1e-12)
        if nk.sum() < 8:
            break
        keep = nk
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    sv = np.linalg.svd(src - src.mean(0), compute_uv=False)
    return {"s": float(s), "R": R.tolist(), "t": t.tolist(),
            "resid_median": float(np.median(res)),
            "resid_max": float(res.max()),
            "resid_inlier_median": float(np.median(res[keep])),
            "n_overlap": int(len(src)), "n_inlier": int(keep.sum()),
            "cond": float(sv[0] / max(sv[-1], 1e-12)),
            "sv": sv.tolist()}


def stitch(runs, trim=0.2):
    """Put every run in run 0's gauge.

    -> (stitched poses, seam records, G) where ``G[c]`` is the CUMULATIVE Sim(3)
    taking run c's own gauge into run 0's.

    ★ G IS CUMULATIVE, NOT INCREMENTAL.  Each fit is against the previous run
    ALREADY IN RUN 0'S GAUGE, so ``G[c]`` is the whole chain, not one seam --
    composing the G's would apply the chain twice.  The incremental seam is
    ``G[c-1]^-1 G[c]``.  Getting this backwards makes an audit report nonsense
    (rot errors of 178 deg after ten seams) while the track itself is fine; it
    cost one full audit run here.
    """
    glob = [runs[0]["pose"].copy()]
    G = [(1.0, np.eye(3), np.zeros(3))]
    seams = []
    for c in range(1, len(runs)):
        prev, cur = runs[c - 1], runs[c]
        shared = np.intersect1d(prev["frames"], cur["frames"])
        if len(shared) < 8:
            seams.append({"pair": [c - 1, c], "n_overlap": int(len(shared)),
                          "ok": False, "why": "overlap < 8 frames"})
            # carry the previous gauge forward unchanged rather than guessing
            G.append(G[-1])
            glob.append(apply_sim3(cur["pose"], *G[-1]))
            continue
        si = np.searchsorted(cur["frames"], shared)
        di = np.searchsorted(prev["frames"], shared)
        f = fit_seam(cur["pose"][si, :3], glob[c - 1][di, :3], trim=trim)
        f["pair"], f["ok"] = [c - 1, c], True
        f["frames"] = [int(shared[0]), int(shared[-1])]
        seams.append(f)
        s_c, R_c, t_c = f["s"], np.array(f["R"]), np.array(f["t"])
        G.append((s_c, R_c, t_c))
        glob.append(apply_sim3(cur["pose"], s_c, R_c, t_c))
        # rotation residual on the overlap, in degrees -- the half of the seam
        # the translation fit never sees
        Ra = mat_of(glob[c][si, 3:7])
        Rb = mat_of(glob[c - 1][di, 3:7])
        M = np.einsum("nij,njk->nik", Ra.transpose(0, 2, 1), Rb)
        cos = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
        f["rot_resid_deg_median"] = float(np.median(np.degrees(np.arccos(cos))))
    return glob, seams, G


def canonical(runs, glob, own=48):
    """Each run owns its FIRST ``own`` frames.  -> (frames [M], pose [M, 9]).

    The first ``own`` frames of a run are the shallowest part of its teacher
    cache, and taking exactly the stride means the blocks tile the sequence with
    no gap and no duplicate.  The tail of each run exists only to carry the seam.
    """
    fr, po = [], []
    for c, r in enumerate(runs):
        n = min(own, r["L"]) if c == len(runs) - 1 else own
        n = min(n, r["L"])
        fr.append(r["frames"][:n])
        po.append(glob[c][:n])
    return np.concatenate(fr), np.concatenate(po)


# ─────────────────────────────────────────────────────────────────────────────
# Validation -- docs/long_supervision_plan.md gate G1
# ─────────────────────────────────────────────────────────────────────────────

def rel_targets(pose, frames, delta, S=48):
    """(rot, dir, log-ratio) for the pairs the LOSS forms, window by window.

    ★ THE NORMALISER MUST BE THE ONE THE LOSS USES.  An earlier version divided
    by the RMS step of the WHOLE track.  On a long scene the chained scale drifts
    (kth_night_01: 0.024 after 200 seams), so that global figure is meaningless
    and differs between two stitches of the same geometry -- which made the
    invariance gate fail on tracks whose loss-relevant targets agree to 1e-6.
    ``long_loss.v_local`` normalises by the CURRENT 48-frame window; so does this.
    """
    pos = {int(f): i for i, f in enumerate(frames)}
    lo, hi = int(frames[0]), int(frames[-1])
    dR_l, dir_l, r_l = [], [], []
    for t in range(lo, hi - S + 2, S):
        if t + S - 1 not in pos:
            break
        w = [pos[t + k] for k in range(S)]
        v = np.sqrt((np.linalg.norm(np.diff(pose[w, :3], axis=0), axis=1) ** 2).mean())
        ii = [pos[t + k - delta] for k in range(S) if t + k - delta in pos]
        jj = [pos[t + k] for k in range(S) if t + k - delta in pos]
        if not ii:
            continue
        ii, jj = np.array(ii), np.array(jj)
        R = mat_of(pose[:, 3:7])
        d = np.einsum("nij,nj->ni", R[ii].transpose(0, 2, 1), pose[jj, :3] - pose[ii, :3])
        n = np.linalg.norm(d, axis=1)
        dR_l.append(np.einsum("nij,njk->nik", R[ii].transpose(0, 2, 1), R[jj]))
        dir_l.append(d / np.maximum(n[:, None], 1e-12))
        r_l.append(np.log(np.maximum(n, 1e-12) / max(v, 1e-12)))
    if not r_l:
        return None
    return np.concatenate(dR_l), np.concatenate(dir_l), np.concatenate(r_l)


def validate(runs, glob, seams, frames, pose, stride, deltas=(48, 96, 192, 319)):
    out, ok = {}, True

    d = np.diff(frames)
    out["canonical_contiguous"] = bool((d == 1).all())
    out["canonical_unique"] = bool(len(np.unique(frames)) == len(frames))
    out["canonical_range"] = [int(frames[0]), int(frames[-1])]
    out["canonical_n"] = int(len(frames))
    ok &= out["canonical_contiguous"] and out["canonical_unique"]

    good = [s for s in seams if s.get("ok")]
    out["n_seams"] = len(seams)
    out["n_seams_ok"] = len(good)
    if good:
        out["seam_resid_median_m"] = float(np.median([s["resid_median"] for s in good]))
        out["seam_resid_worst_m"] = float(max(s["resid_max"] for s in good))
        out["seam_rot_resid_deg_median"] = float(np.median(
            [s.get("rot_resid_deg_median", np.nan) for s in good]))
        out["seam_scale_spread"] = [float(min(s["s"] for s in good)),
                                    float(max(s["s"] for s in good))]
        out["seam_cond_worst"] = float(max(s["cond"] for s in good))
        out["seam_overlap_exact"] = bool(all(s["n_overlap"] == stride for s in good))
    ok &= (len(good) == len(seams))

    out["pairs"] = {int(dd): (0 if rel_targets(pose, frames, dd) is None
                              else int(len(rel_targets(pose, frames, dd)[2])))
                    for dd in deltas}

    # ★ THE END-TO-END GAUGE TEST.  Hit every raw run with its OWN random Sim(3)
    # -- the exact freedom stitching claims to remove -- re-stitch, and require
    # the relative targets to come out identical.  This catches a dropped
    # rotation transport, a composition in the wrong order, and a translation
    # fitted to the extrinsic instead of the centre, none of which the residual
    # numbers above would flag.
    rng = np.random.default_rng(0)
    pert = []
    for r in runs:
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        pert.append({**r, "pose": apply_sim3(r["pose"], float(rng.uniform(0.4, 2.5)),
                                             Q, rng.normal(size=3) * 5)})
    g2, _, _ = stitch(pert)
    f2, p2 = canonical(pert, g2)
    worst = 0.0
    for dd in deltas:
        a, b = rel_targets(pose, frames, dd), rel_targets(p2, f2, dd)
        if a is None or b is None:
            continue
        M = np.einsum("nij,njk->nik", a[0].transpose(0, 2, 1), b[0])
        cos = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
        worst = max(worst, float(np.degrees(np.arccos(cos)).max()),
                    float(np.abs(a[2] - b[2]).max()) * 57.29578,
                    float(np.degrees(np.arccos(np.clip((a[1] * b[1]).sum(-1), -1, 1))).max()))
    out["sim3_invariance_worst_deg_or_scaled"] = worst
    out["sim3_invariant"] = bool(worst < 0.5)
    ok &= out["sim3_invariant"]

    # ★ A SEPARATE, REAL GATE ON SEAM QUALITY.  The invariance test above asks
    # "is the chain a function of the geometry"; it does NOT ask "is the geometry
    # any good".  paralleldomain4d passes nothing here: 18 seams take its
    # cumulative scale to 0.0014, i.e. every seam shrinks the world by ~30%, and
    # a Delta=319 pair spans seven of them.  That is 12x of pure fiction in the
    # target.  The per-seam log-scale is the number to gate on, and it separates
    # cleanly -- 0.03 on MCD against 0.36 there.
    if good:
        inc = [float(np.log(s2["s"] / s1["s"])) for s1, s2 in zip(good, good[1:])]
        inc = [x for x in inc if np.isfinite(x)]
        if inc:
            out["seam_log_scale_per_seam_median"] = float(np.median(np.abs(inc)))
            out["seam_log_scale_span7"] = float(np.median(np.abs(inc)) * 7)
            out["seam_scale_ok"] = bool(out["seam_log_scale_per_seam_median"] < 0.10)
            # ★ REPORTED, NOT GATED -- because it does not rank the way GT does.
            # Checked against the GT seam audit on five MCD scenes: this figure
            # calls kth_night_04 (GT m=7 |log s| 0.449, the worst) no worse than
            # kth_night_01 (0.097, among the best), and it fails paralleldomain4d
            # while a normalised fit residual passes it.  The two proxies rank
            # the corpora in OPPOSITE orders and neither matches GT.  A gate that
            # deletes artifacts on a metric known not to track the truth is worse
            # than no gate: the numbers go in the index and the decision is made
            # where GT exists.  See docs/long_supervision_result.md.
            #   ok &= out["seam_scale_ok"]   <- deliberately not enforced

    out["PASS"] = bool(ok)
    return out


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank", help="pose-only L96s48 bank root")
    ap.add_argument("--out", default=None, help="write artifacts here")
    ap.add_argument("--own", type=int, default=None,
                    help="canonical frames per run (default: the bake stride)")
    ap.add_argument("--trim", type=float, default=0.2)
    ap.add_argument("--check", action="store_true", help="validate, write nothing")
    a = ap.parse_args()

    runs, idx = load_bank(a.bank)
    stride = int(idx.get("stride") or np.median(np.diff([r["t0"] for r in runs])))
    own = a.own or stride
    print(f"[bank] {os.path.basename(a.bank)}  {len(runs)} runs  L={runs[0]['L']} "
          f"stride={stride}  overlap={runs[0]['L'] - stride}  "
          f"pose_only={idx.get('pose_only')}  "
          f"tail_dropped={idx.get('_dropped_tail_runs', 0)}")

    glob, seams, _ = stitch(runs, trim=a.trim)
    frames, pose = canonical(runs, glob, own=own)
    rep = validate(runs, glob, seams, frames, pose, stride)

    print(f"[seam] {rep['n_seams_ok']}/{rep['n_seams']} fitted  "
          f"median {rep.get('seam_resid_median_m', float('nan')) * 1000:.2f} mm  "
          f"worst {rep.get('seam_resid_worst_m', float('nan')) * 1000:.2f} mm  "
          f"rot {rep.get('seam_rot_resid_deg_median', float('nan')):.4f} deg")
    print(f"[seam] scale spread {rep.get('seam_scale_spread')}  "
          f"worst cond {rep.get('seam_cond_worst', float('nan')):.1f}")
    print(f"[canon] {rep['canonical_n']} frames {rep['canonical_range']}  "
          f"contiguous={rep['canonical_contiguous']} unique={rep['canonical_unique']}")
    print(f"[pairs] {rep['pairs']}")
    print(f"[gauge] independent Sim(3) per run -> worst target drift "
          f"{rep['sim3_invariance_worst_deg_or_scaled']:.2e}  "
          f"invariant={rep['sim3_invariant']}")
    if "seam_log_scale_per_seam_median" in rep:
        print(f"[qual]  per-seam |log s| median "
              f"{rep['seam_log_scale_per_seam_median']:.4f}  "
              f"(x7 spans = {rep['seam_log_scale_span7']:.3f})  "
              f"ok={rep['seam_scale_ok']}")
    print(f"[G1] {'PASS' if rep['PASS'] else 'FAIL'}")

    if a.check or not a.out:
        return 0 if rep["PASS"] else 1

    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "canonical.npz"),
             frames=frames.astype(np.int64), pose_enc=pose.astype(np.float32))
    # ★ ALSO WRITE IT AS A ONE-RUN POSE-ONLY BANK.  The canonical track is
    # contiguous and lives in ONE gauge, which is exactly what a bank run is --
    # so emitting it in bank format lets the trainer reach Delta=319 through the
    # same LabelBank.poses / long_pair_index path the in-run rungs already use,
    # instead of a second indexing scheme that would have to be verified
    # separately.  It also makes the K_t=2 candidate and this one
    # interchangeable at the call site: both are "a pose-only bank".
    np.savez(os.path.join(a.out, f"run_00000_t{int(frames[0])}.npz"),
             pose_enc=pose.astype(np.float32))
    np.savez(os.path.join(a.out, "stitched_full.npz"),
             run_id=np.concatenate([np.full(r["L"], c) for c, r in enumerate(runs)]),
             frames=np.concatenate([r["frames"] for r in runs]),
             pose_enc=np.concatenate(glob).astype(np.float32))
    with open(os.path.join(a.out, "seams.json"), "w") as f:
        json.dump(seams, f, indent=1)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump({"kind": "long_pose_track",
                   "pose_only": True,
                   "source_bank": os.path.abspath(a.bank),
                   "source_L": runs[0]["L"], "stride": stride, "own": own,
                   "n_source_runs": len(runs), "scene": idx.get("scene"),
                   "ckpt_sha256": idx.get("ckpt_sha256"),
                   "teacher_interval": idx.get("teacher_interval"),
                   "burn_in": idx.get("burn_in"),
                   # LabelBank-compatible: ONE run, because the stitched track is
                   # one contiguous stretch in one gauge.
                   "runs": [{"file": f"run_00000_t{int(frames[0])}.npz",
                             "t0": int(frames[0]), "L": int(len(frames)),
                             "burn_in": idx.get("burn_in"),
                             "scale_frames": idx.get("scale_frames"),
                             "teacher_interval": idx.get("teacher_interval"),
                             "shape_depth": [], "has_depth": False,
                             "has_conf": False}],
                   "report": rep}, f, indent=1)
    print(f"[write] {a.out}  canonical {pose.nbytes / 1e6:.2f} MB")
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
