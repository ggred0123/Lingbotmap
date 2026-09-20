"""Is a long-Delta teacher target good enough to train on?  GT-based audit.

docs/long_supervision_design.md sections 7 and 9; docs/long_supervision_plan.md gate G1.

Two questions, one script.

  DECISION   Two candidates supply Delta=319: an L96/stride48 bank chained
             through seam Sim(3)s, and an L=480 K_t=2 bank where the pair is
             in-run and there is no seam at all.  Both are baked pose-only, both
             are one measurement away from being trusted, and their failure modes
             are in different places.  So measure both against GT at the SAME
             deltas and pick on the numbers.

             ★ THE CRITERION IS MAD, NOT MEDIAN BIAS.  A bank is static: the
             same label is reused every epoch, so a teacher's random drift is a
             fixed error the student can fit, not noise that averages out.
             Median bias can be calibrated away; spread cannot.  Reported side by
             side so the choice is visible.

  SEAM       For the stitched candidate only: a 4 mm overlap fit residual does
             not bound the error of SIXTY composed transforms.  Fit each run to
             GT independently, derive the true seam, and compare -- per seam and
             after composing 10 / 20 / 60 of them.

Usage:
    python experiments/seam_audit.py --gt mcd --frames data/mcd/kth_day_10/frames_10hz \
        --track labels/kth_day_10_long --bank labels/kth_day_10_L480K2 \
        --seams labels/kth_day_10_L96s48
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.train.label_bank import image_names                  # noqa: E402
from lingbot_map.utils.rotation import quat_to_mat                    # noqa: E402
from mcd_eval import gt_camera_poses, load_extrinsic, umeyama         # noqa: E402
from stitch_bank import fit_seam, load_bank, stitch                   # noqa: E402

S = 48


# ─────────────────────────────────────────────────────────────────────────────

def gt_by_disk_index(frames_dir, calib, sensor):
    """GT camera centres and quats indexed the way the BANK indexes frames.

    ★ THE TWO INDEXINGS DO NOT AGREE (trainer.make_gt_score says the same thing):
    the bank walks the frames directory, meta.npz lists only the frames that have
    GT.  Map name -> row and leave a -1 where there is none.
    """
    meta = np.load(os.path.join(frames_dir, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(calib, sensor)
    gp, gq = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    row_of = {str(n): i for i, n in enumerate(meta["names"])}
    rows = np.array([row_of.get(n, -1) for n in image_names(frames_dir)], np.int64)
    return gp, gq, rows


def mat(q):
    return quat_to_mat(torch.as_tensor(np.ascontiguousarray(q))).numpy()


def window_pairs(frames, delta, lo, hi):
    """The pairs the trainer would actually form: windows on the S grid.

    Yields (t, k, i, j) with j = t + k the supervised frame and i = j - delta the
    detached anchor, for every window [t, t+S) whose whole anchor set is >= lo.
    """
    out = []
    for t in range(lo, hi - S + 1, S):
        k = np.arange(max(0, delta - (t - lo)), S)
        if len(k) == 0:
            continue
        out.append((t, k, t + k - delta, t + k))
    return out


def r_and_rel(C, R, i, j, t):
    """(rot [n,3,3], dir [n,3], log-ratio [n]) for pairs (i, j), window at t."""
    RiT = R[i].transpose(0, 2, 1)
    d = np.einsum("nab,nb->na", RiT, C[j] - C[i])
    dR = np.einsum("nab,nbc->nac", RiT, R[j])
    step = np.linalg.norm(np.diff(C[t:t + S], axis=0), axis=1)
    v = np.sqrt((step ** 2).mean() + 1e-16)
    n = np.linalg.norm(d, axis=1)
    return dR, d / np.maximum(n[:, None], 1e-12), np.log(np.maximum(n, 1e-12) / v), n / v


def geo_deg(A, B):
    M = np.einsum("nab,nbc->nac", A.transpose(0, 2, 1), B)
    return np.degrees(np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def mad(x):
    return float(np.median(np.abs(x - np.median(x))))


def delta_stats(blocks, gp, gq, rows, deltas, tau=0.5):
    """Per-delta target error against GT, over the pairs the trainer would form.

    ``blocks`` is a list of (frames, pose) in ONE gauge each -- the whole
    stitched track is one block; an in-run bank is one block per run.
    """
    Rg_all = mat(gq)
    acc = {d: {"rot": [], "dir": [], "b": [], "n": 0, "masked": 0} for d in deltas}
    for frames, pose in blocks:
        f0 = int(frames[0])
        C = pose[:, :3].astype(np.float64)
        R = mat(pose[:, 3:7])
        # GT rows for this block, contiguous or skip
        gr = rows[frames]
        if (gr < 0).any() or (np.diff(gr) != 1).any():
            continue
        Cg = gp[gr]
        Rg = Rg_all[gr]
        for d in deltas:
            for t, k, i, j in window_pairs(frames, d, f0, f0 + len(frames)):
                li, lj, lt = i - f0, j - f0, t - f0
                a = r_and_rel(C, R, li, lj, lt)
                b = r_and_rel(Cg, Rg, li, lj, lt)
                keep = b[3] > tau                       # GT-side near-zero mask
                acc[d]["masked"] += int((~keep).sum())
                if not keep.any():
                    continue
                acc[d]["rot"].append(geo_deg(a[0][keep], b[0][keep]))
                acc[d]["dir"].append(np.degrees(np.arccos(np.clip(
                    (a[1][keep] * b[1][keep]).sum(-1), -1, 1))))
                acc[d]["b"].append((a[2] - b[2])[keep])
                acc[d]["n"] += int(keep.sum())
    out = {}
    for d, v in acc.items():
        if not v["b"]:
            out[d] = {"n": 0}
            continue
        b = np.concatenate(v["b"])
        rot = np.concatenate(v["rot"])
        dr = np.concatenate(v["dir"])
        out[d] = {"n": v["n"], "masked": v["masked"],
                  "b_median": float(np.median(b)), "b_MAD": mad(b),
                  "b_q05": float(np.quantile(b, 0.05)),
                  "b_q95": float(np.quantile(b, 0.95)),
                  "rot_median_deg": float(np.median(rot)), "rot_MAD_deg": mad(rot),
                  "dir_median_deg": float(np.median(dr)), "dir_MAD_deg": mad(dr)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Seam composition (stitched candidate only)
# ─────────────────────────────────────────────────────────────────────────────

def sim3_mul(a, b):
    """(s,R,t) composition: apply b then a."""
    (sa, Ra, ta), (sb, Rb, tb) = a, b
    return (sa * sb, Ra @ Rb, sa * (Ra @ tb) + ta)


def sim3_inv(x):
    s, R, t = x
    return (1.0 / s, R.T, -(R.T @ t) / s)


def seam_audit(bank_root, gp, rows, trim=0.2, spans=(1, 2, 4, 7)):
    """Does the chained gauge stay true to GT, and how does the error compose?

    ★ THE QUANTITY THAT MATTERS IS RELATIVE, NOT ABSOLUTE.  A long pair spans
    ``m = ceil(Delta / stride)`` runs, so what corrupts its target is the error
    of the m-step relative transform ``A_c^-1 A_{c+m}``, not how far the whole
    chain has wandered from run 0.  An absolute drift that is smooth across the
    sequence barely touches a Delta=319 pair; a jitter between neighbours ruins
    every pair that crosses it.  So report the error per span, and let
    ``spans`` map onto the deltas (stride 48: m=1,2,4,7 -> 48, 96, 192, 336).

    ``A_c`` is the cumulative fit (run c's gauge -> run 0's), ``T_c`` the same
    thing derived from GT by fitting each run to GT on its own.
    """
    runs, idx = load_bank(bank_root)
    _, seams, G = stitch(runs, trim=trim)

    Ggt = []
    for r in runs:
        gr = rows[r["frames"]]
        if (gr < 0).any() or (np.diff(gr) != 1).any():
            Ggt.append(None); continue
        s, R, t = umeyama(r["pose"][:, :3].astype(np.float64), gp[gr])
        Ggt.append((s, R, t))
    ok0 = Ggt[0] is not None
    T = [None if (g is None or not ok0) else sim3_mul(sim3_inv(Ggt[0]), g) for g in Ggt]

    def err(x, y):
        e = sim3_mul(sim3_inv(x), y)
        return (float(np.log(e[0])), float(geo_deg(e[1][None], np.eye(3)[None])[0]),
                float(np.linalg.norm(e[2])))

    per_span = {}
    for m in spans:
        rows_ = []
        for c in range(len(runs) - m):
            if T[c] is None or T[c + m] is None:
                continue
            dA = sim3_mul(sim3_inv(G[c]), G[c + m])
            dT = sim3_mul(sim3_inv(T[c]), T[c + m])
            ls, rd, tm = err(dT, dA)
            # the span's own GT displacement, so the translation error is
            # readable as a fraction rather than as metres at an unknown speed
            f0, f1 = runs[c]["frames"][0], runs[c + m]["frames"][0]
            base = float(np.linalg.norm(gp[rows[f1]] - gp[rows[f0]])) or 1.0
            rows_.append((ls, rd, tm / base))
        if rows_:
            arr = np.array(rows_)
            per_span[m] = {"n": len(rows_), "delta": m * (idx.get("stride") or 48),
                           "log_scale_median": float(np.median(np.abs(arr[:, 0]))),
                           "log_scale_MAD": mad(arr[:, 0]),
                           "rot_deg_median": float(np.median(arr[:, 1])),
                           "rot_deg_MAD": mad(arr[:, 1]),
                           "trans_rel_median": float(np.median(arr[:, 2])),
                           "trans_rel_MAD": mad(arr[:, 2])}

    cumulative = {}
    for N in (10, 20, len(runs) - 1):
        if N < 1 or N >= len(T) or T[N] is None:
            continue
        ls, rd, tm = err(T[N], G[N])
        base = float(np.linalg.norm(gp[rows[runs[N]["frames"][0]]] -
                                    gp[rows[runs[0]["frames"][0]]])) or 1.0
        cumulative[N] = {"log_scale": ls, "rot_deg": rd, "trans_m": tm,
                         "trans_rel": tm / base}
    return per_span, cumulative, seams


# ─────────────────────────────────────────────────────────────────────────────

def blocks_stitched(d):
    z = np.load(os.path.join(d, "canonical.npz"))
    return [(z["frames"], z["pose_enc"].astype(np.float64))]


def blocks_bank(root):
    runs, _ = load_bank(root)
    return [(r["frames"], r["pose"]) for r in runs]


def show(name, st, deltas):
    print(f"\n=== {name} ===")
    print(f"{'D':>5} {'n':>7} {'b_med':>9} {'b_MAD':>9} {'b_q05':>9} {'b_q95':>9} "
          f"{'rot_med':>9} {'dir_med':>9}")
    for d in deltas:
        v = st.get(d, {"n": 0})
        if not v["n"]:
            print(f"{d:>5} {'--':>7}"); continue
        print(f"{d:>5} {v['n']:>7} {v['b_median']:>9.4f} {v['b_MAD']:>9.4f} "
              f"{v['b_q05']:>9.4f} {v['b_q95']:>9.4f} "
              f"{v['rot_median_deg']:>9.4f} {v['dir_median_deg']:>9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--calib", default="data/mcd/calib/hhs_calib.yaml")
    ap.add_argument("--sensor", default="d455b_color")
    ap.add_argument("--track", default=None, help="stitched track dir (stitch_bank --out)")
    ap.add_argument("--bank", default=None, help="in-run long bank (K_t=2)")
    ap.add_argument("--seams", default=None, help="raw L96s48 bank, for the seam audit")
    ap.add_argument("--deltas", type=int, nargs="+", default=[48, 96, 192, 319])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gp, gq, rows = gt_by_disk_index(a.frames, a.calib, a.sensor)
    print(f"[gt] {len(gp)} GT rows, {(rows >= 0).sum()} of "
          f"{len(rows)} disk frames mapped")
    rep = {}

    if a.track:
        rep["stitched"] = delta_stats(blocks_stitched(a.track), gp, gq, rows, a.deltas)
        show(f"P1 stitched  {os.path.basename(a.track)}", rep["stitched"], a.deltas)
    if a.bank:
        rep["inrun"] = delta_stats(blocks_bank(a.bank), gp, gq, rows, a.deltas)
        show(f"P2 in-run    {os.path.basename(a.bank)}", rep["inrun"], a.deltas)

    if "stitched" in rep and "inrun" in rep:
        print("\n=== decision: MAD(b_Delta), lower wins (bias is calibratable, spread is not) ===")
        for d in a.deltas:
            p1, p2 = rep["stitched"].get(d, {}), rep["inrun"].get(d, {})
            if not p1.get("n") or not p2.get("n"):
                a1 = "n/a" if not p1.get("n") else "%.4f" % p1["b_MAD"]
                a2 = "n/a" if not p2.get("n") else "%.4f" % p2["b_MAD"]
                print(f"  D={d:<5} P1 {a1}   P2 {a2}")
                continue
            win = "P1" if p1["b_MAD"] < p2["b_MAD"] else "P2"
            print(f"  D={d:<5} P1 {p1['b_MAD']:.4f}   P2 {p2['b_MAD']:.4f}   -> {win}")

    if a.seams:
        per, comp, seams = seam_audit(a.seams, gp, rows)
        rep["seam_per_span"], rep["seam_cumulative"] = per, comp
        print(f"\n=== chained gauge vs GT, PER SPAN  ({len(seams)} seams) ===")
        print("  the error a long pair actually sees: A_c^-1 A_c+m against GT")
        print(f"  {'m':>3} {'~D':>5} {'n':>5} {'|log s|':>9} {'s MAD':>9} "
              f"{'rot deg':>9} {'rot MAD':>9} {'trans/GT':>9}")
        for m, v in sorted(per.items()):
            print(f"  {m:>3} {v['delta']:>5} {v['n']:>5} "
                  f"{v['log_scale_median']:>9.4f} {v['log_scale_MAD']:>9.4f} "
                  f"{v['rot_deg_median']:>9.4f} {v['rot_deg_MAD']:>9.4f} "
                  f"{v['trans_rel_median']:>9.4f}")
        print("=== cumulative drift from run 0 (context, not the pair error) ===")
        for N, v in sorted(comp.items()):
            print(f"  run {N:>3}  log-scale {v['log_scale']:+.4f}  "
                  f"rot {v['rot_deg']:.3f} deg  trans {v['trans_m']:.2f} m "
                  f"({v['trans_rel']:.3f} of GT span)")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1, default=float)
        print(f"\n[write] {a.out}")


if __name__ == "__main__":
    main()
