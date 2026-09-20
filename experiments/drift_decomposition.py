"""Which channel breaks along a step ladder -- rotation, translation direction, or scale?

Three rival explanations of the Oxford K=1 degradation are on record (rotation
starvation at lam_rot=1, the A1P support jump in L_rot's share, per-step clipping)
next to the null-space one (A1PC is blind to a common pose+depth scale and to
per-frame depth scale).  They make different predictions about WHICH quantity
degrades with training steps, so this script reads the saved Oxford K=1
benchmark outputs (traj.txt + depth/*.exr, no inference) and decomposes the
error per checkpoint step:

  rotation   rpe_rot@1, rpe_rot@48 (gauge-free), abs orientation error after
             the global Sim(3) fit
  direction  per-step translation-direction angle after the global Sim(3) fit
  scale      per-step pose-scale ratio to GT, rho_i = s_glob*|dc_pred|/|dc_gt|:
               breath = std(log rho) INSIDE 48-frame windows
               drift  = std across windows of the window-median log rho
             plus the windowed Sim(3) scale s_w against s_glob
  depth      per-frame median depth ratio to base_k1 on the SAME frame
             (scene content cancels), same breath / drift / end-to-end split
  pose/depth log(step ratio to base) - log(depth ratio to base), the quantity
             L_motion-depth pins in training

Null-space prediction: scale / depth / ratio rise with step while rotation
stays flat.  Rotation-starvation prediction: rotation rises.

Runs under the bench venv (OpenEXR):
  /NHNHOME/.../.venv-bench/bin/python experiments/drift_decomposition.py \
      --runs c0off s0on s0off v6i v7d v7e v7f gtsup teasup --workers 32
Per-frame depth medians are cached under experiments/results/drift_decomp/.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

WS = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/bench_ws/oxford/oxford"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "experiments", "results", "drift_decomp")
OUT = os.path.join(ROOT, "experiments", "results", "drift_decomposition.json")

W, STRIDE = 48, 24
SCENES = sorted(d for d in os.listdir(WS)
                if os.path.isdir(os.path.join(WS, d)) and d != "eval")


# ── geometry ────────────────────────────────────────────────────────────────

def load_traj(path):
    a = np.loadtxt(path)
    M = a[:, 1:].reshape(-1, 3, 4)
    return a[:, 0].astype(int), M[:, :, :3], M[:, :, 3]


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(C)
    Wm = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Wm[2, 2] = -1
    R = U @ Wm @ Vt
    s = np.trace(np.diag(sig) @ Wm) / max((S ** 2).sum() / len(src), 1e-12)
    return s, R, mu_d - s * R @ mu_s


def geo_deg(A, B):
    M = np.einsum("nij,njk->nik", A.transpose(0, 2, 1), B)
    cos = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def rel_rot(R, k):
    return np.einsum("nij,njk->nik", R[:-k].transpose(0, 2, 1), R[k:])


def windows(n):
    starts = list(range(0, max(1, n - W + 1), STRIDE))
    if starts[-1] + W < n:
        starts.append(n - W)
    return [(s, min(s + W, n)) for s in starts]


# ── depth medians (cached) ───────────────────────────────────────────────────

def depth_medians(scene: str, method: str) -> Optional[np.ndarray]:
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f"{scene}__{method}.npy")
    if os.path.exists(cp):
        return np.load(cp)
    ddir = os.path.join(WS, scene, method, "depth")
    if not os.path.isdir(ddir):
        return None
    import OpenEXR, Imath  # bench venv only
    files = sorted(f for f in os.listdir(ddir) if f.endswith(".exr"))
    med = np.zeros(len(files), dtype=np.float64)
    for i, f in enumerate(files):
        ex = OpenEXR.InputFile(os.path.join(ddir, f))
        dw = ex.header()["dataWindow"]
        h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1
        d = np.frombuffer(ex.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT)),
                          dtype=np.float32).reshape(h, w)
        med[i] = float(np.median(d))
    np.save(cp, med)
    return med


# ── one (scene, method) ──────────────────────────────────────────────────────

def analyse(scene: str, method: str, base: str = "base_k1") -> Optional[Dict[str, float]]:
    mdir = os.path.join(WS, scene, method)
    if not os.path.exists(os.path.join(mdir, "traj.txt")):
        return None
    _, Rp, cp = load_traj(os.path.join(mdir, "traj.txt"))
    _, Rg, cg = load_traj(os.path.join(WS, scene, "gt", "traj.txt"))
    _, Rb, cb = load_traj(os.path.join(WS, scene, base, "traj.txt"))
    n = min(len(cp), len(cg), len(cb))
    Rp, cp, Rg, cg, Rb, cb = Rp[:n], cp[:n], Rg[:n], cg[:n], Rb[:n], cb[:n]

    s, Ra, ta = umeyama(cp, cg)
    al = (s * (Ra @ cp.T)).T + ta
    ate = float(np.sqrt(((al - cg) ** 2).sum(1).mean()))

    out = {"ate": ate, "s_glob": float(s)}

    # rotation
    out["rot1"] = float(geo_deg(rel_rot(Rp, 1), rel_rot(Rg, 1)).mean())
    k = min(W, n - 1)
    out["rot48"] = float(geo_deg(rel_rot(Rp, k), rel_rot(Rg, k)).mean())
    out["absrot"] = float(geo_deg(np.einsum("ij,njk->nik", Ra, Rp), Rg).mean())

    # direction (after the global fit) and per-step scale ratio to GT
    dp = al[1:] - al[:-1]
    dg = cg[1:] - cg[:-1]
    np_, ng = np.linalg.norm(dp, axis=1), np.linalg.norm(dg, axis=1)
    ok = ng > 1e-3
    cosd = np.clip((dp * dg).sum(1) / np.maximum(np_ * ng, 1e-12), -1, 1)
    out["dir1"] = float(np.degrees(np.arccos(cosd[ok])).mean())
    lrho = np.log(np.maximum(np_, 1e-12)) - np.log(np.maximum(ng, 1e-12))
    lrho = np.where(ok, lrho, np.nan)

    wins = windows(n - 1)
    def win_stats(x):
        inside = [np.nanstd(x[a:b]) for a, b in wins]
        meds = [np.nanmedian(x[a:b]) for a, b in wins]
        return (float(np.nanmean(inside)), float(np.nanstd(meds)),
                float(meds[-1] - meds[0]), float(np.nanmax(meds) - np.nanmin(meds)))

    out["scale_breath"], out["scale_drift"], out["scale_end"], out["scale_range"] = win_stats(lrho)

    # windowed Sim(3) scale against the global one
    ls = []
    for a, b in windows(n):
        sw, _, _ = umeyama(cp[a:b], cg[a:b])
        ls.append(np.log(max(sw, 1e-12) / max(s, 1e-12)))
    out["wscale_std"] = float(np.std(ls))
    out["wscale_range"] = float(np.max(ls) - np.min(ls))

    # depth channel, relative to base on the same frame
    dm, db = depth_medians(scene, method), depth_medians(scene, base)
    if dm is not None and db is not None:
        m = min(len(dm), len(db), n)
        ldelta = np.log(dm[:m]) - np.log(db[:m])
        out["depth_breath"], out["depth_drift"], out["depth_end"], out["depth_range"] = win_stats(ldelta)
        # pose step ratio to base (raw units of each run), then pose/depth
        sb = np.linalg.norm(cb[1:] - cb[:-1], axis=1)
        sp_ = np.linalg.norm(cp[1:] - cp[:-1], axis=1)
        okb = (sb > 1e-6) & (sp_ > 1e-6)
        lpi = np.where(okb, np.log(np.maximum(sp_, 1e-12)) - np.log(np.maximum(sb, 1e-12)), np.nan)
        lr = lpi[:m - 1] - 0.5 * (ldelta[:m - 1] + ldelta[1:m])
        out["ratio_breath"], out["ratio_drift"], out["ratio_end"], out["ratio_range"] = win_stats(lr)
        out["pose_vs_base_drift"] = win_stats(lpi)[1]
    return out


def _job(args):
    scene, method = args
    try:
        return scene, method, analyse(scene, method)
    except Exception as e:  # keep the sweep going; report at the end
        return scene, method, {"error": repr(e)}


# ── ladders ──────────────────────────────────────────────────────────────────

def ladder(run: str, suffix: str = "k1") -> List[int]:
    steps = set()
    for sc in SCENES:
        for d in os.listdir(os.path.join(WS, sc)):
            if d.startswith(f"sd_{run}s") and d.endswith(f"_{suffix}"):
                steps.add(int(d[len(f"sd_{run}s"):-len(f"_{suffix}")]))
    return sorted(steps)


KEYS = ["ate", "rot1", "rot48", "absrot", "dir1", "scale_breath", "scale_drift",
        "scale_end", "wscale_std", "depth_breath", "depth_drift", "depth_end",
        "ratio_breath", "ratio_drift", "ratio_end"]


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r) if r == r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["c0off", "s0on", "s0off", "v6i", "v6f",
                                                  "v7d", "v7e", "v7f", "gtsup", "teasup",
                                                  "gtmag", "gtopt"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--suffix", default="k1")
    args = ap.parse_args()

    jobs = [(sc, "base_k1") for sc in SCENES]
    plan = {}
    for run in args.runs:
        st = ladder(run, args.suffix)
        plan[run] = st
        for s in st:
            jobs += [(sc, f"sd_{run}s{s}_{args.suffix}") for sc in SCENES]
    print(f"{len(jobs)} (scene, method) jobs over {len(SCENES)} scenes; ladders: "
          + ", ".join(f"{r}:{len(s)}" for r, s in plan.items()), flush=True)

    per = {}
    # depth medians for base first (every job needs them) -- do it serially to avoid races
    for sc in SCENES:
        depth_medians(sc, "base_k1")
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (sc, m, res) in enumerate(ex.map(_job, jobs, chunksize=2)):
            if res is not None:
                per.setdefault(m, {})[sc] = res
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    errors = {m: {sc: r["error"] for sc, r in d.items() if "error" in r} for m, d in per.items()}
    errors = {m: e for m, e in errors.items() if e}
    if errors:
        print("errors:", json.dumps(errors, indent=1)[:2000])

    def agg(method):
        rows = [r for r in per.get(method, {}).values() if "error" not in r]
        return {k: float(np.median([r[k] for r in rows if k in r])) for k in KEYS
                if any(k in r for r in rows)} | {"n_scenes": len(rows)}

    table = {"base": agg("base_k1"), "runs": {}}
    for run, st in plan.items():
        table["runs"][run] = {str(s): agg(f"sd_{run}s{s}_{args.suffix}") for s in st}
        # monotonicity of each metric along the ladder (base as step 0)
        xs = [0] + st
        trend = {}
        for k in KEYS:
            ys = [table["base"].get(k, np.nan)] + [table["runs"][run][str(s)].get(k, np.nan) for s in st]
            if all(y == y for y in ys):
                trend[k] = spearman(xs, ys)
        table["runs"][run]["_spearman_vs_step"] = trend
    with open(OUT, "w") as f:
        json.dump({"table": table, "per_scene": per, "W": W, "stride": STRIDE}, f, indent=1)

    # ── print ────────────────────────────────────────────────────────────────
    cols = ["ate", "rot1", "rot48", "absrot", "dir1", "scale_breath", "scale_drift",
            "wscale_std", "depth_breath", "depth_drift", "depth_end", "ratio_drift", "ratio_end"]
    hdr = f"{'step':>5} " + " ".join(f"{c:>12}" for c in cols)
    for run, st in plan.items():
        print(f"\n===== {run}  (median over {table['base']['n_scenes']} Oxford K=1 scenes; step 0 = base) =====")
        print(hdr)
        rows = [("0", table["base"])] + [(str(s), table["runs"][run][str(s)]) for s in st]
        for s, r in rows:
            print(f"{s:>5} " + " ".join(f"{r.get(c, float('nan')):12.4f}" for c in cols))
        tr = table["runs"][run]["_spearman_vs_step"]
        print(f"{'rho':>5} " + " ".join(f"{tr.get(c, float('nan')):12.2f}" for c in cols)
              + "   <- Spearman(step, metric)")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
