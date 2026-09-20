"""MCD continuous-time ground truth: uniform cumulative B-spline evaluator.

MCD ships ``spline.csv`` (basalt-style uniform cumulative B-spline over SO(3) x R^3)
plus ``pose_inW.csv`` (the same trajectory sampled at 10 Hz).  The official reader
is the ``ceva`` package, which wraps basalt and does not build here, so this is a
standalone NumPy implementation.

``pose_inW.csv`` is used as the correctness oracle: evaluating the spline at those
timestamps must reproduce them.  ``self_test()`` does exactly that and also resolves
the control-point indexing convention empirically instead of assuming one.

Cumulative form (Sommer et al., "Efficient Derivative Computation for Cumulative
B-Splines on Lie Groups"), order k, local parameter u in [0,1):

    p(u) = P_i + sum_{s=1..k-1} Bc_s(u) * (P_{i+s} - P_{i+s-1})
    R(u) = R_i * prod_{s=1..k-1} exp( Bc_s(u) * log(R_{i+s-1}^T R_{i+s}) )

with Bc_s the cumulative uniform B-spline basis (Bc_0 == 1).
"""

import numpy as np
from scipy.interpolate import BSpline


# ── SO(3) helpers (quaternions are xyzw, matching MCD and lingbot_map) ───────

def quat_to_mat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R):
    """Shepperd's method; returns xyzw."""
    R = np.asarray(R, dtype=np.float64)
    m = lambda i, j: R[..., i, j]
    tr = m(0, 0) + m(1, 1) + m(2, 2)
    q = np.empty(R.shape[:-2] + (4,))
    big = tr > 0
    s = np.sqrt(np.maximum(tr + 1.0, 1e-20)) * 2
    q[big] = np.stack([(m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s,
                       (m(1, 0) - m(0, 1)) / s, 0.25 * s], -1)[big]
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = (~big) & (m(i, i) >= m(j, j)) & (m(i, i) >= m(k, k))
        if not np.any(sel):
            continue
        s = np.sqrt(np.maximum(1.0 + m(i, i) - m(j, j) - m(k, k), 1e-20)) * 2
        v = np.zeros(R.shape[:-2] + (4,))
        v[..., i] = 0.25 * s
        v[..., j] = (m(j, i) + m(i, j)) / s
        v[..., k] = (m(k, i) + m(i, k)) / s
        v[..., 3] = (m(k, j) - m(j, k)) / s
        q[sel] = v[sel]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def so3_log(R):
    """Rotation matrix -> rotation vector, stable near 0 and pi."""
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    ax = np.stack([R[..., 2, 1] - R[..., 1, 2],
                   R[..., 0, 2] - R[..., 2, 0],
                   R[..., 1, 0] - R[..., 0, 1]], -1)
    sin = np.linalg.norm(ax, axis=-1) / 2
    theta = np.arctan2(sin, cos)
    small = sin < 1e-10
    scale = np.where(small, 0.5, theta / np.maximum(2 * sin, 1e-20))
    return ax * scale[..., None]


def so3_exp(w):
    w = np.asarray(w, dtype=np.float64)
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    small = th < 1e-10
    a = np.where(small, 1.0, np.sin(th) / np.maximum(th, 1e-20))
    b = np.where(small, 0.5, (1 - np.cos(th)) / np.maximum(th ** 2, 1e-20))
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -w[..., 2], w[..., 1]
    K[..., 1, 0], K[..., 1, 2] = w[..., 2], -w[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -w[..., 1], w[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    return I + a[..., None] * K + b[..., None] * (K @ K)


# ── cumulative uniform B-spline basis ────────────────────────────────────────

def cumulative_basis(order, u):
    """Bc[s](u) for s=0..order-1 on a uniform knot vector, u in [0,1)."""
    k = order
    # Uniform B-spline basis on integer knots; evaluate the k basis functions
    # that are active on the span [0,1).
    knots = np.arange(-(k - 1), k + 1, dtype=np.float64)
    B = np.empty((k, len(np.atleast_1d(u))))
    for s in range(k):
        c = np.zeros(k)
        c[s] = 1.0
        B[s] = BSpline(knots, c, k - 1, extrapolate=False)(np.atleast_1d(u))
    B = np.nan_to_num(B)
    # cumulative: Bc[s] = sum_{j>=s} B[j]
    return np.cumsum(B[::-1], axis=0)[::-1]


class McdSplineGT:
    def __init__(self, spline_csv):
        with open(spline_csv) as f:
            hdr = f.readline()
        meta = dict(p.split(":") for p in
                    [x.strip() for x in hdr.strip().split(",")])
        self.dt = float(meta["Dt"])
        self.order = int(meta["Order"])
        self.t0 = float(meta["MinTime"])
        self.t1 = float(meta["MaxTime"])
        d = np.loadtxt(spline_csv, delimiter=",", skiprows=1)
        self.ct = d[:, 1]
        self.cp = d[:, 2:5]
        self.cq = d[:, 5:9]
        self.cR = quat_to_mat(self.cq)
        self.offset = 0          # resolved by self_test()
        print(f"[spline] order={self.order} dt={self.dt} "
              f"ctrl={len(self.cp)} span={self.t1 - self.t0:.1f}s")

    def eval(self, ts, offset=None):
        """Evaluate at timestamps ts -> (positions [N,3], quaternions xyzw [N,4])."""
        k, off = self.order, self.offset if offset is None else offset
        ts = np.atleast_1d(np.asarray(ts, dtype=np.float64))
        x = (ts - self.t0) / self.dt
        i = np.floor(x).astype(int) + off
        u = x - np.floor(x)
        i = np.clip(i, 0, len(self.cp) - k)

        Bc = cumulative_basis(k, u)                      # [k, N]
        P = self.cp[i[:, None] + np.arange(k)[None, :]]  # [N,k,3]
        R = self.cR[i[:, None] + np.arange(k)[None, :]]  # [N,k,3,3]

        pos = P[:, 0].copy()
        for s in range(1, k):
            pos += Bc[s][:, None] * (P[:, s] - P[:, s - 1])

        rot = R[:, 0].copy()
        for s in range(1, k):
            d = so3_log(np.einsum("nij,njk->nik",
                                  R[:, s - 1].transpose(0, 2, 1), R[:, s]))
            rot = np.einsum("nij,njk->nik", rot, so3_exp(Bc[s][:, None] * d))
        return pos, mat_to_quat(rot)

    def self_test(self, pose_csv, n=4000):
        """Resolve the indexing convention against the 10 Hz discrete poses."""
        d = np.loadtxt(pose_csv, delimiter=",", skiprows=1)
        m = (d[:, 1] > self.t0 + 1.0) & (d[:, 1] < self.t1 - 1.0)
        d = d[m][:n]
        ts, pg, qg = d[:, 1], d[:, 2:5], d[:, 5:9]

        best = None
        for off in range(-(self.order), 2):
            p, q = self.eval(ts, offset=off)
            ep = np.linalg.norm(p - pg, axis=-1)
            dot = np.abs(np.sum(q * qg, axis=-1)).clip(max=1.0)
            er = np.degrees(2 * np.arccos(dot))
            score = np.median(ep)
            print(f"  offset={off:>3}: pos median {np.median(ep):.6f} m  "
                  f"max {ep.max():.4f} m | rot median {np.median(er):.5f}°")
            if best is None or score < best[1]:
                best = (off, score, np.median(ep), np.median(er), ep.max())
        self.offset = best[0]
        print(f"[spline] chosen offset={self.offset}  "
              f"pos_median={best[2]*1000:.3f} mm  rot_median={best[3]:.5f}°  "
              f"pos_max={best[4]*1000:.2f} mm")
        return best[2], best[3]


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/mcd/gt"
    s = McdSplineGT(f"{root}/spline.csv")
    pos_err, rot_err = s.self_test(f"{root}/pose_inW.csv")
    ok = pos_err < 1e-3 and rot_err < 0.01
    print(("\nPASS" if ok else "\nFAIL") +
          f" — spline reproduces discrete GT to {pos_err*1000:.3f} mm / {rot_err:.5f}°")
