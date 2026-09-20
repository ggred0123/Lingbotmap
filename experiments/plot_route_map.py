"""The drift drawn as a map: the actual campus route, and where the model thinks it went.

An error curve says the number; a map says what the number means. The deployed run
is placed on the route with one global Sim(3) -- the same alignment a downstream
consumer of the trajectory would apply -- and the residual is drawn as spokes
joining each GT point to where the model put it. Spoke length IS the error.

Fresh runs cannot be placed globally (each window carries its own gauge), so they
appear as per-window segments aligned inside their own window. That is the fair
comparison: it shows fresh reproducing the local route shape everywhere, while
making no claim about global placement.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcd_eval import umeyama

C_LS, C_FD = "#eb6834", "#2a78d6"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8b8a85"
SURFACE, GRID = "#fcfcfb", "#e4e3df"


def ground_plane(g):
    """Top-down basis from the route's dominant plane."""
    c = g - g.mean(0)
    _, _, V = np.linalg.svd(c, full_matrices=False)
    return g.mean(0), V[:2].T


def style(ax, box=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    if box:
        ax.set_aspect("equal", adjustable="datalim")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom_at", type=int, default=7400)
    ap.add_argument("--zoom_len", type=int, default=600)
    ap.add_argument("--anchor_on", type=int, default=200,
                    help="frames used to anchor the drawing at the start")
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    g, ls_glob, dist = d["gt"], d["ls_global"], d["dist"]
    segs, rng = d["fd_segments"], d["fd_ranges"]

    # Re-anchor at the start.  A single Sim(3) fitted over the whole route is the
    # right thing for an ATE number, but it is the wrong thing for a drift picture:
    # least squares spreads the residual everywhere, so the estimate looks wrong
    # from frame 0 even where it is locally fine (0.3 m at 10 m travelled).  The
    # model's own scale also drifts 9.7 -> 19.0 along the route, so no single
    # scale fits both ends.  Aligning on the opening segment instead makes the
    # drawing show what it claims to show: error starting at zero and accumulating.
    # (Composing a similarity onto ls_global is equivalent to re-fitting the raw
    # poses, so nothing needs re-running.)
    n0 = args.anchor_on
    s0, R0, t0 = umeyama(ls_glob[:n0], g[:n0])
    ls = (s0 * (R0 @ ls_glob.T)).T + t0
    err = np.linalg.norm(ls - g, axis=1)
    err_glob = d["ls_err"]

    mu, P = ground_plane(g)
    G, L = (g - mu) @ P, (ls - mu) @ P

    fig = plt.figure(figsize=(15.5, 8.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.30, wspace=0.16,
                          left=0.05, right=0.985, top=0.845, bottom=0.075)

    # ── left: the whole route ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[:, 0]); style(ax)
    step = max(1, len(G) // 320)
    spokes = np.stack([G[::step], L[::step]], axis=1)
    ax.add_collection(LineCollection(spokes, colors=C_LS, linewidths=0.7,
                                     alpha=0.45, zorder=2))
    ax.plot(L[:, 0], L[:, 1], color=C_LS, lw=1.6, zorder=3, alpha=0.95,
            solid_capstyle="round")
    ax.plot(G[:, 0], G[:, 1], color=SURFACE, lw=5.0, zorder=4)
    ax.plot(G[:, 0], G[:, 1], color=INK, lw=2.6, zorder=5, solid_capstyle="round")
    ax.plot(G[0, 0], G[0, 1], "o", color=INK, ms=9, mec=SURFACE, mew=2, zorder=7)
    ax.annotate("start", (G[0, 0], G[0, 1]), textcoords="offset points",
                xytext=(10, 10), color=INK, fontsize=10, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3.5, foreground=SURFACE)])
    k = int(np.argmax(err))
    ax.plot([G[k, 0], L[k, 0]], [G[k, 1], L[k, 1]], color=C_LS, lw=2.2, zorder=6)
    ax.annotate(f"drifted {err[k]:.0f} m\n(at {dist[k]:.0f} m travelled)",
                ((G[k, 0] + L[k, 0]) / 2, (G[k, 1] + L[k, 1]) / 2),
                textcoords="offset points", xytext=(14, 0), color=C_LS,
                fontsize=10, fontweight="bold", va="center",
                path_effects=[pe.withStroke(linewidth=3.5, foreground=SURFACE)])
    ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_title("Where the model thinks it walked  ·  aligned on the first 200 frames",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=9)
    ax.text(0, -0.075, "black = surveyed route   |   orange = deployed run, Sim(3) fitted on the "
            "opening segment only   |   spoke length = drift at that point\n"
            f"drift peaks at {err.max():.0f} m around {dist[int(np.argmax(err))]:.0f} m travelled, then "
            f"falls back to {err[-1]:.0f} m only because the route loops home "
            f"(for reference, a single Sim(3) fitted over the whole route gives "
            f"ATE rmse {np.sqrt((err_glob**2).mean()):.0f} m)",
            transform=ax.transAxes, ha="left", color=INK3, fontsize=8.5)

    # ── top right: fresh, placed per window ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); style(ax)
    ax.plot(G[:, 0], G[:, 1], color=INK, lw=2.2, zorder=3, solid_capstyle="round")
    for s_, (a, b) in zip(segs, rng):
        S_ = (s_ - mu) @ P
        ax.plot(S_[:, 0], S_[:, 1], color=C_FD, lw=2.6, zorder=4,
                solid_capstyle="round")
    ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_title(f"Fresh runs on the same route  ·  {len(segs)} windows",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax.text(0, -0.135, "each window re-anchored and aligned within itself — "
            "the local route shape survives everywhere",
            transform=ax.transAxes, ha="left", color=INK3, fontsize=8.5)

    # ── bottom right: zoom on the worst stretch ──────────────────────────────
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    a, b = args.zoom_at, min(args.zoom_at + args.zoom_len, len(G))
    Gz, Lz = G[a:b], L[a:b]
    st = max(1, (b - a) // 60)
    ax.add_collection(LineCollection(np.stack([Gz[::st], Lz[::st]], axis=1),
                                     colors=C_LS, linewidths=0.9, alpha=0.5, zorder=2))
    ax.plot(Lz[:, 0], Lz[:, 1], color=C_LS, lw=2.0, zorder=3, solid_capstyle="round")
    ax.plot(Gz[:, 0], Gz[:, 1], color=SURFACE, lw=5.0, zorder=4)
    ax.plot(Gz[:, 0], Gz[:, 1], color=INK, lw=2.6, zorder=5, solid_capstyle="round")
    inside = [(s_, r) for s_, r in zip(segs, rng) if r[0] >= a and r[1] <= b]
    for s_, _ in inside:
        S_ = (s_ - mu) @ P
        ax.plot(S_[:, 0], S_[:, 1], color=C_FD, lw=2.6, zorder=6,
                solid_capstyle="round")
    ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_title(f"Zoom · {dist[a]:.0f}–{dist[b-1]:.0f} m travelled",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax.text(0, -0.135, f"deployed run is {err[a:b].mean():.0f} m off here on average; "
            "fresh sits on the route",
            transform=ax.transAxes, ha="left", color=INK3, fontsize=8.5)

    fig.suptitle("A 1,401 m walk: the deployed streaming run loses the route, a fresh run does not",
                 color=INK, fontsize=15, fontweight="bold", x=0.05, ha="left", y=0.968)
    fig.text(0.05, 0.928, "MCD kth_day_06 · 8,894 frames @10 Hz · survey-grade ground truth · "
             f"deployment keyframe_interval K={int(d['K'])}",
             color=INK2, fontsize=9.5, ha="left")
    fig.legend(handles=[Line2D([], [], color=INK, lw=2.6, label="GT  surveyed route"),
                        Line2D([], [], color=C_LS, lw=2, label="LS  deployed streaming run"),
                        Line2D([], [], color=C_FD, lw=2.6, label="FD  fresh run (per window)")],
               loc="lower left", bbox_to_anchor=(0.048, 0.868), frameon=False,
               fontsize=9.5, labelcolor=INK2, ncol=3, handlelength=1.8, columnspacing=2.0)

    fig.savefig(args.out, dpi=140, facecolor=SURFACE)
    print(f"[saved] {args.out}")
    print(f"\n  route {dist[-1]:.0f} m")
    print(f"  start-anchored drift : " + "  ".join(
        f"{dist[i]:.0f}m={err[i]:.1f}m" for i in
        [int(len(err)*f) for f in (0.02, 0.1, 0.25, 0.5, 0.75, 0.99)]))
    print(f"  start-anchored max   : {err.max():.1f} m at {dist[int(np.argmax(err))]:.0f} m")
    print(f"  whole-route Sim(3)   : ATE rmse {np.sqrt((err_glob**2).mean()):.2f} m "
          f"(the ATE number; not a drift picture)")


if __name__ == "__main__":
    main()
