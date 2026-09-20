"""Verification figure for the v4 finding: the long run's state degrades, fresh does not.

Four panels, each answering one question:
  A  does error grow with distance?          (whole run, shipped inference_streaming)
  B  what does the failure look like?        (worst window, top-down trajectory)
  C  is it the whole window or a blow-up?    (per-frame error inside the window)
  D  does the geometry itself break?         (predicted depth, same frame, LS vs FD)

Ground truth is drawn as neutral ink, not a coloured series -- it is the reference
the others are measured against, not a peer. That leaves three coloured series,
which is the documented all-pairs-safe cap for the reference palette.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# Reference categorical palette, light mode, slots 1-3 (validated all-pairs).
C = {"LS": "#eb6834", "FD": "#2a78d6", "LD": "#1baf7a"}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8b8a85"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
LABEL = {"LS": "LS  long-sparse (deployed)", "LD": "LD  long-dense (fork teacher)",
         "FD": "FD  fresh-dense (fresh teacher)"}


def style(ax):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--single", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = np.load(args.window, allow_pickle=True)
    sr = json.load(open(args.single))
    rows = sr["rows"]

    fig = plt.figure(figsize=(13.5, 10.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.22,
                          left=0.065, right=0.98, top=0.845, bottom=0.075)

    # ── A: does error grow with distance? ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    d = [r["dist"] for r in rows]
    rot = [r["rpe_rot"] for r in rows]
    ax.plot(d, rot, color=C["LS"], lw=2, marker="o", ms=5,
            mec=SURFACE, mew=1.5, zorder=3)
    z = np.polyfit(d, rot, 1)
    ax.plot(d, np.poly1d(z)(d), color=C["LS"], lw=1, ls=(0, (4, 3)), alpha=0.55, zorder=2)
    ax.set_xlabel("distance travelled from anchor (m)")
    ax.set_ylabel("local rotation error (deg / 0.5 s)")
    ax.set_title("A · the deployed run degrades with distance",
                 color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.annotate(f"{rot[0]:.2f}°", (d[0], rot[0]), textcoords="offset points",
                xytext=(6, -12), color=INK2, fontsize=8.5)
    ax.annotate(f"{rot[-1]:.2f}°  ({rot[-1]/rot[0]:.1f}×)", (d[-1], rot[-1]),
                textcoords="offset points", xytext=(-64, 6), color=C["LS"],
                fontsize=9, fontweight="bold")
    ax.text(0, -0.20, "whole 1,401 m run · shipped inference_streaming · K=28",
            transform=ax.transAxes, ha="left", color=INK3, fontsize=8)

    # ── B: what the failure looks like ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); style(ax)
    g = w["gt_pos"]
    # project onto the window's dominant horizontal plane for a top-down view
    ctr = g - g.mean(0)
    _, _, V = np.linalg.svd(ctr, full_matrices=False)
    P = V[:2].T
    gp = ctr @ P
    # noisy estimates first, then FD, then GT on top with a surface halo
    for tag in ("LS", "LD"):
        t = (w[f"{tag}_traj"] - g.mean(0)) @ P
        ax.plot(t[:, 0], t[:, 1], color=C[tag], lw=1.5, zorder=3, alpha=0.9,
                solid_capstyle="round")
        j = int(len(t) * (0.86 if tag == "LS" else 0.12))
        ax.annotate(tag, (t[j, 0], t[j, 1]), textcoords="offset points",
                    xytext=(6, 6), color=C[tag], fontsize=10, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])
    t = (w["FD_traj"] - g.mean(0)) @ P
    ax.plot(t[:, 0], t[:, 1], color=SURFACE, lw=4.5, zorder=5)
    ax.plot(t[:, 0], t[:, 1], color=C["FD"], lw=2.2, zorder=6, solid_capstyle="round")
    ax.plot(gp[:, 0], gp[:, 1], color=SURFACE, lw=5.5, zorder=7)
    ax.plot(gp[:, 0], gp[:, 1], color=INK, lw=2.4, zorder=8, solid_capstyle="round")
    ax.annotate("GT + FD\n(overlapping)", (gp[-1, 0], gp[-1, 1]),
                textcoords="offset points", xytext=(-30, 20), color=INK, fontsize=9,
                fontweight="bold", ha="center",
                path_effects=[pe.withStroke(linewidth=3.5, foreground=SURFACE)])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_title(f"B · worst window, top-down  ({w['dist'][0]:.0f}–{w['dist'][-1]:.0f} m)",
                 color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.text(0, -0.20, "each trajectory Sim(3)-aligned to GT over this window",
            transform=ax.transAxes, ha="left", color=INK3, fontsize=8)

    # ── C: per-frame error inside the window ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    dd = w["dist"]
    for tag in ("LS", "LD", "FD"):
        ax.plot(dd, w[f"{tag}_ate"], color=C[tag], lw=1.7, zorder=3,
                alpha=0.95 if tag == "FD" else 0.85)
    ax.set_xlabel("distance travelled from anchor (m)")
    ax.set_ylabel("position error vs GT (m)")
    ax.set_title("C · error is sustained across the window, not a spike",
                 color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
    # rmse callouts stacked in clear space rather than on the traces
    for i, tag in enumerate(("LS", "LD", "FD")):
        e = w[f"{tag}_ate"]
        ax.text(0.985, 0.95 - i * 0.09, f"{tag}   rmse {np.sqrt((e**2).mean()):.2f} m",
                transform=ax.transAxes, ha="right", va="top", color=C[tag],
                fontsize=9.5, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3.5, foreground=SURFACE)])
    ax.text(0, -0.20, "fresh stays flat while both long-context runs drift",
            transform=ax.transAxes, ha="left", color=INK3, fontsize=8)

    # ── D: predicted depth, same frame ───────────────────────────────────────
    have = sorted(int(k.split("_")[-1]) for k in w.files
                  if k.startswith("LS_depth_") and f"FD_depth_{k.split('_')[-1]}" in w.files)
    j = have[len(have) // 2] if have else None
    if j is not None:
        sub = gs[1, 1].subgridspec(1, 2, wspace=0.06)
        dls, dfd = w[f"LS_depth_{j}"], w[f"FD_depth_{j}"]
        # normalise each by its own median: the two runs live in different gauges
        nls, nfd = dls / np.median(dls), dfd / np.median(dfd)
        vmax = float(np.percentile(np.concatenate([nls.ravel(), nfd.ravel()]), 97))
        for i, (tag, im) in enumerate((("LS", nls), ("FD", nfd))):
            a = fig.add_subplot(sub[0, i])
            a.imshow(im, cmap="magma", vmin=0, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_color(C[tag]); s.set_linewidth(2.4)
            a.set_title(tag, color=C[tag], fontsize=10, fontweight="bold", pad=5)
        fig.text(0.545, 0.415, f"D · predicted depth, same input frame (idx {j})",
                 ha="left", color=INK, fontsize=11, fontweight="bold")
        fig.text(0.545, 0.392, "each normalised by its own median · "
                 "identical pixels in, different geometry out",
                 ha="left", color=INK3, fontsize=8)
    else:
        ax = fig.add_subplot(gs[1, 1]); style(ax)
        ax.text(0.5, 0.5, "no depth maps in dump", ha="center", color=INK3)

    fig.suptitle("Streaming state degrades with distance; a fresh run on the same frames does not",
                 color=INK, fontsize=14.5, fontweight="bold", x=0.065, ha="left", y=0.972)
    fig.text(0.065, 0.936,
             f"MCD kth_day_06 · 8,894 frames @10 Hz · 1,401 m · survey-grade GT   |   "
             f"worst window t0={int(w['t0'])}, L={int(w['L'])}, K={int(w['K'])}, "
             f"teacher K_t={int(w['Kt'])}",
             color=INK2, fontsize=9.5, ha="left")
    handles = [Line2D([], [], color=INK, lw=2.6, label="GT  survey-grade")] + \
              [Line2D([], [], color=C[t], lw=2, label=LABEL[t]) for t in ("LS", "LD", "FD")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.063, 0.877),
               frameon=False, fontsize=9, labelcolor=INK2, ncol=4, handlelength=1.8,
               columnspacing=1.8)

    fig.savefig(args.out, dpi=145, facecolor=SURFACE)
    print(f"[saved] {args.out}")

    print("\n--- table view (accessibility: never colour alone) ---")
    print(f"{'cond':<5} {'ATE rmse(m)':>12} {'ATE med(m)':>11} {'rot RPE(deg)':>13} {'scale':>8}")
    for tag in ("LS", "LD", "FD"):
        e, r = w[f"{tag}_ate"], w[f"{tag}_rot"]
        print(f"{tag:<5} {np.sqrt((e**2).mean()):>12.3f} {np.median(e):>11.3f} "
              f"{r.mean():>13.3f} {float(w[f'{tag}_scale']):>8.3f}")


if __name__ == "__main__":
    main()
