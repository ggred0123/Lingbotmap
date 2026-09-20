"""Two figures for the Phase 1 training report.

  A  did the deployed run actually get better, and where?   (LS ATE vs distance)
  B  what does the regulariser buy and cost?                (L2-SP trade-off)

Palette and axis styling follow experiments/plot_worst_window.py so the training
figures sit next to the Phase 0 diagnosis figures without a visual seam.  The
trained model keeps the LS colour it is measured on and is drawn as the solid
series; the released model is the muted reference it must beat.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Labels are Korean; without this every glyph renders as a tofu box.
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

C = {"LS": "#eb6834", "FD": "#2a78d6", "LD": "#1baf7a"}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8b8a85"
SURFACE, GRID = "#fcfcfb", "#e4e3df"


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


def rows(name):
    with open(os.path.join(R, name)) as f:
        return {r["t0"]: r for r in json.load(f)["rows"]}


def main():
    base = rows("ate_vs_distance.json")
    best = rows("ate_vs_distance_6b.json")
    t0s = sorted(base)
    d = np.array([base[t]["dist"] for t in t0s])
    b_ls = np.array([base[t]["ls_ate"] for t in t0s])
    n_ls = np.array([best[t]["ls_ate"] for t in t0s])
    b_fd = np.array([base[t]["fd_ate"] for t in t0s])
    n_fd = np.array([best[t]["fd_ate"] for t in t0s])

    # Training covered TRAINING frames 80..3000.  Those are sorted(listdir) indices;
    # GT/distance is indexed by meta.npz rows, which start 11 frames later -- so the
    # boundary is meta row 2989, not 3000.  Taking a window t0 instead (2880) would
    # draw ~15 m of genuinely-trained route as untrained.
    meta = np.load(os.path.join(os.path.dirname(R), "..", "data", "mcd",
                                "kth_day_06", "frames_10hz", "meta.npz"), allow_pickle=True)
    gp = meta["gt_pos"]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))])
    trained_m = float(cum[3000 - 11])

    # ── A ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)

    ax.axvspan(0, trained_m, color=INK3, alpha=0.10, lw=0, zorder=1)
    ax.text(trained_m / 2, 0.72, "학습에 쓴 구간", ha="center", va="center",
            fontsize=8, color=INK3, zorder=5, transform=ax.get_xaxis_transform())

    ax.fill_between(d, n_ls, b_ls, where=n_ls <= b_ls, color=C["LS"], alpha=0.13, lw=0, zorder=2)
    ax.plot(d, b_ls, color=INK3, lw=1.4, ls="--", zorder=3, label="LS  릴리즈 모델")
    ax.plot(d, n_ls, color=C["LS"], lw=2.0, zorder=4, label="LS  학습 후 (게이트 6b)")
    ax.plot(d, b_fd, color=C["FD"], lw=1.0, alpha=0.5, zorder=3, label="FD  fresh 실행 (릴리즈)")
    ax.plot(d, n_fd, color=C["FD"], lw=1.4, zorder=4, label="FD  fresh 실행 (학습 후)")

    ax.set_xlabel("앵커로부터의 이동 거리 (m)")
    ax.set_ylabel("궤적 오차 ATE (m)")
    ax.set_xlim(0, d.max() * 1.01)
    ax.set_ylim(0, max(b_ls.max(), n_ls.max()) * 1.06)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2, ncol=2)
    leg.set_zorder(6)
    ax.set_title("오염된 배포 경로(LS)는 전 구간에서 개선됐고, fresh 실행(FD)은 거의 변하지 않는다",
                 fontsize=9.5, color=INK, loc="left", pad=10)
    fig.tight_layout()
    out_a = os.path.join(R, "fig_training_ate.png")
    fig.savefig(out_a, facecolor=SURFACE)
    plt.close(fig)
    print(f"[saved] {out_a}")

    # ── B ────────────────────────────────────────────────────────────────────
    # Paired medians, not medians-of-values: the windows are identical across runs,
    # so the paired statistic is the one that answers "did this window get better".
    # (Medians-of-values put the three settings in a different order and overstate
    # the LS cost -- 0.191 m instead of 0.094 m for 1e-2.)
    pts = [("1e-3", "ate_vs_distance_6b.json"),
           ("3e-3", "ate_vs_distance_l2sp3e3.json"),
           ("1e-2", "ate_vs_distance_l2sp1e2.json")]

    rng = np.random.default_rng(0)

    def boot_ci(vals, block=4, n=4000):
        """Block bootstrap CI on the median; blocks absorb window-to-window
        correlation along the route."""
        v = np.asarray(vals)
        N, nb = len(v), len(v) // block
        starts = rng.integers(0, N - block + 1, size=(n, nb))
        idx = (starts[..., None] + np.arange(block)).reshape(n, -1)
        m = np.median(v[idx], axis=1)
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    # Marginal per-setting CIs overlap heavily and would understate the result.
    # What was actually tested is the PAIRED contrast against 1e-3, window by
    # window, so that is what gets drawn -- a forest plot with a zero line.
    ref = rows(dict(pts)["1e-3"])
    entries = []
    for name in ("3e-3", "1e-2"):
        r = rows(dict(pts)[name])
        d_ls = [ref[t]["ls_ate"] - r[t]["ls_ate"] for t in t0s]   # + : 오염 경로가 더 좋음
        d_fd = [ref[t]["fd_ate"] - r[t]["fd_ate"] for t in t0s]   # + : fresh 퇴행이 더 적음
        entries.append((f"L2-SP {name}", "fresh 퇴행 감소", d_fd, C["FD"]))
        entries.append((f"L2-SP {name}", "오염 경로 개선", d_ls, C["LS"]))

    fig, ax = plt.subplots(figsize=(7.6, 3.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    from math import comb

    def sign_p(vals):
        """Two-sided sign test.  Reported alongside the CI because the CI lower
        bound for one contrast sits at +1e-5 m -- 'excludes zero' is true there
        but says nothing on its own."""
        n = len(vals); k = sum(1 for x in vals if x > 0)
        tail = min(sum(comb(n, i) for i in range(k, n + 1)),
                   sum(comb(n, i) for i in range(0, k + 1)))
        return min(2 * tail / 2 ** n, 1.0)

    ax.axvline(0, color=INK2, lw=1.1, zorder=3)
    ys = list(range(len(entries)))[::-1]
    for y, (grp, metric, vals, col) in zip(ys, entries):
        m = np.median(vals)
        lo, hi = boot_ci(vals)
        p = sign_p(vals)
        strong = (lo > 0 or hi < 0) and p < 0.0125      # Bonferroni, 4 contrasts
        weak = (lo > 0 or hi < 0) and not strong
        note = (f"p={p:.3f}  경계" if weak else
                f"p={p:.3f}" if strong else "0 포함 — 구분 안 됨")
        ax.plot([lo, hi], [y, y], color=col, lw=2.0, alpha=.8 if (strong or weak) else .35,
                zorder=4, solid_capstyle="butt")
        ax.scatter([m], [y], s=58, color=col, zorder=5, edgecolor=SURFACE, linewidth=1.4,
                   alpha=1.0 if (strong or weak) else .5)
        ax.text(hi + 0.006, y, note, va="center", fontsize=8.5,
                color=INK2 if (strong or weak) else INK3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{g}  ·  {mt}" for g, mt, _, _ in entries], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.set_xlim(-0.33, 0.16)
    ax.set_xlabel("L2-SP 1e-3 대비 차이 (m) — 오른쪽이 더 좋음")
    ax.grid(axis="y", visible=False)
    ax.set_title("세 설정 중 무엇이 낫다고 말할 근거가 아직 없다\n"
                 "fresh 퇴행 감소는 방향만 일관될 뿐 경계선이고, 개선 손실은 아예 구분되지 않는다",
                 fontsize=9.5, color=INK, loc="left", pad=10)
    note = ("막대 = 블록 부트스트랩 95% CI (창별 짝지은 차) · p = 부호검정 양측 · "
            "비교 4건이므로 다중비교 보정(p<0.0125) 시 어느 것도 유의하지 않다")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.text(0.012, 0.022, note, fontsize=7.8, color=INK3, ha="left", va="bottom")
    out_b = os.path.join(R, "fig_l2sp_tradeoff.png")
    fig.savefig(out_b, facecolor=SURFACE)
    plt.close(fig)
    print(f"[saved] {out_b}")


if __name__ == "__main__":
    main()


def generalization():
    """C  does it hold on a scene the training never saw, and is the fresh-run
    regression real or a training-scene artefact?

    Two panels because the two questions have opposite polarity: on the left more
    is better, on the right zero is the good answer.  Same paired-contrast
    statistics as panel B -- the windows are identical within a scene.
    """
    from math import comb

    def sign_p(v):
        n = len(v); k = sum(1 for x in v if x > 0)
        t = min(sum(comb(n, i) for i in range(k, n + 1)),
                sum(comb(n, i) for i in range(0, k + 1)))
        return min(2 * t / 2 ** n, 1.0)

    rng2 = np.random.default_rng(0)

    def ci(vals, block=4, n=20000):
        v = np.asarray(vals); N = len(v); nb = max(N // block, 1)
        s = rng2.integers(0, N - block + 1, size=(n, nb))
        idx = (s[..., None] + np.arange(block)).reshape(n, -1)
        m = np.median(v[idx], axis=1)
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    scenes = [("kth_day_09\n미학습 장면", "ate_kth09_baseline.json", "ate_kth09_6b.json"),
              ("kth_day_06\n학습 장면", "ate_vs_distance.json", "ate_vs_distance_6b.json")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), dpi=200,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)

    for ax, (field, col, title, xlab) in zip(axes, [
            ("ls_ate", C["LS"], "오염 경로 개선 — 클수록 좋음", "LS ATE 개선 (m)"),
            ("fd_ate", C["FD"], "fresh 실행 변화 — 0이면 무변화", "FD ATE 변화 (m)")]):
        style(ax); ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK2, lw=1.1, zorder=3)
        labs = []
        for y, (name, bf, nf) in zip([1, 0], scenes):
            B, N = rows(bf), rows(nf)
            t = sorted(B)
            # LS: baseline - trained (positive = better).  FD: trained - baseline
            # (positive = worse), so each panel reads in its own natural direction.
            v = ([B[x][field] - N[x][field] for x in t] if field == "ls_ate"
                 else [N[x][field] - B[x][field] for x in t])
            lo, hi = ci(v); p = sign_p(v)
            good = sum(1 for x in v if x > 0)
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=col, lw=2.2, alpha=.85 if sig else .35,
                    zorder=4, solid_capstyle="butt")
            ax.scatter([np.median(v)], [y], s=64, color=col, zorder=5,
                       edgecolor=SURFACE, linewidth=1.4, alpha=1 if sig else .5)
            tag = (f"p<0.0001" if p < 1e-4 else f"p={p:.3f}") + ("" if sig else "  구분 안 됨")
            ax.text(hi + (hi - lo) * 0.06 + 0.004, y, f"{tag}   {good}/{len(v)}창",
                    va="center", fontsize=8.2, color=INK if sig else INK3)
            labs.append(name)
        ax.set_yticks([1, 0]); ax.set_yticklabels(labs, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)

    axes[0].set_xlim(-0.05, 1.35)
    axes[1].set_xlim(-0.035, 0.115)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.012, 0.02,
             "막대 = 블록 부트스트랩 95% CI (창별 짝지은 차) · p = 부호검정 양측 · "
             "개선은 두 장면에서 같은 크기로 나타나고, fresh 퇴행은 학습 장면에서만 나타난다",
             fontsize=7.8, color=INK3, ha="left", va="bottom")
    out = os.path.join(R, "fig_generalization.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    generalization()
