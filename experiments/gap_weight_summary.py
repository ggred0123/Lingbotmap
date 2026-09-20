"""Aggregate gapw_*.json into the two tables the decision needs.

★ MEDIAN OVER WINDOWS, NOT MEAN.  Same rule the rest of this project settled on
(experiments-ledger sec 05): a scene-wise mean here is set by one outlier window,
and every headline number that was later withdrawn came from one.
"""
import glob, json, statistics as st, sys

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                         else "experiments/results/gapw_v7f_*.json"))
if not files:
    raise SystemExit("no gapw_*.json yet")
recs = []
for f in files:
    r = json.load(open(f))
    r.setdefault("scene", f.split("gapw_")[-1].rsplit("_t", 1)[0])
    recs.append(r)
print(f"{len(recs)} windows\n")

def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")

# ── band table: where does the rotation update point? ───────────────────────
print("  per-band gradient of the local rot term")
print(f"  {'window':<26}{'band':>8}{'pairs':>8}{'resid deg':>11}{'|g|':>11}"
      f"{'cos(rot0)':>11}{'cos(GT)':>10}{'cos(GT)nd':>11}")
bands = [r["band"] for r in recs[0]["band_rows"]]
for r in recs:
    tag = f"{r.get('scene','?')} t{r['t0']} K{r['K']}"
    for b in r["band_rows"]:
        print(f"  {tag:<26}{b['band']:>8}{b['pair_frac']:>8.1%}"
              f"{b['resid_deg']:>11.3f}{b['g_rot']:>11.3e}"
              f"{b['cos_band_vs_rot0']:>+11.4f}"
              f"{b.get('cos_band_gt', float('nan')):>+10.4f}"
              f"{b.get('cos_band_gt_nodepth', float('nan')):>+11.4f}")
        tag = ""

print(f"\n  band medians over {len(recs)} windows")
print(f"  {'band':>8}{'resid deg':>11}{'|g|':>11}{'cos(GT)':>10}{'cos(GT)nd':>11}"
      f"{'wins GT+':>10}")
for bn in bands:
    rows = [b for r in recs for b in r["band_rows"] if b["band"] == bn]
    pos = sum(1 for b in rows if b.get("cos_band_gt_nodepth", 0) > 0)
    print(f"  {bn:>8}{med([b['resid_deg'] for b in rows]):>11.3f}"
          f"{med([b['g_rot'] for b in rows]):>11.3e}"
          f"{med([b.get('cos_band_gt') for b in rows]):>+10.4f}"
          f"{med([b.get('cos_band_gt_nodepth') for b in rows]):>+11.4f}"
          f"{pos:>7}/{len(rows)}")

# ── gamma table: does the reweighted update point at GT more? ───────────────
gammas = [r["gamma"] for r in recs[0]["gamma_rows"]]
print(f"\n  gamma sweep -- change in cos(update, GT) against gamma=0")
print(f"  {'window':<26}" + "".join(f"{'g=' + f'{g:g}':>12}" for g in gammas))
for r in recs:
    base = next(x for x in r["gamma_rows"] if x["gamma"] == 0.0)
    tag = f"{r.get('scene','?')} t{r['t0']} K{r['K']}"
    print(f"  {tag:<26}" + "".join(
        f"{x['cos_local_gt_nodepth'] - base['cos_local_gt_nodepth']:>+12.4f}"
        for x in r["gamma_rows"]))

print(f"\n  medians over {len(recs)} windows")
print(f"  {'gamma':>6}{'mean gap':>10}{'cos(rot,GT)':>13}{'cos(rot,GT)nd':>15}"
      f"{'cos(loc,GT)nd':>15}{'d cos nd':>10}{'wins':>8}")
for gm in gammas:
    rows = [x for r in recs for x in r["gamma_rows"] if x["gamma"] == gm]
    d = []
    for r in recs:
        base = next(x for x in r["gamma_rows"] if x["gamma"] == 0.0)
        cur = next(x for x in r["gamma_rows"] if x["gamma"] == gm)
        d.append(cur["cos_local_gt_nodepth"] - base["cos_local_gt_nodepth"])
    print(f"  {gm:>6g}{med([x['mean_gap'] for x in rows]):>10.1f}"
          f"{med([x.get('cos_rot_gt') for x in rows]):>+13.4f}"
          f"{med([x.get('cos_rot_gt_nodepth') for x in rows]):>+15.4f}"
          f"{med([x.get('cos_local_gt_nodepth') for x in rows]):>+15.4f}"
          f"{med(d):>+10.4f}{sum(1 for x in d if x > 0):>5}/{len(d)}")

# ── the target audit ────────────────────────────────────────────────────────
if recs[0].get("gap_audit"):
    print(f"\n  target audit -- rel-rot error vs GT (deg), median over windows")
    print(f"  {'band':>8}{'GT turn':>10}{'teacher':>10}{'student':>10}"
          f"{'headroom':>10}{'teacher/turn':>14}")
    for bn in recs[0]["gap_audit"]["bands"]:
        v = [r["gap_audit"]["bands"][bn] for r in recs if r.get("gap_audit")]
        print(f"  {bn:>8}{med([x['gt_deg'] for x in v]):>10.3f}"
              f"{med([x['teacher_deg'] for x in v]):>10.3f}"
              f"{med([x['student_deg'] for x in v]):>10.3f}"
              f"{med([x['student_deg'] - x['teacher_deg'] for x in v]):>+10.3f}"
              f"{med([x['teacher_deg'] / max(1e-9, x['gt_deg']) for x in v]):>14.4f}")

print(f"\n  closure check (should be ~1e-7): "
      + " ".join(f"{r['band_closure_rel']:.1e}" for r in recs))
