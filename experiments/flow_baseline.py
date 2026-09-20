"""Visual keyframe baseline: median sparse optical flow between frames i and i+K.

experiments/kf_baseline.py measures the same thing in METRES, but that needs GT
poses.  SlowTV is monocular YouTube and never will have them, and neither will
most of the candidate corpora -- yet the mixture bug they cause is real:
K_DIST_V5 attaches one K label to every dataset, while the visual displacement
that label stands for is a property of the corpus.  Pixels at the model's own
input width are the common axis, and they are what the network actually sees.

    experiments/flow_baseline.py name=/path/to/frames[:stride] ...

`stride` is the loader's frame stride (Oxford's dataset config uses 12), so the
reported K matches the K the benchmark passes to the model, not raw frame count.
"""
import sys
import glob
import numpy as np
import cv2

KS = (1, 4, 8, 12, 28)
WIDTH = 518          # the model's input width -- keep the units it sees
N_PAIRS = 40


def med_flow(files, step, starts, W=WIDTH):
    """`starts` is shared across every K so the K columns compare like with like."""
    if len(files) < step + 8:
        return None
    out = []
    for i in starts:
        if i + step >= len(files):
            continue
        a = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(files[i + step], cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        s = W / a.shape[1]
        a = cv2.resize(a, (W, int(a.shape[0] * s)))
        b = cv2.resize(b, (W, int(b.shape[0] * s)))
        p0 = cv2.goodFeaturesToTrack(a, maxCorners=300, qualityLevel=0.01, minDistance=7)
        if p0 is None or len(p0) < 20:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=4)
        if p1 is None:
            continue
        m = st.ravel() == 1
        if m.sum() < 20:
            continue
        out.append(np.median(np.linalg.norm((p1 - p0)[m].reshape(-1, 2), axis=1)))
    if not out:
        return None
    q1, q3 = np.percentile(out, [25, 75])
    return float(np.median(out)), float(q1), float(q3), len(out)


def listing(d):
    fs = []
    for e in ("png", "jpg", "jpeg", "PNG", "JPG"):
        fs += glob.glob(f"{d}/*.{e}")
    return sorted(fs)


print(f"{'corpus':<18}{'stride':>7}" + "".join(f"{'K='+str(k):>8}" for k in KS), flush=True)
for arg in sys.argv[1:]:
    name, _, spec = arg.partition("=")
    path, _, sd = spec.partition(":")
    stride = int(sd) if sd else 1
    files = listing(path)
    if not files:
        print(f"{name:<18}{stride:>7}   no frames", flush=True)
        continue
    # one shared start set, sized for the largest step so every K uses it
    span = len(files) - stride * max(KS) - 1
    if span < 1:
        print(f"{name:<18}{stride:>7}   too short ({len(files)} frames)", flush=True)
        continue
    starts = np.linspace(0, span, N_PAIRS).astype(int)
    cells = []
    for K in KS:
        r = med_flow(files, stride * K, starts)
        cells.append(f"{r[0]:8.1f}" if r else "      --")
    r28 = med_flow(files, stride * KS[-1], starts)
    iqr = f"  [K=28 IQR {r28[1]:.0f}-{r28[2]:.0f}, n={r28[3]}]" if r28 else ""
    print(f"{name:<18}{stride:>7}" + "".join(cells) + iqr, flush=True)
