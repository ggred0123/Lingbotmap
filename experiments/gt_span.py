"""Largest frame span of a scene on which every frame is GT-scorable.

A bank built to the raw PNG count can include frames that ``meta.npz`` has no GT
row for -- extraction keeps lead-in images whose pose spline has not started yet,
and trailing images past the last GT sample.  Training does not care (student
inputs and bank labels share one indexing), but the probe's GT scorer refuses a
window whose rows are missing or non-contiguous, by design: a silent skip there
is the section 5.5 failure mode -- it looks fine and measures the wrong thing.

So cap the bank at the contiguous GT-backed range and the probe can score any
window the bank hands it.  Prints ``LO HI`` for ``--span``.

    python experiments/gt_span.py data/mcd/kth_day_09/frames_10hz
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.train.label_bank import image_names


def gt_span(frames_dir: str):
    """(lo, hi, n_png, n_gt) -- the longest run of frames with consecutive GT rows."""
    names = image_names(frames_dir)
    meta_path = os.path.join(frames_dir, "meta.npz")
    if not os.path.exists(meta_path):
        return 0, len(names), len(names), 0          # no GT: fall back to all PNGs
    meta = np.load(meta_path, allow_pickle=True)
    row_of = {str(n): i for i, n in enumerate(meta["names"])}
    rows = np.array([row_of.get(n, -1) for n in names], dtype=np.int64)

    best = (0, 0)
    i = 0
    while i < len(rows):
        if rows[i] < 0:
            i += 1
            continue
        j = i + 1
        while j < len(rows) and rows[j] == rows[j - 1] + 1:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    return best[0], best[1], len(names), len(meta["names"])


if __name__ == "__main__":
    d = sys.argv[1]
    lo, hi, n_png, n_gt = gt_span(d)
    if "--verbose" in sys.argv:
        print(f"{os.path.basename(os.path.dirname(d))}: {n_png} png, {n_gt} gt rows "
              f"-> GT-contiguous span [{lo}, {hi}) = {hi - lo} frames "
              f"(dropping {lo} lead-in, {n_png - hi} trailing)", file=sys.stderr)
    print(lo, hi)
