#!/usr/bin/env python
"""Fill SlowTV per-sequence intrinsics, decoupled from the frame export.

Why this is a separate pass:

  Over 14 finished sequences, frame export and COLMAP each took ~half the wall
  clock (578 min vs 531 min).  But they are bound by disjoint resources --
  export saturates the node's single Lustre client while the CPU sits 82% idle;
  COLMAP is CPU-bound (~6 cores measured) and leaves Lustre idle.  Run serially
  in one pipeline, each stage waits on a resource the other is not using.

  COLMAP also reads only `n_imgs` frames (~96 MB) per sequence, so it is not
  tied to the node that did the export.  Hence: export writes frames, this
  fills intrinsics afterwards, many sequences at once, anywhere.

Idempotent: a sequence is 'done' when intrinsics.txt exists, so re-running only
picks up what is missing.  A stale/partial colmap/<seq> dir is recomputed rather
than skipped (estimate_intrinsics would otherwise treat its existence as done).

  experiments/slowtv_intrinsics.py                 # every sequence missing one
  experiments/slowtv_intrinsics.py --jobs 8
  experiments/slowtv_intrinsics.py --seqs 20 39    # inclusive index range
"""
import argparse, os, sys, time
from multiprocessing import Pool

REPO = os.environ.get('SLOWTV_REPO',
                      '/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/tools/slowtv_monodepth')
sys.path.insert(0, REPO)

import src.devkits.slow_tv as stv                    # noqa: E402
from src.paths import DATA_PATHS as PATHS            # noqa: E402

N_IMGS, INTERVAL = 200, 1          # match export_slow_tv.py's settings
SEEDS = [42, 195, 335, 558, 724]


def targets(lo, hi):
    root = PATHS['slow_tv']
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        if lo is not None and not (lo <= int(d.name) <= hi):
            continue
        if (d/'intrinsics.txt').exists():
            continue
        if not any(d.glob('*.png')):
            continue                                  # not exported yet
        out.append(d)
    return out


def run(seq_dir):
    seq = seq_dir.name
    colmap_dir = PATHS['slow_tv']/'colmap'
    stale = (colmap_dir/seq).is_dir()                 # partial from an earlier attempt
    t0 = time.time()
    for seed in SEEDS:
        try:
            stv.estimate_intrinsics(seq_dir, save_root=colmap_dir, n_imgs=N_IMGS,
                                    interval=INTERVAL, seed=seed, overwrite=stale)
            stale = True                              # retries must not skip either
            if (seq_dir/'intrinsics.txt').exists():
                return seq, 'ok', time.time()-t0, seed
        except Exception as e:
            print(f'  [{seq}] seed {seed} failed: {type(e).__name__}', flush=True)
            stale = True
    return seq, 'FAILED', time.time()-t0, None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=8,
                    help='concurrent sequences; COLMAP uses ~6 cores each')
    ap.add_argument('--seqs', type=int, nargs=2, metavar=('FROM', 'TO'), default=None)
    a = ap.parse_args()

    lo, hi = (a.seqs if a.seqs else (None, None))
    todo = targets(lo, hi)
    print(f'[intrinsics] {len(todo)} sequences missing intrinsics, {a.jobs} at a time')
    if not todo:
        sys.exit(0)

    ok = 0
    with Pool(a.jobs) as pool:
        for seq, status, secs, seed in pool.imap_unordered(run, todo):
            ok += status == 'ok'
            print(f'[intrinsics] {seq}: {status} in {secs/60:.1f} min'
                  + (f' (seed {seed})' if seed is not None else ''), flush=True)
    print(f'[intrinsics] done: {ok}/{len(todo)} succeeded')
