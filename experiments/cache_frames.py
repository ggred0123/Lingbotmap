"""Pre-render a scene's preprocessed frames to a float32 memmap.

WHY: the trainer holds every scene's images resident as one float32 tensor --
143 GB for the 10-scene corpus, 216 GB RSS per process in practice.  Two
concurrent runs then exceed what the container will give us and one of them dies
silently, mid-setup, with no traceback (observed three times).

But a training step only ever touches ~96 frames of ONE scene: the supervised
window [t, t+S) and the advance [t, next_t).  The rest is resident for nothing.

Writing the SAME float32 tensor to disk and mapping it back makes those pages
reclaimable and, because both runs map the same files, shared between processes.
Numerically it is byte-identical to what the trainer computes today -- no
quantisation, no dtype change -- which matters here because the repo's
equivalence tests are bit-exact.

    python experiments/cache_frames.py kth_day_10 tuhh_day_03 ...
    python experiments/cache_frames.py --all

The trainer picks the cache up automatically when it exists.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.train.label_bank import image_names

TRAIN_SCENES = ["kth_day_10", "kth_night_01", "kth_night_04", "kth_night_05",
                "tuhh_day_02", "tuhh_day_03", "tuhh_day_04",
                "tuhh_night_07", "tuhh_night_08", "tuhh_night_09"]


def cache_path(frames_dir: str, image_size: int, patch_size: int) -> str:
    return os.path.join(frames_dir, f"_cache_{image_size}_{patch_size}.npy")


def build(frames_dir: str, n_frames: int, image_size: int, patch_size: int,
          chunk: int = 512, uint8: bool = False) -> str:
    """Write [n_frames, 3, H, W] float32 (or uint8, see --uint8).  Chunked so
    peak RSS stays ~chunk frames."""
    out = cache_path(frames_dir, image_size, patch_size)
    names = image_names(frames_dir)[:n_frames]
    if os.path.exists(out):
        mm = np.load(out, mmap_mode="r")
        if mm.shape[0] >= len(names):
            print(f"  [skip] {out} already has {mm.shape[0]} >= {len(names)} frames")
            return out
        del mm

    probe = load_and_preprocess_images([os.path.join(frames_dir, names[0])],
                                       mode="crop", image_size=image_size,
                                       patch_size=patch_size)
    _, C, H, W = probe.shape
    tmp = out + ".partial"
    # ★ uint8 IS A DIFFERENT CONTRACT from the float32 cache above: the values
    # are round(x * 255), so a pixel can move by up to 0.5/255 against what the
    # trainer computes from the PNG.  It is 4x smaller (3.7 GB against 14.6 GB
    # per 8000-frame scene), which is what makes a corpus of dozens of SlowTV
    # chunks fit on the shared Lustre (docs/coverage-corpus-plan.md §6 E2).
    # MemmapFrames divides by 255 on read.  Never use it for a scene that a
    # bit-exact equivalence test reads.
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8 if uint8 else np.float32,
                                   shape=(len(names), C, H, W))
    for i in range(0, len(names), chunk):
        part = names[i:i + chunk]
        arr = load_and_preprocess_images([os.path.join(frames_dir, n) for n in part],
                                         mode="crop", image_size=image_size,
                                         patch_size=patch_size)
        if uint8:
            mm[i:i + len(part)] = np.clip(np.rint(arr.numpy() * 255.0), 0, 255).astype(np.uint8)
        else:
            mm[i:i + len(part)] = arr.numpy()
        del arr
        print(f"  {i + len(part)}/{len(names)}", flush=True)
    mm.flush()
    del mm
    os.replace(tmp, out)                       # atomic: a partial file is never picked up
    print(f"  [done] {out}  {os.path.getsize(out) / 1e9:.1f} GB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", default="data/mcd")
    ap.add_argument("--bank_prefix", default="",
                    help="labels/<prefix><scene> holds the bank.  SlowTV banks "
                         "are named labels/slowtv_00000 while the frames live "
                         "in data/slow_tv/00000, so the two names differ.")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the cached frame count (0 = whatever the bank "
                         "covers).  A SlowTV sequence is up to 181k frames = "
                         "435 GB of cache; the bank is built over a span, so "
                         "cache the span and not the video.")
    ap.add_argument("--bank_suffix", default="",
                    help="labels/<prefix><scene><suffix>.  The re-bakes are named "
                         "labels/<scene>_L48, i.e. the variant is a SUFFIX, which "
                         "--bank_prefix cannot express.  A shorter L tiles its runs "
                         "differently and can end a few frames past the L=240 bank, "
                         "so the cache built for that one comes up short and the "
                         "trainer refuses to start.")
    ap.add_argument("--frames_subdir", default="frames_10hz",
                    help="frame directory inside <root>/<scene>.  MCD and SlowTV "
                         "both use frames_10hz; the corpora added for the v6 "
                         "mixture (ParallelDomain-4D, DL3DV, ScanNet, ...) ship "
                         "their frames in an 'images' directory instead.")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--uint8", action="store_true",
                    help="store round(x*255) as uint8 (4x smaller; MemmapFrames "
                         "divides by 255 on read).  Only for scenes that no "
                         "bit-exact test reads -- see build().")
    args = ap.parse_args()

    scenes = TRAIN_SCENES if args.all else args.scenes
    if not scenes:
        raise SystemExit("give scene names or --all")
    for s in scenes:
        d = os.path.join(args.root, s, args.frames_subdir)
        # Cache exactly what the trainer would load: the bank's covered range.
        import json
        idx = os.path.join("labels", f"{args.bank_prefix}{s}{args.bank_suffix}", "index.json")
        n = len(image_names(d))
        if os.path.exists(idx):
            runs = json.load(open(idx))["runs"]
            n = min(n, runs[-1]["t0"] + runs[-1]["L"])
        elif not args.limit:
            print(f"[{s}] WARNING no bank at {idx} -- caching all {n} frames "
                  f"({n * 2.4 / 1000:.0f} GB).  Build the bank first, or pass "
                  f"--limit, or this writes the whole video to disk.")
        if args.limit:
            n = min(n, args.limit)
        print(f"[{s}] {n} frames -> {d}")
        build(d, n, args.image_size, args.patch_size, uint8=args.uint8)


if __name__ == "__main__":
    main()
