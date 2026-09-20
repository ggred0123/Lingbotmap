"""Extract an image window from an MCD camera rosbag + the matching GT poses.

MCD ships ROS1 bags; ``rosbags`` reads them without a ROS install.  Images are
written as ``<idx:06d>_<sec>_<nsec>.png`` so the model's frame order and the GT
timestamps stay tied together, and a ``meta.npz`` carries the timestamps plus the
spline-evaluated ground-truth body poses for exactly those frames.

  python mcd_extract.py --bag .../kth_day_06_d455b.bag --topic /d455b/color/image_raw \
      --gt .../mcd/gt --start 9000 --count 400 --stride 3 --out .../frames_win0
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcd_gt import McdSplineGT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", default="/d455b/color/image_raw")
    ap.add_argument("--gt", required=True, help="dir with spline.csv and pose_inW.csv")
    ap.add_argument("--start", type=int, default=0, help="first message index")
    ap.add_argument("--count", type=int, default=400, help="frames to write")
    ap.add_argument("--stride", type=int, default=3, help="30Hz/stride (3 -> 10Hz, matching GT)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ts_store = get_typestore(Stores.ROS1_NOETIC)

    stamps, names = [], []
    with Reader(args.bag) as r:
        con = [c for c in r.connections if c.topic == args.topic]
        if not con:
            raise SystemExit(f"topic {args.topic} not in bag")
        want_end = args.start + args.count * args.stride
        for i, (c, _t, raw) in enumerate(r.messages(connections=con)):
            if i < args.start:
                continue
            if i >= want_end or len(names) >= args.count:
                break
            if (i - args.start) % args.stride:
                continue
            m = ts_store.deserialize_ros1(raw, c.msgtype)
            arr = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, -1)
            if m.encoding == "bgr8":
                arr = arr[..., ::-1]
            elif m.encoding == "mono8":
                arr = np.repeat(arr, 3, axis=2)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            name = f"{len(names):06d}_{m.header.stamp.sec}_{m.header.stamp.nanosec:09d}.png"
            Image.fromarray(np.ascontiguousarray(arr)).save(os.path.join(args.out, name))
            stamps.append(t)
            names.append(name)

    stamps = np.array(stamps)
    print(f"[extract] {len(stamps)} frames  span {stamps[-1]-stamps[0]:.1f}s  "
          f"dt {np.diff(stamps).mean()*1000:.1f}ms")

    gt = McdSplineGT(os.path.join(args.gt, "spline.csv"))
    gt.self_test(os.path.join(args.gt, "pose_inW.csv"), n=2000)
    inside = (stamps > gt.t0) & (stamps < gt.t1)
    if not inside.all():
        print(f"[warn] {(~inside).sum()} frames fall outside the GT span; they are dropped")
    pos, quat = gt.eval(stamps[inside])

    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    print(f"[gt] {inside.sum()} poses  path {d.sum():.1f} m  "
          f"step mean {d.mean()*100:.1f} cm  speed {d.sum()/(stamps[inside][-1]-stamps[inside][0]):.2f} m/s")

    np.savez(os.path.join(args.out, "meta.npz"),
             names=np.array(names)[inside], stamps=stamps[inside],
             gt_pos=pos, gt_quat=quat, topic=args.topic, bag=args.bag)
    print(f"[saved] {args.out}/meta.npz")


if __name__ == "__main__":
    main()
