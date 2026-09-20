#!/usr/bin/env python3
"""Repack official ETH3D archives into the 'custom_undistorted' layout.

benchmark/datasets/eth3d.py reads a pi3-flavoured variant of ETH3D:

    {root}/{scene}/images/custom_undistorted/{name}.JPG
    {root}/{scene}/ground_truth_depth/custom_undistorted/{name}.JPG   (raw float32)
    {root}/{scene}/custom_undistorted_cam/{name}.npz                  (intrinsics, extrinsics)

The official download gives the same pixels under different names, plus COLMAP
text calibration instead of per-frame npz:

    {raw}/{scene}/images/dslr_images_undistorted/{name}.JPG
    {raw}/{scene}/dslr_calibration_undistorted/{cameras.txt,images.txt}

So this script symlinks the image tree under its expected name and converts
cameras.txt + images.txt into one npz per frame.  Images are left where they are
(they are ~6 GB); only names are added.

This handles images and calibration only.  Depth needs resampling onto the
undistorted grid, which is what setup_local/warp_eth3d_depth.py does — run it
next, or the adapter will not find any depth map.

Usage:
    python setup_local/convert_eth3d.py [--raw DIR] [--out DIR] [--copy]
    python setup_local/warp_eth3d_depth.py --convention z
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

DEFAULT_RAW = Path('/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/data_bench/eth3d_raw')
DEFAULT_OUT = Path('/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/data_bench/eth3d')

# Names the official archives use, in the order we prefer them.
IMAGE_SUBDIRS = ('dslr_images_undistorted', 'dslr_images')
CALIB_SUBDIR = 'dslr_calibration_undistorted'


def qvec_to_rotmat(qw, qx, qy, qz):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def read_cameras(path: Path):
    """Parse COLMAP cameras.txt -> {camera_id: 3x3 K}."""
    cams = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        cam_id, model = int(parts[0]), parts[1]
        params = [float(x) for x in parts[4:]]
        if model == 'PINHOLE':
            fx, fy, cx, cy = params[:4]
        elif model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            raise ValueError(
                f'{path}: camera model {model!r} is not pinhole -- these images '
                'were expected to be the undistorted set'
            )
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        cams[cam_id] = K
    return cams


def read_images(path: Path):
    """Parse COLMAP images.txt -> [(name, camera_id, 4x4 W2C)].

    Every image entry is two lines: the pose line, then its 2D points line.
    Rather than assuming strict alternation (a view with no observations would
    write an empty second line and shift the parity), pose lines are recognised
    by their last field being an image path.
    """
    entries = []
    for line in path.read_text().splitlines():
        p = line.split()
        if not p or line.startswith('#') or len(p) < 10:
            continue
        if not p[-1].lower().endswith(('.jpg', '.jpeg', '.png')):
            continue                                  # POINTS2D line
        qw, qx, qy, qz = (float(x) for x in p[1:5])
        tx, ty, tz = (float(x) for x in p[5:8])
        cam_id = int(p[8])
        name = ' '.join(p[9:])                    # e.g. dslr_images_undistorted/DSC_0286.JPG
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec_to_rotmat(qw, qx, qy, qz)
        w2c[:3, 3] = (tx, ty, tz)
        entries.append((name, cam_id, w2c))
    return entries


def link_dir(src: Path, dst: Path, copy: bool):
    """Expose src under the name dst (symlink by default)."""
    if dst.is_symlink() or dst.exists():
        return 'exists'
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
        return 'copied'
    dst.symlink_to(src, target_is_directory=True)
    return 'linked'


def find_subdir(parent: Path):
    """Return the first of IMAGE_SUBDIRS that exists under parent."""
    for name in IMAGE_SUBDIRS:
        if (parent / name).is_dir():
            return parent / name
    return None


def convert_scene(scene_dir: Path, out_root: Path, copy: bool) -> str:
    scene = scene_dir.name
    out_dir = out_root / scene

    images_src = find_subdir(scene_dir / 'images')
    if images_src is None:
        return f'{scene}: SKIP (no images/{"|".join(IMAGE_SUBDIRS)})'
    link_dir(images_src, out_dir / 'images' / 'custom_undistorted', copy)

    # Depth is deliberately NOT linked here: the official maps are rendered
    # against the distorted images (6048x4032) while these images are pinhole
    # and a different size, so the adapter's reshape would fail.  Linking them
    # would look done and crash later; warp_eth3d_depth.py writes the real ones.
    depth_src = find_subdir(scene_dir / 'ground_truth_depth')
    depth_note = 'depth: run warp_eth3d_depth.py' if depth_src else 'no depth available'

    calib = scene_dir / CALIB_SUBDIR
    cams = read_cameras(calib / 'cameras.txt')
    entries = read_images(calib / 'images.txt')

    cam_out = out_dir / 'custom_undistorted_cam'
    cam_out.mkdir(parents=True, exist_ok=True)
    for name, cam_id, w2c in entries:
        stem = Path(name).stem
        np.savez(cam_out / f'{stem}.npz',
                 intrinsics=cams[cam_id].astype(np.float32),
                 extrinsics=w2c.astype(np.float64))

    return f'{scene}: {len(entries)} frames, {depth_note}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, default=DEFAULT_RAW,
                    help='where download_eth3d.sh extracted the archives')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help='raw_data_root for configs/datasets/eth3d.yaml')
    ap.add_argument('--copy', action='store_true',
                    help='copy image trees instead of symlinking (~6 GB extra)')
    args = ap.parse_args()

    if not args.raw.is_dir():
        raise SystemExit(f'raw dir not found: {args.raw} (run setup_local/download_eth3d.sh first)')

    scenes = sorted(d for d in args.raw.iterdir()
                    if d.is_dir() and (d / CALIB_SUBDIR).is_dir())
    if not scenes:
        raise SystemExit(f'no scenes with {CALIB_SUBDIR}/ under {args.raw}')

    args.out.mkdir(parents=True, exist_ok=True)
    for scene_dir in scenes:
        print(convert_scene(scene_dir, args.out, args.copy), flush=True)
    print(f'\nwrote {args.out}  ({len(scenes)} scenes)')
    print('next: python setup_local/warp_eth3d_depth.py --convention z')


if __name__ == '__main__':
    main()
