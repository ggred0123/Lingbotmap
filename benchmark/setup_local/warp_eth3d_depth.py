#!/usr/bin/env python3
"""Warp ETH3D ground-truth depth onto the undistorted (pinhole) image grid.

ETH3D ships depth maps rendered against the **original, distorted** images
(6048x4032, THIN_PRISM_FISHEYE), while the undistorted images it also ships are
pinhole and a different size (e.g. 6205x4135).  benchmark/datasets/eth3d.py
reshapes the depth file with the RGB image's shape, so the two must live on the
same grid -- hence this warp, which is what makes the 'custom_undistorted'
layout custom.

For every pixel of the undistorted image we take its viewing ray, project that
ray into the distorted image with ETH3D's documented THIN_PRISM_FISHEYE model,
and read the depth there.  Both images are the same camera at the same instant,
so the sampled value describes the same surface point; only the pixel grid
changes.  The map depends solely on the camera pair, so it is built once per
camera and reused across that camera's frames.

Depth convention (z along the optical axis vs. Euclidean ray length) is not
stated in the ETH3D docs, so `--check` measures it against the sparse COLMAP
points: for observations whose 3D position is known, it compares the stored
depth to both candidates and reports which one matches.  `--convention` then
applies the answer (z transfers unchanged; ray length is divided by the ray
norm to become z).

Usage:
    python setup_local/warp_eth3d_depth.py --check                 # decide the convention
    python setup_local/warp_eth3d_depth.py --convention z          # write the depth maps
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

DEFAULT_RAW = Path('/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/data_bench/eth3d_raw')
DEFAULT_OUT = Path('/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/data_bench/eth3d')

CALIB_D = 'dslr_calibration_jpg'            # distorted:  THIN_PRISM_FISHEYE
CALIB_U = 'dslr_calibration_undistorted'    # undistorted: PINHOLE


# --------------------------------------------------------------------------
# COLMAP-style text parsing
# --------------------------------------------------------------------------

def read_cameras(path: Path):
    """-> {camera_id: (model, width, height, [params])}"""
    cams = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        p = line.split()
        cams[int(p[0])] = (p[1], int(p[2]), int(p[3]), [float(x) for x in p[4:]])
    return cams


def read_images(path: Path, with_points=False):
    """-> {stem: dict(camera_id, qvec, tvec, points2d)}

    Pose lines are identified by their last field being an image path, so an
    observation line that happens to be empty cannot shift the parity.
    """
    entries = {}
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        p = line.split()
        if not p or line.startswith('#') or len(p) < 10:
            continue
        if not p[-1].lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = Path(p[-1]).stem
        rec = {
            'camera_id': int(p[8]),
            'qvec': np.array([float(x) for x in p[1:5]], dtype=np.float64),
            'tvec': np.array([float(x) for x in p[5:8]], dtype=np.float64),
        }
        if with_points and i + 1 < len(lines):
            vals = lines[i + 1].split()
            if vals and len(vals) % 3 == 0:
                arr = np.array(vals, dtype=np.float64).reshape(-1, 3)
                rec['points2d'] = arr                    # x, y, point3d_id
        entries[stem] = rec
    return entries


def read_points3d(path: Path):
    """-> {point3d_id: xyz}"""
    pts = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        p = line.split()
        pts[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return pts


def qvec_to_rotmat(q):
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


# --------------------------------------------------------------------------
# THIN_PRISM_FISHEYE projection (https://www.eth3d.net/documentation)
# --------------------------------------------------------------------------

def project_thin_prism_fisheye(x, y, params):
    """Project normalized image-plane coords (x/z, y/z) to distorted pixels.

    params: fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1
    """
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params

    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    # theta/r -> 1 as r -> 0; guard the singularity at the optical axis.
    scale = np.where(r > 1e-12, theta / np.maximum(r, 1e-12), 1.0)
    ud = x * scale
    vd = y * scale

    th2 = theta * theta
    tr = 1.0 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4)))
    un = ud * tr + 2.0 * p1 * ud * vd + p2 * (th2 + 2.0 * ud * ud) + sx1 * th2
    vn = vd * tr + 2.0 * p2 * ud * vd + p1 * (th2 + 2.0 * vd * vd) + sy1 * th2

    return fx * un + cx, fy * vn + cy


def build_map(cam_u, cam_d):
    """Undistorted pixel grid -> distorted pixel coordinates (map_x, map_y)."""
    _model_u, w_u, h_u, (fx, fy, cx, cy) = cam_u
    _model_d, w_d, h_d, params_d = cam_d

    # (0,0) is the image's top-left corner, so pixel centres sit at +0.5.
    us = (np.arange(w_u, dtype=np.float64) + 0.5 - cx) / fx
    vs = (np.arange(h_u, dtype=np.float64) + 0.5 - cy) / fy
    x = np.broadcast_to(us[None, :], (h_u, w_u))
    y = np.broadcast_to(vs[:, None], (h_u, w_u))

    map_x, map_y = project_thin_prism_fisheye(x, y, params_d)

    # Ray norm, so a Euclidean-distance depth can be converted to z later.
    ray_norm = np.sqrt(x * x + y * y + 1.0).astype(np.float32)

    outside = (map_x < 0) | (map_x >= w_d) | (map_y < 0) | (map_y >= h_d)
    # ETH3D pixel coordinates are corner-based ((0,0) = image corner) while
    # cv2.remap reads them as centre-based, so shift by half a pixel: with
    # INTER_NEAREST, round(c - 0.5) == floor(c), the pixel the coordinate is in.
    map_x = np.where(outside, -2.0, map_x - 0.5).astype(np.float32)
    map_y = np.where(outside, -2.0, map_y - 0.5).astype(np.float32)
    return map_x, map_y, ray_norm


def load_depth(path: Path, h: int, w: int):
    d = np.fromfile(str(path), dtype=np.float32).reshape(h, w)
    return np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)


# --------------------------------------------------------------------------
# Convention check against the sparse COLMAP points
# --------------------------------------------------------------------------

def check_convention(raw: Path, scenes, max_images=3):
    """Compare stored depth to z and to ray length at known 3D observations."""
    print('scene / image            n_obs      median |Δ| vs z    median |Δ| vs dist')
    for scene in scenes:
        calib = raw / scene / CALIB_D
        if not calib.is_dir():
            print(f'{scene}: no {CALIB_D} (download multi_view_training_dslr_jpg.7z)')
            continue
        cams = read_cameras(calib / 'cameras.txt')
        imgs = read_images(calib / 'images.txt', with_points=True)
        pts3d = read_points3d(calib / 'points3D.txt')

        for stem, rec in list(imgs.items())[:max_images]:
            depth_path = raw / scene / 'ground_truth_depth' / 'dslr_images' / f'{stem}.JPG'
            if not depth_path.exists() or 'points2d' not in rec:
                continue
            _m, w, h, _p = cams[rec['camera_id']]
            depth = load_depth(depth_path, h, w)

            R = qvec_to_rotmat(rec['qvec'])
            t = rec['tvec']

            obs = rec['points2d']
            obs = obs[obs[:, 2] >= 0]
            if len(obs) == 0:
                continue

            xy = obs[:, :2]
            ids = obs[:, 2].astype(np.int64)
            keep = np.array([i in pts3d for i in ids])
            xy, ids = xy[keep], ids[keep]
            if len(ids) == 0:
                continue

            X = np.stack([pts3d[i] for i in ids])           # world
            P = X @ R.T + t                                  # camera frame
            z = P[:, 2]
            dist = np.linalg.norm(P, axis=1)

            col = np.clip(xy[:, 0].astype(int), 0, w - 1)
            row = np.clip(xy[:, 1].astype(int), 0, h - 1)
            d = depth[row, col]

            valid = d > 0
            if valid.sum() < 10:
                continue
            e_z = np.median(np.abs(d[valid] - z[valid]))
            e_r = np.median(np.abs(d[valid] - dist[valid]))
            print(f'{scene}/{stem:<12} {valid.sum():>8}   {e_z:>16.4f}   {e_r:>18.4f}')


# --------------------------------------------------------------------------
# Warp
# --------------------------------------------------------------------------

def warp_scene(raw: Path, out: Path, scene: str, convention: str, force: bool):
    calib_d, calib_u = raw / scene / CALIB_D, raw / scene / CALIB_U
    cams_d, cams_u = read_cameras(calib_d / 'cameras.txt'), read_cameras(calib_u / 'cameras.txt')
    imgs_d, imgs_u = read_images(calib_d / 'images.txt'), read_images(calib_u / 'images.txt')

    src_dir = raw / scene / 'ground_truth_depth' / 'dslr_images'
    dst_dir = out / scene / 'ground_truth_depth' / 'custom_undistorted'
    if dst_dir.is_symlink():            # convert_eth3d.py's provisional symlink
        dst_dir.unlink()
    dst_dir.mkdir(parents=True, exist_ok=True)

    maps = {}
    written = skipped = 0
    for stem, rec_u in imgs_u.items():
        src = src_dir / f'{stem}.JPG'
        dst = dst_dir / f'{stem}.JPG'
        if not src.exists() or stem not in imgs_d:
            skipped += 1
            continue
        if dst.exists() and not force:
            continue

        key = (rec_u['camera_id'], imgs_d[stem]['camera_id'])
        if key not in maps:
            maps[key] = build_map(cams_u[key[0]], cams_d[key[1]])
        map_x, map_y, ray_norm = maps[key]

        _m, w_d, h_d, _p = cams_d[key[1]]
        depth = load_depth(src, h_d, w_d)
        warped = cv2.remap(depth, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        if convention == 'dist':
            warped = warped / ray_norm            # ray length -> z
        warped.astype(np.float32).tofile(str(dst))
        written += 1

    return written, skipped, len(maps)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--check', action='store_true',
                    help='report which depth convention the data uses, then exit')
    ap.add_argument('--convention', choices=['z', 'dist'], default='z',
                    help="what the stored depth means: 'z' along the optical axis "
                         "(transferred unchanged) or 'dist' ray length (converted to z)")
    ap.add_argument('--scenes', nargs='*', help='limit to these scenes')
    ap.add_argument('--force', action='store_true', help='rewrite existing depth files')
    args = ap.parse_args()

    scenes = args.scenes or sorted(
        d.name for d in args.raw.iterdir()
        if d.is_dir() and (d / 'ground_truth_depth').is_dir()
    )

    if args.check:
        check_convention(args.raw, scenes)
        return

    for scene in scenes:
        if not (args.raw / scene / CALIB_D).is_dir():
            print(f'{scene}: SKIP (no {CALIB_D})')
            continue
        written, skipped, n_maps = warp_scene(args.raw, args.out, scene,
                                              args.convention, args.force)
        print(f'{scene}: wrote {written} depth maps '
              f'({n_maps} camera map(s), {skipped} without source depth)', flush=True)


if __name__ == '__main__':
    main()
