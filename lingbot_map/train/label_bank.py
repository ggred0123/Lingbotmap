"""T4 -- offline fresh-teacher label bank.

docs/phase1-plan.md §3-T4, docs/self-distill-ver4-fixed.md §3.1-3.2.

The teacher's output depends only on the observation window it was given, never
on the student's state.  So there is no reason to run a teacher inside the
training loop: generate the labels once, offline, and read them back.  (This is
where we are ahead of StreamVGGT, which calls its teacher live.)

★ THE BUDGET IS THE WHOLE POINT.  A fresh teacher is only a clean target while
its own cache stays inside the length it was trained for:

    scale_frames + B + L <= 320  keyframes        (v4 §3.2, measured)

    L=200 (inside)   teacher cache 280 kf   FD ATE rmse 0.223 m
    L=600 (outside)  teacher cache 680 kf   FD ATE rmse 2.652 m   <- 12x worse

Past the budget the "teacher" is just another degraded run.  So one fresh run
supervises a FINITE span -- ~240 frames ~= 38 m at 10 Hz with B=72 -- and a long
rollout has to be covered by many short runs.

★ AND EACH RUN HAS ITS OWN GAUGE.  Every §3.3 loss term is normalised inside the
window, so a supervised window that straddles two runs would be comparing two
different Sim(3) frames and the normalisation would silently absorb the seam.
``LabelBank.windows()`` therefore never returns a window that crosses a run
boundary.  Consistency *between* runs is supervised by nobody (v4 §3.3-3) --
that is a known blind spot, not something this module papers over.

Usage:
    python -m lingbot_map.train.label_bank --ckpt ... --frames data/.../frames_10hz \\
        --out labels/kth_day_06 --span 0 2000 --burn_in 72 --L 240
"""

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# v4 §3.2: the teacher's KV cache may hold at most this many keyframes before its
# own quality falls off the measured cliff.
TEACHER_CACHE_BUDGET = 320


def max_supervised_frames(burn_in: int, scale_frames: int = 8,
                          teacher_interval: int = 1,
                          budget: int = TEACHER_CACHE_BUDGET) -> int:
    """Longest span one fresh run may supervise, in FRAMES.

    The budget is counted in keyframes (v4 §3.1: "B는 키프레임 단위로 셀 것"), so
    a teacher interval K_t > 1 buys coverage linearly:

        L_frames <= (budget - scale_frames - B) * K_t

    K_t in {2, 4} is v4 §3.2's extension handle (38 m -> 76-152 m).  It is NOT
    validated: every FD measurement was taken at K_t = 1, so §6-Q5 asks for a
    one-off label-quality check per K_t before anyone relies on it.
    """
    return max(0, budget - scale_frames - burn_in) * max(teacher_interval, 1)


IMAGE_EXT = (".png", ".jpg", ".jpeg")


# ── frame ordering ──────────────────────────────────────────────────────────
# ★ A BARE sorted() IS NOT VIDEO ORDER, and for 65% of this project's sampling
# weight it was not even close.  experiments/corpus_order_audit.py measured, on
# the corpus every run before 2026-09-08 trained on:
#
#   paralleldomain4d  19 views of ONE instant adjacent (camera0-15, yaw-0/60/
#                     neg-60) -> 0.00% of adjacent pairs are a time step
#   unrealstereo4k    cam0/cam1 alternating every frame          -> 0.00%
#   dynamicreplica    300 left frames then 300 right frames      -> a teleport
#   scannet           "0, 1, 10, 100, 1000" lexicographic         -> 91.0%
#
# The bank was baked through this same function, so inputs and labels agreed
# with each other while neither was a video, and every consistency check passed.
# Fixing it HERE fixes the trainer, the baker and experiments/cache_frames.py at
# once -- which is the reason the original docstring insisted they share it.
_ORDER_CACHE: Dict[str, List[str]] = {}

#: ``<time>_<view>``  -- paralleldomain4d (``..005_camera0``, ``..005_yaw-60``)
#:                       unrealstereo4k   (``00000_cam0``)
_RE_TIME_VIEW = re.compile(r"^(?P<t>\d+)_(?P<v>[A-Za-z][\w.-]*)$")
#: ``<view>-<time>``  -- dynamicreplica   (``009850-3_obj_source_left-0000``)
_RE_VIEW_TIME = re.compile(r"^(?P<v>.*_(?:left|right))-(?P<t>\d+)$")
_RE_INT = re.compile(r"\d+")


def _view_time(stem: str):
    """(view, time) if the name carries both, else None."""
    for rx in (_RE_TIME_VIEW, _RE_VIEW_TIME):
        m = rx.match(stem)
        if m:
            return m.group("v"), int(m.group("t"))
    return None


def image_names(frames_dir: str) -> List[str]:
    """Frame filenames in VIDEO order: one camera, sorted by time numerically.

    ★ meta.npz LIVES IN THE FRAMES DIRECTORY.  A bare sorted(os.listdir(...))
    picks it up as frame 0 or frame N depending on the naming, and the failure is
    a PIL UnidentifiedImageError several minutes into a run -- or, worse, a silent
    off-by-one between the student's inputs and the bank's labels if the extra
    entry sorts in the middle.  Both this module and the trainer must use this
    function so their indexing cannot drift apart.

    Two things beyond that filter:

    ``one camera``  when the names carry a view field and more than one view is
        present, every view but ``min(view)`` is dropped -- cam0, camera0, left.
        Keeping them interleaves frames that share a timestamp, and asking a
        pose head for the ego-motion between two simultaneous views is asking
        for motion that does not exist.
    ``by time``     the time field is compared as an INTEGER, so scannet's
        ``0, 1, 10, 100`` sorts as 0, 1, 10, 100 rather than as text.

    Both are no-ops on a zero-padded single-camera dump (mcd, slowtv, dl3dv,
    replica), which is why the defect survived: the datasets anyone looked at
    were already fine.
    """
    hit = _ORDER_CACHE.get(frames_dir)
    if hit is not None:
        return hit
    raw = [n for n in os.listdir(frames_dir) if n.lower().endswith(IMAGE_EXT)]
    keyed = {n: _view_time(os.path.splitext(n)[0]) for n in raw}
    if raw and all(v is not None for v in keyed.values()):
        views = {v[0] for v in keyed.values()}
        keep = min(views)
        names = sorted((n for n in raw if keyed[n][0] == keep),
                       key=lambda n: keyed[n][1])
        if len(views) > 1:
            print(f"[frames] {frames_dir}: {len(views)} views "
                  f"({len(raw)} files) -> keeping '{keep}' only, {len(names)} frames")
    else:
        def k(n):
            m = _RE_INT.search(os.path.splitext(n)[0])
            return (int(m.group()) if m else -1, n)
        names = sorted(raw, key=k)
    _ORDER_CACHE[frames_dir] = names
    return names


@dataclass
class RunSpec:
    """One fresh teacher run: anchor, burn in, then supervise."""
    t0: int                  # first supervised frame (absolute index)
    L: int                   # supervised frames
    burn_in: int             # B, keyframes of burn-in before t0
    scale_frames: int        # sf, anchor length
    teacher_interval: int    # K_t

    @property
    def a0(self) -> int:
        """Absolute frame where the teacher's anchor starts."""
        return self.t0 - self.burn_in * self.teacher_interval - self.scale_frames

    @property
    def end(self) -> int:
        return self.t0 + self.L

    def validate(self, n_frames: int) -> None:
        cap = max_supervised_frames(self.burn_in, self.scale_frames,
                                    self.teacher_interval)
        if self.L > cap:
            raise ValueError(
                f"run at t0={self.t0} asks for L={self.L} but the teacher budget "
                f"allows {cap} (scale_frames={self.scale_frames}, B={self.burn_in}, "
                f"K_t={self.teacher_interval}, budget={TEACHER_CACHE_BUDGET}). "
                f"Past it the teacher is no longer a clean target -- v4 §3.2 "
                f"measured 0.223 m -> 2.652 m FD ATE across this line.")
        if self.a0 < 0:
            raise ValueError(
                f"run at t0={self.t0} needs {self.burn_in * self.teacher_interval + self.scale_frames} "
                f"frames of anchor+burn-in before it, but the sequence starts at 0")
        if self.end > n_frames:
            raise ValueError(f"run needs frames up to {self.end}, have {n_frames}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def stream_collect(model, images, lo: int, hi: int, scale_frames: int,
                   interval: int, dtype, dev, kf_phase_origin: int,
                   keys=("pose_enc", "depth", "depth_conf"),
                   collect_from: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Stream frames [lo, hi) and collect outputs from ``collect_from`` on.

    Same streaming primitive as ``experiments/fork_rig.step_frames``, but it also
    returns ``depth_conf`` (the loss weights by the teacher's Sigma^D) and can
    discard the burn-in frames as it goes, so a long run does not hold them.
    """
    start = lo if collect_from is None else collect_from
    acc = {k: [] for k in keys}
    for i in range(lo, hi):
        is_kf = (interval <= 1) or ((i - kf_phase_origin) % interval == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(images[:, i:i + 1].to(dev),
                                num_frame_for_scale=scale_frames,
                                num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        if i >= start:
            for k in keys:
                if k in out:
                    acc[k].append(out[k].detach().float().cpu())
        del out
    return {k: torch.cat(v, dim=1)[0] for k, v in acc.items() if v}


@torch.no_grad()
def generate_run(model, images, spec: RunSpec, dtype, dev,
                 keys=("pose_enc", "depth", "depth_conf")) -> Dict[str, torch.Tensor]:
    """Anchor -> burn in -> collect L frames of teacher output.

    ``keys=("pose_enc",)`` bakes a pose-only bank; see ``write_run``."""
    spec.validate(images.shape[1])
    sf = spec.scale_frames
    model.clean_kv_cache()
    with torch.amp.autocast("cuda", dtype=dtype):
        model.forward(images[:, spec.a0:spec.a0 + sf].to(dev),
                      num_frame_for_scale=sf, num_frame_per_block=sf,
                      causal_inference=True)
    # keyframe phase is anchored at the first streamed frame so K_t > 1 lands on
    # a stable grid relative to this run's own anchor
    origin = spec.a0 + sf
    return stream_collect(model, images, origin, spec.end, sf,
                          spec.teacher_interval, dtype, dev, origin,
                          keys=keys, collect_from=spec.t0)


def plan_runs(span: Tuple[int, int], L: int, burn_in: int, scale_frames: int,
              teacher_interval: int, stride: Optional[int] = None) -> List[RunSpec]:
    """Tile [lo, hi) with runs.  ``stride`` defaults to L (contiguous, no overlap).

    An overlap (stride < L) is what v4 §3.3-3 suggests for the cheap fix to the
    run-to-run scale blind spot: shared frames let a later consistency term pin
    the scale ratio between neighbouring runs.  This module only makes the
    overlap available; it does not implement that term.
    """
    lo, hi = span
    stride = L if stride is None else stride
    lead = burn_in * max(teacher_interval, 1) + scale_frames
    runs = []
    t0 = max(lo, lead)
    while t0 < hi:
        length = min(L, hi - t0)
        if length < 2:
            break
        runs.append(RunSpec(t0=t0, L=length, burn_in=burn_in,
                            scale_frames=scale_frames,
                            teacher_interval=teacher_interval))
        t0 += stride
    return runs


def write_run(out_dir: str, idx: int, spec: RunSpec, data: Dict[str, torch.Tensor],
              store_conf: bool = True) -> dict:
    """One .npz per run.  Depth/conf in fp16 -- they are targets, not accumulators."""
    os.makedirs(out_dir, exist_ok=True)
    name = f"run_{idx:05d}_t{spec.t0}.npz"
    arrays = {"pose_enc": data["pose_enc"].numpy().astype(np.float32)}
    # ★ A POSE-ONLY BANK IS A REAL BANK.  The long-supervision targets
    # (docs/long_supervision_design.md section 6) consume rot/dir/scale and
    # nothing else, so baking depth for them would double the label store
    # (~426 GB) to serve a term that never reads it.  Local A1PC keeps using the
    # existing L=240 bank, which still carries depth and conf.
    if "depth" in data:
        d = data["depth"]
        arrays["depth"] = (d[..., 0] if d.dim() == 4 else d).numpy().astype(np.float16)
    if store_conf and "depth_conf" in data:
        arrays["depth_conf"] = data["depth_conf"].numpy().astype(np.float16)
    path = os.path.join(out_dir, name)
    np.savez(path, **arrays)
    return {"file": name, **asdict(spec),
            # absent in a pose-only bake; keep the key so index.json stays one shape
            "shape_depth": list(arrays["depth"].shape) if "depth" in arrays else [],
            "has_depth": "depth" in arrays,
            "has_conf": "depth_conf" in arrays,
            "bytes": os.path.getsize(path)}


# ─────────────────────────────────────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────────────────────────────────────

class LabelBank:
    """Read side.  Samples supervised windows that never cross a run boundary."""

    def __init__(self, root: str, cache_runs: int = 4):
        self.root = root
        with open(os.path.join(root, "index.json")) as f:
            self.index = json.load(f)
        self.runs = self.index["runs"]
        # .npz is a zip archive, so it cannot be memory-mapped -- reading one
        # pulls the whole run (~195 MB) into host RAM.  Keep a small LRU so a
        # trainer walking forward does not accumulate the entire bank.
        self._cache_runs = max(1, cache_runs)
        self._cache: Dict[int, dict] = {}
        self._order: List[int] = []

    def __len__(self):
        return len(self.runs)

    @property
    def frames_covered(self) -> int:
        return sum(r["L"] for r in self.runs)

    def windows(self, S: int, stride: int = 1) -> List[Tuple[int, int]]:
        """(run_id, offset) for every supervised window of length S.

        A window lies inside exactly one run: each run carries its own Sim(3)
        gauge and every loss term normalises within the window, so a window that
        straddled two runs would be normalising across a gauge seam.
        """
        out = []
        for rid, r in enumerate(self.runs):
            for off in range(0, r["L"] - S + 1, stride):
                out.append((rid, off))
        return out

    def _load(self, rid: int) -> dict:
        if rid not in self._cache:
            path = os.path.join(self.root, self.runs[rid]["file"])
            self._cache[rid] = dict(np.load(path))
            self._order.append(rid)
            while len(self._order) > self._cache_runs:
                self._cache.pop(self._order.pop(0), None)
        return self._cache[rid]

    def poses(self, rid: int, lo: int, hi: int, device=None) -> torch.Tensor:
        """Teacher poses for run-offsets [lo, hi).  No depth, no confidence.

        ``L_long`` needs a past anchor's pose and nothing else, and ``get`` would
        slice the depth and conf arrays alongside it.  The anchor is always in
        the SAME run as the window being supervised (long_loss.long_pair_index
        guarantees ``a >= 0`` inside the run), so ``_load`` is already a cache
        hit and this costs no I/O.

        Staying inside one run is not an optimisation: each run carries its own
        Sim(3), so an anchor taken from a neighbouring run would express the
        relative target in a different gauge.
        """
        r = self.runs[rid]
        if lo < 0 or hi > r["L"]:
            raise IndexError(
                f"poses (run {rid}, [{lo}, {hi})) runs outside a run of length "
                f"{r['L']}; a long anchor must stay inside its own run (gauge)")
        out = torch.from_numpy(np.ascontiguousarray(
            self._load(rid)["pose_enc"][lo:hi])).float()
        return out.to(device) if device is not None else out

    def get(self, rid: int, offset: int, S: int, device=None) -> dict:
        """Teacher labels for one window, plus the absolute frame indices."""
        r = self.runs[rid]
        if offset + S > r["L"]:
            raise IndexError(
                f"window (run {rid}, offset {offset}, S={S}) runs past the end of "
                f"a run of length {r['L']}; windows must not cross runs (gauge)")
        z = self._load(rid)
        sl = slice(offset, offset + S)
        out = {
            "pose_enc": torch.from_numpy(np.ascontiguousarray(z["pose_enc"][sl])).float(),
            "depth": torch.from_numpy(np.ascontiguousarray(z["depth"][sl])).float(),
            "t0": r["t0"] + offset,
            "run_id": rid,
        }
        if "depth_conf" in z:
            out["depth_conf"] = torch.from_numpy(
                np.ascontiguousarray(z["depth_conf"][sl])).float()
        if device is not None:
            for k in ("pose_enc", "depth", "depth_conf"):
                if k in out:
                    out[k] = out[k].to(device)
        return out


# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.models.gct_stream import GCTStream

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--span", type=int, nargs=2, default=None,
                    help="[lo, hi) absolute frame range to cover (default: all)")
    ap.add_argument("--L", type=int, default=240, help="supervised frames per run")
    ap.add_argument("--burn_in", type=int, default=72, help="B, in keyframes")
    ap.add_argument("--stride", type=int, default=None,
                    help="run spacing; < L gives overlap for a run-to-run scale term")
    ap.add_argument("--teacher_interval", type=int, default=1, help="K_t")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    ap.add_argument("--no_conf", action="store_true",
                    help="skip depth_conf (halves the bank; the loss then runs unweighted)")
    ap.add_argument("--pose_only", action="store_true",
                    help="bake pose_enc alone -- the long-supervision target bank "
                         "(docs/long_supervision_design.md section 6).  3.5 kB per "
                         "run against 78 MB with depth, because L_long reads "
                         "rot/dir/scale and never touches depth.  Local A1PC must "
                         "keep pointing at a bank that HAS depth.")
    ap.add_argument("--dry_run", action="store_true",
                    help="plan and validate the runs, write nothing")
    args = ap.parse_args()

    sf = args.num_scale_frames
    cap = max_supervised_frames(args.burn_in, sf, args.teacher_interval)
    print(f"[budget] scale_frames={sf} B={args.burn_in} K_t={args.teacher_interval} "
          f"-> L <= {cap} frames; requested L={args.L}"
          + ("  OK" if args.L <= cap else "  EXCEEDS BUDGET"))

    names = image_names(args.frames)
    # ★ CLAMP THE SPAN TO THE IMAGES THAT EXIST.  Callers count the frames
    # directory with `ls | wc -l`, which also counts the float32 frame cache
    # (_cache_<size>_<patch>.npy) that cache_frames.py writes THERE, so --span
    # arrives one too large on every scene that has been cached.  At L=240 the
    # run tiling rarely landed on the boundary and nothing noticed; at L=48 it
    # tiles 5x finer and RunSpec.validate raises "run needs frames up to N, have
    # N-1" on scene after scene.  Clamp here rather than in each caller: this is
    # the only place that knows how many images there actually are.
    span = tuple(args.span) if args.span else (0, len(names))
    if span[1] > len(names):
        print(f"[span] clamped {span[1]} -> {len(names)} (the frames directory "
              f"holds non-image entries; {os.path.basename(args.frames)})")
        span = (span[0], len(names))
    runs = plan_runs(span, args.L, args.burn_in, sf, args.teacher_interval, args.stride)
    if not runs:
        raise SystemExit(f"no runs fit in span {span}: each needs "
                         f"{args.burn_in * args.teacher_interval + sf} frames of lead-in")
    need = max(r.end for r in runs)
    print(f"[plan] {len(runs)} runs over frames [{runs[0].t0}, {runs[-1].end}) "
          f"of span {span}; reading {need} images")
    for r in runs:
        r.validate(len(names))

    if args.dry_run:
        for r in runs[:5]:
            print(f"   t0={r.t0:>6} L={r.L:>4} anchor@{r.a0:>6}")
        if len(runs) > 5:
            print(f"   ... {len(runs) - 5} more")
        return

    dev, dtype = torch.device("cuda"), torch.bfloat16
    images = load_and_preprocess_images(
        [os.path.join(args.frames, n) for n in names[:need]],
        mode="crop", image_size=args.image_size, patch_size=args.patch_size).unsqueeze(0)

    model = GCTStream(
        img_size=args.image_size, patch_size=args.patch_size, enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=sf, kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True, use_sdpa=True, camera_num_iterations=4,
    )
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(sd.get("model", sd), strict=False)
    del sd
    # docs/add_loss.md step 3 wants proof that a bank IS the frozen theta_0's
    # output.  The path alone does not establish that -- load_state_dict runs
    # with strict=False, so a checkpoint that silently failed to populate half
    # the model would still write a plausible-looking bank.  Record the digest
    # AND the load result, and make the loss branch able to assert on them.
    _h = hashlib.sha256()
    with open(args.ckpt, "rb") as _f:
        for _chunk in iter(lambda: _f.read(1 << 24), b""):
            _h.update(_chunk)
    ckpt_sha = _h.hexdigest()
    model = model.to(dev).eval()
    print(f"[model] loaded (missing={len(miss)}, unexpected={len(unexp)})")

    os.makedirs(args.out, exist_ok=True)
    entries, t_start = [], time.time()
    for i, spec in enumerate(runs):
        t = time.time()
        data = generate_run(model, images, spec, dtype, dev,
                            keys=("pose_enc",) if args.pose_only
                            else ("pose_enc", "depth", "depth_conf"))
        e = write_run(args.out, i, spec, data, store_conf=not args.no_conf)
        entries.append(e)
        del data
        print(f"  [{i + 1}/{len(runs)}] t0={spec.t0:>6} L={spec.L:>4}  "
              f"{time.time() - t:5.1f}s  {e['bytes'] / 1e6:6.1f} MB", flush=True)

    index = {
        "frames": os.path.abspath(args.frames),
        "n_frames_available": len(names),
        "scale_frames": sf, "burn_in": args.burn_in,
        "teacher_interval": args.teacher_interval,
        "kv_cache_sliding_window": args.kv_cache_sliding_window,
        "image_size": args.image_size, "patch_size": args.patch_size,
        "budget": TEACHER_CACHE_BUDGET, "max_L_allowed": cap,
        # ★ so a consumer can tell a long-target bank from a local one WITHOUT
        # opening a 146 MB run file.  trainer.py refuses a pose-only bank as the
        # local bank and refuses a depth bank as nothing.
        "pose_only": bool(args.pose_only),
        "stride": args.stride if args.stride is not None else args.L,
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_sha256": ckpt_sha,
        "ckpt_bytes": os.path.getsize(args.ckpt),
        "ckpt_missing_keys": len(miss),
        "ckpt_unexpected_keys": len(unexp),
        "scene": os.path.basename(os.path.dirname(os.path.abspath(args.frames))),
        "runs": entries,
    }
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    total = sum(e["bytes"] for e in entries) / 1e9
    print(f"[done] {len(entries)} runs, {sum(e['L'] for e in entries)} supervised "
          f"frames, {total:.2f} GB in {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
