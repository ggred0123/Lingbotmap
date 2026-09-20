"""
LingbotMap method - Streaming 3D reconstruction with causal transformer.

Wraps the upstream ``lingbot-map`` package (``methods/lingbot-map_repo``,
imported as the ``lingbot_map`` Python module) for benchmark evaluation.
Both streaming and windowed inference modes are supported via ``GCTStream``.
"""

import os
import logging
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from benchmark.method.base import BaseMethod
from benchmark.core.loader import BSSLoader


# Mirrors lingbot-map/demo.py:413-421 — above this frame count, the KV cache
# grows unbounded, so auto-bump the keyframe interval. Used as the default for
# the configurable ``auto_keyframe_threshold`` constructor argument.
_DEFAULT_AUTO_KEYFRAME_THRESHOLD = 320


def _resolve_keyframe_interval(
    cfg_val, num_frames: int, threshold: int = _DEFAULT_AUTO_KEYFRAME_THRESHOLD
) -> int:
    """Resolve a raw config value into a concrete keyframe interval.

    ``None``, ``0``, or the string ``"auto"`` triggers auto-selection:
    ``1`` when ``num_frames <= threshold`` else ``ceil(num_frames / threshold)``.
    An explicit positive int is returned as-is.
    """
    if cfg_val is None or cfg_val == 0 or (isinstance(cfg_val, str) and cfg_val.lower() == "auto"):
        if num_frames <= threshold:
            return 1
        return (num_frames + threshold - 1) // threshold
    return int(cfg_val)


class LingbotMapMethod(BaseMethod):
    """
    LingbotMap model adapter for benchmark evaluation.

    Supports streaming and windowed inference modes via the upstream
    ``GCTStream`` model exposed by the ``lingbot_map`` package.
    """

    def __init__(
        self,
        checkpoint: str = None,
        device: str = 'cuda',
        mode: str = 'streaming',
        use_amp: bool = True,
        use_sdpa: bool = False,
        image_size: int = 518,
        patch_size: int = 14,
        enable_3d_rope: bool = True,
        num_scale_frames: int = 8,
        max_frame_num: int = 1024,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        window_size: int = 64,
        overlap_size: Optional[int] = None,
        keyframe_interval: Any = "auto",
        auto_keyframe_threshold: int = _DEFAULT_AUTO_KEYFRAME_THRESHOLD,
        reanchor_keyframes: int = 0,
        reanchor_overlap: int = 32,
        flow_threshold: float = 0.0,
        max_non_keyframe_gap: int = 30,
        align: int = 14,
        area_budget: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(
            align=align,
            area_budget=area_budget,
            logger=logger,
        )

        self.checkpoint = checkpoint
        self.device = device
        self.mode = mode
        self.use_amp = use_amp
        self.use_sdpa = use_sdpa
        self.image_size = image_size
        self.patch_size = patch_size
        self.enable_3d_rope = enable_3d_rope
        self.num_scale_frames = num_scale_frames
        self.max_frame_num = max_frame_num
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.keyframe_interval = keyframe_interval
        self.auto_keyframe_threshold = int(auto_keyframe_threshold)
        self.reanchor_keyframes = int(reanchor_keyframes)
        self.reanchor_overlap = int(reanchor_overlap)
        self.flow_threshold = flow_threshold
        self.max_non_keyframe_gap = max_non_keyframe_gap

        if self.mode not in ('streaming', 'windowed'):
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'streaming' or 'windowed'")

        if self.auto_keyframe_threshold <= 0:
            raise ValueError(
                f"auto_keyframe_threshold must be a positive int, got {self.auto_keyframe_threshold}"
            )

        self._load_model()

    def _load_model(self):
        """Load LingbotMap (GCTStream) model from checkpoint."""
        # ★ A CAP SO A BENCH CANNOT STARVE A TRAINER SHARING THE CARD.
        # The caching allocator never gives memory back, so a long bench keeps
        # growing: measured 7.9 GB at start and 140 GB six minutes later on a
        # 319-frame grid.  A trainer that was still filling its stream pool at
        # that moment died of OOM and lost the run.  BENCH_GPU_MEM_FRACTION
        # bounds this process instead, so the worst case is a slower bench
        # rather than a dead 12-hour training run.
        frac = os.environ.get("BENCH_GPU_MEM_FRACTION")
        if frac and torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(float(frac))
                total = torch.cuda.get_device_properties(0).total_memory / 2**30
                print(f"  → GPU memory capped at {float(frac):.0%} "
                      f"({float(frac) * total:.0f} GiB of {total:.0f})")
            except Exception as exc:                              # noqa: BLE001
                print(f"  → could not cap GPU memory: {exc}")
        if self.mode == 'windowed':
            from lingbot_map.models.gct_stream_window import GCTStream
        else:
            from lingbot_map.models.gct_stream import GCTStream

        print(f"  → Building LingbotMap model (mode: {self.mode})")
        self.model = GCTStream(
            img_size=self.image_size,
            patch_size=self.patch_size,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.use_sdpa,
        )

        if self.checkpoint:
            print(f"  → Loading checkpoint: {self.checkpoint}")
            ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"    Missing keys: {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
            print("    Checkpoint loaded.")

        self.model = self.model.to(self.device).eval()

    def _prepare_images(self, rgb_list):
        """Convert list of HxWx3 uint8 numpy arrays to [S, 3, H, W] tensor in [0, 1]."""
        from torchvision import transforms as TF

        to_tensor = TF.ToTensor()
        images = torch.stack([to_tensor(rgb) for rgb in rgb_list])
        return images.to(self.device)

    def _run_inference(self, images):
        """Run LingbotMap inference and return raw predictions dict."""
        if self.use_amp:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        print(f"  → Running {self.mode} inference (dtype: {dtype})")

        num_frames = images.shape[0]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            if self.mode == 'streaming':
                keyframe_interval = _resolve_keyframe_interval(
                    self.keyframe_interval, num_frames, self.auto_keyframe_threshold
                )
                if keyframe_interval != self.keyframe_interval:
                    print(
                        f"  → Auto-selected keyframe_interval={keyframe_interval} "
                        f"(num_frames={num_frames}, raw={self.keyframe_interval!r}, "
                        f"threshold={self.auto_keyframe_threshold})"
                    )
                if self.reanchor_keyframes > 0:
                    predictions = self._run_streaming_reanchored(
                        images, keyframe_interval)
                else:
                    predictions = self.model.inference_streaming(
                        images,
                        num_scale_frames=self.num_scale_frames,
                        keyframe_interval=keyframe_interval,
                        output_device=torch.device("cpu"),
                    )
            else:
                predictions = self.model.inference_windowed(
                    images,
                    window_size=self.window_size,
                    overlap_size=self.overlap_size,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=self.keyframe_interval,
                    flow_threshold=self.flow_threshold,
                    max_non_keyframe_gap=self.max_non_keyframe_gap,
                    output_device=torch.device("cpu"),
                )

        return predictions

    def _run_streaming_reanchored(self, images, keyframe_interval):
        """Streaming inference that re-anchors every M keyframes and stitches.

        docs/eval-metric-audit.html section 10 candidate 4, the one the document
        marks as "a diagnostic first": the stitched teacher bank re-anchors every
        48 frames OFFLINE, and this does the same thing ONLINE.  Drift is then
        bounded by one re-anchor period, so the M at which ATE comes back to the
        base level is a direct read of how deep the model holds its scale.

        Mechanics.  The sequence is cut into segments of ``M * keyframe_interval``
        raw frames.  Each segment is an independent ``inference_streaming`` call,
        so it clears the KV cache and re-runs its own scale-frame phase -- a fresh
        gauge every time.  Consecutive segments share ``reanchor_overlap`` frames,
        and a robust Sim(3) fitted on those shared camera centres carries the new
        segment into the running gauge.  This is experiments/stitch_bank.py's seam,
        with the same trimmed fit and the same pose convention.

        ★ THE POSE CONVENTION IS THE PART THAT FAILS SILENTLY.  pose_enc[:3] is the
        camera CENTRE in world coordinates and pose_enc[3:7] the cam->world
        quaternion in XYZW; the seam has to transport BOTH (C -> sRC+t and
        R -> R_seam R).  Mapping the centres alone leaves every rotation in the old
        gauge and the trajectory silently bends at each seam.

        Costs one extra scale phase plus ``reanchor_overlap`` re-run frames per
        segment -- about 4% at M=64, K=12 -- and no retraining.
        """
        import numpy as np
        import sys as _sys
        from pathlib import Path as _Path
        _exp = str(_Path(__file__).resolve().parents[2] / "experiments")
        if _exp not in _sys.path:
            _sys.path.insert(0, _exp)
        from stitch_bank import apply_sim3, fit_seam           # noqa: E402

        M = int(self.reanchor_keyframes)
        K = max(1, int(keyframe_interval))
        sf = int(self.num_scale_frames)
        S = int(images.shape[0])
        period = M * K

        # ★ THE SEAM IS FITTED ON KEYFRAMES, AND THE OVERLAP IS COUNTED IN THEM.
        # Measured the other way first -- 32 RAW frames of overlap -- and ATE on
        # oxford_long (K=12) went 2.6 -> 12.1 m while the keyframe-level rpe
        # improved.  eval-metric-audit.html section 01(c) says why: at K>1 the
        # frames between keyframes leave no KV behind, so consecutive raw poses
        # are independent one-shot estimates whose step directions are close to
        # random.  A Sim(3) fitted on 32 of those is fitted on noise, over a
        # baseline of only 32/12 keyframes of real motion.  Keyframe centres are
        # the cached, mutually consistent ones, so the seam uses those alone.
        # At K=1 the two readings coincide, so K=1 results are unaffected.
        ov_kf = max(8, int(self.reanchor_overlap))
        ov = ov_kf * K
        if period < ov + sf:
            raise ValueError(
                f"re-anchor period {period} (M={M} x K={K}) must exceed the "
                f"{ov}-frame overlap plus {sf} scale frames; raise "
                "_reanchor_keyframes or lower _reanchor_overlap")

        # ★ EVERY SEGMENT STARTS ON A MULTIPLE OF K so that all segments share
        # one keyframe phase (global keyframes land at f = sf mod K).  Without
        # that the grids are offset by up to K/2 and there are no shared
        # keyframes to fit the seam on at all.
        segs, s = [], 0
        while True:
            e = min(S, s + (sf if s == 0 else ov) + period)
            if S - e < max(2 * sf, K):         # never leave an unstitchable tail
                e = S
            segs.append((s, e))
            if e >= S:
                break
            s = max(s + K, ((e - ov) // K) * K)
        print(f"  → re-anchor every {M} keyframes ({period} raw frames): "
              f"{len(segs)} segments, overlap {ov_kf} kf ({ov} frames)")

        glob_pose = np.zeros((S, 9), dtype=np.float64)   # stitched, in seg 0's gauge
        owned = {}                                        # frame -> (seg, local idx, scale)
        seams = []
        prev_end = 0
        for j, (a, b) in enumerate(segs):
            pred = self.model.inference_streaming(
                images[a:b],
                num_scale_frames=self.num_scale_frames,
                keyframe_interval=K,
                output_device=torch.device("cpu"),
            )
            pose = pred["pose_enc"][0].float().numpy().astype(np.float64)
            if j == 0:
                glob_pose[a:b] = pose
                scale = 1.0
            else:
                n = prev_end - a                          # shared frames
                # keyframe offsets inside the shared span: global f = sf mod K
                kf = np.array([i for i in range(n)
                               if (a + i - sf) % K == 0 and a + i >= sf], dtype=int)
                if len(kf) < 8:
                    raise RuntimeError(
                        f"re-anchor seam {j} shares only {len(kf)} keyframes "
                        f"({n} frames); raise _reanchor_overlap")
                seam = fit_seam(pose[kf, :3], glob_pose[a + kf, :3])
                seams.append(seam)
                scale = seam["s"]
                pose = apply_sim3(pose, seam["s"], np.asarray(seam["R"]),
                                  np.asarray(seam["t"]))
                glob_pose[prev_end:b] = pose[n:]
            for f in range(prev_end if j else a, b):
                owned[f] = (j, f - a, scale)
            prev_end = b
            self._seg_cache = getattr(self, "_seg_cache", {})
            self._seg_cache[j] = pred

        if seams:
            r = [s["resid_median"] for s in seams]
            sc = np.array([s["s"] for s in seams])
            cond = np.array([s["cond"] for s in seams])
            # |log s| > 0.2 is a seam that rescales the world by more than 20%;
            # a large `cond` means the overlap was nearly a straight line, where
            # the Sim(3) scale is barely determined at all.
            bad = int((np.abs(np.log(np.maximum(sc, 1e-9))) > 0.2).sum())
            print(f"  → {len(seams)} seams, median residual "
                  f"{np.median(r):.4f} (max {max(r):.4f}), scale spread "
                  f"{sc.min():.3f}-{sc.max():.3f}, median scale {np.median(sc):.3f}, "
                  f"|log s|>0.2 on {bad}/{len(seams)}, max cond {cond.max():.0f}")

        # ── reassemble, taking each frame from the segment that owns it ──────
        keys = [k for k in self._seg_cache[0] if k != "pose_enc"]
        out = {"pose_enc": torch.from_numpy(glob_pose).float().unsqueeze(0)}
        for k in keys:
            v0 = self._seg_cache[0][k]
            if not torch.is_tensor(v0) or v0.dim() < 2 or v0.shape[1] != (segs[0][1] - segs[0][0]):
                out[k] = v0                                # not a per-frame tensor
                continue
            rows = []
            for f in range(S):
                j, li, sc = owned[f]
                t = self._seg_cache[j][k][:, li:li + 1]
                # depth and world points live in the segment's gauge: the seam's
                # scale applies to them exactly as it applies to the centres.
                if k in ("depth", "world_points") and sc != 1.0:
                    t = t * float(sc)
                rows.append(t)
            out[k] = torch.cat(rows, dim=1)
        self._seg_cache = {}
        return out

    def _process_outputs(self, predictions, image_shape):
        """Convert model predictions to benchmark output format.

        Args:
            predictions: Raw model outputs with 'pose_enc', 'depth', 'depth_conf', etc.
            image_shape: (H, W) of the processed images.

        Returns:
            Tuple of (rgb_list, depth_list, pose_list, intrinsics_list, confidence_list)
        """
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

        # Decode pose encoding to extrinsic + intrinsic
        # pose_encoding_to_extri_intri() output is C2W directly (no inverse needed)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], image_shape
        )

        extrinsic = extrinsic.float().cpu().numpy().squeeze(0)  # [S, 3, 4]
        intrinsic = intrinsic.float().cpu().numpy().squeeze(0)  # [S, 3, 3]
        depth = predictions["depth"].float().cpu().numpy().squeeze(0)  # [S, H, W, 1]

        # Extract processed images
        if "images" in predictions:
            images = predictions["images"].float().cpu().numpy().squeeze(0)  # [S, 3, H, W]
        else:
            images = None

        num_frames = extrinsic.shape[0]
        print(f"  → Extracting {num_frames} frames")

        rgb_list = []
        depth_list = []
        pose_list = []
        intrinsics_list = []
        confidence_list = []

        for i in range(num_frames):
            # RGB: [3, H, W] float [0,1] -> [H, W, 3] uint8
            if images is not None:
                rgb = images[i].transpose(1, 2, 0)
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                rgb_list.append(rgb)

            # Pose: 3x4 C2W -> 4x4 C2W
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :] = extrinsic[i].astype(np.float32)
            pose_list.append(pose)

            # Intrinsics: 3x3 K -> [fx, fy, cx, cy]
            K = intrinsic[i]
            intrinsics_list.append(np.array(
                [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32
            ))

            # Depth: [H, W, 1] -> [H, W]
            depth_frame = depth[i]
            if depth_frame.ndim == 3 and depth_frame.shape[-1] == 1:
                depth_frame = depth_frame.squeeze(-1)
            depth_list.append(depth_frame.astype(np.float32))

            # Confidence
            if "depth_conf" in predictions:
                conf = predictions["depth_conf"][0, i].float().cpu().numpy()
                confidence_list.append(conf.astype(np.float32))

        return rgb_list, depth_list, pose_list, intrinsics_list, confidence_list

    def process_scene(self, gt_artifact) -> Dict[str, Any]:
        """Process a scene with LingbotMap inference."""
        loader = BSSLoader(gt_artifact, resize_context=self.resize_context)
        input_rgb_list = loader.load_rgb_list()
        self.logger.info(f"Image size for processing: {loader.get_processing_dimensions()} (HxW)")

        print(f"  → Processing {len(input_rgb_list)} frames with LingbotMap (mode: {self.mode})")

        # Prepare and run inference
        images = self._prepare_images(input_rgb_list)
        image_shape = images.shape[-2:]  # (H, W)
        predictions = self._run_inference(images)

        # Convert outputs
        rgb_list, depth_list, pose_list, intrinsics_list, confidence_list = \
            self._process_outputs(predictions, image_shape)

        if len(depth_list) != len(input_rgb_list):
            print(f"  → WARNING: Output frames ({len(depth_list)}) != input frames ({len(input_rgb_list)})")

        # Assemble results
        print(f"  → Assembling {len(rgb_list)} frames in standard format")
        frame_results = {
            'rgb': rgb_list,
            'depth': depth_list,
            'pose': pose_list,
            'intrinsics': intrinsics_list,
        }

        if confidence_list:
            frame_results['confidence'] = confidence_list

        return {
            'frame': frame_results,
            'global': {},
        }
