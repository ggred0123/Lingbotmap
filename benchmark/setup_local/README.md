# Local benchmark setup

How the shipped `benchmark/` pipeline is wired on this machine, and what to type
to run it. Everything below is already configured — this file is the record, not
a to-do list.

## Environment

There are no conda envs here; the container's system Python runs everything.
The benchmark needs `opencv / open3d / evo / OpenEXR / plyfile / trimesh`, none
of which were installed, so they live in an isolated venv:

    /NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-bench

It was created with `--system-site-packages`, so it reuses the container's torch
(2.10.0a0, CUDA) and adds only the missing pieces. Installing into it cannot
disturb a running trainer, which is why it exists.

Two consequences for the configs:

- Method configs carry **no `env:` field**, so `run.py` executes the model
  in-process instead of dispatching through `conda run`.
- `_use_sdpa: true` everywhere — flashinfer is not installed, and the FlashInfer
  attention path raises without it. The SDPA path honours
  `kv_cache_sliding_window` eviction (`CausalAttention._apply_kv_cache_eviction_causal`),
  so the history-length sweep below is real under SDPA.

`datasets/__init__.py` and `methods/__init__.py` were added: the container ships
HuggingFace `datasets`, and a namespace package loses to a regular package found
later on `sys.path`, so `datasets.oxford_spires` failed to import until these
directories became regular packages.

## Running

`benchmark/bench.sh` wraps the phases with the venv, pins `CUDA_VISIBLE_DEVICES`
(default GPU 1, since training usually holds GPU 0) and caps threads:

    benchmark/bench.sh prepare  configs/oxford.yaml
    benchmark/bench.sh run      configs/oxford.yaml
    benchmark/bench.sh evaluate configs/oxford.yaml
    benchmark/bench.sh all      configs/oxford.yaml      # the three in order
    benchmark/bench.sh report   /path/to/bench_ws/oxford

    GPU=0 THREADS=16 benchmark/bench.sh run configs/kitti.yaml
    benchmark/bench.sh prepare configs/oxford.yaml --debug   # first scene only

A single (method, scene) job, bypassing the config's method list:

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$(cd .. && pwd) \
      ../../.venv-bench/bin/python run_worker.py \
        --config configs/oxford.yaml --method base_h64 \
        --dataset oxford --scene bodleian-library-02

Outputs (BSS workspace) go to `/NHNHOME/.../youngmin/bench_ws/{dataset}/`.
Phases resume: completed scenes are skipped unless `--force`.

Two helpers:

    ./setup_local/prepare_ready.sh          # prepare every ready dataset (CPU only,
                                            # safe to run while a trainer holds the GPUs)
    python setup_local/summarize.py /path/to/bench_ws/*   # metric vs. history-length table

`summarize.py` parses method names as `{tag}_h{history}` and prints one markdown
table per dataset — baseline and self-distilled side by side, averaged over
whatever scenes have been evaluated so far, with the scene count in the last
column. `report.py` still generates the full HTML report.

## Methods

`_checkpoint` is the only thing separating the two model variants:

| config | checkpoint |
|---|---|
| `base_h*` | `ckpt/lingbot-map.pt` (released baseline) |
| `sd_a1_h*` | `bench_ckpt/sd_a1_step200.pt` (self-distilled A1, step 200) |

`bench_ckpt/sd_a1_step200.pt` is `ckpt_train/a1.step200.pt` with the optimizer
state stripped (11.5 GB → 4.6 GB); the weights are byte-identical. To benchmark
another run:

    python - <<'EOF'
    import torch
    ck = torch.load('ckpt_train/a2.step200.pt', map_location='cpu',
                    weights_only=False, mmap=True)
    torch.save({'model': ck['model'], 'step': ck.get('step')},
               'bench_ckpt/sd_a2_step200.pt')
    EOF

then copy a `sd_a1_h*.yaml` to `sd_a2_h*.yaml` and repoint `_checkpoint`.

### History-length sweep

`_kv_cache_sliding_window` is the number of recent frames the KV cache keeps
(plus `_kv_cache_scale_frames: 8` anchor frames that are never evicted). The
sweep is `h ∈ {16, 32, 64, 128, 256}`, 64 being the released default, for both
checkpoints — 10 method configs in total.

`configs/oxford.yaml`, `configs/kitti.yaml` and `configs/vbr.yaml` list all 10
(ATE / RPE per history length). The point-cloud configs list only `base_h64` and
`sd_a1_h64`; add more if you want the sweep there too.

## Datasets

| config | root | scenes | status |
|---|---|---:|---|
| `oxford` / `oxford_long` | `seonghyun/R3R/data/oxford_spires` | 10 | ready |
| `kitti_504x280` | `seonghyun/R3R/data/kitti/dataset` | 11 | ready |
| `vbr` | `seonghyun/R3R/data/vbr` | 7 | ready |
| `seven_scenes` | `seonghyun/R3R/data/7scenes` | 18 | ready (test split already extracted) |
| `tum` | `seonghyun/R3R/data/tum` | 9 | ready (bonus, not requested) |
| `eth3d` | `youngmin/data_bench/eth3d` | 11 | ready (downloaded + converted here) |
| `neural_rgbd` | `youngmin/data_bench/neural_rgbd_data` | 9 | downloading (slow host) |

Scene counts match the upstream README's result tables, so numbers are
comparable to the published `lingbot-map.pt` rows.

`droid_w` and `tat` still hold `/path/to/...` placeholders — that data is not on
this machine.

### ETH3D

The adapter expects a pi3-flavoured layout (`images/custom_undistorted`,
`ground_truth_depth/custom_undistorted`, `custom_undistorted_cam/*.npz`) that no
official archive ships, and the gap is not just naming. Per ETH3D's own docs the
GT depth maps are rendered against the **original, distorted** images
(6048x4032, THIN_PRISM_FISHEYE), while the undistorted images are pinhole and a
different size (6205x4135, 6220x4141, … — it varies per camera). The adapter
reshapes the depth file with the RGB image's shape, so the two grids have to be
made to match. That is what makes this variant "custom".

Three steps, already run:

    ./setup_local/download_eth3d.sh                                      # images + per-scene depth
    ../../.venv-bench/bin/python setup_local/convert_eth3d.py            # images + per-frame npz
    ../../.venv-bench/bin/python setup_local/warp_eth3d_depth.py --convention z

`warp_eth3d_depth.py` takes each undistorted pixel's viewing ray, projects it
into the distorted image with ETH3D's documented THIN_PRISM_FISHEYE model, and
samples the depth there — same camera, same instant, so only the grid changes.
The map depends only on the camera pair, so it is built once per camera (1–6 per
scene) and reused across frames. 416 depth maps, ~4 minutes.

The distorted calibration holding the fisheye parameters is not in the
undistorted archive, so the script chain also needs
`multi_view_training_dslr_jpg.7z` (5.1 GB), from which only
`*/dslr_calibration_jpg/*` is extracted.

**Depth convention.** ETH3D does not document whether depth is z along the
optical axis or Euclidean ray length, and the two differ enough to matter, so
`--check` measures it instead of assuming: for observations whose 3D position is
known from the sparse COLMAP points, it compares the stored value against both
candidates. The answer is unambiguous — median |Δ| 0.002–0.04 m against z versus
0.08–3.8 m against ray length — hence `--convention z`, which transfers the
value unchanged.

### Neural RGB-D

`download_neural_rgbd.sh` pulls the 7.2 GB zip; its layout already matches the
adapter. The TUM host is slow (~200 KB/s), so this runs for hours — it is
started under tmux (`tmux attach -t dl_nrgbd`) and is resumable (`wget -c`).

## Verified

Full `prepare → run → evaluate` on Oxford Spires *bodleian-library-02*
(320 frames at stride 12, GPU shared with a live trainer, ~5.5 fps):

| method | ATE | RPE-trans | RPE-rot |
|---|---:|---:|---:|
| `base_h16` | 32.207 | 2.084 | 2.482 |
| `base_h64` | 4.318 | 1.208 | 1.255 |
| `sd_a1_h64` | 14.457 | 1.483 | 2.494 |

Single scene, so read it as a smoke test, not a result: it confirms both
checkpoints load, the history knob changes behaviour, and metrics land in the
same range as the upstream Oxford row (dataset-level ATE 5.374).

The point-cloud path (open3d ICP, the fragile one) was checked the same way on
7-Scenes *chess/seq-03* with `base_h64` — chamfer 0.032, acc 0.039, comp 0.024,
F1 79.9, alongside ATE 0.043 and AUC. The upstream 7-Scenes row is chamfer 0.040
/ F1 82.4 over all 18 sequences. ETH3D *courtyard* likewise ran end to end:
chamfer 0.190, F1 72.4, ATE 0.303 (upstream 11-scene row: 0.128 / 86.8 / 0.439).

The ETH3D conversion carries its own checks, since a silent geometry error there
would poison every ETH3D number:

- Pose / intrinsics parsing — reprojecting the sparse COLMAP points through the
  parsed pose and K lands within 0.67–0.82 px median, with no systematic offset,
  on 25 MP images.
- The finished depth warp — comparing warped depth against the same known 3D
  points gives 0.21 cm median error (p90 2.2 cm), which covers the fisheye
  projection, the half-pixel corner-vs-centre convention, and the resampling
  together. Valid-pixel coverage is 11.3%, matching ETH3D's note that the
  rendered depth is sparse at full DSLR resolution.

All five ready adapters were also instantiated directly and load frame 0 with
the expected keys and resolutions.

## Cost warning

`configs/kitti.yaml` and `configs/vbr.yaml` list 10 methods over long sequences
(KITTI seq 00 alone is 4541 frames; VBR campus_train0 is 12042). That is a large
number of GPU-hours per config, and the trainer is using the same GPUs. Trim the
`methods:` list, or run one history length at a time, before launching the full
sweep.

## Prepared workspaces

`prepare` has already been run for everything that is ready, so `run` can start
directly:

| workspace | scenes | frames |
|---|---:|---|
| `bench_ws/oxford` | 10 | 320 per scene (stride 12) |
| `bench_ws/kitti` | 11 | 23,201 total (seq 00: 4,541) |
| `bench_ws/vbr` | 7 | up to 12,042 per scene |
| `bench_ws/seven_scenes` | 18 | 200 per sequence (stride 5) |
| `bench_ws/eth3d` | 11 | 14–76 per scene |
