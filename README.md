# streaming3d-self-distill

On-policy self-distillation for streaming 3D reconstruction: fixing the
long-horizon collapse of [LingBot-Map](README_lingbot_map.md)'s Direct mode
without test-time stitching.

This repository is a research fork of LingBot-Map (Geometric Context
Transformer). The released model, demo and benchmark are unchanged and
documented in [README_lingbot_map.md](README_lingbot_map.md); everything under
`lingbot_map/train/` and `experiments/` is new.

## Problem

LingBot-Map is trained on sequences of up to 320 views. Its Direct mode
(no state reset, one absolute pose + depth per frame) drifts and eventually
collapses on long rollouts (~3 000 frames) because the KV cache it reads from
is its *own* hidden state: the model is never trained on the state distribution
it produces at inference. This is exposure bias, in the imitation-learning
sense. The upstream fix is VO mode (short windows + Sim(3) stitching at test
time); the goal here is to amortize that into the weights so a single
feed-forward stream stays metric over kilometres.

## Method (DAgger-style)

* **Rollout.** Stream the student in Direct mode, no grad, over long unlabeled
  video to build on-policy caches (`RolloutPool` in `trainer.py`).
* **Expert query.** A frozen teacher (= the released weights, θ₀) is run on a
  short window `[t−w, t]` from a fresh state; its locally accurate poses,
  depths and confidences are the labels (`label_bank.py`, baked offline once
  per scene into `labels/<dataset>_<scene>/`).
* **Gauge-free supervision.** Student and teacher live in different
  coordinate frames, so absolute poses cannot be compared. Losses are relative
  pose (rotation, direction, magnitude), motion, and scale-aligned depth
  (`losses.py`), plus optional run-scale absolute terms (`abs_loss.py`) and
  long-range seam terms (`long_loss.py`).
* **Truncated gradient.** Only the last window attends with grad to a detached
  cache prefix, so memory is bounded and no context parallelism is needed.
* **Identity branch.** With probability `p_identity` a step supervises the
  student on a *fresh* (teacher-forced) state instead of a rolled one — the
  off-policy control and the only anchor to θ₀ when `--l2sp 0`.

## Repository layout

```
lingbot_map/train/
  trainer.py         the training loop: rollout pool, state sampler, probes, DDP, resume
  losses.py          gauge-invariant pair losses and presets (A1PC etc.)
  label_bank.py      frozen-teacher label baking + bank sampling
  abs_loss.py        run-gauge absolute / scale terms (gtabs cell)
  long_loss.py       long-range (stitched) supervision
  lora.py            LoRA adapters (see below)
experiments/
  launch_clean.sh    launch one arm (torchrun, detached); MANIFEST=<file> selects the corpus
  watchdog_*.sh      keep an arm alive on its card, resume from the last checkpoint
  chain_*.sh         score an arm once it finishes (Oxford K=1 grid, MCD ladder)
  build_banks_generic.sh / build_banks_slowtv.sh   bake teacher banks for a corpus
  cache_frames.py    memmap frame cache (--uint8 is lossless for the 518x14 crop)
  score_k1_c0off.sh  Oxford Spires K=1 benchmark for a finished arm
  mcd_distance_ladder.py / mcd_distance_score.py   held-out MCD scenes at 3 frame strides
  coverage_ladder_score.py                         scene-count ladder D(N) + slopes
  corpus/*.txt       corpus manifests (NAME:FRAMES:BANK:DATASET per line)
  test_*.py          CPU tests (losses, long loss, abs loss, state mixture, lora, ...)
benchmark/           upstream evaluation harness + this fork's method/dataset configs
preprocess/          dataset preprocessing (Oxford Spires, SlowTV, MCD, ...)
```

Large artefacts stay out of git: `data/` (frames + caches), `labels/` (teacher
banks), `ckpt_train/`, `bench_ckpt/`, `wandb/`, `experiments/logs/`,
`experiments/results/`. Design notes and result ledgers live in `docs/`
(local, see `.gitignore`).

## Pipeline

1. **Bake teacher banks** for every scene of a corpus (one GPU, ~40 min per
   8 000-frame scene at L=240):

   ```bash
   DATASET=slowtv ROOT=data/slow_tv SUBDIR=frames_10hz SPAN=8000 CACHE_FLAGS=--uint8 \
     bash experiments/build_banks_generic.sh 00000c2 00001c2
   ```

   A scene needs ≥ burn_in + 8 + L (= 320) frames to yield a single run.

2. **Train one arm.** `launch_clean.sh <arm> <p_identity> [trainer flags]`
   writes `ckpt_train/<arm>.step*.pt` every `save_every` steps and
   `experiments/results/train_<arm>.json`. Corpora come from a manifest:

   ```bash
   MANIFEST=experiments/corpus/rung18.txt TARGET=275 \
     experiments/watchdog_broad.sh myarm 0 --gt_calib data/mcd/calib/hhs_calib.yaml --gt_sensor d455b_color
   ```

   The recipe every arm in the ledger shares: preset `A1PC`, `--sampler unified
   --p_identity 0.35 --k_dist v5 --pool 20 --fresh_pool 5`, AdamW lr 1e-5,
   encoder (`patch_embed`) frozen, save every 25 steps, 275 steps.

3. **Score.** `chain_rung.sh <arm> <gpu>` waits for the arm, strips the
   optimizer into `bench_ckpt/sd_<arm>_step*.pt`, runs Oxford Spires K=1 at
   steps 150/200/250/275 and the MCD hold-out distance ladder, and
   `coverage_ladder_score.py` folds everything into the D(N) table.

## LoRA adapters (`lingbot_map/train/lora.py`)

Every full-fine-tune arm so far improves the scenes its correction branch
rolled on and degrades every scene it did not — same campus, other campus,
Oxford alike — and the damage grows monotonically with steps. LoRA makes the
update a rank-`r` subspace of the frozen released weights, so rank is a
capacity dial for that co-adaptation.

```bash
# r=16 on the 24 global (cache-reading) blocks: qkv / proj / fc1 / fc2, 6.3 M trainable of 1.16 B
experiments/launch_clean.sh lora16 0.35 --lora_rank 16 --lora_alpha 32 --lr 1e-4
```

| flag | default | meaning |
|---|---|---|
| `--lora_rank` | 0 (off) | rank `r`; > 0 freezes the base and trains only the adapter |
| `--lora_alpha` | 2·r | delta = (alpha / r)·B·A |
| `--lora_dropout` | 0 | dropout on the adapter input |
| `--lora_targets` | `global` | comma list of `global`, `frame`, `camera`, `depth` (trainer `PARAM_GROUPS`) |
| `--lora_modules` | `qkv,proj,fc1,fc2` | which `nn.Linear` leaves inside each target |

Checkpoint format is unchanged for every reader: `ckpt["model"]` holds the
**merged** weights (released key set, adapter folded in), so
`_strip_optim.py`, the benchmark and the MCD ladder need no changes;
`ckpt["lora"]` holds the adapter alone and is what `--resume` restores on top
of θ₀. `B` is zero-initialised, so step 0 is exactly the released model, and
the rollout pool always sees the adapted weights. Verify with
`python experiments/test_lora.py [--real]`.

## Status (Sep 2026)

* In-domain gain vs. hold-out damage is reproduced across GT vs. teacher
  labels, window-local vs. run-gauge scale terms, and 10- vs. 25-scene
  corpora. The identity-only arm (no on-policy correction) is harmless.
* Distance regime is ruled out: scenes the model trained on stay at base
  accuracy at Oxford's 1.5 m/frame over 470 m; held-out scenes of the *same*
  campus degrade as much as Oxford does.
* Scene-count ladder N = 2 / 5 / 10 / 18 (same outdoor-walking domain): every
  slope of D(N) has a 90 % CI spanning zero. N = 45 (SlowTV Natural chunks)
  is baked and queued.
* Next: LoRA rank ladder (r = 4 / 16 / 64) on the 10-scene recipe, and a
  hold-out scorer for genuinely different domains (driving, indoor).

## Setup

Follow the upstream installation section of [README_lingbot_map.md](README_lingbot_map.md)
(PyTorch, the CUDA extensions, optional FlashInfer). Training uses the SDPA
path. The released checkpoint is expected at
`../ckpt/lingbot-map.pt` (override with `CKPT=`).

## License / citation

Apache-2.0, inherited from LingBot-Map — see [LICENSE.txt](LICENSE.txt) and the
citation in [README_lingbot_map.md](README_lingbot_map.md).
