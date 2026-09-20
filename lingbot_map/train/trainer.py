"""T5 -- the trainer.  docs/phase1-plan.md §3-T5, gate 6.

Every piece below already exists and was measured; this assembles them.

    prefix        Level-0 rollout: no grad, detached at every step
    window        T1/T2-e masked parallel path, S=48, checkpointing ON
    labels        T4 offline bank (teacher never runs in this loop)
    loss          T3, gauge-invariant, unlabeled
    encoder       frozen (or scaled LR) -- its grad norm is ~6x global_blocks
    L2-SP         decoupled pull toward the released weights

★ THE ROLLOUT POOL IS THE WHOLE DESIGN PROBLEM.  A supervised window at frame t
needs the student's streaming state at t, and that state is produced by the
student itself walking frames 0..t.  Re-rolling per step is impossible (t=5248
costs ~5 minutes).  So each pool slot owns a LIVE stream: it is restored, the
window is supervised without touching it (the masked path is read-only -- that
property is what makes this legal), and then it is advanced past the window with
the freshly updated weights and re-snapshotted.

That gives on-policy-with-lag data, which is the DAgger regime v4 §3.3-1 already
flagged.  Interleaving M streams decorrelates consecutive steps; without it every
step would see the next 48 frames of the same walk.

★ THE LOGGED LOSS IS PREQUENTIAL.  A window is scored at the pre-update weights
and each window is visited once, so the training curve is already a held-out
curve.  It still mixes learning with scene difficulty, so a fixed PROBE set --
frozen states, never advanced, re-scored every N steps -- isolates the weight
change on an unchanging problem.  Gate 6 ("loss 하강") is read off the probe.

──────────────────────────────────────────────────────────────────────────────
[한국어 개요]
이 파일은 학생(student) 모델이 스스로 만든 롤아웃(rollout) 위에서, 교사(teacher)
모델이 미리 만들어 둔 라벨(오프라인 뱅크)을 흉내 내도록(self-distillation) 학습시키는
전체 학습 루프(trainer)이다.

핵심 구성 요소:
    prefix   레벨-0 롤아웃: gradient 없이(no grad), 매 스텝마다 detach 됨
    window   T1/T2-e 마스킹된 병렬 경로. 윈도우 길이 S=48, 체크포인팅 ON
    labels   T4 오프라인 뱅크 (이 루프에서는 교사 모델이 직접 돌지 않는다)
    loss     T3, gauge-invariant(게이지 불변), 라벨 없는(unlabeled) 손실
    encoder  고정(frozen)되거나 낮은 LR 사용 -- encoder의 grad norm이 global_blocks의
             약 6배로 매우 크기 때문
    L2-SP    배포된(released) 가중치 쪽으로 당기는 decoupled 페널티

★ 롤아웃 풀(pool)이 설계의 전부다.  프레임 t에서 지도학습(supervise)하려면 학생이
t까지 프레임 0..t를 실제로 걸어와서(walk) 만든 스트리밍 상태(state)가 필요하다.
매 스텝마다 처음부터 다시 굴리는 것은 불가능하다(t=5248이면 ~5분 소요).  그래서
풀의 각 슬롯은 "살아있는(LIVE) 스트림"을 하나씩 소유한다: 상태를 복원(restore)하고,
그 창(window)을 (마스킹된 read-only 경로로) 건드리지 않고 지도학습만 한 다음,
방금 업데이트된 가중치로 그 창을 지나쳐서(advance) 다시 스냅샷(snapshot)한다.

이렇게 하면 "지연이 있는 온-폴리시(on-policy-with-lag)" 데이터가 만들어지며,
이는 DAgger 방식(v4 §3.3-1)에서 이미 지적된 문제다.  M개의 스트림을 번갈아
(interleave) 사용하는 이유는 연속된 스텝끼리의 상관관계를 없애기 위함이다 --
안 그러면 매 스텝이 같은 walk의 다음 48프레임만 보게 된다.

★ 기록되는 loss는 prequential(사전순차적) 하다.  창(window)은 업데이트 "전"
가중치로 채점되고, 각 창은 딱 한 번만 방문되므로 학습 곡선 자체가 이미
held-out(홀드아웃) 곡선이다.  다만 이는 "학습 정도"와 "장면(scene) 난이도"가
섞여 있는 곡선이라서, 고정된 PROBE 집합 -- 절대 진행(advance)시키지 않고
동결된(frozen) 상태를 N스텝마다 다시 채점하는 것 -- 을 두어 가중치 변화만을
분리해서 본다.  Gate 6 ("loss 하강")은 바로 이 probe에서 읽는다.
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from lingbot_map.train.label_bank import LabelBank, image_names
from lingbot_map.train.losses import DEPTH_MODES, PRESETS, SelfDistillLoss
from lingbot_map.train import long_loss as LL
from lingbot_map.train import abs_loss as AL
from lingbot_map.train import lora as LORA
from lingbot_map.train import wandb_log

# [KR] 모델 파라미터를 5개 그룹으로 나눠서 grad norm을 그룹별로 따로 로깅하기 위한
# (표시이름, 파라미터 이름 prefix) 목록. grad_norms()에서 사용.
PARAM_GROUPS = [
    ("encoder", "aggregator.patch_embed"),
    ("frame_blocks", "aggregator.frame_blocks"),
    ("global_blocks", "aggregator.global_blocks"),
    ("camera_head", "camera_head."),
    ("depth_head", "depth_head."),
]



class MemmapFrames:
    """A [1, N, 3, H, W] stand-in backed by a disk memmap.
    ★ torch.from_numpy ON A READ-ONLY MEMMAP SEGFAULTS.  It hands torch a buffer
    it believes is writable and whose lifetime it does not own; the first launch
    that tried it died with SIGSEGV on the very line.  So never wrap the mapping
    -- index it, and materialise only the frames asked for.

    The trainer's access pattern is always ``images[:, a:b]`` and the slices are
    tiny: one frame while rolling, S=48 for a window, ~320 for the fresh branch's
    anchor+burn-in span.  Each __getitem__ copies just that, so resident memory
    is bounded by the slice rather than by the scene (216 GB per process before,
    which is what made two concurrent runs unschedulable), and the page cache is
    shared between runs reading the same file.

    [KR] 디스크에 memmap(메모리 매핑)으로 저장된 이미지 시퀀스를,
    [1, N, 3, H, W] 모양의 텐서인 척 흉내내는 래퍼(wrapper) 클래스.
    - 전체 장면(scene)을 통째로 메모리에 올리면 프로세스당 216GB까지 차지해서
      두 개의 학습을 동시에 돌릴 수 없었기 때문에 만들어짐.
    - numpy memmap은 "읽기 전용(read-only)" 파일 매핑인데, 이걸 그대로
      torch.from_numpy()에 넘기면 torch가 "내가 쓸 수 있고 내가 수명을 관리하는
      버퍼"로 착각해서 세그폴트(SIGSEGV)가 난다. 그래서 반드시 슬라이스한 뒤
      .copy()로 실제 메모리 복사본을 만들어서 넘겨야 한다.
    - 실제 학습 코드가 요청하는 슬라이스 크기는 항상 작다: 롤아웃 중엔 프레임 1개,
      지도학습 창(window)은 S=48개, fresh 브랜치의 anchor+burn-in 구간은 ~320개.
      그래서 이 클래스는 "요청받은 만큼만" 복사하도록 강제해서 메모리 사용량을
      장면 전체 크기가 아니라 "슬라이스 크기"로 묶어둔다(bound).
    """

    #: largest slice a caller may ask for, in frames.  The trainer reads 1, 8,
    #: 48 (a window) or 320 (fresh_step's anchor+burn-in span); anything larger
    #: is an unbounded slice that would copy most of the scene.
    # [KR] 한 번에 요청 가능한 최대 프레임 수. 정상적인 호출은 1(롤아웃 한 프레임),
    # 8(scale frame), 48(window), 320(fresh anchor+burn-in) 정도이므로, 이보다
    # 크면 "무한정 슬라이스"로 보고 에러를 낸다(장면 전체를 복사하는 버그 방지).
    MAX_SLICE = 4096

    def __init__(self, path: str, n: int):
        # [KR] mmap_mode="r"로 열면 실제 데이터는 디스크에 남아있고, 필요한
        # 부분만 OS 페이지 캐시를 통해 읽어온다 (지연 로딩).
        import numpy as np
        self._mm = np.load(path, mmap_mode="r")
        if self._mm.shape[0] < n:
            raise SystemExit(f"{path} has {self._mm.shape[0]} frames, need {n}")
        self._n = n
        # A uint8 cache (cache_frames.py --uint8) holds round(x*255); it is
        # rescaled to float32 in __getitem__, so every consumer sees the same
        # [0, 1] float tensor as from a float32 cache.  nbytes reports the
        # float32 size the trainer would otherwise hold, so the "[scene] ... GB"
        # line stays comparable across cache dtypes.
        self._u8 = self._mm.dtype == np.uint8
        self.nbytes = 4 * n * int(np.prod(self._mm.shape[1:]))

    def __getitem__(self, key):
        # [KR] images[:, a:b] 형태의 호출만 허용한다. 그 외 인덱싱 패턴은
        # 이 클래스의 설계 의도(안전한 bounded copy)를 벗어나므로 명시적으로 거부한다.
        import numpy as np
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None)):
            raise TypeError(f"MemmapFrames supports images[:, a:b], got {key!r}")
        sl = key[1]
        # ★ REFUSE AN OPEN-ENDED SLICE.  Every __getitem__ here COPIES (that is
        # the whole point of the class), so ``images[:, a0:]`` materialises the
        # rest of the FILE -- 15-24 GB at 0.3-3 GB/s.  FreshPool._sub did exactly
        # that twice per identity step until it was fixed; the trainer's real
        # access pattern is 1, 8, 48 or 320 frames, so anything unbounded or
        # large is a bug at the call site rather than a legitimate read.
        if isinstance(sl, slice):
            _lo = 0 if sl.start is None else sl.start
            _hi = self._mm.shape[0] if sl.stop is None else sl.stop
            if _hi - _lo > self.MAX_SLICE:
                raise ValueError(
                    f"MemmapFrames: images[:, {sl.start}:{sl.stop}] would copy "
                    f"{_hi - _lo} frames ({(_hi - _lo) * self.nbytes / max(self._n, 1) / 1e9:.1f} GB). "
                    f"Slice a bounded range at the call site (see FreshPool._frames).")
        # .copy() and NOT ascontiguousarray: a leading-axis slice of a C-contiguous
        # memmap is already contiguous, so ascontiguousarray returns the read-only
        # view untouched and torch.from_numpy is back to wrapping a mapping it does
        # not own -- the exact SIGSEGV this class exists to avoid.  Force the copy.
        # [KR] 반드시 .copy()를 써야 한다. ascontiguousarray()는 이미 연속된
        # (contiguous) 메모리라고 판단하면 아무것도 복사하지 않고 원본 read-only
        # view를 그대로 반환해버리므로, 결국 이 클래스가 막으려던 SIGSEGV가
        # 재발한다. unsqueeze(0)으로 배치 차원 [1, ...]을 추가해서 반환.
        if self._u8:
            # astype allocates a fresh float32 array, so this is a copy too
            return torch.from_numpy(self._mm[sl].astype(np.float32) * (1.0 / 255.0)).unsqueeze(0)
        return torch.from_numpy(self._mm[sl].copy()).unsqueeze(0)

    def numel(self):
        # [KR] float32(4바이트) 기준 전체 원소 개수. GB 단위 로그 출력용.
        return self.nbytes // 4


@dataclass
class Scene:
    """One MCD sequence: its frames, its bank, and its OWN 0-based frame index.

    ★ SCENE-LOCAL INDEXING IS A CONSTRAINT, NOT A STYLE CHOICE.  The two obvious
    shortcuts were both tried and both fail:

      concatenating bank run lists -- every MCD bank starts at t0=80, so
        ``locate()`` returns the FIRST scene's run for every query and every
        later scene becomes silently unreachable.  Not an error, just a corpus
        that quietly shrinks to one sequence.

      one images tensor with a global frame offset -- the masked path's
        ``_check_prefix_consistency`` (aggregator/stream.py:566-602) predicts the
        cached keyframe count from ``window_start`` and raises when it disagrees,
        because window_start must be the index the stream actually rolled from.

    So each scene owns its tensor, its bank, and its indices, and a stream
    carries the scene it belongs to.

    [KR] 하나의 MCD 시퀀스(장면)를 나타내는 데이터 클래스.
    핵심 설계 원칙: "장면마다 자기만의 0-based 프레임 인덱스를 갖는다."
    - 여러 장면의 뱅크(run 목록)를 하나로 이어붙이면, 모든 뱅크가 t0=80에서
      시작하기 때문에 locate()가 항상 "첫 번째 장면의 run"만 찾게 되어 뒤쪽
      장면들이 조용히(에러 없이) 사라져버린다.
    - 여러 장면의 이미지를 하나의 텐서 + 전역(global) 오프셋으로 합치면,
      마스킹 경로의 _check_prefix_consistency가 window_start로부터 캐시된
      키프레임 개수를 역산하는데, window_start가 "그 stream이 실제로 걸어온
      시작점"이 아니면 계산이 어긋나서 에러가 난다.
    그래서 장면마다 자기 텐서, 자기 뱅크, 자기 인덱스를 따로 갖고, 하나의
    스트림(RolloutStream)은 자신이 속한 장면을 참조로 들고 다닌다.
    """
    name: str
    frames: str
    images: torch.Tensor                    # [1, N, 3, H, W] on the host
    bank: LabelBank
    skip_runs: set = field(default_factory=set)   # [KR] probe용으로 학습에서 제외된 run id들
    covered: tuple = (0, 0)                       # [KR] 뱅크가 커버하는 (시작, 끝) 프레임 범위
    gt: object = None                       # gt_score closure, or None  [KR] GT 채점 함수(옵션)
    #: optional SECOND, pose-only bank supplying the long rungs the local bank
    #: cannot reach (Delta > L - S).  Either the stitched L96s48 track written as
    #: a one-run bank, or a K_t=2 bank whose runs are long enough for an in-run
    #: pair -- the two candidates are interchangeable here on purpose.
    long_bank: object = None
    # ★ v5: which CORPUS this sequence belongs to.  Dataset-balanced sampling
    # (v5 "Dataset Sampling") selects a dataset first and a sequence second, so
    # the sampler needs the grouping the scene list does not otherwise carry.
    # Inferred from the frames path (data/mcd/... -> "mcd") unless the --scene
    # spec names it explicitly.
    # [KR] 이 시퀀스가 어느 데이터셋(코퍼스)에 속하는지. --dataset_weights로
    # 데이터셋 단위 균형 샘플링을 할 때 그룹핑 키로 쓰인다.
    dataset: str = "mcd"


# ─────────────────────────────────────────────────────────────────────────────
# Rollout pool
# ─────────────────────────────────────────────────────────────────────────────

def next_valid_window(bank: LabelBank, t: int, S: int,
                      skip_runs=()) -> Optional[int]:
    """Smallest start >= t whose whole S-window sits inside ONE bank run.

    Windows may not straddle runs (each run has its own Sim(3); see
    LabelBank.windows), so a stream that walks off the end of a run jumps to the
    next run's start rather than spanning the seam.

    ``skip_runs`` are held out for probing.  Holding out whole RUNS rather than
    individual windows is what makes the probe honest: training windows advance
    by S but a stream can still partially overlap a neighbouring window, so only
    a whole-run reservation guarantees the probe frames are never trained on.

    [KR] t 이상인 시작점 중에서, S 프레임짜리 창(window) 전체가 "하나의 run"
    안에 완전히 들어가는 가장 작은 시작점을 찾는다.
    - run마다 자기만의 Sim(3) 좌표계(gauge)가 있어서, 창이 두 run에 걸쳐
      있으면 안 된다. 그래서 run의 끝에 도달하면 다음 run의 시작으로 점프한다.
    - skip_runs는 probe(평가용 홀드아웃)를 위해 학습에서 제외된 run들. 개별
      window가 아니라 run 전체를 통째로 제외해야, 학습 window가 S만큼씩
      전진하면서 이웃 window와 겹치더라도 probe 프레임은 절대 학습에 쓰이지
      않는다는 것이 보장된다.
    """
    best = None
    for rid, r in enumerate(bank.runs):
        if rid in skip_runs:
            continue
        if t + S <= r["t0"] + r["L"]:
            start = max(t, r["t0"])
            if start + S <= r["t0"] + r["L"]:
                best = start if best is None else min(best, start)
    return best


def locate(bank: LabelBank, t: int, S: int):
    # [KR] 절대 프레임 t가 어느 run에 속하는지 찾고, 그 run 내에서의 상대
    # 오프셋(off)을 함께 반환한다. [t, t+S) 구간이 한 run 안에 완전히 들어가지
    # 않으면 KeyError.
    for rid, r in enumerate(bank.runs):
        if r["t0"] <= t and t + S <= r["t0"] + r["L"]:
            return rid, t - r["t0"]
    raise KeyError(f"[{t}, {t + S}) is not inside a single run")


def long_term(long_crit, bank: LabelBank, hist: dict, t: int, S: int,
              stu_pose: torch.Tensor, dev,
              alt_bank: Optional[LabelBank] = None, alt_deltas=(),
              max_depth: int = 0, long_crit_alt=None):
    """L_long for ONE supervised window, possibly from TWO teacher sources.

    Returns ``(loss_or_None, parts)``.  ``None`` means nothing fired -- the window
    sits too early in its run, the rollout has not walked back far enough since
    its last re-anchor, or the depth cap excluded it.  All three are normal.

    ★ THE TERMS ARE SPLIT ACROSS SOURCES ON PURPOSE, and the GT audit is why.
    Measured against GT on five MCD scenes at Delta=192:

        rotation   in-run L240  1.31 deg      stitched  9.5 - 11.9 deg
        scale      in-run L240  bias -0.245   stitched  bias 0.068 - 0.090

    The stitched track re-anchors every 48 frames, so its labels sit at teacher
    depth 80-127 instead of 272 and its SCALE is three times less biased -- but
    each seam contributes a rotation error, and seven of them span a Delta=319
    pair, which is where the 9.5-11.9 deg comes from.  Taking rotation from the
    stitched track would be feeding the loss the seam error; taking scale from
    the in-run bank would be feeding it the teacher's depth-driven scale drift.

    ★ AND BOTH ENDPOINTS OF ANY ONE PAIR STAY IN ONE GAUGE.  Each source brings
    its own window poses (``tea_win``), so a pair is never half stitched and half
    in-run -- that would make the "relative" target the difference between two
    teachers rather than a trajectory relation.
    """
    rid, off = locate(bank, t, S)
    # ★ THE TEACHER DEPTH THIS WINDOW SITS AT.  In the in-run bank Delta and depth
    # are entangled (long rungs only fit at deep offsets) and the teacher's scale
    # bias tracks DEPTH, so a run that does not report or bound this cannot
    # attribute anything to Delta.
    depth = int(bank.runs[rid]["burn_in"]) + off
    empty = {"long_depth": float(depth), "long_depth_skipped": 0.0}
    if max_depth and depth > max_depth:
        return None, {**empty, "long_depth_skipped": 1.0}

    total, parts = None, dict(empty)

    if long_crit is not None and long_crit.terms:
        pairs = LL.build_pairs(long_crit.ladder, hist,
                               lambda lo, hi: bank.poses(rid, lo, hi, device=dev),
                               bank.runs[rid]["t0"], off, S, dev)
        if pairs:
            tea = bank.poses(rid, off, off + S, device=dev)
            L, p = long_crit(stu_pose, tea, pairs)
            parts.update(p)
            total = L if total is None else total + L

    if long_crit_alt is not None and long_crit_alt.terms and alt_bank is not None:
        try:
            arid, aoff = locate(alt_bank, t, S)
        except KeyError:
            arid = None
        if arid is not None:
            atea = alt_bank.poses(arid, aoff, aoff + S, device=dev)
            apairs = LL.build_pairs(long_crit_alt.ladder, hist,
                                    lambda lo, hi: alt_bank.poses(arid, lo, hi, device=dev),
                                    alt_bank.runs[arid]["t0"], aoff, S, dev,
                                    tea_win=atea)
            if apairs:
                L, p = long_crit_alt(stu_pose, atea, apairs)
                parts.update(p)
                total = L if total is None else total + L

    if total is None:
        for d in (long_crit.ladder if long_crit else ()):
            parts.setdefault(f"long_n_d{d}", 0.0)
        return None, parts
    return total, parts


# ─────────────────────────────────────────────────────────────────────────────
# v5 state mixture: keyframe interval, rollout horizon, dataset
# ─────────────────────────────────────────────────────────────────────────────

#: docs/self-distill-ver5.md "Student Keyframe Sampling".  K=1 and K=28 carry a
#: quarter each; the five intermediate values share the remaining half.
K_DIST_V5 = ((1, 25), (2, 10), (4, 10), (8, 10), (12, 10), (16, 10), (28, 25))

#: "Rollout-Horizon Sampling" -- raw frames, log-spaced.  0 means "no horizon":
#: walk until the scene's bank coverage ends, which is the pre-v5 behaviour.
HORIZONS_V5 = (320, 960, 1920, 3840)


def h_deck(horizons, pool: int) -> List[int]:
    """One rollout horizon per stream, dealt round-robin over the grid.

    ★ THE SAME STEP-WEIGHTING BUG ``k_deck`` FIXES, ON THE OTHER AXIS.  ``--horizons``
    is drawn per rollout and the loss averages per step, but a horizon-3840
    rollout contributes 80 steps and a horizon-320 one contributes 7.  Measured
    on v6a: a uniform draw over {320, 960, 1920, 3840} realised step shares of
    15.7 / 26.4 / 35.0 / 20.4 %, so the shallow end -- the only place a K=1
    student is ever supervised below cache depth 320 -- was the one that got
    squeezed.  Only 69 of 802 correction steps landed there.

    Fixing the horizon to the stream makes the step share exactly
    ``(streams at h) / pool``, for the same reason it does for K: ``pool.next()``
    is a strict round robin.  Unlike K the grid need not divide the pool -- the
    horizons are dealt cyclically, so an uneven pool differs from exact by at
    most one stream per value, which is reported at startup rather than hidden.
    """
    hs = [h for h in horizons if h] or [0]
    return [hs[i % len(hs)] for i in range(pool)]


def parse_k_dist(spec: str):
    """``"1:25,2:10,28:25"`` -> [(1, .25), (2, .10), (28, .25)] (normalised).

    ``"fixed"`` returns None, which every caller reads as "use --K everywhere",
    so a v5 trainer reproduces the pre-v5 run without a second code path.
    """
    if not spec or spec.lower() == "fixed":
        return None
    pairs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, w = tok.partition(":")
        try:
            ki, wf = int(k), (float(w) if w else 1.0)
        except ValueError:
            raise SystemExit(f"--k_dist: cannot parse {tok!r}; want K:weight")
        if ki < 1 or wf < 0:
            raise SystemExit(f"--k_dist: K must be >=1 and weight >=0, got {tok!r}")
        pairs.append((ki, wf))
    tot = sum(w for _, w in pairs)
    if not pairs or tot <= 0:
        raise SystemExit(f"--k_dist: no positive weight in {spec!r}")
    return [(k, w / tot) for k, w in pairs]


def k_deck(k_dist, pool: int, default_K: int) -> List[int]:
    """The pool's per-stream keyframe intervals, as an exact multiset.

    ★ WHY K IS ASSIGNED TO STREAMS INSTEAD OF DRAWN PER ROLLOUT.  ``--k_dist``
    names a distribution over ROLLOUTS, but the loss averages over STEPS, and a
    rollout contributes ``horizon / S`` of them.  The draw is unbiased -- horizon
    is drawn independently of K, so E[steps | K] does not depend on K -- but its
    variance is set by the number of ROLLOUTS, not steps, and the blocks are
    wildly unequal (a horizon-3840 rollout is 80 steps, a horizon-320 one is 7).

    Measured on v5c: 300 steps produced 19 rollouts of 1-22 steps, an effective
    sample size of ``(sum n)^2 / sum n^2 = 11.3``.  The realized step-weighted
    mixture was 44% K=12 and 0% K=8 against a declared 10/10.  Simulation of the
    same schedule says that is TYPICAL, not unlucky: mean L1 from the declared
    mixture 0.56, and at least one K missing entirely in 55% of runs.  v5b, the
    cell the A-vs-B comparison rests on, drew no K=1, K=4 or K=16 at all.

    ``pool.next()`` is a strict round robin, so every stream receives exactly
    ``1 / pool`` of the correction steps.  Assigning K per STREAM therefore makes
    the step-weighted mixture exactly ``(streams at K) / pool`` -- zero variance,
    independent of step count, horizon grid and reset timing.  The cost is that
    ``pool * P(K)`` must be an integer for every K, i.e. the pool has to be a
    multiple of the mixture's common denominator (20 for the ver5 mixture, since
    lcm(4, 10) = 20).  That is checked here rather than left to silently skew.

    Returned SORTED, so ``deck[rank::world]`` splits it evenly: the global
    multiset is exact whatever the split, and sorting keeps the expensive deep
    K=1 snapshots spread across ranks instead of landing on one.
    """
    if not k_dist:
        return [default_K] * pool
    counts, deck = [], []
    for kv, p in k_dist:
        c = pool * p
        if abs(c - round(c)) > 1e-9:
            import math
            from fractions import Fraction
            den = 1
            for _, q in k_dist:                     # lcm of the weights' denominators
                d = Fraction(q).limit_denominator(1000).denominator
                den = den * d // math.gcd(den, d)
            raise SystemExit(
                f"--pool {pool} x P(K={kv})={p:.4f} = {c:.4f} is not an integer, so the "
                f"per-stream assignment cannot realise --k_dist.  The smallest pool that "
                f"matches this mixture is {den} (use it or a multiple).  Alternatively "
                f"pass --k_dist fixed to take --K for every stream.")
        counts.append((kv, int(round(c))))
    for kv, c in counts:
        deck += [kv] * c
    if len(deck) != pool:
        raise SystemExit(f"--k_dist rounds to {len(deck)} streams, not --pool {pool}")
    return sorted(deck)


class StateSampler:
    """Draws (dataset, scene, K, horizon) for a rollout, and the per-step branch.

    ★ TWO DIFFERENT KINDS OF RANDOMNESS, AND THEY MUST NOT SHARE A STREAM.

    ``branch()`` decides identity-vs-correction for a training STEP.  Under DDP
    every rank must reach the same answer: the two branches build different
    autograd graphs, and DDP's ``find_unused_parameters`` reducer derives the
    unused set from the LOCAL graph, so ranks that disagree about which branch
    ran would wait on gradients that never arrive -- a hang, not an error.  It is
    therefore keyed on the step index alone, with no rank term and no carried
    state, so it is identical on every rank and survives a resume unchanged.

    ``k()`` / ``horizon()`` / ``scene()`` describe one rank's own walk.  Those
    SHOULD differ across ranks -- that is the point of partitioning scenes -- so
    they come from a rank-seeded generator.  Nothing downstream couples them
    across ranks: both branches push the same modules through the same masked
    path, so the used-parameter set does not depend on K.
    """

    def __init__(self, k_dist, horizons, p_identity, dataset_weights,
                 seed: int = 0, rank: int = 0):
        import random
        self.k_dist = k_dist
        self.horizons = tuple(horizons) if horizons else (0,)
        self.p_identity = float(p_identity)
        self.dataset_weights = dataset_weights or {}
        self._rng = random.Random((seed * 1000003) ^ (rank * 9176) ^ 0x5EED)
        self._seed = seed

    # -- per-rollout draws (rank-local) ---------------------------------------

    def k(self, default: int) -> int:
        """UNUSED by the pool since K moved to the stream (see ``k_deck``).

        Kept because ``experiments/test_state_mixture.py`` checks that the draw
        matches the declared distribution, which is still the property the deck
        has to reproduce -- and because a caller that genuinely wants an i.i.d.
        per-rollout draw has nowhere else to get one.
        """
        if not self.k_dist:
            return default
        r, acc = self._rng.random(), 0.0
        for kv, p in self.k_dist:
            acc += p
            if r <= acc:
                return kv
        return self.k_dist[-1][0]

    def horizon(self, cap: int) -> int:
        """Raw-frame budget for one rollout, clipped to what the scene has.

        ``cap`` is how many supervised frames remain ahead of the start.  A
        horizon longer than the scene is not an error -- it just means the
        rollout ends at the scene boundary instead of at the horizon -- but
        clipping it here keeps the LOGGED horizon equal to the one actually
        walked, which is what the age plots are read against.
        """
        cands = [h for h in self.horizons if h and h <= cap]
        if not cands:
            return cap if 0 not in self.horizons else 0
        return self._rng.choice(cands)

    def scene(self, by_dataset) -> int:
        """Dataset first, then a sequence inside it (v5 "Dataset Sampling").

        Frame-balanced sampling would let the larger corpus dominate purely by
        frame count -- which is the domain over-specialisation this stage exists
        to fix, so the first multi-domain run samples the dataset uniformly by
        WEIGHT and the sequence uniformly inside it.
        """
        names = [d for d in by_dataset if by_dataset[d]]
        if not names:
            raise SystemExit("[sampler] no scene to draw from")
        if len(names) == 1:
            return self._rng.choice(by_dataset[names[0]])
        w = [max(0.0, float(self.dataset_weights.get(d, 1.0))) for d in names]
        if sum(w) <= 0:
            w = [1.0] * len(names)
        r, acc = self._rng.random() * sum(w), 0.0
        for name, wi in zip(names, w):
            acc += wi
            if r <= acc:
                return self._rng.choice(by_dataset[name])
        return self._rng.choice(by_dataset[names[-1]])

    # -- per-step draw (rank-invariant) ---------------------------------------

    def branch(self, step: int) -> str:
        """"identity" or "correction" for this step -- SAME on every rank.

        Hashed from (seed, step) rather than drawn from ``self._rng`` so that a
        resume at step N replays exactly the schedule an uninterrupted run would
        have taken, and so the two ranks never diverge.
        """
        if self.p_identity <= 0:
            return "correction"
        if self.p_identity >= 1:
            return "identity"
        import hashlib
        h = hashlib.sha256(f"{self._seed}:{step}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / float(1 << 64)
        return "identity" if u < self.p_identity else "correction"


@dataclass
class RolloutStream:
    """One live student walk: where it is, and its streaming state."""
    sid: int
    t: int
    start: int
    scene: int = 0                          # index into RolloutPool.scenes
    state: dict = field(default=None, repr=False)
    windows_done: int = 0
    resets: int = 0
    # ★ v5.  K is a property of the STREAM, assigned once at pool construction
    # and never redrawn.  It must at minimum be constant for the life of a
    # rollout, because the masked path's ``_check_prefix_consistency`` predicts
    # the cached keyframe count as ``ceil((t0 - sf) / K)`` and that identity only
    # holds if one K produced the whole prefix.  Pinning it to the STREAM is
    # strictly stronger, and it is what makes the mixture exact -- see
    # ``k_deck`` for why drawing per rollout could not.
    K: int = 1
    horizon: int = 0                        # raw-frame budget, 0 = to scene end
    anchor_t: int = 0                       # where THIS rollout began, for age
    # ★ L_long: absolute frame -> [9] student pose, filled by ``_roll``.  This is
    # the ONLY record of where the student thought it was in the past; the
    # rollout is no_grad and otherwise throws the output away.
    #
    # ★ CLEARED AT EVERY RE-ANCHOR (``_rebuild``).  A rollout is one Sim(3)
    # gauge -- the anchor block fixes it -- so a pose from before the re-anchor
    # is expressed in a DIFFERENT gauge.  Pairing across that boundary would
    # feed the long term a seam and call it trajectory error.  This is the one
    # place the whole term can be silently wrong.
    hist: dict = field(default_factory=dict, repr=False)


class RolloutPool:
    """M interleaved student walks, snapshotted on the host between turns."""

    def __init__(self, model, scenes: List[Scene], starts, S, sf, K, dtype, dev,
                 sampler: Optional[StateSampler] = None, stream_k=None,
                 stream_h=None):
        """``starts`` is a list of (scene_idx, t0) pairs -- one per stream.

        ``sampler`` is v5's state mixture.  Without it every stream takes ``K``
        and walks to the end of its scene, which is the pre-v5 behaviour
        bit-for-bit; with it each RESET redraws (scene, horizon).

        ``stream_k`` is this rank's slice of the pool-wide K deck (``k_deck``),
        one interval per stream, held for the whole run.  ``None`` means every
        stream takes ``K``.  A resumed stream keeps the K in the checkpoint, so
        the deck only ever seeds a fresh pool.
        """
        from grad_probe import snapshot_state_cpu, restore_state_cpu, detach_caches
        self._snap, self._restore, self._detach = (
            snapshot_state_cpu, restore_state_cpu, detach_caches)
        self.model, self.scenes = model, scenes
        self.S, self.sf, self.K, self.dtype, self.dev = S, sf, K, dtype, dev
        #: frames of student pose history to keep per stream.  0 = RECORD
        #: NOTHING, which is what a --lam_long 0 run wants: no dict write and no
        #: host copy, i.e. the pre-L_long step with no new work of any kind.
        #: main() sets it to max(long_deltas) + 2S when the term is on.
        self.hist_span = 0
        self.sampler = sampler
        # scene indices grouped by corpus, for dataset-balanced redraws
        self.by_dataset: Dict[str, List[int]] = {}
        for i, sc in enumerate(scenes):
            self.by_dataset.setdefault(sc.dataset, []).append(i)
        #: raw frames actually walked per dataset -- v5 asks for per-dataset
        #: revisit counts, because a 50/50 dataset-balanced sampler over corpora
        #: of very different size revisits the small one far more often.
        self.frames_seen: Dict[str, int] = {d: 0 for d in self.by_dataset}
        if stream_k is not None and len(stream_k) != len(starts):
            raise SystemExit(f"stream_k has {len(stream_k)} entries for "
                             f"{len(starts)} streams")
        self.stream_k = list(stream_k) if stream_k is not None else None
        # Fixed per stream for the same reason as K -- see ``h_deck``.  None
        # keeps the pre-v6b behaviour: redrawn from the sampler on every reset.
        self.stream_h = list(stream_h) if stream_h is not None else None
        self.streams: List[RolloutStream] = []
        for i, entry in enumerate(starts):
            k_i = self.stream_k[i] if self.stream_k is not None else K
            if isinstance(entry, dict):
                # Resume: the rollout is CONTINUED, not redrawn.  ``start`` is
                # the saved position, so _rebuild rolls back to exactly it.
                st = RolloutStream(sid=entry.get("sid", i), t=entry["t"],
                                   start=entry["t"], scene=entry["scene"],
                                   # the checkpoint's K wins: a resumed rollout
                                   # must keep the interval its prefix was rolled
                                   # at, or _check_prefix_consistency raises on
                                   # the first step after the restore.
                                   K=entry.get("K", k_i),
                                   horizon=entry.get("horizon", 0),
                                   anchor_t=entry.get("anchor_t", entry["t"]),
                                   windows_done=entry.get("windows_done", 0),
                                   resets=entry.get("resets", 0))
                self._rebuild(st)
            else:
                sc, t0 = entry
                st = RolloutStream(sid=i, t=t0, start=t0, scene=sc, K=k_i, anchor_t=t0)
                self.reset(st)
            self.streams.append(st)
        self._cursor = 0

    # -- streaming primitives -------------------------------------------------

    def _roll(self, sc: int, lo: int, hi: int, K: Optional[int] = None,
              hist: Optional[dict] = None):
        """Advance the live model over scene ``sc``'s [lo, hi), no grad, detached.

        ``K`` is the WALKING stream's interval, not the pool's: under v5 each
        stream owns one, and the phase origin stays the absolute frame ``sf`` so
        the schedule matches ``gca_mask.frame_roles`` exactly.
        """
        K = self.K if K is None else K
        images = self.scenes[sc].images
        for i in range(lo, hi):
            is_kf = (K <= 1) or ((i - self.sf) % K == 0)
            if not is_kf:
                self.model._set_skip_append(True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
                out = self.model.forward(images[:, i:i + 1].to(self.dev),
                                         num_frame_for_scale=self.sf,
                                         num_frame_per_block=1, causal_inference=True)
            if not is_kf:
                self.model._set_skip_append(False)
            # ★ L_long: frame i's pose at the weights that were current when the
            # stream walked it -- exactly the detached anchor the long term
            # wants, and the only place it exists.  9 floats per frame.
            if hist is not None:
                hist[i] = out["pose_enc"][0, 0].detach().float().cpu()
            del out
            self._detach(self.model)
        if hist is not None and self.hist_span:
            for f in [f for f in hist if f < hi - self.hist_span]:
                del hist[f]
        self.frames_seen[self.scenes[sc].dataset] = (
            self.frames_seen.get(self.scenes[sc].dataset, 0) + max(0, hi - lo))

    def _anchor(self, sc: int):
        """The scale-frame block that fixes this scene's gauge.  Per scene: the
        unit is set at the anchor, so a stream must anchor in its OWN sequence."""
        self.model.clean_kv_cache()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            self.model.forward(self.scenes[sc].images[:, :self.sf].to(self.dev),
                               num_frame_for_scale=self.sf,
                               num_frame_per_block=self.sf, causal_inference=True)
        self._detach(self.model)

    def roll_to(self, sc: int, t0: int) -> dict:
        """Fresh anchor -> roll to t0 -> snapshot.  Used for the frozen probes."""
        self._anchor(sc)
        self._roll(sc, self.sf, t0)
        return self._snap(self.model)

    def reset(self, st: RolloutStream):
        """Begin a NEW rollout: redraw (scene, horizon), anchor, roll, snapshot.

        ★ K IS NOT REDRAWN.  It belongs to the stream, not to the rollout -- see
        ``k_deck`` for the measurement that moved it there.  Only the scene and
        the horizon are resampled.

        ★ EVERY ROLLOUT STARTS AT ITS SCENE'S FIRST VALID WINDOW, and depth comes
        from the HORIZON rather than from a deep start.  Starting deep instead
        would mean re-rolling the whole prefix from the anchor on every reset --
        ~50 ms/frame measured, so a reset at t=3000 costs 2.5 min and a 320-frame
        horizon would spend more time rolling in than training.  Sampling the
        horizon instead makes the roll-in a fixed 72 frames and puts the whole
        320..3840 raw-frame age range in the distribution anyway, which is the
        axis v5 asks the results to be plotted against.

        The cost is that frames beyond ``max(horizons)`` of a scene are never
        visited: with --horizons 320 960 1920 3840 a 9684-frame sequence is
        trained over its first ~3900.  ``--horizons 0`` restores the pre-v5
        walk-to-the-end behaviour, and adding a larger value to the list buys
        depth at a proportional cost in wall clock.
        """
        if self.sampler is not None:
            # ★ ALWAYS REDRAW THE SCENE, EVEN WITH ONE DATASET.  Gating this on
            # "more than one corpus" pinned every stream to the scene it started
            # on: with --horizons set, a 10-scene MCD run would reset ~30x per
            # scene and land back on the same sequence every time, so the corpus
            # a stream ever sees is one sequence.  That is the same "corpus
            # silently shrinks to what the streams happen to sit on" failure the
            # pool-size warning exists for, arriving by a different route.
            # With a single dataset ``scene()`` is a uniform draw over it.
            st.scene = self.sampler.scene(self.by_dataset)
            # ★ AND START AT THE FRONT, INCLUDING THE FIRST TIME.  ``starts``
            # spreads the initial pool through each scene, which under v5 would
            # give the first rollout of every stream a different roll-in depth
            # and an age offset that no later rollout shares.  Zeroing it here
            # makes the very first rollout obey the same contract as every
            # reset after it, so raw_frame_age means one thing throughout.
            st.start = 0
            sc = self.scenes[st.scene]
            t0 = next_valid_window(sc.bank, 0, self.S, sc.skip_runs)
            cap = max(0, sc.covered[1] - (t0 if t0 is not None else sc.covered[0]))
            if self.stream_h is not None:
                # Clip to the scene, exactly as StateSampler.horizon does, so the
                # LOGGED horizon is the one actually walked.
                h = self.stream_h[st.sid % len(self.stream_h)]
                st.horizon = h if (h and h <= cap) else cap
            else:
                st.horizon = self.sampler.horizon(cap)
        st.anchor_t = -1                    # _rebuild fills it in from the real t0
        self._rebuild(st)
        st.resets += 1

    def _rebuild(self, st: RolloutStream):
        """Anchor -> roll to ``st.start`` -> snapshot, drawing NOTHING.

        ★ SEPARATE FROM ``reset`` BECAUSE A RESUME MUST NOT REDRAW.  The pool is
        reconstructed from the checkpoint at startup, and if that path went
        through ``reset`` the sampler would hand every stream a new (scene, K,
        horizon) and drop it back at the front of the scene -- silently
        restarting the rollout the checkpoint exists to continue, and breaking
        the "one K per rollout" contract mid-walk.  ``reset`` draws and then
        calls this; the resume path calls this alone.
        """
        sc = self.scenes[st.scene]
        t0 = next_valid_window(sc.bank, st.start, self.S, sc.skip_runs)
        if t0 is None:
            t0 = next_valid_window(sc.bank, 0, self.S, sc.skip_runs)
        if t0 is None:
            raise SystemExit(
                f"scene {sc.name}: no window of {self.S} frames survives holding "
                f"out runs {sorted(sc.skip_runs)} -- lower --probe_runs_per_scene")
        # ★ THE GAUGE BOUNDARY.  Everything before this re-anchor was measured
        # in a different Sim(3); see RolloutStream.hist.
        st.hist.clear()
        self._anchor(st.scene)
        self._roll(st.scene, self.sf, t0, st.K, st.hist if self.hist_span else None)
        st.t = t0
        if st.anchor_t < 0:
            st.anchor_t = t0
        st.state = self._snap(self.model)

    def activate(self, st: RolloutStream):
        self._restore(self.model, st.state, self.dev)
        self._detach(self.model)

    def advance(self, st: RolloutStream):
        """Walk past the supervised window with the freshly updated weights.

        Two things can end a rollout: walking off the end of the scene (as
        before) or exhausting the sampled raw-frame horizon (v5).  Either way
        ``reset`` starts a fresh one with a newly drawn (scene, K, horizon).
        """
        sc = self.scenes[st.scene]
        target = next_valid_window(sc.bank, st.t + self.S, self.S, sc.skip_runs)
        if target is None:                  # walked off the end of its scene
            st.start = 0
            self.reset(st)
            return
        if st.horizon and (target - st.anchor_t) >= st.horizon:
            # The horizon is a budget on the rollout, not on the window: stop
            # before the step that would take the state past it, so the logged
            # raw_frame_age never exceeds the horizon it is attributed to.
            st.start = 0
            self.reset(st)
            return
        self._roll(st.scene, st.t, target, st.K,
                   st.hist if self.hist_span else None)
        st.t = target
        st.state = self._snap(self.model)

    def load(self, state):
        """Restore an arbitrary snapshot (used by the frozen probes)."""
        self._restore(self.model, state, self.dev)
        self._detach(self.model)

    def next(self) -> RolloutStream:
        st = self.streams[self._cursor % len(self.streams)]
        self._cursor += 1
        return st

    def bytes_resident(self) -> int:
        tot = 0
        for st in self.streams:
            for d in ([st.state["agg"]] if st.state["agg"] else []) + (st.state["cam"] or []):
                for v in d.values():
                    if torch.is_tensor(v):
                        tot += v.numel() * v.element_size()
        return tot


# ─────────────────────────────────────────────────────────────────────────────
# Collapse monitoring
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def health(out) -> Dict[str, float]:
    """Cheap degeneracy detectors.  A collapsed depth head is flat; a collapsed
    pose head stops moving.  Both are silent in the loss for a while."""
    d = out["depth"].detach().float()[0, ..., 0]              # [S, H, W]
    S = d.shape[0]
    flat = d.reshape(S, -1)
    contrast = (flat.std(dim=1) / flat.mean(dim=1).clamp(min=1e-6)).mean()
    p = out["pose_enc"].detach().float()[0]
    step = (p[1:, :3] - p[:-1, :3]).norm(dim=-1)
    return {
        "depth_median": float(flat.median()),
        "depth_contrast": float(contrast),      # -> 0 means a constant depth map
        "pose_step_mean": float(step.mean()),   # -> 0 means a frozen camera
        "pose_step_std": float(step.std()),
    }


def grad_norms(model) -> Dict[str, float]:
    out = {}
    for label, pre in PARAM_GROUPS:
        sq = sum(float(p.grad.detach().float().pow(2).sum())
                 for n, p in model.named_parameters()
                 if n.startswith(pre) and p.grad is not None)
        out[label] = sq ** 0.5
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, lr, wd, encoder_lr_scale):
    """Encoder frozen (scale 0) or on a reduced LR.

    §3-T5: the encoder's grad norm dominates every other group (measured 5.7x
    global_blocks in experiments/train_step_check.py).  It sits OUTSIDE the
    contaminated cross-frame path -- it is per-frame -- so letting it take the
    update means the frame-level module absorbs the distillation signal instead
    of the cache-reading policy that the whole project is about.
    """
    enc, rest = [], []
    for n, p in model.named_parameters():
        (enc if n.startswith("aggregator.patch_embed") else rest).append((n, p))
    if encoder_lr_scale == 0.0:
        for _, p in enc:
            p.requires_grad_(False)
    # Only what still requires grad goes to AdamW.  Without --lora_rank that is
    # every non-encoder parameter (unchanged); with it, the adapter alone --
    # inject_lora froze the base, and a frozen parameter in an optimizer group
    # is only dead weight in opt.state_dict().
    rest = [p for _, p in rest if p.requires_grad]
    enc = [p for _, p in enc if p.requires_grad]
    if encoder_lr_scale == 0.0 or not enc:
        groups = [{"params": rest, "lr": lr, "lr_scale": 1.0}]
    else:
        groups = [{"params": rest, "lr": lr, "lr_scale": 1.0},
                  {"params": enc, "lr": lr * encoder_lr_scale,
                   "lr_scale": encoder_lr_scale}]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)


def _export_weights(model, args) -> Dict[str, object]:
    """The weight entries of a checkpoint.

    Full fine-tune: {"model": state_dict} as before.  LoRA: {"model": MERGED
    state_dict, "lora": adapter, "lora_cfg": flags} -- "model" keeps the released
    key set so _strip_optim.py / the benchmark / the MCD ladder read it
    unchanged, and --resume rebuilds from "lora" (lora.py module docstring).
    """
    if args.lora_rank > 0:
        return {"model": LORA.merged_state_dict(model),
                "lora": LORA.lora_state_dict(model),
                "lora_cfg": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                             "dropout": args.lora_dropout, "targets": args.lora_targets,
                             "modules": args.lora_modules}}
    return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}


class L2SP:
    """Decoupled pull toward the released weights: p -= lr * coef * (p - p0).

    Kept out of the graph on purpose -- as a penalty term it would add a full
    parameter-sized backward every step for no benefit.
    """

    def __init__(self, model, coef):
        self.coef = coef
        self.ref = ({n: p.detach().clone() for n, p in model.named_parameters()
                     if p.requires_grad} if coef > 0 else {})

    @torch.no_grad()
    def apply(self, model, lr):
        if not self.ref:
            return
        for n, p in model.named_parameters():
            if n in self.ref:
                p.add_(p - self.ref[n], alpha=-lr * self.coef)

    def bytes(self):
        return sum(v.numel() * v.element_size() for v in self.ref.values())


def make_gt_score(frames_dir: str, calib: str, sensor: str):
    """Umeyama-aligned local ATE/rot of one window against GT, for ONE scene.

    METRIC ONLY -- GT never enters the loss (the method is unlabelled by
    construction, v4 section 3.3).  The probe loss says "imitates the teacher
    better", not "is closer to GT", and those came apart in practice: two probes
    disagreed on which lambda was better and only GT settled it (gate 6b).

    ★ THE TWO INDEXINGS DO NOT AGREE.  The trainer and label_bank walk the frames
    directory; the eval scripts walk meta["names"].  For kth_day_06 the directory
    holds 8907 entries and meta lists 8894 -- lead-in frames with no GT.  Training
    is unaffected (student inputs and bank labels share one indexing) but scoring
    with the training index is off by 11 frames, which barely moves ATE after the
    Sim(3) fit and inflates relative rotation ~10x.  So map name -> row and refuse
    to score anything non-contiguous, rather than silently measuring the wrong
    window -- the section 5.5 failure mode.
    """
    import numpy as np
    from mcd_eval import load_extrinsic, gt_camera_poses
    from ate_vs_distance import local_scores

    meta = np.load(os.path.join(frames_dir, "meta.npz"), allow_pickle=True)
    T, _, _ = load_extrinsic(calib, sensor)
    gp, gq = gt_camera_poses(meta["gt_pos"], meta["gt_quat"], T)
    disk = image_names(frames_dir)
    row_of = {str(n): i for i, n in enumerate(meta["names"])}
    gt_row = np.array([row_of.get(n, -1) for n in disk], dtype=np.int64)

    def score(pose_enc, t, S_):
        rows = gt_row[t:t + S_]
        if len(rows) < S_ or rows[0] < 0 or (np.diff(rows) != 1).any():
            raise RuntimeError(
                f"window [{t}, {t + S_}) of {frames_dir} does not map to a "
                f"contiguous GT range ({len(disk)} files vs {len(meta['names'])} "
                f"GT rows) -- refusing to score against misaligned GT")
        g0 = int(rows[0])
        pe = pose_enc.detach().float().cpu().double().numpy()
        a, med, rot, sc = local_scores(pe[:, :3], pe[:, 3:7],
                                       gp[g0:g0 + S_], gq[g0:g0 + S_])
        return {"gt_ate": a, "gt_ate_med": med, "gt_rot_deg": rot, "gt_scale": sc}

    off = int(gt_row[gt_row >= 0][0]) - int(np.argmax(gt_row >= 0))
    print(f"[gt] {os.path.basename(os.path.dirname(frames_dir))}: GT scoring ON "
          f"({os.path.basename(calib)}, {sensor}) -- metric only; "
          f"frame -> meta row offset {off:+d} "
          f"({len(disk)} files vs {len(meta['names'])} GT rows)")
    return score


def lr_at(step, base_lr, warmup):
    return base_lr * min(1.0, (step + 1) / max(warmup, 1))




def supervised_step(model, images, bank, t, S, sf, K, crit, dtype, dev, grad=True,
                    fwd=None, return_lab=False):
    """One masked-window forward + loss against the bank.  Does not touch state.

    [KR] "교정(correction)" 브랜치의 핵심 함수: 이미 롤아웃으로 만들어진
    스트리밍 상태(캐시) 위에서, [t, t+S) window를 마스킹된(masked) 경로로
    forward하고, 그 결과를 뱅크의 라벨(교사가 미리 만들어둔 라벨)과 비교해서
    loss를 낸다. model.masked_window() 컨텍스트 안에서 병렬로(parallel) 계산
    하지만, 실제로 캐시 상태 자체는 건드리지 않는다(read-only 성질 -- 그래서
    나중에 pool.advance()로 안전하게 "다시" 이 구간을 순차적으로 걸을 수 있음).
    """
    rid, off = locate(bank, t, S)
    lab = bank.get(rid, off, S, device=dev)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        with model.masked_window(window_start=t, keyframe_interval=K):
            with torch.amp.autocast("cuda", dtype=dtype):
                out = (fwd or model)(images[:, t:t + S].to(dev),
                                     num_frame_for_scale=sf, num_frame_per_block=S,
                                     causal_inference=False)
        total, parts = crit(out["pose_enc"][0].float(), lab["pose_enc"],
                            out["depth"][0].float(), lab["depth"],
                            lab.get("depth_conf"))
    # ★ v5 step 1 asks for burn_in_effective PER WINDOW, and this is the only
    # place that knows it.  The banks were generated at --L 240, so one teacher
    # rollout supervises 240 contiguous frames; a window at offset ``off`` inside
    # such a run is scored against a teacher whose cache was already ``off``
    # frames deeper than the nominal burn-in when it produced that target.  The
    # spec's 8 + 72 + 48 = 128 teacher window is the off == 0 case only.
    r = bank.runs[rid]
    parts["run"] = float(rid)
    parts["offset"] = float(off)
    parts["burn_in_eff"] = float(r["burn_in"] + off)
    if return_lab:
        # L_abs scores the same window against the same labels; hand them back
        # rather than slicing the bank (and copying the depth to the card) twice
        return total, parts, out, lab
    return total, parts, out



def fresh_step(model, scene, rid, S, crit, dtype, dev, fwd=None):
    """docs/add_loss.md step 3 -- L_fresh = D(S_theta^fresh, T_theta0^fresh).

    The long branch asks "with a drifted cache, do you still agree with a clean
    run".  Nothing in it asks "are you still the model you started as", and with
    L2-SP removed nothing else does either: 1200 steps of pulling the contaminated
    state toward the teacher can drag the FRESH behaviour along with it, and the
    loss would never notice, because the teacher it is scored against was frozen
    before any of that happened.

    So run the student the way the teacher ran -- same anchor, same burn-in, same
    K_t -- and score it against the SAME bank labels.  At step 0 the two are the
    same weights and this is ~0 by construction; it grows only as fresh behaviour
    moves, which is exactly the thing to hold still.

    ★ RELATIVE INDICES ARE WHAT MAKE THIS LEGAL.  ``gca_mask.frame_roles`` fixes
    the anchor at absolute frames [0, sf) and the keyframe phase at sf, so a run
    anchored at a0 != 0 cannot be described in absolute coordinates.  Slicing the
    scene from a0 makes the anchor land on [0, sf) and the phase origin on sf --
    and the teacher's own phase origin was a0+sf (label_bank.generate_run), so the
    two coincide exactly.  No change to the verified mask code.

    [KR] "fresh(신선함) 보존" 브랜치: docs/add_loss.md step 3의
    L_fresh = D(학생의 fresh 결과, theta_0(초기 가중치)의 fresh 결과).
    - 목적: L_long(장기 상관 항)은 "드리프트된 캐시로도 여전히 깨끗한 롤아웃과
      동의하는가"만 묻는다. "지금도 초기 모델과 같은 행동을 하는가"는 아무도
      묻지 않으므로, L2-SP를 끄면 1200스텝 동안 교사 쪽으로 당기다가 fresh한
      행동(초기 상태에서의 반응) 자체가 조용히 망가져도 loss는 눈치채지 못한다
      (채점 대상인 교사 라벨이 그 전에 이미 고정(frozen)되었기 때문).
    - 방법: 교사가 이 창을 만들었을 때와 완전히 동일한 방식(같은 앵커, 같은
      burn-in, 같은 K_t)으로 학생을 다시 굴려서, 같은 뱅크 라벨로 채점한다.
      스텝 0에서는 두 가중치가 같으므로 이 loss는 구성상 거의 0이고, fresh
      행동이 실제로 변할 때만 커진다.
    - 상대 인덱스를 쓰는 이유: gca_mask.frame_roles가 앵커를 절대 프레임
      [0, sf)에 고정하므로, 앵커가 a0!=0인 run은 절대 좌표로 표현 불가능하다.
      장면을 a0부터 슬라이스하면 앵커가 다시 [0, sf)에 오고, 위상 원점도
      sf에 와서 교사의 원점(a0+sf, label_bank.generate_run)과 정확히 일치한다.
    """
    r = scene.bank.runs[rid]
    sf_b, B, Kt = r["scale_frames"], r["burn_in"], r["teacher_interval"]
    a0 = r["t0"] - B * Kt - sf_b
    ws = r["t0"] - a0                       # window start in the sliced frame
    # ★ Bank runs are NOT all >= S.  generate_run stops at the end of a sequence,
    # so the last run of a scene is whatever was left (42 frames on rank1's run
    # 18), and asking a 42-frame run for a 48-frame window raises out of
    # label_bank.get.  The long branch never hit this because its stream only
    # ever enters runs via next_valid_window, which checks the fit; the fresh
    # branch picks a run directly and has to check for itself.
    if a0 < 0 or r["L"] < S:
        return None, None
    sub = scene.images[:, a0:r["t0"] + S]   # view, no copy

    from grad_probe import detach_caches
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.forward(sub[:, :sf_b].to(dev), num_frame_for_scale=sf_b,
                      num_frame_per_block=sf_b, causal_inference=True)
    detach_caches(model)
    for i in range(sf_b, ws):
        is_kf = (Kt <= 1) or ((i - sf_b) % Kt == 0)
        if not is_kf:
            model._set_skip_append(True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(sub[:, i:i + 1].to(dev), num_frame_for_scale=sf_b,
                          num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        detach_caches(model)

    lab = scene.bank.get(rid, 0, S, device=dev)
    with model.masked_window(window_start=ws, keyframe_interval=Kt):
        with torch.amp.autocast("cuda", dtype=dtype):
            out = (fwd or model)(sub[:, ws:ws + S].to(dev), num_frame_for_scale=sf_b,
                                 num_frame_per_block=S, causal_inference=False)
    total, parts = crit(out["pose_enc"][0].float(), lab["pose_enc"],
                        out["depth"][0].float(), lab["depth"], lab.get("depth_conf"))
    # Same fields FreshPool.supervise publishes, so the two fresh modes are
    # interchangeable to everything downstream -- the v5 logger and the unified
    # sampler both read these and would otherwise KeyError on the window path.
    # This branch always scores offset 0, so burn_in_eff is the nominal B.
    parts["depth_kf"] = float(ws)
    parts["offset"] = 0.0
    parts["run"] = float(rid)
    parts["burn_in_eff"] = float(B)
    parts.update(health(out))
    del out
    return total, parts


@dataclass
class FreshStream:
    """One dense (K_t=1) walk over a single bank run.

    [KR] fresh_mode="walk"에서 쓰이는, 뱅크의 run 하나를 촘촘하게(dense,
    교사의 K_t 그대로) 걷는 walk 하나. RolloutStream과 비슷한 역할이지만
    "correction"이 아니라 "identity/fresh" 브랜치 전용이다.
    """
    scene: int = 0
    rid: int = -1
    a0: int = 0                             # run anchor, absolute frame index
    ws: int = 0                             # sf + B*K_t, in a0-relative coords
    sf: int = 8
    Kt: int = 1
    L: int = 0
    offset: int = 0                         # label offset inside the run  [KR] run 내에서 현재 라벨 오프셋
    state: dict = field(default=None, repr=False)
    runs_done: int = 0
    windows_done: int = 0
    #: same contract as RolloutStream.hist -- absolute frame -> [9] pose,
    #: cleared in begin_run because that is where this walk re-anchors.
    hist: dict = field(default_factory=dict, repr=False)


class FreshPool:
    """The fresh branch as a WALK instead of a per-step replay.

    [KR] ─────────────────────────────────────────────────────────────────
    fresh_step()은 매 스텝마다 교사의 prefix를 처음부터 재현(replay)하고
    항상 offset 0만 채점하기 때문에, 240프레임짜리 run 중 겨우 48프레임
    (깊이 80~128)만 지도학습되고 나머지 192프레임의 라벨은 전혀 안 쓰인다.
    측정 결과 오히려 손상이 심한 쪽(깊이 190 이후)은 이 좁은 범위 밖에 있었다.

    FreshPool은 이를 개선해서, run 하나당 앵커를 "한 번만" 하고 그 뒤로는
    S 프레임씩 순차적으로 걸어가며(walk) 매번 다른 offset을 채점한다.
    그러면 run 하나가 깊이 sf+B*K_t부터 320(=배포 예산 전체)까지 훑게 되고,
    비용도 오히려 더 싸다(80프레임짜리 prefix를 스텝마다 X run당 한 번만
    지불하면 되므로). 유일하게 달라지는 특성: offset o에서의 상태가 이제
    "현재(최신) 가중치"로 만들어진다는 점(오래된 prefix + 최근 advance들의
    조합) -- 이는 롱풀(long pool)이 원래 갖고 있던 온폴리시 드리프트와 같은
    성질이며, 라벨 자체는 여전히 theta_0의 것이므로 여전히 초기 모델 쪽으로
    당기는 보존(preservation) 항으로 기능한다.
    ─────────────────────────────────────────────────────────────────────

    ★ WHY.  ``fresh_step`` rebuilds the teacher's prefix from scratch every step
    and always scores ``bank.get(rid, 0, S)`` -- offset ZERO.  So the dense
    branch only ever sees the cache between depth ``sf + B*K_t`` and
    ``sf + B*K_t + S`` (80 -> 128 keyframes with this bank), and the labels for
    the other L-S = 192 frames of every 240-frame run are never used at all.

    That placement was assumed, not measured, and the measurement says it is in
    the wrong half.  Oxford bodleian-library-02 is 320 frames, so auto-K is 1
    and the frame index IS the cache depth; local ATE (per-window Sim(3)) of the
    distilled checkpoints against the frozen baseline:

        depth [ 80,180)   a2.s250 / base = 1.13     <- what this branch covers
        depth [180,280)   a2.s250 / base = 3.76
        depth [220,320)   a2.s250 / base = 2.50

    The damage is entirely past depth ~190; the replayed branch sits inside the
    healthy half and never sees the broken one.  It is not scene difficulty --
    the GT step size is flat (1.62-1.74 m/frame) across the whole run -- and it
    is not caused by L_fresh, since a1 (lam_fresh=0) breaks in the same place,
    only harder (ratio 4.4 over [220,320)).

    ★ WHAT CHANGES.  Keep the dense state instead of throwing it away: anchor
    once per run, then walk S frames at a time.  One run then sweeps depth
    ``sf + B*K_t`` -> 320, which is the WHOLE deployment budget -- the bank is
    sized ``8 + 72 + 240 = 320`` for exactly this reason (``max_L_allowed``).
    Nothing new has to be generated; the labels already exist.

    ★ IT IS ALSO CHEAPER.  The sf+B prefix costs 80 sequential single-frame
    forwards and was paid once per STEP; here it is paid once per RUN, i.e.
    amortised over ceil(L/S) = 5 windows (16/step), and each step adds the same
    S-frame advance the long pool already does.  80 -> ~64 forwards per step for
    5x the depth coverage.

    ★ THE EQUIVALENCE THAT MAKES IT LEGAL is the same one ``fresh_step``
    documents: ``gca_mask.frame_roles`` pins the anchor to absolute frames
    [0, sf) and the keyframe phase to sf, so a run anchored at a0 != 0 is only
    describable in a0-relative coordinates.  Slicing the scene at a0 puts the
    anchor back on [0, sf), and the teacher's own phase origin was a0+sf
    (``label_bank.generate_run``), so the two coincide at EVERY offset -- not
    just at offset 0.  The step-0 sanity check therefore gets stronger, not
    weaker: the loss is ~0 by construction across the whole depth sweep.

    One property does change and is worth naming: the state at offset o is now
    produced by the CURRENT weights (a stale prefix plus recent advances), not
    replayed from theta_0.  That is the same on-policy drift the long pool has
    always had, and the labels are still theta_0's, so the branch keeps pulling
    back toward the starting model.  It is a preservation term with an on-policy
    input distribution, not a pure output-space anchor.
    """

    def __init__(self, model, scenes: List[Scene], S, dtype, dev, n_streams=1,
                 cursor=0, keep_hist: bool = False,
                 dataset_weights: Optional[Dict[str, float]] = None):
        # [KR] fresh 브랜치가 걸을 수 있는 (장면, run) 후보 목록을 만든다.
        # 장면-우선(scene-major) 순서로 정렬해서, 연속된 run들이 서로 다른
        # 장면에서 오도록 한다(probe 집합을 만드는 방식과 같은 이유).
        # a0<0(시퀀스 시작보다 앞서 앵커링해야 하는 run)이거나 L<S(윈도우보다
        # 짧은 run, 보통 시퀀스 맨 마지막 남은 조각)는 후보에서 제외한다.
        from grad_probe import snapshot_state_cpu, restore_state_cpu, detach_caches
        self._snap, self._restore, self._detach = (
            snapshot_state_cpu, restore_state_cpu, detach_caches)
        self.model, self.scenes, self.S = model, scenes, S
        self.dtype, self.dev = dtype, dev
        #: record past poses for L_long.  Off by default so the identity branch
        #: is byte-identical to its pre-L_long self when the term is disabled.
        self.keep_hist = keep_hist

        # Candidates, SCENE-MAJOR.  Ordering run-major would spend the first
        # len(runs[0]) run changes inside one sequence; scene-major makes
        # consecutive runs come from different scenes, which is the same reason
        # the probe set is built this way.  a0 < 0 is dropped (a run whose
        # anchor would fall before the start of the sequence) and so is L < S,
        # which fresh_step had to check for itself -- the last run of a scene is
        # whatever was left over (40 frames on kth_day_06's run 12).
        per_scene = []
        for sc in scenes:
            keep = []
            for rid, r in enumerate(sc.bank.runs):
                if rid in sc.skip_runs or r["L"] < S:
                    continue
                if r["t0"] - r["burn_in"] * r["teacher_interval"] - r["scale_frames"] < 0:
                    continue
                keep.append(rid)
            per_scene.append(keep)
        # ★ --dataset_weights APPLIES HERE TOO, and used not to.  Scene-major
        # was chosen so consecutive runs come from different scenes, and that
        # part is kept -- but with 214 scenes whose per-dataset counts differ 19x
        # it silently decided the MIXTURE as well: the first sweep of _cand is
        # one run from each scene, which was 70% dl3dv (150 of 214 scenes)
        # against a requested 10%, and measured, the identity arm's training
        # windows hit 100% dl3dv by step 300 and 68% over the whole run.  The
        # correction branch's sampler has honoured these weights since v5; this
        # path simply never asked.
        #
        # The per-dataset lists stay scene-major, and they are interleaved by a
        # smooth weighted round-robin -- deterministic, no RNG -- so the cursor
        # stays a plain sequential index and --resume still restores the exact
        # position it stored.
        groups: Dict[str, List[Tuple[int, int]]] = {}
        for j in range(max((len(h) for h in per_scene), default=0)):
            for si, h in enumerate(per_scene):
                if j < len(h):
                    groups.setdefault(scenes[si].dataset, []).append((si, h[j]))
        wts = {d: max(1e-9, float((dataset_weights or {}).get(d, 1.0)))
               for d in groups}
        deadline = {d: 0.0 for d in groups}
        take = {d: 0 for d in groups}
        self._cand = []
        for _ in range(sum(len(g) for g in groups.values())):
            d = min(deadline, key=lambda x: (deadline[x], x))
            g = groups[d]
            self._cand.append(g[take[d] % len(g)])
            take[d] += 1
            deadline[d] += 1.0 / wts[d]
        if groups:
            share = {d: sum(1 for si, _ in self._cand
                            if scenes[si].dataset == d) / max(len(self._cand), 1)
                     for d in groups}
            tw = sum(wts.values())
            print("[fresh] candidate mixture "
                  + "  ".join(f"{d}={share[d]:.0%}(want {wts[d] / tw:.0%})"
                              for d in sorted(groups)))
        if not self._cand:
            raise SystemExit("[fresh] no bank run survives (L >= S, a0 >= 0, not "
                             "held out) -- lower --S or --probe_runs_per_scene")
        self._cursor = cursor
        self.streams = [FreshStream() for _ in range(max(1, n_streams))]
        self._turn = 0
        for st in self.streams:
            self.begin_run(st)

    # -- primitives -----------------------------------------------------------

    def _frames(self, st: FreshStream, lo: int, hi: int):
        """a0-relative ``[lo, hi)`` as ABSOLUTE indices into the scene.

        ★ NEVER SLICE OPEN-ENDED INTO ``scene.images``.  It is a MemmapFrames and
        its ``__getitem__`` must copy (wrapping a read-only mapping SIGSEGVs --
        see the class docstring), so the old ``images[:, st.a0:]`` copied every
        frame from the anchor to the END OF THE FILE: 15 GB for kth_day_10, 23.6
        GB for kth_night_01, at a measured 0.3-3 GB/s.  ``_cand`` is scene-major
        so the runs actually used are the first few of each scene, i.e. a0 in
        {0, 240, 480} -- very nearly the whole file, every time.

        It ran twice per identity step (supervise + advance->_walk), which is
        what made identity steps cost 21.9 s against correction's 14.0 s and
        spike to 108 s.  Bound the slice at the call site instead.

        [KR] st.a0을 기준으로 한 상대 좌표 [lo, hi)를, 장면(scene)의 절대
        인덱스로 변환해서 슬라이스를 반환한다. 절대 열린(open-ended) 슬라이스
        (images[:, a0:])를 절대 쓰면 안 된다 -- MemmapFrames는 요청받은 만큼만
        복사하도록 설계되어 있는데, 열린 슬라이스는 "파일 끝까지" 복사해버려서
        장면 하나당 최대 23.6GB를 매번 복사하게 된다. 그래서 항상 여기서
        [lo, hi) 범위를 명시적으로 제한한다.
        """
        return self.scenes[st.scene].images[:, st.a0 + lo:st.a0 + hi]

    def _walk(self, st: FreshStream, lo: int, hi: int):
        """Advance the live model over a0-relative [lo, hi), no grad, detached.

        [KR] a0 기준 상대 좌표 [lo, hi) 구간을 한 프레임씩 실제로 걸어간다
        (no_grad, detach). Kt(교사 키프레임 간격)마다 한 번만 캐시에 append.
        """
        for i in range(lo, hi):
            is_kf = (st.Kt <= 1) or ((i - st.sf) % st.Kt == 0)
            if not is_kf:
                self.model._set_skip_append(True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
                out = self.model.forward(self._frames(st, i, i + 1).to(self.dev),
                                         num_frame_for_scale=st.sf,
                                         num_frame_per_block=1, causal_inference=True)
            if not is_kf:
                self.model._set_skip_append(False)
            # ``_walk`` is a0-relative; hist keys are ABSOLUTE, like the long
            # pool's, so build_pairs can index both with bank frame ids.
            if self.keep_hist:
                st.hist[st.a0 + i] = out["pose_enc"][0, 0].detach().float().cpu()
            del out
            self._detach(self.model)

    def begin_run(self, st: FreshStream):
        """Move this stream to the next run: anchor, burn in, snapshot.

        [KR] 후보 목록에서 다음 (장면, run)을 골라, 그 run의 교사와 완전히
        같은 방식으로 앵커링하고 burn-in만큼 걸은 뒤 스냅샷을 저장한다.
        새 run으로 넘어가는 것이므로 게이지(gauge)가 바뀌어 hist도 초기화한다.
        """
        si, rid = self._cand[self._cursor % len(self._cand)]
        self._cursor += 1
        sc = self.scenes[si]
        r = sc.bank.runs[rid]
        st.scene, st.rid = si, rid
        st.sf, st.Kt, st.L = r["scale_frames"], r["teacher_interval"], r["L"]
        st.a0 = r["t0"] - r["burn_in"] * st.Kt - st.sf
        st.ws = r["t0"] - st.a0                  # == sf + burn_in*K_t
        st.offset = 0
        st.hist.clear()                      # re-anchor -> new gauge
        self.model.clean_kv_cache()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            self.model.forward(self._frames(st, 0, st.sf).to(self.dev),
                               num_frame_for_scale=st.sf,
                               num_frame_per_block=st.sf, causal_inference=True)
        self._detach(self.model)
        self._walk(st, st.sf, st.ws)
        st.state = self._snap(self.model)
        st.runs_done += 1

    def activate(self, st: FreshStream):
        # [KR] RolloutPool.activate와 동일한 역할: 이 fresh 스트림의 스냅샷을
        # 라이브 모델에 복원한다.
        self._restore(self.model, st.state, self.dev)
        self._detach(self.model)

    def next(self) -> FreshStream:
        # [KR] fresh 스트림들 사이에서도 라운드로빈으로 다음 것을 고른다.
        st = self.streams[self._turn % len(self.streams)]
        self._turn += 1
        return st

    # -- the branch itself ----------------------------------------------------

    def supervise(self, st: FreshStream, crit, fwd=None, with_health=False,
                  long_crit=None, lam_long: float = 0.0, alt_deltas=(),
                  max_depth: int = 0, long_crit_alt=None,
                  abs_crit=None, abs_units=None, abs_max_offset: int = 0):
        """One masked window at the CURRENT offset, scored against the bank.

        Call with this stream's state installed (``activate``), and run the
        backward before anything swaps the cache -- checkpoint recompute pairs
        the mask built here with whatever cache is live at backward time.  See
        the long note in the training loop.

        [KR] 현재 offset에서의 window 하나를 마스킹 forward하고 뱅크 라벨로
        채점한다. 반드시 activate() 직후, 그리고 다른 무언가가 캐시를 바꾸기
        "전"에 backward까지 끝내야 한다 (체크포인트 재계산이 forward 시점의
        mask를 backward 시점에 "라이브인 캐시"와 짝지어 재사용하기 때문 --
        학습 루프의 긴 주석 참고). lam_long>0이면 L_long도 같은 out으로
        추가 forward 없이 계산해서 합친다.
        """
        sc = self.scenes[st.scene]
        ws = st.ws + st.offset
        lab = sc.bank.get(st.rid, st.offset, self.S, device=self.dev)
        with self.model.masked_window(window_start=ws, keyframe_interval=st.Kt):
            with torch.amp.autocast("cuda", dtype=self.dtype):
                out = (fwd or self.model)(self._frames(st, ws, ws + self.S).to(self.dev),
                                          num_frame_for_scale=st.sf,
                                          num_frame_per_block=self.S,
                                          causal_inference=False)
        total, parts = crit(out["pose_enc"][0].float(), lab["pose_enc"],
                            out["depth"][0].float(), lab["depth"],
                            lab.get("depth_conf"))
        # ★ THE IDENTITY LADDER IS FREE.  This walk is dense at the bank's own
        # K_t and its labels ARE theta_0's output, so "same K, same context,
        # frozen target" -- what the design doc asks for -- holds by
        # construction; nothing new has to be generated.  The whole branch is
        # scaled by lam_fresh at the call site, so the long part rides along
        # with it and stays one preservation term rather than two objectives.
        if long_crit is not None and lam_long > 0:
            l_long, long_parts = long_term(
                long_crit, sc.bank, st.hist,
                sc.bank.runs[st.rid]["t0"] + st.offset, self.S,
                out["pose_enc"][0].float(), self.dev,
                alt_bank=getattr(sc, "long_bank", None), alt_deltas=alt_deltas,
                max_depth=max_depth, long_crit_alt=long_crit_alt)
            parts.update(long_parts)
            if l_long is not None:
                total = total + lam_long * l_long
        # ★ L_abs ON THE IDENTITY BRANCH IS THE SAME TERM AT THE IDENTITY GAUGE.
        # This walk anchors where the teacher did, so student and reference
        # share one unit by construction: s = 1, R = I, t = 0, no history fit
        # (docs/gtabs-plan.md §2-3).  Expected residual is the global offset
        # results-ledger N14 measured (+9%), nothing offset-dependent -- it is
        # here so both branches run one code path, not for its signal.
        if abs_crit is not None:
            l_abs, abs_parts = AL.abs_term(
                abs_crit, sc.bank, None, sc.bank.runs[st.rid]["t0"] + st.offset, self.S,
                out["pose_enc"][0].float(), out["depth"][0].float(), lab, self.dev,
                identity=True, max_offset=abs_max_offset, units=abs_units)
            parts.update(abs_parts)
            if l_abs is not None:
                total = total + l_abs
        # The whole point of this class: log WHERE in the depth sweep each step
        # landed, so "did the branch actually reach past 190" is answerable from
        # the run log instead of by re-deriving it from offsets.
        parts["depth_kf"] = float(ws)
        parts["offset"] = float(st.offset)
        parts["run"] = float(st.rid)
        # v5: the identity branch's teacher window is 8 + 72 + 48 only at offset
        # 0; every later offset of the same run is a deeper effective burn-in.
        # Logged here for the same reason as in supervised_step.
        parts["burn_in_eff"] = float(sc.bank.runs[st.rid]["burn_in"] + st.offset)
        # ★ COLLAPSE DETECTION MUST NOT DEPEND ON THE BRANCH.  Under the unified
        # sampler an identity step runs no long window, so ``health(out)`` in the
        # training loop has nothing to read and a run that collapsed on identity
        # steps would show no contrast reading at all on those steps -- the gate
        # would be evaluated on a subsample without anyone saying so.  Computed
        # here, before ``out`` is dropped, so every step has one.
        if with_health:
            parts.update(health(out))
        del out
        return total, parts

    def advance(self, st: FreshStream):
        """Walk past the supervised window with the freshly updated weights.

        Same contract as RolloutPool.advance, and it must be called with this
        stream's state installed.  At the end of a run the stream re-anchors on
        the next one, which is where the sf+B prefix cost is paid.

        [KR] 방금 지도학습한 window를 지나쳐서 S프레임만큼 걷는다. run 끝에
        도달하면 begin_run()으로 다음 run에 새로 앵커링한다(이때만 sf+B
        prefix 비용을 지불).
        """
        nxt = st.offset + self.S
        st.windows_done += 1
        if nxt + self.S > st.L:
            self.begin_run(st)
            return
        self._walk(st, st.ws + st.offset, st.ws + nxt)
        st.offset = nxt
        st.state = self._snap(self.model)

    def bytes_resident(self) -> int:
        tot = 0
        for st in self.streams:
            if not st.state:
                continue
            for d in ([st.state["agg"]] if st.state["agg"] else []) + (st.state["cam"] or []):
                for v in d.values():
                    if torch.is_tensor(v):
                        tot += v.numel() * v.element_size()
        return tot


def _cap_threads():
    """Hold this process to a fixed CPU budget.

    ★ OMP_NUM_THREADS IS NOT THE WHOLE BUDGET.  It caps torch's INTRAOP pool;
    the INTEROP pool defaults to the core count independently (72 here), and
    ``load_and_preprocess_images`` adds a 16-thread ThreadPoolExecutor on top.
    Measured: one trainer with OMP_NUM_THREADS=32 still ran 100 threads.  Two of
    them on 72 cores is the thrash that docs/phase1-plan.md section 5.2 warns
    about ("프로세스당 199스레드", model init 19 s -> 10 min+), and it is how the
    first concurrent A1/A2 pair lost a run with no traceback.

    set_num_interop_threads must be called before any inter-op parallel work has
    started, so this runs first thing in main() and tolerates the RuntimeError if
    something upstream already initialised the pool.

    [KR] 이 프로세스가 쓰는 CPU 스레드 수를 고정 예산으로 제한한다.
    OMP_NUM_THREADS 같은 환경변수만으로는 부족하다(torch의 intraop 풀만 제한
    되고, interop 풀과 이미지 전처리용 스레드풀은 별개로 코어 수만큼 커짐).
    한 프로세스가 72코어에서 199스레드까지 뛴 적이 있고, 두 학습을 동시에
    돌리면 스레드 경합(thrash)으로 모델 초기화가 19초에서 10분+로 늘어난다.
    torch.set_num_threads()/set_num_interop_threads()는 이 빌드에서 CUDA
    초기화와 순서가 겹치면 SIGSEGV가 나서 쓸 수 없고, 대신 환경변수만으로
    OpenMP 런타임 스레드 수를 제한한다.
    """
    n = int(os.environ.get("LINGBOT_THREADS", "0"))
    if n <= 0:
        return
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)
    # ★ ENV VARS ONLY.  torch.set_num_threads() SIGSEGVs both ranks under
    # torchrun on this build -- isolated by bisection: the identical script
    # without the call runs clean, with it dies, in either order relative to
    # cuda.set_device()/init_process_group().  OMP_NUM_THREADS is read by the
    # OpenMP runtime at first use and achieves the same cap without the crash.
    # ★ DO NOT call torch.set_num_interop_threads() here either.  On this build it makes
    # the process SIGSEGV later, inside build_model, once several scene memmaps
    # are open -- reproduced deterministically, and it disappears the moment this
    # call is removed.  It also was not buying anything: with intraop capped, the
    # interop pool sits idle (measured 1 runnable thread of 67, load 16.6 on 72
    # cores), because interop threads only dispatch parallel op graphs.  The
    # thread COUNT was never the problem; runnable threads were always few.
    print(f"[cpu] OMP/MKL capped at {n}; torch reports intraop "
          f"{torch.get_num_threads()} / interop {torch.get_num_interop_threads()}")


class _nullctx:
    # [KR] 아무것도 하지 않는 컨텍스트 매니저(no-op). 조건부로 with 블록을
    # 켜고 끄고 싶을 때 자리표시자(placeholder)로 사용.
    def __enter__(self): return self
    def __exit__(self, *a): return False


def ddp_setup():
    """torchrun-style init.  Returns (rank, world, is_main).

    ★ WHY DDP FITS HERE.  A single rollout's activations do not split across
    cards (v4 section 5.1), so model parallelism buys nothing -- but the pool is
    already M independent walks, and giving each rank its own walks makes the
    optimizer step average over WORLD windows instead of one.  That directly
    attacks the variance the prequential curve shows: consecutive steps come
    from different scenes, so a batch of one is a batch of one scene.

    [KR] torchrun 스타일의 DDP(분산 데이터 병렬) 초기화. (rank, world, 이
    프로세스가 rank 0(메인)인지)를 반환한다.
    - 왜 모델 병렬(model parallelism)이 아니라 DDP인가: 롤아웃 하나의
      연산(activation)은 여러 카드로 나눌 수 없다. 대신 이미 풀 자체가 M개의
      독립적인 walk이므로, rank마다 자기만의 walk들을 맡기면 optimizer
      step이 (world 수만큼의) 여러 window를 평균 내게 되어, "배치 크기 1 =
      장면 1개"에서 오는 분산(variance)을 직접적으로 줄인다.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, True
    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl")
    return rank, world, rank == 0


def main():
    # [KR] ─────────────────────────────────────────────────────────────────
    # main()의 전체 흐름 (매우 길기 때문에 미리 지도를 그려둔다):
    #   1) 커맨드라인 인자 파싱 (--ckpt, --scene, v5 state mixture 관련 옵션들,
    #      L_long/L_fresh 옵션들, wandb 옵션들 등)
    #   2) DDP 초기화, CPU 스레드 제한
    #   3) --scene 스펙으로부터 Scene 객체들 생성 (이미지 로딩/memmap, LabelBank)
    #   4) 모델 빌드, optimizer 빌드, (world>1이면) DDP로 감싸기
    #   5) loss(SelfDistillLoss)와 L_long(LongPoseLoss) 구성
    #   6) 장면마다 probe용 run을 홀드아웃(hold-out)
    #   7) --pool 크기만큼 스트림 배정, v5 K/horizon deck 구성
    #   8) --resume이 있으면 체크포인트에서 가중치/옵티마이저/스트림 위치 복원
    #   9) RolloutPool 구성 (+ FreshPool, --lam_fresh > 0이고 walk 모드일 때)
    #   10) 고정 probe 상태들 구축 (theta_0 기준)
    #   11) 메인 학습 루프: 매 스텝마다 branch(identity/correction) 결정 ->
    #       fresh/long forward -> backward -> optimizer.step() -> advance ->
    #       로깅 -> (주기적으로) probe 재채점 -> (주기적으로) 체크포인트 저장
    #   12) 마지막 요약 출력(gate 6 판정) 및 결과 저장
    # ─────────────────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "experiments"))
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from phase0_density_sweep import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", default=None, help="single-scene shorthand")
    ap.add_argument("--bank", default=None, help="single-scene shorthand")
    ap.add_argument("--scene", nargs="*", default=[], metavar="NAME:FRAMES:BANK[:DATASET]",
                    help="repeatable, one per training sequence.  The optional "
                         "fourth field is the corpus this sequence belongs to, "
                         "for --dataset_weights; it defaults to the parent of "
                         "the frames path (data/mcd/... -> mcd).  Each scene keeps "
                         "its own images tensor, its own LabelBank and its own "
                         "0-based frame index -- see the Scene docstring for why "
                         "merging them is not an option.")
    ap.add_argument("--allow_frames_mismatch", action="store_true",
                    help="skip the check that --bank was generated from --frames")
    ap.add_argument("--out", default="experiments/results/train_log.json")
    ap.add_argument("--save", default=None,
                    help="write the trained weights here.  Saved as {'model': sd} "
                         "so experiments/phase0_density_sweep.build_model reads it "
                         "exactly like the released checkpoint.")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--K", type=int, default=28, help="student keyframe interval")
    ap.add_argument("--pool", type=int, default=4, help="interleaved rollout streams")
    ap.add_argument("--pool_starts", type=int, nargs="*", default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--encoder_lr_scale", type=float, default=0.0,
                    help="0 = freeze (default); 0.1 = reduced LR (§3-T5)")
    ap.add_argument("--l2sp", type=float, default=0.0,
                    help="decoupled pull toward the released weights")
    ap.add_argument("--clip", type=float, default=1.0)

    # ── LoRA (lingbot_map/train/lora.py) ─────────────────────────────────────
    # [KR] --lora_rank > 0 이면 released 가중치를 동결하고 저랭크 adapter만
    # 학습한다. 저장되는 ckpt["model"]은 adapter를 접어 넣은(merged) 일반
    # state_dict이므로 벤치/채점기는 그대로 읽는다. --resume은 ckpt["lora"]를
    # 읽는다. LoRA는 보통 full FT보다 큰 lr(1e-4 근처)이 필요하다.
    ap.add_argument("--lora_rank", type=int, default=0,
                    help="0 = full fine-tune (default); r > 0 = LoRA of rank r on --lora_targets")
    ap.add_argument("--lora_alpha", type=float, default=None,
                    help="LoRA scale numerator (delta = alpha/r * BA); default 2r")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_targets", default="global",
                    help="comma list of global,frame,camera,depth (PARAM_GROUPS prefixes)")
    ap.add_argument("--lora_modules", default="qkv,proj,fc1,fc2",
                    help="comma list of nn.Linear leaf names to wrap inside each target")

    # ── v5 state mixture (docs/self-distill-ver5.md) ─────────────────────────
    # [KR] v5 "상태 혼합" 관련 옵션들: 학생 키프레임 간격 K 분포(--k_dist),
    # 롤아웃 수평선(--horizons), identity/correction 샘플러(--sampler,
    # --p_identity), 데이터셋 균형(--dataset_weights). 여기 모든 기본값은
    # v5 이전 동작과 동일하므로, 값을 하나라도 바꾸기 전까지는 기존 커맨드가
    # 그대로 동작한다.
    # Every default here is the pre-v5 behaviour, so adding these flags to an
    # existing command line changes nothing until one is set.
    ap.add_argument("--k_dist", default="fixed",
                    help="student keyframe-interval distribution, 'K:weight' "
                         "comma-separated.  'fixed' (default) uses --K for every "
                         "rollout, i.e. the pre-v5 behaviour.  'v5' expands to "
                         "the docs/self-distill-ver5.md mixture "
                         "1:25,2:10,4:10,8:10,12:10,16:10,28:25.  K is drawn ONCE "
                         "per rollout and held until reset -- see RolloutStream.")
    ap.add_argument("--horizons", type=int, nargs="*", default=[0],
                    help="raw-frame rollout horizons to sample from.  [0] "
                         "(default) means no horizon: a stream walks to the end "
                         "of its scene, the pre-v5 behaviour.  v5 asks for "
                         "320 960 1920 3840.  A rollout ends at its horizon and "
                         "resets with a newly drawn (scene, K, horizon).")
    ap.add_argument("--fixed_horizon", type=int, default=0,
                    help="1 = deal --horizons round-robin to the streams and keep "
                         "each stream's horizon across resets, so the step-weighted "
                         "share is exact (see h_deck).  0 = pre-v6b behaviour, "
                         "redrawn every reset, which squeezes the shallow end.")
    ap.add_argument("--sampler", default="split", choices=["split", "unified"],
                    help="'split' (default) is the pre-v5 shape: the long branch "
                         "runs every step and the fresh branch is added to it, "
                         "weighted by --lam_fresh.  'unified' is v5's single "
                         "objective: each step draws ONE state type -- identity "
                         "with probability --p_identity, correction otherwise -- "
                         "so a step costs one branch instead of two.")
    ap.add_argument("--p_identity", type=float, default=0.35,
                    help="sampler=unified: share of steps drawn as identity / "
                         "preservation states (v5 asks for 30-40%%).  These are "
                         "reset, dense-K=1, teacher-matched windows; the rest are "
                         "long-history correction states at the sampled K.")
    ap.add_argument("--dataset_weights", default=None,
                    help="dataset-balanced sampling, 'name:weight' comma-"
                         "separated, e.g. 'mcd:50,slowtv:50'.  Selects a dataset "
                         "first and a sequence inside it second, so a larger "
                         "corpus cannot dominate by frame count alone.  Unset = "
                         "uniform over scenes, which is frame-balanced only if "
                         "the scenes are the same length.")
    ap.add_argument("--sampler_seed", type=int, default=0,
                    help="seed for the v5 state mixture.  The per-step branch "
                         "draw is keyed on (seed, step) with no rank term, so "
                         "every DDP rank takes the same branch; the per-rollout "
                         "draws are additionally rank-seeded so ranks differ.")

    ap.add_argument("--probe_every", type=int, default=10)
    ap.add_argument("--probe_runs_per_scene", type=int, default=2,
                    help="runs held out of TRAINING on every scene (hygiene)")
    ap.add_argument("--probe_max", type=int, default=6,
                    help="how many held-out windows are actually SCORED each "
                         "probe pass.  Hold-out is per scene and cheap; scoring "
                         "is a forward pass each, so it is capped separately.")
    ap.add_argument("--resume", default=None,
                    help="checkpoint written by --save_every.  Restores weights, "
                         "optimizer state, the step counter and every stream's "
                         "(scene, position), so a killed run picks up where it "
                         "stopped instead of from zero.")
    ap.add_argument("--save_every", type=int, default=0,
                    help="checkpoint every N steps (0 = only at the end).  A "
                         "full-corpus epoch is ~15 h; an end-only save loses "
                         "everything to one crash.")
    ap.add_argument("--probe_runs", type=int, nargs="*", default=None,
                    help="bank run indices held out for probing; streams never "
                         "enter them (default: two spread through the bank)")
    ap.add_argument("--preset", default="A1", choices=sorted(PRESETS),
                    help="loss configuration (docs/add_loss.md §5).  B0 = the "
                         "four-term loss of gate 6; B1 = gate 6b (lam_rot 30, "
                         "lam_mag 0.1); A1 = B1 with L_mag replaced by "
                         "L_motion-depth.  Individual --lam_* override it.")
    ap.add_argument("--lam_rot", type=float, default=None,
                    help="L_rot is in RADIANS, so lam_rot=1 gives it ~2%% of the "
                         "loss despite having the best FAR/NEAR discrimination "
                         "(7.5x). ~57 converts it to degrees; gate 6b used 30.")
    ap.add_argument("--lam_dir", type=float, default=None,
                    help="L_dir already takes ~half of the global_blocks gradient "
                         "at lam=1 (measured 49%% at FAR), so this is the knob "
                         "that was missing, not a formality.")
    ap.add_argument("--lam_mag", type=float, default=None,
                    help="0 in A1: L_mag fits the pose scale away, which is the "
                         "very quantity L_motion-depth exists to supervise.")
    ap.add_argument("--lam_motion", type=float, default=None,
                    help="L_motion-depth.  PROVISIONAL VALUE -- 0.5 was chosen to "
                         "occupy roughly L_mag's old share, not from a gradient "
                         "measurement.  Re-measure with term_grad_probe.py.")
    ap.add_argument("--lam_depth", type=float, default=None)
    ap.add_argument("--depth_mode", default=None, choices=DEPTH_MODES,
                    help="override the preset's depth gauge x residual.  Unset "
                         "(default) keeps whatever --preset chose.  'global' / "
                         "'global_linear' fit ONE scale for the whole window "
                         "instead of one per frame: every other mode quotients "
                         "away S scales where only one is a Sim(3) freedom, so "
                         "L_depth reads exactly 0 under per-frame depth scale "
                         "drift.  Use these to measure whether that blind spot "
                         "is hiding anything (see losses._DEPTH_MODES).")
    # ── L_long, docs/long_supervision_design.md ─────────────────────────────
    # [KR] L_long(장기 상대 포즈 손실) 관련 옵션들: 가중치(--lam_long), Delta
    # ladder(--long_deltas), 항별 가중치, 별도 long_bank 사용 여부 등.
    ap.add_argument("--lam_long", type=float, default=0.0,
                    help="weight of the long-relative pose term.  0 builds no "
                         "module at all, so the step is the pre-L_long step "
                         "bit-for-bit -- not a 0 * x that still allocates a graph.")
    ap.add_argument("--long_deltas", type=int, nargs="+", default=[48, 96, 192],
                    help="the Delta ladder, in frames.  Every entry must be >= S: "
                         "below that the anchor falls inside the supervised window "
                         "and A1PC's pairs='all' already covers it.")
    ap.add_argument("--long_lam_delta", default="",
                    help="per-Delta weights as 48:1,96:1,192:1.  Empty = flat.  "
                         "Set these from the GRADIENT share (term_grad_probe.py), "
                         "not from the loss scalar: clipping fires every step.")
    ap.add_argument("--long_terms", default="rot,dir,scale",
                    help="terms taken from the LOCAL bank.  Ablation cells: "
                         "'rot,dir' (B), 'scale' (C), all (D).")
    ap.add_argument("--long_alt_terms", default="",
                    help="terms taken from --long_bank_suffix instead.  The GT "
                         "audit says the split should be rot,dir from the in-run "
                         "bank (1.31 deg at Delta=192) and scale from the stitched "
                         "track (bias 0.07 against the in-run -0.245), i.e. "
                         "--long_terms rot,dir --long_alt_terms scale.")
    ap.add_argument("--long_lam_rot", type=float, default=15.0)
    ap.add_argument("--long_lam_dir", type=float, default=1.9)
    ap.add_argument("--long_lam_scale", type=float, default=1.0)
    ap.add_argument("--long_tau", type=float, default=0.5,
                    help="drop a pair whose teacher long displacement is below "
                         "tau * v_local: a loop or a round trip makes both the log "
                         "magnitude and the direction noise.")
    ap.add_argument("--long_huber_delta", type=float, default=0.1,
                    help="log-Huber transition, in RELATIVE error units (0.1 = 10%%).")
    ap.add_argument("--long_bank_suffix", default="",
                    help="a SECOND, pose-only bank per scene at <bank><suffix> "
                         "(e.g. '_long'), supplying the rungs the local bank "
                         "cannot reach.  Written by experiments/stitch_bank.py "
                         "(stitched L96s48) or baked directly at K_t=2.")
    ap.add_argument("--long_alt_deltas", type=int, nargs="*", default=[],
                    help="which rungs come from --long_bank_suffix.  Explicit "
                         "rather than inferred: a rung silently switching "
                         "teachers is the kind of thing that shows up as a "
                         "slightly worse curve three days later.")
    ap.add_argument("--long_vlocal", default="grad", choices=["detach", "grad"],
                    help="whether v_local carries gradient in L_scale.  GRAD is "
                         "the default: the term is supposed to constrain BOTH "
                         "sides of the long/local ratio.  Detaching makes it read "
                         "'assume this window's local scale is right and fit the "
                         "long displacement to it' -- and nothing constrains that "
                         "reference, because A1PC is invariant to a joint "
                         "(pose, depth) scaling (measured: identical to the last "
                         "digit at k = 0.25, 1, 2, 10).  The 1/sigma gradient "
                         "blow-up that motivated detaching is fixed by "
                         "--long_scale_gauge_norm instead.")
    ap.add_argument("--long_scale_gauge_norm", type=int, default=1,
                    help="multiply L_scale by a detached v_local, cancelling the "
                         "1/sigma in its gradient.  Measured spread over a 1000x "
                         "gauge sweep: 509x (grad, off) / 160x (detached) / 2.5x "
                         "(grad, on).  The loss VALUE then scales with the gauge, "
                         "so read long_r_resid_d* for a comparable scalar.")
    ap.add_argument("--long_weight_norm", type=int, default=1,
                    help="divide the long total by the sum of the lambda_delta that "
                         "actually fired.  ★ WITHOUT THIS THE OBJECTIVE CHANGES "
                         "WITH WINDOW POSITION: a rung only fires where its anchor "
                         "fits inside the run, so on the 48-grid the lambda-sum is "
                         "0/1/3/3/7 at offsets 0/48/96/144/192 -- a different "
                         "local:long mix at every offset, which is not a clean "
                         "long-supervision on/off.")
    ap.add_argument("--long_max_teacher_depth", type=int, default=0,
                    help="skip long pairs whose window sits deeper than this many "
                         "teacher keyframes (burn_in + offset).  0 = no limit.  "
                         "★ THIS IS THE CONFOUND CONTROL.  Delta and teacher depth "
                         "are entangled by construction -- Delta=192 only fits at "
                         "the LAST window of a 240-frame run -- and the teacher's "
                         "scale bias is a function of depth, not of Delta "
                         "(measured: -0.082 at depth 128, -0.263 at 272, while "
                         "across Delta at fixed depth it is flat).  Capping the "
                         "depth is how a run supervises long baselines without "
                         "also selecting the worst labels.")
    ap.add_argument("--long_rot_huber_deg", type=float, default=1.0,
                    help="degrees below which L_rot goes quadratic.  acos is an L1 "
                         "on the angle -- constant gradient magnitude however small "
                         "the residual -- so a converged pair still gets a full-size "
                         "pull, which is wrong on the identity branch where the "
                         "residual is zero by construction.  0 restores plain acos.")
    # ── L_abs, docs/gtabs-plan.md ─────────────────────────────────────────
    # [KR] run 게이지 감독 항. --abs_mode off이면 아무것도 만들지 않아 기존
    # 경로와 비트 단위로 같다(gtctrl). scale = 공통 scale 2항(gtscale),
    # paper = 논문 Eq.1의 run-게이지 판(gtpaper; dir/motion/depth-median은
    # 명시하지 않는 한 0으로 꺼진다).
    ap.add_argument("--abs_mode", default="off", choices=["off", "scale", "paper"],
                    help="run-gauge supervision (lingbot_map/train/abs_loss.py).  "
                         "'off' builds nothing.  'scale' = L_trans-scale + "
                         "L_depth-scale: one common scale per run, the channel "
                         "A1PC is blind to (gtscale).  'paper' = the paper's Eq.1 "
                         "in the run gauge: L_rot + L_rel-trans-l1 + L_abs-pos + "
                         "L_abs-rot + L_depth-scale (+ L_trans-scale), with "
                         "lam_dir/lam_motion/lam_depth zeroed unless given (gtpaper).")
    ap.add_argument("--abs_fit", default="prefix", choices=["prefix", "hist"],
                    help="how the run gauge is fitted: 'prefix' once on the run's "
                         "first 48 frames and held for the run; 'hist' refit per "
                         "window on the whole history [t0, t0+off).  Default set "
                         "by experiments/gauge_precheck.py (docs/gtabs-plan.md §5-3).")
    ap.add_argument("--lam_trans_scale", type=float, default=0.0)
    ap.add_argument("--lam_depth_scale", type=float, default=0.0)
    ap.add_argument("--lam_abs_pos", type=float, default=0.0)
    ap.add_argument("--lam_abs_rot", type=float, default=0.0)
    ap.add_argument("--lam_rel_trans", type=float, default=0.0,
                    help="the five L_abs weights.  Set from the GRADIENT share "
                         "(experiments/term_grad_probe.py --abs_mode ...), never from "
                         "the loss scalar: clipping fires every step.  A term at 0 "
                         "is skipped, not multiplied by 0.")
    ap.add_argument("--abs_min_path", type=float, default=10.0,
                    help="degeneracy guard: a fit range whose reference path is "
                         "shorter than this many median steps cannot fix a scale; "
                         "the window then uses the depth-ratio fallback and is "
                         "logged fit_mode=fallback.")
    ap.add_argument("--abs_max_offset", type=int, default=0,
                    help="skip L_abs on correction windows deeper than this run "
                         "offset (0 = never).  For the theta_0-bank variant, where "
                         "teacher drift contaminates deep targets.  ★ Under DDP a "
                         "skipped window changes the graph on that rank only; keep "
                         "at 0 with world > 1 unless every rank skips together.")
    ap.add_argument("--abs_tau", type=float, default=0.5,
                    help="drop pairs whose reference displacement is below tau * u")
    ap.add_argument("--abs_huber_delta", type=float, default=0.1,
                    help="log-Huber transition of L_trans-scale, relative units")
    ap.add_argument("--abs_rot_huber_deg", type=float, default=1.0)
    ap.add_argument("--abs_rot_exclude", default="",
                    help="json {scene: [run ids]} from gauge_precheck.py whose "
                         "orientation spread disqualifies L_abs-rot on those runs")
    ap.add_argument("--hist_span", type=int, default=0,
                    help="frames of student pose history kept per stream.  0 = "
                         "auto: max(long ladder + 2S, longest bank run + S) when "
                         "L_long or L_abs is on, else nothing is recorded.  The "
                         "prefix fit needs the run's first 48 frames to survive "
                         "until its last window: run 240 + S 48 = 288.")
    ap.add_argument("--lam_fresh", type=float, default=0.0,
                    help="weight of the fresh-preservation branch, docs/add_loss.md "
                         "step 3.  L_total = L_long + lam_fresh * L_fresh.  0 "
                         "disables the branch entirely (and costs nothing).  With "
                         "--l2sp 0 this is the ONLY thing anchoring the model to "
                         "its starting point.")
    ap.add_argument("--fresh_mode", default="window", choices=["window", "walk"],
                    help="how the fresh branch places its supervision.  'window' "
                         "is the shipped behaviour: replay the teacher prefix "
                         "every step and score offset 0, i.e. cache depth "
                         "sf+B -> sf+B+S only (80 -> 128 here).  'walk' keeps "
                         "the dense state and advances S frames per step, so a "
                         "run sweeps sf+B -> 320 -- the whole deployment budget. "
                         "See FreshPool for the measurement that motivates it. "
                         "Default stays 'window' so a2 reproduces bit-for-bit.")
    ap.add_argument("--fresh_pool", type=int, default=1,
                    help="fresh_mode=walk: how many dense walks to interleave.  "
                         "1 costs one extra host snapshot (~1.7 GB) and makes "
                         "consecutive steps sweep the depth range in order; "
                         "raise it to decorrelate steps at that cost each.")
    ap.add_argument("--wandb", type=int, default=1,
                    help="1 = log to Weights & Biases (default), 0 = off.  The "
                         "run never fails because of wandb; every hook degrades "
                         "to a no-op.")
    ap.add_argument("--wandb_project", default="streaming3d-self-distill")
    ap.add_argument("--wandb_name", default=None, help="run name (default: preset+time)")
    ap.add_argument("--wandb_group", default=None, help="groups the A1/A2/A3 ablation cells")
    ap.add_argument("--wandb_tags", nargs="*", default=[])
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--gt_calib", default=None,
                    help="calib yaml -> turns on the GT probe.  METRIC ONLY: GT "
                         "never enters the loss (the method is unlabelled by "
                         "construction, v4 §3.3).  Poses come from <frames>/meta.npz.")
    ap.add_argument("--gt_sensor", default="d455b_color")
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=20000)
    args = ap.parse_args()

    rank, world, is_main = ddp_setup()
    # ★ AFTER ddp_setup(), never before.  torch.set_num_threads() ahead of
    # torch.cuda.set_device() / init_process_group SIGSEGVs both ranks under
    # torchrun -- isolated to exactly this ordering (the same script with the
    # call moved after cuda init runs clean).  Threads are a CPU concern and the
    # CUDA context has to exist first.
    _cap_threads()
    dev, dtype, sf, S = torch.device("cuda"), torch.bfloat16, args.num_scale_frames, args.S

    def _dataset_of(frames_dir: str) -> str:
        """``data/mcd/kth_day_10/frames_10hz`` -> ``mcd``.

        Two levels up from the frames directory is the sequence, three is the
        corpus.  Only a default -- the --scene spec's fourth field wins -- but a
        correct default matters because getting it wrong makes --dataset_weights
        silently balance groups that are not the ones it names.
        """
        p = os.path.abspath(frames_dir)
        return os.path.basename(os.path.dirname(os.path.dirname(p))) or "unknown"

    specs = []
    for x in args.scene:
        f = x.split(":", 3)
        if len(f) < 3:
            raise SystemExit(f"--scene must be NAME:FRAMES:BANK[:DATASET], got {x!r}")
        specs.append((f[0], f[1], f[2], f[3] if len(f) > 3 else _dataset_of(f[1])))
    if args.frames and args.bank:
        specs.append((os.path.basename(os.path.dirname(os.path.abspath(args.frames))),
                      args.frames, args.bank, _dataset_of(args.frames)))
    if not specs:
        raise SystemExit("give --scene NAME:FRAMES:BANK (repeatable) or --frames+--bank")
    if world > 1:
        # [KR] world>1(DDP)이면 장면을 rank들에 "분할(partition)"한다(복제가
        # 아님) -- rank마다 다른 시퀀스를 걷게 해서, 스텝마다 서로 다른
        # 장면의 window들이 평균되도록(=배치 다양성 확보) 하기 위함.
        # Scenes are PARTITIONED, not replicated: each rank walks its own
        # sequences, so the two windows averaged into one step come from
        # different scenes -- which is the point.  It also halves the per-rank
        # memmap set.
        #
        # ★ v5: the stride partition is DATASET-INTERLEAVED because the caller is
        # asked to interleave the --scene list, but nothing enforces it.  With
        # --dataset_weights set, a rank that ends up holding only one corpus
        # cannot honour the weights at all, so say so rather than sampling a
        # distribution the run does not have.
        all_specs = specs
        specs = specs[rank::world]
        if not specs:
            raise SystemExit(f"rank {rank}: no scenes (have {len(all_specs)}, world {world})")
        if args.dataset_weights:
            _have = {x[3] for x in specs}
            _all = {x[3] for x in all_specs}
            if _have != _all:
                print(f"[ddp] WARNING rank {rank} holds datasets {sorted(_have)} of "
                      f"{sorted(_all)} -- --dataset_weights cannot be honoured on "
                      f"this rank.  Interleave --scene so every rank gets every "
                      f"corpus (rank r takes scenes r, r+{world}, ...).")
        print(f"[ddp] rank {rank}/{world} takes {[x[0] for x in specs]}", flush=True)

    # ★ BUILD THE MODEL BEFORE OPENING ANY MEMMAP.  With the scene mappings
    # already in place, CUDA/checkpoint initialisation inside build_model
    # SIGSEGVs -- reproduced on both cards, single-GPU and under torchrun, always
    # at the same point (all scenes load, then the crash), and it disappears when
    # the order is reversed.  The interaction is between the ~143 GB of file
    # mappings and whatever address space CUDA reserves at context creation; the
    # cheap fix is to let CUDA go first.
    model = build_model(args.ckpt, dev, args.image_size, args.patch_size,
                        args.max_frame_num, args.kv_cache_sliding_window, sf)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    # LoRA goes in here, after the released weights are on the card and before
    # build_optimizer / L2SP / DDP look at requires_grad.  inject_lora freezes
    # the base, so from this point "trainable" means the adapter.
    if args.lora_rank > 0:
        _alpha = args.lora_alpha if args.lora_alpha is not None else 2.0 * args.lora_rank
        _wrapped = LORA.inject_lora(
            model, args.lora_rank, _alpha, args.lora_dropout,
            targets=args.lora_targets.split(","), modules=args.lora_modules.split(","))
        print(LORA.lora_summary(model), flush=True)
        print(f"[lora] first/last wrapped: {_wrapped[0]} .. {_wrapped[-1]}", flush=True)

    scenes: List[Scene] = []
    # [KR] --scene 스펙마다 Scene 객체를 하나씩 만든다: 뱅크 로드, (memmap
    # 캐시가 있으면 그걸, 없으면 이미지를 직접 로드), 필요시 long_bank 로드.
    for name, frames_dir, bank_dir, dataset in specs:
        bk = LabelBank(bank_dir)
        # Nothing else checks that these two describe the same sequence, and with
        # a corpus this size a silent mismatch is a matter of time: the student
        # would read one scene's pixels and score them against another's labels,
        # producing a loss that is large, stable, and completely meaningless.
        recorded = bk.index.get("frames") if hasattr(bk, "index") else None
        if recorded and not args.allow_frames_mismatch:
            if os.path.abspath(recorded) != os.path.abspath(frames_dir):
                raise SystemExit(
                    f"scene {name}: bank {bank_dir} was generated from\n"
                    f"  {recorded}\nbut --scene points at\n  {os.path.abspath(frames_dir)}\n"
                    f"pass --allow_frames_mismatch only if you are certain.")
        cov = (bk.runs[0]["t0"], bk.runs[-1]["t0"] + bk.runs[-1]["L"])
        # Prefer a memmapped cache (experiments/cache_frames.py).  A step touches
        # ~96 frames of one scene; holding all 10 scenes resident costs 216 GB of
        # RSS per process, which is what makes two concurrent runs unschedulable.
        # The cache holds the SAME float32 values, so this changes nothing
        # numerically -- it only makes the pages reclaimable and shareable.
        _cache = os.path.join(frames_dir,
                              f"_cache_{args.image_size}_{args.patch_size}.npy")
        if os.path.exists(_cache):
            imgs = MemmapFrames(_cache, cov[1])
            _src = "memmap"
        else:
            names = image_names(frames_dir)[:cov[1]]
            imgs = load_and_preprocess_images(
                [os.path.join(frames_dir, n) for n in names], mode="crop",
                image_size=args.image_size, patch_size=args.patch_size).unsqueeze(0)
            _src = "resident"
        lbk = None
        if args.long_bank_suffix:
            _lp = bank_dir + args.long_bank_suffix
            if os.path.exists(os.path.join(_lp, "index.json")):
                lbk = LabelBank(_lp, cache_runs=2)
        scenes.append(Scene(name=name, frames=frames_dir, images=imgs, bank=bk,
                            covered=cov, dataset=dataset, long_bank=lbk))
        print(f"[scene] {name:<16} {dataset:<8} {len(bk):>3} runs  frames {cov}  "
              f"{bk.frames_covered:>6} supervised  "
              f"{imgs.numel() * 4 / 1e9:.1f} GB {_src}")
    _by_ds = {}
    for s_ in scenes:
        _by_ds.setdefault(s_.dataset, []).append(s_)
    print(f"[corpus] {len(scenes)} scenes, "
          f"{sum(s_.bank.frames_covered for s_ in scenes)} supervised frames, "
          f"{sum(s_.images.numel() * 4 for s_ in scenes) / 1e9:.0f} GB resident")
    for _d, _ss in sorted(_by_ds.items()):
        print(f"[corpus]   {_d:<8} {len(_ss):>2} scenes, "
              f"{sum(x.bank.frames_covered for x in _ss):>7} supervised frames")

    opt = build_optimizer(model, args.lr, args.wd, args.encoder_lr_scale)
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # [KR] 모델을 DDP로 감싼다. 롤아웃(no_grad)은 DDP를 거치지 않고 직접
        # model로 실행하고, 지도학습 forward 두 개(fresh, correction)만 DDP
        # 래퍼(fwd)를 거쳐서 gradient가 스텝당 정확히 한 번씩 all-reduce
        # 되도록 한다. find_unused_parameters=True는 방어적인 게 아니라
        # 필수다 -- 이 학습 경로에서 절대 안 쓰이는 파라미터(예:
        # camera_head.empty_pose_tokens)가 있어서 안 켜면 DDP가 "gradient를
        # 못 받았다"고 에러를 낸다. 이게 안전한 이유는 스텝당 backward가
        # "정확히 한 번"이기 때문(위의 브랜치별 backward 순서 규칙 참고).
        # The rollout is no_grad and must NOT go through DDP; only the two
        # supervised forwards do, so gradients are all-reduced exactly once per
        # branch.  gradient_as_bucket_view keeps the extra copy off the card.
        # find_unused_parameters is REQUIRED, not defensive: index 867 is
        # camera_head.empty_pose_tokens, which this training path never touches
        # (it is the placeholder for absent pose slots, and every supervised
        # window is full), so it receives no gradient in either branch and DDP
        # otherwise raises "did not receive grad".
        #
        # It is only safe because there is exactly ONE backward per step.  Under
        # the earlier two-backward + no_sync design it would have been actively
        # wrong: a parameter whose gradient arrived during the no_sync branch
        # would be marked unused on the syncing branch, skip the all-reduce, and
        # leave every rank with its own value -- silent divergence, no error.
        fwd = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", rank))],
                  gradient_as_bucket_view=True, broadcast_buffers=False,
                  find_unused_parameters=True)
    else:
        fwd = model
    l2sp = L2SP(model, args.l2sp)
    lam_over = {f"lam_{k}": getattr(args, f"lam_{k}")
                for k in ("rot", "dir", "mag", "motion", "depth")
                if getattr(args, f"lam_{k}") is not None}
    # [KR] --preset(A1/B0/B1 등)으로 기본 손실 가중치를 정하고, 개별
    # --lam_* / --depth_mode 옵션이 있으면 그것으로 덮어쓴다.
    if args.depth_mode:
        lam_over["depth_mode"] = args.depth_mode
    if args.abs_mode == "paper":
        # the paper's Eq.1 has no direction, motion-depth or median-depth term;
        # an explicit --lam_* still wins so the ablation stays one flag away
        for k in ("dir", "motion", "depth"):
            lam_over.setdefault(f"lam_{k}", 0.0)
    crit = SelfDistillLoss.from_preset(args.preset, **lam_over)
    abs_crit, abs_units, abs_rot_excl = None, None, None
    if args.abs_mode != "off":
        _lam_abs = dict(lam_trans_scale=args.lam_trans_scale, lam_depth_scale=args.lam_depth_scale,
                        lam_abs_pos=args.lam_abs_pos, lam_abs_rot=args.lam_abs_rot,
                        lam_rel_trans=args.lam_rel_trans)
        if args.abs_mode == "scale":
            _extra = {k: v for k, v in _lam_abs.items()
                      if k in ("lam_abs_pos", "lam_abs_rot", "lam_rel_trans") and v != 0.0}
            if _extra:
                raise SystemExit(f"--abs_mode scale supervises the common scale only; "
                                 f"{sorted(_extra)} belong to --abs_mode paper")
        if not any(v != 0.0 for v in _lam_abs.values()):
            raise SystemExit(f"--abs_mode {args.abs_mode} with every lam_* at 0 -- nothing to train")
        abs_crit = AL.RunGaugeLoss(**_lam_abs, huber_delta=args.abs_huber_delta,
                                   rot_huber_deg=args.abs_rot_huber_deg, tau=args.abs_tau)
        abs_units = AL.RunUnitCache()
        if args.abs_rot_exclude:
            with open(args.abs_rot_exclude) as _f:
                _ex = json.load(_f)
            _ex = _ex.get("abs_rot_exclude", _ex)
            abs_rot_excl = {k: set(int(x) for x in v) for k, v in _ex.items()}
        print(f"[abs] mode={args.abs_mode} fit={args.abs_fit} "
              f"lam={ {k[4:]: v for k, v in _lam_abs.items() if v != 0.0} } "
              f"tau={args.abs_tau} huber={args.abs_huber_delta} rot_huber={args.abs_rot_huber_deg}deg "
              f"min_path={args.abs_min_path} max_offset={args.abs_max_offset or 'none'}"
              + (f" rot_exclude={ {k: sorted(v) for k, v in abs_rot_excl.items()} }" if abs_rot_excl else ""))
    long_crit = long_crit_alt = None
    # [KR] --lam_long > 0이면 L_long(장기 상대 포즈) 모듈을 구성한다. 0이면
    # long_crit이 None으로 남아서, L_long 관련 코드가 전혀 실행되지 않는다
    # (이전 성능 특성을 그대로 유지하기 위함).
    if args.lam_long > 0:
        _ladder = tuple(sorted({int(d) for d in args.long_deltas}))
        _bad = [d for d in _ladder if d < S]
        if _bad:
            raise SystemExit(
                f"--long_deltas {_bad} are below --S {S}.  A Delta smaller than the "
                f"window puts the anchor INSIDE the window: it would pick up "
                f"gradient (breaking the detached-past premise) and duplicate "
                f"A1PC's pairs='all', which already supervises gaps 1..{S - 1}.")
        _lamd = {}
        for _tok in filter(None, args.long_lam_delta.split(",")):
            _k, _v = _tok.split(":")
            _lamd[int(_k)] = float(_v)
        if set(_lamd) - set(_ladder):
            raise SystemExit(f"--long_lam_delta names {sorted(set(_lamd) - set(_ladder))} "
                             f"which are not on the ladder {list(_ladder)}")
        _terms = tuple(t.strip() for t in args.long_terms.split(",") if t.strip())
        _aterms = tuple(t.strip() for t in args.long_alt_terms.split(",") if t.strip())
        if set(_terms) & set(_aterms):
            raise SystemExit(
                f"--long_terms and --long_alt_terms overlap on "
                f"{sorted(set(_terms) & set(_aterms))}: a term must have ONE "
                f"teacher source, or the same relation is supervised twice against "
                f"two different teachers.")
        if _aterms and not args.long_bank_suffix:
            raise SystemExit("--long_alt_terms needs --long_bank_suffix")
        long_crit = LL.LongPoseLoss(
            ladder=_ladder, lam_delta=_lamd,
            lam_rot=args.long_lam_rot, lam_dir=args.long_lam_dir,
            lam_scale=args.long_lam_scale, tau=args.long_tau,
            huber_delta=args.long_huber_delta,
            detach_vlocal=(args.long_vlocal == "detach"),
            scale_gauge_norm=bool(args.long_scale_gauge_norm),
            weight_norm=bool(args.long_weight_norm),
            rot_huber_deg=args.long_rot_huber_deg, terms=_terms)
        if _aterms:
            _al = tuple(sorted({int(d) for d in (args.long_alt_deltas or _ladder)}))
            if [d for d in _al if d < S]:
                raise SystemExit(f"--long_alt_deltas below --S {S}: {_al}")
            long_crit_alt = LL.LongPoseLoss(
                ladder=_al, lam_delta=_lamd,
                lam_rot=args.long_lam_rot, lam_dir=args.long_lam_dir,
                lam_scale=args.long_lam_scale, tau=args.long_tau,
                huber_delta=args.long_huber_delta,
                detach_vlocal=(args.long_vlocal == "detach"),
                scale_gauge_norm=bool(args.long_scale_gauge_norm),
                weight_norm=bool(args.long_weight_norm),
                rot_huber_deg=args.long_rot_huber_deg,
                terms=_aterms, prefix="longalt")
        # ★ A BANK WHOSE RUNS ARE THE WINDOW CANNOT SUPPLY A SINGLE LONG PAIR.
        # With L=48 (the _L48 re-bake) a run IS one window, so off is always 0
        # and every Delta indexes before the start of the run.  L_long would then
        # be masked out on every step of the run and the cell would look like a
        # null result instead of a misconfiguration.  Refuse at startup.
        _need = S + min(_ladder)
        _dead = [sc_.name for sc_ in scenes
                 if all(r["L"] < _need for r in sc_.bank.runs)]
        if _dead:
            raise SystemExit(
                f"--lam_long {args.lam_long} needs bank runs of at least "
                f"S + min(delta) = {_need} frames, but {len(_dead)} scene(s) have "
                f"none: {_dead[:5]}{' ...' if len(_dead) > 5 else ''}.  The _L48 "
                f"bake cannot serve L_long -- point --scene at the L=240 banks.")
        _cov = {d: sum(1 for sc_ in scenes for r in sc_.bank.runs
                       if r["L"] >= S + d) for d in _ladder}
        print(f"[long] vlocal={args.long_vlocal} gauge_norm={args.long_scale_gauge_norm} "
              f"weight_norm={args.long_weight_norm} "
              f"max_teacher_depth={args.long_max_teacher_depth or 'none'}")
        print(f"[long] lam={args.lam_long} ladder={list(_ladder)} "
              f"terms={long_crit.terms}"
              + (f" | ALT terms={long_crit_alt.terms} ladder={list(long_crit_alt.ladder)}"
                 if long_crit_alt is not None else "")
              + f" tau={args.long_tau} "
              f"lam(rot,dir,scale)=({args.long_lam_rot},{args.long_lam_dir},"
              f"{args.long_lam_scale}) lam_delta={_lamd or 'flat'}")
        print(f"[long] runs able to serve each delta: {_cov} "
              f"of {sum(len(sc_.bank.runs) for sc_ in scenes)} runs")
        # ★ COVERAGE OF THE ALTERNATE (stitched) SOURCE, PER CORPUS.  A scene
        # with no stitched track contributes NO long pairs for the rungs routed
        # there, so the objective differs by corpus -- dl3dv and replica have no
        # track at all (their sequences are below the 447 frames a Delta=319 pair
        # needs).  Silent, and worth a line in the log rather than a surprise in
        # the ablation.
        if args.long_bank_suffix:
            _by = {}
            for sc_ in scenes:
                d = _by.setdefault(sc_.dataset, [0, 0])
                d[1] += 1
                d[0] += sc_.long_bank is not None
            print(f"[long] alt bank '{args.long_bank_suffix}' for delta="
                  f"{sorted(set(int(x) for x in args.long_alt_deltas))}: "
                  + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(_by.items())))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train / 1e6:.0f} M "
          f"(encoder {'frozen' if args.encoder_lr_scale == 0 else f'{args.encoder_lr_scale}x LR'})"
          f"   L2-SP ref {l2sp.bytes() / 1e9:.1f} GB")
    print(f"[loss] preset={args.preset}  lam(rot,dir,mag,depth,motion)={crit.lam}"
          f"  mag={crit.mag_mode}  depth={crit.depth_mode}")

    if args.wandb and is_main:
        cfg = {**vars(args),
               "lam_resolved": dict(zip(("rot", "dir", "mag", "depth", "motion"), crit.lam)),
               "mag_mode": crit.mag_mode, "depth_mode": crit.depth_mode,
               "trainable_M": n_train / 1e6}
        wandb_log.init(cfg,
                       name=args.wandb_name or f"{args.preset}-{time.strftime('%m%d-%H%M')}",
                       project=args.wandb_project, group=args.wandb_group,
                       tags=args.wandb_tags, mode=args.wandb_mode)

    # ── hold-out, per scene ──────────────────────────────────────────────────
    # [KR] 장면마다 probe(평가)용으로 run 전체를 홀드아웃한다(학습에서 절대
    # 쓰지 않음). --gt_calib가 있으면 GT 채점 함수도 여기서 만든다.
    # Whole RUNS are reserved, not windows: training windows advance by S but a
    # stream can still partially overlap a neighbour, so only a whole-run
    # reservation guarantees the probe frames are never trained on.  At corpus
    # scale this is per scene, so every sequence contributes held-out ground.
    for sc in scenes:
        n = len(sc.bank.runs)
        if args.probe_runs is not None and len(scenes) == 1:
            hold = [r for r in args.probe_runs if 0 <= r < n]
        else:
            k = max(0, min(args.probe_runs_per_scene, n - 1))
            hold = sorted({(i + 1) * n // (k + 1) for i in range(k)})
        sc.skip_runs = set(hold)
        # ★ ONLY WHERE GT ACTUALLY EXISTS.  make_gt_score reads <frames>/meta.npz,
        # which only the MCD corpus has; attaching it to every scene would raise
        # at startup on the first dl3dv/scannet entry.  A mixed corpus therefore
        # gets the GT probe on its MCD windows and nothing on the rest -- which is
        # the point: it is a METRIC, and one grounded window beats none.
        if sc.gt is None and args.gt_calib and \
                os.path.exists(os.path.join(sc.frames, "meta.npz")):
            sc.gt = make_gt_score(sc.frames, args.gt_calib, args.gt_sensor)

    # ── streams: one per scene by default ────────────────────────────────────
    # [KR] --pool 개수만큼 스트림을 만들어 장면들에 라운드로빈으로 배정하고,
    # 각 스트림의 시작 위치(scene, t0)를 정한다. --pool은 "전체(all-rank)"
    # 스트림 수이므로 world로 나눠서 이 rank가 맡을 몫만 계산한다.
    # With fewer streams than scenes the corpus silently shrinks to whatever the
    # streams happen to sit on, which is exactly the failure that makes a big
    # corpus look like a small one.
    # --pool is the TOTAL stream count across ranks; each rank owns its share.
    # Without the division every rank builds the full pool, so world=2 silently
    # doubles both the walks and the 6 GB-per-stream host snapshots.
    if args.pool and world > 1 and args.pool % world:
        raise SystemExit(
            f"--pool {args.pool} is not divisible by world {world}; the K deck is "
            f"split as deck[rank::world] and an uneven split would give the ranks "
            f"different stream counts, so the realised mixture would not be the "
            f"declared one.  Use a multiple of {world}.")
    n_pool = (args.pool // world) if args.pool else len(scenes)
    n_pool = max(1, n_pool)
    if n_pool < len(scenes):
        print(f"[pool] WARNING {n_pool} streams for {len(scenes)} scenes -- "
              f"{len(scenes) - n_pool} scenes will never be visited")
    counts: Dict[int, int] = {}
    for i in range(n_pool):
        counts[i % len(scenes)] = counts.get(i % len(scenes), 0) + 1
    starts = []
    for si, cnt in sorted(counts.items()):
        sc = scenes[si]
        span = max(1, sc.covered[1] - sc.covered[0] - S)
        for j in range(cnt):
            starts.append((si, sc.covered[0] + j * span // cnt))

    # ── v5 state mixture ─────────────────────────────────────────────────────
    # [KR] --k_dist, --horizons, --dataset_weights 문자열을 실제 구조로 파싱
    # 하고, 이 중 하나라도 켜져 있으면(mixture_on) StateSampler를 만든다.
    k_dist = parse_k_dist(
        ",".join(f"{k}:{w}" for k, w in K_DIST_V5) if args.k_dist == "v5"
        else args.k_dist)
    horizons = [h for h in (args.horizons or [0]) if h >= 0]
    ds_w = None
    if args.dataset_weights:
        ds_w = {}
        for tok in args.dataset_weights.split(","):
            tok = tok.strip()
            if not tok:
                continue
            nm, _, w = tok.partition(":")
            try:
                ds_w[nm] = float(w) if w else 1.0
            except ValueError:
                raise SystemExit(f"--dataset_weights: cannot parse {tok!r}")
        unknown = set(ds_w) - {sc.dataset for sc in scenes}
        if unknown:
            raise SystemExit(
                f"--dataset_weights names {sorted(unknown)} but the scenes carry "
                f"{sorted({sc.dataset for sc in scenes})}.  Fix the fourth field "
                f"of --scene, or the weights silently balance nothing.")
    mixture_on = bool(k_dist) or any(horizons) or ds_w or args.sampler == "unified"
    sampler = StateSampler(k_dist, horizons, args.p_identity, ds_w,
                           seed=args.sampler_seed, rank=rank) if mixture_on else None

    # ── the K deck ───────────────────────────────────────────────────────────
    # [KR] k_deck()/h_deck()로 전체 풀의 K/horizon 배정 목록을 만들고, 이
    # rank가 맡을 슬라이스(deck[rank::world])를 뽑는다. 스트림 수와 deck
    # 길이가 안 맞으면 반복해서 채운다.
    # One interval per stream for the WHOLE pool, then this rank's slice.  The
    # deck is sorted, so deck[rank::world] hands each rank an even share of every
    # K -- the global multiset is exact either way, but sorting keeps the deep
    # K=1 snapshots (the largest) from all landing on one card.  See ``k_deck``.
    _pool_total = args.pool if args.pool else len(scenes) * world
    _deck = k_deck(k_dist, _pool_total, args.K)
    stream_k = _deck[rank::world] if world > 1 else _deck
    stream_h = None
    if args.fixed_horizon and any(horizons):
        _hd = h_deck(horizons, _pool_total)
        stream_h = _hd[rank::world] if world > 1 else _hd
        import collections as _c
        _hc = _c.Counter(_hd)
        print(f"[v6b] horizon deck over {_pool_total} streams: "
              + "  ".join(f"h{h}x{n} ({n / _pool_total:.0%})" for h, n in sorted(_hc.items()))
              + f"   -- rank {rank} holds {sorted(stream_h)}")
        print("[v6b]   horizon is fixed per STREAM, so its step share is exact too.")
    if len(stream_k) != n_pool:                 # e.g. --pool unset, scenes uneven
        stream_k = (stream_k * ((n_pool // max(len(stream_k), 1)) + 1))[:n_pool]
    if k_dist:
        _glob: Dict[int, int] = {}
        for _k in _deck:
            _glob[_k] = _glob.get(_k, 0) + 1
        print(f"[v5] K deck over {_pool_total} streams: "
              + "  ".join(f"K{k}x{v} ({v / _pool_total:.0%})" for k, v in sorted(_glob.items()))
              + f"   -- rank {rank} holds {sorted(stream_k)}")
        print("[v5]   K is fixed per STREAM, so the step-weighted mixture is exact "
              "and independent of --steps, --horizons and reset timing.")

    # ★ AN UNREACHABLE HORIZON IS A SILENT NO-OP, so say so at startup.  A stream
    # gets steps*(1-p_identity)/n_pool turns and advances S frames per turn, so
    # any horizon past that is never the thing that ends a rollout -- the run
    # simply stops mid-walk.  v5c drew 1920 and 3840 for 10 of its 19 rollouts
    # and reached a maximum age of 1008.
    if sampler is not None and any(horizons):
        _reach = args.steps * (1.0 - (args.p_identity if args.sampler == "unified" else 0.0)) \
                 / max(n_pool, 1) * S
        _far = [h for h in horizons if h and h > _reach]
        if _far:
            print(f"[v5] WARNING horizons {_far} exceed the reachable age {_reach:.0f} "
                  f"(steps={args.steps}, pool/rank={n_pool}, p_identity={args.p_identity}) "
                  f"-- those rollouts never hit their horizon.  Need steps >= "
                  f"{max(_far) / S * n_pool / max(1e-9, 1.0 - args.p_identity):.0f}, "
                  f"or lower --pool / --p_identity / max(--horizons).")
    if sampler is not None:
        print(f"[v5] state mixture ON -- sampler={args.sampler}"
              + (f" p_identity={args.p_identity}" if args.sampler == "unified" else "")
              + f"  K={'x'.join(f'{k}@{p:.0%}' for k, p in k_dist) if k_dist else f'fixed {args.K}'}"
              + f"  horizons={horizons if any(horizons) else 'scene end'}"
              + (f"  datasets={ds_w}" if ds_w else ""))
        if args.sampler == "unified" and args.lam_fresh <= 0:
            raise SystemExit(
                "--sampler unified needs --lam_fresh > 0: the identity branch IS "
                "the fresh branch, and lam_fresh is its weight in the single "
                "objective.  v5's unified loss weights the two state types "
                "equally, so pass --lam_fresh 1.0.")
        if args.sampler == "unified" and args.fresh_mode != "walk":
            print("[v5] NOTE --sampler unified with --fresh_mode window replays "
                  "the teacher prefix every identity step and always scores "
                  "offset 0, so burn_in_eff is pinned at 72.  'walk' sweeps the "
                  "whole 320-keyframe budget for the same cost -- see FreshPool.")

    step0 = 0
    fresh_cursor0 = 0
    # ★ KEEP THETA_0 SO THE PROBE STAYS THE SAME PROBLEM ACROSS A RESUME.
    # pool.roll_to() rolls a probe's frozen state with whatever weights the model
    # currently holds.  Fresh, that is theta_0; after a resume it is the resumed
    # weights, so the probe silently rebases -- its own step-0 reading becomes a
    # new baseline and the series cannot be compared to the one before the crash.
    # Gate 6 is read off that series, so the states are built at theta_0 either
    # way and only the SCORING weights change.
    theta0_sd = None
    # [KR] --resume이 주어지면: (1) 재개 직전의 현재 가중치를 theta0_sd로
    # 잠깐 보관(probe가 여전히 theta_0 기준으로 만들어지도록), (2) 체크포인트
    # 로드해서 모델/옵티마이저/스텝 카운터/스트림 위치를 전부 복원한다.
    if args.resume:
        theta0_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Stream positions are stored as scene INDICES, which only mean anything
        # against the scene list they were written with.  Map through NAMES so a
        # run can resume across a different partition -- notably single-GPU (all
        # 10 scenes) -> DDP (5 per rank), where the indices would otherwise put
        # every walk in the wrong sequence.
        _names = _ck.get("scenes") or []
        _local = {sc_.name: i for i, sc_ in enumerate(scenes)}
        _missing = [n for n in _names if n not in _local] if world == 1 else []
        if _missing:
            raise SystemExit(
                f"--resume checkpoint covers scenes {_names}; this run is missing "
                f"{_missing}.  Add them with --scene or start fresh.")
        if args.lora_rank > 0:
            # ckpt["model"] is the MERGED weights (see lora.py); loading it over
            # the frozen base would fold the adapter in twice once ckpt["lora"]
            # is restored on top.  The base stays theta_0 from --ckpt and only
            # the adapter comes from the checkpoint.
            if "lora" not in _ck:
                raise SystemExit(f"--resume {args.resume} has no ['lora'] entry but "
                                 f"--lora_rank {args.lora_rank} was given")
            LORA.load_lora_state(model, _ck["lora"])
        else:
            model.load_state_dict(_ck["model"], strict=True)
        model.to(dev)
        if "opt" in _ck:
            opt.load_state_dict(_ck["opt"])
        step0 = int(_ck.get("step", 0))
        # Without this the walk restarts on the FIRST run of the first
        # scene after every resume, so a run that saves every 50 steps
        # re-covers the same ~10 runs forever and the depth sweep never
        # moves on.  Same failure the "streams" field exists to prevent.
        fresh_cursor0 = int(_ck.get("fresh_cursor", 0))
        if _ck.get("streams") and _names:
            mine = []
            for x in sorted(_ck["streams"],
                            key=lambda y: (y.get("scene_name") or "", y["t"])):
                nm = x.get("scene_name") or (
                    _names[x["scene"]] if x["scene"] < len(_names) else None)
                if nm in _local:                      # this rank owns that scene
                    # ★ CARRY K AND THE HORIZON, not just the position.  A
                    # rollout is (scene, K, horizon, age); restoring only the
                    # position would give the resumed walk a fresh K over a
                    # prefix that was rolled at the old one, and the masked
                    # path's _check_prefix_consistency would raise on the very
                    # first step -- loudly, but only after the pool has been
                    # rebuilt, which is minutes of rolling wasted.
                    mine.append({"sid": x.get("sid", len(mine)),
                                 "scene": _local[nm], "t": x["t"],
                                 "K": x.get("K", args.K),
                                 "horizon": x.get("horizon", 0),
                                 "anchor_t": x.get("anchor_t", x["t"]),
                                 "windows_done": x.get("windows_done", 0),
                                 "resets": x.get("resets", 0)})
            if mine:
                starts = mine
            else:
                print(f"[resume] rank {rank}: none of the saved streams belong to "
                      f"this rank's scenes -- keeping the fresh placement")
        _desc = ([(scenes[a["scene"]].name, a["t"], f"K{a['K']}") for a in starts]
                 if starts and isinstance(starts[0], dict)
                 else [(scenes[a].name, b) for a, b in starts])
        print(f"[resume] {args.resume} -- step {step0}, streams at {_desc}")
        del _ck

    t_setup = time.time()
    pool = RolloutPool(model, scenes, starts, S, sf, args.K, dtype, dev,
                       sampler=sampler, stream_k=stream_k, stream_h=stream_h)
    if long_crit is not None:
        # One window of margin past the longest rung: nothing older can ever be
        # indexed, and an untrimmed dict would grow with the rollout horizon.
        #
        # ★ THE ALT LADDER COUNTS TOO, AND v7e IS WHAT HAPPENS WHEN IT DOES NOT.
        # Sizing this from long_crit alone gave 192 + 2S = 288 while the alt bank
        # asked for Delta=319, so every Delta=319 anchor was deleted here one
        # step before build_pairs looked for it.  build_pairs skips a delta whose
        # anchors are missing -- no error, no warning -- so the rung logged
        # n_d319 = 0 on 952 of 952 windows that had the bank, except the 30 that
        # sat at raw_frame_age EXACTLY 288 and caught 17 partial pairs on the
        # prune boundary.  The run looked healthy and scored like the control.
        _ladders = list(long_crit.ladder)
        if long_crit_alt is not None:
            _ladders += list(long_crit_alt.ladder)
        pool.hist_span = max(_ladders) + 2 * S
    if abs_crit is not None:
        # the prefix fit reads the run's first 48 frames from hist at the run's
        # LAST window, so the history must reach back a whole run plus a window
        _need = max(r["L"] for sc_ in scenes for r in sc_.bank.runs) + S
        pool.hist_span = max(pool.hist_span, _need)
    if args.hist_span > 0:
        pool.hist_span = max(pool.hist_span, args.hist_span)
    if abs_crit is not None or long_crit is not None:
        print(f"[hist] keeping {pool.hist_span} frames of student pose history per stream")
    if k_dist:
        # A resume restores each stream's saved K, and the streams are re-matched
        # by scene name -- so if the pool size or the mixture changed since the
        # checkpoint, the multiset silently stops being the deck.  Check it.
        _have: Dict[int, int] = {}
        for _st in pool.streams:
            _have[_st.K] = _have.get(_st.K, 0) + 1
        _want: Dict[int, int] = {}
        for _k in stream_k:
            _want[_k] = _want.get(_k, 0) + 1
        if _have != _want:
            print(f"[v5] WARNING rank {rank} holds K {dict(sorted(_have.items()))} but the "
                  f"deck says {dict(sorted(_want.items()))} -- a resume across a changed "
                  f"--pool or --k_dist.  The realised mixture is the former, not the latter.")
    print(f"[pool] {len(pool.streams)} streams "
          f"{[(scenes[st.scene].name, st.t, f'K{st.K}', f'h{st.horizon}') for st in pool.streams]} in "
          f"{time.time() - t_setup:.0f}s, "
          f"{pool.bytes_resident() / 1e9:.1f} GB of host snapshots")

    # ── frozen probes ────────────────────────────────────────────────────────
    # [KR] 위에서 홀드아웃한 run들 중 일부를 실제로 채점(score)할 "고정
    # (frozen)" 상태로 만든다. theta_0(초기) 가중치로 롤아웃해서 만든 뒤,
    # 학습 중에는 절대 advance하지 않고 매번 다시 load()해서 채점만 한다.
    # Held out above; scored here.  Hold-out is free, scoring is a forward pass
    # each, so the count that gets SCORED is capped separately and spread across
    # scenes rather than exhausting one.
    t_p = time.time()
    # Round-robin over SCENES, not over run ids: sorting by run id fills the
    # probe set from whichever scenes happen to have the fewest runs, so a cap of
    # 6 over 10 scenes would score six windows of the four shortest sequences and
    # call it a held-out set.  One window per scene first, then second windows.
    per_scene = [sorted(sc.skip_runs) for sc in scenes]
    cand = [(si, h[j]) for j in range(max((len(h) for h in per_scene), default=0))
            for si, h in enumerate(per_scene) if j < len(h)]
    # ★ A RUN SHORTER THAN THE WINDOW CANNOT BE PROBED.  ``generate_run`` stops at
    # the end of a sequence, so a scene's last run is whatever was left over -- 4
    # frames on some dl3dv scenes.  ``t_probe`` below clamps the centring offset
    # at 0, so an L < S run yields t_probe = t0 and ``locate(bank, t0, S)`` then
    # asks for [t0, t0+S) which no single run contains: KeyError, at startup,
    # after every scene has been mapped.
    #
    # The other two places that pick a run already filter this way -- fresh_step
    # (``if a0 < 0 or r["L"] < S: return None, None``), FreshPool._cand, and the
    # window-mode fresh branch's ``train_runs`` -- and the long branch only ever
    # enters a run through ``next_valid_window``, which checks the fit.  The probe
    # was the one path that picked a run directly without checking.
    #
    # It stayed latent while the corpus was MCD: those scenes are 6000-9600
    # frames, so 20-40 runs, and the held-out ids land nowhere near the tail.  A
    # 2-run dl3dv scene holds out run 1 -- which IS the tail -- so 167 of 278
    # banks trip it the moment the new corpora are in the list.
    #
    # Only the PROBE set is filtered, not ``skip_runs``: a short tail run is
    # untrainable anyway (next_valid_window rejects it), so reserving it costs
    # training nothing, whereas holding out a long run instead would take real
    # ground away from a 2-run scene.
    cand = [(si, rid) for si, rid in cand if scenes[si].bank.runs[rid]["L"] >= S]
    if not cand:
        raise SystemExit(
            f"[probe] no held-out run of >= {S} frames survives across "
            f"{len(scenes)} scenes -- every reserved run is a short tail.  Lower "
            f"--S, raise --probe_runs_per_scene, or add a scene with more runs.")
    probes = []
    _cur_sd = None
    if theta0_sd is not None:
        _cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(theta0_sd, strict=True); model.to(dev)
    for si, rid in cand[:max(1, args.probe_max)]:
        sc = scenes[si]
        r = sc.bank.runs[rid]
        t_probe = r["t0"] + max(0, (r["L"] - S) // 2)
        probes.append({"scene": si, "t": t_probe, "run": rid,
                       "state": pool.roll_to(si, t_probe)})
    if _cur_sd is not None:
        model.load_state_dict(_cur_sd, strict=True); model.to(dev)
        del _cur_sd, theta0_sd
    print(f"[probe] {len(probes)} scored windows "
          f"{[(scenes[p['scene']].name, p['t']) for p in probes]} "
          f"(held out per scene: "
          f"{ {s_.name: sorted(s_.skip_runs) for s_ in scenes} }) "
          f"in {time.time() - t_p:.0f}s")

    log = {"meta": {**vars(args),
                    "scenes": [{"name": s_.name, "frames": s_.frames,
                                "bank": s_.bank.root, "covered": list(s_.covered),
                                "runs": len(s_.bank), "held_out": sorted(s_.skip_runs),
                                "ckpt_sha256": s_.bank.index.get("ckpt_sha256")}
                               for s_ in scenes],
                    "starts": starts},
           "steps": [], "probes": []}

    #: supervised frames per corpus, the denominator of the revisit rate
    ds_frames: Dict[str, int] = {}
    for sc in scenes:
        ds_frames[sc.dataset] = ds_frames.get(sc.dataset, 0) + sc.bank.frames_covered

    def run_probes(step):
        # [KR] 모든 probe 상태를 다시 load()해서 --K로(샘플링된 K가 아니라)
        # 고정 채점하고, gate 6 판정을 위한 loss/health 통계를 모은다. probe는
        # "가중치 변화"만 측정하는 게 목적이므로 항상 같은 문제(같은 K, 같은
        # 상태)를 유지해야 한다 -- 매번 다른 K로 채점하면 학습 진행과 문제
        # 난이도 변화가 섞여서 곡선을 해석할 수 없게 된다.
        # ★ THE PROBE STAYS AT --K, NOT AT THE SAMPLED K.  Its whole purpose is
        # to be the SAME problem at every step so a change in its value is the
        # weights and nothing else (see the module docstring); drawing a fresh K
        # per pass would make the series a mixture of different problems and the
        # gate unreadable.  Under v5 that means the probe measures one policy --
        # so it reports progress on that policy, not on the mixture.  Policy
        # breadth is an EVAL question, answered by the K grid in
        # docs/self-distill-ver5.md "Evaluation", not by this probe.
        rec = {"step": step, "loss": [], "parts": []}
        for p_ in probes:
            sc = scenes[p_["scene"]]
            pool.load(p_["state"])
            total, parts, out = supervised_step(model, sc.images, sc.bank, p_["t"],
                                                S, sf, args.K, crit, dtype, dev,
                                                grad=False)
            extra = sc.gt(out["pose_enc"][0], p_["t"], S) if sc.gt else {}
            rec["loss"].append(parts["loss"])
            rec["parts"].append({**parts, **health(out), **extra,
                                 "scene": sc.name, "t": p_["t"]})
            del out, total
        rec["mean"] = sum(rec["loss"]) / len(rec["loss"])
        # ★ AVERAGE OVER THE WINDOWS THAT HAVE GT, NOT ALL OR NOTHING.  The old
        # ``all(...)`` meant a mixed corpus reported no GT at all, because only the
        # MCD windows carry it -- and that is exactly how v6i and v7d ran for 1250
        # steps with their mid-training judgement resting on the teacher-imitation
        # probe alone.  That probe improved while the GT metric did not.
        _g = [p for p in rec["parts"] if "gt_ate" in p]
        if _g:
            rec["gt_ate_mean"] = sum(p["gt_ate"] for p in _g) / len(_g)
            rec["gt_rot_mean"] = sum(p["gt_rot_deg"] for p in _g) / len(_g)
            rec["gt_n"] = len(_g)
        # ★ v5 asks for PER-DATASET FRAME REVISIT COUNTS, and this is where they
        # become visible.  A 50/50 dataset-balanced sampler over corpora of very
        # different size revisits the small one far more often per frame -- v5's
        # own arithmetic is ~15x for MCD against SlowTV -- which works against
        # the domain over-specialisation the stage exists to fix.  Reporting
        # frames walked AND the resulting revisit rate makes the asymmetry a
        # number in the log instead of an assumption about the sampler.
        rec["frames_seen"] = dict(pool.frames_seen)
        rec["revisits"] = {d: (pool.frames_seen.get(d, 0) / max(1, n))
                           for d, n in ds_frames.items()}
        log["probes"].append(rec)
        wandb_log.log_probe(rec, step)
        # ★ ONLY THE WINDOWS THAT CARRY GT.  The aggregate above averages over the
        # GT-bearing subset (a mixed corpus has GT on its MCD windows alone), so
        # this line must select the same subset -- indexing every part was safe
        # only while the aggregate required ALL of them to have it.
        _gp = [p for p in rec["parts"] if "gt_ate" in p]
        gt_str = ("   | GT ate " + " ".join(f"{p['gt_ate']:.3f}" for p in _gp)
                  + "  rot " + " ".join(f"{p['gt_rot_deg']:.2f}" for p in _gp)
                  + f"  (n={len(_gp)}/{len(rec['parts'])})"
                  if _gp else "")
        print(f"  [probe @ {step:>3}] mean {rec['mean']:.4f}  "
              + "  ".join(f"{v:.4f}" for v in rec["loss"])
              + f"   contrast {rec['parts'][0]['depth_contrast']:.3f}" + gt_str, flush=True)
        if len(ds_frames) > 1:
            print("  [revisit] " + "  ".join(
                f"{d}: {pool.frames_seen.get(d, 0)}f / {n}f = {rec['revisits'][d]:.2f}x"
                for d, n in sorted(ds_frames.items())), flush=True)
        return rec

    fresh_cursor: Dict[int, int] = {i: 0 for i in range(len(scenes))}
    fpool = None
    if args.lam_fresh > 0:
        print(f"[fresh] preservation branch ON, lam_fresh={args.lam_fresh} "
              f"mode={args.fresh_mode} "
              f"(L2-SP {'OFF -- this is the only anchor to theta_0' if args.l2sp == 0 else f'also on at {args.l2sp}'})")
        if args.fresh_mode == "walk":
            # ★ BUILT AFTER THE PROBES.  pool.roll_to() clean_kv_cache()s the
            # live model and the probe section swaps in theta_0 to build them;
            # constructing here means the walk's burn-in is rolled with the
            # weights training actually starts from, and its host snapshot is
            # unaffected by anything the probes do to the live cache afterwards.
            t_f = time.time()
            fpool = FreshPool(model, scenes, S, dtype, dev,
                              n_streams=args.fresh_pool, cursor=fresh_cursor0,
                              keep_hist=long_crit is not None,
                              dataset_weights=ds_w)
            print(f"[fresh] {len(fpool.streams)} dense walk(s) over "
                  f"{len(fpool._cand)} runs, at "
                  f"{[(scenes[x.scene].name, x.rid) for x in fpool.streams]}, "
                  f"depth {fpool.streams[0].ws} -> "
                  f"{fpool.streams[0].ws + fpool.streams[0].L} kf per run, in "
                  f"{time.time() - t_f:.0f}s, "
                  f"{fpool.bytes_resident() / 1e9:.1f} GB of host snapshots")

    if is_main:
        run_probes(0)
    t_start = time.time()
    # [KR] ═══════════════════════ 메인 학습 루프 ═══════════════════════
    for step in range(step0, args.steps):
        # ── v5 unified sampler ───────────────────────────────────────────────
        # [KR] 이번 스텝이 "both"(correction 창 + fresh 항)인지, 아니면
        # "identity"(fresh 항만)인지, "correction"(교정 창만)인지 결정한다.
        # "both"는 v5 이전 방식(매 스텝 long 브랜치 + lam_fresh로 가중된 fresh
        # 브랜치를 둘 다 계산). --sampler unified는 v5의 방식으로, 매 스텝
        # "하나의" 상태 타입만 뽑아서 브랜치 하나만 계산(비용 절감).
        # "both" is the pre-v5 shape: the long branch every step, plus the fresh
        # branch weighted by lam_fresh.  Under --sampler unified each step draws
        # ONE state type from the single objective instead, which is what makes
        # a step cost one branch rather than two.
        #
        # ★ THE DRAW IS RANK-INVARIANT BY CONSTRUCTION (StateSampler.branch).
        # The two branches build different graphs, and DDP's
        # find_unused_parameters reducer derives the unused set from the local
        # graph -- ranks that disagreed about the branch would each wait for a
        # gradient the other never produces, and the run would HANG rather than
        # fail.  Nothing downstream re-checks this, so it is enforced at the
        # source: branch() hashes (seed, step) and touches no rank state.
        branch = "both" if args.sampler != "unified" else sampler.branch(step)
        if branch == "identity" and fpool is None and args.lam_fresh <= 0:
            branch = "correction"

        st = None
        if branch != "identity":
            st = pool.next()
            pool.activate(st)
        lr = lr_at(step, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_scale"]

        t_sup = st.t if st is not None else -1
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        sc = scenes[st.scene] if st is not None else scenes[0]
        # ★ FRESH BRANCH FIRST, then restore, then the long window.
        # fresh_step calls clean_kv_cache() and rebuilds the cache from its own
        # anchor.  With the long window's graph already alive, that clear does not
        # take -- the old cache tensors are still referenced, the fresh rollout
        # appends on top, and the next masked window sees KV of exactly twice the
        # size the mask was built for ("GCA mask ... does not match attention").
        # Running fresh first and re-activating leaves the LONG cache current when
        # its forward happens, which is the one the mask is built from.
        # [KR] 순서가 중요하다: fresh 브랜치를 "먼저" 계산하고 backward까지
        # 끝낸 다음, long(correction) 스트림 캐시를 다시 activate한다.
        # fresh_step은 clean_kv_cache()로 캐시를 비우고 자기 앵커부터 다시
        # 짓는데, 만약 long window의 그래프가 이미 살아있는 상태에서 이걸
        # 하면 옛날 캐시 텐서들이 여전히 참조되고 있어서 clean이 제대로 안
        # 먹고, fresh 롤아웃이 그 위에 덧붙여져서 캐시 크기가 두 배가 되어
        # 마스크 크기 불일치 에러가 난다. fresh를 먼저 끝내고 다시
        # activate하면, long window의 forward 시점에는 항상 long 캐시가
        # "라이브"인 상태를 유지할 수 있다.
        fresh_parts = None
        f_total = None
        fst = None
        fresh_si = None                   # which scene the identity state came from
        if branch in ("both", "identity") and (fpool is not None or args.lam_fresh > 0):
            if fpool is not None:
                # ★ SAME ORDERING RULE AS BELOW: activate, forward, backward, and
                # only then let anything else touch the cache.  The walk's state is
                # a host snapshot like the long stream's, so the two never share the
                # live cache -- they take turns holding it.
                fst = fpool.next()
                fpool.activate(fst)
                f_total, fresh_parts = fpool.supervise(
                    fst, crit, fwd=fwd, with_health=(branch == "identity"),
                    long_crit=long_crit, lam_long=args.lam_long,
                    alt_deltas=args.long_alt_deltas,
                    max_depth=args.long_max_teacher_depth,
                    long_crit_alt=long_crit_alt,
                    abs_crit=abs_crit, abs_units=abs_units,
                    abs_max_offset=args.abs_max_offset)
                fresh_si = fst.scene
                (args.lam_fresh * f_total).backward()
                f_total = None
            else:
                # Round-robin over this scene's TRAINING runs so the preserved
                # windows are not always the same stretch of route.  Under
                # --sampler unified there is no long stream to take the scene
                # from, so the sampler picks one -- dataset-balanced like every
                # other rollout draw.
                # ★ INDEX, NOT THE Scene OBJECT.  Scene is a dataclass, so its
                # generated __eq__ compares every field including ``images`` --
                # a tensor, whose elementwise comparison is not a bool.  Any
                # ``scenes.index(sc)`` here raises "truth value of a Tensor is
                # ambiguous", and only on the window-mode path, which no
                # existing run takes.
                si_f = st.scene if st is not None else 0
                if branch == "identity" and sampler is not None:
                    si_f = sampler.scene(pool.by_dataset)
                sc_f = scenes[si_f]
                train_runs = [i for i in range(len(sc_f.bank.runs))
                              if i not in sc_f.skip_runs and sc_f.bank.runs[i]["L"] >= S]
                if train_runs:
                    rid_f = train_runs[fresh_cursor[si_f] % len(train_runs)]
                    fresh_cursor[si_f] += 1
                    f_total, fresh_parts = fresh_step(model, sc_f, rid_f, S, crit,
                                                      dtype, dev, fwd=fwd)
                    fresh_si = si_f
                    # ★ BACKWARD HERE, WHILE THE FRESH CACHE IS STILL INSTALLED.
                    # See the note below activate().
                    if f_total is not None:
                        (args.lam_fresh * f_total).backward()
                        f_total = None
            if st is not None:
                pool.activate(st)          # back to this stream's long state

        if branch == "identity":
            # [KR] identity 스텝이면 correction(long) 창은 아예 안 돌린다 --
            # 이 스텝의 gradient는 오직 fresh(보존) 항 하나뿐이다.
            # ★ THE IDENTITY STEP IS COMPLETE HERE.  No long window runs, so the
            # gradient of this step is the preservation term alone -- which is
            # the point: v5's objective is one expectation over a state mixture,
            # and an identity draw contributes an identity state, not an
            # identity state plus a correction state.  Nothing after this may
            # touch the long stream: it was never activated, and advancing it
            # would walk a state the step did not train.
            if fresh_parts is None:
                raise SystemExit(
                    "[v5] identity step produced no loss -- the fresh branch is "
                    "off.  --sampler unified requires --lam_fresh > 0.")
            parts = dict(fresh_parts)
            h = {k: v for k, v in parts.items()
                 if k in ("depth_median", "depth_contrast",
                          "pose_step_mean", "pose_step_std")}
            total = None
        else:
            total, parts, out, lab = supervised_step(model, sc.images, sc.bank, st.t, S, sf,
                                                     st.K, crit, dtype, dev, fwd=fwd,
                                                     return_lab=True)
            h = health(out)

        # [KR] ★★★ 매우 중요한 버그 픽스: 브랜치마다 backward를 따로, 그것도
        # 각자의 캐시가 설치돼(installed) 있는 "동안"에 실행해야 한다.
        # 두 loss를 더해서 backward를 한 번만 하면(195스텝 동안 실제로 그렇게
        # 돌았음) 조용히 틀린 결과가 나온다: 체크포인트 재계산(checkpoint
        # recompute)이 forward 시점의 mask를, backward 시점에 "그때 라이브인
        # 캐시"와 짝지어 재사용하기 때문에, restore_state_cpu가 이미 캐시를
        # long 스트림 것으로 바꿔놓은 뒤에 backward를 하면 fresh의 gradient가
        # long 캐시를 기준으로 잘못 재계산된다. 그래서 각 브랜치는 forward
        # 직후 즉시 backward까지 끝내고 나서야 다음 브랜치로 넘어간다.
        # ★ ONE BACKWARD PER BRANCH, EACH WHILE ITS OWN CACHE IS INSTALLED.
        # Summing the two and running a single backward is what a DDP reduction
        # wants, and it ran for 195 steps -- but it is WRONG, and the way it is
        # wrong is silent.  aggregator/stream.py:651 passes ``kv_cache=self.kv_cache``
        # from INSIDE the checkpointed ``run()``, so the cache is looked up on the
        # module when the block is called; ``attn_mask=ctx["mask"]`` is a closure
        # over the forward.  Checkpoint recompute therefore pairs the mask built
        # at the fresh forward with WHATEVER CACHE IS LIVE AT BACKWARD TIME -- and
        # restore_state_cpu has by then replaced agg.kv_cache with the long
        # stream's dict.  So L_fresh's gradient was recomputed against the long
        # cache, not its own.  Both caches sit near the sliding-window cap, so the
        # lengths usually coincide and nothing complains; at step ~196 they
        # differed by 4 tokens and it finally raised
        #   "GCA mask (1,1,50016,125120) does not match attention [KV=125124]".
        # Running each backward before the cache is swapped is the fix, and it
        # also lowers peak memory: the fresh graph is freed before the long
        # forward allocates.  Under DDP this is the ordinary pattern -- one
        # forward, one backward, one reduction each -- and re-reducing an already
        # averaged gradient is idempotent, so the accumulation stays exact.
        if total is not None:
            # ── L_long, on the window the local loss just scored ──────────────
            # [KR] correction 창의 loss(total)에, 방금 계산한 out을 재사용해서
            # L_long을 더한다. 새로운 forward도 backward도 만들지 않고, 하나의
            # backward가 두 loss를 동시에 처리하게 한다(위의 "브랜치마다 한
            # 번의 backward" 규칙과 충돌하지 않음 -- 이건 같은 브랜치 안에서
            # 항을 더하는 것뿐).
            # ★ NO SECOND FORWARD AND NO SECOND BACKWARD.  Adding the term to
            # ``total`` before the single backward is what keeps the cache
            # ordering rule above intact, and it leaves the set of parameters
            # this step touches unchanged -- so a rank with no usable long pair
            # and a rank with 48 of them still agree about which gradients DDP
            # must reduce.
            if long_crit is not None:
                l_long, long_parts = long_term(
                    long_crit, sc.bank, st.hist, st.t, S,
                    out["pose_enc"][0].float(), dev,
                    alt_bank=sc.long_bank, alt_deltas=args.long_alt_deltas,
                    max_depth=args.long_max_teacher_depth,
                    long_crit_alt=long_crit_alt)
                parts.update(long_parts)
                if l_long is not None:
                    total = total + args.lam_long * l_long
            # ── L_abs, same window, same out, run gauge from st.hist ─────────
            # [KR] correction 창의 run 게이지 항. 추가 forward 없음. fit은
            # st.hist(학생 롤아웃 포즈 기록)와 뱅크 참조 포즈로 detach 상태에서
            # 맞춘다. 항은 항상 계산된다(fallback 포함) -- DDP rank 간 그래프
            # 일치를 위해서다.
            if abs_crit is not None:
                l_abs, abs_parts = AL.abs_term(
                    abs_crit, sc.bank, st.hist, st.t, S,
                    out["pose_enc"][0].float(), out["depth"][0].float(), lab, dev,
                    fit_mode=args.abs_fit, identity=False,
                    min_path=args.abs_min_path, max_offset=args.abs_max_offset,
                    units=abs_units, rot_exclude=abs_rot_excl, scene=sc.name)
                parts.update(abs_parts)
                if l_abs is not None:
                    total = total + l_abs
            del lab
            total.backward()
            del out
        del total, f_total
        gn = grad_norms(model)
        gclip = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.clip))
        opt.step()
        l2sp.apply(model, lr)
        torch.cuda.synchronize()
        t_step = time.time() - t0

        t1 = time.time()
        # [KR] optimizer.step()으로 가중치가 업데이트된 뒤, 각 스트림을
        # "방금 지도학습한 window를 지나쳐서" 새 가중치로 걷게 한다(advance).
        # 이게 바로 파일 맨 위 docstring이 말하는 "on-policy-with-lag" 만들기.
        if fst is not None:
            # Advance the dense walk with the freshly updated weights too -- the
            # same contract RolloutPool.advance documents -- then put the long
            # stream's cache back, because pool.advance() rolls from st.t on
            # whatever is live.  Skipping that restore would append the long
            # stream's next S frames on top of the fresh walk's cache.
            fpool.activate(fst)
            fpool.advance(fst)
            if st is not None:
                pool.activate(st)
        if st is not None:
            pool.advance(st)
            st.windows_done += 1
        t_adv = time.time() - t1

        # ── v5 per-window record ─────────────────────────────────────────────
        # [KR] v5 문서가 요구하는 지도학습 창(window) 단위 기록: 어느
        # 데이터셋/시퀀스, K, raw_frame_age(원본 프레임 나이), kf_age(실제
        # 유지된 키프레임 나이), window_start, axis_a_gap(학생-교사 히스토리
        # 격차) 등을 기록해서 나중에 K/나이 축으로 분석할 수 있게 한다.
        # "Log the following for every supervised window: dataset_id,
        # sequence_id, K, raw_frame_age, retained_keyframe_age, window_start" --
        # plus burn_in_eff, which step 1 asks for separately because the banks
        # were built at L=240 and a window at offset j has an effective teacher
        # burn-in of 72 + j rather than the nominal 72.
        if branch == "identity":
            # An identity state is by construction reset, dense and short: its
            # age is the teacher's own burn-in depth, not a long-rollout age.
            _sc_log = scenes[fresh_si] if fresh_si is not None else sc
            v5 = {"branch": "identity", "is_identity": 1.0, "K": 1,
                  "dataset": _sc_log.dataset, "scene": _sc_log.name,
                  "window_start": float(parts.get("depth_kf", 0.0)),
                  "raw_frame_age": 0.0,
                  "kf_age": float(parts.get("burn_in_eff", 0.0)),
                  # ★ THE AXIS THE SUPERVISION ACTUALLY LIVES ON, and the one
                  # nothing logged until now.  A teacher run re-anchors at
                  # a0 = t0 - 80, the student anchors once per rollout, so
                  #     axis_a_gap = (student raw history) - (teacher raw history)
                  #                = t - (80 + offset) = t0 - 80 = 240 * run_id
                  # -- independent of K.  It is 0 while a rollout is still inside
                  # the FIRST bank run (t0 = 80 -> a0 = 0 = the student's own
                  # anchor), which is when student and teacher walked the very
                  # same frames.  An identity state is teacher-matched by
                  # construction, so its gap is always 0.
                  "axis_a_gap": 0.0,
                  "horizon": 0, "resets": 0}
        else:
            v5 = {"branch": "correction", "is_identity": 0.0, "K": st.K,
                  "dataset": sc.dataset, "scene": sc.name,
                  "window_start": float(t_sup),
                  # raw frames this rollout has walked since it was anchored
                  "raw_frame_age": float(t_sup - st.anchor_t),
                  # keyframes actually RETAINED over that walk -- the axis that
                  # separates K=1 from K=28 at equal raw age, and the one the
                  # gain is expected to decay along
                  "kf_age": float((t_sup - sf + st.K - 1) // max(st.K, 1)),
                  # see the identity branch above: how much longer the student's
                  # raw history is than the teacher's for THIS window
                  "axis_a_gap": float(t_sup - (sf + parts["burn_in_eff"])),
                  "horizon": st.horizon, "resets": st.resets}
        rec = {"step": step, "sid": (st.sid if st is not None else -1),
               "t": t_sup, "next_t": (st.t if st is not None else -1),
               "scene_idx": (st.scene if st is not None else -1),
               "lr": lr, **parts, **h, **v5,
               **({f"fresh_{k}": v for k, v in fresh_parts.items()}
                  if fresh_parts and branch != "identity" else {}),
               "grad_norm": gn, "grad_total": gclip,
               "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
               "step_s": t_step, "advance_s": t_adv}
        log["steps"].append(rec)
        wandb_log.log_step(rec, step)
        if step % 5 == 0 or step == args.steps - 1:
            print(f"  [{step:>3}] {'ID ' if branch == 'identity' else 'COR'} "
                  f"s{rec['sid']} {v5['scene'][:12]:<12} "
                  f"K{v5['K']:<2} t={int(v5['window_start']):>5} "
                  f"age={int(v5['raw_frame_age']):>4}/{int(v5['kf_age']):>4}kf "
                  f"loss {parts['loss']:.4f} "
                  f"(rot {parts['L_rot_deg']:.3f} dir {parts['L_dir']:.3f} "
                  f"mot {parts['L_motion_depth']:.3f} dep {parts['L_depth_si']:.4f})  "
                  f"|g| {gclip:.2f}  contrast {h['depth_contrast']:.3f}  "
                  # L_long: the value and WHICH deltas actually fired.  A silent
                  # zero here means every pair was masked or the window sits too
                  # early in its run -- both legal, both worth seeing.
                  # both sources, each with its own ladder: the terms can be
                  # split (rot/dir from one teacher, scale from another) and a
                  # line that only knew about one would show a silent zero.
                  + "".join(
                      f"{tag} {parts[f'L_{tag}']:.3f}"
                      f"[{'/'.join(str(int(parts.get(f'{tag}_n_d{d}', 0))) for d in cr.ladder)}]  "
                      for tag, cr in (("long", long_crit), ("longalt", long_crit_alt))
                      if cr is not None and f"L_{tag}" in parts)
                  + (f"fresh@{rec['fresh_depth_kf']:.0f}kf "
                     f"({rec['fresh_loss']:.3f})  " if "fresh_depth_kf" in rec else "")
                  # L_abs: the weighted total, the two scale residuals in
                  # relative units, the fitted scale and HOW it was fitted --
                  # a run that fell back on most windows must be visible here
                  + (f"abs {parts['L_abs']:.3f}"
                     + (f" ts {parts['L_trans_scale']:.3f}" if "L_trans_scale" in parts else "")
                     + (f" ds {parts['L_depth_scale']:.3f}" if "L_depth_scale" in parts else "")
                     + (f" ap {parts['L_abs_pos']:.2f}" if "L_abs_pos" in parts else "")
                     + (f" ar {parts['abs_rot_deg']:.2f}°" if "abs_rot_deg" in parts else "")
                     + (f" rt {parts['L_rel_trans']:.2f}" if "L_rel_trans" in parts else "")
                     + f" s={parts['fit_s']:.3f} {parts['fit_mode']}@{int(parts['abs_offset'])}  "
                     if "L_abs" in parts else "")
                  + f"{t_step:.1f}+{t_adv:.1f}s  {rec['peak_gb']:.0f}GB", flush=True)
        # [KR] NaN 판정은 반드시 "전체 rank가 함께" 내려야 한다. 각 rank는
        # 자기 window만 채점하므로 한 rank에서만 NaN이 뜰 수 있는데, 그
        # rank만 로컬 판단으로 break하면 그 rank는 dist.barrier()로 가고
        # 다른 rank는 다음 스텝의 backward/all-reduce로 진행해버려서 NCCL
        # 집합 통신(collective)이 서로 어긋나 "행(hang)"이 나버린다(에러가
        # 아니라 그냥 멈춤 -- ~20초/스텝이면 조용히 런 전체를 태워먹는다).
        # 그래서 all_reduce로 깃발을 합쳐서 모든 rank가 "같이" 결정한다.
        # ★ THE NaN VERDICT MUST BE COLLECTIVE.  Each rank scores its own window,
        # so one rank can see nan while the other does not.  Breaking on the
        # LOCAL value sends that rank to dist.barrier() while the other proceeds
        # to the next step's backward and its all-reduce -- mismatched collectives
        # on the NCCL group, i.e. a HANG, not an abort.  At ~20 s/step that quietly
        # burns the whole run.  All-reduce the flag so both ranks decide together.
        _nan = torch.tensor([0.0 if parts["loss"] == parts["loss"] else 1.0], device=dev)
        if world > 1:
            import torch.distributed as dist
            dist.all_reduce(_nan)
        if _nan.item() > 0:
            if is_main:
                print("  [ABORT] loss is nan on at least one rank")
            break
        if is_main and args.probe_every and (step + 1) % args.probe_every == 0:
            run_probes(step + 1)
        if args.save_every and args.save and (step + 1) % args.save_every == 0:
            # [KR] 체크포인트 저장을 위해 모든 rank의 스트림 위치를 모은다
            # (all_gather_object는 collective라서 반드시 모든 rank가 도달해야
            # 함 -- 그래서 이 블록은 is_main 조건 없이 모든 rank가 실행).
            # ★ ONLY RANK 0 WRITES, BUT EVERY RANK'S WALKS MUST BE IN THE FILE.
            # Saving pool.streams alone stores rank 0's five walks; on resume the
            # other rank finds none of its scenes, falls back to the fresh
            # placement, and restarts all of its walks at frame 80 -- re-training
            # ground already covered, which is the exact failure the "streams"
            # field exists to prevent.  Observed on the first resume of this run.
            # Gather first (collective -- every rank must reach it), write after.
            _streams = [{"sid": x.sid, "scene_name": scenes[x.scene].name,
                         "scene": x.scene, "t": x.t,
                         # v5: a rollout is (scene, K, horizon, age).  All four
                         # travel together or the resumed walk is a different
                         # rollout wearing the old one's position.
                         "K": x.K, "horizon": x.horizon, "anchor_t": x.anchor_t,
                         "resets": x.resets,
                         "windows_done": x.windows_done} for x in pool.streams]
            if world > 1:
                import torch.distributed as dist
                _buf = [None] * world
                dist.all_gather_object(_buf, _streams)
                _streams = [y for part in _buf for y in part]
        if is_main and args.save_every and args.save and (step + 1) % args.save_every == 0:
            _p = f"{os.path.splitext(args.save)[0]}.step{step + 1}.pt"
            os.makedirs(os.path.dirname(_p) or ".", exist_ok=True)
            torch.save({**_export_weights(model, args),
                        "opt": opt.state_dict(), "step": step + 1,
                        # where every stream had walked to.  Without this a resume
                        # restarts all 10 walks at frame 80 and re-trains ground
                        # the model has already seen, which is worse than useless:
                        # it silently changes the data distribution mid-run.
                        "streams": _streams,
                        # where the dense walk had got to; see the resume note.
                        "fresh_cursor": (fpool._cursor if fpool is not None else 0),
                        "scenes": [sc_.name for sc_ in scenes],
                        "meta": log["meta"]}, _p)
            with open(args.out, "w") as _f:          # log alongside, so a crash
                json.dump(log, _f, indent=1)         # still leaves a readable run
            print(f"  [ckpt] {_p}", flush=True)

    if world > 1:
        import torch.distributed as dist
        dist.barrier()
    if not is_main:
        return
    # [KR] ── 최종 요약 출력 ──────────────────────────────────────────────
    # 학습 루프가 끝난 뒤, "첫 probe(p0)" vs "마지막 probe(pN)"를 비교해서
    # loss가 실제로 하강했는지(gate 6), depth/pose가 붕괴하지 않았는지를
    # 판정하고, 실제로 학습에 반영된 K 혼합 비율 등을 출력/저장한다.
    log["wall_s"] = time.time() - t_start
    p0, pN = log["probes"][0], log["probes"][-1]
    print(f"\n{'=' * 78}")
    print(f"  {len(log['steps'])} steps in {log['wall_s'] / 60:.1f} min "
          f"({log['wall_s'] / max(len(log['steps']), 1):.1f} s/step)")
    print(f"  probe loss  {p0['mean']:.4f} -> {pN['mean']:.4f}  "
          f"({(pN['mean'] - p0['mean']) / p0['mean'] * 100:+.1f}%)")
    c0 = p0["parts"][0]["depth_contrast"]
    cN = pN["parts"][0]["depth_contrast"]
    print(f"  depth contrast {c0:.3f} -> {cN:.3f}   "
          f"pose step {p0['parts'][0]['pose_step_mean']:.4f} -> "
          f"{pN['parts'][0]['pose_step_mean']:.4f}")
    if "gt_ate_mean" in p0:
        print(f"  GT (probe windows)  ate {p0['gt_ate_mean']:.4f} -> {pN['gt_ate_mean']:.4f} m "
              f"({(pN['gt_ate_mean'] - p0['gt_ate_mean']) / p0['gt_ate_mean'] * 100:+.1f}%)   "
              f"rot {p0['gt_rot_mean']:.3f} -> {pN['gt_rot_mean']:.3f} deg "
              f"({(pN['gt_rot_mean'] - p0['gt_rot_mean']) / p0['gt_rot_mean'] * 100:+.1f}%)")
        log["gt"] = {"ate": [p0["gt_ate_mean"], pN["gt_ate_mean"]],
                     "rot": [p0["gt_rot_mean"], pN["gt_rot_mean"]]}
    # [KR] "선언된" K 분포가 아니라 "실제로 학습에 쓰인" K 분포를 보고한다
    # (--k_dist는 롤아웃 단위 분포지만 loss는 스텝 단위 평균이라서 실제
    # 비율이 다를 수 있음 -- k_deck의 주석 참고). axis-A gap(학생-교사 히스토리
    # 격차) 통계도 함께 보고.
    # ★ THE REALIZED K MIXTURE, NOT THE DECLARED ONE.  --k_dist names a
    # distribution over ROLLOUTS; the loss averages over STEPS, and a rollout
    # contributes horizon/S of them.  A 300-step run drew 19 rollouts of 1-22
    # steps each, so the step-weighted mixture came out 44% K=12 and 0% K=8
    # against a declared 10/10 -- the cell's identity is what it TRAINED on, so
    # print that rather than leaving it to be re-derived from the json.  Also
    # report the axis-A coverage, since "how deep did the student's history get
    # past the teacher's" is the axis ver5 asks the results to be plotted on.
    _cor = [r for r in log["steps"] if r.get("branch") != "identity"]
    _n_id = len(log["steps"]) - len(_cor)
    if _cor:
        _kc: Dict[int, int] = {}
        for r in _cor:
            _kc[int(r["K"])] = _kc.get(int(r["K"]), 0) + 1
        log["k_realized"] = {"by_step_correction": _kc,
                             "identity_steps": _n_id, "total_steps": len(log["steps"])}
        print("  K mixture ACTUALLY TRAINED (correction steps): "
              + "  ".join(f"K{k}:{v} ({v / len(_cor):.0%})" for k, v in sorted(_kc.items())))
        print(f"  branch split: correction {len(_cor)} / identity {_n_id} "
              f"({_n_id / max(len(log['steps']), 1):.0%}); identity always runs K=1 "
              f"(every bank has teacher_interval=1), so K=1 masked-window exposure "
              f"over ALL steps is {(_n_id + _kc.get(1, 0)) / max(len(log['steps']), 1):.0%}")
        _ga = sorted(r.get("axis_a_gap", 0.0) for r in _cor)
        _z = sum(1 for g in _ga if g <= 0)
        print(f"  axis-A gap (student raw history - teacher's): max {_ga[-1]:.0f}  "
              f"median {_ga[len(_ga) // 2]:.0f}  zero on {_z}/{len(_cor)} "
              f"({_z / len(_cor):.0%}) correction steps")
        log["axis_a"] = {"max": _ga[-1], "median": _ga[len(_ga) // 2], "zero_frac": _z / len(_cor)}
    # [KR] Gate 6 판정: (1) probe loss가 하강했는가, (2) depth contrast와
    # pose step 크기가 초기값의 절반 이상을 유지해서 "붕괴"하지 않았는가.
    descending = pN["mean"] < p0["mean"]
    alive = cN > 0.5 * c0 and pN["parts"][0]["pose_step_mean"] > 0.5 * p0["parts"][0]["pose_step_mean"]
    log["gate6"] = {"loss_descending": bool(descending), "no_collapse": bool(alive)}
    print(f"  GATE 6: loss descending {'YES' if descending else 'NO'}   "
          f"no collapse {'YES' if alive else 'NO'}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({**_export_weights(model, args), "meta": log["meta"], "gate6": log["gate6"]},
                   args.save)
        log["saved_weights"] = args.save
        print(f"[saved] {args.save}  ({os.path.getsize(args.save) / 1e9:.1f} GB)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(log, f, indent=1)
    print(f"[saved] {args.out}")

    wandb_log.log_summary({
        "steps": len(log["steps"]), "wall_min": log["wall_s"] / 60,
        "s_per_step": log["wall_s"] / max(len(log["steps"]), 1),
        "probe_loss_first": p0["mean"], "probe_loss_last": pN["mean"],
        "probe_loss_pct": (pN["mean"] - p0["mean"]) / p0["mean"] * 100,
        **({"gt_ate_first": p0["gt_ate_mean"], "gt_ate_last": pN["gt_ate_mean"],
            "gt_rot_first": p0["gt_rot_mean"], "gt_rot_last": pN["gt_rot_mean"]}
           if "gt_ate_mean" in p0 else {}),
        **log["gate6"], "preset": args.preset,
    })
    wandb_log.save_file(args.out)
    wandb_log.finish()


if __name__ == "__main__":
    main()
