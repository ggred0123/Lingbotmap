"""Scheduling tests for ``FreshPool`` -- the dense (K_t=1) fresh walk.

The pool cannot be exercised end-to-end without a GPU, but everything that can
be gotten WRONG here is bookkeeping: which frames get rolled, what
``window_start`` the mask is built at, which label offset is scored, and when a
stream re-anchors.  All of that is testable with a stub model, and all of it is
off-by-one territory that a training run would surface only as a slightly worse
curve.

The invariant that matters most:

    cache depth at the supervised window == the teacher's cache depth for the
    labels being scored == sf + B*K_t + offset

because that equality is the entire justification for scoring theta_0's labels
against a state the student walked itself (see FreshPool's docstring).
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from lingbot_map.train.trainer import FreshPool          # noqa: E402

SF, B, KT, L, S = 8, 72, 1, 240, 48


class StubBank:
    """Just the two things FreshPool touches: ``runs`` and ``get``."""

    def __init__(self, t0s, L=L):
        self.runs = [{"t0": t0, "L": L, "burn_in": B, "scale_frames": SF,
                      "teacher_interval": KT} for t0 in t0s]
        self.gets = []

    def get(self, rid, offset, S_, device=None):
        r = self.runs[rid]
        # the real bank raises here; make the test fail the same way
        assert offset + S_ <= r["L"], f"offset {offset} + {S_} > L {r['L']}"
        self.gets.append((rid, offset))
        return {"pose_enc": torch.zeros(S_, 9), "depth": torch.zeros(S_, 4, 4)}


class StubScene:
    def __init__(self, name, t0s, n_frames=4000, skip=()):
        self.name = name
        self.bank = StubBank(t0s)
        self.skip_runs = set(skip)
        self.images = torch.zeros(1, n_frames, 3, 4, 4)


class StubModel:
    """Records every frame index it is asked to run, in a0-relative coords."""

    def __init__(self):
        self.rolled = []          # (n_frames, num_frame_per_block)
        self.cleans = 0
        self.windows = []         # (window_start, keyframe_interval)

    def clean_kv_cache(self):
        self.cleans += 1

    def _set_skip_append(self, v):
        pass

    def forward(self, imgs, num_frame_for_scale=None, num_frame_per_block=None,
                causal_inference=None):
        self.rolled.append(imgs.shape[1])
        return {"pose_enc": torch.zeros(1, num_frame_per_block or 1, 9),
                "depth": torch.zeros(1, num_frame_per_block or 1, 4, 4)}

    __call__ = forward

    def masked_window(self, window_start, keyframe_interval):
        self.windows.append((window_start, keyframe_interval))

        class _N:
            def __enter__(s): return s
            def __exit__(s, *a): return False
        return _N()


def _pool(monkeypatch, scenes, n_streams=1, cursor=0):
    """FreshPool with the host-snapshot helpers stubbed out."""
    import grad_probe
    monkeypatch.setattr(grad_probe, "snapshot_state_cpu", lambda m: {"tag": id(m)})
    monkeypatch.setattr(grad_probe, "restore_state_cpu", lambda m, s, d: None)
    monkeypatch.setattr(grad_probe, "detach_caches", lambda m, **k: None)
    model = StubModel()
    return FreshPool(model, scenes, S, torch.float32, "cpu",
                     n_streams=n_streams, cursor=cursor), model


def _crit(sp, tp, sd, td, tc=None):
    return torch.zeros(()), {"loss": 0.0}


# ── the walk itself ──────────────────────────────────────────────────────────

def test_burn_in_is_paid_once_per_run_not_once_per_step(monkeypatch):
    """The whole cost argument: 80 forwards per RUN, not per step."""
    sc = StubScene("s", [80, 320, 560])
    pool, model = _pool(monkeypatch, [sc])
    st = pool.streams[0]
    assert model.cleans == 1
    # one sf-frame anchor block + (ws - sf) single frames
    assert model.rolled[0] == SF
    assert len(model.rolled) == 1 + (SF + B * KT - SF)
    before = len(model.rolled)

    # four advances stay inside the run: S single frames each, no re-anchor
    for _ in range(4):
        pool.advance(st)
    assert model.cleans == 1, "re-anchored while the run still had windows left"
    assert len(model.rolled) - before == 4 * S


def test_one_run_sweeps_the_whole_deployment_budget(monkeypatch):
    """sf + B + L == 320 is not a coincidence -- it is the cache budget."""
    sc = StubScene("s", [80, 320])
    pool, model = _pool(monkeypatch, [sc])
    st = pool.streams[0]
    depths = []
    for _ in range(L // S):
        pool.supervise(st, _crit)
        depths.append(model.windows[-1][0])
        pool.advance(st)
    assert depths == [80, 128, 176, 224, 272]
    # the last window ENDS at the budget
    assert depths[-1] + S == SF + B * KT + L == 320
    # and the shipped 'window' mode only ever supervises the first of these
    assert depths[0] == SF + B * KT


def test_window_start_equals_teacher_cache_depth(monkeypatch):
    """window_start must equal sf + B*K_t + offset at every step."""
    sc = StubScene("s", [80, 320])
    pool, model = _pool(monkeypatch, [sc])
    st = pool.streams[0]
    for _ in range(6):
        off = st.offset
        pool.supervise(st, _crit)
        ws, kt = model.windows[-1]
        assert ws == SF + B * KT + off
        assert kt == KT
        assert sc.bank.gets[-1] == (st.rid, off)
        pool.advance(st)


def test_frames_walked_are_contiguous_and_match_the_windows(monkeypatch):
    """No gap and no overlap between what is supervised and what is rolled."""
    sc = StubScene("s", [80, 320])
    pool, model = _pool(monkeypatch, [sc])
    st = pool.streams[0]
    covered = SF + B * KT           # frames already in the cache after burn-in
    for _ in range(4):
        pool.supervise(st, _crit)
        assert model.windows[-1][0] == covered
        n_before = len(model.rolled)
        pool.advance(st)
        assert len(model.rolled) - n_before == S
        covered += S


# ── run rotation ─────────────────────────────────────────────────────────────

def test_stream_reanchors_when_the_run_runs_out(monkeypatch):
    sc = StubScene("s", [80, 320, 560])
    pool, model = _pool(monkeypatch, [sc])
    st = pool.streams[0]
    first = st.rid
    for _ in range(L // S):
        pool.advance(st)
    assert st.rid != first and st.offset == 0
    assert model.cleans == 2


def test_labels_never_run_past_the_end_of_a_run(monkeypatch):
    """StubBank.get asserts the real IndexError condition."""
    sc = StubScene("s", [80, 320, 560])
    pool, model = _pool(monkeypatch, [sc])
    st = pool.streams[0]
    for _ in range(40):
        pool.supervise(st, _crit)
        pool.advance(st)


def test_candidates_are_scene_major(monkeypatch):
    """Consecutive runs come from different scenes, as for the probe set."""
    scenes = [StubScene("a", [80, 320, 560]), StubScene("b", [80, 320]),
              StubScene("c", [80])]
    pool, _ = _pool(monkeypatch, scenes)
    assert [c[0] for c in pool._cand] == [0, 1, 2, 0, 1, 0]


def test_short_and_held_out_runs_are_dropped(monkeypatch):
    """L < S, held-out runs, and a0 < 0 must never be walked.

    ``fresh_step`` had to check ``r["L"] < S`` itself and hit the 40-frame
    tail run of kth_day_06; the pool filters once, up front.
    """
    sc = StubScene("s", [80, 320, 560, 800])
    sc.bank.runs[2]["L"] = 40                  # the leftover tail
    sc.skip_runs = {1}                         # held out for probing
    sc.bank.runs[0]["t0"] = 40                 # a0 = 40 - 80 < 0
    pool, _ = _pool(monkeypatch, [sc])
    assert [rid for _, rid in pool._cand] == [3]


def test_no_candidate_is_a_loud_failure(monkeypatch):
    sc = StubScene("s", [80])
    sc.bank.runs[0]["L"] = 10
    with pytest.raises(SystemExit):
        _pool(monkeypatch, [sc])


# ── interleaving and resume ──────────────────────────────────────────────────

def test_streams_interleave_round_robin(monkeypatch):
    scenes = [StubScene("a", [80, 320]), StubScene("b", [80, 320])]
    pool, _ = _pool(monkeypatch, scenes, n_streams=2)
    assert pool.streams[0].scene != pool.streams[1].scene
    assert [pool.next() is pool.streams[i % 2] for i in range(4)] == [True] * 4


def test_cursor_resume_continues_the_sweep_instead_of_restarting(monkeypatch):
    """A resume must not send the walk back to the first run of scene 0."""
    scenes = [StubScene("a", [80, 320, 560]), StubScene("b", [80, 320, 560])]
    p0, _ = _pool(monkeypatch, scenes)
    visited = [(p0.streams[0].scene, p0.streams[0].rid)]
    for _ in range(3):
        p0.begin_run(p0.streams[0])
        visited.append((p0.streams[0].scene, p0.streams[0].rid))

    resumed, _ = _pool(monkeypatch, scenes, cursor=p0._cursor)
    here = (resumed.streams[0].scene, resumed.streams[0].rid)
    assert here not in visited, f"resume re-covered {here}"
    # and it is exactly where an uninterrupted run would have been next
    p0.begin_run(p0.streams[0])
    assert here == (p0.streams[0].scene, p0.streams[0].rid)

    restarted, _ = _pool(monkeypatch, scenes)          # cursor=0, the old bug
    assert (restarted.streams[0].scene, restarted.streams[0].rid) == visited[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
