"""Weights & Biases logging for the self-distillation trainer.

★ THE TWO LOSSES ARE NOT THE SAME QUANTITY, so they get different namespaces.

    train_prequential/*   the supervised window's loss, scored at the PRE-update
                          weights, each window visited once.  Already a held-out
                          curve -- but it also mixes scene difficulty and, once
                          several scenes are in the pool, scene identity.  It
                          goes DOWN for reasons that have nothing to do with
                          learning (an easier stretch of route) and UP likewise.
    probe_frozen/*        a fixed set of held-out windows, frozen streaming state,
                          re-scored every N steps.  Input and target never move,
                          so a change here is the weights and nothing else.
                          docs/phase1-plan.md section 3-T5: the gate is read off
                          THIS one.

Naming them apart is not cosmetic.  In gate 6 the prequential curve fell 5.5%
while the probe fell 10.4%, and in gate 6b the two probes disagreed in sign --
a single "loss" panel would have hidden both.

Everything here is best-effort: a wandb outage, a missing package, or a bad key
must never take down a run that costs an hour of GPU time.  Every entry point
swallows its exceptions and degrades to a no-op.
"""

import os
import socket
import time
import sys
from typing import Any, Dict, Optional

# wandb lives outside the container's system python, in the workspace:
#     <workspace>/pylibs        (pip install --target)
#
# ★ APPENDED, never prepended.  ``pip install --target`` copies the whole
# dependency closure, and 11 of those 18 packages already exist in this
# container at DIFFERENT versions -- protobuf 7.35 vs 6.33, packaging 26.3 vs
# 25.0 (nvidia-dali pins <=25.0), pydantic, pyyaml, requests, urllib3.  Putting
# the directory on PYTHONPATH would shadow all of them ahead of site-packages
# and quietly re-version the torch stack underneath a multi-hour training run.
# Appending inverts the precedence: anything the container already has wins, and
# only the genuinely missing modules (wandb, sentry_sdk, opentelemetry) resolve
# here.
_PYLIBS = os.environ.get("LINGBOT_PYLIBS") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "pylibs"))
if os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.append(_PYLIBS)

# The API key is NOT in source.  WANDB_API_KEY in the environment wins; otherwise
# it is read from a gitignored file <repo>/.wandb_key (one line), so the tree
# can be committed and pushed without rotating the key.
_KEY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".wandb_key"))


def _api_key() -> Optional[str]:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    try:
        with open(_KEY_FILE) as f:
            return f.read().strip() or None
    except OSError:
        return None


_PROJECT = "streaming3d-self-distill"

_run = None
_enabled = False


def init(config: Dict[str, Any], name: Optional[str] = None,
         project: str = _PROJECT, group: Optional[str] = None,
         tags=(), mode: str = "online") -> bool:
    """Start a run.  Returns whether logging is live; never raises."""
    global _run, _enabled
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed -- logging disabled "
              "(pip install --user --break-system-packages wandb)")
        return False
    try:
        _k = _api_key()
        if _k:
            os.environ.setdefault("WANDB_API_KEY", _k)
        _run = wandb.init(
            project=project, name=name, group=group, tags=list(tags),
            mode=mode, config=config,
            settings=wandb.Settings(host=socket.gethostname()),
        )
        _enabled = True
        print(f"[wandb] {_run.url}")
        return True
    except Exception as e:                                   # noqa: BLE001
        print(f"[wandb] init failed ({type(e).__name__}: {e}) -- logging disabled")
        _run = None
        _enabled = False
        return False


def _flat(prefix: str, d: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    """One level of nesting is all the trainer produces (grad_norm is a dict)."""
    for k, v in d.items():
        if isinstance(v, dict):
            _flat(f"{prefix}{k}/", v, out)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = v
    return out


#: keys of the step record that describe WHERE the step happened rather than how
#: it went -- useful as wandb metrics but conceptually separate from the loss
_POSITION = ("sid", "t", "next_t", "scene", "scene_idx", "dist_m",
             # where the dense fresh walk is in its depth sweep -- the
             # thing FreshPool exists to move, so it belongs next to the
             # long stream's position rather than among the loss terms.
             "fresh_depth_kf", "fresh_offset", "fresh_run",
             # ★ v5 state-mixture coordinates.  Every one of these describes
             # WHICH state the step drew, not how well it went, so they belong
             # in the position namespace -- and they are the axes v5 asks the
             # results to be plotted against ("Results must be plotted against
             # both raw-frame age and retained-keyframe age").  Filing them as
             # losses would put them on the loss panel and average them across
             # a mixture, which is precisely the reading the plan forbids.
             "K", "raw_frame_age", "kf_age", "window_start", "horizon",
             "resets", "burn_in_eff", "run", "offset", "is_identity",
             "dataset", "branch",
             # how much longer the student's raw history is than the teacher's
             # for this window (= t0 - 80 = 240 * run_id, independent of K).
             # 0 while the rollout is still inside the first bank run.
             "axis_a_gap",
             "fresh_burn_in_eff", "fresh_K")


def log_step(rec: Dict[str, Any], step: int) -> None:
    """One optimizer step.  ``rec`` is the trainer's per-step dict."""
    if not _enabled:
        return
    try:
        import wandb
        m: Dict[str, Any] = {}
        grads = rec.get("grad_norm") or {}
        for k, v in rec.items():
            if k in ("step", "grad_norm"):
                continue
            if k in _POSITION:
                m[f"stream/{k}"] = v
            elif k in ("step_s", "advance_s", "peak_gb"):
                m[f"cost/{k}"] = v
            elif k in ("lr", "grad_total"):
                m[f"optim/{k}"] = v
            elif isinstance(v, dict):
                _flat(f"train_prequential/{k}/", v, m)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                m[f"train_prequential/{k}"] = v
        # ★ THE MIXTURE MUST BE SPLITTABLE AFTER THE FACT.  v5 asks for each
        # loss component logged separately FOR EVERY K, and a single
        # train_prequential/loss series averages the sampled K values into one
        # curve that answers no question the plan asks.  Mirroring the loss under
        # its own K and branch gives a per-K series without a second log call.
        # ★ IDENTITY STEPS DO NOT BELONG IN by_K.  They run at K=1 because every
        # bank carries teacher_interval=1 (checked: 278/278), so filing them under
        # by_K/K1 mixes two structurally different populations -- an identity state
        # is reset, teacher-matched and shallow (its student/teacher raw-history
        # gap is 0 by construction), a K=1 correction state is a long rollout whose
        # gap grows to the horizon.  In v5c that made by_K/K1 86% identity, so the
        # "K=1 policy" curve was mostly the preservation term.  by_K is the
        # CORRECTION policy axis; identity gets its own namespace.
        _K, _br = rec.get("K"), rec.get("branch")
        if isinstance(_K, (int, float)) and not isinstance(_K, bool) and _br != "identity":
            for _k in ("loss", "L_rot_deg", "L_dir", "L_motion_depth", "L_depth_si",
                       # ★ L_long SPLIT BY K IS NOT OPTIONAL.  The student rolls
                       # at K from the deck while every bank label is K_t=1, and
                       # that mismatch accumulates with Delta -- so a long term
                       # averaged over K hides the axis its own bias audit
                       # (design doc section 9) asks about.
                       "L_long", "long_pairs"):
                if isinstance(rec.get(_k), (int, float)):
                    m[f"by_K/K{int(_K)}/{_k}"] = rec[_k]
            for _k in [k for k in rec if k.startswith("long_") and
                       isinstance(rec[k], (int, float))]:
                m[f"by_K/K{int(_K)}/{_k}"] = rec[_k]
            m[f"by_K/K{int(_K)}/grad_total"] = rec.get("grad_total")
        if _br == "identity":
            for _k in ("loss", "L_rot_deg", "L_dir", "L_motion_depth", "L_depth_si",
                       "burn_in_eff", "L_long", "long_pairs"):
                if isinstance(rec.get(_k), (int, float)):
                    m[f"by_identity/{_k}"] = rec[_k]
        if _br:
            for _k in ("loss", "grad_total"):
                if isinstance(rec.get(_k), (int, float)):
                    m[f"by_branch/{_br}/{_k}"] = rec[_k]
        for g, v in grads.items():
            m[f"grad/{g}"] = v
        # gradient SHARE, not just norms: docs/add_loss.md section 5 asks for the
        # share because that is what says which module the update actually moves.
        tot = sum(v for v in grads.values() if v == v)
        if tot > 0:
            for g, v in grads.items():
                m[f"grad_share/{g}"] = v / tot
        wandb.log(m, step=step)
    except Exception:                                        # noqa: BLE001
        pass


def log_probe(rec: Dict[str, Any], step: int) -> None:
    """The frozen held-out probes.  ``rec`` is one entry of log['probes']."""
    if not _enabled:
        return
    try:
        import wandb
        m: Dict[str, Any] = {"probe_frozen/loss_mean": rec.get("mean")}
        # v5 "log per-dataset frame revisit counts" -- the sampler's actual
        # coverage, not the weights it was configured with.
        for _d, _n in (rec.get("frames_seen") or {}).items():
            m[f"corpus/frames_seen/{_d}"] = _n
        for _d, _r in (rec.get("revisits") or {}).items():
            m[f"corpus/revisits/{_d}"] = _r
        for i, v in enumerate(rec.get("loss", [])):
            m[f"probe_frozen/p{i}/loss"] = v
        for i, parts in enumerate(rec.get("parts", [])):
            _flat(f"probe_frozen/p{i}/", parts, m)
        # a scalar per term averaged over probes, so the panel is readable when
        # the probe count grows past two
        parts_list = rec.get("parts", [])
        if parts_list:
            for k in parts_list[0]:
                vals = [p[k] for p in parts_list
                        if isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool)]
                if vals:
                    m[f"probe_frozen/mean/{k}"] = sum(vals) / len(vals)
        wandb.log({k: v for k, v in m.items() if v is not None}, step=step)
    except Exception:                                        # noqa: BLE001
        pass


def log_summary(d: Dict[str, Any]) -> None:
    """End-of-run scalars (gate verdicts, totals)."""
    if not _enabled:
        return
    try:
        import wandb
        for k, v in _flat("", d, {}).items():
            wandb.run.summary[k] = v
        for k, v in d.items():
            if isinstance(v, (str, bool)):
                wandb.run.summary[k] = v
    except Exception:                                        # noqa: BLE001
        pass


def save_file(path: str) -> None:
    """Attach an artifact (the json log, a checkpoint manifest)."""
    if not _enabled or not os.path.exists(path):
        return
    try:
        import wandb
        wandb.save(path, policy="now")
    except Exception:                                        # noqa: BLE001
        pass


def finish() -> None:
    global _run, _enabled
    if not _enabled:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:                                        # noqa: BLE001
        pass
    finally:
        _run, _enabled = None, False


def enabled() -> bool:
    return _enabled
