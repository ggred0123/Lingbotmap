#!/usr/bin/env python3
"""Recover per-step records the trainer's results json lost across a resume.

``trainer.py`` re-initialises ``log["steps"]`` on --resume and rewrites
``experiments/results/train_<arm>.json`` at the next checkpoint, so every step
before the resume survives only in the text log (every 5th step) and in wandb
(every step).  This pulls every wandb run named ``<arm>`` and writes the union
of their per-step rows, keyed by step, to ``train_<arm>_wandb.json``;
``gtabs_report.py`` merges those rows under the json's own where the json has
a hole.  Keys are stored without the ``train_prequential/`` prefix so they
match the trainer's ``parts`` names.

    python3 experiments/gtabs_wandb_fill.py gtscale
Run from anywhere except the repo root: the repo's ``wandb/`` run directory
shadows the package there (PYTHONPATH must reach the pylibs install).
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYLIBS = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/pylibs"
PROJECT = "ggred0123-korea-university/streaming3d-self-distill"


def main():
    arms = sys.argv[1:] or ["gtscale"]
    os.chdir("/tmp")                                   # keep repo/wandb/ off the path
    sys.path = [p for p in sys.path if not p.endswith("streaming3d-self-distill")]
    sys.path.insert(0, PYLIBS)
    # key: env WANDB_API_KEY, else the gitignored <repo>/.wandb_key (wandb_log.py)
    if not os.environ.get("WANDB_API_KEY") and os.path.isfile(os.path.join(ROOT, ".wandb_key")):
        os.environ["WANDB_API_KEY"] = open(os.path.join(ROOT, ".wandb_key")).read().strip()
    import wandb
    api = wandb.Api(timeout=120)
    for arm in arms:
        by_step = {}
        runs = list(api.runs(PROJECT, filters={"display_name": arm}))
        runs.sort(key=lambda r: r.created_at)
        for r in runs:
            n = 0
            for row in r.scan_history():
                st = row.get("_step")
                if st is None:
                    continue
                rec = {k.split("/", 1)[1]: v for k, v in row.items() if k.startswith("train_prequential/")}
                if not rec:
                    continue
                # the state-mixture coordinates live under stream/; the branch is
                # the one the report needs to split identity from correction
                for k in ("is_identity", "K", "raw_frame_age", "window_start", "branch"):
                    if f"stream/{k}" in row:
                        rec[k] = row[f"stream/{k}"]
                rec["step"] = int(st)
                rec["_wandb_run"] = r.id
                by_step[int(st)] = rec                 # later runs win (a resume re-trains nothing)
                n += 1
            print(f"[wandb] {arm} run {r.id} ({r.state}, created {r.created_at}): {n} step rows")
        out = os.path.join(ROOT, "experiments", "results", f"train_{arm}_wandb.json")
        json.dump([by_step[k] for k in sorted(by_step)], open(out, "w"))
        print(f"[wandb] wrote {out}: {len(by_step)} steps")


if __name__ == "__main__":
    main()
