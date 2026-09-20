#!/usr/bin/env python3
"""Tabulate benchmark results straight out of a BSS workspace.

`report.py` builds the full HTML report; this prints the one table the
history-length sweep is actually about — metric vs. history length, baseline
next to self-distilled — plus a point-cloud table when those metrics exist.

Method names are read as `{tag}_h{history}` (e.g. `base_h64`, `sd_a1_h128`);
anything not matching that shape is grouped under history `-`.

Usage:
    python setup_local/summarize.py /path/to/bench_ws/oxford [...]
    python setup_local/summarize.py /path/to/bench_ws/*        # all at once
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

METHOD_RE = re.compile(r'^(?P<tag>.+)_h(?P<hist>\d+)$')

TABLES = [
    # (metrics json, columns, lower-is-better flags)
    ('traj.json',   [('ate', 'ATE'), ('rpe_trans', 'RPE-t'), ('rpe_rot', 'RPE-r°')], True),
    ('points.json', [('chamfer', 'Chamfer'), ('accuracy', 'Acc'),
                     ('completeness', 'Comp'), ('f1', 'F1')], None),
]


def collect(workspace: Path):
    """workspace/{dataset}/{scene}/{method}/eval/*.json -> {(ds, metric_file, tag, hist): {metric: [values]}}"""
    out = defaultdict(lambda: defaultdict(list))
    scenes = defaultdict(set)
    for eval_dir in workspace.glob('*/*/*/eval'):
        method_dir = eval_dir.parent
        scene_dir = method_dir.parent
        dataset = scene_dir.parent.name
        if 'report' in eval_dir.parts[len(workspace.parts):]:
            continue                                    # skip report/artifacts copies
        m = METHOD_RE.match(method_dir.name)
        tag, hist = (m['tag'], int(m['hist'])) if m else (method_dir.name, None)
        for fname, cols, _ in TABLES:
            path = eval_dir / fname
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            key = (dataset, fname, tag, hist)
            for metric, _label in cols:
                if metric in data:
                    out[key][metric].append(float(data[metric]))
            scenes[key].add(scene_dir.name)
    return out, scenes


def render(out, scenes):
    for fname, cols, _ in TABLES:
        keys = sorted((k for k in out if k[1] == fname),
                      key=lambda k: (k[0], k[3] if k[3] is not None else -1, k[2]))
        if not keys:
            continue
        datasets = sorted({k[0] for k in keys})
        for dataset in datasets:
            dk = [k for k in keys if k[0] == dataset]
            tags = sorted({k[2] for k in dk})
            hists = sorted({k[3] for k in dk}, key=lambda h: (h is None, h))

            header = ['history'] + [f'{t} {lbl}' for t in tags for _m, lbl in cols] + ['#scenes']
            print(f'\n## {dataset} — {fname.replace(".json", "")}')
            print('| ' + ' | '.join(header) + ' |')
            print('|' + '|'.join(['---'] * len(header)) + '|')

            for h in hists:
                row = [str(h) if h is not None else '-']
                n = 0
                for t in tags:
                    vals = out.get((dataset, fname, t, h), {})
                    n = max(n, len(scenes.get((dataset, fname, t, h), ())))
                    for metric, _lbl in cols:
                        v = vals.get(metric)
                        row.append(f'{statistics.fmean(v):.3f}' if v else '·')
                row.append(str(n))
                print('| ' + ' | '.join(row) + ' |')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workspaces', nargs='+', type=Path,
                    help='BSS workspace dirs, e.g. bench_ws/oxford')
    args = ap.parse_args()

    for ws in args.workspaces:
        if not ws.is_dir():
            print(f'skip {ws} (not a directory)')
            continue
        out, scenes = collect(ws)
        if not out:
            print(f'\n# {ws.name}: no eval/*.json yet')
            continue
        print(f'\n# {ws.name}   ({ws})')
        render(out, scenes)
    print()


if __name__ == '__main__':
    main()
