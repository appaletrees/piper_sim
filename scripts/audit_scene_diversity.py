#!/usr/bin/env python3
"""Measure reset-time variation and split overlap without running a controller.

Sampled layouts are proposals, not qualified demonstrations. Shared meshes and
task language mean a seed split is not an unseen-object or unseen-task benchmark.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from piper_sim.config import ROOT, load
from piper_sim.tasks.clean_table import CleanTableTask
from piper_sim.randomization import stream_seed


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def distribution(values):
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                    np.percentile(values, [0, 5, 50, 95, 100]).tolist())) if values else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--nominal-fraction', type=float, default=0.)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.samples < 2 or not 0 <= args.nominal_fraction <= 1 or not 0 <= args.validation_fraction < 1:
        parser.error('Invalid sample count or fractions')
    module = importlib.util.spec_from_file_location('collector', ROOT/'scripts/collect_clean_table.py')
    collector = importlib.util.module_from_spec(module)
    module.loader.exec_module(collector)
    cfg = load('configs/clean_table_workcell.yaml')
    task_cfg = yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    task_cfg['domain_randomization'] = 'configs/domain_randomization/clean_table.yaml'
    task = CleanTableTask(task_cfg, cfg)
    rows, failures, scalars = [], [], {}
    layouts = {}
    def value(name, x):
        scalars.setdefault(name, []).append(float(x))
    for seed in range(args.samples):
        nominal = stream_seed(seed, 'collection/nominal')/2**64 < args.nominal_fraction
        split = collector.collection_split(seed, nominal, args.validation_fraction)
        try:
            layout = task.reset(seed, not nominal)
        except ValueError as error:
            failures.append({'seed': seed, 'nominal': nominal, 'error': str(error)})
            continue
        poses = {name: layout['poses'][name] for name in task.active_names}
        layout_hash = digest(poses)
        layouts.setdefault(layout_hash, set()).add(split)
        params = task.domain_parameters or {}
        content = layout['content']
        row = {'seed': seed, 'split': split, 'nominal': nominal,
            'layout_sha256': layout_hash, 'cup_count': content['cup_count'],
            'active_cup_colors': {n: c for n, c in content['cup_colors'].items() if n in task.active_names},
            'tabletop': params.get('tabletop_material', 'nominal'),
            'domain_without_seed_sha256': digest({k: v for k, v in params.items() if k != 'seed'})}
        rows.append(row)
        for name, pose in poses.items():
            for axis, x in zip('xy', pose['position']):
                value('layout/'+name+'/'+axis+'_m', x)
        for name, p in params.get('objects', {}).items():
            if name in task.active_names:
                for key, x in p.items(): value('physics/'+name+'/'+key, x)
        for name, p in params.get('cameras', {}).items():
            for key, x in p.items():
                if np.isscalar(x): value('sensor/'+name+'/'+key, x)
                else:
                    for i, v in enumerate(x): value('sensor/'+name+'/'+key+'/'+str(i), v)
        if params: value('appearance/light_scale', params['light_scale'])
    repeated = Counter(row['layout_sha256'] for row in rows)
    colors = Counter(str(color) for row in rows for color in row['active_cup_colors'].values())
    report = {'status': 'reset_sampler_diagnostic', 'requested': args.samples,
        'generated': len(rows), 'failures': failures, 'nominal_fraction': args.nominal_fraction,
        'validation_fraction': args.validation_fraction,
        'split_counts': dict(Counter(row['split'] for row in rows)),
        'cup_counts': dict(Counter(row['cup_count'] for row in rows)),
        'active_cup_color_counts': dict(colors), 'tabletop_counts': dict(Counter(row['tabletop'] for row in rows)),
        'unique_layouts': len(layouts), 'duplicate_layout_groups': {k: v for k, v in repeated.items() if v > 1},
        'exact_layout_cross_split_groups': [key for key, splits in layouts.items() if len(splits) > 1],
        'distributions': {k: distribution(v) for k, v in scalars.items()}, 'rows': rows,
        'limitations': [
            'No physics, IK, grasp success, accepted RGB yield or sim-to-real quality is measured here.',
            'Layout fingerprints exclude seed and inactive objects; domain fingerprints exclude seed.',
            'Exact split overlap only; near-duplicate trajectories or visuals require recorded-episode comparisons.',
            'Identical meshes, language template and randomization ranges are shared across splits.',
            'Seed holdout tests in-distribution layout variation, not unseen object/task/environment generalization.',
            'Camera calibration, robot dimensions, gravity and control rate are intentionally fixed.',
            'Sampling varies rigid properties, colors, layout, lighting and sensor effects; textures/models remain a finite asset library.',
        ], 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (ROOT/'configs/clean_table_workcell.yaml', ROOT/'configs/tasks/clean_table_rigid.yaml',
             ROOT/'configs/domain_randomization/clean_table.yaml', ROOT/'src/piper_sim/tasks/clean_table.py',
             ROOT/'src/piper_sim/randomization/__init__.py', ROOT/'scripts/collect_clean_table.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'generated', 'split_counts', 'cup_counts',
                                          'tabletop_counts', 'unique_layouts', 'exact_layout_cross_split_groups')}))


if __name__ == '__main__':
    main()
