#!/usr/bin/env python3
"""Independently audit recorded cup poses; never changes the episode or physics."""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.tasks.clean_table import CleanTableTask
from piper_sim.transforms import pose_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    with h5py.File(args.episode) as source:
        metadata = json.loads(source.attrs['metadata'])
        saved = metadata['task_scene']
        cfg = copy.deepcopy(saved['config'])
        cfg['domain_randomization'] = saved.get('domain_profile')
        task = CleanTableTask(cfg, metadata['config'])
        task.reset(saved['layout']['seed'], saved['layout']['randomized'])
        assert list(task.active_names) == saved['layout']['active_objects']
        cups = [n for n in task.active_names if task.specs[n]['kind'] == 'cup']
        times = source['next_observation/sim_time'][:]
        expected = 1/metadata['fps']
        uniform = bool(len(times) > 1 and np.allclose(np.diff(times), expected, atol=1e-7, rtol=0))
        hold, first_table, first_basket = 0., None, None
        for index, raw in enumerate(source['next_observation/extras_json']):
            poses = json.loads(raw)['objects']
            stack = task.stack_metrics(poses)
            hold = hold+expected if stack['nested'] and stack['on_table'] else 0.
            if first_table is None and hold >= cfg['stacking']['settle_seconds']:
                first_table = float(times[index])
            basket = pose_matrix(poses['basket']['position'], poses['basket']['quaternion_wxyz'])
            p = (np.asarray(poses[cups[0]]['position'])-basket[:3, 3])@basket[:3, :3]
            size = np.asarray(task.specs['basket']['size_m'])
            if first_basket is None and np.all(abs(p[:2]) < size[:2]/2) and 0 <= p[2] < size[2]:
                first_basket = float(times[index])
            # Recheck all final objects for the required containment dwell.
            if index >= len(times)-max(60, int(2*metadata['fps'])):
                metrics = task.metrics(poses, expected)
        sequence = first_table is not None and first_basket is not None and first_table < first_basket
        final_cups = all(metrics[n]['success'] for n in cups) and stack['nested']
        full_containment = all(metrics[n]['success'] for n in task.required)
        result = {'scope': 'Recorded-pose audit; not a new rollout or a grasp/contact benchmark',
                  'episode': str(args.episode), 'frames': len(times), 'uniform_clock': uniform,
                  'first_stable_table_stack_s': first_table, 'first_base_cup_in_basket_s': first_basket,
                  'stacked_on_table_before_basket': bool(sequence), 'final_stack': stack,
                  'all_cups_nested_and_contained': bool(final_cups),
                  'all_required_objects_contained': bool(full_containment),
                  'source_episode_accepted': bool(source.attrs.get('accepted', False)),
                  'passed': bool(uniform and sequence and final_cups and full_containment)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
