import importlib.util
from pathlib import Path

import numpy as np
import pytest

from piper_sim.tasks.clean_table_meshes import box, build_parts

spec = importlib.util.spec_from_file_location('release_support',
    Path(__file__).resolve().parents[1]/'scripts/audit_release_support.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_support_distinguishes_floor_from_existing_cargo_and_excludes_self():
    specs = {'basket': {'kind': 'basket', 'size_m': [.1, .1, .1], 'wall_m': .003},
             'item': {'target': 'basket'}, 'cargo': {}}
    parts = {'basket': build_parts(specs['basket']),
             'item': [box('item', [.01, .01, .01], [0, 0, .005], 'test', collision=True)],
             'cargo': [box('cargo', [.02, .02, .01], [0, 0, .005], 'test', collision=True)]}
    poses = {n: {'position': [0., 0., z], 'quaternion_wxyz': [1., 0., 0., 0.]}
             for n, z in (('basket', .2), ('item', .233), ('cargo', .203))}
    result = module.geometry('item', poses, specs, parts, list(specs))
    assert result['basket_local_min_height_above_floor_m'] == pytest.approx(.03)
    assert result['basket_local_xy_margin_m'] == pytest.approx(.042)
    assert [s['object'] for s in result['support_envelopes']] == ['cargo', 'basket']
    assert result['support_envelopes'][0]['sampled_min_vertical_gap_m'] == pytest.approx(.02)
    # A basket tilt changes vertical clearance, but not its local floor distance.
    rotation = module.pose_matrix([0, 0, 0], [np.cos(.2), np.sin(.2), 0, 0])
    for pose in poses.values():
        pose['position'] = (rotation[:3, :3]@pose['position']).tolist()
        pose['quaternion_wxyz'] = [np.cos(.2), np.sin(.2), 0, 0]
    tilted = module.geometry('item', poses, specs, parts, list(specs))
    assert tilted['basket_local_min_height_above_floor_m'] == pytest.approx(.03)
