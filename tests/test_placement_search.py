from types import SimpleNamespace

import numpy as np
import pytest

from piper_sim.experts.placement_search import VerticalSupport, validate_profile, support_surface, diverse_candidates


def test_support_height_handles_tilted_faces_overlap_and_holes():
    triangles = np.array([[[0, 0, 0], [1, 0, 1], [0, 1, 0]],
                          [[0, 0, 2], [.2, 0, 2], [0, .2, 2]],
                          [[.5, .5, 0], [.5, .5, 1], [.5, 1, 0]]], float)
    surface = VerticalSupport(triangles)
    result = surface.height([[.05, .05], [.3, .2], [2., 2.]])
    np.testing.assert_allclose(result[:2], [2., .3])
    assert result[2] == -np.inf
    # Reversing winding cannot change vertical intersections.
    np.testing.assert_array_equal(result, VerticalSupport(triangles[:, ::-1]).height([[.05, .05], [.3, .2], [2., 2.]]))


def test_support_includes_current_occupants_but_excludes_payload_and_visual_mesh():
    part = {'points': [[0, 0, 0], [1, 0, 0], [0, 1, 0]], 'faces': [[0, 1, 2]], 'collision': True}
    task = SimpleNamespace(active_names=['basket', 'occupant', 'held'], parts={n:[part] for n in ('basket', 'occupant', 'held')})
    task.parts['basket'] = [part, {**part, 'points': [[0, 0, 100], [1, 0, 100], [0, 1, 100]], 'collision':False}]
    positions = {'basket': 0., 'occupant': .05, 'held': .20}
    def body(n): return SimpleNamespace(xmat=np.eye(3).ravel(), xpos=np.array([0., 0., positions[n.removeprefix('task/')]]))
    teacher = SimpleNamespace(b=SimpleNamespace(task=task, data=SimpleNamespace(body=body)))
    surface, obstacles = support_surface(teacher, 'basket', ['held'])
    assert obstacles == ['basket', 'occupant']
    assert surface.height([[.1, .1]])[0] == pytest.approx(.05)
    positions['occupant'] = .08
    changed, _ = support_surface(teacher, 'basket', ['held'])
    assert changed.height([[.1, .1]])[0] == pytest.approx(.08)


def test_profile_rejects_unknown_and_unbounded_search():
    valid = {'kinds': ['chopstick', 'side_dish'], 'xy_radius_m': .05, 'xy_step_m': .025,
             'wall_margin_m': .012, 'support_gap_m': .01,
             'extra_release_heights_m': [0., .015, .03, .06], 'max_candidates': 216}
    assert validate_profile(valid) == valid
    assert validate_profile(None) is None
    for key, bad in [('kinds', ['plate']), ('xy_step_m', .0001), ('support_gap_m', -1.),
                     ('wall_margin_m', float('nan')), ('max_candidates', 1000),
                     ('extra_release_heights_m', [.01]), ('extra_release_heights_m', [0., 0.])]:
        with pytest.raises(ValueError): validate_profile({**valid, key: bad})
    with pytest.raises(ValueError): validate_profile({**valid, 'typo': 1})


def test_candidate_budget_retains_yaw_and_slot_diversity():
    rows = []
    for yaw in (0, 15, -15):
        for xy in ([0., 0.], [-.05, 0.], [0., .05]):
            rows.append((-xy[1], np.array(xy), .02, np.r_[xy, .1], np.eye(3), np.zeros((3, 3)), yaw))
    selected = diverse_candidates(rows, np.zeros(2), np.array([-.4, 0., .2]), 6)
    assert [c[6] for c in selected] == [0, 15, -15, 0, 15, -15]
    for c in selected[:3]: np.testing.assert_array_equal(c[1], [0., 0.])
    for c in selected[3:]: np.testing.assert_array_equal(c[1], [-.05, 0.])
    assert len(diverse_candidates(rows, np.zeros(2), np.array([-.4, 0., .2]), 20)) == 9


def test_supported_choice_rechecks_both_motion_segments_without_mutating_state():
    pytest.importorskip('mujoco')
    from piper_sim.experts.placement_search import search
    from piper_sim.experts.clean_table import SortFailure
    points = np.array([[x, y, z] for x in (-.02, .02) for y in (-.02, .02) for z in (0., .04)])
    floor = {'points': [[-.18, -.155, 0], [.18, -.155, 0], [.18, .155, 0], [-.18, .155, 0]],
             'faces': [[0, 1, 2], [0, 2, 3]], 'collision': True}
    task = SimpleNamespace(active_names=['basket', 'item'], parts={'basket': [floor]},
        specs={'basket': {'size_m': [.36, .31, .14], 'wall_m': .003}, 'item': {'kind': 'side_dish'}},
        config={'drop_slots_m': {'item': [0., 0.]}}, surface_points={'item': points})
    bodies = {name: SimpleNamespace(xmat=np.eye(3).ravel(), xpos=np.array([0., 0., z]))
              for name, z in [('task/basket', 0.), ('task/item', .2)]}
    before = {k: v.xpos.copy() for k, v in bodies.items()}
    teacher = SimpleNamespace(b=SimpleNamespace(task=task, data=SimpleNamespace(body=bodies.__getitem__, qpos=np.zeros(8)),
        indices={'right': {'qpos': np.arange(8)}}, sim_time=1.),
        q=np.zeros(14), events=[], tcp=lambda side:(np.array([0., 0., .2]), np.eye(3)),
        carried_objects=lambda: ['item'], solve=lambda side,p,r,*args:np.r_[p, np.zeros(3)])
    checks = []
    teacher.check_joint_path = lambda side,a,b,**kw: checks.append((a.copy(), b.copy(), kw))
    profile = dict(kinds=['side_dish'], xy_radius_m=.025, xy_step_m=.025, wall_margin_m=.01,
                   support_gap_m=.01, extra_release_heights_m=[0.], max_candidates=9)
    _, high, low, _, _, _ = search(teacher, 'right', 'item', 'basket', profile)
    assert high[2] == pytest.approx(.175) and low[2] == pytest.approx(.01)
    assert len(checks) > 2
    assert checks[2][2]['gripper'][0] == 0.
    assert checks[-1][2]['gripper'][1] == .07
    for key, pos in before.items(): np.testing.assert_array_equal(bodies[key].xpos, pos)
    np.testing.assert_array_equal(teacher.q, np.zeros(14))
    def reject(*args): raise SortFailure('payload obstruction')
    teacher.check_joint_path = reject
    with pytest.raises(SortFailure, match='No supported container placement'):
        search(teacher, 'right', 'item', 'basket', profile)
    assert len(teacher.events) == 1
    def reject_opening(*args, **kwargs):
        if 'gripper' in kwargs: raise SortFailure('opening hits basket')
    teacher.check_joint_path = reject_opening
    with pytest.raises(SortFailure, match='opening hits basket'):
        search(teacher, 'right', 'item', 'basket', profile)
    assert len(teacher.events) == 1
