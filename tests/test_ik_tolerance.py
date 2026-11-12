"""Declared planning slack must not degrade already accurate solutions."""
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip('mujoco')
from piper_sim.config import ROOT
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure


def teacher(errors, tolerance=None):
    obj = object.__new__(CleanTableTeacher)
    obj.q = np.zeros(14)
    obj.b = SimpleNamespace(sim_time=12., task=SimpleNamespace(
        config={'reference': {'joint14_rad_fraction': [0.]*14}}))
    obj.current = 'cup_01'; obj.events = []
    obj.ik_tolerance = tolerance or {'position_m': .006, 'orientation_rad': .07}
    calls = []
    def solve(*args):
        i = len(calls); calls.append(args)
        ep, er = errors[min(i, len(errors)-1)]
        return np.full(6, i), ep, er
    obj.ik = {'left': SimpleNamespace(solve=solve)}
    return obj, calls


def test_tries_all_original_seeds_before_relaxed_fallback():
    obj, calls = teacher([(.005, .025), (.002, .02)])
    np.testing.assert_array_equal(obj.solve('left', [.3, .3, .1], np.eye(3)), np.ones(6))
    assert len(calls) == 2 and obj.events == []


def test_best_feasible_fallback_is_logged_and_real_failures_still_reject():
    # Lowest weighted cost is outside the angular budget; choose a feasible one.
    obj, calls = teacher([(.0001, .071), (.005, .03), (.0055, .04)])
    np.testing.assert_array_equal(obj.solve('left', [.3, .3, .1], np.eye(3)), np.ones(6))
    assert len(calls) == 5
    assert obj.events[-1]['position_error_m'] == .005
    assert obj.events[-1]['orientation_error_rad'] == .03
    obj, _ = teacher([(.007, .02)])
    with pytest.raises(SortFailure, match='7.0 mm'):
        obj.solve('left', [.3, .3, .1], np.eye(3))


def test_default_and_stricter_tolerances_are_respected():
    for tolerance in ({'position_m': .003, 'orientation_rad': .035},
                      {'position_m': .001, 'orientation_rad': .01}):
        obj, _ = teacher([(.002, .02)], tolerance)
        if tolerance['position_m'] == .003:
            obj.solve('left', [.3, .3, .1], np.eye(3))
        else:
            with pytest.raises(SortFailure):obj.solve('left', [.3, .3, .1], np.eye(3))


def test_practical_revision_changes_duration_gate_only_and_declares_planning_slack():
    p = ROOT/'configs/collection'
    original = yaml.safe_load((p/'quality_policy_tcp.yaml').read_text())
    revision = yaml.safe_load((p/'quality_policy_practical_v2.yaml').read_text())
    assert revision['dataset']['max_duration_s'] == 420.
    revision['dataset']['max_duration_s'] = original['dataset']['max_duration_s']
    assert original == revision
    profile = yaml.safe_load((p/'clean_table_practical_v2.yaml').read_text())
    assert profile['pacing']['ik_tolerance']['position_m'] == .01
    assert np.rad2deg(profile['pacing']['ik_tolerance']['orientation_rad']) == pytest.approx(5.)
    assert profile['dataset'] == {**yaml.safe_load((p/'clean_table_rim_damped.yaml').read_text())['dataset'],
                                  'max_duration_s': 420.}
