"""Capture must establish preload as well as remove excess compression."""
import copy

import pytest

from piper_sim.experts.gripper_preload import relax_preload, validate_profile


PROFILE = {'kinds': ['chopstick'], 'target_tracking_error_m': .0003,
           'min_tracking_error_m': .0002, 'max_tracking_error_m': .0004, 'max_iterations': 6}


def test_weak_bilateral_capture_closes_with_bounded_position_targets():
    target = .00599
    commands, samples = [], []

    def width():
        return .006-(.006-target)*.2

    def move(value):
        nonlocal target
        assert 0 < value < target  # weak initial contact needs additional closure
        target = value
        commands.append(value)

    relax_preload(profile=validate_profile(PROFILE), read_width=width, read_target=lambda: target,
                  has_bilateral_contact=lambda: True, move=move, report=samples.append)
    assert commands
    assert .0002 <= width()-target <= .0004
    assert len(commands) <= PROFILE['max_iterations']


def test_preload_window_accepts_established_capture_without_extra_motion():
    relax_preload(profile=validate_profile(PROFILE), read_width=lambda: .006,
                  read_target=lambda: .0057, has_bilateral_contact=lambda: True,
                  move=lambda _: pytest.fail('Already inside window'), report=lambda _: None)


def test_weak_capture_with_no_contact_cannot_be_promoted_by_window():
    with pytest.raises(RuntimeError, match='Bilateral contact'):
        relax_preload(profile=validate_profile(PROFILE), read_width=lambda: .006,
                      read_target=lambda: .00599, has_bilateral_contact=lambda: False,
                      move=lambda _: pytest.fail('Stop on missing contact'), report=lambda _: None)


@pytest.mark.parametrize('value', [-1., True, float('nan'), .0003, .0005])
def test_invalid_minimum(value):
    profile = copy.deepcopy(PROFILE);profile['min_tracking_error_m'] = value
    with pytest.raises(ValueError): validate_profile(profile)
