import copy

import pytest

from piper_sim.experts.gripper_preload import relax_preload, validate_profile


PROFILE = {'kinds': ['chopstick', 'pen'], 'target_tracking_error_m': .0003,
           'max_tracking_error_m': .0004, 'max_iterations': 6}


def test_relaxation_uses_positions_and_stops_at_budget():
    # A compliant object opens back toward its unloaded 6 mm cross-section.
    target = 0.
    commands, samples = [], []

    def width():
        return .006-(.006-target)*.2

    def move(value):
        nonlocal target
        assert target < value < width()
        target = value
        commands.append(value)

    relax_preload(profile=PROFILE, read_width=width, read_target=lambda: target,
                  has_bilateral_contact=lambda: True, move=move, report=samples.append)
    assert width()-target <= PROFILE['max_tracking_error_m']
    assert 1 < len(commands) <= PROFILE['max_iterations']
    assert len(samples) == len(commands)+1
    assert all(s['tracking_error_m'] > .0 for s in samples)


def test_no_movement_when_already_within_budget():
    relax_preload(profile=PROFILE, read_width=lambda: .006, read_target=lambda: .0057,
                  has_bilateral_contact=lambda: True,
                  move=lambda _: pytest.fail('Unneeded motion'), report=lambda _: None)


def test_loss_of_contact_is_not_treated_as_relaxation_success():
    with pytest.raises(RuntimeError, match='Bilateral contact'):
        relax_preload(profile=PROFILE, read_width=lambda: .005, read_target=lambda: .005,
                      has_bilateral_contact=lambda: False,
                      move=lambda _: pytest.fail('Must stop'), report=lambda _: None)


def test_nonconvergence_is_bounded_and_explicit():
    target = 0.
    commands = []

    def move(value):
        nonlocal target
        target = value
        commands.append(value)

    with pytest.raises(RuntimeError, match='did not converge'):
        relax_preload(profile=PROFILE, read_width=lambda: target+.001,
                      read_target=lambda: target, has_bilateral_contact=lambda: True,
                      move=move, report=lambda _: None)
    assert len(commands) == PROFILE['max_iterations']


@pytest.mark.parametrize('field,value', [
    ('kinds', []), ('kinds', ['pen', 'pen']), ('max_iterations', True),
    ('target_tracking_error_m', float('nan')), ('target_tracking_error_m', -.001),
    ('max_tracking_error_m', .0002), ('max_iterations', 21)])
def test_invalid_budget_rejected(field, value):
    profile = copy.deepcopy(PROFILE);profile[field] = value
    with pytest.raises(ValueError): validate_profile(profile)
