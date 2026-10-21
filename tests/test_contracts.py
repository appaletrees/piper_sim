import numpy as np
import pytest
from piper_sim.contracts import Command, LOWER, UPPER, vector
from piper_sim.safety import CommandGuard

HOME = np.array([0, .8, -.6, 0, 0, 0, .04]*2)


def test_invalid_shapes_and_nan():
    for value in ([0]*13, [float("nan")]*14, [float("inf")]*14):
        with pytest.raises(ValueError):
            vector(value)


def test_alignment_freshness_sequence_and_slew():
    guard = CommandGuard()
    target = HOME.copy()
    target[0] += .1
    accepted = guard.accept(Command(target, 10, 1), HOME, 10)
    assert accepted.position[0] == pytest.approx(.005)
    for command in (Command(target, 9, 2), Command(target, 11, 2),
                    Command(target, 10, 1)):
        with pytest.raises(ValueError):
            guard.accept(command, HOME, 10.01)
    outside = UPPER.copy()
    outside[0] += .1
    with pytest.raises(ValueError, match="bounds"):
        CommandGuard().accept(Command(outside, 10, 1), HOME, 10)
    far = HOME.copy()
    far[0] = .3
    with pytest.raises(ValueError, match="aligned"):
        CommandGuard().accept(Command(far, 10, 1), HOME, 10)


def test_tracking_error_and_reversed_clock():
    guard = CommandGuard()
    guard.accept(Command(HOME, 10, 1), HOME, 10)
    with pytest.raises(ValueError, match="Clock"):
        guard.accept(Command(HOME, 10, 2), HOME, 10)
    measured = HOME.copy()
    measured[0] += .5
    with pytest.raises(ValueError, match="tracking"):
        guard.accept(Command(HOME, 10.01, 2), measured, 10.01)
