"""Rest-to-rest C3 trajectories with explicit velocity/acceleration/jerk bounds."""
import numpy as np


def smooth_trajectory(start, end, dt, *, velocity, acceleration, jerk, seconds=0.):
    """Return targets excluding start, including end; all limits are per channel.

    Seventh-order smoothstep has zero first THREE derivatives at both ends.
    Duration uses analytic extrema, then rounds UP to a control tick. Adjacent
    segments and holds can therefore join without velocity/acceleration jumps.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    limits = [np.broadcast_to(np.asarray(x, float), start.shape) for x in (velocity, acceleration, jerk)]
    if (start.shape != end.shape or dt <= 0 or seconds < 0
            or not np.isfinite([dt, seconds]).all()
            or not all(np.isfinite(x).all() for x in (start, end, *limits))
            or any(np.any(x <= 0) for x in limits)):
        raise ValueError('Finite endpoints, positive dt/limits and nonnegative duration required')
    distance = np.abs(end-start)
    duration = max(seconds, float(np.max(2.1875*distance/limits[0])),
                   float(np.sqrt(np.max(7.514*distance/limits[1]))),
                   float(np.cbrt(np.max(52.5*distance/limits[2]))))
    steps = max(1, int(np.ceil(duration/dt)))
    t = np.arange(1, steps+1, dtype=float)/steps
    blend = t**4*(35+t*(-84+t*(70-20*t)))
    result = start+blend[:, None]*(end-start)
    result[-1] = end
    return result
