"""C3 rest-to-rest and retimed C2 joint paths; no command clipping."""
import numpy as np


def _convex_spline(points, parameters, samples):
    """Clamped B-spline: exact endpoints, interior points are a control polygon.

    De Boor evaluation uses convex combinations. The curve stays in the convex
    hull of the input joint vectors; it is NOT an exact Cartesian waypoint path.
    Three or fewer points use a single lower-degree polynomial (no internal joins).
    """
    degree = min(3, len(points)-1)
    internal = [np.mean(parameters[j:j+degree]) for j in range(1, len(points)-degree)]
    knots = np.r_[np.zeros(degree+1), internal, np.ones(degree+1)]
    span = np.clip(np.searchsorted(knots, samples, side='right')-1, degree, len(points)-1)
    values = points[span[:, None]-degree+np.arange(degree+1)].copy()
    for level in range(1, degree+1):
        for j in range(degree, level-1, -1):
            left = knots[span-degree+j]
            alpha = ((samples-left)/(knots[span+1+j-level]-left))[:, None]
            # Difference form preserves locally constant coordinates exactly.
            values[:, j] = values[:, j-1]+alpha*(values[:, j]-values[:, j-1])
    return values[:, degree]


def path_trajectory(waypoints, dt, *, velocity, acceleration, jerk, seconds=0., interpolation='natural', sampling=None):
    """C2 cubic joint path, with one C3 time law for the whole motion.

    Natural interpolation passes through IK waypoints without stopping. Optional
    convex interpolation approximates interior waypoints within their joint-space
    convex hull; callers MUST check the resulting path for geometric clearance.
    Retime the actual
    control-rate samples against all three derivative limits, including joins
    to stationary commands at both ends. No command clipping or frame dropping.
    Optional sampling receives copies of the retained control polygon and its
    parameters, plus the final sample parameters, for parameter-matched FK checks.
    """
    points = np.asarray(waypoints, dtype=float)
    if interpolation not in ('natural', 'convex'):
        raise ValueError('Unknown joint path interpolation')
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('At least two finite vector waypoints required')
    limits = [np.broadcast_to(np.asarray(x, float), points.shape[1:]) for x in (velocity, acceleration, jerk)]
    if (not np.isfinite([dt, seconds]).all() or dt <= 0 or seconds < 0
            or any(not np.isfinite(x).all() or np.any(x <= 0) for x in limits)):
        raise ValueError('Positive finite time step and motion limits required')
    distance = np.linalg.norm(np.diff(points, axis=0)/limits[0], axis=1)
    points = points[np.r_[True, distance > 1e-10]]
    if len(points) == 1:
        if sampling is not None:
            sampling.update(points=points.copy(), knots=np.array([0.]), samples=np.array([0.]))
        return points.copy()
    knots = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0)/limits[0], axis=1))]
    knots /= knots[-1]
    h = np.diff(knots)
    # Natural cubic spline: continuous acceleration at every internal knot.
    if interpolation == 'natural':
        matrix = np.eye(len(points))
        rhs = np.zeros_like(points)
        for i in range(1, len(points)-1):
            matrix[i, i-1:i+2] = [h[i-1], 2*(h[i-1]+h[i]), h[i]]
            rhs[i] = 6*((points[i+1]-points[i])/h[i]-(points[i]-points[i-1])/h[i-1])
        second = np.linalg.solve(matrix, rhs)
    steps = max(4, int(np.ceil(max(seconds, np.max(np.sum(abs(np.diff(points, axis=0)), axis=0)/limits[0]))/dt)))
    for _ in range(40):
        t = np.arange(1, steps+1)/steps
        s = t**4*(35+t*(-84+t*(70-20*t)))
        if interpolation == 'convex':
            path = _convex_spline(points, knots, s)
        else:
            i = np.clip(np.searchsorted(knots, s, side='right')-1, 0, len(h)-1)
            a, b = (knots[i+1]-s)/h[i], (s-knots[i])/h[i]
            path = (a[:, None]*points[i]+b[:, None]*points[i+1]
                    +((a**3-a)[:, None]*second[i]+(b**3-b)[:, None]*second[i+1])*h[i, None]**2/6)
        constant = np.ptp(points, axis=0) == 0
        path[:, constant] = points[0, constant]
        path[-1] = points[-1]
        padded = np.vstack([np.tile(points[0], (4, 1)), path, np.tile(points[-1], (4, 1))])
        scale = max(float(np.max(abs(np.diff(padded, n=order, axis=0))/dt**order/bound))**(1/order)
                    for order, bound in enumerate(limits, 1))
        if scale <= 1+1e-9:
            if sampling is not None:
                sampling.update(points=points.copy(), knots=knots.copy(), samples=s.copy())
            return path
        steps = max(steps+1, int(np.ceil(steps*scale*1.02)))
    raise ValueError('Could not retime the joint path within derivative limits')


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
