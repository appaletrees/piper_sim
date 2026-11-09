"""Robot/scene and self-contact audit on the compiled MuJoCo collision model.

Adjacent bearing exclusions belong to the model. Nonadjacent same-arm contacts
must not be discarded here. This does not certify the underlying convex hulls.
"""
import weakref

import mujoco
import numpy as np


_MODEL_INDICES = weakref.WeakKeyDictionary()


def _index(model):
    """Cache immutable topology only, without retaining the simulation model."""
    index = _MODEL_INDICES.get(model)
    if index is None:
        names = tuple(model.body(i).name for i in range(model.nbody))
        ids = model.geom_bodyid.copy()
        robot = np.array([name.startswith(('left/', 'right/')) for name in names])
        fingers = np.array([name in ('left/link7', 'left/link8', 'right/link7', 'right/link8')
                            for name in names])
        index = names, ids, robot[ids], fingers[ids]
        _MODEL_INDICES[model] = index
    return index


def _deepest(data, selected, names, bodies):
    indices = np.flatnonzero(selected)
    if not len(indices):
        return 0., None
    i = indices[np.argmin(data.contact.dist[indices])]
    return float(-data.contact.dist[i]), [names[int(bodies[g])] for g in data.contact.geom[i]]


def robot_penetration(model, data, target=None):
    names, bodies, robot, fingers = _index(model)
    geoms = data.contact.geom
    selected = (data.contact.dist < 0) & np.any(robot[geoms], axis=1)
    if target and 'task/'+target in names:
        target_id = names.index('task/'+target)
        intended = np.any(bodies[geoms] == target_id, axis=1) & np.any(fingers[geoms], axis=1)
        selected &= ~intended
    return _deepest(data, selected, names, bodies)


def carried_obstacle_penetration(model, data, carried):
    """Payload versus scene contacts; internal cup-stack and gripper contacts excluded."""
    names, bodies, robot, _ = _index(model)
    carried_names = {'task/'+n for n in carried}
    carried_ids = [i for i, name in enumerate(names) if name in carried_names]
    geoms = data.contact.geom
    payload = np.isin(bodies[geoms], carried_ids)
    selected = ((data.contact.dist < 0) & (payload[:, 0] != payload[:, 1])
                & ~np.any(robot[geoms], axis=1))
    return _deepest(data, selected, names, bodies)


def carried_obstacle_clearance(model, data, carried, distance):
    """Nearest signed payload/scene distance, capped at the requested margin.

    Tests near misses as well as existing contacts. Internal payload pairs,
    robot and non-colliding visual geoms are excluded. Topology is cached but
    current collision masks and world positions are always read afresh.
    """
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('Clearance search distance must be positive and finite')
    names, bodies, robot, _ = _index(model)
    carried_names = {'task/'+n for n in carried}
    carried_ids = [i for i, name in enumerate(names) if name in carried_names]
    payload = np.isin(bodies, carried_ids)
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    nearest, pair = distance, None
    for a in np.flatnonzero(payload & active):
        compatible = ((model.geom_contype[a] & model.geom_conaffinity) |
                      (model.geom_conaffinity[a] & model.geom_contype)) != 0
        lower_bound = np.linalg.norm(data.geom_xpos-data.geom_xpos[a], axis=1)-model.geom_rbound-model.geom_rbound[a]
        close = (lower_bound < nearest) | (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        for b in np.flatnonzero(~payload & ~robot & active & compatible & close):
            value = mujoco.mj_geomDistance(model, data, int(a), int(b), nearest, None)
            if value < nearest:
                nearest, pair = value, [names[int(bodies[a])], names[int(bodies[b])]]
    return float(nearest), pair


def robot_obstacle_clearance(model, data, side, obstacles, distance):
    """Signed distance from one arm to selected task obstacles, capped at distance.

    This supplements contact rejection with a positive transit margin. It does
    not replace self/inter-arm checks or treat intentional base mounts as free
    space: only explicitly selected task bodies participate in this query.
    """
    if side not in ('left', 'right') or not np.isfinite(distance) or distance <= 0:
        raise ValueError('Expected an arm side and a positive finite distance')
    names, bodies, _, _ = _index(model)
    arm = np.array([names[int(body)].startswith(side+'/') for body in bodies])
    selected = np.isin(bodies, [i for i, name in enumerate(names) if name in {'task/'+n for n in obstacles}])
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    nearest, pair = distance, None
    for a in np.flatnonzero(arm & active):
        compatible = ((model.geom_contype[a] & model.geom_conaffinity) |
                      (model.geom_conaffinity[a] & model.geom_contype)) != 0
        lower_bound = np.linalg.norm(data.geom_xpos-data.geom_xpos[a], axis=1)-model.geom_rbound-model.geom_rbound[a]
        close = (lower_bound < nearest) | (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        for b in np.flatnonzero(selected & active & compatible & close):
            value = mujoco.mj_geomDistance(model, data, int(a), int(b), max(nearest, 0.), None)
            if value < nearest:
                nearest, pair = value, [names[int(bodies[a])], names[int(bodies[b])]]
    return float(nearest), pair
