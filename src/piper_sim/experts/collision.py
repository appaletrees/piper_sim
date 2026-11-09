"""Robot/scene and self-contact audit on the compiled MuJoCo collision model.

Adjacent bearing exclusions belong to the model. Nonadjacent same-arm contacts
must not be discarded here. This does not certify the underlying convex hulls.
"""
import weakref

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
