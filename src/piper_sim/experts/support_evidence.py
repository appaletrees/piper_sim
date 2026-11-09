"""Read-only, world-frame payload contact forces from the current solver buffers.

Call after forward dynamics, as with other physics evidence. A force observation
does not establish stable placement or identify real hardware parameters.
"""
import mujoco
import numpy as np


def payload_support(model, data, target):
    body = model.body('task/'+target).id
    gravity = np.asarray(model.opt.gravity)
    magnitude = float(np.linalg.norm(gravity))
    if magnitude <= 0:
        raise ValueError('Support evidence requires nonzero gravity')
    up = -gravity/magnitude
    rows = []
    force_sum = np.zeros(3)
    for index, contact in enumerate(data.contact):
        if contact.efc_address < 0 or min(contact.geom) < 0:
            continue
        bodies = model.geom_bodyid[contact.geom]
        if body not in bodies or bodies[0] == bodies[1]:
            continue
        k = 0 if bodies[0] == body else 1
        other = model.body(int(bodies[1-k])).name
        if other.startswith(('left/', 'right/')):
            continue
        local = np.empty(6)
        mujoco.mj_contactForce(model, data, index, local)
        sign = 1 if k == 1 else -1
        frame = contact.frame.reshape(3, 3)
        force = sign*frame.T@local[:3]
        normal = sign*frame[0]
        supports = float(normal@up) >= .5
        if supports:
            force_sum += force
        rows.append({'other_body': other, 'geoms': [model.geom(int(g)).name for g in contact.geom],
                     'position_world_m': contact.pos.tolist(), 'distance_m': float(contact.dist),
                     'force_on_payload_world_N': force.tolist(), 'normal_up_component': float(normal@up),
                     'upward_facing_support': supports})
    weight = float(model.body_mass[body]*magnitude)
    if weight <= 0:
        raise ValueError('Support evidence requires positive payload mass')
    return {'sim_time': float(data.time), 'payload': target, 'mass_kg': float(model.body_mass[body]),
            'gravity_m_s2': magnitude, 'up_axis_world': up.tolist(), 'weight_N': weight,
            'support_force_world_N': force_sum.tolist(), 'support_fraction_of_weight': float(force_sum@up/weight),
            'contacts': rows,
            'scope': 'Solver-buffer contact forces from non-robot bodies with inward payload normal at least 0.5 along anti-gravity; no static-equilibrium claim.'}
