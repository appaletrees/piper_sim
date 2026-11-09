"""Explicit contact-triggered demo assistance, never a hidden physics success.

The optional weld starts at the observed relative pose (no state placement),
and is removed after the finger servo is unloaded during opening. The teacher
then finishes opening without the constraint. Its use is recorded as assisted.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def add_grasp_constraints(xml):
    root = ET.fromstring(xml)
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    for side in ('left', 'right'):
        for body in root.findall('.//body'):
            name = body.get('name','').removeprefix('task/')
            if not body.get('name','').startswith('task/') or name in ('basket','trash_bin','placemat'):
                continue
            ET.SubElement(equality, 'weld', name=f'demo_grasp/{side}/{name}',
                          body1=side+'/link6', body2='task/'+name, active='false',
                          solref='.006 1', solimp='.999 .9999 .001', torquescale='.1')
    return ET.tostring(root, encoding='unicode')


def finger_contacts(backend, side, name):
    """Names of fingers actually in contact, with positive normal force."""
    return [finger for finger, evidence in finger_contact_evidence(backend, side, name).items()
            if evidence['contact_above_threshold']]


def finger_contact_evidence(backend, side, name):
    """Solver normal forces; the existing per-contact 0.005 N gate is unchanged."""
    model, data = backend.model, backend.data
    obj = model.body('task/'+name).id
    fingers = {model.body(side+'/link'+str(i)).id: str(i) for i in (7, 8)}
    bodies = model.geom_bodyid[data.contact.geom]
    selected = np.any(bodies == obj, axis=1) & np.any(np.isin(bodies, list(fingers)), axis=1)
    result = {str(i): {'normal_force_N': 0., 'contact_count': 0, 'contact_above_threshold': False}
              for i in (7, 8)}
    force = np.zeros(6)
    for index in np.flatnonzero(selected):
        finger = next(fingers[b] for b in bodies[index] if b in fingers)
        mujoco.mj_contactForce(model, data, index, force)
        evidence = result[finger]
        evidence['normal_force_N'] += float(max(0., force[0]))
        evidence['contact_count'] += 1
        evidence['contact_above_threshold'] |= bool(force[0] > .005)
    return result


class ContactGraspAssist:
    def __init__(self, backend):
        self.backend = backend
        self.active = None

    def engage(self, side, name):
        if self.active is not None:
            raise RuntimeError('Release the current grasp before engaging another')
        b = self.backend
        contacts = finger_contacts(b, side, name)
        if len(contacts) != 2:
            raise RuntimeError(f'No finger contact on both sides of {name}; assistance refused')
        hand, obj = b.data.body(side+'/link6'), b.data.body('task/'+name)
        rotation = hand.xmat.reshape(3, 3)
        relative_rotation = rotation.T @ obj.xmat.reshape(3, 3)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, relative_rotation.reshape(-1))
        eq = b.model.equality(f'demo_grasp/{side}/{name}').id
        b.model.eq_data[eq, 3:6] = rotation.T @ (obj.xpos-hand.xpos)
        b.model.eq_data[eq, 6:10] = quaternion
        b.data.eq_active[eq] = True
        self.active = (eq, side, name)
        return {'side': side, 'object': name, 'contact_fingers': contacts,
                'simulation_time': float(b.data.time), 'assisted': True}

    def release(self):
        if self.active is not None:
            self.backend.data.eq_active[self.active[0]] = False
            self.active = None
