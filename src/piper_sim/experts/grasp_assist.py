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
        for name in ('paper', 'towel', 'spoon', 'bowl', 'plate', 'side_dish', 'chopstick_a', 'chopstick_b'):
            ET.SubElement(equality, 'weld', name=f'demo_grasp/{side}/{name}',
                          body1=side+'/link6', body2='task/'+name, active='false',
                          solref='.006 1', solimp='.999 .9999 .001', torquescale='.1')
    return ET.tostring(root, encoding='unicode')


def finger_contacts(backend, side, name):
    """Names of fingers actually in contact, with positive normal force."""
    model, data = backend.model, backend.data
    obj = model.body('task/'+name).id
    fingers = {model.body(side+'/link'+str(i)).id: str(i) for i in (7, 8)}
    found = set()
    for index, contact in enumerate(data.contact):
        bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
        if obj not in bodies:
            continue
        finger = next((fingers[b] for b in bodies if b in fingers), None)
        if finger is not None:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            if force[0] > .005:
                found.add(finger)
    return sorted(found)


class ContactGraspAssist:
    def __init__(self, backend):
        self.backend = backend
        self.active = None

    def engage(self, side, name):
        if self.active is not None:
            raise RuntimeError('Release the current grasp before engaging another')
        b = self.backend
        contacts = finger_contacts(b, side, name)
        if not contacts:
            raise RuntimeError(f'No finger contact with {name}; assistance refused')
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
