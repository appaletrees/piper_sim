"""Bounded Piper IK on scratch data; never writes the live simulation state."""
import numpy as np
import mujoco

from ..contracts import LOWER, UPPER


def rotation_error(desired, actual):
    """SO(3) logarithm in world coordinates, including the half-turn case."""
    r = desired @ actual.T
    v = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]) / 2
    s = np.linalg.norm(v)
    c = np.clip((np.trace(r)-1)/2, -1., 1.)
    if s > 1e-7:
        return v*np.arctan2(s, c)/s
    if c > 0:
        return v
    _, axes = np.linalg.eigh((r+r.T)/2)
    return axes[:, -1]*np.pi


def grasp_rotation(tilt_degrees, yaw_degrees, roll_degrees=0):
    """Link6 +Z approaches the object; +Y is the jaw opening direction.

    At zero tilt +Z points down. Positive tilt points it forward along yaw.
    Roll rotates the fingers around the approach axis, in degrees.
    """
    t, y, a = np.deg2rad([tilt_degrees, yaw_degrees, roll_degrees])
    c, s = np.cos(y), np.sin(y)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ry = np.array([[-np.cos(t), 0, np.sin(t)], [0, 1, 0],
                   [-np.sin(t), 0, -np.cos(t)]])
    roll = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return rz @ ry @ roll


class PiperIK:
    def __init__(self, backend, side, tcp_offset_m=(0, 0, .1375)):
        self.backend = backend
        self.model = backend.model
        self.data = mujoco.MjData(self.model)
        self.qids = backend.indices[side]['qpos'][:6]
        self.dofs = backend.indices[side]['dof'][:6]
        self.body_id = self.model.body(side+'/link6').id
        self.offset = np.asarray(tcp_offset_m, dtype=float)

    def forward(self, joints):
        self.data.qpos[:] = self.backend.data.qpos
        self.data.qpos[self.qids] = joints
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        rotation = self.data.xmat[self.body_id].reshape(3, 3).copy()
        return self.data.xpos[self.body_id]+rotation@self.offset, rotation

    def solve(self, position, rotation, seed, *, iterations=200, rotation_weight=.12):
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        q = np.asarray(seed, dtype=float).copy()
        if (position.shape != (3,) or rotation.shape != (3, 3) or q.shape != (6,)
                or not all(np.isfinite(a).all() for a in (position, rotation, q))):
            raise ValueError('IK requires finite position[3], rotation[3,3], seed[6]')
        q = np.clip(q, LOWER[:6]+1e-5, UPPER[:6]-1e-5)
        for _ in range(iterations):
            p, r = self.forward(q)
            ep, er = position-p, rotation_error(rotation, r)
            error = np.r_[ep, er*rotation_weight]
            if np.linalg.norm(ep) < .0003 and np.linalg.norm(er) < .005:
                break
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, jr, p, self.body_id)
            jac = np.vstack([jp[:, self.dofs], jr[:, self.dofs]*rotation_weight])
            delta = jac.T @ np.linalg.solve(jac@jac.T+np.eye(6)*1e-5, error)
            delta *= min(1., .2/max(np.linalg.norm(delta), 1e-10))
            cost = np.linalg.norm(error)
            for scale in (1., .5, .25, .1, .02):
                candidate = np.clip(q+scale*delta, LOWER[:6]+1e-5, UPPER[:6]-1e-5)
                pc, rc = self.forward(candidate)
                if np.linalg.norm(np.r_[position-pc, rotation_error(rotation, rc)*rotation_weight]) < cost:
                    break
            else:
                break
            q = candidate
        p, r = self.forward(q)
        return q, float(np.linalg.norm(position-p)), float(np.linalg.norm(rotation_error(rotation, r)))
