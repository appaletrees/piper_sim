"""Read-only solver evidence, kept in privileged extras, outside VLA inputs."""
import mujoco
import numpy as np

from .grasp_assist import finger_contact_evidence


def physics_snapshot(backend, target=None):
    model, data = backend.model, backend.data
    arms = {}
    for side, ids in backend.indices.items():
        hand = data.body(side+'/link6')
        tcp = hand.xpos+hand.xmat.reshape(3, 3) @ np.array([0., 0., .1375])
        jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, tcp, hand.id)
        dofs = ids['dof']
        drive = data.qfrc_actuator[dofs]+data.qfrc_applied[dofs]
        arms[side] = {
            'joint_position': data.qpos[ids['qpos']].tolist(),
            'joint_velocity': data.qvel[dofs].tolist(),
            'actuator_generalized_force': data.qfrc_actuator[dofs].tolist(),
            'applied_generalized_force': data.qfrc_applied[dofs].tolist(),
            'combined_drive_generalized_force': drive.tolist(),
            'constraint_generalized_force': data.qfrc_constraint[dofs].tolist(),
            'tcp_position_world_m': tcp.tolist(),
            'tcp_linear_velocity_world_m_s': (jp@data.qvel).tolist(),
            'tcp_angular_velocity_world_rad_s': (jr@data.qvel).tolist(),
        }
        if target is not None:
            arms[side]['target_finger_contacts'] = finger_contact_evidence(backend, side, target)
    return {'schema_version': 'piper.physics_evidence.v1', 'sim_time': float(data.time),
            'target': target, 'arms': arms}
