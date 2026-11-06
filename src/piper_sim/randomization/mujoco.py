"""Reset-time MuJoCo property application; always restore immutable baseline."""
import numpy as np
import mujoco


class MujocoRandomizer:
    def __init__(self, model, task):
        self.model, self.task = model, task
        self.baseline = {key: getattr(model, key).copy() for key in (
            'body_mass', 'body_inertia', 'geom_friction', 'mat_rgba', 'mat_shininess',
            'mat_specular', 'light_pos', 'light_diffuse', 'light_ambient', 'light_specular')}

    def apply(self, parameters):
        m, base = self.model, self.baseline
        for key, value in base.items():
            getattr(m, key)[:] = value
        if parameters:
            for name, p in parameters['objects'].items():
                body = m.body('task/'+name).id
                m.body_mass[body] *= p['mass_scale']
                m.body_inertia[body] *= p['mass_scale']
                m.geom_friction[m.geom_bodyid == body] *= p['friction_scale']
            prefix = 'clean' if self.task.config['name'] == 'clean_table' else self.task.config['name']
            for name, p in parameters['materials'].items():
                mid = m.material(prefix+'/'+name).id
                m.mat_rgba[mid, :3] = np.clip(base['mat_rgba'][mid, :3]*p['color_scale'], 0, 1)
                # Keep renderer-specific nominal specular values at delta=0.
                m.mat_shininess[mid] = np.clip(base['mat_shininess'][mid]-p['roughness_delta'], 0, 1)
                m.mat_specular[mid] = np.clip(base['mat_specular'][mid]-.3*p['roughness_delta'], 0, 1)
            for key in ('light_diffuse', 'light_ambient', 'light_specular'):
                getattr(m, key)[:] *= parameters['light_scale']
            m.light_pos[:] += parameters['light_position_delta_m']
        # Mass changes need dependent constants refreshed; scratch data avoids
        # changing any live state. Called only at reset, before episode recording.
        mujoco.mj_setConst(m, mujoco.MjData(m))
