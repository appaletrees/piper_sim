#!/usr/bin/env python3
"""Fresh teacher/external execution comparison with identical position actions.

No stored state placement, object writes, assistance or RGB. Both models start
from the same reset and receive the same full-precision 14D targets. The teacher
cycles target identities to expose hidden parameter switching; the fixed
executor must remain identical to ordinary SimulationSession.step execution.
This short free-space diagnostic is not a complete manipulation success.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import yaml

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import load
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.position_execution import audit_position_execution
from piper_sim.simulation import SimulationSession


class TeacherSession(SimulationSession):
    teacher = None

    def read(self):
        obs = super().read()
        if self.teacher:
            obs.extras['teacher'] = self.teacher.telemetry()
        return obs

    def metadata(self):
        return {**super().metadata(), 'teacher': self.teacher.metadata()}


def integration_state(backend):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(backend.model, kind))
    mujoco.mj_getState(backend.model, backend.data, result, kind)
    return result


def run(cfg, settings, root, fixed):
    cfg = copy.deepcopy(cfg)
    if not fixed:
        cfg['mujoco'].pop('position_execution')
    teacher_session = TeacherSession(MujocoBackend(copy.deepcopy(cfg)), record_dir=root)
    external = None
    try:
        external = SimulationSession(MujocoBackend(copy.deepcopy(cfg)), record_dir=root)
        teacher = RigidTableTeacher(teacher_session, settings=settings)
        teacher_session.teacher = teacher
        results = []
        for reset_index, seed in enumerate((1, 2)):
            teacher_session.reset(seed, True)
            external.reset(seed, True)
            teacher.q = teacher_session.target.copy()
            # Fresh reset is the only state initialization in either execution.
            initial_equal = np.array_equal(integration_state(teacher.b), integration_state(external.backend))
            teacher_session.start_recording('teacher_'+str(reset_index), 'execution diagnostic', extras_storage='compressed')
            external.start_recording('external_'+str(reset_index), 'execution diagnostic', extras_storage='compressed')
            start = teacher.q.copy()
            delta = np.array([.1, .2, -.3, .05, .05, .05, .02]*2)
            maxima = {'public_position': 0., 'native_qpos': 0., 'native_qvel': 0.}
            full_equal = True
            for i in range(150):
                s = float(np.clip((i-30)/90, 0., 1.))
                blend = 35*s**4-84*s**5+70*s**6-20*s**7
                command = start+blend*delta
                name = ('cup_00', 'pen', 'chopstick_a', 'plate', 'side_dish')[i//30]
                teacher.current = name
                teacher.phase = 'execution_diagnostic'
                teacher.configure_object_contact(name)
                teacher.q = command.copy()
                a = teacher.step();b = external.step(command)
                full_equal &= np.array_equal(integration_state(teacher.b), integration_state(external.backend))
                for key, va, vb in [('public_position', a.position, b.position),
                                   ('native_qpos', teacher.b.data.qpos, external.backend.data.qpos),
                                   ('native_qvel', teacher.b.data.qvel, external.backend.data.qvel)]:
                    maxima[key] = max(maxima[key], float(np.max(np.abs(va-vb))))
            paths = [Path(s.stop_recording(success=False)['path']) for s in (teacher_session, external)]
            audits = []
            for path in paths:
                digest = sha256(path)
                with h5py.File(path, 'r') as f:
                    audits.append(audit_position_execution(f, json.loads(f.attrs['metadata'])))
                if sha256(path) != digest:
                    raise RuntimeError('Diagnostic source changed')
            results.append({'seed': seed, 'transitions': 150, 'sim_duration_s': 5.,
                'initial_integration_state_exact': initial_equal, 'every_integration_state_exact': bool(full_equal),
                'max_absolute_differences': maxima, 'execution_audits': audits,
                'source_hashes': {str(p): sha256(p) for p in paths}})
            if fixed and (not initial_equal or not full_equal or not all(a['passed'] for a in audits)):
                raise RuntimeError('Shared execution failed exact comparison')
        return results
    finally:
        teacher_session.close()
        if external:
            external.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load('configs/clean_table_position_execution.yaml')
    cfg['task_scene'] = yaml.safe_load(Path('configs/tasks/clean_table_rigid.yaml').read_text())
    cfg['task_scene']['domain_randomization'] = 'configs/domain_randomization/clean_table.yaml'
    settings = yaml.safe_load(Path('configs/collection/clean_table_tcp_slender.yaml').read_text())
    result = {'scope': __doc__, 'script_sha256': sha256(__file__), 'training_ready': False}
    for fixed in (False, True):
        label = 'shared_fixed' if fixed else 'legacy_teacher_vs_external'
        result[label] = run(cfg, settings, args.output/label, fixed)
        atomic_json(args.output/'comparison.json', result)
        print(label, [{k: v for k, v in r.items() if k not in ('execution_audits', 'source_hashes')} for r in result[label]], flush=True)


if __name__ == '__main__':
    main()
