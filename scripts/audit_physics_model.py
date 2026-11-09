#!/usr/bin/env python3
"""Inspect the actually compiled MuJoCo model versus its URDF; no hardware I/O.

This is a parameter/provenance audit, not real-device dynamics identification.
No rollout, object manipulation, or acceptance verdict is produced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from piper_sim.config import ROOT, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/clean_table_workcell.yaml')
    parser.add_argument('--task-scene', default='configs/tasks/clean_table_rigid.yaml')
    parser.add_argument('--collection-profile', default='configs/collection/clean_table.yaml')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output already exists')
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.simulation import SimulationSession
    from piper_sim.experts.clean_table import CleanTableTeacher
    cfg = load(args.config)
    cfg['task_scene'] = yaml.safe_load(Path(args.task_scene).read_text())
    settings = yaml.safe_load(Path(args.collection_profile).read_text())
    urdf = ROOT/cfg['robot']['urdf']
    root = ET.parse(urdf).getroot()
    result = {'status': 'parameter_audit_only', 'hardware_calibrated': False,
              'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  urdf, Path(args.config), Path(args.task_scene), Path(args.collection_profile),
                  ROOT/'src/piper_sim/mujoco_asset.py', ROOT/'src/piper_sim/robot_asset.py')},
              'urdf_joints': [{'name': j.get('name'), 'type': j.get('type'),
                  'limit': dict(j.find('limit').attrib) if j.find('limit') is not None else None,
                  'dynamics': dict(j.find('dynamics').attrib) if j.find('dynamics') is not None else None}
                  for j in root.findall('joint')], 'robot_bodies': [], 'actuators': [], 'equalities': []}
    backend = MujocoBackend(cfg)
    try:
        session = SimulationSession(backend)
        teacher = CleanTableTeacher(session, settings=settings)
        model = backend.model
        result['teacher'] = teacher.metadata()
        result['runtime'] = {'control_hz': 1/(backend.dt*backend.decimation),
                             'physics_tick_hz': 1/backend.dt, 'integration_hz': 1/model.opt.timestep,
                             'substeps_per_physics_tick': backend.substeps,
                             'integration_steps_per_action': backend.substeps*backend.decimation,
                             'nq': model.nq, 'nv': model.nv, 'ngeom': model.ngeom,
                             'contact_count_at_reset': backend.data.ncon}
        result['collision_model'] = backend.manifest['collision_model']
        result['inertia_model'] = backend.manifest['inertia_model']
        for side in ('left', 'right'):
            for link in root.findall('link'):
                inertial = link.find('inertial')
                if inertial is None: continue
                body = model.body(side+'/'+link.get('name'))
                inertia = inertial.find('inertia')
                ixx, ixy, ixz, iyy, iyz, izz = [float(inertia.get(k)) for k in ('ixx','ixy','ixz','iyy','iyz','izz')]
                principal = np.linalg.eigvalsh([[ixx,ixy,ixz],[ixy,iyy,iyz],[ixz,iyz,izz]])
                mass = float(inertial.find('mass').get('value'))
                compiled = model.body_inertia[body.id]
                result['robot_bodies'].append({'name': body.name, 'source_mass_kg': mass,
                    'compiled_mass_kg': float(model.body_mass[body.id]),
                    'source_principal_inertia_kg_m2': principal.tolist(),
                    'compiled_principal_inertia_kg_m2': sorted(compiled.tolist()),
                    'mass_matches_source': bool(np.isclose(mass, model.body_mass[body.id], rtol=1e-7)),
                    'inertia_matches_source': bool(np.allclose(principal, np.sort(compiled), rtol=1e-6, atol=1e-10)),
                    'positive_and_triangle_valid': bool(np.all(compiled > 0) and 2*max(compiled) <= sum(compiled)+1e-10)})
            for joint in (*[f'joint{i}' for i in range(1,7)], 'gripper'):
                index = model.actuator(side+'/'+joint).id
                result['actuators'].append({'name': model.actuator(index).name,
                    'gainprm': model.actuator_gainprm[index].tolist(),
                    'biasprm': model.actuator_biasprm[index].tolist(),
                    'ctrlrange': model.actuator_ctrlrange[index].tolist(),
                    'forcerange': model.actuator_forcerange[index].tolist()})
        for i in range(model.neq):
            result['equalities'].append({'name': model.equality(i).name, 'type': int(model.eq_type[i]),
                                        'active': bool(backend.data.eq_active[i])})
        result['unidentified_or_incomplete'] = [
            'URDF values are source declarations, not measurements of this robot.',
            'CAD camera/bracket mass and inertia are not included in the declared robot inertia model.',
            'MuJoCo joint damping/armature and teacher contact friction/control gains are simulation choices.',
            'URDF arm velocity=10 rad/s is not a verified device capability; collection command/measured gates are lower.',
            'Teacher applies robot bias feedforward through qfrc_applied; actuator force ranges alone do not bound total joint drive.',
            'Online penetration gate includes nonadjacent same-arm contacts; source adjacent body exclusions and convex hulls still limit self-collision coverage.',
            'Isaac runtime parameters/dynamic responses are not measured by this MuJoCo-only script.',
        ]
    finally:
        backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'runtime': result['runtime'],
                      'robot_inertias_match_source': all(r['inertia_matches_source'] for r in result['robot_bodies']),
                      'hardware_calibrated': False}))


if __name__ == '__main__':
    main()
