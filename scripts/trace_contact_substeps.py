#!/usr/bin/env python3
"""Opt-in read-only force trace around selected phases of a fresh full rollout.

Never replays rounded HDF5 actions or reconstructs solver state. The normal demo
runs from reset. Forces/Jacobians left by mj_step describe its force evaluation
at time_before; qpos_after/qvel_after describe the integrated state at time_after.
Only Euler/implicitfast single-step integration is supported. No extra forward,
step1, controls, forces, contacts or model parameters are written by this tool.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256


class ContactTrace:
    def __init__(self, model, qpos, dofs, actuators, geoms, max_steps=30000,
                 max_contacts=1000000):
        if model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER,
                                        mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
            raise ValueError('Only Euler/implicitfast force-evaluation timing is supported')
        self.model = model
        self.qpos = np.asarray(qpos, dtype=int)
        self.dofs = np.asarray(dofs, dtype=int)
        self.actuators = np.asarray(actuators, dtype=int)
        self.geoms = set(geoms)
        self.max_steps, self.max_contacts = max_steps, max_contacts
        self.rows, self.contacts = [], []
        self.projected = np.empty(model.nv)

    def before(self, data):
        if len(self.rows) >= self.max_steps:
            raise RuntimeError('Contact trace row limit reached; refusing silent truncation')
        return {'time_before': float(data.time),
                'qpos_before': data.qpos[self.qpos].copy(),
                'qvel_before': data.qvel[self.dofs].copy(),
                'control': data.ctrl[self.actuators].copy(),
                'gain': self.model.actuator_gainprm[self.actuators, :3].copy(),
                'bias': self.model.actuator_biasprm[self.actuators, :3].copy(),
                'force_range': self.model.actuator_forcerange[self.actuators].copy(),
                'noslip_iterations': int(self.model.opt.noslip_iterations)}

    def after(self, data, row, phase, target):
        model = self.model
        if abs(data.time-row['time_before']-model.opt.timestep) > 1e-9:
            raise ValueError('Trace requires exactly one contiguous native step')
        row.update(time_after=float(data.time), phase=phase, target=target,
                   qpos_after=data.qpos[self.qpos].copy(),
                   qvel_after=data.qvel[self.dofs].copy(),
                   solver_acceleration=data.qacc[self.dofs].copy(),
                   actuator_force=data.actuator_force[self.actuators].copy(),
                   actuator_length=data.actuator_length[self.actuators].copy(),
                   actuator_velocity=data.actuator_velocity[self.actuators].copy(),
                   constraint_count=int(data.nefc))
        for field in ('qfrc_actuator', 'qfrc_applied', 'qfrc_bias', 'qfrc_passive',
                      'qfrc_constraint', 'qfrc_smooth'):
            row[field] = getattr(data, field)[self.dofs].copy()
        # A separate result vector is the only writable argument. This supports
        # both sparse and dense constraint Jacobians, without extra dynamics.
        components = []
        for kind in range(8):
            forces = np.where(data.efc_type == kind, data.efc_force, 0.)
            mujoco.mj_mulJacTVec(model, data, self.projected, forces)
            components.append(self.projected[self.dofs].copy())
        row['constraint_force_by_type'] = np.asarray(components)
        row['contact_offset'] = len(self.contacts)
        for index in range(data.ncon):
            contact = data.contact[index]
            if int(contact.geom1) not in self.geoms and int(contact.geom2) not in self.geoms:
                continue
            if len(self.contacts) >= self.max_contacts:
                raise RuntimeError('Contact trace contact limit reached; refusing silent truncation')
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, index, force)
            self.contacts.append((len(self.rows), int(contact.geom1), int(contact.geom2),
                                  float(contact.dist), contact.pos.copy(),
                                  contact.frame.copy(), force))
        row['contact_count'] = len(self.contacts)-row['contact_offset']
        self.rows.append(row)

    def save(self, directory, metadata):
        directory = Path(directory)
        arrays = {key: np.asarray([row[key] for row in self.rows])
                  for key in self.rows[0]} if self.rows else {}
        names = ('contact_row', 'contact_geom1', 'contact_geom2', 'contact_distance',
                 'contact_position_world', 'contact_frame', 'contact_force_local')
        for index, name in enumerate(names):
            arrays[name] = np.asarray([row[index] for row in self.contacts])
        if any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError('Object arrays are not allowed in trace artifacts')
        destination = directory/'trace.npz'
        with destination.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        metadata.update(schema_version=1, rows=len(self.rows), contacts=len(self.contacts),
                        trace_sha256=sha256(destination), mujoco_version=mujoco.__version__,
                        qpos_indices=self.qpos.tolist(), dof_indices=self.dofs.tolist(),
                        actuator_indices=self.actuators.tolist(),
                        actuator_names=[self.model.actuator(int(i)).name for i in self.actuators],
                        geom_names=[self.model.geom(i).name for i in range(self.model.ngeom)],
                        constraint_types={k: int(v) for k, v in mujoco.mjtConstraint.__members__.items()},
                        integrator=int(self.model.opt.integrator), integration_dt=float(self.model.opt.timestep),
                        timing='Solver force/Jacobian/contact arrays are from the evaluation at time_before. '
                               'qpos_after and qvel_after are the integrated state at time_after. '
                               'Finite differences of integrated velocity need not equal solver qacc under implicit integration.',
                        status='diagnostic_only_not_training_data')
        atomic_json(directory/'trace.json', metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--trace-output', type=Path, required=True)
    parser.add_argument('--trace-objects', default='bowl,plate,side_dish')
    parser.add_argument('--trace-phases', default='unload_gripper,release')
    args, forwarded = parser.parse_known_args()
    args.trace_output.mkdir(parents=True, exist_ok=False)
    from piper_sim.experts.clean_table import CleanTableTeacher
    objects, phases = set(args.trace_objects.split(',')), set(args.trace_phases.split(','))
    original_step, original_teacher = mujoco.mj_step, CleanTableTeacher.step
    current, tracer, result, error = None, None, None, None
    metadata = {'command': forwarded, 'selected_objects': sorted(objects),
                'selected_phases': sorted(phases), 'script_sha256': sha256(Path(__file__)),
                'scope': 'Read-only force diagnostic of fresh physical rollout; no additional forwards or state writes.'}

    def teacher_step(teacher, *a, **kw):
        nonlocal current
        previous, current = current, teacher
        try:
            return original_teacher(teacher, *a, **kw)
        finally:
            current = previous

    def physics_step(model, data, *a, **kw):
        nonlocal tracer
        selected = (current is not None and model is current.b.model and data is current.b.data
                    and current.current in objects and current.phase in phases)
        if not selected:
            return original_step(model, data, *a, **kw)
        if a or kw.get('nstep', 1) != 1:
            raise ValueError('Diagnostic requires native single-step calls')
        if tracer is None:
            ids = current.b.indices['right']
            geoms = [i for i in range(model.ngeom)
                     if model.body(int(model.geom_bodyid[i])).name.startswith('right/')]
            tracer = ContactTrace(model, ids['qpos'], ids['dof'], ids['ctrl'], geoms)
        row = tracer.before(data)
        value = original_step(model, data, *a, **kw)
        tracer.after(data, row, current.phase, current.current)
        return value

    try:
        mujoco.mj_step, CleanTableTeacher.step = physics_step, teacher_step
        spec = importlib.util.spec_from_file_location('contact_trace_demo', Path(__file__).with_name('demo_clean_table.py'))
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        sys.argv = [str(spec.origin), *forwarded]
        result = demo.main()
        return result
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        mujoco.mj_step, CleanTableTeacher.step = original_step, original_teacher
        metadata.update(returncode=result, error=error)
        if tracer is not None:
            tracer.save(args.trace_output, metadata)
        else:
            atomic_json(args.trace_output/'trace.json', dict(metadata, rows=0, status='no_selected_steps'))


if __name__ == '__main__':
    raise SystemExit(main())
