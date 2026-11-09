#!/usr/bin/env python3
"""Fresh full physical trial of a plate chord grasp and low flat placement.

Only grasp and placement position commands change. The actual dynamic held
transform is used by the placement search, not the ideal static hypothesis.
Original motor/contact parameters, physical grasp and quality gates remain.
No live object writes, assistance, or promotion to production/training data.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure
from piper_sim.experts.mujoco_ik import grasp_rotation
from piper_sim.experts.support_evidence import payload_support
from piper_sim.transforms import pose_matrix
import probe_flat_plate as flat
import trace_contact_substeps as tracing


PROFILE = {'mode': 'plate_chord_flat_release_v1', 'object': 'plate',
           'radius_m': .090, 'azimuth_deg': 45, 'tilt_deg': 30, 'height_m': .012,
           'open_width_m': .095, 'scope': __doc__,
           'calibration': 'Unidentified experiment, not calibrated hardware grasp parameters',
           'placement': {**flat.PROFILE, 'max_candidates': 288, 'release_width_m': .095}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--geometry-evidence', type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if any(arg.split('=')[0] in ('--objects', '--rgb', '--viewer') for arg in forwarded):
        raise ValueError('Full state-only physical trial required')
    evidence = json.loads(args.geometry_evidence.read_text())
    candidate = next(row for row in evidence['candidates'] if all(row[key]==PROFILE[key]
                     for key in ('radius_m','azimuth_deg','tilt_deg','height_m')))
    if candidate['stage']!='coupled_geometry_feasible' or not candidate['isolated_state_restored']:
        raise ValueError('Expected matching isolated coupled geometry evidence')
    profile = {**PROFILE, 'script_sha256': sha256(__file__),
               'geometry_evidence': str(args.geometry_evidence),
               'geometry_evidence_sha256': sha256(args.geometry_evidence),
               'geometry_evidence_scope': 'Hypothetical rigid grasp only; current rollout must revalidate actual held geometry.'}
    old_grasp, old_place = CleanTableTeacher.plan_grasp, CleanTableTeacher.placement
    old_meta, old_telemetry = CleanTableTeacher.metadata, CleanTableTeacher.telemetry
    old_profile = flat.PROFILE

    def grasp(t, name):
        if name!=profile['object']:return old_grasp(t, name)
        pose = t.object_pose(name);matrix = pose_matrix(pose['position'], pose['quaternion_wxyz'])
        a = np.deg2rad(profile['azimuth_deg'])
        offset = np.array([-profile['radius_m']*np.cos(a), -profile['radius_m']*np.sin(a), profile['height_m']])
        return ('right', matrix[:3, 3]+matrix[:3, :3]@offset,
                matrix[:3, :3]@grasp_rotation(profile['tilt_deg'], profile['azimuth_deg']), profile['open_width_m'])

    def placement(t, side, name, target):
        if name!=profile['object']:return old_place(t, side, name, target)
        diagnostics = {}
        try:
            choice = flat.search_flat_plate(t, diagnostics)
        except SortFailure:
            t.events.append({'phase': 'experimental_flat_placement_rejected', 'object': name,
                             'sim_time': t.b.sim_time, 'diagnostics': diagnostics})
            raise
        selected = diagnostics['attempts'][-1]
        up = np.asarray(selected['predicted_object_rotation_world'])[:, 2]
        bin_up = t.b.data.body('task/'+target).xmat.reshape(3, 3)[:, 2]
        t.events.append({'phase': 'placement_pose_selected', 'object': name, 'sim_time': t.b.sim_time,
                         'strategy': profile['mode'], 'predicted_object_up_world': up.tolist(),
                         'predicted_tilt_from_container_up_rad': float(np.arccos(np.clip(up@bin_up, -1., 1.))),
                         'scope': 'Measured held transform; sampled support geometry, not measured support force.',
                         'diagnostics': diagnostics})
        return choice

    def metadata(t):return {**old_meta(t), 'experimental_chord_release': profile}

    def telemetry(t):
        value = old_telemetry(t)
        if t.current==profile['object']:
            value['payload_support'] = payload_support(t.b.model, t.b.data, t.current)
        return value

    try:
        CleanTableTeacher.plan_grasp, CleanTableTeacher.placement = grasp, placement
        CleanTableTeacher.metadata, CleanTableTeacher.telemetry = metadata, telemetry
        flat.PROFILE = profile['placement']
        sys.argv = [tracing.__file__, *forwarded]
        return tracing.main(extra_metadata=profile)
    finally:
        CleanTableTeacher.plan_grasp, CleanTableTeacher.placement = old_grasp, old_place
        CleanTableTeacher.metadata, CleanTableTeacher.telemetry = old_meta, old_telemetry
        flat.PROFILE = old_profile


if __name__ == '__main__':raise SystemExit(main())
