#!/usr/bin/env python3
"""Screen a hypothetical deeper rim grasp against flat placement geometry.

ISOLATED GEOMETRY ONLY. Translates the plate relative to a recorded hand pose
to represent moving the original rim grasp inward. The recorded hand/object
orientation is an explicit hypothesis, not predicted by physics. Restores the
complete isolated integration state; never runs dynamics or writes the source.
"""
import argparse
import sys

import mujoco
import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.experts.grasp_assist import finger_contacts
from probe_supported_lowering import integration_state
import probe_flat_plate as base
import probe_flat_plate_routes as routes


def query(teacher, diagnostics, inset):
    b = teacher.b
    state = integration_state(b.model, b.data)
    original = b.data.body('task/plate').xpos.copy()
    rotation = b.data.body('task/plate').xmat.reshape(3, 3).copy()
    radial = np.array([-.0687, -.0518, 0.])
    delta = -radial/np.linalg.norm(radial)*inset
    index = b.model.joint('task/plate/free').qposadr[0]
    diagnostics['grasp_hypothesis'] = {'scope': __doc__, 'script_sha256': sha256(__file__),
        'inset_m': inset, 'initial_grasp_offset_delta_object_m': delta.tolist(),
        'retained_orientation_assumption': 'Same as the recorded held object; must be remeasured in a fresh physical grasp.'}
    try:
        b.data.qpos[index:index+3] = original-rotation@delta
        mujoco.mj_forward(b.model, b.data)
        diagnostics['grasp_hypothesis']['geometric_finger_contacts'] = sorted(finger_contacts(b, 'right', 'plate'))
        return routes.search_with_routes(teacher, diagnostics)
    finally:
        mujoco.mj_setState(b.model, b.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(b.model, b.data)
        mujoco.mj_setState(b.model, b.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        if not np.array_equal(state, integration_state(b.model, b.data)):
            raise RuntimeError('Hypothetical query did not restore isolated state')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--grasp-inset-m', type=float, required=True)
    args, forwarded = parser.parse_known_args()
    if not np.isfinite(args.grasp_inset_m) or not 0 <= args.grasp_inset_m <= .025:
        raise ValueError('Expected inset within 0..25 mm for this bounded experiment')
    base.PROFILE = {**base.PROFILE, 'max_candidates': 288}
    base.search_flat_plate = lambda teacher, evidence: query(teacher, evidence, args.grasp_inset_m)
    sys.argv = [base.__file__, *forwarded]
    base.main()
