#!/usr/bin/env python3
"""Fresh full physical rollout with one change: an 8 mm deeper plate rim grasp.

No assumed grasp transform is injected. Actual slip, contacts and rotation are
measured, and the existing planner uses the actual held pose. Original dynamics,
quality gates and release policy remain unchanged. State-only experiment.
"""
import argparse
import sys

import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import CleanTableTeacher
import trace_contact_substeps as tracing


PROFILE = {'mode': 'plate_rim_inset_8mm_v1', 'object': 'plate', 'inset_m': .008,
           'scope': __doc__, 'calibration': 'unidentified experimental grasp offset',
           'unchanged': ['robot and object geometry', 'motor gains and force limits',
                         'contact materials', 'quality and collision limits', 'placement and release policy']}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    _, forwarded = parser.parse_known_args()
    if any(arg.split('=')[0] in ('--objects', '--rgb', '--viewer') for arg in forwarded):
        raise ValueError('This probe requires a full state-only rollout')
    profile = {**PROFILE, 'script_sha256': sha256(__file__)}
    original_grasp, original_metadata = CleanTableTeacher.plan_grasp, CleanTableTeacher.metadata

    def grasp(teacher, name):
        side, position, rotation, width = original_grasp(teacher, name)
        if name == profile['object']:
            obj = teacher.b.data.body('task/'+name)
            object_r = obj.xmat.reshape(3, 3)
            local = object_r.T@(position-obj.xpos)
            local[2] = 0.
            radius = np.linalg.norm(local)
            if radius <= profile['inset_m']:
                raise ValueError('Rim inset exceeds original grasp radius')
            position = position-object_r@local/radius*profile['inset_m']
        return side, position, rotation, width

    def metadata(teacher):
        return {**original_metadata(teacher), 'experimental_rim_grasp': profile}

    try:
        CleanTableTeacher.plan_grasp, CleanTableTeacher.metadata = grasp, metadata
        sys.argv = [tracing.__file__, *forwarded]
        return tracing.main(extra_metadata=profile)
    finally:
        CleanTableTeacher.plan_grasp, CleanTableTeacher.metadata = original_grasp, original_metadata


if __name__ == '__main__':
    raise SystemExit(main())
