#!/usr/bin/env python3
"""Controlled vertical-approach variant of the plate chord-grasp experiment.

Relative to probe_chord_release.py, only the plate pregrasp approach changes
from tool-axis descent to world-vertical descent. All original physical gates,
geometry, forces, closing, lifting and flat-placement checks remain in effect.
Geometry-only screening is a necessary preflight, not a physical success.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import CleanTableTeacher
import probe_plate_chord as geometry
import probe_chord_capture_feasibility as capture
import probe_chord_release as trial


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--geometry-only', action='store_true')
    args, forwarded = parser.parse_known_args()
    profile = {'mode': 'world_vertical_plate_approach_v1', 'scope': __doc__,
               'script_sha256': sha256(__file__),
               'offset_axis_world': [0, 0, 1], 'object': 'plate'}
    if not args.geometry_only:
        evidence_parser = argparse.ArgumentParser(add_help=False)
        evidence_parser.add_argument('--geometry-evidence', type=Path, required=True)
        evidence_args, _ = evidence_parser.parse_known_args(forwarded)
        evidence = json.loads(evidence_args.geometry_evidence.read_text())
        if evidence.get('vertical_approach_experiment') != profile:
            raise ValueError('Matching vertical approach geometry provenance required')
        if not all(k in evidence for k in ('open_approach_screen', 'necessary_force_screen')):
            raise ValueError('Open approach and necessary force screening required')
    old_approach = CleanTableTeacher.approach_position
    old_publish, old_profile, old_argv = geometry.atomic_json, trial.PROFILE, sys.argv

    def approach(t, name, p, r, distance):
        return np.asarray(p) + [0, 0, distance] if name == 'plate' else old_approach(t, name, p, r, distance)

    try:
        CleanTableTeacher.approach_position = approach
        if args.geometry_only:
            geometry.atomic_json = lambda path, value: old_publish(path, {
                **value, 'vertical_approach_experiment': profile})
            sys.argv = [capture.__file__, *forwarded]
            return capture.main()
        trial.PROFILE = {**old_profile, 'approach_experiment': profile}
        sys.argv = [trial.__file__, *forwarded]
        return trial.main()
    finally:
        CleanTableTeacher.approach_position = old_approach
        geometry.atomic_json, trial.PROFILE, sys.argv = old_publish, old_profile, old_argv


if __name__ == '__main__':
    raise SystemExit(main())
