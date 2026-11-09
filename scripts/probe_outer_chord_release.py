#!/usr/bin/env python3
"""Move the failed plate chord-grasp target outward by 5 mm.

This single controlled change keeps the original tool-axis approach, grasp
orientation, jaw opening, closing, lifting, motor and contact parameters.
Require the additional open-approach and necessary force preflight checks.
The resulting fresh rollout still determines physical success or failure.
"""
import argparse
import json
from pathlib import Path

from piper_sim.collection_jobs import sha256
import probe_chord_release as trial


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--geometry-evidence', type=Path, required=True)
    args, _ = parser.parse_known_args()
    evidence = json.loads(args.geometry_evidence.read_text())
    if not all(key in evidence for key in ('open_approach_screen', 'necessary_force_screen')):
        raise ValueError('Open approach and necessary force screening required')
    if 'vertical_approach_experiment' in evidence:
        raise ValueError('Original tool-axis approach evidence required')
    original = trial.PROFILE
    try:
        trial.PROFILE = {**original, 'radius_m': .095,
                         'radius_experiment': {'scope': __doc__,
                            'script_sha256': sha256(__file__),
                            'baseline_radius_m': .090, 'outward_shift_m': .005}}
        return trial.main()
    finally:
        trial.PROFILE = original


if __name__ == '__main__':
    raise SystemExit(main())
