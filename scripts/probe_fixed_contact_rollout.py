#!/usr/bin/env python3
"""Record state-only full episodes for an explicitly experimental fixed material.

Use the ordinary demo/recorder and independent closed-file audit, with a declared
teacher substitution. No RGB or LeRobot export is exposed by this diagnostic.
The production contact profile is not changed by running this script.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

from diagnose_fixed_contact import FixedMaterialTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--rolling-m', type=float, default=.002)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite([args.scale, args.rolling_m]).all() or args.scale <= 0 or args.rolling_m < 0:
        parser.error('Expected positive scale and nonnegative rolling friction length')
    if args.output.exists(): parser.error('Choose a new output')
    scripts = Path(__file__).parent
    provenance = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(__file__), scripts/'diagnose_fixed_contact.py')}
    class ProbeTeacher(FixedMaterialTeacher):
        def __init__(self, *a, **kw):
            super().__init__(*a, friction_scale=args.scale, rolling_m=args.rolling_m, **kw)

        def metadata(self):
            value = super().metadata()
            value.pop('plate_pad_rolling_friction_m', None)
            value['name'] = 'rigid_table_fixed_material_diagnostic_v1'
            value['finger_friction'] = [args.scale, .015*args.scale, args.rolling_m*args.scale]
            value['experimental_contact_profile'] = {
                'calibrated': False, 'constant_within_episode': True,
                'finger_friction': value['finger_friction'],
                'solref': [self.settings['controller']['finger_contact_time_constant_s'], 1.],
                'solimp': [.95, .99, .001, .5, 2.],
                'scope': __doc__, 'implementation_sha256': provenance,
                'control': 'Existing target-specific motor force/kp and numerical noslip refinement retained; '
                           'no target-specific material changes, attachments or live pose assignments.'}
            return value
    import piper_sim.experts.rigid_table as module
    original = module.RigidTableTeacher
    argv = sys.argv
    try:
        module.RigidTableTeacher = ProbeTeacher
        spec = importlib.util.spec_from_file_location('fixed_contact_demo', scripts/'demo_clean_table.py')
        demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)
        sys.argv = [str(spec.origin), '--record', '--seed', str(args.seed),
            '--domain-randomization', 'configs/domain_randomization/clean_table.yaml',
            '--output', str(args.output)]
        return demo.main()
    finally:
        module.RigidTableTeacher, sys.argv = original, argv


if __name__ == '__main__':
    raise SystemExit(main())
